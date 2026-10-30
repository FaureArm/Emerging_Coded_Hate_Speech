"""Streaming consecutive reference/discovery slices, configured in run_pipeline.py.

Requires Python 3.14+ for standard-library Zstandard support. Sizes are maximum
uncompressed decimal GB; whole thread records are preserved. Existing reference
prefixes are verified against the archive; completed slice pairs reuse checksums.
"""

from pathlib import Path
from contextlib import ExitStack
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256, file_digest
import json
import tarfile


def byte_limit(size_gb):
    """Convert a positive decimal-GB budget to bytes."""
    try:
        value = Decimal(str(size_gb))
    except InvalidOperation as error:
        raise ValueError("Corpus sizes must be positive, finite numbers.") from error
    if not value.is_finite() or value <= 0:
        raise ValueError("Corpus sizes must be positive, finite numbers.")
    limit = int(value * 1_000_000_000)
    if limit < 1:
        raise ValueError("Corpus size must be at least one byte.")
    return limit


def copy_slice(source, pending, limit, output, existing=None, progress_every_threads=10000):
    """Consume a prefix under the budget, returning its first excluded line.

    When existing is supplied, compare bytes instead of writing. Keeping the
    excluded line is essential: it becomes the first discovery record.
    """
    digest = sha256()
    stats = {
        "bytes": 0,
        "threads": 0,
        "posts": 0,
        "first_thread_id": None,
        "last_thread_id": None,
    }
    line = pending if pending is not None else source.readline()
    while line:
        if stats["bytes"] + len(line) > limit:
            break
        if not line.endswith(b"\n"):
            raise ValueError("Encountered an unterminated NDJSON record.")
        thread = json.loads(line)
        if not isinstance(thread, dict) or not isinstance(thread.get("posts"), list):
            raise ValueError("Expected a thread object with a posts list.")
        posts = thread["posts"]
        openers = [post for post in posts if post.get("resto") == 0]
        if len(openers) != 1:
            raise ValueError("Expected exactly one opening post per thread.")
        thread_id = int(openers[0]["no"])
        if existing is not None:
            if existing.readline() != line:
                raise ValueError(
                    "Existing reference differs from the archive prefix. "
                    "Check its source and the reference size setting."
                )
        else:
            output.write(line)
        digest.update(line)
        stats["bytes"] += len(line)
        stats["threads"] += 1
        stats["posts"] += len(posts)
        if stats["first_thread_id"] is None:
            stats["first_thread_id"] = thread_id
        stats["last_thread_id"] = thread_id
        if progress_every_threads and stats["threads"] % progress_every_threads == 0:
            print(
                f"  {stats['threads']:,} threads / {stats['bytes'] / 1e9:.3f} GB",
                flush=True,
            )
        line = source.readline()
    if not stats["threads"]:
        raise ValueError(
            "No threads fit: the budget is too small or the archive ended."
        )
    if existing is not None and existing.read(1):
        raise ValueError(
            "Existing reference extends beyond the requested prefix. "
            "Use the original size setting or a new reference filename."
        )
    stats["sha256"] = digest.hexdigest()
    stats["maximum_bytes"] = limit
    stats["next_excluded_thread_bytes"] = len(line)
    stats["source_exhausted"] = not bool(line)
    return stats, line


def slice_configuration(archive_path, archive_member, reference_path, discovery_path,
                        reference_size_gb, discovery_size_gb):
    """Validate the requested layout without extracting or creating files."""
    archive_path = Path(archive_path).resolve()
    reference_path, discovery_path = Path(reference_path).resolve(), Path(discovery_path).resolve()
    manifest_path = discovery_path.with_suffix(".slices.json")
    paths = [archive_path, reference_path, discovery_path, manifest_path,
             reference_path.with_name(reference_path.name + ".partial"),
             discovery_path.with_name(discovery_path.name + ".partial")]
    if len(set(paths)) != len(paths):
        raise ValueError("Automatic slices need distinct reference and discovery paths; choose a new DISCOVERY_SOURCE.")
    if reference_path.suffix != ".ndjson" or discovery_path.suffix != ".ndjson":
        raise ValueError("Both slice output paths must end in .ndjson.")
    if not archive_path.is_file():
        raise FileNotFoundError(f"Raiders archive not found: {archive_path}")
    if not isinstance(archive_member, str) or not archive_member.strip():
        raise ValueError("The NDJSON archive member name must be nonempty.")
    archive_stat = archive_path.stat()
    return {
        "format_version": 1,
        "source_archive": str(archive_path), "source_member": archive_member,
        "source_archive_bytes": archive_stat.st_size,
        "source_archive_mtime_ns": archive_stat.st_mtime_ns,
        "reference_path": str(reference_path), "discovery_path": str(discovery_path),
        "reference_maximum_bytes": byte_limit(reference_size_gb),
        "discovery_maximum_bytes": byte_limit(discovery_size_gb),
    }


def prepare_slices(archive_path, archive_member, reference_path, discovery_path,
                   reference_size_gb, discovery_size_gb):
    """Create a consecutive pair once, or verify/reuse its completed manifest.

    An existing reference prefix is allowed so the original 1 GB file and bank
    remain reusable. An existing discovery file needs this helper's manifest.
    Changing a size requires new output names for the changed files.
    """
    configuration = slice_configuration(archive_path, archive_member, reference_path, discovery_path,
                                        reference_size_gb, discovery_size_gb)
    reference_path, discovery_path = Path(configuration["reference_path"]), Path(configuration["discovery_path"])
    archive_path = Path(configuration["source_archive"])
    manifest_path = discovery_path.with_suffix(".slices.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("configuration") != configuration:
            raise ValueError("Slice settings or archive identity differ. Choose new slice output filenames.")
        for label, path in [("reference", reference_path), ("discovery", discovery_path)]:
            if path.stat().st_size != manifest[label]["bytes"]:
                raise ValueError(f"Saved {label} slice size changed: {path}")
            with path.open("rb") as source:
                if file_digest(source, "sha256").hexdigest() != manifest[label]["sha256"]:
                    raise ValueError(f"Saved {label} slice checksum changed: {path}")
        print("Reusing verified reference/discovery slices; no archive decompression.", flush=True)
        return manifest

    reference_partial = reference_path.with_name(reference_path.name + ".partial")
    discovery_partial = discovery_path.with_name(discovery_path.name + ".partial")
    for path in (discovery_path, reference_partial, discovery_partial):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}. Choose a new slice filename.")
    reuse_reference = reference_path.exists()
    if reuse_reference and not reference_path.is_file():
        raise ValueError(f"Reference path is not a file: {reference_path}")

    # Import lazily: extraction stays local and is never required on Colab.
    from compression import zstd

    # Failure leaves .partial files for inspection, never a completed manifest.
    with ExitStack() as stack:
        compressed = stack.enter_context(zstd.open(archive_path, "rb"))
        archive = stack.enter_context(tarfile.open(fileobj=compressed, mode="r|"))
        member = next((item for item in archive if item.name == archive_member), None)
        if member is None or not member.isfile():
            raise ValueError(f"NDJSON member not found: {archive_member}")
        source = stack.enter_context(archive.extractfile(member))
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        discovery_path.parent.mkdir(parents=True, exist_ok=True)
        print("Verifying existing reference prefix..." if reuse_reference else "Extracting reference prefix...", flush=True)
        if reuse_reference:
            with reference_path.open("rb") as existing:
                reference, pending = copy_slice(source, None, configuration["reference_maximum_bytes"], None, existing)
        else:
            with reference_partial.open("xb") as output:
                reference, pending = copy_slice(source, None, configuration["reference_maximum_bytes"], output)
        print("Extracting following discovery slice...", flush=True)
        with discovery_partial.open("xb") as output:
            discovery, _ = copy_slice(source, pending, configuration["discovery_maximum_bytes"], output)

    if slice_configuration(archive_path, archive_member, reference_path, discovery_path,
                           reference_size_gb, discovery_size_gb) != configuration:
        raise ValueError("Archive changed during extraction; outputs have not been published.")
    manifest = {
        "configuration": configuration,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "Consecutive complete threads in archive order; not random sampling",
        "reference": {"path": str(reference_path), "reused_existing_prefix": reuse_reference,
                      "first_source_line": 1, **reference},
        "discovery": {"path": str(discovery_path), "first_source_line": reference["threads"] + 1, **discovery},
        "limitations": ["Slice membership does not remove copied text or cross-thread quotations",
                        "Archive identity uses path, byte size and modification time; slice contents use SHA-256"],
    }
    if discovery_path.exists() or (not reuse_reference and reference_path.exists()):
        raise FileExistsError("Slice output appeared during extraction; refusing to overwrite it.")
    if not reuse_reference:
        reference_partial.rename(reference_path)
    discovery_partial.rename(discovery_path)
    with manifest_path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    print(f"Reference: {reference['threads']:,} threads, {reference['bytes']:,} bytes", flush=True)
    print(f"Discovery: {discovery['threads']:,} threads, {discovery['bytes']:,} bytes", flush=True)
    print(f"Slice manifest: {manifest_path}", flush=True)
    return manifest
