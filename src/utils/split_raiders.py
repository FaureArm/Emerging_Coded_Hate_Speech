"""Create consecutive reference/discovery NDJSON slices without unpacking Raiders.

Run with Python 3.14+: .venv/bin/python src/utils/split_raiders.py
Sizes are decimal GB and are maximums: records are never cut to meet a size.
An existing reference file is reused only if it exactly matches the requested
archive prefix. Discovery starts with the very next complete thread.
This separates source records, not duplicate text or cross-thread quotations.
"""

from pathlib import Path

# -------------------------- Editable parameters --------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_PATH = PROJECT_ROOT / "data/raiders/pol_0616-1119_labeled.tar.zst"
ARCHIVE_MEMBER = "pol_062016-112019_labeled.ndjson"

REFERENCE_SIZE_GB = 1.0
REFERENCE_PATH = PROJECT_ROOT / "data/raiders/raiders_first_1GB.ndjson"
REUSE_EXISTING_REFERENCE = True

DISCOVERY_SIZE_GB = 3.0
DISCOVERY_PATH = PROJECT_ROOT / "data/raiders/raiders_next_3GB.ndjson"
MANIFEST_PATH = PROJECT_ROOT / "data/raiders/raiders_split.info.json"
PROGRESS_EVERY_THREADS = 10000
# -------------------------------------------------------------------------

from contextlib import ExitStack
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
import tarfile


def byte_limit(size_gb):
    """Convert a positive decimal-GB budget to bytes."""
    value = Decimal(str(size_gb))
    if not value.is_finite() or value <= 0:
        raise ValueError("Corpus sizes must be positive, finite numbers.")
    limit = int(value * 1_000_000_000)
    if limit < 1:
        raise ValueError("Corpus size must be at least one byte.")
    return limit


def copy_slice(source, pending, limit, output, existing=None):
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
        if PROGRESS_EVERY_THREADS and stats["threads"] % PROGRESS_EVERY_THREADS == 0:
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


def main():
    # Python 3.14's standard library provides streaming Zstandard support.
    from compression import zstd

    reference_limit = byte_limit(REFERENCE_SIZE_GB)
    discovery_limit = byte_limit(DISCOVERY_SIZE_GB)
    archive_path = Path(ARCHIVE_PATH).resolve()
    reference_path = Path(REFERENCE_PATH).resolve()
    discovery_path = Path(DISCOVERY_PATH).resolve()
    manifest_path = Path(MANIFEST_PATH).resolve()
    reference_partial = reference_path.with_name(reference_path.name + ".partial")
    discovery_partial = discovery_path.with_name(discovery_path.name + ".partial")
    paths = [
        archive_path,
        reference_path,
        discovery_path,
        manifest_path,
        reference_partial,
        discovery_partial,
    ]
    if len(set(paths)) != len(paths):
        raise ValueError(
            "Archive, output, manifest and temporary paths must be distinct."
        )
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Set ARCHIVE_PATH to your compressed archive: {archive_path}"
        )
    reuse = reference_path.exists() and REUSE_EXISTING_REFERENCE
    for path in [discovery_path, manifest_path, discovery_partial] + (
        [] if reuse else [reference_path, reference_partial]
    ):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}. Choose a new path.")

    # Failure leaves .partial files for inspection, never a misleading final file.
    with ExitStack() as stack:
        compressed = stack.enter_context(zstd.open(archive_path, "rb"))
        archive = stack.enter_context(tarfile.open(fileobj=compressed, mode="r|"))
        member = next((item for item in archive if item.name == ARCHIVE_MEMBER), None)
        if member is None or not member.isfile():
            raise ValueError(f"NDJSON member not found: {ARCHIVE_MEMBER}")
        source = stack.enter_context(archive.extractfile(member))
        for path in (reference_path, discovery_path, manifest_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        print(
            "Verifying existing reference..." if reuse else "Writing reference...",
            flush=True,
        )
        if reuse:
            with reference_path.open("rb") as existing:
                reference, pending = copy_slice(
                    source, None, reference_limit, None, existing
                )
        else:
            with reference_partial.open("xb") as output:
                reference, pending = copy_slice(source, None, reference_limit, output)
        print("Writing consecutive discovery slice...", flush=True)
        with discovery_partial.open("xb") as output:
            discovery, _ = copy_slice(source, pending, discovery_limit, output)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_archive": str(archive_path),
        "source_archive_bytes": archive_path.stat().st_size,
        "source_member": ARCHIVE_MEMBER,
        "selection": "Consecutive complete threads in archive order; not random sampling",
        "reference": {
            "path": str(reference_path),
            "reused": reuse,
            "first_source_line": 1,
            **reference,
        },
        "discovery": {
            "path": str(discovery_path),
            "first_source_line": reference["threads"] + 1,
            **discovery,
        },
        "limitations": ["No duplicate-text or cross-thread-quotation filtering"],
    }
    if not reuse:
        reference_partial.rename(reference_path)
    discovery_partial.rename(discovery_path)
    with manifest_path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    print(f"Reference: {reference['threads']:,} threads, {reference['bytes']:,} bytes")
    print(f"Discovery: {discovery['threads']:,} threads, {discovery['bytes']:,} bytes")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
