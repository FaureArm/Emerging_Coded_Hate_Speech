"""Edit SETTINGS below, then run: .venv/bin/python run_pipeline.py

Prepares reference text locally; all BERT computation runs on Colab.
Use --check to validate settings and print the plan without creating any files
or contacting Colab. Results are inspected in src/Results.ipynb afterwards.
"""

from pathlib import Path
import argparse
import hashlib
import inspect
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.vocabulary import HATE_CATEGORIES, vocabulary_for_category

# --------------------------- EDIT SETTINGS HERE ---------------------------
HATE_CATEGORY = "islamophobia"  # Or pass --category; seeds and anchors come from vocabulary.py.
# False: use the existing input files below (including a shared file).
# True: create/reuse separate consecutive slices locally before the Colab run.
PREPARE_SLICES_FROM_ARCHIVE = False
RAW_ARCHIVE_PATH = "data/raiders/pol_0616-1119_labeled.tar.zst"
RAW_ARCHIVE_MEMBER = "pol_062016-112019_labeled.ndjson"
REFERENCE_SLICE_SIZE_GB = 1.0  # First uncompressed decimal GB, retaining whole threads.
DISCOVERY_SLICE_SIZE_GB = 10.0  # Following GB; only used when slice preparation is enabled.

# With automatic slices, these are output paths and MUST differ.
# Example: keep REFERENCE_SOURCE and change DISCOVERY_SOURCE to raiders_next_3GB.ndjson.
REFERENCE_SOURCE = "data/raiders/raiders_1GB.ndjson"
DISCOVERY_SOURCE = "data/raiders/raiders_10GB.ndjson"

# None uses <category>_v1. Use a new name when reference/model settings change.
# The old islamophobia_v3 bank contains antisemitism anchors; do not reuse it.
REFERENCE_VERSION = None
MAX_REFERENCES_PER_ANCHOR = 100  # None keeps all eligible distinct contexts.
REFERENCE_RANDOM_SEED = 42

MAX_DISCOVERY_POSTS = 10000
DISCOVERY_SAMPLING_METHOD = "first_matching"  # Or "uniform_time": balance eligible posts across UTC days.
DISCOVERY_RANDOM_SEED = 42  # Used only by uniform_time; independent of reference selection.
CANDIDATE_BATCH_SIZE = 16
RUN_NAME = None  # None creates a unique timestamped name; otherwise use a new name.

BERT_CHECKPOINT = "google-bert/bert-base-uncased"
MODEL_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"  # Same revision as your first run.
CONTEXT_TOKENS_EACH_SIDE = 64

COLAB_EXECUTABLE = None  # Auto-detect PATH or ~/.local/bin/colab; or set an absolute path.
EXISTING_COLAB_SESSION = None  # None creates a fresh session; "hmapp" reuses that session.
COLAB_GPU = "T4"  # Used only when creating a session.
REMOTE_STEP_TIMEOUT_SECONDS = 7200
TRANSFER_MAX_ATTEMPTS = 5  # Includes the first attempt; file transfers only.
TRANSFER_TIMEOUT_SECONDS = 180  # Maximum time for one upload/download attempt.
TRANSFER_RETRY_DELAY_SECONDS = 5  # Exponential backoff, capped at 60 seconds.
# --------------------------------------------------------------------------

TRANSFER_CHUNK_BYTES = 16 * 1024 * 1024


def file_digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def split_archive(archive_path, parts_directory, chunk_bytes):
    """Use bounded memory; reuse valid parts if an earlier split was interrupted."""
    archive_path, parts_directory = Path(archive_path), Path(parts_directory)
    parts_directory.mkdir(parents=True, exist_ok=True)
    parts = []
    with archive_path.open("rb") as source:
        while chunk := source.read(chunk_bytes):
            name = f"part-{len(parts):06d}"
            digest = hashlib.sha256(chunk).hexdigest()
            part_path = parts_directory / name
            if not part_matches(part_path, {"bytes": len(chunk), "sha256": digest}):
                part_path.write_bytes(chunk)
            parts.append({"name": name, "bytes": len(chunk), "sha256": digest})
    manifest = {"bytes": archive_path.stat().st_size, "sha256": file_digest(archive_path), "parts": parts}
    write_json_atomic(parts_directory / "transfer.json", manifest)
    return manifest


def write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def part_matches(path, part):
    path = Path(path)
    return path.is_file() and path.stat().st_size == part["bytes"] and file_digest(path) == part["sha256"]


def combine_archive_parts(parts_directory, archive_path, manifest):
    """Reject missing/corrupt parts; never concatenate unrelated files by glob."""
    parts_directory, archive_path = Path(parts_directory), Path(archive_path)
    if archive_path.exists():
        if part_matches(archive_path, manifest):
            return
        raise ValueError(f"Existing archive differs from the transfer: {archive_path}")
    partial = archive_path.with_name(archive_path.name + ".partial")
    with partial.open("wb") as output:
        for index, part in enumerate(manifest["parts"]):
            if part["name"] != f"part-{index:06d}":
                raise ValueError("Unexpected transfer part order or filename.")
            source_path = parts_directory / part["name"]
            if source_path.stat().st_size != part["bytes"] or file_digest(source_path) != part["sha256"]:
                raise ValueError(f"Corrupt transfer part: {source_path.name}")
            with source_path.open("rb") as source:
                shutil.copyfileobj(source, output)
    if not part_matches(partial, manifest):
        raise ValueError("Reassembled archive checksum differs from the original.")
    partial.replace(archive_path)


def validate_zip_members(archive, destination, expected_root=None):
    """Only extract ordinary relative paths inside the intended directory."""
    destination = Path(destination).resolve()
    for member in archive.infolist():
        member_path = Path(member.filename)
        if (not member_path.parts or member_path.is_absolute() or ".." in member_path.parts
                or "\\" in member.filename or (member.external_attr >> 16) & 0o170000 == 0o120000
                or not (destination / member_path).resolve().is_relative_to(destination)):
            raise ValueError(f"Unsafe ZIP member: {member.filename}")
        if expected_root is not None and member_path.parts[0] != expected_root:
            raise ValueError(f"Expected only {expected_root}/ in ZIP, found {member.filename}")


def extract_validated_archive(archive_path, destination, validator):
    """Publish an extracted bank/run only after its contents pass validation."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=destination.parent) as temporary:
        with zipfile.ZipFile(archive_path) as archive:
            validate_zip_members(archive, temporary, destination.name)
            archive.extractall(temporary)
        staged = Path(temporary) / destination.name
        validator(staged)
        if destination.exists():
            raise FileExistsError(f"Output appeared during extraction: {destination}")
        staged.rename(destination)


class ColabClient:
    def __init__(self, executable, session_name, timeout_seconds):
        self.executable = executable
        self.session_name = session_name
        self.timeout_seconds = timeout_seconds

    def command(self, action, *arguments):
        command = [self.executable, action, "-s", self.session_name, *map(str, arguments)]
        if action not in {"upload", "download"}:
            subprocess.run(command, check=True)
            return
        for attempt in range(1, TRANSFER_MAX_ATTEMPTS + 1):
            try:
                subprocess.run(command, check=True, timeout=TRANSFER_TIMEOUT_SECONDS)
                return
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if attempt == TRANSFER_MAX_ATTEMPTS:
                    raise
                delay = min(TRANSFER_RETRY_DELAY_SECONDS * 2 ** (attempt - 1), 60)
                print(f"{action.capitalize()} failed (attempt {attempt}/{TRANSFER_MAX_ATTEMPTS}); "
                      f"retrying in {delay} seconds.", flush=True)
                time.sleep(delay)

    def execute(self, code):
        # CLI v0.7.5 may print a remote Python error yet exit successfully.
        # Require an exact success marker reached only after all code succeeds.
        success_marker = "HMAPP_SUCCESS_" + uuid.uuid4().hex
        command = [self.executable, "exec", "-s", self.session_name,
                   "--timeout", str(self.timeout_seconds)]
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8") as process:
            process.stdin.write(code + f"\nprint({success_marker!r}, flush=True)\n")
            process.stdin.close()
            succeeded = False
            for line in process.stdout:
                if line.strip() == success_marker:
                    succeeded = True
                else:
                    print(line, end="", flush=True)
            return_code = process.wait()
        if return_code or not succeeded:
            raise RuntimeError(f"Colab step did not complete in session {self.session_name}. See output above.")


def remote_helpers():
    return "from pathlib import Path\nimport hashlib, json, shutil, zipfile\n" + "\n".join(
        inspect.getsource(function) for function in
        (file_digest, write_json_atomic, part_matches, split_archive, combine_archive_parts, validate_zip_members)
    )


def upload_inputs(client, archive_path, temporary_directory, remote_workspace):
    parts_directory = Path(temporary_directory) / "upload_parts"
    manifest = split_archive(archive_path, parts_directory, TRANSFER_CHUNK_BYTES)
    remote_parts = remote_workspace / "upload_parts"
    client.execute(remote_helpers() + f'''
remote_parts = Path({str(remote_parts)!r})
remote_parts.mkdir(parents=True, exist_ok=True)
missing_parts = [part["name"] for part in {manifest['parts']!r}
                 if not part_matches(remote_parts / part["name"], part)]
write_json_atomic(remote_parts / "missing.json", missing_parts)
''')
    status_path = parts_directory / "missing.json"
    client.command("download", str(remote_parts / "missing.json").removeprefix("/"), status_path)
    missing_parts = set(json.loads(status_path.read_text()))
    print(f"Reusing {len(manifest['parts']) - len(missing_parts)}/{len(manifest['parts'])} verified uploaded parts.", flush=True)
    for index, part in enumerate(manifest["parts"], 1):
        if part["name"] not in missing_parts:
            continue
        print(f"Uploading input part {index}/{len(manifest['parts'])}", flush=True)
        client.command("upload", parts_directory / part["name"],
                       str(remote_parts / part["name"]).removeprefix("/"))
    client.execute(remote_helpers() + f'''
archive_path = Path({str(remote_workspace / 'inputs.zip')!r})
combine_archive_parts({str(remote_parts)!r}, archive_path, {manifest!r})
with zipfile.ZipFile(archive_path) as archive:
    validate_zip_members(archive, {str(remote_workspace)!r})
    archive.extractall({str(remote_workspace)!r})
''')


def download_archive(client, remote_archive, destination, temporary_directory):
    remote_parts = remote_archive.with_name(remote_archive.name + ".parts")
    client.execute(remote_helpers() + f"\nsplit_archive({str(remote_archive)!r}, {str(remote_parts)!r}, {TRANSFER_CHUNK_BYTES})")
    local_parts = Path(temporary_directory) / (destination.name + ".parts")
    local_parts.mkdir(exist_ok=True)
    client.command("download", str(remote_parts / "transfer.json").removeprefix("/"), local_parts / "transfer.json")
    manifest = json.loads((local_parts / "transfer.json").read_text())
    for index, part in enumerate(manifest["parts"]):
        if part["name"] != f"part-{index:06d}":
            raise ValueError("Unexpected download part name.")
        if part_matches(local_parts / part["name"], part):
            print(f"Reusing downloaded {destination.name}: part {index + 1}/{len(manifest['parts'])}", flush=True)
            continue
        print(f"Downloading {destination.name}: part {index + 1}/{len(manifest['parts'])}", flush=True)
        client.command("download", str(remote_parts / part["name"]).removeprefix("/"), local_parts / part["name"])
    staged_archive = Path(temporary_directory) / destination.name
    combine_archive_parts(local_parts, staged_archive, manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if part_matches(destination, manifest):
            return
        raise FileExistsError(f"Refusing to overwrite {destination}")
    staged_archive.rename(destination)


def validate_bank(bank_directory, preparation_manifest):
    from utils.reference_bank import load_reference_bank, validate_discovery_bank
    anchors, _ = vocabulary_for_category(HATE_CATEGORY)
    vectors, metadata, posts, manifest = load_reference_bank(bank_directory)
    validate_discovery_bank(vectors, metadata, posts, manifest, anchors, category=HATE_CATEGORY)
    saved_preparation = json.loads((bank_directory / "preparation_manifest.json").read_text())
    if (manifest["preparation_config"] != preparation_manifest["config"]
            or saved_preparation["file_checksums"] != preparation_manifest["file_checksums"]):
        raise ValueError("Saved bank uses different reference selections. Choose a new REFERENCE_VERSION.")
    configuration = manifest["embedding_config"]
    if (configuration["checkpoint"] != BERT_CHECKPOINT
            or MODEL_REVISION not in {configuration["requested_revision"], configuration["resolved_revision"]}
            or configuration["context_tokens_each_side"] != CONTEXT_TOKENS_EACH_SIDE):
        raise ValueError("Saved bank uses different BERT settings. Choose a new REFERENCE_VERSION.")


def validate_results(run_directory):
    from utils.results import load_run
    run_info, _, _ = load_run(run_directory)
    anchors, seeds = vocabulary_for_category(HATE_CATEGORY)
    if (run_info["category"] != HATE_CATEGORY or run_info["anchor_terms"] != anchors
            or run_info["seed_words"] != seeds):
        raise ValueError("Downloaded run category, anchors or seeds do not match the selected vocabulary.")
    retrieval = run_info["retrieval"]
    expected_seed = DISCOVERY_RANDOM_SEED if DISCOVERY_SAMPLING_METHOD == "uniform_time" else None
    if (retrieval["sampling_method"] != DISCOVERY_SAMPLING_METHOD
            or retrieval["sampling_random_seed"] != expected_seed):
        raise ValueError("Downloaded run discovery sampling settings do not match the launcher.")
    for name, expected in run_info["artifact_checksums"].items():
        if Path(name).name != name or file_digest(run_directory / name) != expected:
            raise ValueError(f"Invalid downloaded artifact: {name}")


def configured_paths(saved_run_name=None, allow_existing_outputs=False):
    from utils.discovery import validate_discovery_sampling
    validate_discovery_sampling(DISCOVERY_SAMPLING_METHOD, DISCOVERY_RANDOM_SEED)
    anchors, _ = vocabulary_for_category(HATE_CATEGORY)
    reference_version = REFERENCE_VERSION if REFERENCE_VERSION is not None else f"{HATE_CATEGORY}_v1"
    run_name = saved_run_name or RUN_NAME or f"raiders_{HATE_CATEGORY}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    for label, name in [("REFERENCE_VERSION", reference_version), ("RUN_NAME", run_name)]:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError(f"{label} must be a plain name containing letters, numbers, underscores or hyphens.")
    for name, value in [("MAX_DISCOVERY_POSTS", MAX_DISCOVERY_POSTS), ("CANDIDATE_BATCH_SIZE", CANDIDATE_BATCH_SIZE),
                        ("REMOTE_STEP_TIMEOUT_SECONDS", REMOTE_STEP_TIMEOUT_SECONDS),
                        ("TRANSFER_MAX_ATTEMPTS", TRANSFER_MAX_ATTEMPTS),
                        ("TRANSFER_TIMEOUT_SECONDS", TRANSFER_TIMEOUT_SECONDS)]:
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if type(TRANSFER_RETRY_DELAY_SECONDS) is not int or TRANSFER_RETRY_DELAY_SECONDS < 0:
        raise ValueError("TRANSFER_RETRY_DELAY_SECONDS must be a nonnegative integer.")
    if MAX_REFERENCES_PER_ANCHOR is not None and (type(MAX_REFERENCES_PER_ANCHOR) is not int or MAX_REFERENCES_PER_ANCHOR < 1):
        raise ValueError("MAX_REFERENCES_PER_ANCHOR must be positive or None.")
    if type(REFERENCE_RANDOM_SEED) is not int or type(CONTEXT_TOKENS_EACH_SIDE) is not int or CONTEXT_TOKENS_EACH_SIDE < 0:
        raise ValueError("Reference seed and context size must be integers; context size cannot be negative.")
    executable = COLAB_EXECUTABLE or shutil.which("colab") or str(Path.home() / ".local/bin/colab")
    executable = shutil.which(str(Path(executable).expanduser()))
    if executable is None:
        raise FileNotFoundError("Colab CLI not found. Set COLAB_EXECUTABLE to the installed colab executable.")
    paths = {
        "reference_source": (PROJECT_ROOT / Path(REFERENCE_SOURCE).expanduser()).resolve(),
        "discovery_source": (PROJECT_ROOT / Path(DISCOVERY_SOURCE).expanduser()).resolve(),
        "reference_version": reference_version,
        "reference_corpus": PROJECT_ROOT / "data/raiders/reference_corpora" / reference_version,
        "reference_bank": PROJECT_ROOT / "artifacts/reference_banks" / reference_version,
        "bank_zip": PROJECT_ROOT / "artifacts" / f"reference_bank_{reference_version}.zip",
        "run": PROJECT_ROOT / "artifacts" / run_name,
        "run_zip": PROJECT_ROOT / "artifacts" / f"{run_name}.zip",
    }
    if type(PREPARE_SLICES_FROM_ARCHIVE) is not bool:
        raise ValueError("PREPARE_SLICES_FROM_ARCHIVE must be True or False.")
    paths["slice_configuration"] = None
    if PREPARE_SLICES_FROM_ARCHIVE:
        from utils.split_raiders import slice_configuration
        paths["slice_configuration"] = slice_configuration(
            (PROJECT_ROOT / Path(RAW_ARCHIVE_PATH).expanduser()).resolve(), RAW_ARCHIVE_MEMBER,
            paths["reference_source"], paths["discovery_source"],
            REFERENCE_SLICE_SIZE_GB, DISCOVERY_SLICE_SIZE_GB,
        )
    else:
        for name in ["reference_source", "discovery_source"]:
            if not paths[name].is_file() or paths[name].suffix != ".ndjson":
                raise FileNotFoundError(f"Choose an existing NDJSON slice for {name}: {paths[name]}")
    for name in ["run", "run_zip"]:
        if paths[name].exists() and not allow_existing_outputs:
            raise FileExistsError(f"Output already exists: {paths[name]}. Choose a new RUN_NAME.")
    for manifest_path, config_key in [
        (paths["reference_corpus"] / "manifest.json", "config"),
        (paths["reference_bank"] / "bank_manifest.json", "preparation_config"),
    ]:
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            config = manifest[config_key]
            if (config.get("category") != HATE_CATEGORY
                    or (config_key == "preparation_config" and manifest.get("category") != HATE_CATEGORY)
                    or config.get("anchors") != anchors):
                raise ValueError(
                    f"Saved reference vocabulary at {manifest_path} does not match {HATE_CATEGORY!r}. "
                    "Choose a new REFERENCE_VERSION; renaming an existing bank cannot change its anchors."
                )
    return paths, executable, anchors


def configuration_snapshot(paths):
    """Reject changed experiments on resume; transport settings may be adjusted."""
    names = (
        "HATE_CATEGORY", "REFERENCE_VERSION", "MAX_REFERENCES_PER_ANCHOR", "REFERENCE_RANDOM_SEED",
        "MAX_DISCOVERY_POSTS", "DISCOVERY_SAMPLING_METHOD", "DISCOVERY_RANDOM_SEED", "CANDIDATE_BATCH_SIZE", "RUN_NAME",
        "BERT_CHECKPOINT", "MODEL_REVISION", "CONTEXT_TOKENS_EACH_SIDE",
        "EXISTING_COLAB_SESSION", "COLAB_GPU", "TRANSFER_CHUNK_BYTES",
    )
    anchors, seed_words = vocabulary_for_category(HATE_CATEGORY)
    return {
        "project_root": str(PROJECT_ROOT),
        "settings": {name: globals()[name] for name in names},
        "vocabulary": {"anchors": anchors, "seed_words": seed_words},
        "sources": {
            name: {"path": str(paths[name]), "bytes": paths[name].stat().st_size,
                   "mtime_ns": paths[name].stat().st_mtime_ns}
            for name in ("reference_source", "discovery_source")
        },
        "slice_configuration": paths["slice_configuration"],
    }


def run_pipeline(check_only=False, resume=None):
    checkpoint = None
    if resume is not None:
        checkpoint = Path(resume).expanduser().resolve()
        if checkpoint.parent != (PROJECT_ROOT / "artifacts").resolve() or not checkpoint.name.startswith(".pipeline-"):
            raise ValueError("Resume must point to a saved artifacts/.pipeline-* directory.")
        state = json.loads((checkpoint / "checkpoint.json").read_text())
        if state.get("format_version") != 1:
            raise ValueError("Unsupported pipeline checkpoint.")
        paths, executable, anchors = configured_paths(state["settings"]["run_name"], allow_existing_outputs=True)
        if configuration_snapshot(paths) != state["configuration"]:
            raise ValueError("Settings or source files changed since the checkpoint. Restore them or start a new run.")
        print(f"Resuming session {state['settings']['colab_session']} from {checkpoint}", flush=True)
    else:
        paths, executable, anchors = configured_paths()
    anchors, seed_words = vocabulary_for_category(HATE_CATEGORY)
    print(f"Category: {HATE_CATEGORY}; {len(anchors)} anchors; retrieval seeds: {seed_words}")
    print(f"Reference version: {paths['reference_version']} (cap {MAX_REFERENCES_PER_ANCHOR} per anchor)")
    print(f"Reference source: {paths['reference_source']}")
    print(f"Discovery: up to {MAX_DISCOVERY_POSTS:,} matching posts from {paths['discovery_source']}")
    print(f"Discovery selection: {DISCOVERY_SAMPLING_METHOD}")
    if DISCOVERY_SAMPLING_METHOD == "uniform_time":
        print(f"Balance UTC days using seed {DISCOVERY_RANDOM_SEED}; two full scans on Colab before embedding selected posts.")
    print(f"Results: {paths['run']}")
    print(f"Colab CLI: {executable}")
    if paths["slice_configuration"] is not None:
        print(f"Local slice preparation: first {REFERENCE_SLICE_SIZE_GB} GB for references, "
              f"then {DISCOVERY_SLICE_SIZE_GB} GB for discovery; complete threads only.")
    if check_only:
        print("Configuration paths checked. No files created or Colab calls made; cache contents are checked during a run.")
        return paths["run"]

    if checkpoint is not None:
        if not part_matches(checkpoint / "inputs.zip", state["input_archive"]):
            raise ValueError("Saved input archive is missing or corrupt; cannot resume this checkpoint.")
        bank_validator = lambda folder: validate_bank(folder, state["preparation_manifest"])
        if paths["reference_bank"].exists():
            bank_validator(paths["reference_bank"])
        return continue_pipeline(paths, executable, checkpoint, state)

    slice_manifest = None
    if paths["slice_configuration"] is not None:
        from utils.split_raiders import prepare_slices
        print("Preparing or verifying local archive slices", flush=True)
        slice_manifest = prepare_slices(
            paths["slice_configuration"]["source_archive"], RAW_ARCHIVE_MEMBER,
            paths["reference_source"], paths["discovery_source"],
            REFERENCE_SLICE_SIZE_GB, DISCOVERY_SLICE_SIZE_GB,
        )

    from utils.reference_review import prepare_references
    print("1/6: Prepare or validate saved reference text", flush=True)
    _, occurrences, _, preparation_manifest = prepare_references(
        paths["reference_source"], paths["reference_corpus"], anchors, HATE_CATEGORY,
        max_per_anchor=MAX_REFERENCES_PER_ANCHOR, random_seed=REFERENCE_RANDOM_SEED,
    )
    if len(occurrences[["anchor", "post_id"]].drop_duplicates()) < 5:
        raise ValueError("Fewer than five reference vectors are possible. Expand the reference source before encoding.")
    bank_validator = lambda folder: validate_bank(folder, preparation_manifest)
    if paths["reference_bank"].exists():
        bank_validator(paths["reference_bank"])
    elif paths["bank_zip"].exists():
        extract_validated_archive(paths["bank_zip"], paths["reference_bank"], bank_validator)
    reuse_bank = paths["reference_bank"].is_dir()
    print("Reusing encoded reference bank." if reuse_bank else "Reference bank will be encoded on Colab.")

    invocation_id = "hmapp_" + uuid.uuid4().hex
    session_name = EXISTING_COLAB_SESSION or invocation_id
    remote_workspace = Path("/content") / invocation_id
    settings = {
        "slice_preparation": slice_manifest,
        "category": HATE_CATEGORY, "reference_version": paths["reference_version"], "reference_source": str(paths["reference_source"]),
        "discovery_source": str(paths["discovery_source"]), "max_references_per_anchor": MAX_REFERENCES_PER_ANCHOR,
        "reference_random_seed": REFERENCE_RANDOM_SEED, "seed_words": seed_words,
        "max_discovery_posts": MAX_DISCOVERY_POSTS, "candidate_batch_size": CANDIDATE_BATCH_SIZE,
        "discovery_sampling_method": DISCOVERY_SAMPLING_METHOD, "discovery_random_seed": DISCOVERY_RANDOM_SEED,
        "bert_checkpoint": BERT_CHECKPOINT, "model_revision": MODEL_REVISION,
        "context_tokens_each_side": CONTEXT_TOKENS_EACH_SIDE, "reference_bank_reused": reuse_bank,
        "run_name": paths["run"].name, "colab_session": session_name,
        "requested_gpu": COLAB_GPU if not EXISTING_COLAB_SESSION else None,
    }
    paths["run"].parent.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(tempfile.mkdtemp(prefix=".pipeline-", dir=paths["run"].parent))
    try:
        print("2/6: Package code and selected inputs", flush=True)
        input_archive = checkpoint / "inputs.zip"
        with zipfile.ZipFile(input_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for source in sorted((PROJECT_ROOT / "src").rglob("*.py")):
                archive.write(source, source.relative_to(PROJECT_ROOT))
            archive.write(PROJECT_ROOT / "run_pipeline.py", "run_pipeline.py")
            archive.write(paths["discovery_source"], "inputs/discovery.ndjson")
            reference_folder = paths["reference_bank"] if reuse_bank else paths["reference_corpus"]
            remote_folder = "reference_bank" if reuse_bank else "reference_corpus"
            for source in sorted(reference_folder.iterdir()):
                if source.is_file():
                    archive.write(source, f"{remote_folder}/{source.name}")
            archive.writestr("pipeline_settings.json", json.dumps(settings, indent=2) + "\n")
        state = {
            "format_version": 1, "configuration": configuration_snapshot(paths),
            "settings": settings, "preparation_manifest": preparation_manifest,
            "remote_workspace": str(remote_workspace),
            "input_archive": {"bytes": input_archive.stat().st_size, "sha256": file_digest(input_archive)},
            "completed_steps": [],
        }
        write_json_atomic(checkpoint / "checkpoint.json", state)
    except BaseException:
        # No remote work has started, so an incomplete package cannot be resumed.
        shutil.rmtree(checkpoint)
        raise
    return continue_pipeline(paths, executable, checkpoint, state)


def continue_pipeline(paths, executable, checkpoint, state):
    """Persist each completed stage; never automatically repeat a failed BERT job."""
    settings = state["settings"]
    session_name = settings["colab_session"]
    remote_workspace = Path(state["remote_workspace"])
    client = ColabClient(executable, session_name, REMOTE_STEP_TIMEOUT_SECONDS)
    completed = state["completed_steps"]

    def complete(step):
        completed.append(step)
        write_json_atomic(checkpoint / "checkpoint.json", state)

    bank_validator = lambda folder: validate_bank(folder, state["preparation_manifest"])
    reuse_bank = settings["reference_bank_reused"]
    try:
        print(f"3/6: Use Colab session {session_name}", flush=True)
        if "session" not in completed:
            if not EXISTING_COLAB_SESSION:
                client.command("new", "--gpu", COLAB_GPU)
            complete("session")
        print(f"Remote workspace (retained on failure): {remote_workspace}", flush=True)
        if "environment" not in completed:
            client.command("install", "pandas", "numpy", "nltk", "ftfy", "emoji", "torch", "transformers", "scikit-learn", "tqdm")
        # Also confirms that the saved session is still reachable on resume.
        client.execute("import pandas, numpy, nltk, ftfy, emoji, torch, transformers, sklearn, tqdm\n"
                       "if not torch.cuda.is_available():\n    raise RuntimeError('A CUDA GPU is required for this pipeline.')")
        if "environment" not in completed:
            complete("environment")
        if "upload" not in completed:
            upload_inputs(client, checkpoint / "inputs.zip", checkpoint, remote_workspace)
            complete("upload")

        print("4/6: Encode references if needed; save the bank locally", flush=True)
        remote_bank = remote_workspace / "reference_bank" if reuse_bank else remote_workspace / "reference_banks" / settings["reference_version"]
        if not reuse_bank:
            if "bank_encoded" not in completed:
                arguments = ["python", str(remote_workspace / "src/build_reference_bank.py"),
                             "--reference-corpus", str(remote_workspace / "reference_corpus"),
                             "--output", str(remote_bank), "--checkpoint", BERT_CHECKPOINT,
                             "--revision", MODEL_REVISION, "--context-tokens", str(CONTEXT_TOKENS_EACH_SIDE), "--device", "cuda"]
                client.execute(f'''from pathlib import Path
import subprocess, shutil
if not Path({str(remote_bank / 'bank_manifest.json')!r}).is_file():
    subprocess.run({arguments!r}, cwd={str(remote_workspace)!r}, check=True)
if not Path({str(remote_bank) + '.zip'!r}).is_file():
    shutil.make_archive({str(remote_bank)!r}, "zip", root_dir={str(remote_bank.parent)!r}, base_dir={remote_bank.name!r})
''')
                complete("bank_encoded")
            if "bank_downloaded" not in completed:
                download_archive(client, Path(str(remote_bank) + ".zip"), paths["bank_zip"], checkpoint)
                if not paths["reference_bank"].exists():
                    extract_validated_archive(paths["bank_zip"], paths["reference_bank"], bank_validator)
                bank_validator(paths["reference_bank"])
                complete("bank_downloaded")

        print("5/6: Run discovery and candidate embeddings", flush=True)
        remote_run = remote_workspace / "artifacts" / paths["run"].name
        if "discovery" not in completed:
            arguments = ["python", str(remote_workspace / "src/run_reproduction.py"),
                         "--input", str(remote_workspace / "inputs/discovery.ndjson"),
                         "--reference-bank", str(remote_bank), "--output", str(remote_run),
                         "--max-posts", str(MAX_DISCOVERY_POSTS), "--batch-size", str(CANDIDATE_BATCH_SIZE),
                         "--sampling-method", settings["discovery_sampling_method"],
                         "--sampling-random-seed", str(settings["discovery_random_seed"]),
                         "--device", "cuda", "--category", settings["category"]]
            client.execute(f'''from pathlib import Path
import subprocess
if not Path({str(remote_run / 'run_info.json')!r}).is_file():
    subprocess.run({arguments!r}, cwd={str(remote_workspace)!r}, check=True)
''')
            complete("discovery")
        if "run_archived" not in completed:
            client.execute(f'''import shutil
shutil.copyfile({str(remote_workspace / 'pipeline_settings.json')!r}, {str(remote_run / 'pipeline_settings.json')!r})
shutil.copyfile({str(remote_workspace / 'run_pipeline.py')!r}, {str(remote_run / 'pipeline_launcher.py')!r})
shutil.make_archive({str(remote_run)!r}, "zip", root_dir={str(remote_run.parent)!r}, base_dir={remote_run.name!r})
''')
            complete("run_archived")

        print("6/6: Download, verify and extract results", flush=True)
        if "results" not in completed:
            download_archive(client, Path(str(remote_run) + ".zip"), paths["run_zip"], checkpoint)
            if not paths["run"].exists():
                extract_validated_archive(paths["run_zip"], paths["run"], validate_results)
            complete("results")
        validate_results(paths["run"])
        if not EXISTING_COLAB_SESSION and "stopped" not in completed:
            client.command("stop")
            complete("stopped")
    except BaseException:
        print(f"Pipeline checkpoint retained at {checkpoint}\n"
              f"Resume with: .venv/bin/python run_pipeline.py --category {settings['category']} --resume {checkpoint}",
              file=sys.stderr, flush=True)
        raise
    shutil.rmtree(checkpoint)
    print(f"Completed. Open src/Results.ipynb and set RUN_DIRECTORY to:\n{paths['run']}", flush=True)
    return paths["run"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=HATE_CATEGORIES, default=HATE_CATEGORY,
                        help="Select seeds and anchors from vocabulary.py (overrides HATE_CATEGORY)")
    parser.add_argument("--check", action="store_true", help="Validate settings/paths without preparing data or contacting Colab")
    parser.add_argument("--resume", type=Path, help="Resume a retained artifacts/.pipeline-* checkpoint using its existing Colab session")
    arguments = parser.parse_args()
    HATE_CATEGORY = arguments.category
    run_pipeline(check_only=arguments.check, resume=arguments.resume)
