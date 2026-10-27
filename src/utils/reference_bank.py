"""Read prepared reference spans and persist reusable contextual reference vectors.

No model is loaded here. Saved text is never cleaned again, and metadata row i
always describes vector row i. The full Raiders archive is not needed on Colab.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from .reference_review import file_sha256, validate_offsets


CORPUS_FILES = ("reference_posts.jsonl", "reference_occurrences.csv")
POST_METADATA = [
    "post_id",
    "board",
    "thread_id",
    "source_line",
    "text_sha256",
    "normalized_text_sha256",
]


def _read_posts(path):
    with Path(path).open(encoding="utf-8") as source:
        return pd.DataFrame([json.loads(line) for line in source])


def _verify_files(folder, checksums):
    for name, expected in checksums.items():
        if Path(name).name != name:
            raise ValueError(f"Expected a plain filename in manifest: {name}")
        if file_sha256(folder / name) != expected:
            raise ValueError(f"Checksum mismatch: {folder / name}")


def load_reference_corpus(corpus_dir):
    """Validate the exported corpus before any model download or inference."""
    folder = Path(corpus_dir)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config"].get("format_version") != 2:
        raise ValueError(
            "Expected an automatic reference corpus from Prepare_Reference.ipynb."
        )
    checksums = manifest["file_checksums"]
    if set(checksums) != set(CORPUS_FILES):
        raise ValueError("Reference manifest must identify both exported corpus files.")
    _verify_files(folder, checksums)
    posts = _read_posts(folder / "reference_posts.jsonl")
    occurrences = pd.read_csv(
        folder / "reference_occurrences.csv", keep_default_na=False
    )
    if posts.empty or occurrences.empty:
        raise ValueError(
            "The reference corpus is empty; prepare matching contexts first."
        )
    if not set(POST_METADATA + ["text"]).issubset(posts.columns):
        raise ValueError("Reference posts are missing source IDs or text hashes.")
    required = {
        "occurrence_id",
        "post_id",
        "thread_id",
        "anchor",
        "matched_text",
        "start",
        "end",
    }
    if not required.issubset(occurrences.columns):
        raise ValueError("Reference occurrences are missing IDs or span metadata.")
    for column in ("post_id", "thread_id", "start", "end"):
        if not pd.api.types.is_integer_dtype(occurrences[column]):
            raise ValueError(f"Occurrence {column} must contain integer values.")
    if set(posts["post_id"]) != set(occurrences["post_id"]):
        raise ValueError("Saved reference posts and occurrence post IDs do not match.")
    if occurrences.duplicated(["post_id", "anchor", "start", "end"]).any():
        raise ValueError(
            "Duplicate spans would be dropped by embed_anchors; fix preparation first."
        )
    validate_offsets(posts, occurrences)
    for post in posts.itertuples(index=False):
        if not isinstance(post.text, str) or not post.text.strip():
            raise ValueError(f"Empty or non-text reference post: {post.post_id}")
        if sha256(post.text.encode()).hexdigest() != post.text_sha256:
            raise ValueError(f"Text checksum mismatch: {post.post_id}")
        normalized = " ".join(post.text.casefold().split())
        if sha256(normalized.encode()).hexdigest() != post.normalized_text_sha256:
            raise ValueError(f"Normalized text checksum mismatch: {post.post_id}")
    if not set(occurrences["anchor"]).issubset(manifest["config"]["anchors"]):
        raise ValueError("Occurrence anchor not present in the preparation vocabulary.")
    if manifest["reference_counts"] != {
        "posts": len(posts),
        "occurrences": len(occurrences),
    }:
        raise ValueError("Saved counts do not match the reference corpus.")
    return posts, occurrences, manifest


def _check_vectors(vectors, metadata, label):
    if (
        not isinstance(vectors, np.ndarray)
        or vectors.ndim != 2
        or vectors.shape[0] != len(metadata)
        or vectors.shape[1] == 0
        or len(metadata) == 0
    ):
        raise ValueError(
            f"{label}: vectors must be a nonempty matrix aligned with metadata."
        )
    if not np.issubdtype(vectors.dtype, np.floating) or not np.isfinite(vectors).all():
        raise ValueError(f"{label}: vectors must contain finite floating-point values.")
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError(f"{label}: zero vectors cannot be used for cosine scoring.")


def save_reference_bank(
    bank_dir,
    corpus_dir,
    preparation_manifest,
    posts,
    occurrence_metadata,
    occurrence_vectors,
    reference_metadata,
    reference_vectors,
    embedding_config,
    run_info,
):
    """Save a complete new bank without overwriting an existing bank directory.

    Retains occurrence vectors and the existing helper's averages per anchor/post.
    The current preparation selects one occurrence per anchor/post, so those two
    vector sets are currently equivalent apart from row ordering.
    """
    bank_dir, corpus_dir = Path(bank_dir), Path(corpus_dir)
    if bank_dir.exists():
        raise FileExistsError(
            f"Reference bank already exists: {bank_dir}. Use a new bank name."
        )
    current_posts, current_occurrences, current_manifest = load_reference_corpus(
        corpus_dir
    )
    if current_manifest != preparation_manifest:
        raise ValueError("Preparation manifest changed while embedding.")
    pd.testing.assert_frame_equal(
        posts.reset_index(drop=True), current_posts.reset_index(drop=True)
    )
    # The existing embedding helper must preserve exactly the selected occurrence rows.
    expected = current_occurrences.reset_index(drop=True)
    actual = occurrence_metadata.reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[expected.columns], expected)
    _check_vectors(occurrence_vectors, actual, "Occurrences")
    _check_vectors(reference_vectors, reference_metadata, "References")
    if occurrence_vectors.shape[1] != reference_vectors.shape[1]:
        raise ValueError("Occurrence and reference embedding dimensions differ.")
    if reference_vectors.shape[1] != embedding_config["hidden_size"]:
        raise ValueError("Vector dimension does not match the model configuration.")
    if reference_metadata.duplicated(["anchor", "post_id"]).any():
        raise ValueError("Expected one scoring reference per anchor/post pair.")
    groups = actual.groupby(["anchor", "post_id"], sort=False)
    if len(reference_metadata) != groups.ngroups:
        raise ValueError(
            "Reference metadata does not cover all selected anchor/post pairs."
        )
    for row_number, row in enumerate(reference_metadata.itertuples(index=False)):
        group = groups.get_group((row.anchor, row.post_id))
        if row.n_occurrences != len(group):
            raise ValueError("Reference occurrence count mismatch.")
        np.testing.assert_allclose(
            reference_vectors[row_number],
            occurrence_vectors[group.index].mean(axis=0),
            rtol=1e-5,
            atol=1e-6,
        )

    def enrich(metadata):
        result = metadata.reset_index(drop=True).copy()
        result.insert(0, "embedding_row", np.arange(len(result)))
        extra = [c for c in POST_METADATA if c == "post_id" or c not in result.columns]
        result = result.merge(
            posts[extra], on="post_id", how="left", sort=False, validate="many_to_one"
        )
        if result["embedding_row"].tolist() != list(range(len(result))):
            raise ValueError("Metadata enrichment changed embedding row order.")
        return result

    occurrence_metadata = enrich(actual)
    reference_metadata = enrich(reference_metadata)
    coverage = (
        reference_metadata.groupby("anchor")
        .agg(
            reference_vectors=("embedding_row", "size"),
            posts=("post_id", "nunique"),
            threads=("thread_id", "nunique"),
            selected_occurrences=("n_occurrences", "sum"),
        )
        .reindex(preparation_manifest["config"]["anchors"], fill_value=0)
    )
    bank_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".reference_bank_", dir=bank_dir.parent
    ) as temporary:
        staged = Path(temporary) / "bank"
        staged.mkdir()
        np.save(staged / "reference_vectors.npy", reference_vectors, allow_pickle=False)
        np.save(
            staged / "anchor_occurrence_embeddings.npy",
            occurrence_vectors,
            allow_pickle=False,
        )
        reference_metadata.to_csv(staged / "reference_metadata.csv", index=False)
        occurrence_metadata.to_csv(
            staged / "anchor_occurrence_metadata.csv", index=False
        )
        coverage.to_csv(staged / "anchor_coverage.csv", index_label="anchor")
        for name in CORPUS_FILES:
            shutil.copyfile(corpus_dir / name, staged / name)
        (staged / "preparation_manifest.json").write_text(
            json.dumps(preparation_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "format_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "category": preparation_manifest["config"]["category"],
            "embedding_config": embedding_config,
            "run_info": run_info,
            "reference_shape": list(reference_vectors.shape),
            "occurrence_shape": list(occurrence_vectors.shape),
            "reference_dtype": str(reference_vectors.dtype),
            "preparation_config": preparation_manifest["config"],
            "metadata_alignment": "CSV embedding_row i describes NPY row i (zero-based)",
            "file_checksums": {
                p.name: file_sha256(p) for p in sorted(staged.iterdir())
            },
        }
        (staged / "bank_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if bank_dir.exists():
            raise FileExistsError(f"Reference bank appeared during export: {bank_dir}")
        staged.rename(bank_dir)
    return manifest


def load_reference_bank(bank_dir, expected_embedding_config=None):
    """Load scoring vectors, aligned metadata and source posts without running BERT.

    Supply expected model/revision/context/pooling settings when wiring discovery
    scoring. Discovery post/thread/text-overlap checks remain the caller's task.
    """
    folder = Path(bank_dir)
    manifest = json.loads((folder / "bank_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported reference bank format.")
    required = {
        "reference_vectors.npy",
        "reference_metadata.csv",
        "reference_posts.jsonl",
        "anchor_occurrence_embeddings.npy",
        "anchor_occurrence_metadata.csv",
        "reference_occurrences.csv",
        "preparation_manifest.json",
        "anchor_coverage.csv",
    }
    if set(manifest["file_checksums"]) != required:
        raise ValueError("Incomplete reference bank manifest.")
    for key, expected in (expected_embedding_config or {}).items():
        if manifest["embedding_config"].get(key) != expected:
            raise ValueError(
                f"Reference/discovery embedding configuration mismatch: {key}"
            )
    _verify_files(folder, manifest["file_checksums"])
    vectors = np.load(folder / "reference_vectors.npy", allow_pickle=False)
    metadata = pd.read_csv(folder / "reference_metadata.csv", keep_default_na=False)
    posts = _read_posts(folder / "reference_posts.jsonl")
    _check_vectors(vectors, metadata, "References")
    if (
        list(vectors.shape) != manifest["reference_shape"]
        or str(vectors.dtype) != manifest["reference_dtype"]
    ):
        raise ValueError("Reference vectors do not match the bank manifest.")
    if metadata["embedding_row"].tolist() != list(range(len(vectors))):
        raise ValueError("Reference metadata rows are out of order.")
    if not set(metadata["post_id"]).issubset(posts["post_id"]):
        raise ValueError("Reference source posts are missing.")
    return vectors, metadata, posts, manifest
