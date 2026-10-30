"""Internal Colab worker: encode a prepared reference corpus once.

Normally invoked by run_pipeline.py. No local notebook preparation is required.
"""

import argparse
from pathlib import Path

import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter

# colab exec executes code in a notebook kernel, not necessarily from src/.
if Path("src/utils").is_dir() and str(Path("src").resolve()) not in sys.path:
    sys.path.insert(0, str(Path("src").resolve()))

from utils.reference_bank import load_reference_corpus, load_reference_bank, save_reference_bank
from utils.reference_review import file_sha256


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default="google-bert/bert-base-uncased")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--context-tokens", type=int, default=64)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    return parser.parse_args(arguments)


def main(arguments=None):
    arguments = parse_arguments(arguments)
    started_at = datetime.now(timezone.utc).isoformat()
    started = perf_counter()
    corpus_dir = arguments.reference_corpus.resolve()
    bank_dir = arguments.output.resolve()
    archive_path = Path(str(bank_dir) + ".zip")
    if bank_dir.exists() or archive_path.exists():
        raise FileExistsError(f"Bank or ZIP already exists at {bank_dir}. Choose a new bank name.")
    if type(arguments.context_tokens) is not int or arguments.context_tokens < 0:
        raise ValueError("arguments.context_tokens must be a nonnegative integer.")
    if arguments.device not in {"auto", "cuda", "cpu"}:
        raise ValueError("arguments.device must be auto, cuda or cpu.")

    posts, occurrences, preparation_manifest = load_reference_corpus(corpus_dir)
    print(f"Validated {len(posts):,} posts and {len(occurrences):,} anchor occurrences.", flush=True)
    print("Using the exact saved text and offsets; no rematching or cleaning.", flush=True)

    # Heavy imports and downloads occur only when this script is explicitly run.
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModel, AutoTokenizer
    from utils import embeddings

    device = torch.device("cuda" if arguments.device == "auto" and torch.cuda.is_available()
                          else "cpu" if arguments.device == "auto" else arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this runtime.")
    print(f"Loading {arguments.checkpoint} on {device}...", flush=True)
    resolved_revision = HfApi().model_info(arguments.checkpoint, revision=arguments.revision).sha
    if not resolved_revision:
        raise ValueError("Could not resolve a Hugging Face model commit; use a Hub model checkpoint.")
    tokenizer = AutoTokenizer.from_pretrained(arguments.checkpoint, revision=resolved_revision, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with character offsets is required.")
    model = AutoModel.from_pretrained(arguments.checkpoint, revision=resolved_revision).to(device)
    model.eval()

    # Original Raiders post IDs, not fresh DataFrame row numbers.
    embedding_posts = posts.set_index("post_id")[["text"]].rename(columns={"text": "Text for BERT"})
    print("Embedding selected anchor spans in context...", flush=True)
    (
        occurrence_metadata, _, occurrence_vectors, _, _,
        reference_vectors, reference_metadata, _,
    ) = embeddings.embed_anchors(occurrences, embedding_posts, model, tokenizer, device, arguments.context_tokens)

    embedding_config = {
        "procedure": embeddings.EMBEDDING_PROCEDURE,
        "checkpoint": arguments.checkpoint,
        "requested_revision": arguments.revision,
        "resolved_revision": resolved_revision,
        "tokenizer_checkpoint": arguments.checkpoint,
        "tokenizer_revision": resolved_revision,
        "context_tokens_each_side": arguments.context_tokens,
        "layer": "last_hidden_state",
        "pooling": "mean of subword tokens overlapping the selected anchor span; special tokens excluded",
        "aggregation": "mean of selected occurrence vectors per (anchor, post_id)",
        "text_transform": "none; exact saved reference text",
        "model_eval": True,
        "model_dtype": str(next(model.parameters()).dtype),
        "hidden_size": int(model.config.hidden_size),
        "max_position_embeddings": int(model.config.max_position_embeddings),
        "embedding_helper_sha256": file_sha256(Path(embeddings.__file__)),
    }
    run_info = {
        "started_at_utc": started_at,
        "finished_embedding_at_utc": datetime.now(timezone.utc).isoformat(),
        "loading_and_embedding_seconds": perf_counter() - started,
        "device": str(device),
        "package_versions": {name: version(name) for name in
                             ("torch", "transformers", "numpy", "pandas", "tqdm", "tokenizers", "huggingface-hub")},
    }
    save_reference_bank(bank_dir, corpus_dir, preparation_manifest, posts,
                        occurrence_metadata, occurrence_vectors,
                        reference_metadata, reference_vectors, embedding_config, run_info)
    # Check the persisted matrices, metadata, configuration and checksums before zipping.
    saved_vectors, saved_metadata, _, _ = load_reference_bank(bank_dir, embedding_config)
    print(f"Saved and verified {saved_vectors.shape[0]:,} reference vectors of dimension {saved_vectors.shape[1]}.")
    print(f"Anchors represented: {saved_metadata['anchor'].nunique()} / {len(preparation_manifest['config']['anchors'])}")
    print(f"Reference bank: {bank_dir}")
    shutil.make_archive(str(bank_dir), "zip", root_dir=bank_dir.parent, base_dir=bank_dir.name)
    print(f"Download archive: {archive_path} ({archive_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Finished in {perf_counter() - started:.1f} seconds.")


if __name__ == "__main__":
    main()
