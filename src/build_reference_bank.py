"""Build BERT reference vectors once from Prepare_Reference.ipynb's saved corpus.

Run from the repository root: .venv/bin/python src/build_reference_bank.py
Relative paths below are resolved from the working directory, normally /content
on Colab. The script refuses to overwrite an existing bank or download archive.

Colab CLI commands (run yourself from the repository root; adjust the category
folder below and the parameters together if using a different prepared corpus):

    colab new -s references --gpu T4
    colab install -s references pandas numpy torch transformers tqdm
    zip -r reference_bank_inputs.zip src/utils \
      data/raiders/reference_corpora/antisemitism_v1 -x '*/__pycache__/*'
    colab upload -s references reference_bank_inputs.zip content/reference_bank_inputs.zip
    echo 'import zipfile; zipfile.ZipFile("/content/reference_bank_inputs.zip").extractall("/content")' | colab exec -s references
    colab exec -s references -f src/build_reference_bank.py --timeout 7200
    colab download -s references content/artifacts/reference_banks/antisemitism_v1.zip artifacts/reference_bank_antisemitism_v1.zip

Only the prepared corpus and helpers are uploaded, not the 1 GB reference source.
The existing discovery runner is not yet connected to this bank.
"""

from pathlib import Path

# -------------------------- Editable parameters --------------------------
REFERENCE_CORPUS_DIR = Path("data/raiders/reference_corpora/antisemitism_v1")
REFERENCE_BANK_DIR = Path("artifacts/reference_banks/antisemitism_v1")
CHECKPOINT = "google-bert/bert-base-uncased"
MODEL_REVISION = "main"  # Resolved to one commit before loading model and tokenizer.
CONTEXT_TOKENS = 64      # Surrounding subword tokens on EACH side, as in the runner.
DEVICE = "auto"         # auto / cuda / cpu
CREATE_ZIP = True
# -------------------------------------------------------------------------

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


def main():
    started_at = datetime.now(timezone.utc).isoformat()
    started = perf_counter()
    corpus_dir = REFERENCE_CORPUS_DIR.resolve()
    bank_dir = REFERENCE_BANK_DIR.resolve()
    archive_path = Path(str(bank_dir) + ".zip")
    if bank_dir.exists() or (CREATE_ZIP and archive_path.exists()):
        raise FileExistsError(f"Bank or ZIP already exists at {bank_dir}. Choose a new bank name.")
    if type(CONTEXT_TOKENS) is not int or CONTEXT_TOKENS < 0:
        raise ValueError("CONTEXT_TOKENS must be a nonnegative integer.")
    if DEVICE not in {"auto", "cuda", "cpu"}:
        raise ValueError("DEVICE must be auto, cuda or cpu.")

    posts, occurrences, preparation_manifest = load_reference_corpus(corpus_dir)
    print(f"Validated {len(posts):,} posts and {len(occurrences):,} anchor occurrences.", flush=True)
    print("Using the exact saved text and offsets; no rematching or cleaning.", flush=True)

    # Heavy imports and downloads occur only when this script is explicitly run.
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModel, AutoTokenizer
    from utils import embeddings

    device = torch.device("cuda" if DEVICE == "auto" and torch.cuda.is_available()
                          else "cpu" if DEVICE == "auto" else DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this runtime.")
    print(f"Loading {CHECKPOINT} on {device}...", flush=True)
    resolved_revision = HfApi().model_info(CHECKPOINT, revision=MODEL_REVISION).sha
    if not resolved_revision:
        raise ValueError("Could not resolve a Hugging Face model commit; use a Hub model checkpoint.")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, revision=resolved_revision, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with character offsets is required.")
    model = AutoModel.from_pretrained(CHECKPOINT, revision=resolved_revision).to(device)
    model.eval()
    embeddings.configure_embeddings(model, tokenizer, device, CONTEXT_TOKENS)

    # Original Raiders post IDs, not fresh DataFrame row numbers.
    embedding_posts = posts.set_index("post_id")[["text"]].rename(columns={"text": "Text for BERT"})
    print("Embedding selected anchor spans in context...", flush=True)
    (
        occurrence_metadata, _, occurrence_vectors, _, _,
        reference_vectors, reference_metadata, _,
    ) = embeddings.embed_anchors(occurrences, embedding_posts)

    embedding_config = {
        "checkpoint": CHECKPOINT,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": resolved_revision,
        "tokenizer_checkpoint": CHECKPOINT,
        "tokenizer_revision": resolved_revision,
        "context_tokens_each_side": CONTEXT_TOKENS,
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
    if CREATE_ZIP:
        shutil.make_archive(str(bank_dir), "zip", root_dir=bank_dir.parent, base_dir=bank_dir.name)
        print(f"Download archive: {archive_path} ({archive_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Finished in {perf_counter() - started:.1f} seconds.")


if __name__ == "__main__":
    main()
