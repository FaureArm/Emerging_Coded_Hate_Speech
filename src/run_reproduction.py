"""One discovery experiment against a previously encoded Raiders reference bank.

From any working directory:
    python /path/to/project/src/run_reproduction.py --max-posts 1000

Normally invoked by the configurable root run_pipeline.py launcher.
This script never builds reference vectors from discovery posts.
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import random
import subprocess
from time import perf_counter

from utils.vocabulary import HATE_CATEGORIES, vocabulary_for_category


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=HATE_CATEGORIES, default="antisemitism",
                        help="Select both retrieval seeds and anchors from utils/vocabulary.py")
    parser.add_argument("--input", type=project_path, default=PROJECT_ROOT / "data/raiders/raiders_first_1GB.ndjson")
    parser.add_argument("--reference-bank", type=project_path, default=None,
                        help="Encoded bank; defaults to artifacts/reference_banks/<category>_v1")
    parser.add_argument("--output", type=project_path, default=None)
    parser.add_argument("--max-posts", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16, help="Candidate BERT window batch size")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parsed = parser.parse_args(arguments)
    if parsed.reference_bank is None:
        parsed.reference_bank = PROJECT_ROOT / "artifacts/reference_banks" / f"{parsed.category}_v1"
    if parsed.output is None:
        parsed.output = PROJECT_ROOT / "artifacts" / (
            f"raiders_{parsed.category}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    return parsed


def enrich_candidate_metadata(metadata, posts):
    source_columns = ["post_id", "post_key", "thread_id", "thread_key", "board", "normalized_text_sha256"]
    enriched = metadata.merge(posts.reset_index(drop=True)[source_columns], on="post_id",
                              how="left", sort=False, validate="many_to_one")
    if len(enriched) != len(metadata) or enriched["post_key"].isna().any():
        raise ValueError("Candidate metadata lost its source identity.")
    if enriched["post_id"].tolist() != metadata["post_id"].tolist():
        raise ValueError("Source metadata join changed embedding row order.")
    return enriched


def code_identity():
    from utils.reference_review import file_sha256

    def git_output(*arguments):
        try:
            result = subprocess.run(["git", "-C", str(PROJECT_ROOT), *arguments], capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else None
        except FileNotFoundError:
            return None
    return {
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "source_files_sha256": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
            for path in sorted((PROJECT_ROOT / "src").rglob("*.py"))
        },
    }


def run_experiment(arguments):
    import numpy as np
    import pandas as pd
    from utils.artifacts import save_discovery_run
    from utils.candidates import prepare_candidates
    from utils.discovery import retrieve_discovery_posts
    from utils.preprocessing import load_nlp_tools
    from utils.reference_bank import load_reference_bank, validate_discovery_bank
    from utils.reference_review import file_sha256
    from utils.scoring import score_fixed_reference_bank, summarize_rankings

    category = arguments.category
    anchors, seed_words = vocabulary_for_category(category)

    if arguments.max_posts < 1 or arguments.batch_size < 1:
        raise ValueError("--max-posts and --batch-size must be positive.")
    if arguments.output.exists():
        raise FileExistsError(f"Run directory already exists: {arguments.output}. Choose a new --output.")
    if not (arguments.reference_bank / "bank_manifest.json").is_file():
        raise FileNotFoundError(
            f"No encoded reference bank at {arguments.reference_bank}. "
            "Use run_pipeline.py to prepare and encode it on Colab, or set --reference-bank. "
            "Discovery will not generate replacement reference vectors."
        )
    started_at = datetime.now(timezone.utc).isoformat()
    experiment_started = perf_counter()
    stage_times = {}
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    print(f"Category: {category}; {len(anchors)} anchors; retrieval seeds: {seed_words}", flush=True)
    print("1/5: Validating the fixed reference bank", flush=True)
    stage_started = perf_counter()
    reference_vectors, reference_metadata, reference_posts, bank_manifest = load_reference_bank(arguments.reference_bank)
    reference_metadata = validate_discovery_bank(reference_vectors, reference_metadata, reference_posts, bank_manifest,
                                                 anchors, category=category)
    bank_identifier = file_sha256(arguments.reference_bank / "bank_manifest.json")
    reference_coverage = pd.read_csv(arguments.reference_bank / "anchor_coverage.csv", keep_default_na=False)
    stage_times["reference_loading"] = perf_counter() - stage_started

    print("2/5: Retrieving independent discovery posts", flush=True)
    stage_started = perf_counter()
    posts, seed_summary, retrieval_info = retrieve_discovery_posts(
        arguments.input, seed_words, reference_posts, arguments.max_posts
    )
    stage_times["retrieval"] = perf_counter() - stage_started
    print(f"Retrieved {len(posts):,} posts in {posts.thread_id.nunique():,} threads.", flush=True)

    print("3/5: Existing lexical/POS/TF-IDF candidate extraction", flush=True)
    stage_started = perf_counter()
    stop_words, sentence_tokenizer, word_tokenizer = load_nlp_tools()
    occurrences, tfidf_candidates, stage_counts = prepare_candidates(posts, stop_words, sentence_tokenizer, word_tokenizer)
    stage_times["candidate_preparation"] = perf_counter() - stage_started

    print("4/5: Contextual candidate embeddings", flush=True)
    from utils.embeddings import EMBEDDING_PROCEDURE, embed_candidates, load_embedding_model
    import torch

    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    # No training, and no reduced-precision matrix multiplication in this baseline.
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    stage_started = perf_counter()
    embedding_configuration = bank_manifest["embedding_config"]
    model, tokenizer, device = load_embedding_model(embedding_configuration, arguments.device)
    stage_times["model_loading"] = perf_counter() - stage_started
    stage_started = perf_counter()
    (
        occurrence_metadata, _, _, _, candidate_vectors, candidate_metadata,
    ) = embed_candidates(
        occurrences, posts.rename(columns={"text": "Text for BERT"}), arguments.batch_size,
        model, tokenizer, device, embedding_configuration["context_tokens_each_side"],
    )
    occurrence_metadata = enrich_candidate_metadata(occurrence_metadata, posts)
    candidate_metadata = enrich_candidate_metadata(candidate_metadata, posts)
    candidate_metadata.insert(0, "embedding_row", np.arange(len(candidate_metadata)))
    stage_times["candidate_embedding"] = perf_counter() - stage_started

    print("5/5: Top-five reference scoring and export", flush=True)
    stage_started = perf_counter()
    candidate_post_scores = score_fixed_reference_bank(candidate_vectors, candidate_metadata, reference_vectors, reference_metadata)
    rankings, relevance_threshold = summarize_rankings(candidate_post_scores, occurrence_metadata)
    stage_times["scoring"] = perf_counter() - stage_started
    stage_counts["candidate_post_embeddings"] = len(candidate_metadata)
    stage_counts["ranked_candidates"] = len(rankings)
    stage_counts["relevance_survivors"] = int(rankings["passes_relevance"].sum())
    run_info = {
        "format_version": 1, "status": "complete", "started_at_utc": started_at,
        "category": category, "seed_words": seed_words, "anchor_terms": anchors,
        "seed_provenance": f"Experimental seeds from vocabulary.py for {category}; not Wu's published seed selection",
        "requested_max_posts": arguments.max_posts, "retrieval": retrieval_info,
        "reference_bank": {"path": str(arguments.reference_bank), "identifier_sha256": bank_identifier,
                           "manifest": bank_manifest},
        "embedding_config": embedding_configuration, "candidate_embedding_procedure": EMBEDDING_PROCEDURE,
        "candidate_batch_size": arguments.batch_size, "device": str(device), "random_seed": random_seed,
        "tfidf_threshold": stage_counts["tfidf_threshold"], "relevance_threshold": relevance_threshold,
        "nearest_reference_vectors": 5, "ngram_policy": stage_counts["ngram_policy"],
        "frequency_filter_applied": False, "stage_seconds": stage_times,
        "calculation_seconds": perf_counter() - experiment_started,
        "finished_calculation_at_utc": datetime.now(timezone.utc).isoformat(),
        "tf32": False, "determinism": "eval mode and fixed seeds; cross-device bitwise equality is not guaranteed",
        "software": {"packages": {package: version(package) for package in
                      ("numpy", "pandas", "torch", "transformers", "tokenizers", "nltk", "scikit-learn", "emoji")},
                     **code_identity()},
        "limitations": [
            "Consecutive historical source slice; retrieval is seed-conditioned and stops in file order",
            "Reference anchor matches are assumed suitable contexts, not validated hate labels",
            "Thread holdout is defined by this bank's selected source threads, not its entire sampling pool",
            "Normalized full-text equality excludes copies, not shared passages or paraphrases",
            "Legacy stopword-before-ngram extraction retained; noncontiguous occurrences are flagged",
            "Existing noun/adjective POS rules are provisional, not verified exact Wu rules",
            "Reference preparation selects one occurrence per anchor/thread and caps contexts; candidate repeats are averaged",
            "Distinct posts/threads do not establish independent adoption; similarity is not a hate label",
        ],
    }
    save_discovery_run(arguments.output, run_info, posts, seed_summary, reference_coverage,
                       stage_counts, tfidf_candidates, occurrence_metadata, candidate_vectors,
                       candidate_metadata, candidate_post_scores, rankings)
    print(f"Completed run: {arguments.output}", flush=True)
    return arguments.output


def main(arguments=None):
    return run_experiment(parse_arguments(arguments))


if __name__ == "__main__":
    main()
