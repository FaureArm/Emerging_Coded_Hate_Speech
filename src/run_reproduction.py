"""Run the calculations from Reproduce.ipynb and export their results.
HOW TO RUN
Meant to be run in a colab cli session

colab new -s reproduction --gpu T4

colab install -s reproduction pandas numpy nltk ftfy emoji torch transformers scikit-learn tqdm

zip -r reproduction_inputs.zip utils \
  "data/Unmasking Antisemitism SRI Data Set - Reporting Layer.csv" \
  -x '*/__pycache__/*'

colab upload -s reproduction reproduction_inputs.zip content/reproduction_inputs.zip

echo 'import zipfile; zipfile.ZipFile("/content/reproduction_inputs.zip").extractall("/content")' | colab exec -s reproduction

colab exec -s reproduction -f run_reproduction.py --timeout 7200

Once finished running, download the artifacts with:
colab download -s reproduction content/artifacts.zip .artifacts/artifacts_colab.zip

"""

from pathlib import Path

DATA_SOURCE = Path("data/raiders/raiders_antisemitism.csv")
ARTIFACTS_DIR = Path("/content/artifacts_raiders")

checkpoint = "google-bert/bert-base-uncased"  # could try different models, for ex cased
context_tokens = 64
candidate_batch_size = 16


import json
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter

import ftfy
import nltk
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.tokenize import NLTKWordTokenizer, PunktTokenizer
from transformers import AutoModel, AutoTokenizer

from utils.anchors import anchor_terms, find_anchor_occurrences
from utils.artifacts import save_artifacts
from utils.candidates import (
    apply_pos_filter,
    build_ngrams,
    build_pos_ngrams,
    extract_tfidf_candidates,
    find_candidate_occurrences,
)
from utils.embeddings import configure_embeddings, embed_anchors, embed_candidates
from utils.preprocessing import (
    cleanup,
    configure_preprocessing,
    tokenize_with_positions,
)
from utils.scoring import (
    aggregate_candidate_counts,
    aggregate_candidate_scores,
    compute_cosine_similarities,
    compute_relevance_scores,
    filter_by_occurrences,
    filter_by_relevance,
    rank_candidates,
)


def main():
    started_at = datetime.now(timezone.utc).isoformat()
    start_time = perf_counter()
    ART = ARTIFACTS_DIR.resolve()

    if not DATA_SOURCE.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_SOURCE.resolve()}. "
            "Set DATA_SOURCE to a CSV available on the machine running this script."
        )

    print(f"Data source: {DATA_SOURCE.resolve()}", flush=True)
    print(f"Artifacts folder: {ART}", flush=True)

    # load the data
    df = pd.read_csv(DATA_SOURCE, keep_default_na=False)

    if "Post Text" not in df.columns:
        raise ValueError("The input CSV must contain a 'Post Text' column.")

    data = df.dropna(subset=["Post Text"]).copy()
    if data.empty:
        raise ValueError("The input CSV contains no non-missing post texts.")

    ART.mkdir(parents=True, exist_ok=True)
    print(f"Posts with text: {len(data)} / {len(df)}", flush=True)
    print("1/6: Preprocessing and POS tagging", flush=True)

    # Download the missing NLTK stopwords and punkt_tab resources
    nltk.download("stopwords")
    nltk.download("punkt_tab")

    stop_words = set(stopwords.words("english"))
    sentence_tokenizer = PunktTokenizer("english")
    word_tokenizer = NLTKWordTokenizer()

    # Preserve readable sentences for BERT
    data["Text for BERT"] = data["Post Text"].apply(
        lambda text: ftfy.fix_encoding(text) if isinstance(text, str) else text
    )

    # POS tagging happens during initial tokenization
    nltk.download("averaged_perceptron_tagger_eng")

    configure_preprocessing(stop_words, sentence_tokenizer, word_tokenizer)

    # split into two new cols the different types of data

    # text as is required for BERT
    # aka just cleaned
    data["Text for Candidate Extraction"] = data["Text for BERT"].apply(cleanup)

    # text as is required for TFIDF
    # list of words, with position, stop-words applied, etc
    data["Word Tokens"] = data["Text for Candidate Extraction"].apply(
        tokenize_with_positions
    )

    print("2/6: Candidate generation and TF-IDF", flush=True)
    data = build_ngrams(data)

    # Tag the existing unigram, bigram, and trigram candidates
    nltk.download("averaged_perceptron_tagger_eng")

    data = build_pos_ngrams(data)

    # Each tagged candidate contains (word, POS tag) pairs
    data = apply_pos_filter(data)

    (
        documents,
        vectorizer,
        tfidf,
        terms,
        max_scores,
        delta,
        selected_terms,
    ) = extract_tfidf_candidates(data)

    print(f"{len(selected_terms)} selected terms", flush=True)
    print("3/6: Anchor matching and contextual embeddings", flush=True)

    ### Contextual embedding ###
    # https://github.com/UKPLab/sentence-transformers ??
    ## Anchor terms ##

    # retrieve the entire post text before our preprocessing
    # so that we apply the BERT preprocessing to the entire text isntead
    embedding_posts = data[["Post Text", "Text for BERT"]].copy()

    embedding_posts["Text for BERT"] = embedding_posts["Post Text"].apply(
        lambda text: ftfy.fix_encoding(text) if isinstance(text, str) else text
    )

    occurrences, anchor_occurrences = find_anchor_occurrences(
        embedding_posts, anchor_terms
    )

    # applying BERT context to anchor words
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if anchor_occurrences.empty:
        raise ValueError("No anchor occurrences found. Check the matching cell first.")

    print(f"Device: {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model = AutoModel.from_pretrained(checkpoint).to(device)
    model.eval()  # Disable training behavior such as dropout.

    configure_embeddings(model, tokenizer, device, context_tokens)

    (
        anchor_occurrence_metadata,
        occurrence_vectors,
        anchor_occurrence_embeddings,
        reference_vectors,
        reference_records,
        anchor_embeddings,
        anchor_metadata,
        anchor_reference_post_ids,
    ) = embed_anchors(anchor_occurrences, embedding_posts)

    print(f"Anchor/post embeddings: {anchor_embeddings.shape}", flush=True)
    print(
        "Anchors without matches:",
        sorted(set(anchor_terms) - set(anchor_metadata["anchor"])),
    )

    # to visualise which anchors dominate the reference database
    anchor_coverage = (
        anchor_metadata.groupby("anchor")
        .agg(
            n_vectors=("post_id", "size"),
            n_posts=("post_id", "nunique"),
        )
        .reindex(anchor_terms.keys(), fill_value=0)
        .sort_values("n_vectors", ascending=False)
    )

    print("4/6: Candidate matching and contextual embeddings", flush=True)
    (
        candidate_terms,
        candidate_records,
        ngram_columns,
        candidate_occurrences,
        matched_candidates,
        unmatched_candidates,
    ) = find_candidate_occurrences(data, selected_terms)

    configure_embeddings(model, tokenizer, device, context_tokens)
    model.eval()

    (
        candidate_occurrence_metadata,
        candidate_occurrence_embeddings,
        candidate_post_vectors,
        candidate_post_records,
        candidate_embeddings,
        candidate_metadata,
    ) = embed_candidates(candidate_occurrences, embedding_posts, candidate_batch_size)

    print(f"Candidate/post embeddings: {candidate_embeddings.shape}", flush=True)
    print("Candidates without matches:", unmatched_candidates)
    print("5/6: Cosine similarities, relevance scores and filtering", flush=True)

    cosine_similarities = compute_cosine_similarities(
        candidate_embeddings, anchor_embeddings
    )

    (
        candidate_post_ids,
        anchor_post_ids,
        candidate_ids_column,
        anchor_ids_row,
        same_post,
        eligible_similarities,
        eligible_anchor_counts,
        sorted_similarities,
        top_five_similarities,
        relevance_scores,
        candidate_post_scores,
    ) = compute_relevance_scores(
        cosine_similarities, candidate_metadata, anchor_metadata, embedding_posts
    )

    candidate_scores = aggregate_candidate_scores(candidate_post_scores)
    ranked_candidates = rank_candidates(candidate_scores)
    relevance_threshold, filtered_candidates = filter_by_relevance(
        candidate_scores, ranked_candidates
    )
    filtered_candidates_relevance = filtered_candidates

    save_artifacts(
        ART,
        data,
        terms,
        max_scores,
        delta,
        candidate_post_scores,
        candidate_scores,
        candidate_embeddings,
        candidate_metadata,
        anchor_embeddings,
        anchor_metadata,
    )
    ranked_candidates.to_csv(ART / "ranked_candidates.csv", index=False)
    filtered_candidates_relevance.to_csv(
        ART / "filtered_candidates_relevance.csv", index=False
    )

    candidate_scores = aggregate_candidate_counts(candidate_post_scores)
    ranked_candidates = rank_candidates(candidate_scores)

    (
        relevance_threshold,
        median_occurrences,
        occurrence_threshold,
        filtered_candidates,
    ) = filter_by_occurrences(candidate_scores, ranked_candidates)
    filtered_candidates_frequency = filtered_candidates

    print(
        "6/6: Exporting additional results and creating the download archive",
        flush=True,
    )

    candidate_scores.to_csv(ART / "candidate_scores_with_counts.csv", index=False)
    ranked_candidates.to_csv(ART / "ranked_candidates_with_counts.csv", index=False)
    filtered_candidates_frequency.to_csv(
        ART / "filtered_candidates_frequency.csv", index=False
    )
    anchor_coverage.to_csv(ART / "anchor_coverage.csv", index_label="anchor")

    anchor_occurrence_metadata.to_pickle(ART / "anchor_occurrence_metadata.pkl")
    candidate_occurrence_metadata.to_pickle(ART / "candidate_occurrence_metadata.pkl")
    np.save(ART / "anchor_occurrence_embeddings.npy", anchor_occurrence_embeddings)
    np.save(
        ART / "candidate_occurrence_embeddings.npy", candidate_occurrence_embeddings
    )
    np.save(ART / "cosine_similarities.npy", cosine_similarities)
    np.save(ART / "eligible_similarities.npy", eligible_similarities)
    np.save(ART / "eligible_anchor_counts.npy", eligible_anchor_counts)
    np.save(ART / "top_five_similarities.npy", top_five_similarities)

    run_info = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "calculation_and_export_seconds": perf_counter() - start_time,
        "data_source": str(DATA_SOURCE.resolve()),
        "artifacts_dir": str(ART),
        "checkpoint": checkpoint,
        "context_tokens": context_tokens,
        "candidate_batch_size": candidate_batch_size,
        "device": str(device),
        "input_rows": len(df),
        "posts_used": len(data),
        "selected_terms": len(selected_terms),
        "unmatched_candidates": unmatched_candidates,
        "anchor_terms": anchor_terms,
        "anchor_post_embeddings": len(anchor_metadata),
        "candidate_post_embeddings": len(candidate_metadata),
        "tfidf_threshold": float(delta),
        "relevance_threshold": float(relevance_threshold),
        "median_occurrences": float(median_occurrences),
        "minimum_occurrences": int(occurrence_threshold),
        "exclude_same_post": True,
        "exclude_identical_text": True,
        "package_versions": {
            package: version(package)
            for package in (
                "emoji",
                "ftfy",
                "nltk",
                "numpy",
                "pandas",
                "scikit-learn",
                "torch",
                "transformers",
                "tqdm",
            )
        },
    }
    (ART / "run_info.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    archive_path = shutil.make_archive(
        str(ART), "zip", root_dir=ART.parent, base_dir=ART.name
    )
    print(f"Artifacts folder: {ART}", flush=True)
    print(f"Download archive: {archive_path}", flush=True)
    print(f"Finished in {perf_counter() - start_time:.1f} seconds.", flush=True)


if __name__ == "__main__":
    main()
