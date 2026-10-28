import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarities(candidate_embeddings, anchor_embeddings):
    cosine_similarities = cosine_similarity(candidate_embeddings,anchor_embeddings)

    return cosine_similarities


def aggregate_candidate_counts(candidate_post_scores):
    # rank from highest to lowest relevance
    candidate_scores = (
        candidate_post_scores
        .groupby("candidate", as_index=False)
        .agg(
            relevance=("relevance", "mean"),
            n_posts=("post_id", "nunique"),
            n_occurrences=("n_occurrences", "sum"), # NB: I added number of occurences/posts because
            # I suspect that a lot of these pass the final threshold only bc appear once
        )
    )

    return candidate_scores


def score_fixed_reference_bank(candidate_embeddings, candidate_metadata, reference_embeddings,
                               reference_metadata, batch_size=4096):
    """Top-five mean per candidate/post, with independent source and text identities."""
    from .reference_bank import _check_vectors

    _check_vectors(candidate_embeddings, candidate_metadata, "Candidates")
    _check_vectors(reference_embeddings, reference_metadata, "References")
    if candidate_embeddings.shape[1] != reference_embeddings.shape[1]:
        raise ValueError("Candidate/reference vector dimensions differ.")
    if len(reference_embeddings) < 5:
        raise ValueError("HMApp requires at least five eligible reference vectors; this bank has fewer.")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Scoring batch_size must be positive.")
    for column in ("post_key", "thread_key", "normalized_text_sha256"):
        if column not in candidate_metadata or column not in reference_metadata:
            raise ValueError(f"Cannot verify separation without {column} metadata.")
        if set(candidate_metadata[column]) & set(reference_metadata[column]):
            raise ValueError(f"Reference/discovery overlap in {column}; fix retrieval before scoring.")
    if candidate_metadata.duplicated(["candidate", "post_key"]).any():
        raise ValueError("Expected one vector per candidate/source post.")
    score_dtype = np.float32 if candidate_embeddings.dtype == reference_embeddings.dtype == np.float32 else np.float64
    scores = np.empty(len(candidate_embeddings), dtype=score_dtype)
    for batch_start in range(0, len(candidate_embeddings), batch_size):
        batch_end = batch_start + batch_size
        similarities = compute_cosine_similarities(candidate_embeddings[batch_start:batch_end], reference_embeddings)
        scores[batch_start:batch_end] = np.sort(similarities, axis=1)[:, -5:].mean(axis=1)
    post_scores = candidate_metadata.copy()
    post_scores["relevance"] = scores
    return post_scores


def summarize_rankings(candidate_post_scores, candidate_occurrences):
    """Keep the original relevance ranking, with frequency and contiguity diagnostics."""
    candidate_scores = aggregate_candidate_counts(candidate_post_scores)
    thread_counts = candidate_post_scores.groupby("candidate")["thread_key"].nunique()
    contiguous_counts = candidate_occurrences.groupby("candidate")["is_contiguous"].sum()
    candidate_scores["n_threads"] = candidate_scores["candidate"].map(thread_counts)
    candidate_scores["n_contiguous_occurrences"] = candidate_scores["candidate"].map(contiguous_counts).astype(int)
    candidate_scores["n_noncontiguous_occurrences"] = (
        candidate_scores["n_occurrences"] - candidate_scores["n_contiguous_occurrences"]
    )
    # Tied scores use candidate text for reproducible display order.
    rankings = candidate_scores.sort_values(["relevance", "candidate"], ascending=[False, True]).reset_index(drop=True)
    rankings.insert(0, "original_rank", np.arange(1, len(rankings) + 1))
    relevance_threshold = float(rankings["relevance"].mean())
    rankings["passes_relevance"] = rankings["relevance"] > relevance_threshold
    return rankings, relevance_threshold
