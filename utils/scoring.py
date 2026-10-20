import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarities(candidate_embeddings, anchor_embeddings):
    cosine_similarities = cosine_similarity(candidate_embeddings,anchor_embeddings)

    return cosine_similarities


def compute_relevance_scores(cosine_similarities, candidate_metadata, anchor_metadata):
    # NB: since we dont have a corpus large enough to separate anchor occurrences
    # and candidate discovery, instead we exclude comparisons between anchors and
    # candidate that are from the same post, to avoid bias in cosine similarity
    # this solution would ideally only be temporary


    # Retrieve the post IDs corresponding to the embedding rows
    candidate_post_ids = candidate_metadata["post_id"].to_numpy()
    anchor_post_ids = anchor_metadata["post_id"].to_numpy()

    # we create a boolean matrix to keep track of if
    # an anchor and  candidate are from the same post
    # ->candidate IDs as a column
    # ->anchor IDs as a row
    candidate_ids_column = candidate_post_ids.reshape(-1, 1)
    anchor_ids_row = anchor_post_ids.reshape(1, -1)

    same_post = candidate_ids_column == anchor_ids_row


    # exclude similarities between embeddings from the same post
    eligible_similarities = cosine_similarities.copy()
    eligible_similarities[same_post] = -np.inf


    # check that every candidate has at least five eligible references
    # not very practical, could try to implement a func to avg over the
    # number of references (checkin if -np.inf and keeping this somewhere)
    # but not worth the hassle for now.
    eligible_anchor_counts = len(anchor_post_ids) - same_post.sum(axis=1)

    if (eligible_anchor_counts < 5).any():
        raise ValueError(
            "Some candidates have fewer than five eligible anchor vectors."
        )

    # this is a result of our temp solution to disregard similarities
    # from anchors and candidates from same posts. Normally, we would only
    # have the lines after to execute otherwise

    # -- - - - -- - - -

    # average the five highest eligible similarities
    sorted_similarities = np.sort(eligible_similarities, axis=1)
    top_five_similarities = sorted_similarities[:, -5:]
    relevance_scores = top_five_similarities.mean(axis=1)


    # create scores to their candidate/post pairs
    candidate_post_scores = candidate_metadata.copy()
    candidate_post_scores["relevance"] = relevance_scores

    return (
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
    )


def aggregate_candidate_scores(candidate_post_scores):
    # gives us the mean of each candidate's relevance across posts
    candidate_scores = (
        candidate_post_scores
        .groupby("candidate", as_index=False)["relevance"]
        .mean()
    )

    return candidate_scores


def rank_candidates(candidate_scores):
    # rank from highest to lowest relevance
    ranked_candidates = candidate_scores.sort_values(
        "relevance",
        ascending=False,
    ).reset_index(drop=True)

    return ranked_candidates


def filter_by_relevance(candidate_scores, ranked_candidates):
    # filter all candidates below the mean of all candidates
    relevance_threshold = candidate_scores["relevance"].mean()

    filtered_candidates = ranked_candidates.loc[
        ranked_candidates["relevance"] > relevance_threshold
    ].reset_index(drop=True)

    print(f"Relevance threshold: {relevance_threshold:.4f}")
    print(f"Candidates before filtering: {len(ranked_candidates)}")
    print(f"Candidates after filtering: {len(filtered_candidates)}")

    return (relevance_threshold, filtered_candidates)


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


def filter_by_occurrences(candidate_scores, ranked_candidates):
    # filter all candidates below the mean of all candidates, as well as min requirement of number of occurences

    # relevance filter
    relevance_threshold = candidate_scores["relevance"].mean()

    # number of occurences filter
    median_occurrences = candidate_scores["n_occurrences"].median()
    occurrence_threshold = max(2, int(np.ceil(median_occurrences)))

    filtered_candidates = ranked_candidates.loc[
        (ranked_candidates["relevance"] > relevance_threshold)
        & (ranked_candidates["n_occurrences"] >= occurrence_threshold)
    ].reset_index(drop=True)

    print(f"Relevance threshold: {relevance_threshold:.4f}")
    print(f"Minimum occurrences: {occurrence_threshold}")
    print(f"Candidates before filtering: {len(ranked_candidates)}")
    print(f"Candidates after filtering: {len(filtered_candidates)}")

    return (
        relevance_threshold,
        median_occurrences,
        occurrence_threshold,
        filtered_candidates,
    )
