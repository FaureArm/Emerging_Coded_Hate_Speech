import numpy as np
import pandas as pd


def save_artifacts(
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
):
    ART.mkdir(exist_ok=True)

    # Candidate extraction: preserve all TF-IDF values.
    pd.DataFrame({
        "candidate": terms,
        "max_tfidf": max_scores,
        "selected": max_scores > delta,
    }).to_csv(ART / "tfidf_candidates.csv", index=False)

    # Preserve your intermediate lexical/POS results.
    data.to_pickle(ART / "preprocessed_data.pkl")

    # Scoring outputs.
    candidate_post_scores.to_csv(
        ART / "candidate_post_scores.csv",
        index=False,
    )

    candidate_scores.to_csv(
        ART / "candidate_scores.csv",
        index=False,
    )

    # Save embeddings and their matching metadata.
    np.save(ART / "candidate_embeddings.npy", candidate_embeddings)
    candidate_metadata.to_csv(
        ART / "candidate_metadata.csv",
        index=False,
    )

    np.save(ART / "anchor_embeddings.npy", anchor_embeddings)
    anchor_metadata.to_csv(
        ART / "anchor_metadata.csv",
        index=False,
    )

    print("Artifacts saved.")


def load_artifacts(ART):
    data = pd.read_pickle(ART / "preprocessed_data.pkl")

    tfidf_candidates = pd.read_csv(
        ART / "tfidf_candidates.csv",
        keep_default_na=False,
    )

    candidate_scores = pd.read_csv(
        ART / "candidate_scores.csv",
        keep_default_na=False,
    )

    candidate_post_scores = pd.read_csv(
        ART / "candidate_post_scores.csv",
        keep_default_na=False,
    )

    candidate_metadata = pd.read_csv(
        ART / "candidate_metadata.csv",
        keep_default_na=False,
    )

    anchor_metadata = pd.read_csv(
        ART / "anchor_metadata.csv"
    )

    return (
        data,
        tfidf_candidates,
        candidate_scores,
        candidate_post_scores,
        candidate_metadata,
        anchor_metadata,
    )
