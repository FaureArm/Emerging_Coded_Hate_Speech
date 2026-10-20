import re

import numpy as np


def gram_to_string(gram):
    if isinstance(gram, str):
        return gram.casefold()
    return " ".join(gram).casefold()


def extract_vocabulary(columns, data):
    vocabulary = set()

    for column in columns:
        for grams in data[column]:
            for gram in grams:
                vocabulary.add(gram_to_string(gram))

    return vocabulary


def recover_vocabularies(data, tfidf_candidates, candidate_scores):
    vocab_before_pos = extract_vocabulary([
        "Unigrams",
        "Bigrams",
        "Trigrams",
    ], data)

    vocab_after_pos = extract_vocabulary([
        "Unigrams Filtered",
        "Bigrams Filtered",
        "Trigrams Filtered",
    ], data)

    vocab_after_tfidf = set(
        tfidf_candidates.loc[
            tfidf_candidates["selected"],
            "candidate",
        ]
    )

    vocab_scored = set(candidate_scores["candidate"])

    print("Before POS:", len(vocab_before_pos))
    print("After POS:", len(vocab_after_pos))
    print("After TF-IDF:", len(vocab_after_tfidf))
    print("Scored candidates:", len(vocab_scored))

    return (vocab_before_pos, vocab_after_pos, vocab_after_tfidf, vocab_scored,)


def count_posts_with_term(term, data):
    # Match a complete expression, allowing varying whitespace.
    pattern = re.compile(
        r"(?<!\w)"
        + r"\s+".join(re.escape(w) for w in term.split())
        + r"(?!\w)",
        flags=re.IGNORECASE,
    )

    return sum(
        bool(pattern.search(text))
        for text in data["Text for BERT"]
        if isinstance(text, str)
    )


def check_raw_occurrences(gold, data):
    gold["raw_posts"] = gold["term"].apply(
        count_posts_with_term,
        args=(data,),
    )

    gold["in_raw"] = gold["raw_posts"] > 0

    return gold


def track_terms(
    gold,
    vocab_before_pos,
    vocab_after_pos,
    vocab_after_tfidf,
    vocab_scored,
    candidate_scores,
):
    gold["generated"] = gold["term"].isin(
        vocab_before_pos
    )

    gold["after_pos"] = gold["term"].isin(
        vocab_after_pos
    )

    gold["after_tfidf"] = gold["term"].isin(
        vocab_after_tfidf
    )

    gold["scored"] = gold["term"].isin(
        vocab_scored
    )

    # Global HMApp ranking.
    ranking = (
        candidate_scores
        .sort_values("relevance", ascending=False)
        .reset_index(drop=True)
    )

    ranking["rank"] = np.arange(1, len(ranking) + 1)

    rank_lookup = ranking.set_index("candidate")["rank"]
    score_lookup = ranking.set_index("candidate")["relevance"]

    gold["rank"] = gold["term"].map(rank_lookup)
    gold["relevance"] = gold["term"].map(score_lookup)

    # Published HMApp relevance cutoff.
    threshold = candidate_scores["relevance"].mean()

    gold["passes_hmapp"] = (
        gold["relevance"] > threshold
    )

    return (gold, ranking, rank_lookup, score_lookup, threshold)


def failure_stage(row):
    if not row["in_raw"]:
        return "Absent from raw corpus"

    if not row["generated"]:
        return "N-gram generation / preprocessing"

    if not row["after_pos"]:
        return "POS filtering"

    if not row["after_tfidf"]:
        return "TF-IDF filtering"

    if not row["scored"]:
        return "Occurrence matching / embedding"

    if not row["passes_hmapp"]:
        return "Semantic relevance cutoff"

    return "Retained by HMApp"


def identify_failure_stages(gold):
    gold["outcome"] = gold.apply(
        failure_stage,
        axis=1,
    )

    return gold


def candidate_survival_rates(positives, stages):
    rates = {}
    for stage in stages:
        rate = positives[stage].mean()
        rates[stage] = rate
    return rates
