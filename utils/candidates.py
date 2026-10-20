import pandas as pd
from nltk.util import bigrams, trigrams


def build_ngrams(data):
    # construct unigram, bigram and trigram features
    # def: a n-gram is a sequence of n adjacent words

    # word-token tokenisation is necessary for this step,
    # but was done when creating the 'word token' col

    data["Unigrams"] = data["Word Tokens"].apply(
        lambda tokens: [token["word"] for token in tokens]
    )
    data["Bigrams"] = data["Unigrams"].apply(
        lambda words: list(bigrams(words))
    )

    data["Trigrams"] = data["Unigrams"].apply(
        lambda words: list(trigrams(words))
    )

    return data


# update:
def build_tagged_ngrams(word_records, size):
    """Group words with their saved sentence-level POS tags"""
    tagged_words = [
        (record["word"], record["pos"])
        for record in word_records
    ]

    return [
        tagged_words[start:start + size]
        for start in range(len(tagged_words) - size + 1)
    ]


def build_pos_ngrams(data):
    # Unigrams are strings; bigrams and trigrams are tuples of words
    data["Unigrams POS"] = data["Word Tokens"].apply(
        lambda records: build_tagged_ngrams(records, 1)
    )

    data["Bigrams POS"] = data["Word Tokens"].apply(
        lambda records: build_tagged_ngrams(records, 2)
    )

    data["Trigrams POS"] = data["Word Tokens"].apply(
        lambda records: build_tagged_ngrams(records, 3)
    )

    return data


# Provisional POS rules: these are not verified HMApp rules.
# Keep single nouns/adjectives, and noun/adjective phrases ending in a noun.
# Penn Treebank tags: NN* = nouns; JJ* = adjectives.
def keep_candidate(tagged_gram):
    tags = [tag for word, tag in tagged_gram]
    if not tags:
        return False
    # Reject candidates containing tokens made entirely of symbols. NB: could be worth keeping?
    if any(
        not any(char.isalnum() for char in word)
        for word, tag in tagged_gram
    ):
        return False
    if len(tags) == 1:
        return tags[0].startswith(("NN", "JJ"))
    return (
        all(tag.startswith(("NN", "JJ")) for tag in tags)
        and tags[-1].startswith("NN")
    )


def filter_candidates(tagged_candidates):
    kept = []
    for gram in tagged_candidates:
        if keep_candidate(gram):
            words = tuple(word for word, tag in gram)
            kept.append(words[0] if len(words) == 1 else words)
    return kept


def apply_pos_filter(data):
    # Preserve original candidates and store survivors separately.
    for column in ("Unigrams", "Bigrams", "Trigrams"):

        data[f"{column} Filtered"] = data[f"{column} POS"].apply(filter_candidates)
        before = data[column].apply(len).sum()
        after = data[f"{column} Filtered"].apply(len).sum()
        print(f"{column}: {before} occurrences before filtering, {after} after")

    return data


def extract_tfidf_candidates(data):
    ### TF-IDF ###
    # For every term t, compute its TF-IDF in every post, then keep its maximum TF-IDF over all posts
    # then compute the mean of all these maxima and keep all terms where their TF-IDF is greater than the mean

    from sklearn.feature_extraction.text import TfidfVectorizer

    # step 1

    # transform post into lists of string to then be used by the vectorizer
    # -> convert n-gram tuples into expression strings
    documents = data.apply(
        lambda row: (
            row["Unigrams Filtered"]
            + [" ".join(gram) for gram in row["Bigrams Filtered"]]
            + [" ".join(gram) for gram in row["Trigrams Filtered"]]
        ),
        axis=1,
    ).tolist()

    # we already calculated the n-grams, so we can use them directly
    # in our vectorizer
    vectorizer = TfidfVectorizer(analyzer=lambda candidates: candidates)
    # calculate TFIDF scores
    tfidf = vectorizer.fit_transform(documents)

    terms = vectorizer.get_feature_names_out()

    # HMApp's threshold calculation
    max_scores = tfidf.max(axis=0).toarray().ravel()
    delta = max_scores.mean()

    selected_terms = terms[max_scores > delta]

    return (documents, vectorizer, tfidf, terms, max_scores, delta, selected_terms,)


def find_candidate_occurrences(data, selected_terms):
    candidate_terms = set(selected_terms)
    candidate_records = []

    ngram_columns = [
        (1, "Unigrams Filtered"),
        (2, "Bigrams Filtered"),
        (3, "Trigrams Filtered"),
    ]

    for post_id, post in data.iterrows():
        text = post["Text for BERT"]
        word_records = post["Word Tokens"]

        for size, column in ngram_columns:
            # Keep expressions that survived both POS filtering and TF-IDF
            accepted_candidates = set()

            for gram in post[column]:
                if size == 1:
                    candidate = gram
                else:
                    candidate = " ".join(gram)

                if candidate in candidate_terms:
                    accepted_candidates.add(candidate)

            # Locate those expressions in the already-tokenized post
            for index in range(len(word_records) - size + 1):
                candidate_words = word_records[index:index + size]

                # Check the tags belonging to this particular occurrence.
                tagged_candidate = [
                    (record["word"], record["pos"])
                    for record in candidate_words
                ]

                if not keep_candidate(tagged_candidate):
                    continue

                words = [record["word"] for record in candidate_words]
                candidate = " ".join(words)

                if candidate not in accepted_candidates:
                    continue

                # Retrieve the saved positions of the constituent words
                word_spans = []
                for record in candidate_words:
                    word_spans.append((record["start"], record["end"]))

                start = candidate_words[0]["start"]
                end = candidate_words[-1]["end"]

                candidate_records.append({
                    "post_id": post_id,
                    "candidate": candidate,
                    "matched_text": text[start:end],
                    "start": start,
                    "end": end,
                    "word_spans": tuple(word_spans),
                })

    candidate_occurrences = pd.DataFrame(
        candidate_records,
        columns=[
            "post_id",
            "candidate",
            "matched_text",
            "start",
            "end",
            "word_spans",
        ],
    )

    matched_candidates = set(candidate_occurrences["candidate"])
    unmatched_candidates = sorted(candidate_terms - matched_candidates)

    return (
        candidate_terms,
        candidate_records,
        ngram_columns,
        candidate_occurrences,
        matched_candidates,
        unmatched_candidates,
    )
