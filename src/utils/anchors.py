import re

import pandas as pd


# define anchor terms
# as defined in the paper

#TODO: keep in a csv or smth

# Keep the existing runner import compatible with the shared vocabulary.
from .vocabulary import anchor_terms_by_hate_category

anchor_terms = anchor_terms_by_hate_category["antisemitism"]


def find_anchor_occurrences(embedding_posts, anchor_terms):
    # locate anchor occurences and record their positions
    # in our discovery corpus

    occurrences = []

    for anchor, variants in anchor_terms.items():
        for variant in variants:
            # match the expression case-insensitively
            # also allow for whitespace before and after the expression
            pattern = re.compile(
                r"(?<!\w)"
                + r"\s+".join(re.escape(word) for word in variant.split())
                + r"(?!\w)",
                flags=re.IGNORECASE,
            )

            for post_id, text in embedding_posts["Text for BERT"].items():
                if not isinstance(text, str):
                    continue

                for match in pattern.finditer(text):
                    occurrences.append({
                        "post_id": post_id,
                        "anchor": anchor,
                        "matched_text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                    })

    anchor_occurrences = pd.DataFrame(
        occurrences,
        columns=["post_id", "anchor", "matched_text", "start", "end"],
    )

    return (occurrences, anchor_occurrences)
