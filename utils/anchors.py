import re

import pandas as pd


# define anchor terms
# as defined in the paper

#TODO: keep in a csv or smth

anchor_terms = {
    "kike": ["kike"],
    "blood libel": ["blood libel"],
    "the goyim know": ["the goyim know"],
    "great replacement": ["great replacement"],
    "holocaust/gas chambers": ["holocaust", "gas chambers"],
    "holohoax": ["holohoax"],
    "jew coup": ["jew coup"],
    "jew down": ["jew down"],
    "jewish feature": ["jewish feature"],
    "jewish figure": ["jewish figure"],
    "jewish lighting": ["jewish lighting"],
    "jewish power": ["jewish power"],
    "kosher tax": ["kosher tax"],
    "laughing jew": ["laughing jew"],
    "nazi symbols": ["nazi symbols"],
    "protocols of the elders of zion": ["protocols of the elders of zion"],
    "qanon": ["qanon"],
    "smirking merchant": ["smirking merchant"],
    "zionism is racism": ["zionism is racism"],
    "zionist occupied government": ["zionist occupied government"],
}


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
