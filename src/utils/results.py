"""Offline artifact inspection. No torch, transformers, NLTK downloads or BERT."""

from hashlib import sha256
from html import escape
import json
from pathlib import Path
import re

import pandas as pd


def phrase_pattern(phrases):
    alternatives = [r"\s+".join(re.escape(word) for word in phrase.split()) for phrase in phrases]
    if not alternatives:
        return re.compile(r"(?!x)x")
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.IGNORECASE)


def bar_chart_html(values, title):
    """Small offline bars, using the same HTML approach as Prepare_Reference.ipynb."""
    maximum = max(float(values.max()), 1.0) if len(values) else 1.0
    rows = []
    for label, count in values.items():
        rows.append(
            f'<tr><td style="padding-right:12px">{escape(str(label))}</td>'
            f'<td style="width:320px"><div style="background:#16878b;height:14px;'
            f'width:{100 * float(count) / maximum:.1f}%"></div></td>'
            f'<td>{float(count):g}</td></tr>'
        )
    return "<b>" + escape(title) + "</b><table>" + "".join(rows) + "</table>"


def inspection_view(rankings, excluded_phrases=(), minimum_posts=1, minimum_threads=1):
    """Display filtering only; original scores, ranks and threshold flags stay unchanged."""
    excluded_pattern = phrase_pattern(excluded_phrases)
    excluded = rankings["candidate"].map(lambda term: bool(excluded_pattern.search(term)))
    selected = rankings.loc[
        ~excluded & rankings["n_posts"].ge(minimum_posts) & rankings["n_threads"].ge(minimum_threads)
    ].copy().reset_index(drop=True)
    selected.insert(0, "display_position", range(1, len(selected) + 1))
    return selected


def highlight_word_spans(text, word_spans):
    """Escape source HTML, then highlight the actual candidate words, including gapped n-grams."""
    pieces, previous_end = [], 0
    for start, end in sorted(set(tuple(span) for span in word_spans)):
        if not previous_end <= start < end <= len(text):
            raise ValueError("Invalid or overlapping candidate word offsets.")
        pieces.extend([escape(text[previous_end:start]), "<mark>", escape(text[start:end]), "</mark>"])
        previous_end = end
    pieces.append(escape(text[previous_end:]))
    return '<div style="white-space:pre-wrap">' + "".join(pieces) + "</div>"


def load_run(run_directory):
    """Load small tables only; embeddings stay on disk and the reference bank is not required."""
    folder = Path(run_directory)
    if not (folder / "run_info.json").is_file():
        raise FileNotFoundError(f"No completed run_info.json in {folder}; choose a completed discovery run.")
    run_info = json.loads((folder / "run_info.json").read_text(encoding="utf-8"))
    if run_info.get("format_version") != 1 or run_info.get("status") != "complete":
        raise ValueError("Unsupported or incomplete discovery run.")
    tables = {}
    filenames = ["discovery_posts", "retrieval_summary", "reference_coverage", "tfidf_candidates",
                 "candidate_occurrences", "candidate_metadata", "candidate_post_scores", "ranked_candidates"]
    for name in filenames:
        path = folder / f"{name}.csv"
        if sha256(path.read_bytes()).hexdigest() != run_info["artifact_checksums"].get(path.name):
            raise ValueError(f"Saved table changed since export: {path.name}")
        tables[name] = pd.read_csv(path, keep_default_na=False)
    for column in ("raw_post", "matched_seeds"):
        tables["discovery_posts"][column] = tables["discovery_posts"][column].map(json.loads)
    tables["candidate_occurrences"]["word_spans"] = tables["candidate_occurrences"]["word_spans"].map(json.loads)
    stage_path = folder / "candidate_stage_counts.json"
    if sha256(stage_path.read_bytes()).hexdigest() != run_info["artifact_checksums"].get(stage_path.name):
        raise ValueError("Candidate stage counts changed since export.")
    stage_counts = json.loads(stage_path.read_text(encoding="utf-8"))
    posts, metadata, occurrences = (tables[name] for name in
                                     ("discovery_posts", "candidate_metadata", "candidate_occurrences"))
    if not posts["post_key"].is_unique or metadata["embedding_row"].tolist() != list(range(len(metadata))):
        raise ValueError("Post identities or embedding row order are invalid.")
    if not set(metadata["post_key"]).issubset(posts["post_key"]) or not set(occurrences["post_key"]).issubset(posts["post_key"]):
        raise ValueError("Candidate metadata refers to missing discovery posts.")
    return run_info, tables, stage_counts
