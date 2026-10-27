"""Build a reference corpus automatically from Raiders anchor matches; no BERT.

The HTML parser below is copied unchanged from Prepare_Raiders_Posts.ipynb.
It is intentionally separate from raiders.parse_post_html, whose behavior differs.
Never reclean saved text after matching: occurrence offsets address that exact text.
"""

from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import inspect
import json
import os
from pathlib import Path
import random
import re
import tempfile

import pandas as pd


class ReadablePostHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text_fragments = []

    def handle_starttag(self, tag_name, attributes):
        # adds a newline for the tags as below, else nothing
        if tag_name in {"br", "p", "div", "li"}:
            self.text_fragments.append("\n")

    def handle_endtag(self, tag_name):
        # adds a newline for the tags as below, else nothing
        if tag_name in {"p", "div", "li"}:
            self.text_fragments.append("\n")

    def handle_data(self, text_fragment):
        self.text_fragments.append(text_fragment)


def extract_readable_post_text(post_html):
    post_html_parser = ReadablePostHTMLParser()
    post_html_parser.feed(post_html)
    post_html_parser.close()
    # join text fragments into single string
    readable_post_text = "".join(post_html_parser.text_fragments)
    # remove pst-reference markers. could keep them for later tho?
    readable_post_text = re.sub(r">>[0-9]+\b", " ", readable_post_text)
    # replace whitespaces other than newlines with a single space
    readable_post_text = re.sub(r"[^\S\n]+", " ", readable_post_text)
    return "\n".join( #NB this keeps the quote label
        text_line.strip() for text_line in readable_post_text.splitlines()
    ).strip()


def compile_phrase_pattern(phrase):
    return re.compile(
        r"(?<!\w)"
        + r"\s+".join(re.escape(phrase_word) for phrase_word in phrase.split())
        + r"(?!\w)",
        flags=re.IGNORECASE,
    )


POST_COLUMNS = ["board", "post_id", "thread_id", "source_line", "timestamp",
                "raw_post", "raw_html", "text", "text_sha256", "normalized_text_sha256"]
OCCURRENCE_COLUMNS = ["occurrence_id", "post_id", "thread_id", "anchor",
                      "matched_text", "start", "end"]
SELECTION_ASSUMPTION = (
    "Anchor matches in the designated reference corpus are assumed suitable reference "
    "usages for this experiment. They are not manually or LLM validated hate labels."
)
SAMPLING_RULE = (
    "Per anchor, shuffle sorted thread IDs with a fixed per-anchor seed; shuffle "
    "occurrences within each thread; retain one occurrence per thread and one normalized "
    "post text per anchor, up to the cap. Skip duplicate-only threads; never pad rare anchors. "
    "A null cap removes the numerical limit, but retains thread/text diversity rules."
)


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path, write):
    """Replace a complete file only after its new contents have been flushed."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="",
                                         dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            write(output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_json(path, value):
    _atomic_write(path, lambda output: json.dump(value, output, indent=2, ensure_ascii=False))


def _read_csv(path):
    # Preserve literal strings such as "null".
    return pd.read_csv(path, keep_default_na=False)


def _coverage(posts, occurrences, anchors):
    text_keys = posts.set_index("post_id")["normalized_text_sha256"].to_dict()
    rows = []
    for anchor in anchors:
        matches = occurrences.loc[occurrences["anchor"].eq(anchor)]
        n_posts = matches["post_id"].nunique()
        unique_texts = len({text_keys[p] for p in matches["post_id"]})
        rows.append({"anchor": anchor, "occurrences": len(matches),
                     "posts": n_posts, "threads": matches["thread_id"].nunique(),
                     "distinct_normalized_texts": unique_texts,
                     "repeated_text_posts": n_posts - unique_texts})
    return pd.DataFrame(rows)


def validate_offsets(posts, occurrences):
    if posts["post_id"].duplicated().any() or occurrences["occurrence_id"].duplicated().any():
        raise ValueError("Duplicate saved post or occurrence IDs.")
    lookup = posts.set_index("post_id")
    for row in occurrences.itertuples(index=False):
        post = lookup.loc[row.post_id]
        if (row.thread_id != post["thread_id"] or not 0 <= row.start < row.end <= len(post["text"])
                or post["text"][row.start:row.end] != row.matched_text):
            raise ValueError(f"Saved span or thread mismatch: {row.occurrence_id}")


def _extract(source_path, anchors, category, progress_every):
    patterns = {name: [compile_phrase_pattern(v) for v in variants]
                for name, variants in anchors.items()}
    posts, occurrences = [], []
    seen_posts, seen_threads = set(), set()
    stats = {"threads_scanned": 0, "posts_scanned": 0, "empty_text_posts": 0}
    with source_path.open("rb") as source:
        for line_no, line in enumerate(source, 1):
            thread = json.loads(line)
            if not isinstance(thread, dict) or not isinstance(thread.get("posts"), list):
                raise ValueError(f"Expected thread with posts on line {line_no}.")
            raw_posts = thread["posts"]
            if any(not isinstance(p, dict) for p in raw_posts):
                raise ValueError(f"Invalid post on line {line_no}.")
            openers = [p for p in raw_posts if p.get("resto") == 0]
            if len(openers) != 1:
                raise ValueError(f"Expected one thread opener on line {line_no}.")
            thread_id = int(openers[0]["no"])
            if thread_id in seen_threads:
                raise ValueError(f"Repeated source thread ID: {thread_id}")
            seen_threads.add(thread_id)
            stats["threads_scanned"] += 1
            for post in raw_posts:
                post_id = int(post["no"])
                if post_id in seen_posts or post.get("resto") not in {0, thread_id}:
                    raise ValueError(f"Repeated post ID or inconsistent thread: {post_id}")
                seen_posts.add(post_id)
                stats["posts_scanned"] += 1
                raw_html = post.get("com")
                if raw_html is None:
                    raw_html = ""
                if not isinstance(raw_html, str):
                    raise ValueError(f"Non-text com field in post {post_id}")
                text = extract_readable_post_text(raw_html)
                if not text:
                    stats["empty_text_posts"] += 1
                    continue
                matches = {}
                for anchor, variants in patterns.items():
                    for pattern in variants:
                        for match in pattern.finditer(text):
                            # Duplicate variants identifying the same span count once.
                            matches[(anchor, match.start(), match.end())] = match.group()
                if not matches:
                    continue
                text_hash = sha256(text.encode()).hexdigest()
                normalized = " ".join(text.casefold().split())
                posts.append({"board": "pol", "post_id": post_id, "thread_id": thread_id,
                              "source_line": line_no, "timestamp": post.get("time"),
                              "raw_post": post, "raw_html": raw_html, "text": text,
                              "text_sha256": text_hash,
                              "normalized_text_sha256": sha256(normalized.encode()).hexdigest()})
                for (anchor, start, end), matched_text in matches.items():
                    identity = ["pol", post_id, category, anchor, text_hash, start, end]
                    occurrence_id = sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
                    occurrences.append(dict(zip(OCCURRENCE_COLUMNS, [
                        occurrence_id, post_id, thread_id, anchor, matched_text, start, end])))
            if progress_every and line_no % progress_every == 0:
                print(f"Scanned {line_no:,} reference threads; {len(occurrences):,} occurrences.",
                      flush=True)
    return (pd.DataFrame(posts, columns=POST_COLUMNS),
            pd.DataFrame(occurrences, columns=OCCURRENCE_COLUMNS), stats)


def select_reference_occurrences(posts, occurrences, anchors, max_per_anchor, random_seed):
    """Select reference spans automatically, with no annotation or approval gate."""
    text_keys = posts.set_index("post_id")["normalized_text_sha256"].to_dict()
    selected = []
    for anchor in anchors:
        rows = occurrences.loc[occurrences["anchor"].eq(anchor)]
        groups = {key: list(group.sort_values("occurrence_id").itertuples(index=False))
                  for key, group in rows.groupby("thread_id")}
        rng = random.Random(f"{random_seed}:{anchor}")
        threads = sorted(groups)
        rng.shuffle(threads)
        used_texts = set()
        for thread_id in threads:
            candidates = groups[thread_id]
            rng.shuffle(candidates)
            for row in candidates:
                key = text_keys[row.post_id]
                if key not in used_texts:
                    selected.append(row.occurrence_id)
                    used_texts.add(key)
                    break
            if max_per_anchor is not None and len(used_texts) >= max_per_anchor:
                break
    return occurrences.set_index("occurrence_id").loc[selected].reset_index()


def prepare_references(source_path, output_dir, anchors, category, max_per_anchor=15,
                       random_seed=42, progress_every=2000):
    """Extract, select and save references, or reload a matching frozen corpus.

    Uses only the designated reference NDJSON and anchor vocabulary. Saves original
    metadata, exact readable text and selected occurrence offsets; no inferred hate
    labels or vectors. Coverage is returned for notebook display, not used as a gate.
    Reusing an output directory verifies the source, settings and saved files.
    """
    source_path, output_dir = Path(source_path).resolve(), Path(output_dir).resolve()
    if source_path.suffix != ".ndjson":
        raise ValueError("Use the reference thread NDJSON, not a discovery CSV.")
    if not anchors or any(not variants or any(not isinstance(v, str) or not v.strip()
                                             for v in variants) for variants in anchors.values()):
        raise ValueError("Anchors need nonempty variant lists.")
    if max_per_anchor is not None and (type(max_per_anchor) is not int or max_per_anchor < 1):
        raise ValueError("Use a positive integer reference cap, or None.")
    if type(random_seed) is not int:
        raise ValueError("Use an integer random seed.")
    parser_source = "\n".join(inspect.getsource(f) for f in
                              (ReadablePostHTMLParser, extract_readable_post_text, compile_phrase_pattern))
    source_hash = file_sha256(source_path)
    config = {"format_version": 2, "source_path": str(source_path),
              "source_sha256": source_hash, "source_bytes": source_path.stat().st_size,
              "category": category, "anchors": anchors,
              "parser_sha256": sha256(parser_source.encode()).hexdigest(),
              "max_per_anchor": max_per_anchor, "random_seed": random_seed,
              "sampling_rule": SAMPLING_RULE, "selection_assumption": SELECTION_ASSUMPTION,
              "manual_validation": False, "llm_validation": False}
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists():
        if not manifest_path.is_file():
            raise ValueError("Output folder has no completed reference manifest. Choose a new folder.")
        manifest = json.loads(manifest_path.read_text())
        if manifest["config"] != config:
            raise ValueError("Source, vocabulary, parser or settings differ. Choose a new reference folder.")
        for name, expected in manifest["file_checksums"].items():
            if file_sha256(output_dir / name) != expected:
                raise ValueError(f"Saved reference file changed: {name}. Do not edit text or offsets.")
        with (output_dir / "reference_posts.jsonl").open(encoding="utf-8") as source:
            reference_posts = pd.DataFrame([json.loads(line) for line in source], columns=POST_COLUMNS)
        reference_occurrences = _read_csv(output_dir / "reference_occurrences.csv")
        validate_offsets(reference_posts, reference_occurrences)
        coverage = pd.DataFrame(manifest["coverage"])
        print("Loaded saved reference corpus; no resampling or annotation required.")
        return reference_posts, reference_occurrences, coverage, manifest

    posts, occurrences, stats = _extract(source_path, anchors, category, progress_every)
    if file_sha256(source_path) != source_hash:
        raise ValueError("Reference source changed during extraction; nothing saved.")
    validate_offsets(posts, occurrences)
    coverage = _coverage(posts, occurrences, anchors)
    reference_occurrences = select_reference_occurrences(
        posts, occurrences, anchors, max_per_anchor, random_seed)
    reference_posts = posts.loc[posts["post_id"].isin(reference_occurrences["post_id"])].copy()
    validate_offsets(reference_posts, reference_occurrences)
    selected_coverage = _coverage(reference_posts, reference_occurrences, anchors).set_index("anchor")
    for column in ("occurrences", "posts", "threads"):
        coverage[f"selected_{column}"] = coverage["anchor"].map(selected_coverage[column])

    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "reference_posts.jsonl").open("x", encoding="utf-8") as output:
        for post in reference_posts.to_dict(orient="records"):
            output.write(json.dumps(post, ensure_ascii=False) + "\n")
    reference_occurrences.to_csv(output_dir / "reference_occurrences.csv", index=False, mode="x")
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                "scan_counts": stats,
                "reference_counts": {"posts": len(reference_posts), "occurrences": len(reference_occurrences)},
                "coverage": coverage.to_dict(orient="records"),
                "file_checksums": {name: file_sha256(output_dir / name) for name in
                                   ("reference_posts.jsonl", "reference_occurrences.csv")},
                "limitations": [SELECTION_ASSUMPTION,
                                "Archive-order reference source; not random sampling",
                                "Full-text hashes do not detect shared passages or near duplicates",
                                "Discovery must come from a separate source partition; old pilot CSVs overlap"]}
    _write_json(manifest_path, manifest)
    print(f"Saved {len(reference_posts):,} reference posts and {len(reference_occurrences):,} occurrences to {output_dir}")
    return reference_posts, reference_occurrences, coverage, manifest
