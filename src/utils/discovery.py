"""Stream Raiders posts while keeping the saved reference threads out of discovery."""

from collections import Counter
from hashlib import sha256
import inspect
import json
from pathlib import Path

import pandas as pd

from .reference_review import (
    ReadablePostHTMLParser, compile_phrase_pattern, extract_readable_post_text,
)


def normalized_text_hash(text):
    return sha256(" ".join(text.casefold().split()).encode("utf-8")).hexdigest()


def parser_hash():
    # Same contract as the saved reference corpus.
    parser_source = "\n".join(inspect.getsource(function) for function in
                              (ReadablePostHTMLParser, extract_readable_post_text, compile_phrase_pattern))
    return sha256(parser_source.encode()).hexdigest()


def validate_reference_posts(reference_posts):
    required = {"board", "post_id", "thread_id", "text", "normalized_text_sha256"}
    if reference_posts.empty or not required.issubset(reference_posts):
        raise ValueError("Reference posts must include board, post/thread IDs, text and text hashes.")
    if set(reference_posts["board"]) != {"pol"}:
        raise ValueError("This Raiders reader supports 4chan /pol/ only; reference board differs.")
    for column in ("post_id", "thread_id"):
        if not pd.api.types.is_integer_dtype(reference_posts[column]) or reference_posts[column].le(0).any():
            raise ValueError(f"Reference {column} must contain positive source IDs.")
    if reference_posts["post_id"].duplicated().any():
        raise ValueError("Duplicate reference source post IDs.")
    if not reference_posts["text"].map(normalized_text_hash).equals(reference_posts["normalized_text_sha256"]):
        raise ValueError("Reference text does not match its normalized hash.")


def retrieve_discovery_posts(source_path, seed_words, reference_posts, maximum_posts=1000):
    """Keep matching posts, excluding entire fixed-bank threads and copied reference text.

    The bank's saved threads are the frozen holdout; no new random split is made.
    IDs are scoped to 4chan /pol/. Source line numbers address this input file.
    Hash only the consumed complete NDJSON lines, not the unused archive suffix.
    """
    source_path = Path(source_path)
    if source_path.suffix != ".ndjson" or not source_path.is_file():
        raise FileNotFoundError(f"Expected a Raiders NDJSON file: {source_path}")
    if type(maximum_posts) is not int or maximum_posts < 1:
        raise ValueError("maximum_posts must be a positive integer.")
    if not seed_words or any(not isinstance(word, str) or not word.strip() for word in seed_words):
        raise ValueError("Provide nonempty retrieval seed phrases.")
    validate_reference_posts(reference_posts)
    reference_thread_ids = set(reference_posts["thread_id"])
    reference_post_ids = set(reference_posts["post_id"])
    reference_text_hashes = set(reference_posts["normalized_text_sha256"])
    seed_patterns = {word: compile_phrase_pattern(word) for word in dict.fromkeys(seed_words)}
    statistics = Counter({name: 0 for name in (
        "source_records_read", "posts_examined", "reference_threads_excluded",
        "reference_thread_posts_excluded", "reference_text_posts_excluded",
        "empty_text_posts", "seed_matched_posts", "selected_posts", "selected_threads",
    )})
    selected_seed_counts = Counter()
    seen_thread_ids, seen_post_ids, copied_text_threads = set(), set(), set()
    selected_posts = []
    consumed_digest, consumed_bytes = sha256(), 0
    source_stat = source_path.stat()
    stop_reason = "end_of_file"
    with source_path.open("rb") as source:
        for line_number, raw_line in enumerate(source, 1):
            consumed_digest.update(raw_line)
            consumed_bytes += len(raw_line)
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(f"Invalid JSON at source line {line_number}") from error
            if not isinstance(record, dict) or not isinstance(record.get("posts"), list):
                raise ValueError(f"Expected a posts list at source line {line_number}")
            raw_posts = record["posts"]
            if any(not isinstance(post, dict) for post in raw_posts):
                raise ValueError(f"Invalid post at source line {line_number}")
            openers = [post for post in raw_posts if post.get("resto") == 0]
            if len(openers) != 1 or type(openers[0].get("no")) is not int:
                raise ValueError(f"Expected one thread opener with an integer ID at line {line_number}")
            thread_id = openers[0]["no"]
            if thread_id <= 0 or thread_id in seen_thread_ids:
                raise ValueError(f"Invalid or repeated thread ID: {thread_id}")
            seen_thread_ids.add(thread_id)
            statistics["source_records_read"] += 1
            if thread_id in reference_thread_ids:
                statistics["reference_threads_excluded"] += 1
                statistics["reference_thread_posts_excluded"] += len(raw_posts)
                continue
            for raw_post in raw_posts:
                post_id = raw_post.get("no")
                if type(post_id) is not int or post_id <= 0 or raw_post.get("resto") not in {0, thread_id}:
                    raise ValueError(f"Invalid post identity at source line {line_number}")
                if post_id in seen_post_ids or post_id in reference_post_ids:
                    raise ValueError(f"Repeated post ID or inconsistent reference thread: {post_id}")
                seen_post_ids.add(post_id)
                statistics["posts_examined"] += 1
                raw_html = raw_post.get("com")
                if raw_html is None:
                    raw_html = ""
                if not isinstance(raw_html, str):
                    raise ValueError(f"Non-text com field in source post {post_id}")
                post_text = extract_readable_post_text(raw_html)
                if not post_text:
                    statistics["empty_text_posts"] += 1
                    continue
                text_hash = normalized_text_hash(post_text)
                if text_hash in reference_text_hashes:
                    statistics["reference_text_posts_excluded"] += 1
                    copied_text_threads.add(thread_id)
                    continue
                matched_seeds = [word for word, pattern in seed_patterns.items() if pattern.search(post_text)]
                if not matched_seeds:
                    continue
                statistics["seed_matched_posts"] += 1
                selected_seed_counts.update(matched_seeds)
                selected_posts.append({
                    "post_id": post_id, "thread_id": thread_id, "board": "pol", "source": "4chan",
                    "post_key": f"4chan:pol:{post_id}", "thread_key": f"4chan:pol:{thread_id}",
                    "source_line": line_number, "timestamp": raw_post.get("time"),
                    "text": post_text, "normalized_text_sha256": text_hash,
                    "matched_seeds": matched_seeds, "raw_post": raw_post,
                })
                if len(selected_posts) >= maximum_posts:
                    stop_reason = "maximum_posts"
                    break
            if len(selected_posts) >= maximum_posts:
                break
    if (source_path.stat().st_size, source_path.stat().st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
        raise ValueError("Source file changed during retrieval.")
    if not selected_posts:
        raise ValueError(f"No discovery posts survived retrieval and separation. Counts: {dict(statistics)}")
    posts = pd.DataFrame(selected_posts).set_index("post_id", drop=False)
    statistics.update(selected_posts=len(posts), selected_threads=posts["thread_id"].nunique())
    statistics["threads_with_reference_text_exclusions"] = len(copied_text_threads)
    retrieval_info = {
        "counts": dict(statistics), "stop_reason": stop_reason,
        "source_path": str(source_path.resolve()), "source_bytes": source_stat.st_size,
        "consumed_bytes": consumed_bytes, "consumed_prefix_sha256": consumed_digest.hexdigest(),
        "selection": "First matching posts in source order, after reference exclusions; not random sampling",
        "separation": "Exclude every thread represented in the fixed bank; then exclude normalized reference text",
        "reference_thread_count": len(reference_thread_ids), "split_random_seed": None,
        "source_identity": "4chan board + original post/thread ID; this reader is restricted to pol",
        "parser_sha256": parser_hash(),
    }
    seed_summary = pd.DataFrame([
        {"seed": word, "selected_posts": selected_seed_counts[word]} for word in seed_patterns
    ])
    return posts, seed_summary, retrieval_info
