"""Stream Raiders posts while keeping the saved reference threads out of discovery."""

from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from itertools import islice
import random
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


def validate_discovery_sampling(sampling_method, sampling_random_seed):
    if sampling_method not in ("first_matching", "uniform_time"):
        raise ValueError("Discovery sampling method must be 'first_matching' or 'uniform_time'.")
    if type(sampling_random_seed) is not int:
        raise ValueError("Discovery sampling random seed must be an integer.")


def _post_utc_day(post):
    timestamp = post["timestamp"]
    if type(timestamp) is not int or timestamp <= 0:
        raise ValueError(f"uniform_time requires a positive integer Unix timestamp in post {post['post_id']}.")
    try:
        return datetime.fromtimestamp(timestamp, timezone.utc).date().isoformat()
    except (OverflowError, OSError, ValueError) as error:
        raise ValueError(f"Invalid Unix timestamp in post {post['post_id']}: {timestamp}") from error


def _allocate_daily_quotas(eligible_counts, maximum_posts, random_generator):
    """Balance days, redistributing capacity that sparse days cannot fill."""
    remaining_posts = min(maximum_posts, sum(eligible_counts.values()))
    remaining_days = sorted(eligible_counts)
    daily_quotas = {}
    while remaining_days:
        quota_per_day, extra_posts = divmod(remaining_posts, len(remaining_days))
        sparse_days = {day for day in remaining_days if eligible_counts[day] <= quota_per_day}
        if sparse_days:
            for day in sparse_days:
                daily_quotas[day] = eligible_counts[day]
                remaining_posts -= eligible_counts[day]
            remaining_days = [day for day in remaining_days if day not in sparse_days]
        else:
            daily_quotas.update({day: quota_per_day for day in remaining_days})
            for day in random_generator.sample(remaining_days, extra_posts):
                daily_quotas[day] += 1
            break
    return daily_quotas


def _iter_eligible_posts(source_path, seed_patterns, reference_posts, scan_info):
    """Stream validated seed matches; fill scan_info even when the caller stops early."""
    reference_thread_ids = set(reference_posts["thread_id"])
    reference_post_ids = set(reference_posts["post_id"])
    reference_text_hashes = set(reference_posts["normalized_text_sha256"])
    statistics = Counter({name: 0 for name in (
        "source_records_read", "posts_examined", "reference_threads_excluded",
        "reference_thread_posts_excluded", "reference_text_posts_excluded",
        "empty_text_posts", "seed_matched_posts", "selected_posts", "selected_threads",
    )})
    seen_thread_ids, seen_post_ids, copied_text_threads = set(), set(), set()
    consumed_digest, consumed_bytes = sha256(), 0
    source_stat = source_path.stat()
    try:
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
                    yield {
                        "post_id": post_id, "thread_id": thread_id, "board": "pol", "source": "4chan",
                        "post_key": f"4chan:pol:{post_id}", "thread_key": f"4chan:pol:{thread_id}",
                        "source_line": line_number, "timestamp": raw_post.get("time"),
                        "text": post_text, "normalized_text_sha256": text_hash,
                        "matched_seeds": matched_seeds, "raw_post": raw_post,
                    }
    finally:
        if (source_path.stat().st_size, source_path.stat().st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
            raise ValueError("Source file changed during retrieval.")
        statistics["threads_with_reference_text_exclusions"] = len(copied_text_threads)
        scan_info.update(counts=dict(statistics), consumed_bytes=consumed_bytes,
                         consumed_prefix_sha256=consumed_digest.hexdigest())


def retrieve_discovery_posts(source_path, seed_words, reference_posts, maximum_posts=1000,
                             sampling_method="first_matching", sampling_random_seed=42):
    """Select seed matches after excluding fixed-bank threads and copied reference text.

    first_matching stops at the cap. uniform_time reads the full slice twice:
    count eligible posts by UTC day, then reservoir-sample each day's quota.
    Only selected post bodies are retained; source-ID sets still cover the scan.
    The bank's frozen holdout is identical for both methods.
    """
    source_path = Path(source_path)
    if source_path.suffix != ".ndjson" or not source_path.is_file():
        raise FileNotFoundError(f"Expected a Raiders NDJSON file: {source_path}")
    if type(maximum_posts) is not int or maximum_posts < 1:
        raise ValueError("maximum_posts must be a positive integer.")
    validate_discovery_sampling(sampling_method, sampling_random_seed)
    if not seed_words or any(not isinstance(word, str) or not word.strip() for word in seed_words):
        raise ValueError("Provide nonempty retrieval seed phrases.")
    validate_reference_posts(reference_posts)
    source_stat = source_path.stat()
    seed_patterns = {word: compile_phrase_pattern(word) for word in dict.fromkeys(seed_words)}
    scan_info = {}
    daily_counts = []
    if sampling_method == "first_matching":
        with closing(_iter_eligible_posts(source_path, seed_patterns, reference_posts, scan_info)) as eligible_posts:
            selected_posts = list(islice(eligible_posts, maximum_posts))
        stop_reason = "maximum_posts" if len(selected_posts) == maximum_posts else "end_of_file"
        selection = "First matching posts in source order, after reference exclusions; not random sampling"
    else:
        print("Time sampling 1/2: counting eligible posts by UTC day across the full discovery slice.", flush=True)
        eligible_counts = Counter()
        with closing(_iter_eligible_posts(source_path, seed_patterns, reference_posts, scan_info)) as eligible_posts:
            for post in eligible_posts:
                eligible_counts[_post_utc_day(post)] += 1
        if not eligible_counts:
            raise ValueError(f"No discovery posts survived retrieval and separation. Counts: {scan_info['counts']}")
        random_generator = random.Random(sampling_random_seed)
        daily_quotas = _allocate_daily_quotas(eligible_counts, maximum_posts, random_generator)
        daily_reservoirs = {day: [] for day in daily_quotas}
        daily_posts_seen = Counter()
        second_scan_info = {}
        print(f"Time sampling 2/2: selecting up to {maximum_posts:,} posts across {len(daily_quotas):,} UTC days.", flush=True)
        with closing(_iter_eligible_posts(source_path, seed_patterns, reference_posts, second_scan_info)) as eligible_posts:
            for source_position, post in enumerate(eligible_posts):
                day = _post_utc_day(post)
                if day not in daily_quotas:
                    raise ValueError("Source file changed between time-sampling passes.")
                daily_posts_seen[day] += 1
                daily_quota = daily_quotas[day]
                reservoir = daily_reservoirs[day]
                if len(reservoir) < daily_quota:
                    reservoir.append((source_position, post))
                elif daily_quota:
                    replacement_index = random_generator.randrange(daily_posts_seen[day])
                    if replacement_index < daily_quota:
                        reservoir[replacement_index] = (source_position, post)
        if scan_info != second_scan_info or eligible_counts != daily_posts_seen:
            raise ValueError("Source file changed between time-sampling passes.")
        # Restore source order so random reservoir slots do not affect later stages.
        selected_posts = [post for _, post in sorted(
            (entry for reservoir in daily_reservoirs.values() for entry in reservoir),
            key=lambda entry: entry[0],
        )]
        daily_counts = [
            {"date_utc": day, "eligible_posts": eligible_counts[day], "selected_posts": len(daily_reservoirs[day])}
            for day in sorted(eligible_counts)
        ]
        stop_reason = "end_of_file"
        selection = ("Balanced quotas across UTC calendar days with eligible seed matches; "
                     "random posts within each day; sparse-day capacity redistributed; source order retained")
    if (source_path.stat().st_size, source_path.stat().st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
        raise ValueError("Source file changed during retrieval.")
    if not selected_posts:
        raise ValueError(f"No discovery posts survived retrieval and separation. Counts: {scan_info['counts']}")
    posts = pd.DataFrame(selected_posts).set_index("post_id", drop=False)
    scan_info["counts"].update(selected_posts=len(posts), selected_threads=int(posts["thread_id"].nunique()))
    retrieval_info = {
        **scan_info, "stop_reason": stop_reason,
        "source_path": str(source_path.resolve()), "source_bytes": source_stat.st_size,
        "selection": selection, "sampling_method": sampling_method,
        "sampling_random_seed": sampling_random_seed if sampling_method == "uniform_time" else None,
        "scan_passes": 2 if sampling_method == "uniform_time" else 1,
        "time_bin": "UTC calendar day" if sampling_method == "uniform_time" else None,
        "daily_counts": daily_counts,
        "separation": "Exclude every thread represented in the fixed bank; then exclude normalized reference text",
        "reference_thread_count": int(reference_posts["thread_id"].nunique()), "split_random_seed": None,
        "source_identity": "4chan board + original post/thread ID; this reader is restricted to pol",
        "parser_sha256": parser_hash(),
    }
    selected_seed_counts = Counter(seed for post in selected_posts for seed in post["matched_seeds"])
    seed_summary = pd.DataFrame([
        {"seed": word, "selected_posts": selected_seed_counts[word]} for word in seed_patterns
    ])
    return posts, seed_summary, retrieval_info
