"""Bounded thread context and exploratory social evidence for Raiders discovery."""

from collections import defaultdict
from html import unescape
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .discovery import normalized_text_hash
from .reference_review import compile_phrase_pattern, extract_readable_post_text


POST_REFERENCE = re.compile(r">>([0-9]+)\b")
MODERATION_CUE = re.compile(
    r"\b(?:mod(?:s|erator|eration)?|automod|ban(?:ned|ning)?|censor(?:ed|ship)?|"
    r"filter(?:ed|ing)?|algorithm|deleted?)\b", re.IGNORECASE,
)
PLAYFUL_CUE = re.compile(r"\b(?:lol|lmao|rofl|haha+|hehe+)\b", re.IGNORECASE)
PUSHBACK_CUE = re.compile(
    r"\b(?:racist|bigot(?:ed|ry)?|antisemitic|islamophobic|reported|reporting)\b",
    re.IGNORECASE,
)


def load_selected_threads(source_path, selected_posts, reference_posts):
    """Read only source lines for selected threads; exclude copied reference text."""
    selected_lines = dict(zip(selected_posts["source_line"], selected_posts["thread_id"]))
    if len(selected_lines) != selected_posts["thread_id"].nunique():
        raise ValueError("Selected threads have inconsistent source lines.")
    selected_ids = set(selected_posts["post_id"])
    reference_text_hashes = set(reference_posts["normalized_text_sha256"])
    reference_thread_ids = set(reference_posts["thread_id"])
    source_path = Path(source_path)
    source_stat = source_path.stat()
    rows = []
    found_lines = set()
    with source_path.open("rb") as source:
        for line_number, raw_line in enumerate(source, 1):
            if line_number not in selected_lines:
                continue
            record = json.loads(raw_line)
            raw_posts = record.get("posts")
            if not isinstance(raw_posts, list) or not raw_posts:
                raise ValueError(f"Missing posts at selected source line {line_number}.")
            thread_id = selected_lines[line_number]
            if thread_id in reference_thread_ids:
                raise ValueError("A selected discovery thread belongs to the reference bank.")
            openers = [post for post in raw_posts if post.get("resto") == 0]
            if len(openers) != 1 or openers[0].get("no") != thread_id:
                raise ValueError(f"Selected thread changed at source line {line_number}.")
            source_ids = {post.get("no") for post in raw_posts}
            if len(source_ids) != len(raw_posts):
                raise ValueError(f"Repeated post ID in selected thread {thread_id}.")
            source_order = {post.get("no"): index for index, post in enumerate(raw_posts)}
            for post_order, raw_post in enumerate(raw_posts):
                post_id = raw_post.get("no")
                if type(post_id) is not int or raw_post.get("resto") not in {0, thread_id}:
                    raise ValueError(f"Invalid post in selected thread {thread_id}.")
                raw_html = raw_post.get("com") or ""
                if not isinstance(raw_html, str):
                    raise ValueError(f"Non-text comment in post {post_id}.")
                text = extract_readable_post_text(raw_html)
                if not text or normalized_text_hash(text) in reference_text_hashes:
                    continue
                references = list(dict.fromkeys(
                    int(match.group(1)) for match in POST_REFERENCE.finditer(unescape(raw_html))
                    if int(match.group(1)) in source_ids
                    and source_order[int(match.group(1))] < post_order
                ))
                rows.append({
                    "thread_id": thread_id, "post_id": post_id, "post_order": post_order,
                    "timestamp": raw_post.get("time"), "text": text,
                    "reply_to_post_ids": references,
                })
            found_lines.add(line_number)
            if len(found_lines) == len(selected_lines):
                break
    if (source_path.stat().st_size, source_path.stat().st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
        raise ValueError("Discovery source changed during conversation lookup.")
    if found_lines != set(selected_lines):
        raise ValueError("Selected thread source lines are missing.")
    thread_posts = pd.DataFrame(rows)
    if not selected_ids.issubset(set(thread_posts["post_id"])):
        raise ValueError("A selected post vanished or changed during conversation lookup.")
    selected_text = selected_posts.set_index("post_id")["text"]
    for post in thread_posts.itertuples(index=False):
        if post.post_id in selected_ids and post.text != selected_text.loc[post.post_id]:
            raise ValueError(f"Selected post text changed: {post.post_id}.")
    return thread_posts


def attach_conversation_context(selected_posts, thread_posts):
    """Use the opener and an explicitly linked earlier parent; never future replies."""
    by_id = thread_posts.set_index("post_id")
    opener_text = thread_posts.loc[thread_posts["post_order"].eq(0)].set_index("thread_id")["text"]
    contexts = {}
    for post in selected_posts.itertuples(index=False):
        source = by_id.loc[post.post_id]
        earlier = [
            reference_id for reference_id in source["reply_to_post_ids"]
            if reference_id in by_id.index
            and by_id.loc[reference_id, "thread_id"] == post.thread_id
            and by_id.loc[reference_id, "post_order"] < source["post_order"]
        ]
        parent_id = earlier[0] if earlier else None
        parts = []
        if source["post_order"] > 0 and post.thread_id in opener_text:
            parts.append("[THREAD] " + opener_text.loc[post.thread_id])
        if parent_id is not None and parent_id != post.thread_id:
            parts.append("[PARENT] " + by_id.loc[parent_id, "text"])
        contexts[post.post_id] = {
            "conversation_context": "\n".join(parts),
            "conversation_parent_id": parent_id if parent_id is not None else "",
            "conversation_context_source": "explicit_parent" if parent_id is not None else
                                           "thread_opener" if parts else "none",
        }
    enriched = selected_posts.copy()
    for column in ("conversation_context", "conversation_parent_id", "conversation_context_source"):
        enriched[column] = enriched["post_id"].map(lambda post_id: contexts[post_id][column])
    return enriched


def unquoted_text(text):
    """Quoted 4chan lines should not count as a new use or reply reaction."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(">"))


def social_evidence(rankings, thread_posts):
    """Count term reuse and explicit reply cues, without inferring user identities."""
    candidates = rankings["candidate"].tolist()
    patterns = {candidate: compile_phrase_pattern(candidate) for candidate in candidates}
    by_first_word = defaultdict(list)
    for candidate in candidates:
        first_word = re.search(r"\w+", candidate)
        if first_word:
            by_first_word[first_word.group().casefold()].append(candidate)
    matched_posts = defaultdict(list)
    post_candidates = defaultdict(set)
    for post in thread_posts.itertuples(index=False):
        own_text = unquoted_text(post.text)
        words = set(re.findall(r"\w+", own_text.casefold()))
        potential = {candidate for word in words for candidate in by_first_word.get(word, ())}
        for candidate in potential:
            if patterns[candidate].search(own_text):
                matched_posts[candidate].append(post)
                post_candidates[post.post_id].add(candidate)
    reply_counts = defaultdict(lambda: defaultdict(int))
    reply_evidence = []
    for reply in thread_posts.itertuples(index=False):
        targets = list(dict.fromkeys(reply.reply_to_post_ids))
        if not targets:
            continue
        reply_body = unquoted_text(reply.text)
        cues = [name for name, pattern in (
            ("moderation", MODERATION_CUE), ("playful", PLAYFUL_CUE),
            ("pushback", PUSHBACK_CUE),
        ) if pattern.search(reply_body)]
        for candidate in {candidate for target in targets for candidate in post_candidates.get(target, ())}:
            count = reply_counts[candidate]
            count["direct_replies"] += 1
            if cues:
                if "moderation" in cues or "playful" in cues:
                    count["cued_direct_replies"] += 1
                for cue in cues:
                    count[cue + "_replies"] += 1
                reply_evidence.append({
                    "candidate": candidate, "thread_id": reply.thread_id,
                    "reply_post_id": reply.post_id, "target_post_ids": json.dumps(
                        [target for target in targets if candidate in post_candidates.get(target, ())]),
                    "cues": json.dumps(cues), "reply_text": reply.text,
                })
    summary = []
    for candidate in candidates:
        occurrences = matched_posts[candidate]
        by_thread = defaultdict(list)
        for post in occurrences:
            by_thread[post.thread_id].append(post)
        reuse_posts = sum(max(0, len(posts) - 1) for posts in by_thread.values())
        counts = reply_counts[candidate]
        summary.append({
            "candidate": candidate, "observed_thread_posts": len(occurrences),
            "observed_threads": len(by_thread), "reuse_posts": reuse_posts,
            "threads_with_reuse": sum(len(posts) > 1 for posts in by_thread.values()),
            **{name: counts[name] for name in (
                "direct_replies", "cued_direct_replies", "moderation_replies",
                "playful_replies", "pushback_replies",
            )},
        })
    return pd.DataFrame(summary), pd.DataFrame(reply_evidence, columns=[
        "candidate", "thread_id", "reply_post_id", "target_post_ids", "cues", "reply_text",
    ])


def rerank_with_social_evidence(rankings, social_summary, use_uptake, use_reactions,
                                uptake_weight, reaction_weight):
    """Add transparent exploratory scores while retaining HMApp relevance and rank."""
    if any(type(value) is not bool for value in (use_uptake, use_reactions)):
        raise ValueError("Social extension switches must be booleans.")
    if not 0 <= uptake_weight <= 1 or not 0 <= reaction_weight <= 1:
        raise ValueError("Social weights must be between zero and one.")
    enriched = rankings.merge(social_summary, on="candidate", how="left",
                              sort=False, validate="one_to_one")
    if len(enriched) != len(rankings) or enriched["reuse_posts"].isna().any():
        raise ValueError("Social evidence does not cover every ranked candidate.")
    for source, destination, enabled in (
        ("reuse_posts", "uptake_score", use_uptake),
        ("cued_direct_replies", "reaction_score", use_reactions),
    ):
        counts = np.log1p(enriched[source].to_numpy(dtype=float))
        enriched[destination] = counts / counts.max() if enabled and counts.max() else np.zeros(len(counts))
    enriched["experimental_score"] = (
        enriched["relevance"] + uptake_weight * enriched["uptake_score"]
        + reaction_weight * enriched["reaction_score"]
    )
    experimental_order = enriched.sort_values(
        ["experimental_score", "candidate"], ascending=[False, True]).index
    enriched["experimental_rank"] = 0
    enriched.loc[experimental_order, "experimental_rank"] = np.arange(1, len(enriched) + 1)
    return enriched
