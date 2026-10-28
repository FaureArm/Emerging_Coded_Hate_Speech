import numpy as np


def save_discovery_run(output_dir, run_info, posts, seed_summary, reference_coverage,
                       stage_counts, tfidf_candidates, occurrences, embeddings,
                       metadata, post_scores, rankings):
    """Save one complete run. run_info.json is written last as the completion marker."""
    import json
    from pathlib import Path
    from datetime import datetime, timezone
    from time import perf_counter
    from .reference_review import file_sha256

    export_started = perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    exported_posts = posts.reset_index(drop=True).copy()
    for column in ("matched_seeds", "raw_post"):
        exported_posts[column] = exported_posts[column].map(lambda value: json.dumps(value, ensure_ascii=False))
    exported_occurrences = occurrences.copy()
    exported_occurrences["word_spans"] = exported_occurrences["word_spans"].map(
        lambda spans: json.dumps([[int(start), int(end)] for start, end in spans])
    )
    tables = {
        "discovery_posts.csv": exported_posts,
        "retrieval_summary.csv": seed_summary,
        "reference_coverage.csv": reference_coverage,
        "tfidf_candidates.csv": tfidf_candidates,
        "candidate_occurrences.csv": exported_occurrences,
        "candidate_metadata.csv": metadata,
        "candidate_post_scores.csv": post_scores,
        "ranked_candidates.csv": rankings,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)
    np.save(output_dir / "candidate_embeddings.npy", embeddings, allow_pickle=False)
    (output_dir / "candidate_stage_counts.json").write_text(json.dumps(stage_counts, indent=2) + "\n", encoding="utf-8")
    run_info["artifact_checksums"] = {
        path.name: file_sha256(path) for path in sorted(output_dir.iterdir())
    }
    run_info["stage_seconds"]["export"] = perf_counter() - export_started
    run_info["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
