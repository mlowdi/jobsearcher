#!/usr/bin/env python3
"""
Jobsearcher — single entrypoint.

Fetches job ads from jobtechdev, scores them, stores in SQLite,
writes ranked results markdown, and commits to git.
"""

import sys
from datetime import datetime, date
from pathlib import Path

import git

from fetcher import fetch_jobs
from scorer import score_ads
from db import init_db, upsert_ads, record_run

RESULTS_DIR = Path.home() / "job-results"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def write_results(ranked: list[dict], embedding_available: bool, output_file: Path) -> None:
    lines = [f"# Job search results — {date.today()}\n"]

    if embedding_available:
        lines.append("Ranked by combined keyword + embedding similarity score.\n")
    else:
        lines.append("Ranked by keyword score only (embedding unavailable).\n")

    lines += [
        "| Rank | Headline | Company | KW | Sim | URL |",
        "|------|----------|---------|-----|-----|-----|",
    ]

    for i, job in enumerate(ranked, 1):
        headline = job["headline"].replace("|", "\\|")
        company = job["employer"].replace("|", "\\|")
        sim_str = f"{job['similarity']:.3f}" if job["similarity"] is not None else "—"
        lines.append(f"| {i} | {headline} | {company} | {job['kw_score']} | {sim_str} | {job['webpage_url']} |")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    init_db()

    # Step 1: Fetch
    log("Fetching jobs from API...")
    try:
        ads = fetch_jobs()
    except Exception as e:
        log(f"API fetch failed: {e}")
        record_run(0, 0, False, "failed")
        return 2

    log(f"Fetched {len(ads)} ads")
    if not ads:
        log("No jobs found (may be weekend/holiday)")
        record_run(0, 0, False, "success")
        return 0

    # Step 2: Score all ads (store all in DB, take top 8 for output)
    log("Scoring and ranking...")
    try:
        all_scored, embedding_available = score_ads(ads, top_n=20, final_n=len(ads))
    except Exception as e:
        log(f"Scoring failed: {e}")
        record_run(len(ads), 0, False, "failed")
        return 3

    ranked = all_scored[:8]
    log(f"Scored {len(all_scored)} ads, top {len(ranked)} for output (embeddings: {'yes' if embedding_available else 'no'})")

    # Step 3: Store in DB
    new_count = upsert_ads(all_scored)
    log(f"DB: {new_count} new ads, {len(all_scored) - new_count} updated")

    record_run(len(ads), len(all_scored), embedding_available,
               "success" if embedding_available else "partial")

    # Step 4: Write results
    output_file = RESULTS_DIR / f"{date.today().strftime('%Y-%m-%d')}-results.md"
    write_results(ranked, embedding_available, output_file)
    log(f"Results written to {output_file}")

    # Print top matches
    for job in ranked[:5]:
        sim_str = f", sim={job['similarity']:.3f}" if job["similarity"] is not None else ""
        log(f"  kw={job['kw_score']}{sim_str} — {job['headline']}")

    # Step 5: Git commit
    try:
        repo = git.Repo(RESULTS_DIR)
        repo.index.add([str(output_file.relative_to(RESULTS_DIR))])
        if repo.is_dirty() or repo.untracked_files:
            repo.index.commit(f"job results {date.today().strftime('%Y-%m-%d')}")
            log("Committed to git")
        else:
            log("No changes to commit")
    except Exception as e:
        log(f"Git commit failed: {e}")
        return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())
