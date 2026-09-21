#!/usr/bin/env python3
"""One-shot idempotent import of completed corrected oracle evidence."""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Sequence

from db import DB_PATH, connect, migrate

ROOT = Path(__file__).parent
DEFAULT_COHORT = "pre-filtered-first-pass"
DEFAULT_CAMPAIGN = "first-disagreements"


def readonly(path: str | Path) -> sqlite3.Connection:
    resolved = Path(path).expanduser().resolve()
    connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def load_import_rows(
    oracle_path: str | Path,
    adjudication_path: str | Path,
    *,
    cohort: str,
    campaign: str,
) -> list[dict[str, Any]]:
    with closing(readonly(oracle_path)) as connection:
        cohort_row = connection.execute(
            "SELECT candidate_count FROM cohorts WHERE name=?", (cohort,)
        ).fetchone()
        candidates = [
            dict(row)
            for row in connection.execute(
                """
                SELECT c.ad_id,c.queue_order,r.rating,r.rated_at
                FROM candidates c
                JOIN ratings r ON r.cohort=c.cohort AND r.ad_id=c.ad_id
                WHERE c.cohort=? ORDER BY c.queue_order
                """,
                (cohort,),
            )
        ]
        candidate_ids = {
            str(row["ad_id"])
            for row in connection.execute(
                "SELECT ad_id FROM candidates WHERE cohort=?", (cohort,)
            )
        }
        rating_ids = {
            str(row["ad_id"])
            for row in connection.execute(
                "SELECT ad_id FROM ratings WHERE cohort=?", (cohort,)
            )
        }
    if cohort_row is None or int(cohort_row["candidate_count"]) != len(candidates):
        raise ValueError("oracle cohort is incomplete or has an inconsistent candidate count")
    if candidate_ids != rating_ids or len(candidates) != len(candidate_ids):
        raise ValueError("oracle candidate and rating ad IDs are inconsistent")
    if any(int(row["rating"]) not in range(1, 6) for row in candidates):
        raise ValueError("oracle cohort contains a rating outside 1-5")

    with closing(readonly(adjudication_path)) as connection:
        decisions = [
            dict(row)
            for row in connection.execute(
                """
                SELECT ad_id,adjudicated_rating,note,decided_at
                FROM decisions WHERE campaign=?
                """,
                (campaign,),
            )
        ]
        cases = {
            str(row["ad_id"]): int(row["original_rating"])
            for row in connection.execute(
                "SELECT ad_id,original_rating FROM cases WHERE campaign=?", (campaign,)
            )
        }
        reasons: dict[str, list[str]] = {}
        for row in connection.execute(
            """
            SELECT ad_id,reason_code FROM decision_reasons
            WHERE campaign=? ORDER BY ad_id,reason_order
            """,
            (campaign,),
        ):
            reasons.setdefault(str(row["ad_id"]), []).append(str(row["reason_code"]))
    source = {str(row["ad_id"]): row for row in candidates}
    decision_ids = [str(row["ad_id"]) for row in decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError("adjudication contains duplicate ad IDs")
    unknown_reason_ids = set(reasons) - set(decision_ids)
    if unknown_reason_ids:
        raise ValueError(
            f"adjudication reasons reference undecided ad IDs: {', '.join(sorted(unknown_reason_ids)[:5])}"
        )
    for decision in decisions:
        ad_id = str(decision["ad_id"])
        if ad_id not in source:
            raise ValueError(f"adjudication references unknown oracle ad ID {ad_id}")
        rating = int(decision["adjudicated_rating"])
        if rating not in range(1, 6):
            raise ValueError(f"adjudication rating for {ad_id} is outside 1-5")
        if cases.get(ad_id) != int(source[ad_id]["rating"]):
            raise ValueError(f"adjudication original rating disagrees for ad {ad_id}")
        source[ad_id].update(
            corrected_rating=rating,
            reason_code=",".join(reasons.get(ad_id, [])) or None,
            note=decision.get("note") or None,
            event_at=decision.get("decided_at") or source[ad_id]["rated_at"],
        )
    for row in candidates:
        row.setdefault("corrected_rating", int(row["rating"]))
        row.setdefault("reason_code", None)
        row.setdefault("note", None)
        row.setdefault("event_at", row["rated_at"])
    return candidates


def import_labels(
    *,
    target_path: str | Path,
    oracle_path: str | Path,
    adjudication_path: str | Path,
    cohort: str = DEFAULT_COHORT,
    campaign: str = DEFAULT_CAMPAIGN,
) -> dict[str, int]:
    rows = load_import_rows(
        oracle_path, adjudication_path, cohort=cohort, campaign=campaign
    )
    migrate(target_path)
    source = f"oracle-import:{cohort}:{campaign}"
    connection = connect(target_path)
    imported = skipped = missing = 0
    try:
        connection.execute("BEGIN IMMEDIATE")
        try:
            for row in rows:
                ad_id = str(row["ad_id"])
                if connection.execute("SELECT 1 FROM ads WHERE id=?", (ad_id,)).fetchone() is None:
                    missing += 1
                    continue
                if connection.execute(
                    "SELECT 1 FROM triage_events WHERE ad_id=? AND source LIKE 'oracle-import:%'",
                    (ad_id,),
                ).fetchone() is not None:
                    skipped += 1
                    continue
                rating = int(row["corrected_rating"])
                state = "dismissed" if rating <= 2 else "saved"
                connection.execute(
                    """
                    INSERT INTO triage_events(
                        ad_id,state,reason_code,note,created_at,source
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ad_id,
                        state,
                        row["reason_code"],
                        row["note"],
                        str(row["event_at"]),
                        source,
                    ),
                )
                imported += 1
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    finally:
        connection.close()
    return {"imported": imported, "skipped": skipped, "missing": missing}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--oracle", type=Path, default=ROOT / "oracle-ratings.db")
    parser.add_argument("--adjudication", type=Path, default=ROOT / "adjudication.db")
    parser.add_argument("--cohort", default=DEFAULT_COHORT)
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    counts = import_labels(
        target_path=args.db,
        oracle_path=args.oracle,
        adjudication_path=args.adjudication,
        cohort=args.cohort,
        campaign=args.campaign,
    )
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
