#!/usr/bin/env python3
"""Rescore every active ad from available cache without fetching JobTech."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from db import (
    DB_PATH,
    finish_refresh_run,
    integrate_staged_refresh,
    migrate,
    rebuild_duplicates_and_lanes,
    staged_database,
    start_refresh_run,
)
from preferences import score_active_ads

ROOT = Path(__file__).parent


def run_rescore(
    *, db_path: str | Path, model_path: str | Path, active_on: str | None = None
) -> tuple[int, dict[str, object]]:
    migrate(db_path)
    run_id = start_refresh_run(db_path, run_type="rescore")
    try:
        with staged_database(db_path) as staged_path:
            scoring = score_active_ads(
                staged_path, model_path, active_on=active_on,
                endpoint_client=None, force=True
            )
            snapshot = rebuild_duplicates_and_lanes(
                staged_path, scoring_run_id=run_id, active_on=active_on
            )
            status = "partial" if scoring.fallback_count else "success"
            counts = {
                "preference_scored": scoring.scored_count,
                "fallback_count": scoring.fallback_count,
                "four_signal_count": scoring.four_signal_count,
                "active_count": int(snapshot["active_count"]),
                "canonical_count": int(snapshot["canonical_count"]),
            }
            integrate_staged_refresh(
                db_path, staged_path, run_id=run_id, status=status, counts=counts
            )
        return 0, {"run_id": run_id, "status": status, **counts}
    except Exception as exc:
        finish_refresh_run(run_id, db_path, status="failure", error=str(exc))
        return 3, {"run_id": run_id, "status": "failure", "error": str(exc)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "preference.joblib")
    parser.add_argument("--active-on", help="ISO date override for deterministic operations")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code, result = run_rescore(
        db_path=args.db, model_path=args.model, active_on=args.active_on
    )
    print(json.dumps(result, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
