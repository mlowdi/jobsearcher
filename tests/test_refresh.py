from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from db import connect, rebuild_duplicates_and_lanes, start_refresh_run, upsert_ads
from main import advisory_lock, run_refresh
from preferences import score_active_ads
from tests.test_workspace_vertical import ad, write_model

ACTIVE_ON = "2026-08-23"


def arguments(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        db=root / "workspace.db",
        model=root / "model.joblib",
        lock_file=root / "refresh.lock",
        active_on=ACTIVE_ON,
        source_embeddings=False,
        enable_embedding_endpoint=False,
        embedding_url="http://127.0.0.1:9/v1/embeddings",
        embedding_model="fixture-q4",
        embedding_timeout=.2,
        embedding_batch_size=8,
    )


def prepare(root: Path) -> SimpleNamespace:
    root.mkdir(parents=True, exist_ok=True)
    args = arguments(root)
    write_model(args.model)
    upsert_ads([ad("101", "Security Manager", "A", .8)], args.db, seen_on=ACTIVE_ON)
    score_active_ads(args.db, args.model, active_on=ACTIVE_ON, force=True)
    rebuild_duplicates_and_lanes(args.db, scoring_run_id=None,
                                 active_on=ACTIVE_ON, snapshot_at="before")
    return args


def lane_state(path: Path) -> list[tuple[object, ...]]:
    connection = connect(path, writable=False)
    try:
        return [tuple(row) for row in connection.execute(
            "SELECT ad_id,lane,lane_rank,snapshot_at,scoring_run_id FROM lane_snapshots ORDER BY ad_id")]
    finally:
        connection.close()


def visible_workspace_state(path: Path) -> dict[str, list[tuple[object, ...]]]:
    connection = connect(path, writable=False)
    try:
        return {
            table: [tuple(row) for row in connection.execute(
                f"SELECT * FROM {table} ORDER BY 1"
            )]
            for table in (
                "ads", "preference_scores", "embedding_cache",
                "duplicate_groups", "lane_snapshots",
            )
        }
    finally:
        connection.close()


class RefreshTest(unittest.TestCase):
    def test_no_hit_refresh_still_rebuilds_complete_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = prepare(Path(temporary))
            code, result = run_refresh(args, fetch=lambda: [])
            self.assertEqual(code, 0)
            self.assertIn(result["status"], {"success", "partial"})
            after = lane_state(args.db)
            self.assertEqual(len(after), 1)
            self.assertNotEqual(after[0][3], "before")
            self.assertIsNotNone(after[0][4])
            connection = connect(args.db, writable=False)
            try:
                run = connection.execute("SELECT * FROM refresh_runs ORDER BY id DESC LIMIT 1").fetchone()
                self.assertEqual(run["total_fetched"], 0)
                self.assertIn(run["status"], {"success", "partial"})
            finally:
                connection.close()

    def test_fetch_source_and_preference_failures_preserve_prior_snapshot(self) -> None:
        cases = (
            ("fetch", lambda args: run_refresh(
                args, fetch=lambda: (_ for _ in ()).throw(RuntimeError("fetch injected")))),
            ("source", lambda args: run_refresh(
                args, fetch=lambda: [dict(id="202")],
                source_scorer=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("source injected")))),
            ("preference", lambda args: run_refresh(
                args,
                fetch=lambda: [dict(id="202")],
                source_scorer=lambda *a, **k: ([
                    ad("101", "Changed headline", "Changed employer", .1),
                    ad("202", "Analyst", "B", .4),
                ], False),
                preference_scorer=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("preference injected")))),
        )
        for name, execute in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                args = prepare(Path(temporary)); before = visible_workspace_state(args.db)
                code, result = execute(args)
                self.assertIn(code, {2, 3})
                self.assertEqual(result["status"], "failure")
                self.assertEqual(visible_workspace_state(args.db), before)
                connection = connect(args.db, writable=False)
                try:
                    run = connection.execute("SELECT status,error FROM refresh_runs ORDER BY id DESC LIMIT 1").fetchone()
                    self.assertEqual(run["status"], "failure")
                    self.assertIn("injected", run["error"])
                finally:
                    connection.close()

    def test_new_run_closes_prior_unfinished_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            db_path = Path(temporary) / "workspace.db"
            first = start_refresh_run(db_path)
            second = start_refresh_run(db_path)
            connection = connect(db_path, writable=False)
            try:
                stale = connection.execute(
                    "SELECT status,completed_at,error FROM refresh_runs WHERE id=?",
                    (first,),
                ).fetchone()
                current = connection.execute(
                    "SELECT status,completed_at FROM refresh_runs WHERE id=?",
                    (second,),
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(stale["status"], "failure")
            self.assertIsNotNone(stale["completed_at"])
            self.assertIn("terminated", stale["error"])
            self.assertEqual(current["status"], "running")
            self.assertIsNone(current["completed_at"])

    def test_advisory_lock_rejects_overlap_without_starting_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = arguments(root)
            with advisory_lock(args.lock_file):
                code, result = run_refresh(args, fetch=lambda: [])
            self.assertEqual(code, 75)
            self.assertEqual(result["status"], "locked")
            self.assertFalse(args.db.exists())


if __name__ == "__main__":
    unittest.main()
