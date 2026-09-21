from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from db import connect, normalized_text_hash, put_cached_vector, rebuild_duplicates_and_lanes, upsert_ads
from preferences import full_embedding_text, score_active_ads, title_embedding_text
from rescore import run_rescore
from tests.test_workspace_vertical import ad, write_model

ACTIVE_ON = "2026-08-23"


class RescoreTest(unittest.TestCase):
    def test_rescore_uses_cache_only_and_rebuilds_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); db_path, model_path = root / "workspace.db", root / "model.joblib"
            write_model(model_path)
            rows = [ad("101", "Security Manager", "A", .8), ad("102", "Nurse", "B", .2)]
            upsert_ads(rows, db_path, seen_on=ACTIVE_ON)
            score_active_ads(db_path, model_path, active_on=ACTIVE_ON)
            artifact = joblib.load(model_path)
            connection = connect(db_path)
            try:
                with connection:
                    for text in (title_embedding_text(rows[0]), full_embedding_text(rows[0])):
                        put_cached_vector(connection, text_hash=normalized_text_hash(text),
                            model_name=artifact["cache_model_name"],
                            vector=np.asarray([1, 1, 0, 0], dtype=np.float32), dimensions=4)
            finally: connection.close()
            code, result = run_rescore(db_path=db_path, model_path=model_path, active_on=ACTIVE_ON)
            self.assertEqual(code, 0)
            self.assertEqual(result["status"], "partial")
            self.assertEqual((result["four_signal_count"], result["fallback_count"]), (1, 1))
            connection = connect(db_path, writable=False)
            try:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM lane_snapshots").fetchone()[0], 2)
                self.assertEqual(tuple(connection.execute(
                    "SELECT run_type,status FROM refresh_runs"
                ).fetchone()), ("rescore", "partial"))
            finally: connection.close()

    def test_failed_rescore_preserves_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); db_path, model_path = root / "workspace.db", root / "model.joblib"
            write_model(model_path)
            upsert_ads([ad("101", "Security Manager", "A", .8)], db_path, seen_on=ACTIVE_ON)
            score_active_ads(db_path, model_path, active_on=ACTIVE_ON)
            rebuild_duplicates_and_lanes(db_path, scoring_run_id=None,
                                         active_on=ACTIVE_ON, snapshot_at="before")
            code, result = run_rescore(db_path=db_path, model_path=root / "missing.joblib",
                                       active_on=ACTIVE_ON)
            self.assertEqual(code, 3)
            self.assertEqual(result["status"], "failure")
            connection = connect(db_path, writable=False)
            try:
                self.assertEqual(connection.execute("SELECT DISTINCT snapshot_at FROM lane_snapshots").fetchone()[0], "before")
            finally: connection.close()


if __name__ == "__main__":
    unittest.main()
