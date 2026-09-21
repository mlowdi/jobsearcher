from __future__ import annotations

import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from db import (
    append_triage_event,
    connect,
    decode_float32,
    duplicate_key,
    encode_float32,
    migrate,
    rebuild_duplicates_and_lanes,
    upsert_ads,
)
from fetcher import _remember_hit

ADS_SCHEMA = """
CREATE TABLE ads (
 id TEXT PRIMARY KEY, headline TEXT NOT NULL, employer TEXT, employment_type TEXT,
 publication_date TEXT, application_deadline TEXT, webpage_url TEXT,
 description_text TEXT, municipality TEXT, region TEXT, occupation_group TEXT,
 kw_raw INTEGER, kw_score INTEGER, similarity REAL, final_score REAL,
 first_seen TEXT, last_seen TEXT, query_source TEXT
);
CREATE TABLE runs (
 id INTEGER PRIMARY KEY AUTOINCREMENT, run_date TEXT NOT NULL,
 total_fetched INTEGER,total_scored INTEGER,embedding_available BOOLEAN,status TEXT
);
"""


def source_ad(ad_id: str, **overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "id": ad_id, "headline": "Old headline", "employer": "Old employer",
        "employment_type": "Old employment", "publication_date": "2026-08-01",
        "application_deadline": "2026-09-01", "webpage_url": "https://old.test",
        "description_text": "Old description", "municipality": "Old municipality",
        "region": "Old region", "occupation_group": "Old occupation", "kw_raw": 1,
        "kw_score": 1, "similarity": None, "final_score": 0.1,
        "query_source": "tagged:first",
    }
    result.update(overrides)
    return result


def insert_ranked_ads(path: Path, count: int) -> None:
    connection = connect(path)
    try:
        with connection:
            for index in range(count):
                ad_id = f"{index:04d}"
                connection.execute(
                    """INSERT INTO ads(
                       id,headline,employer,publication_date,application_deadline,
                       final_score,first_seen,last_seen) VALUES (?,?,?,?,?,?,?,?)""",
                    (ad_id, f"Role {index}", f"Employer {index}",
                     f"2026-08-{(index % 20) + 1:02d}", "2026-09-30",
                     (count - index) / count, "2026-08-01", "2026-08-23"),
                )
                spread = 1.0 - (index / (count * 2))
                connection.execute(
                    """INSERT INTO preference_scores(
                       ad_id,model_version,scoring_mode,title_tfidf,full_embedding,
                       title_embedding,source_score,ensemble_score,component_spread,
                       title_hash,description_hash,scored_at)
                       VALUES (?, 'v', 'fallback', .5, NULL, NULL, .5, ?, ?, ?, ?, ?)""",
                    (ad_id, (count - index) / count, spread, "a" * 64, "b" * 64,
                     "2026-08-23T00:00:00+00:00"),
                )
    finally:
        connection.close()


class MigrationAndStorageTest(unittest.TestCase):
    def test_migration_preserves_legacy_rows_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.db"
            connection = sqlite3.connect(path)
            connection.executescript(ADS_SCHEMA)
            connection.execute("INSERT INTO ads(id,headline,first_seen,last_seen) VALUES ('1','Legacy','2020-01-01','2020-01-02')")
            connection.execute("INSERT INTO runs(run_date,total_fetched,status) VALUES ('2020-01-02',7,'success')")
            connection.commit(); connection.close()
            self.assertEqual(migrate(path), [1, 2, 3])
            self.assertEqual(migrate(path), [])
            connection = connect(path)
            try:
                self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
                self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
                self.assertEqual(connection.execute("SELECT headline FROM ads WHERE id='1'").fetchone()[0], "Legacy")
                self.assertEqual(connection.execute("SELECT total_fetched FROM runs").fetchone()[0], 7)
                tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertTrue({"schema_migrations", "triage_events", "preference_scores",
                                 "embedding_cache", "duplicate_groups", "lane_snapshots",
                                 "refresh_runs"} <= tables)
            finally:
                connection.close()

    def test_full_mutable_upsert_preserves_first_seen_and_nonblank_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "upsert.db"
            upsert_ads([source_ad("1")], path, seen_on="2026-08-20")
            replacement = source_ad(
                "1", headline="New headline", employer="New employer",
                employment_type="New employment", publication_date="2026-08-22",
                application_deadline="2026-10-01", webpage_url="https://new.test",
                description_text="New description", municipality="New municipality",
                region="New region", occupation_group="New occupation", kw_raw=9,
                kw_score=8, similarity=.75, final_score=.81, query_source="")
            self.assertEqual(upsert_ads([replacement], path, seen_on="2026-08-23"), 0)
            connection = connect(path, writable=False)
            try:
                row = connection.execute("SELECT * FROM ads WHERE id='1'").fetchone()
                self.assertEqual(row["first_seen"], "2026-08-20")
                self.assertEqual(row["last_seen"], "2026-08-23")
                self.assertEqual(row["query_source"], "tagged:first")
                for key in ("headline", "employer", "employment_type", "publication_date",
                            "application_deadline", "webpage_url", "description_text",
                            "municipality", "region", "occupation_group", "kw_raw",
                            "kw_score", "similarity", "final_score"):
                    self.assertEqual(row[key], replacement[key])
            finally:
                connection.close()
            replacement["query_source"] = "tagged:new"
            upsert_ads([replacement], path, seen_on="2026-08-24")
            connection = connect(path, writable=False)
            try:
                self.assertEqual(connection.execute("SELECT query_source FROM ads WHERE id='1'").fetchone()[0], "tagged:new")
            finally:
                connection.close()

    def test_fetch_hit_merge_keeps_first_tag_with_fresh_fields(self) -> None:
        hits: dict[str, dict[str, object]] = {}
        _remember_hit(hits, {"id": "7", "headline": "Old"}, "occupation_group")
        _remember_hit(hits, {"id": "7", "headline": "Fresh"}, "")
        self.assertEqual(hits["7"]["headline"], "Fresh")
        self.assertEqual(hits["7"]["query_source"], "occupation_group")

    def test_float32_encoding_is_exact_and_validated(self) -> None:
        blob = encode_float32([1.25, -2.5], dimensions=2)
        self.assertEqual(blob, struct.pack("<ff", 1.25, -2.5))
        np.testing.assert_array_equal(decode_float32(blob, 2), np.asarray([1.25, -2.5], dtype=np.float32))
        with self.assertRaises(ValueError): encode_float32([[1.0]])
        with self.assertRaises(ValueError): encode_float32([np.nan])
        with self.assertRaises(ValueError): decode_float32(blob, 3)


class DuplicateAndLaneTest(unittest.TestCase):
    def test_exact_sizes_small_corpus_and_independent_duplicate_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lanes.db"
            migrate(path); insert_ranked_ads(path, 270)
            connection = connect(path)
            try:
                with connection:
                    connection.execute("UPDATE ads SET headline='  Säkerhet  Chef ',employer='Å AB' WHERE id='0000'")
                    connection.execute("UPDATE ads SET headline='säkerhet chef',employer='A\u030a ab' WHERE id='0001'")
            finally: connection.close()
            append_triage_event("0001", "dismissed", source="web", db_path=path)
            result = rebuild_duplicates_and_lanes(path, scoring_run_id=None,
                active_on="2026-08-23", snapshot_at="2026-08-23T07:00:00+00:00")
            self.assertEqual(result["active_count"], 270)
            self.assertEqual(result["canonical_count"], 269)
            connection = connect(path, writable=False)
            try:
                counts = {row["lane"]: row["n"] for row in connection.execute("SELECT lane,COUNT(*) n FROM lane_snapshots GROUP BY lane")}
                self.assertEqual(counts, {"high": 50, "possible": 150, "wildcard": 50, "low": 19})
                group = connection.execute("SELECT ad_id,canonical_ad_id,method,similarity FROM duplicate_groups WHERE ad_id IN ('0000','0001') ORDER BY ad_id").fetchall()
                self.assertEqual({row["canonical_ad_id"] for row in group}, {"0000"})
                self.assertTrue(all(row["method"] == "exact" and row["similarity"] == 1.0 for row in group))
                self.assertIsNone(connection.execute("SELECT lane FROM lane_snapshots WHERE ad_id='0001'").fetchone())
                self.assertEqual(connection.execute("SELECT state FROM triage_events WHERE ad_id='0001'").fetchone()[0], "dismissed")
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM triage_events WHERE ad_id='0000'").fetchone()[0], 0)
                self.assertEqual({row[0] for row in connection.execute("SELECT DISTINCT snapshot_at FROM lane_snapshots")}, {"2026-08-23T07:00:00+00:00"})
            finally: connection.close()
            small = Path(temporary) / "small.db"
            migrate(small); insert_ranked_ads(small, 7)
            rebuild_duplicates_and_lanes(small, scoring_run_id=None, active_on="2026-08-23")
            connection = connect(small, writable=False)
            try:
                self.assertEqual(connection.execute("SELECT DISTINCT lane FROM lane_snapshots").fetchone()[0], "high")
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM lane_snapshots").fetchone()[0], 7)
            finally: connection.close()

    def test_snapshot_and_groups_roll_back_together_on_insert_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "atomic.db"
            migrate(path); insert_ranked_ads(path, 3)
            rebuild_duplicates_and_lanes(path, scoring_run_id=None, active_on="2026-08-23", snapshot_at="before")
            connection = connect(path)
            try:
                before_lanes = [tuple(row) for row in connection.execute("SELECT ad_id,lane,lane_rank,snapshot_at FROM lane_snapshots ORDER BY ad_id")]
                before_groups = [tuple(row) for row in connection.execute("SELECT ad_id,canonical_ad_id,updated_at FROM duplicate_groups ORDER BY ad_id")]
                connection.execute("""CREATE TRIGGER reject_snapshot BEFORE INSERT ON lane_snapshots BEGIN SELECT RAISE(ABORT, 'injected failure'); END""")
                connection.commit()
            finally: connection.close()
            with self.assertRaises(sqlite3.IntegrityError):
                rebuild_duplicates_and_lanes(path, scoring_run_id=None, active_on="2026-08-23", snapshot_at="after")
            connection = connect(path, writable=False)
            try:
                self.assertEqual([tuple(row) for row in connection.execute("SELECT ad_id,lane,lane_rank,snapshot_at FROM lane_snapshots ORDER BY ad_id")], before_lanes)
                self.assertEqual([tuple(row) for row in connection.execute("SELECT ad_id,canonical_ad_id,updated_at FROM duplicate_groups ORDER BY ad_id")], before_groups)
            finally: connection.close()

    def test_duplicate_key_is_unicode_casefolded_and_whitespace_collapsed(self) -> None:
        self.assertEqual(duplicate_key("  SÄKERHET\nCHEF ", "A\u030a  AB"), duplicate_key("säkerhet chef", "Å ab"))


if __name__ == "__main__":
    unittest.main()
