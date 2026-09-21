from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from db import append_triage_event, connect, migrate
from import_labels import import_labels


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_oracle(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE cohorts(name TEXT PRIMARY KEY,candidate_count INTEGER NOT NULL);
        CREATE TABLE candidates(
          cohort TEXT NOT NULL,ad_id TEXT NOT NULL,queue_order INTEGER NOT NULL,
          PRIMARY KEY(cohort,ad_id)
        );
        CREATE TABLE ratings(
          cohort TEXT NOT NULL,ad_id TEXT NOT NULL,rating INTEGER NOT NULL,rated_at TEXT NOT NULL,
          PRIMARY KEY(cohort,ad_id)
        );
        """
    )
    connection.execute("INSERT INTO cohorts VALUES ('pre-filtered-first-pass',4)")
    for order, (ad_id, rating) in enumerate((("1", 1), ("2", 2), ("3", 3), ("4", 5)), 1):
        connection.execute("INSERT INTO candidates VALUES (?,?,?)", ("pre-filtered-first-pass", ad_id, order))
        connection.execute("INSERT INTO ratings VALUES (?,?,?,?)", ("pre-filtered-first-pass", ad_id, rating, f"2026-08-22T00:00:0{order}+00:00"))
    connection.commit(); connection.close()


def create_adjudication(path: Path, *, original_rating: int = 1) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE cases(campaign TEXT,ad_id TEXT,original_rating INTEGER);
        CREATE TABLE decisions(
          campaign TEXT,ad_id TEXT,adjudicated_rating INTEGER,decision_state TEXT,
          note TEXT,decided_at TEXT
        );
        CREATE TABLE decision_reasons(
          campaign TEXT,ad_id TEXT,reason_order INTEGER,reason_code TEXT
        );
        """
    )
    connection.execute("INSERT INTO cases VALUES ('first-disagreements','1',?)", (original_rating,))
    connection.execute("INSERT INTO decisions VALUES ('first-disagreements','1',4,'revised','Strong coordination role','2026-08-23T12:00:00+00:00')")
    connection.executemany(
        "INSERT INTO decision_reasons VALUES ('first-disagreements','1',?,?)",
        [(0, "coordination-delivery-change"), (1, "meaningful-mission")],
    )
    connection.commit(); connection.close()


def add_target_ads(path: Path) -> None:
    migrate(path)
    connection = connect(path)
    try:
        with connection:
            for ad_id in ("1", "2", "3"):
                connection.execute(
                    "INSERT INTO ads(id,headline,application_deadline) VALUES (?,?,?)",
                    (ad_id, f"Role {ad_id}", "2026-09-30"),
                )
    finally:
        connection.close()


class LabelImportTest(unittest.TestCase):
    def test_precedence_idempotence_counts_and_source_immutability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oracle, adjudication, target = root / "oracle.db", root / "adjudication.db", root / "target.db"
            create_oracle(oracle); create_adjudication(adjudication); add_target_ads(target)
            append_triage_event("2", "saved", source="web", db_path=target,
                                created_at="2026-08-23T13:00:00+00:00")
            before = (digest(oracle), digest(adjudication))
            first = import_labels(target_path=target, oracle_path=oracle,
                                  adjudication_path=adjudication)
            second = import_labels(target_path=target, oracle_path=oracle,
                                   adjudication_path=adjudication)
            self.assertEqual(first, {"imported": 3, "skipped": 0, "missing": 1})
            self.assertEqual(second, {"imported": 0, "skipped": 3, "missing": 1})
            self.assertEqual(before, (digest(oracle), digest(adjudication)))
            connection = connect(target, writable=False)
            try:
                imported = connection.execute(
                    """SELECT ad_id,state,reason_code,note,source FROM triage_events
                       WHERE source LIKE 'oracle-import:%' ORDER BY ad_id"""
                ).fetchall()
                self.assertEqual([(row["ad_id"], row["state"]) for row in imported],
                                 [("1", "saved"), ("2", "dismissed"), ("3", "saved")])
                self.assertEqual(imported[0]["reason_code"],
                                 "coordination-delivery-change,meaningful-mission")
                self.assertEqual(imported[0]["note"], "Strong coordination role")
                self.assertEqual(connection.execute(
                    "SELECT COUNT(*) FROM triage_events WHERE ad_id='2'"
                ).fetchone()[0], 2)
            finally:
                connection.close()

    def test_inconsistent_adjudication_rejected_before_target_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oracle, adjudication, target = root / "oracle.db", root / "adjudication.db", root / "target.db"
            create_oracle(oracle); create_adjudication(adjudication, original_rating=2); add_target_ads(target)
            with self.assertRaisesRegex(ValueError, "original rating disagrees"):
                import_labels(target_path=target, oracle_path=oracle,
                              adjudication_path=adjudication)
            connection = connect(target, writable=False)
            try:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM triage_events").fetchone()[0], 0)
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
