from __future__ import annotations

import csv
import hashlib
import io
import re
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from adjudicate import (
    AdjudicationError,
    DEFAULT_CAMPAIGN,
    DEFAULT_COHORT,
    DEFAULT_PORT,
    DEFAULT_PREDICTIONS_CSV,
    DEFAULT_RATINGS_DB,
    DEFAULT_STATE_DB,
    OPTIONAL_SIGNAL_COLUMNS,
    REASON_CODES,
    SELECTION_VERSION,
    build_parser,
    create_app,
)

SOURCE_SCHEMA = """
CREATE TABLE cohorts (
    name TEXT PRIMARY KEY,
    candidate_count INTEGER NOT NULL
);
CREATE TABLE candidates (
    cohort TEXT NOT NULL,
    ad_id TEXT NOT NULL,
    queue_order INTEGER NOT NULL,
    source_score REAL,
    score_bucket INTEGER NOT NULL,
    headline TEXT NOT NULL,
    employer TEXT NOT NULL,
    employment_type TEXT NOT NULL,
    publication_date TEXT NOT NULL,
    application_deadline TEXT NOT NULL,
    webpage_url TEXT NOT NULL,
    description_text TEXT NOT NULL,
    municipality TEXT NOT NULL,
    region TEXT NOT NULL,
    occupation_group TEXT NOT NULL,
    keyword_score INTEGER,
    PRIMARY KEY (cohort, ad_id)
);
CREATE TABLE ratings (
    cohort TEXT NOT NULL,
    ad_id TEXT NOT NULL,
    rating INTEGER NOT NULL,
    rated_at TEXT NOT NULL,
    PRIMARY KEY (cohort, ad_id)
);
CREATE TABLE presentations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cohort TEXT NOT NULL,
    ad_id TEXT NOT NULL,
    presented_at TEXT NOT NULL,
    action TEXT NOT NULL
);
"""

PREDICTION_COLUMNS = (
    "ad_id",
    "headline",
    "employer",
    "rating",
    "keyword_score",
    "source_score",
    "oof_title_tfidf",
    "oof_embedding_mrl256",
    "oof_title_embedding",
    "oof_four_way_ensemble",
    "model_spread",
    "oof_full_tfidf",
    "oof_embedding_plus_scores",
    "oof_centroid",
    "oof_ensemble",
    "absolute_ensemble_error",
)


def case(ad_id: str, rating: int, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "ad_id": ad_id,
        "rating": rating,
        "source_score": 0.42,
        "keyword_score": 6,
        "score_bucket": 3,
        "headline": f"Role {ad_id}",
        "employer": f"Employer {ad_id}",
        "employment_type": "Permanent",
        "publication_date": "2026-08-01",
        "application_deadline": "2026-09-01",
        "webpage_url": f"https://example.test/jobs/{ad_id}",
        "description_text": f"Description for {ad_id}",
        "municipality": "Stockholm",
        "region": "Stockholm",
        "occupation_group": "Other professionals",
        "oof_title_tfidf": 0.31,
        "oof_embedding_mrl256": 0.37,
        "oof_title_embedding": 0.43,
        "oof_four_way_ensemble": 0.45,
        "model_spread": 0.12,
        "oof_full_tfidf": 0.32,
        "oof_embedding_plus_scores": 0.38,
        "oof_centroid": 0.11,
        "oof_ensemble": 0.41,
        "absolute_ensemble_error": 0.2,
    }
    values.update(overrides)
    return values


def create_source(path: Path, cases: list[dict[str, object]]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SOURCE_SCHEMA)
        connection.execute(
            "INSERT INTO cohorts (name, candidate_count) VALUES (?, ?)",
            (DEFAULT_COHORT, len(cases)),
        )
        for queue_order, item in enumerate(cases, start=1):
            connection.execute(
                """
                INSERT INTO candidates (
                    cohort, ad_id, queue_order, source_score, score_bucket,
                    headline, employer, employment_type, publication_date,
                    application_deadline, webpage_url, description_text,
                    municipality, region, occupation_group, keyword_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    DEFAULT_COHORT,
                    item["ad_id"],
                    queue_order,
                    item["source_score"],
                    item["score_bucket"],
                    item["headline"],
                    item["employer"],
                    item["employment_type"],
                    item["publication_date"],
                    item["application_deadline"],
                    item["webpage_url"],
                    item["description_text"],
                    item["municipality"],
                    item["region"],
                    item["occupation_group"],
                    item["keyword_score"],
                ),
            )
            connection.execute(
                """
                INSERT INTO ratings (cohort, ad_id, rating, rated_at)
                VALUES (?, ?, ?, '2026-08-23T10:00:00+00:00')
                """,
                (DEFAULT_COHORT, item["ad_id"], item["rating"]),
            )
        connection.commit()
    finally:
        connection.close()


def prediction_row(item: dict[str, object]) -> dict[str, object]:
    return {column: item[column] for column in PREDICTION_COLUMNS}


def write_predictions(
    path: Path,
    rows: list[dict[str, object]],
    *,
    columns: tuple[str, ...] = PREDICTION_COLUMNS,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rendered_ad_id(data: bytes) -> str:
    match = re.search(rb'name="ad_id" value="([^"]+)"', data)
    if match is None:
        raise AssertionError("rendered page did not contain an ad_id")
    return match.group(1).decode("utf-8")


class AdjudicationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_path = self.root / "ratings.db"
        self.predictions_path = self.root / "predictions.csv"
        self.state_path = self.root / "state.db"
        self.campaign = "test-campaign"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def prepare(
        self,
        cases: list[dict[str, object]],
        *,
        prediction_rows: list[dict[str, object]] | None = None,
    ) -> None:
        create_source(self.source_path, cases)
        rows = (
            [prediction_row(item) for item in cases if item["rating"] != 3]
            if prediction_rows is None
            else prediction_rows
        )
        write_predictions(self.predictions_path, rows)

    def app(self, **overrides: object):
        arguments: dict[str, object] = {
            "ratings_db": self.source_path,
            "predictions_csv": self.predictions_path,
            "state_db": self.state_path,
            "campaign": self.campaign,
            "cohort": DEFAULT_COHORT,
        }
        arguments.update(overrides)
        app = create_app(**arguments)
        app.config.update(TESTING=True)
        return app

    def query(self, sql: str, parameters: tuple[object, ...] = ()) -> list[sqlite3.Row]:
        connection = sqlite3.connect(self.state_path)
        connection.row_factory = sqlite3.Row
        try:
            return connection.execute(sql, parameters).fetchall()
        finally:
            connection.close()

    def original_rating(self, ad_id: str) -> int:
        return int(
            self.query(
                "SELECT original_rating FROM cases WHERE campaign = ? AND ad_id = ?",
                (self.campaign, ad_id),
            )[0]["original_rating"]
        )

    def decide(
        self,
        client,
        ad_id: str,
        *,
        rating: int | None = None,
        reasons: list[str] | None = None,
        note: str = "",
    ):
        selected_rating = self.original_rating(ad_id) if rating is None else rating
        return client.post(
            "/decide",
            data={
                "ad_id": ad_id,
                "rating": str(selected_rating),
                "reasons": reasons or ["other-unclear"],
                "note": note,
            },
        )


class SelectionTests(AdjudicationTestCase):
    def test_exact_deterministic_20_20_20_construction_and_precedence(self) -> None:
        cases: list[dict[str, object]] = []
        for index in range(40):
            cases.append(
                case(
                    f"n{index:03d}",
                    1 if index % 2 == 0 else 2,
                    oof_four_way_ensemble=0.95 - (index // 2) * 0.01,
                    model_spread=0.99 - (index // 2) * 0.01,
                )
            )
        for index in range(40):
            cases.append(
                case(
                    f"p{index:03d}",
                    4 if index % 2 == 0 else 5,
                    oof_four_way_ensemble=0.05 + (index // 2) * 0.01,
                    model_spread=0.50 - (index // 2) * 0.01,
                )
            )
        cases.extend(case(f"maybe{index}", 3) for index in range(3))
        create_source(self.source_path, cases)
        prediction_rows = [
            prediction_row(item) for item in reversed(cases) if item["rating"] != 3
        ]
        write_predictions(self.predictions_path, prediction_rows)

        app = self.app()
        summary = app.config["ADJUDICATION_SUMMARY"]
        self.assertEqual(summary["candidate_count"], 60)
        self.assertEqual(summary["model_overestimate_count"], 20)
        self.assertEqual(summary["model_underestimate_count"], 20)
        self.assertEqual(summary["signal_disagreement_count"], 20)
        self.assertEqual(summary["selection_version"], SELECTION_VERSION)

        rows = self.query(
            """
            SELECT ad_id, queue_order, lane, oof_four_way_ensemble, model_spread
            FROM cases WHERE campaign = ? ORDER BY queue_order
            """,
            (self.campaign,),
        )
        self.assertEqual([row["queue_order"] for row in rows], list(range(1, 61)))
        self.assertEqual(len({row["ad_id"] for row in rows}), 60)
        self.assertEqual(
            [row["ad_id"] for row in rows[:20]],
            [f"n{index:03d}" for index in range(20)],
        )
        self.assertEqual(
            [row["ad_id"] for row in rows[20:40]],
            [f"p{index:03d}" for index in range(20)],
        )
        self.assertEqual(
            [row["ad_id"] for row in rows[40:]],
            [f"n{index:03d}" for index in range(20, 40)],
        )
        self.assertEqual(
            [row["lane"] for row in rows],
            ["model-overestimate"] * 20
            + ["model-underestimate"] * 20
            + ["signal-disagreement"] * 20,
        )
        self.assertTrue(
            all(
                rows[index]["oof_four_way_ensemble"]
                >= rows[index + 1]["oof_four_way_ensemble"]
                for index in range(19)
            )
        )
        self.assertTrue(
            all(
                rows[index]["oof_four_way_ensemble"]
                <= rows[index + 1]["oof_four_way_ensemble"]
                for index in range(20, 39)
            )
        )
        self.assertTrue(
            all(
                rows[index]["model_spread"] >= rows[index + 1]["model_spread"]
                for index in range(40, 59)
            )
        )

    def test_insufficient_corpus_freezes_each_available_case_once(self) -> None:
        cases = [
            case("negative-a", 1, oof_four_way_ensemble=0.9),
            case("negative-b", 2, oof_four_way_ensemble=0.8),
            case("negative-c", 1, oof_four_way_ensemble=0.7),
            case("positive-a", 4, oof_four_way_ensemble=0.1),
            case("positive-b", 5, oof_four_way_ensemble=0.2),
            case("maybe", 3),
        ]
        self.prepare(cases)
        app = self.app()
        summary = app.config["ADJUDICATION_SUMMARY"]
        self.assertEqual(summary["candidate_count"], 5)
        self.assertEqual(summary["model_overestimate_count"], 3)
        self.assertEqual(summary["model_underestimate_count"], 2)
        self.assertEqual(summary["signal_disagreement_count"], 0)
        ids = [row["ad_id"] for row in self.query("SELECT ad_id FROM cases")]
        self.assertEqual(len(ids), len(set(ids)))
        page = app.test_client().get("/")
        self.assertIn(b"0 of 5 adjudicated", page.data)


class InputValidationTests(AdjudicationTestCase):
    def test_duplicate_missing_unknown_and_mismatched_joins_fail_clearly(self) -> None:
        base_cases = [case("negative", 1), case("positive", 4)]

        variants: list[tuple[str, list[dict[str, object]], str]] = []
        base_rows = [prediction_row(item) for item in base_cases]
        variants.append(("duplicate", base_rows + [dict(base_rows[0])], "duplicate prediction ad_id"))
        variants.append(("missing", base_rows[:1], "missing clear-label source ad_id"))
        variants.append(
            (
                "unknown",
                base_rows + [prediction_row(case("not-in-source", 1))],
                "missing from the source cohort clear-label join",
            )
        )
        mismatched = [dict(row) for row in base_rows]
        mismatched[0]["rating"] = 2
        variants.append(("mismatched", mismatched, "prediction/source mismatch"))

        for name, rows, expected in variants:
            with self.subTest(name=name):
                root = self.root / name
                root.mkdir()
                source = root / "ratings.db"
                predictions = root / "predictions.csv"
                state = root / "state.db"
                create_source(source, base_cases)
                write_predictions(predictions, rows)
                with self.assertRaisesRegex(AdjudicationError, expected):
                    create_app(
                        ratings_db=source,
                        predictions_csv=predictions,
                        state_db=state,
                        campaign=self.campaign,
                        cohort=DEFAULT_COHORT,
                    )
                self.assertFalse(state.exists())

    def test_missing_required_prediction_column_fails_before_state_creation(self) -> None:
        cases = [case("negative", 1)]
        create_source(self.source_path, cases)
        columns = tuple(
            column for column in PREDICTION_COLUMNS if column != "model_spread"
        )
        write_predictions(
            self.predictions_path,
            [prediction_row(cases[0])],
            columns=columns,
        )
        with self.assertRaisesRegex(AdjudicationError, "missing required columns: model_spread"):
            self.app()
        self.assertFalse(self.state_path.exists())


class ProvenanceAndImmutabilityTests(AdjudicationTestCase):
    def test_inputs_remain_byte_identical_across_create_actions_export_and_resume(self) -> None:
        cases = [
            case("negative-a", 1, oof_four_way_ensemble=0.9),
            case("negative-b", 2, oof_four_way_ensemble=0.8),
            case("positive", 4, oof_four_way_ensemble=0.1),
        ]
        self.prepare(cases)
        source_before = file_digest(self.source_path)
        predictions_before = file_digest(self.predictions_path)

        app = self.app()
        client = app.test_client()
        first_id = rendered_ad_id(client.get("/").data)
        self.assertEqual(
            self.decide(
                client,
                first_id,
                reasons=["wrong-role-family", "misleading-vague-ad"],
            ).status_code,
            303,
        )
        second_id = rendered_ad_id(client.get("/").data)
        self.assertEqual(
            client.post("/defer", data={"ad_id": second_id}).status_code, 303
        )
        self.assertEqual(client.get("/export.csv").status_code, 200)
        restarted = self.app()
        self.assertEqual(restarted.test_client().get("/").status_code, 200)

        self.assertEqual(file_digest(self.source_path), source_before)
        self.assertEqual(file_digest(self.predictions_path), predictions_before)

    def test_campaign_path_or_current_hash_mismatch_blocks_without_state_mutation(self) -> None:
        cases = [case("negative", 1), case("positive", 4)]
        self.prepare(cases)
        self.app()
        state_before = file_digest(self.state_path)

        alternate_source = self.root / "same-content-different-path.db"
        shutil.copyfile(self.source_path, alternate_source)
        with self.assertRaisesRegex(AdjudicationError, "immutable frozen snapshot") as error:
            self.app(ratings_db=alternate_source)
        self.assertIn("new --campaign", str(error.exception).lower())
        self.assertEqual(file_digest(self.state_path), state_before)

        prediction_bytes = self.predictions_path.read_bytes()
        self.predictions_path.write_bytes(prediction_bytes + b"\n")
        with self.assertRaisesRegex(AdjudicationError, "predictions SHA-256"):
            self.app()
        self.assertEqual(file_digest(self.state_path), state_before)

        self.predictions_path.write_bytes(prediction_bytes)
        connection = sqlite3.connect(self.source_path)
        try:
            connection.execute(
                "UPDATE candidates SET description_text = description_text || ' changed' WHERE ad_id = 'negative'"
            )
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(AdjudicationError, "ratings source SHA-256"):
            self.app()
        self.assertEqual(file_digest(self.state_path), state_before)

    def test_input_mutation_before_state_commit_rolls_back_campaign(self) -> None:
        cases = [case("negative", 1), case("positive", 4)]
        self.prepare(cases)

        def mutate_predictions() -> None:
            with self.predictions_path.open("ab") as output:
                output.write(b"\n")

        with self.assertRaisesRegex(
            AdjudicationError,
            "frozen predictions changed before the campaign transaction could commit",
        ):
            self.app(_before_provenance_commit=mutate_predictions)

        self.assertTrue(self.state_path.exists())
        self.assertEqual(
            int(self.query("SELECT count(*) AS count FROM campaigns")[0]["count"]),
            0,
        )
        self.assertEqual(
            int(self.query("SELECT count(*) AS count FROM cases")[0]["count"]),
            0,
        )
        self.assertEqual(
            int(self.query("SELECT count(*) AS count FROM decisions")[0]["count"]),
            0,
        )

    def test_replacement_after_verified_snapshot_is_rejected_on_restart(self) -> None:
        cases = [
            case("negative", 1, oof_four_way_ensemble=0.91),
            case("positive", 4, oof_four_way_ensemble=0.09),
        ]
        self.prepare(cases)
        frozen_prediction_hash = file_digest(self.predictions_path)

        def replace_after_verification() -> None:
            with self.predictions_path.open("ab") as output:
                output.write(b"\n")

        app = self.app(
            _after_provenance_verification=replace_after_verification
        )
        self.assertEqual(app.config["ADJUDICATION_SUMMARY"]["candidate_count"], 2)
        campaign = self.query(
            "SELECT predictions_csv_sha256 FROM campaigns WHERE name = ?",
            (self.campaign,),
        )[0]
        self.assertEqual(campaign["predictions_csv_sha256"], frozen_prediction_hash)
        frozen_case = self.query(
            """
            SELECT headline, oof_four_way_ensemble
            FROM cases WHERE campaign = ? AND ad_id = 'negative'
            """,
            (self.campaign,),
        )[0]
        self.assertEqual(frozen_case["headline"], "Role negative")
        self.assertEqual(frozen_case["oof_four_way_ensemble"], 0.91)
        self.assertNotEqual(file_digest(self.predictions_path), frozen_prediction_hash)
        self.assertEqual(app.test_client().get("/").status_code, 200)

        state_before_restart = file_digest(self.state_path)
        with self.assertRaisesRegex(AdjudicationError, "predictions SHA-256"):
            self.app()
        self.assertEqual(file_digest(self.state_path), state_before_restart)


class DecisionValidationTests(AdjudicationTestCase):
    def test_confirm_and_revise_with_multiple_reasons_persist_across_restart(self) -> None:
        cases = [
            case("negative-a", 1, oof_four_way_ensemble=0.9),
            case("negative-b", 2, oof_four_way_ensemble=0.8),
            case("positive-a", 4, oof_four_way_ensemble=0.1),
        ]
        self.prepare(cases)
        app = self.app()
        client = app.test_client()

        first_id = rendered_ad_id(client.get("/").data)
        original = self.original_rating(first_id)
        response = self.decide(
            client,
            first_id,
            rating=original,
            reasons=["misleading-vague-ad", "wrong-role-family"],
            note="Confirmed after seeing the frozen signals.",
        )
        self.assertEqual(response.status_code, 303)

        second_id = rendered_ad_id(client.get("/").data)
        second_original = self.original_rating(second_id)
        revised_rating = 5 if second_original != 5 else 1
        response = self.decide(
            client,
            second_id,
            rating=revised_rating,
            reasons=["meaningful-mission", "changed-mind-context"],
            note="Context changed.",
        )
        self.assertEqual(response.status_code, 303)

        restarted = self.app()
        next_id = rendered_ad_id(restarted.test_client().get("/").data)
        self.assertNotIn(next_id, {first_id, second_id})
        decisions = self.query(
            """
            SELECT ad_id, adjudicated_rating, decision_state, note, decided_at
            FROM decisions WHERE campaign = ? ORDER BY ad_id
            """,
            (self.campaign,),
        )
        by_id = {row["ad_id"]: row for row in decisions}
        self.assertEqual(by_id[first_id]["decision_state"], "confirmed")
        self.assertEqual(by_id[first_id]["adjudicated_rating"], original)
        self.assertEqual(by_id[second_id]["decision_state"], "revised")
        self.assertEqual(by_id[second_id]["adjudicated_rating"], revised_rating)
        self.assertTrue(by_id[first_id]["decided_at"])
        reasons = self.query(
            """
            SELECT ad_id, reason_code FROM decision_reasons
            WHERE campaign = ? ORDER BY ad_id, reason_order
            """,
            (self.campaign,),
        )
        reasons_by_id: dict[str, list[str]] = {}
        for row in reasons:
            reasons_by_id.setdefault(row["ad_id"], []).append(row["reason_code"])
        self.assertEqual(
            reasons_by_id[first_id],
            ["wrong-role-family", "misleading-vague-ad"],
        )
        self.assertEqual(
            reasons_by_id[second_id],
            ["meaningful-mission", "changed-mind-context"],
        )

    def test_invalid_id_rating_reasons_or_unknown_reason_write_nothing(self) -> None:
        cases = [case("negative", 1), case("positive", 4)]
        self.prepare(cases)
        app = self.app()
        client = app.test_client()
        ad_id = rendered_ad_id(client.get("/").data)

        def counts() -> tuple[int, int, int]:
            return tuple(
                int(self.query(f"SELECT count(*) AS count FROM {table}")[0]["count"])
                for table in ("decisions", "decision_reasons", "case_events")
            )

        before = counts()
        invalid_requests = (
            ({"ad_id": "outside", "rating": "1", "reasons": "other-unclear"}, 404),
            ({"ad_id": ad_id, "rating": "0", "reasons": "other-unclear"}, 400),
            ({"ad_id": ad_id, "rating": "6", "reasons": "other-unclear"}, 400),
            ({"ad_id": ad_id, "rating": "1"}, 400),
            ({"ad_id": ad_id, "rating": "1", "reasons": "invented-reason"}, 400),
        )
        for data, status in invalid_requests:
            with self.subTest(data=data):
                response = client.post("/decide", data=data)
                self.assertEqual(response.status_code, status)
                self.assertEqual(counts(), before)


class QueueAndExportTests(AdjudicationTestCase):
    def test_defer_rotation_resume_decided_exclusion_and_completion_counts(self) -> None:
        cases = [
            case(f"negative-{index}", 1, oof_four_way_ensemble=0.9 - index * 0.1)
            for index in range(4)
        ]
        self.prepare(cases)
        app = self.app()
        client = app.test_client()

        deferred_id = rendered_ad_id(client.get("/").data)
        self.assertEqual(
            client.post("/defer", data={"ad_id": deferred_id}).status_code, 303
        )
        second_id = rendered_ad_id(client.get("/").data)
        self.assertNotEqual(second_id, deferred_id)

        restarted_client = self.app().test_client()
        self.assertEqual(rendered_ad_id(restarted_client.get("/").data), second_id)

        decided_ids: set[str] = set()
        for _index in range(3):
            current = rendered_ad_id(restarted_client.get("/").data)
            self.assertNotIn(current, decided_ids)
            self.assertNotEqual(current, deferred_id)
            self.assertEqual(self.decide(restarted_client, current).status_code, 303)
            decided_ids.add(current)

        self.assertEqual(rendered_ad_id(restarted_client.get("/").data), deferred_id)
        revised_rating = 5
        self.assertEqual(
            self.decide(
                restarted_client,
                deferred_id,
                rating=revised_rating,
                reasons=["changed-mind-context"],
            ).status_code,
            303,
        )
        complete = restarted_client.get("/")
        self.assertEqual(complete.status_code, 200)
        self.assertIn(b"Campaign complete", complete.data)
        self.assertIn(b"Confirmed", complete.data)
        self.assertIn(b"Revised", complete.data)
        self.assertNotIn(b'name="ad_id"', complete.data)
        states = self.query(
            "SELECT decision_state, count(*) AS count FROM decisions GROUP BY decision_state"
        )
        self.assertEqual(
            {row["decision_state"]: row["count"] for row in states},
            {"confirmed": 3, "revised": 1},
        )

    def test_export_contains_provenance_signals_and_every_case_state(self) -> None:
        cases = [
            case("negative-a", 1, oof_four_way_ensemble=0.9, model_spread=0.3),
            case("negative-b", 2, oof_four_way_ensemble=0.8, model_spread=0.2),
            case("positive", 4, oof_four_way_ensemble=0.1, model_spread=0.1),
        ]
        self.prepare(cases)
        source_hash = file_digest(self.source_path)
        prediction_hash = file_digest(self.predictions_path)
        client = self.app().test_client()

        deferred_id = rendered_ad_id(client.get("/").data)
        client.post("/defer", data={"ad_id": deferred_id})
        decided_id = rendered_ad_id(client.get("/").data)
        self.decide(
            client,
            decided_id,
            reasons=["wrong-role-family", "misleading-vague-ad"],
            note="Structured note, with comma.",
        )

        response = client.get("/export.csv")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.content_type)
        rows = list(csv.DictReader(io.StringIO(response.get_data(as_text=True))))
        self.assertEqual(len(rows), 3)
        by_id = {row["ad_id"]: row for row in rows}

        for row in rows:
            self.assertEqual(row["campaign"], self.campaign)
            self.assertEqual(row["source_cohort"], DEFAULT_COHORT)
            self.assertEqual(row["selection_version"], SELECTION_VERSION)
            self.assertEqual(row["source_db_path"], str(self.source_path.resolve()))
            self.assertEqual(row["source_db_sha256"], source_hash)
            self.assertEqual(
                row["predictions_csv_path"], str(self.predictions_path.resolve())
            )
            self.assertEqual(row["predictions_csv_sha256"], prediction_hash)
            self.assertTrue(row["selection_explanation"])
            self.assertTrue(row["frozen_at"])
            self.assertIn(row["lane"], {
                "model-overestimate",
                "model-underestimate",
                "signal-disagreement",
            })
            for signal in (
                "keyword_score",
                "source_score",
                "oof_title_tfidf",
                "oof_embedding_mrl256",
                "oof_title_embedding",
                "oof_four_way_ensemble",
                "model_spread",
                *OPTIONAL_SIGNAL_COLUMNS,
            ):
                self.assertNotEqual(row[signal], "")

        deferred = by_id[deferred_id]
        self.assertEqual(deferred["case_status"], "deferred")
        self.assertEqual(deferred["adjudicated_rating"], "")
        self.assertEqual(deferred["defer_count"], "1")
        self.assertGreaterEqual(int(deferred["presentation_count"]), 1)
        self.assertTrue(deferred["first_presented_at"])
        self.assertTrue(deferred["last_deferred_at"])

        decided = by_id[decided_id]
        self.assertEqual(decided["case_status"], "decided")
        self.assertEqual(decided["decision_state"], "confirmed")
        self.assertEqual(decided["rating_confirmed"], "1")
        self.assertEqual(
            decided["reason_codes"],
            "wrong-role-family;misleading-vague-ad",
        )
        self.assertEqual(decided["note"], "Structured note, with comma.")
        self.assertTrue(decided["decided_at"])

        untouched = next(
            row for row in rows if row["ad_id"] not in {deferred_id, decided_id}
        )
        self.assertEqual(untouched["case_status"], "undecided")
        self.assertEqual(untouched["adjudicated_rating"], "")


class RenderingAndCliTests(AdjudicationTestCase):
    def test_html_shows_frozen_signals_and_escapes_hostile_ad_content(self) -> None:
        hostile = case(
            "hostile",
            1,
            headline="<script>alert('headline')</script>",
            employer="<b>Employer & Co</b>",
            description_text='<img src=x onerror="alert(1)"> & text',
            municipality="<svg onload=alert(2)>",
            webpage_url="javascript:alert(3)",
            oof_four_way_ensemble=0.91,
            model_spread=0.77,
        )
        self.prepare([hostile])
        client = self.app().test_client()
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Lane: model-overestimate", html)
        self.assertIn("Original human rating", html)
        self.assertIn("Stored keyword score", html)
        self.assertIn("Stored source score", html)
        self.assertIn("Title TF-IDF component", html)
        self.assertIn("Description embedding component", html)
        self.assertIn("Title embedding component", html)
        self.assertIn("Four-way ensemble preference probability", html)
        self.assertIn("Model spread", html)
        for label, value in (
            ("Full-description TF-IDF component", "0.320000"),
            ("Embedding plus stored-scores component", "0.380000"),
            ("Centroid component", "0.110000"),
            ("Prior ensemble preference probability", "0.410000"),
            ("Absolute prior ensemble error", "0.200000"),
        ):
            self.assertRegex(
                html,
                rf"<dt>{re.escape(label)}</dt>\s*<dd>{value}</dd>",
            )
        self.assertIn("out-of-fold signals", html)
        self.assertIn("not hiring probabilities", html)
        self.assertIn("&lt;script&gt;alert(&#39;headline&#39;)&lt;/script&gt;", html)
        self.assertIn("&lt;img src=x onerror=&#34;alert(1)&#34;&gt;", html)
        self.assertNotIn("<script>alert('headline')</script>", html)
        self.assertNotIn('<img src=x onerror="alert(1)">', html)
        self.assertNotIn('href="javascript:', html)
        self.assertRegex(
            html,
            r'name="rating" value="1" required checked',
        )
        for reason in REASON_CODES:
            self.assertIn(f'value="{reason}"', html)

        decision = client.post(
            "/decide",
            data={
                "ad_id": "hostile",
                "rating": "1",
                "reasons": "other-unclear",
                "note": "<script>alert('note')</script>",
            },
            follow_redirects=True,
        )
        self.assertEqual(decision.status_code, 200)
        self.assertNotIn("<script>alert('note')</script>", decision.get_data(as_text=True))
        template_source = (
            Path(__file__).resolve().parents[1] / "templates" / "adjudicate.html"
        ).read_text(encoding="utf-8")
        self.assertNotIn("|safe", template_source)

    def test_optional_signal_headers_or_cells_may_be_absent(self) -> None:
        optional_labels = (
            "Full-description TF-IDF component",
            "Embedding plus stored-scores component",
            "Centroid component",
            "Prior ensemble preference probability",
            "Absolute prior ensemble error",
        )
        required_only_columns = tuple(
            column
            for column in PREDICTION_COLUMNS
            if column not in OPTIONAL_SIGNAL_COLUMNS
        )
        for scenario in ("missing-headers", "blank-cells"):
            with self.subTest(scenario=scenario):
                root = self.root / scenario
                root.mkdir()
                source = root / "ratings.db"
                predictions = root / "predictions.csv"
                state = root / "state.db"
                item = case(f"optional-{scenario}", 1)
                row = prediction_row(item)
                columns = PREDICTION_COLUMNS
                if scenario == "missing-headers":
                    columns = required_only_columns
                else:
                    for signal in OPTIONAL_SIGNAL_COLUMNS:
                        row[signal] = ""
                create_source(source, [item])
                write_predictions(predictions, [row], columns=columns)
                app = create_app(
                    ratings_db=source,
                    predictions_csv=predictions,
                    state_db=state,
                    campaign=scenario,
                    cohort=DEFAULT_COHORT,
                )
                app.config.update(TESTING=True)
                client = app.test_client()
                html = client.get("/").get_data(as_text=True)
                for label in optional_labels:
                    self.assertRegex(
                        html,
                        rf"<dt>{re.escape(label)}</dt>\s*<dd>Not available</dd>",
                    )
                export_row = next(
                    csv.DictReader(
                        io.StringIO(client.get("/export.csv").get_data(as_text=True))
                    )
                )
                for signal in OPTIONAL_SIGNAL_COLUMNS:
                    self.assertEqual(export_row[signal], "")

    def test_help_contains_exact_defaults_and_no_configurable_host(self) -> None:
        help_text = build_parser().format_help()
        for value in (
            str(DEFAULT_RATINGS_DB),
            str(DEFAULT_PREDICTIONS_CSV),
            str(DEFAULT_STATE_DB),
            DEFAULT_CAMPAIGN,
            DEFAULT_COHORT,
            str(DEFAULT_PORT),
        ):
            self.assertIn(value, re.sub(r"\s+", "", help_text))
        self.assertNotIn("--host", help_text)


if __name__ == "__main__":
    unittest.main()
