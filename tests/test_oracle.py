from __future__ import annotations

import csv
import hashlib
import io
import re
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import date
from pathlib import Path

from oracle import (
    DEFAULT_COHORT,
    DEFAULT_MIN_KEYWORD_SCORE,
    OracleError,
    build_parser,
    create_app,
)

TODAY = date(2026, 8, 19)
SOURCE_SCHEMA = """
CREATE TABLE ads (
    id TEXT PRIMARY KEY,
    headline TEXT NOT NULL,
    employer TEXT,
    employment_type TEXT,
    publication_date TEXT,
    application_deadline TEXT,
    webpage_url TEXT,
    description_text TEXT,
    municipality TEXT,
    region TEXT,
    occupation_group TEXT,
    kw_raw INTEGER,
    kw_score INTEGER,
    similarity REAL,
    final_score REAL,
    first_seen TEXT,
    last_seen TEXT,
    query_source TEXT
)
"""
SOURCE_COLUMNS = (
    "id",
    "headline",
    "employer",
    "employment_type",
    "publication_date",
    "application_deadline",
    "webpage_url",
    "description_text",
    "municipality",
    "region",
    "occupation_group",
    "kw_raw",
    "kw_score",
    "similarity",
    "final_score",
    "first_seen",
    "last_seen",
    "query_source",
)


def ad(ad_id: str, score: float, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "id": ad_id,
        "headline": f"Role {ad_id}",
        "employer": f"Employer {ad_id}",
        "employment_type": "Full time",
        "publication_date": "2026-08-01",
        "application_deadline": "2026-09-30",
        "webpage_url": f"https://example.test/jobs/{ad_id}",
        "description_text": f"Full description for {ad_id}.",
        "municipality": "Stockholm",
        "region": "Stockholm County",
        "occupation_group": "Software developers",
        "kw_raw": 2,
        "kw_score": 4,
        "similarity": 0.5,
        "final_score": score,
        "first_seen": "2026-08-01",
        "last_seen": "2026-08-18",
        "query_source": "fixture",
    }
    values.update(overrides)
    return values


def create_source(path: Path, ads: list[dict[str, object]]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(SOURCE_SCHEMA)
        placeholders = ", ".join("?" for _ in SOURCE_COLUMNS)
        connection.executemany(
            f"INSERT INTO ads ({', '.join(SOURCE_COLUMNS)}) VALUES ({placeholders})",
            [tuple(item[column] for column in SOURCE_COLUMNS) for item in ads],
        )
        connection.commit()
    finally:
        connection.close()


def candidate_id(response_data: bytes) -> str:
    match = re.search(rb'name="ad_id" value="([^"]+)"', response_data)
    if match is None:
        raise AssertionError("candidate id was not rendered")
    return match.group(1).decode("utf-8")


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OracleTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_path = self.root / "source.db"
        self.ratings_path = self.root / "ratings.db"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_app(
        self,
        ads: list[dict[str, object]],
        *,
        ratings_path: Path | None = None,
        cohort: str = DEFAULT_COHORT,
        pool_size: int = 500,
        min_keyword_score: int = DEFAULT_MIN_KEYWORD_SCORE,
    ):
        create_source(self.source_path, ads)
        return create_app(
            ads_db=self.source_path,
            ratings_db=ratings_path or self.ratings_path,
            cohort=cohort,
            pool_size=pool_size,
            min_keyword_score=min_keyword_score,
            today=TODAY,
        )


class CohortCreationTests(OracleTestCase):
    def test_eligibility_deadlines_and_deduplication_are_deterministic(self) -> None:
        ads = [
            ad("dup-low", 10, headline="Data Engineer", employer="Acme"),
            ad("dup-z", 80, headline="data engineer", employer="ACME"),
            ad("dup-a", 80, headline="DATA ENGINEER", employer="acme"),
            ad(
                "today",
                30,
                application_deadline="2026-08-19T23:59:59+02:00",
            ),
            ad("future", 40, application_deadline="2026-08-20"),
            ad("expired", 99, application_deadline="2026-08-18"),
            ad("blank", 99, description_text="   "),
            ad("blank-tab", 99, description_text="\t"),
            ad("blank-newline", 99, description_text="\n"),
            ad("blank-nbsp", 99, description_text="\u00a0"),
            ad("malformed", 99, application_deadline="not-a-date"),
        ]
        self.make_app(ads)

        connection = sqlite3.connect(self.ratings_path)
        try:
            candidate_rows = connection.execute(
                "SELECT ad_id, source_score FROM candidates ORDER BY ad_id"
            ).fetchall()
            cohort_row = connection.execute(
                """
                SELECT eligible_count, deduplicated_count, candidate_count
                FROM cohorts WHERE name = ?
                """,
                (DEFAULT_COHORT,),
            ).fetchone()
        finally:
            connection.close()

        self.assertEqual(
            candidate_rows,
            [("dup-a", 80.0), ("future", 40.0), ("today", 30.0)],
        )
        self.assertEqual(cohort_row, (5, 3, 3))

    def test_invalid_source_keyword_scores_are_ineligible(self) -> None:
        self.make_app(
            [
                ad("missing", 90, kw_score=None),
                ad("text", 89, kw_score="not-an-integer"),
                ad("fraction", 88, kw_score=3.5),
                ad("too-low", 87, kw_score=0),
                ad("below-gate", 86, kw_score=2),
                ad("too-high", 85, kw_score=11),
                ad("at-gate", 1, kw_score=3),
                ad("at-ceiling", 0, kw_score=10),
            ]
        )

        connection = sqlite3.connect(self.ratings_path)
        try:
            candidates = connection.execute(
                """
                SELECT ad_id, keyword_score
                FROM candidates ORDER BY ad_id
                """
            ).fetchall()
        finally:
            connection.close()

        self.assertEqual(candidates, [("at-ceiling", 10), ("at-gate", 3)])

    def test_keyword_gate_runs_before_deduplication_and_score_ranking(self) -> None:
        self.make_app(
            [
                ad(
                    "ineligible-high-rank",
                    999,
                    kw_score=2,
                    headline="Security Engineer",
                    employer="Example AB",
                ),
                ad(
                    "eligible-low-rank",
                    -999,
                    kw_score=3,
                    headline="security engineer",
                    employer="EXAMPLE AB",
                ),
            ]
        )

        connection = sqlite3.connect(self.ratings_path)
        try:
            candidates = connection.execute(
                """
                SELECT ad_id, keyword_score, source_score
                FROM candidates
                """
            ).fetchall()
        finally:
            connection.close()

        self.assertEqual(candidates, [("eligible-low-rank", 3, -999.0)])

    def test_pool_is_finite_reproducible_and_stratified_across_five_quintiles(self) -> None:
        ads = [ad(f"ad-{number:02d}", float(number)) for number in range(50)]
        create_source(self.source_path, ads)
        first_ratings = self.root / "first-ratings.db"
        second_ratings = self.root / "second-ratings.db"

        create_app(
            ads_db=self.source_path,
            ratings_db=first_ratings,
            cohort="sample",
            pool_size=10,
            today=TODAY,
        )
        create_app(
            ads_db=self.source_path,
            ratings_db=second_ratings,
            cohort="sample",
            pool_size=10,
            today=TODAY,
        )

        def frozen_rows(path: Path) -> list[tuple[object, ...]]:
            connection = sqlite3.connect(path)
            try:
                return connection.execute(
                    """
                    SELECT ad_id, queue_order, source_score, score_bucket
                    FROM candidates ORDER BY queue_order
                    """
                ).fetchall()
            finally:
                connection.close()

        first = frozen_rows(first_ratings)
        second = frozen_rows(second_ratings)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual([row[1] for row in first], list(range(1, 11)))
        bucket_counts = {bucket: 0 for bucket in range(1, 6)}
        for row in first:
            bucket_counts[int(row[3])] += 1
        self.assertEqual(bucket_counts, {1: 2, 2: 2, 3: 2, 4: 2, 5: 2})
        self.assertLess(min(float(row[2]) for row in first), 10.0)

    def test_existing_cohort_keeps_frozen_membership_and_order(self) -> None:
        ads = [ad(f"ad-{number:02d}", float(number)) for number in range(20)]
        create_source(self.source_path, ads)
        create_app(
            ads_db=self.source_path,
            ratings_db=self.ratings_path,
            cohort="stable",
            pool_size=10,
            today=TODAY,
        )

        connection = sqlite3.connect(self.ratings_path)
        try:
            before = connection.execute(
                """
                SELECT ad_id, queue_order, source_score, score_bucket
                FROM candidates WHERE cohort = 'stable' ORDER BY queue_order
                """
            ).fetchall()
            metadata_before = connection.execute(
                """
                SELECT as_of_date, requested_pool_size, candidate_count
                FROM cohorts WHERE name = 'stable'
                """
            ).fetchone()
        finally:
            connection.close()

        source = sqlite3.connect(self.source_path)
        try:
            new_ad = ad("new-top", 1000)
            placeholders = ", ".join("?" for _ in SOURCE_COLUMNS)
            source.execute(
                f"INSERT INTO ads ({', '.join(SOURCE_COLUMNS)}) VALUES ({placeholders})",
                tuple(new_ad[column] for column in SOURCE_COLUMNS),
            )
            source.execute("UPDATE ads SET final_score = final_score + 500")
            source.commit()
        finally:
            source.close()

        create_app(
            ads_db=self.source_path,
            ratings_db=self.ratings_path,
            cohort="stable",
            pool_size=5,
            today=date(2026, 8, 20),
        )

        connection = sqlite3.connect(self.ratings_path)
        try:
            after = connection.execute(
                """
                SELECT ad_id, queue_order, source_score, score_bucket
                FROM candidates WHERE cohort = 'stable' ORDER BY queue_order
                """
            ).fetchall()
            metadata_after = connection.execute(
                """
                SELECT as_of_date, requested_pool_size, candidate_count
                FROM cohorts WHERE name = 'stable'
                """
            ).fetchone()
        finally:
            connection.close()

        self.assertEqual(after, before)
        self.assertEqual(metadata_after, metadata_before)

    def test_existing_filtered_cohort_rejects_a_different_threshold(self) -> None:
        create_source(self.source_path, [ad("one", 1, kw_score=3)])
        create_app(
            ads_db=self.source_path,
            ratings_db=self.ratings_path,
            cohort="threshold-three",
            min_keyword_score=3,
            today=TODAY,
        )

        with self.assertRaisesRegex(
            OracleError,
            r"created with --min-keyword-score 3, not 4; choose a new --cohort name",
        ):
            create_app(
                ads_db=self.source_path,
                ratings_db=self.ratings_path,
                cohort="threshold-three",
                min_keyword_score=4,
                today=TODAY,
            )

        connection = sqlite3.connect(self.ratings_path)
        try:
            cohort_row = connection.execute(
                """
                SELECT min_keyword_score, candidate_count
                FROM cohorts WHERE name = 'threshold-three'
                """
            ).fetchone()
            candidates = connection.execute(
                """
                SELECT ad_id, queue_order, keyword_score
                FROM candidates WHERE cohort = 'threshold-three'
                """
            ).fetchall()
        finally:
            connection.close()

        self.assertEqual(cohort_row, (3, 1))
        self.assertEqual(candidates, [("one", 1, 3)])


class LegacySchemaMigrationTests(OracleTestCase):
    def test_old_first_pass_preserves_candidates_ratings_and_presentations(self) -> None:
        create_source(self.source_path, [ad("current-source-row", 1, kw_score=10)])
        source_before = file_digest(self.source_path)

        connection = sqlite3.connect(self.ratings_path)
        try:
            connection.executescript(
                """
                CREATE TABLE cohorts (
                    name TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    as_of_date TEXT NOT NULL,
                    source_db_path TEXT NOT NULL,
                    requested_pool_size INTEGER NOT NULL,
                    eligible_count INTEGER NOT NULL,
                    deduplicated_count INTEGER NOT NULL,
                    candidate_count INTEGER NOT NULL,
                    selection_version TEXT NOT NULL
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
                    PRIMARY KEY (cohort, ad_id),
                    UNIQUE (cohort, queue_order),
                    FOREIGN KEY (cohort) REFERENCES cohorts(name) ON DELETE RESTRICT
                );
                CREATE TABLE ratings (
                    cohort TEXT NOT NULL,
                    ad_id TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    rated_at TEXT NOT NULL,
                    PRIMARY KEY (cohort, ad_id),
                    FOREIGN KEY (cohort, ad_id)
                        REFERENCES candidates(cohort, ad_id) ON DELETE RESTRICT
                );
                CREATE TABLE presentations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cohort TEXT NOT NULL,
                    ad_id TEXT NOT NULL,
                    presented_at TEXT NOT NULL,
                    action TEXT NOT NULL,
                    FOREIGN KEY (cohort, ad_id)
                        REFERENCES candidates(cohort, ad_id) ON DELETE RESTRICT
                );
                CREATE INDEX presentations_candidate_idx
                    ON presentations(cohort, ad_id, action);
                """
            )
            connection.execute(
                """
                INSERT INTO cohorts VALUES (
                    'first-pass', '2026-08-19T08:00:00+00:00', '2026-08-19',
                    '/frozen/source.db', 500, 1078, 1029, 2,
                    'sha256-stratified-quintiles-v1'
                )
                """
            )
            candidate_values = [
                (
                    "first-pass",
                    "legacy-rated",
                    1,
                    91.5,
                    5,
                    "Legacy rated role",
                    "Old employer",
                    "Full time",
                    "2026-08-01",
                    "2026-09-30",
                    "https://example.test/legacy-rated",
                    "Frozen rated description",
                    "Stockholm",
                    "Stockholm County",
                    "Security specialists",
                ),
                (
                    "first-pass",
                    "legacy-skipped",
                    2,
                    12.5,
                    1,
                    "Legacy skipped role",
                    "Old employer",
                    "Full time",
                    "2026-08-02",
                    "2026-10-01",
                    "https://example.test/legacy-skipped",
                    "Frozen skipped description",
                    "Gothenburg",
                    "Vastra Gotaland",
                    "Software developers",
                ),
            ]
            connection.executemany(
                """
                INSERT INTO candidates VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                candidate_values,
            )
            connection.execute(
                """
                INSERT INTO ratings VALUES (
                    'first-pass', 'legacy-rated', 4,
                    '2026-08-19T09:00:00+00:00'
                )
                """
            )
            connection.executemany(
                """
                INSERT INTO presentations (
                    cohort, ad_id, presented_at, action
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        "first-pass",
                        "legacy-rated",
                        "2026-08-19T09:00:00+00:00",
                        "rated",
                    ),
                    (
                        "first-pass",
                        "legacy-skipped",
                        "2026-08-19T09:01:00+00:00",
                        "skip",
                    ),
                ],
            )
            connection.commit()
            candidates_before = connection.execute(
                """
                SELECT ad_id, queue_order, source_score, score_bucket,
                       headline, description_text
                FROM candidates ORDER BY queue_order
                """
            ).fetchall()
            ratings_before = connection.execute(
                "SELECT * FROM ratings ORDER BY ad_id"
            ).fetchall()
            presentations_before = connection.execute(
                "SELECT * FROM presentations ORDER BY id"
            ).fetchall()
        finally:
            connection.close()

        app = create_app(
            ads_db=self.source_path,
            ratings_db=self.ratings_path,
            cohort="first-pass",
            min_keyword_score=9,
            today=TODAY,
        )
        page = app.test_client().get("/")

        connection = sqlite3.connect(self.ratings_path)
        try:
            candidates_after = connection.execute(
                """
                SELECT ad_id, queue_order, source_score, score_bucket,
                       headline, description_text
                FROM candidates ORDER BY queue_order
                """
            ).fetchall()
            ratings_after = connection.execute(
                "SELECT * FROM ratings ORDER BY ad_id"
            ).fetchall()
            presentations_after = connection.execute(
                "SELECT * FROM presentations ORDER BY id"
            ).fetchall()
            legacy_threshold = connection.execute(
                """
                SELECT min_keyword_score FROM cohorts
                WHERE name = 'first-pass'
                """
            ).fetchone()
            legacy_keyword_scores = connection.execute(
                """
                SELECT keyword_score FROM candidates
                WHERE cohort = 'first-pass' ORDER BY queue_order
                """
            ).fetchall()
        finally:
            connection.close()

        self.assertEqual(candidates_after, candidates_before)
        self.assertEqual(ratings_after, ratings_before)
        self.assertEqual(presentations_after, presentations_before)
        self.assertEqual(legacy_threshold, (None,))
        self.assertEqual(legacy_keyword_scores, [(None,), (None,)])
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"1 of 2 rated", page.data)
        self.assertEqual(candidate_id(page.data), "legacy-skipped")
        self.assertEqual(file_digest(self.source_path), source_before)


class OracleWebTests(OracleTestCase):
    def test_rating_post_redirects_persists_resumes_and_source_stays_immutable(self) -> None:
        ads = [ad(f"ad-{number}", float(number)) for number in range(3)]
        create_source(self.source_path, ads)
        source_before = file_digest(self.source_path)
        app = create_app(
            ads_db=self.source_path,
            ratings_db=self.ratings_path,
            today=TODAY,
        )
        client = app.test_client()

        first_page = client.get("/")
        self.assertEqual(first_page.status_code, 200)
        self.assertIn(b"0 of 3 rated", first_page.data)
        first_id = candidate_id(first_page.data)

        rating_response = client.post(
            "/rate", data={"ad_id": first_id, "rating": "4"}
        )
        self.assertEqual(rating_response.status_code, 303)
        self.assertEqual(rating_response.headers["Location"], "/")

        connection = sqlite3.connect(self.ratings_path)
        try:
            stored = connection.execute(
                "SELECT rating, rated_at FROM ratings WHERE ad_id = ?", (first_id,)
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual(stored[0], 4)
        self.assertRegex(stored[1], r"^2026-|^2027-")

        next_page = client.get("/")
        self.assertIn(b"1 of 3 rated", next_page.data)
        second_id = candidate_id(next_page.data)
        self.assertNotEqual(second_id, first_id)

        skip_response = client.post(
            "/rate", data={"ad_id": second_id, "action": "skip"}
        )
        self.assertEqual(skip_response.status_code, 303)
        after_skip = client.get("/")
        self.assertNotEqual(candidate_id(after_skip.data), second_id)

        connection = sqlite3.connect(self.ratings_path)
        try:
            skip_event = connection.execute(
                """
                SELECT action FROM presentations
                WHERE ad_id = ? ORDER BY id DESC LIMIT 1
                """,
                (second_id,),
            ).fetchone()
            skipped_rating = connection.execute(
                "SELECT rating FROM ratings WHERE ad_id = ?", (second_id,)
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual(skip_event, ("skip",))
        self.assertIsNone(skipped_rating)

        resumed_app = create_app(
            ads_db=self.source_path,
            ratings_db=self.ratings_path,
            cohort=DEFAULT_COHORT,
            pool_size=1,
            today=TODAY,
        )
        resumed_page = resumed_app.test_client().get("/")
        self.assertIn(b"1 of 3 rated", resumed_page.data)
        self.assertNotEqual(candidate_id(resumed_page.data), first_id)
        self.assertEqual(file_digest(self.source_path), source_before)

    def test_invalid_ratings_and_ids_cannot_create_state(self) -> None:
        app = self.make_app([ad("one", 1), ad("two", 2)])
        client = app.test_client()
        valid_id = candidate_id(client.get("/").data)

        self.assertEqual(
            client.post("/rate", data={"ad_id": valid_id, "rating": "6"}).status_code,
            400,
        )
        self.assertEqual(
            client.post(
                "/rate", data={"ad_id": valid_id, "rating": "3.0"}
            ).status_code,
            400,
        )
        self.assertEqual(
            client.post(
                "/rate", data={"ad_id": "outside", "rating": "5"}
            ).status_code,
            404,
        )
        self.assertEqual(
            client.post(
                "/rate", data={"ad_id": "outside", "action": "skip"}
            ).status_code,
            404,
        )
        self.assertEqual(client.post("/rate", data={"rating": "3"}).status_code, 400)

        connection = sqlite3.connect(self.ratings_path)
        try:
            self.assertEqual(
                connection.execute("SELECT count(*) FROM ratings").fetchone()[0], 0
            )
            self.assertEqual(
                connection.execute("SELECT count(*) FROM presentations").fetchone()[0],
                0,
            )
        finally:
            connection.close()

    def test_candidate_page_contains_decision_fields_but_no_ranking_fields(self) -> None:
        app = self.make_app(
            [
                ad(
                    "visible",
                    42,
                    kw_score=8,
                    headline="Senior Data Engineer",
                    employer="Useful Company",
                    employment_type="Permanent",
                    application_deadline="2026-09-12",
                    webpage_url="https://example.test/apply",
                    description_text="Build reliable data services with a small team.",
                    municipality="Gothenburg",
                    region="Vastra Gotaland",
                )
            ],
            min_keyword_score=7,
        )
        connection = sqlite3.connect(self.ratings_path)
        try:
            connection.execute(
                """
                UPDATE candidates
                SET source_score = 987654.321, score_bucket = 5,
                    queue_order = 314159
                WHERE cohort = ? AND ad_id = 'visible'
                """,
                (DEFAULT_COHORT,),
            )
            connection.commit()
            frozen_signals = connection.execute(
                """
                SELECT
                    c.source_score, c.score_bucket, c.queue_order,
                    c.keyword_score, h.min_keyword_score
                FROM candidates AS c
                JOIN cohorts AS h ON h.name = c.cohort
                WHERE c.cohort = ? AND c.ad_id = 'visible'
                """,
                (DEFAULT_COHORT,),
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual(frozen_signals, (987654.321, 5, 314159, 8, 7))

        page = app.test_client().get("/")
        rendered = page.get_data(as_text=True).lower()
        rendered_body = rendered.partition("<body>")[2]

        self.assertNotIn("987654.321", rendered_body)
        self.assertNotIn("314159", rendered_body)
        self.assertNotIn("bucket 5", rendered_body)
        self.assertNotRegex(rendered_body, r"(?<!\d)7(?!\d)")
        self.assertNotRegex(rendered_body, r"(?<!\d)8(?!\d)")
        for expected in (
            "senior data engineer",
            "useful company",
            "gothenburg",
            "vastra gotaland",
            "2026-09-12",
            "permanent",
            "build reliable data services with a small team.",
            "https://example.test/apply",
            "no, not for me",
            "yes, i want to look into this",
            "skip for now",
        ):
            self.assertIn(expected, rendered)

        for forbidden in (
            "source_score",
            "source score",
            "final_score",
            "score_bucket",
            "score bucket",
            "queue_order",
            "queue rank",
            "kw_score",
            "keyword score",
            "prefilter",
            "threshold",
            "matcher",
            "ranker",
            "similarity",
            "algorithm",
            "model",
        ):
            self.assertNotIn(forbidden, rendered_body)

    def test_csv_export_contains_frozen_metadata_ratings_and_skips(self) -> None:
        app = self.make_app(
            [
                ad("first", 17.5, kw_score=5),
                ad("second", 42.0, kw_score=9),
            ],
            min_keyword_score=5,
        )
        connection = sqlite3.connect(self.ratings_path)
        try:
            frozen_keyword_scores = connection.execute(
                """
                SELECT ad_id, keyword_score
                FROM candidates ORDER BY ad_id
                """
            ).fetchall()
            frozen_threshold = connection.execute(
                "SELECT min_keyword_score FROM cohorts WHERE name = ?",
                (DEFAULT_COHORT,),
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual(frozen_keyword_scores, [("first", 5), ("second", 9)])
        self.assertEqual(frozen_threshold, (5,))

        client = app.test_client()
        rated_id = candidate_id(client.get("/").data)
        client.post("/rate", data={"ad_id": rated_id, "rating": "5"})
        skipped_id = candidate_id(client.get("/").data)
        client.post("/rate", data={"ad_id": skipped_id, "action": "skip"})

        response = client.get("/export.csv")
        self.assertEqual(response.status_code, 200)
        exported = {
            row["ad_id"]: row
            for row in csv.DictReader(io.StringIO(response.get_data(as_text=True)))
        }
        self.assertEqual(set(exported), {"first", "second"})
        self.assertEqual(exported["first"]["keyword_score"], "5")
        self.assertEqual(exported["second"]["keyword_score"], "9")

        rated = exported[rated_id]
        self.assertEqual(rated["rating"], "5")
        self.assertTrue(rated["rated_at"])
        self.assertEqual(rated["last_presentation_action"], "rated")
        self.assertTrue(rated["last_presented_at"])
        self.assertEqual(rated["presentation_count"], "1")
        self.assertEqual(rated["skip_count"], "0")

        skipped = exported[skipped_id]
        self.assertEqual(skipped["rating"], "")
        self.assertEqual(skipped["rated_at"], "")
        self.assertEqual(skipped["last_presentation_action"], "skip")
        self.assertTrue(skipped["last_presented_at"])
        self.assertEqual(skipped["presentation_count"], "1")
        self.assertEqual(skipped["skip_count"], "1")

        for row in exported.values():
            self.assertEqual(row["cohort"], DEFAULT_COHORT)
            self.assertTrue(row["headline"])
            self.assertTrue(row["source_score"])
            self.assertIn(row["score_bucket"], {"1", "2", "3", "4", "5"})
            self.assertEqual(row["source_db_path"], str(self.source_path.resolve()))
            self.assertEqual(row["requested_pool_size"], "500")
            self.assertEqual(row["eligible_count"], "2")
            self.assertEqual(row["deduplicated_count"], "2")
            self.assertEqual(row["candidate_count"], "2")
            self.assertEqual(row["min_keyword_score"], "5")
            self.assertEqual(
                row["selection_version"],
                "sha256-keyword-prefilter-stratified-quintiles-v2",
            )
            self.assertTrue(row["cohort_created_at"])
            self.assertEqual(row["as_of_date"], TODAY.isoformat())
            self.assertTrue(row["first_presented_at"])


class StartupTests(OracleTestCase):
    def test_help_documents_required_and_optional_cli_paths(self) -> None:
        help_text = build_parser().format_help()
        self.assertIn("--ads-db PATH", help_text)
        self.assertIn("--ratings-db PATH", help_text)
        self.assertIn("--cohort", help_text)
        self.assertIn("--pool-size N", help_text)
        self.assertIn("--min-keyword-score N", help_text)
        self.assertIn("from 1 through 10 (default: 3)", help_text)
        self.assertIn("read-only", help_text)

    def test_cli_defaults_and_rejects_invalid_keyword_thresholds(self) -> None:
        args = build_parser().parse_args(["--ads-db", "source.db"])
        self.assertEqual(args.cohort, DEFAULT_COHORT)
        self.assertEqual(args.min_keyword_score, DEFAULT_MIN_KEYWORD_SCORE)

        for invalid in ("0", "11", "3.5", "not-an-integer"):
            with self.subTest(invalid=invalid):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    build_parser().parse_args(
                        ["--ads-db", "source.db", "--min-keyword-score", invalid]
                    )

    def test_unavailable_or_incompatible_source_fails_clearly(self) -> None:
        with self.assertRaisesRegex(OracleError, "source ads database is unavailable"):
            create_app(
                ads_db=self.root / "missing.db",
                ratings_db=self.ratings_path,
                today=TODAY,
            )

        incomplete = self.root / "incomplete.db"
        connection = sqlite3.connect(incomplete)
        try:
            connection.execute("CREATE TABLE ads (id TEXT PRIMARY KEY, headline TEXT)")
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(OracleError, "missing expected columns"):
            create_app(
                ads_db=incomplete,
                ratings_db=self.ratings_path,
                today=TODAY,
            )

    def test_source_and_ratings_database_must_be_separate(self) -> None:
        create_source(self.source_path, [ad("one", 1)])
        before = file_digest(self.source_path)
        with self.assertRaisesRegex(OracleError, "must be separate"):
            create_app(
                ads_db=self.source_path,
                ratings_db=self.source_path,
                today=TODAY,
            )
        self.assertEqual(file_digest(self.source_path), before)


if __name__ == "__main__":
    unittest.main()
