from __future__ import annotations

import re
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from db import connect, migrate, rebuild_duplicates_and_lanes, upsert_ads
from preferences import (
    ARTIFACT_SCHEMA_VERSION,
    CACHE_HASH_NORMALIZATION,
    EMBEDDING_INPUT_NORMALIZATION,
    FULL_TEXT_CONSTRUCTOR,
    SOURCE_SCORE_POLICY,
    TITLE_TEXT_CONSTRUCTOR,
    VECTOR_NORMALIZATION,
    score_active_ads,
)
from web import create_app


ACTIVE_ON = "2026-08-23"


def write_model(path: Path) -> None:
    titles = ["security manager", "security analyst", "nurse", "salesperson"]
    labels = np.asarray([1, 1, 0, 0])
    vectorizer = TfidfVectorizer().fit(titles)
    title_classifier = LogisticRegression(random_state=7, solver="liblinear").fit(
        vectorizer.transform(titles), labels
    )
    vectors = np.asarray(
        [[1, 1, 0, 0], [1, 0.8, 0, 0], [0, 0, 1, 1], [0, 0, 0.8, 1]],
        dtype=float,
    )
    embedding_classifier = LogisticRegression(random_state=7, solver="liblinear").fit(
        vectors, labels
    )
    artifact = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_version": "fixture-v1",
        "seed": 7,
        "title_vectorizer": vectorizer,
        "title_classifier": title_classifier,
        "full_embedding_classifier": embedding_classifier,
        "title_embedding_classifier": embedding_classifier,
        "preprocessing": {
            "title_text_constructor": TITLE_TEXT_CONSTRUCTOR,
            "title_embedding_constructor": TITLE_TEXT_CONSTRUCTOR,
            "full_embedding_constructor": FULL_TEXT_CONSTRUCTOR,
            "cache_hash_normalization": CACHE_HASH_NORMALIZATION,
            "embedding_input_normalization": EMBEDDING_INPUT_NORMALIZATION,
            "vector_normalization": VECTOR_NORMALIZATION,
            "source_score_policy": SOURCE_SCORE_POLICY,
        },
        "dimensions": {
            "full_source": 4,
            "full_used": 4,
            "title_source": 4,
            "title_used": 4,
        },
        "embedding_model_name": "fixture-q8",
        "embedding_model_source": "fixture-q8.gguf",
        "embedding_model_sha256": "a" * 64,
        "embedding_model_digest_status": "declared-manifest",
        "cache_model_name": "fixture-q8@sha256:" + "a" * 64 + ":declared-manifest",
        "source_hashes": {"fixture": "b" * 64},
        "training_counts": {"clear": 4, "positive": 2, "negative": 2},
        "regularization": {"values": {"title_tfidf": 1.0}},
    }
    joblib.dump(artifact, path)


def ad(ad_id: str, headline: str, employer: str, score: float) -> dict[str, object]:
    return {
        "id": ad_id,
        "headline": headline,
        "employer": employer,
        "employment_type": "Full time",
        "publication_date": "2026-08-20",
        "application_deadline": "2026-09-01",
        "webpage_url": f"https://example.test/{ad_id}",
        "description_text": f"Description for {headline}",
        "municipality": "Stockholm",
        "region": "Stockholm",
        "occupation_group": "IT security",
        "kw_raw": 4,
        "kw_score": 4,
        "similarity": None,
        "final_score": score,
        "query_source": "fixture",
    }


class WorkspaceVerticalTest(unittest.TestCase):
    def test_migrate_fallback_lanes_and_append_only_web_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            db_path = root / "workspace.db"
            model_path = root / "model.joblib"
            write_model(model_path)

            self.assertEqual(migrate(db_path), [1, 2, 3])
            self.assertEqual(migrate(db_path), [])
            upsert_ads(
                [
                    ad("101", "Security Manager", "Example AB", 0.9),
                    ad("102", " security  manager ", "EXAMPLE ab", 0.2),
                    ad("103", "Nurse", "Hospital", 0.1),
                ],
                db_path,
                seen_on=ACTIVE_ON,
            )
            summary = score_active_ads(db_path, model_path, active_on=ACTIVE_ON)
            self.assertEqual(summary.active_count, 3)
            self.assertEqual(summary.fallback_count, 3)
            self.assertEqual(summary.four_signal_count, 0)
            snapshot = rebuild_duplicates_and_lanes(
                db_path,
                scoring_run_id=None,
                active_on=ACTIVE_ON,
                snapshot_at="2026-08-23T07:00:00+00:00",
            )
            self.assertEqual(snapshot["active_count"], 3)
            self.assertEqual(snapshot["canonical_count"], 2)

            connection = connect(db_path, writable=False)
            try:
                scores = connection.execute(
                    "SELECT scoring_mode,full_embedding,title_embedding FROM preference_scores"
                ).fetchall()
                self.assertTrue(all(row["scoring_mode"] == "fallback" for row in scores))
                self.assertTrue(all(row["full_embedding"] is None for row in scores))
                lane_ids = {
                    row["ad_id"] for row in connection.execute("SELECT ad_id FROM lane_snapshots")
                }
                self.assertEqual(lane_ids, {"101", "103"})
            finally:
                connection.close()

            app = create_app(
                db_path, secret_key="vertical-test-secret", today=date(2026, 8, 23)
            )
            app.testing = True
            client = app.test_client()
            dashboard = client.get("/")
            self.assertEqual(dashboard.status_code, 200)
            self.assertIn(b"Active ads", dashboard.data)
            inbox = client.get("/inbox/high")
            self.assertEqual(inbox.status_code, 200)
            self.assertIn(b"Security Manager", inbox.data)
            detail = client.get("/ads/101")
            token_match = re.search(rb'name="csrf_token" value="([^"]+)"', detail.data)
            self.assertIsNotNone(token_match)
            response = client.post(
                "/ads/101/triage",
                data={
                    "csrf_token": token_match.group(1).decode(),
                    "state": "saved",
                    "reason": "promising",
                    "note": "<script>not executable</script>",
                    "next": "/state/saved",
                },
            )
            self.assertEqual(response.status_code, 303)
            self.assertEqual(response.headers["Location"], "/state/saved")
            saved = client.get("/state/saved")
            self.assertIn(b"Security Manager", saved.data)
            detail = client.get("/ads/101")
            self.assertIn(b"&lt;script&gt;not executable&lt;/script&gt;", detail.data)
            self.assertNotIn(b"<script>not executable</script>", detail.data)

            connection = sqlite3.connect(db_path)
            try:
                events = connection.execute(
                    "SELECT state,source FROM triage_events WHERE ad_id='101'"
                ).fetchall()
                self.assertEqual(events, [("saved", "web")])
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
