from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import requests

from db import connect, migrate, normalized_text_hash, put_cached_vector, upsert_ads
from preferences import (
    OpenAIEmbeddingClient,
    full_embedding_text,
    score_active_ads,
    title_embedding_text,
)
from tests.test_workspace_vertical import ad, write_model

ACTIVE_ON = "2026-08-23"


class FailingSession:
    def post(self, *args, **kwargs):
        raise requests.ConnectionError("offline fixture")


class FakeResponse:
    def __init__(self, count: int):
        self.count = count

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {"data": [
            {"index": index, "embedding": [1.0, 1.0, 0.0, 0.0]}
            for index in range(self.count)
        ]}


class SuccessfulSession:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def post(self, url, *, json, timeout):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse(len(json["input"]))


def setup_workspace(root: Path) -> tuple[Path, Path, dict[str, object]]:
    root.mkdir(parents=True, exist_ok=True)
    db_path, model_path = root / "workspace.db", root / "model.joblib"
    write_model(model_path)
    row = ad("101", "Security Manager", "Example AB", 0.8)
    upsert_ads([row], db_path, seen_on=ACTIVE_ON)
    return db_path, model_path, row


class PreferenceScoringTest(unittest.TestCase):
    def test_fallback_has_only_two_real_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            db_path, model_path, _ = setup_workspace(Path(temporary))
            summary = score_active_ads(db_path, model_path, active_on=ACTIVE_ON)
            self.assertEqual((summary.fallback_count, summary.four_signal_count), (1, 0))
            connection = connect(db_path, writable=False)
            try:
                row = connection.execute("SELECT * FROM preference_scores").fetchone()
                self.assertEqual(row["scoring_mode"], "fallback")
                self.assertIsNone(row["full_embedding"])
                self.assertIsNone(row["title_embedding"])
                self.assertAlmostEqual(row["ensemble_score"],
                                       (row["title_tfidf"] + row["source_score"]) / 2)
                self.assertAlmostEqual(row["component_spread"],
                                       abs(row["title_tfidf"] - row["source_score"]))
                self.assertIsNone(row["embedding_model_name"])
            finally:
                connection.close()

    def test_cached_vectors_upgrade_to_four_signal_and_validate_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            db_path, model_path, source = setup_workspace(Path(temporary))
            score_active_ads(db_path, model_path, active_on=ACTIVE_ON)
            artifact = joblib.load(model_path)
            connection = connect(db_path)
            try:
                with connection:
                    for text in (title_embedding_text(source), full_embedding_text(source)):
                        put_cached_vector(
                            connection,
                            text_hash=normalized_text_hash(text),
                            model_name=artifact["cache_model_name"],
                            vector=np.asarray([1, 1, 0, 0], dtype=np.float32),
                            dimensions=4,
                        )
            finally:
                connection.close()
            summary = score_active_ads(db_path, model_path, active_on=ACTIVE_ON)
            self.assertEqual((summary.fallback_count, summary.four_signal_count), (0, 1))
            connection = connect(db_path, writable=False)
            try:
                row = connection.execute("SELECT * FROM preference_scores").fetchone()
                self.assertEqual(row["scoring_mode"], "four-signal")
                components = [row["title_tfidf"], row["full_embedding"],
                              row["title_embedding"], row["source_score"]]
                self.assertAlmostEqual(row["ensemble_score"], sum(components) / 4)
                self.assertAlmostEqual(row["component_spread"], max(components) - min(components))
                self.assertEqual(row["embedding_model_name"], artifact["cache_model_name"])
            finally:
                connection.close()

            bad = Path(temporary) / "bad.db"
            _, _, bad_source = setup_workspace(Path(temporary) / "second")
            # The helper's path is distinct; use it rather than the placeholder above.
            bad = Path(temporary) / "second" / "workspace.db"
            connection = connect(bad)
            try:
                with connection:
                    connection.execute(
                        """INSERT INTO embedding_cache(text_hash,model_name,dimensions,vector,created_at)
                           VALUES (?,?,?,?,?)""",
                        (normalized_text_hash(title_embedding_text(bad_source)),
                         artifact["cache_model_name"], 3,
                         np.asarray([1, 2, 3], dtype="<f4").tobytes(), "now"),
                    )
            finally:
                connection.close()
            with self.assertRaisesRegex(ValueError, "cached dimensions"):
                score_active_ads(bad, Path(temporary) / "second" / "model.joblib",
                                 active_on=ACTIVE_ON)

    def test_endpoint_failure_falls_back_and_success_records_distinct_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            db_path, model_path, _ = setup_workspace(root / "failure")
            failing = OpenAIEmbeddingClient(
                url="http://127.0.0.1:9/v1/embeddings", model="fixture-q4",
                timeout=.2, batch_size=8, session=FailingSession())
            summary = score_active_ads(db_path, model_path, active_on=ACTIVE_ON,
                                       endpoint_client=failing)
            self.assertEqual(summary.fallback_count, 1)
            self.assertIn("offline fixture", summary.endpoint_error)
            connection = connect(db_path, writable=False)
            try:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0], 0)
            finally:
                connection.close()

            success_root = root / "success"
            db_path, model_path, _ = setup_workspace(success_root)
            session = SuccessfulSession()
            client = OpenAIEmbeddingClient(
                url="http://127.0.0.1:9090/v1/embeddings", model="fixture-q4",
                timeout=1, batch_size=8, session=session)
            summary = score_active_ads(db_path, model_path, active_on=ACTIVE_ON,
                                       endpoint_client=client)
            self.assertEqual((summary.fallback_count, summary.four_signal_count), (0, 1))
            self.assertIsNone(summary.endpoint_error)
            self.assertTrue(session.calls)
            connection = connect(db_path, writable=False)
            try:
                row = connection.execute("SELECT embedding_model_name,embedding_provenance FROM preference_scores").fetchone()
                self.assertEqual(row["embedding_model_name"], "endpoint:fixture-q4")
                self.assertEqual(set(json.loads(row["embedding_provenance"]).values()), {"endpoint:fixture-q4"})
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0], 2)
            finally:
                connection.close()
            cached = score_active_ads(db_path, model_path, active_on=ACTIVE_ON)
            self.assertEqual((cached.fallback_count, cached.four_signal_count), (0, 1))
            connection = connect(db_path)
            try:
                with connection:
                    connection.execute(
                        "DELETE FROM embedding_cache WHERE model_name='endpoint:fixture-q4'"
                    )
            finally:
                connection.close()
            downgraded = score_active_ads(db_path, model_path, active_on=ACTIVE_ON)
            self.assertEqual((downgraded.fallback_count, downgraded.four_signal_count), (1, 0))
            connection = connect(db_path, writable=False)
            try:
                row = connection.execute(
                    "SELECT scoring_mode,full_embedding,title_embedding,embedding_provenance FROM preference_scores"
                ).fetchone()
                self.assertEqual(row["scoring_mode"], "fallback")
                self.assertIsNone(row["full_embedding"])
                self.assertIsNone(row["title_embedding"])
                self.assertIsNone(row["embedding_provenance"])
            finally:
                connection.close()

    def test_oversized_input_does_not_block_later_ads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            db_path, model_path = root / "workspace.db", root / "model.joblib"
            write_model(model_path)
            oversized = ad("101", "Security Manager", "A", 0.8)
            oversized["description_text"] = "security " * 1000
            normal = ad("102", "Security Strategist", "B", 0.7)
            upsert_ads([oversized, normal], db_path, seen_on=ACTIVE_ON)
            session = SuccessfulSession()
            client = OpenAIEmbeddingClient(
                url="http://127.0.0.1:9090/v1/embeddings",
                model="fixture-q4",
                timeout=1,
                batch_size=8,
                token_budget=256,
                session=session,
            )
            summary = score_active_ads(
                db_path, model_path, active_on=ACTIVE_ON, endpoint_client=client
            )
            self.assertEqual((summary.fallback_count, summary.four_signal_count), (1, 1))
            self.assertIn("1 embedding input(s)", summary.endpoint_error)
            connection = connect(db_path, writable=False)
            try:
                modes = {
                    row["ad_id"]: row["scoring_mode"]
                    for row in connection.execute(
                        "SELECT ad_id,scoring_mode FROM preference_scores"
                    )
                }
            finally:
                connection.close()
            self.assertEqual(modes, {"101": "fallback", "102": "four-signal"})

    def test_endpoint_bounds_are_enforced_without_requests(self) -> None:
        with self.assertRaises(ValueError):
            OpenAIEmbeddingClient(url="http://example.test", model="x", timeout=61)
        with self.assertRaises(ValueError):
            OpenAIEmbeddingClient(url="http://example.test", model="x", batch_size=129)
        with self.assertRaises(ValueError):
            OpenAIEmbeddingClient(url="http://example.test", model="x", token_budget=255)
        client = OpenAIEmbeddingClient(
            url="http://example.test", model="x", token_budget=256,
            session=SuccessfulSession(),
        )
        self.assertEqual(client.estimated_tokens("x" * 600), 200)
        self.assertEqual(len(list(client.batches([("a", "x" * 600), ("b", "y" * 600)]))), 2)
        with self.assertRaisesRegex(RuntimeError, "token budget"):
            client.embed(["x" * 2000], dimensions=4)


if __name__ == "__main__":
    unittest.main()
