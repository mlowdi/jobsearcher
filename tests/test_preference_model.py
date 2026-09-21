from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from preferences import FULL_TEXT_CONSTRUCTOR, TITLE_TEXT_CONSTRUCTOR, full_embedding_text
from train_model import train


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_evidence(root: Path) -> dict[str, Path]:
    oracle = root / "oracle.db"
    connection = sqlite3.connect(oracle)
    connection.executescript(
        """
        CREATE TABLE cohorts(name TEXT PRIMARY KEY,candidate_count INTEGER);
        CREATE TABLE candidates(
          cohort TEXT,ad_id TEXT,queue_order INTEGER,source_score REAL,score_bucket INTEGER,
          headline TEXT,employer TEXT,employment_type TEXT,publication_date TEXT,
          application_deadline TEXT,webpage_url TEXT,description_text TEXT,
          municipality TEXT,region TEXT,occupation_group TEXT,keyword_score INTEGER,
          PRIMARY KEY(cohort,ad_id)
        );
        CREATE TABLE ratings(
          cohort TEXT,ad_id TEXT,rating INTEGER,rated_at TEXT,PRIMARY KEY(cohort,ad_id)
        );
        """
    )
    connection.execute("INSERT INTO cohorts VALUES ('pre-filtered-first-pass',12)")
    ratings = [1, 2, 4, 5, 1, 2, 4, 5, 1, 2, 4, 3]
    rows: list[dict[str, object]] = []
    for index, rating in enumerate(ratings):
        row = {
            "cohort": "pre-filtered-first-pass", "ad_id": str(100 + index),
            "queue_order": index + 1, "source_score": .1 + index * .06,
            "score_bucket": 1, "headline": ("Security leader" if rating >= 4 else "General operator") + f" {index % 3}",
            "employer": f"Employer {index}", "employment_type": "Full time",
            "publication_date": "2026-08-01", "application_deadline": "2026-09-30",
            "webpage_url": f"https://example.test/{index}",
            "description_text": "security strategy coordination" if rating >= 4 else "unrelated routine work",
            "municipality": "Stockholm", "region": "Stockholm",
            "occupation_group": "IT security" if rating >= 4 else "Operations",
            "keyword_score": 5,
        }
        rows.append(row)
        connection.execute(
            "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            tuple(row[key] for key in ("cohort","ad_id","queue_order","source_score","score_bucket",
                  "headline","employer","employment_type","publication_date","application_deadline",
                  "webpage_url","description_text","municipality","region","occupation_group","keyword_score")),
        )
        connection.execute("INSERT INTO ratings VALUES (?,?,?,?)",
                           (row["cohort"], row["ad_id"], rating, "2026-08-22T00:00:00+00:00"))
    connection.commit(); connection.close()

    adjudication = root / "adjudication.db"
    connection = sqlite3.connect(adjudication)
    connection.executescript(
        """
        CREATE TABLE cases(campaign TEXT,ad_id TEXT,original_rating INTEGER);
        CREATE TABLE decisions(campaign TEXT,ad_id TEXT,adjudicated_rating INTEGER,decision_state TEXT,note TEXT);
        """
    )
    connection.execute("INSERT INTO cases VALUES ('first-disagreements','100',1)")
    connection.execute("INSERT INTO decisions VALUES ('first-disagreements','100',4,'revised','corrected')")
    connection.commit(); connection.close()

    rng = np.random.default_rng(20260823)
    full = rng.normal(size=(12, 8)).astype(np.float32)
    title = rng.normal(size=(12, 8)).astype(np.float32)
    full_matrix, title_matrix = root / "full.npy", root / "title.npy"
    np.save(full_matrix, full); np.save(title_matrix, title)
    full_meta = root / "full.json"
    full_meta.write_text(json.dumps({
        "cohort": "pre-filtered-first-pass", "model": "fixture-q8",
        "source_model": "fixture-q8.gguf", "count": 12, "dimensions": 8,
        "ad_ids": [str(row["ad_id"]) for row in rows],
        "text_sha256": [hashlib.sha256(full_embedding_text(row).encode()).hexdigest() for row in rows],
    }), encoding="utf-8")
    title_meta = root / "title.json"
    title_meta.write_text(json.dumps({
        "model": "fixture-q8", "count": 12, "dimensions": 8,
        "ad_ids": [str(row["ad_id"]) for row in rows],
    }), encoding="utf-8")
    report = root / "report.json"
    report.write_text(json.dumps({"selected_C": {
        "title_tfidf": [10, 10, 10, 1, 10],
        "full_embedding_1024": [10, 10, 100, 10, 10],
        "title_embedding_mrl256": [10, 10, 100, 10, 10],
    }}), encoding="utf-8")
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"embedding": {
        "model": "fixture-q8.gguf", "model_sha256": "a" * 64,
    }}), encoding="utf-8")
    stopwords = root / "stopwords.txt"
    stopwords.write_text("och\nmed\n", encoding="utf-8")
    return {"oracle": oracle, "adjudication": adjudication,
            "full_matrix": full_matrix, "full_metadata": full_meta,
            "title_matrix": title_matrix, "title_metadata": title_meta,
            "report": report, "manifest": manifest, "stopwords": stopwords}


class PreferenceModelTest(unittest.TestCase):
    def test_artifact_schema_corrected_labels_exact_text_and_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = build_evidence(root)
            before = {name: digest(path) for name, path in evidence.items()}
            outputs = [root / "model-one.joblib", root / "model-two.joblib"]
            results = []
            for output in outputs:
                results.append(train(
                    oracle_path=evidence["oracle"], adjudication_path=evidence["adjudication"],
                    full_matrix_path=evidence["full_matrix"], full_metadata_path=evidence["full_metadata"],
                    title_matrix_path=evidence["title_matrix"], title_metadata_path=evidence["title_metadata"],
                    report_path=evidence["report"], manifest_path=evidence["manifest"],
                    stopwords_path=evidence["stopwords"], output_path=output))
            self.assertEqual(results[0]["model_version"], results[1]["model_version"])
            self.assertEqual(before, {name: digest(path) for name, path in evidence.items()})
            artifact = joblib.load(outputs[0])
            repeated = joblib.load(outputs[1])
            self.assertEqual(
                artifact["title_vectorizer"].transformer_list[0][1].vocabulary_,
                repeated["title_vectorizer"].transformer_list[0][1].vocabulary_,
            )
            for key in ("title_classifier", "full_embedding_classifier", "title_embedding_classifier"):
                np.testing.assert_array_equal(artifact[key].coef_, repeated[key].coef_)
                np.testing.assert_array_equal(artifact[key].intercept_, repeated[key].intercept_)
            self.assertEqual(artifact["training_counts"], {
                "total": 12, "adjudicated": 1, "changed": 1, "clear": 11,
                "positive": 6, "negative": 5, "excluded_rating_3": 1,
            })
            self.assertEqual(artifact["preprocessing"]["title_text_constructor"], TITLE_TEXT_CONSTRUCTOR)
            self.assertEqual(artifact["preprocessing"]["full_embedding_constructor"], FULL_TEXT_CONSTRUCTOR)
            self.assertIn("metadata-absent", artifact["preprocessing"]["title_metadata_hash_validation"])
            self.assertEqual(artifact["regularization"]["values"], {
                "title_tfidf": 10.0, "full_embedding": 10.0, "title_embedding": 10.0,
            })
            self.assertTrue(artifact["regularization"]["training_fit_not_reported_as_validation"])
            self.assertEqual(artifact["dimensions"], {
                "full_source": 8, "full_used": 8, "title_source": 8, "title_used": 8,
            })
            self.assertEqual(
                artifact["cache_model_name"],
                "fixture-q8@sha256:" + "a" * 64 + ":declared-manifest",
            )
            self.assertEqual(artifact["embedding_model_digest_status"], "declared-manifest")
            self.assertTrue({"title_vectorizer", "title_classifier",
                             "full_embedding_classifier", "title_embedding_classifier",
                             "source_hashes", "model_version"} <= artifact.keys())

    def test_model_digest_must_be_hex_and_match_available_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); evidence = build_evidence(root)
            manifest = json.loads(evidence["manifest"].read_text())
            manifest["embedding"]["model_sha256"] = "z" * 64
            evidence["manifest"].write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "lowercase hex"):
                train(oracle_path=evidence["oracle"], adjudication_path=evidence["adjudication"],
                      full_matrix_path=evidence["full_matrix"], full_metadata_path=evidence["full_metadata"],
                      title_matrix_path=evidence["title_matrix"], title_metadata_path=evidence["title_metadata"],
                      report_path=evidence["report"], manifest_path=evidence["manifest"],
                      stopwords_path=evidence["stopwords"], output_path=root / "bad.joblib")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); evidence = build_evidence(root)
            (root / "fixture-q8.gguf").write_bytes(b"different model bytes")
            with self.assertRaisesRegex(ValueError, "model bytes disagree"):
                train(oracle_path=evidence["oracle"], adjudication_path=evidence["adjudication"],
                      full_matrix_path=evidence["full_matrix"], full_metadata_path=evidence["full_metadata"],
                      title_matrix_path=evidence["title_matrix"], title_metadata_path=evidence["title_metadata"],
                      report_path=evidence["report"], manifest_path=evidence["manifest"],
                      stopwords_path=evidence["stopwords"], output_path=root / "bad.joblib")

    def test_full_metadata_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); evidence = build_evidence(root)
            metadata = json.loads(evidence["full_metadata"].read_text())
            metadata["text_sha256"][0] = "0" * 64
            evidence["full_metadata"].write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "text hashes"):
                train(oracle_path=evidence["oracle"], adjudication_path=evidence["adjudication"],
                      full_matrix_path=evidence["full_matrix"], full_metadata_path=evidence["full_metadata"],
                      title_matrix_path=evidence["title_matrix"], title_metadata_path=evidence["title_metadata"],
                      report_path=evidence["report"], manifest_path=evidence["manifest"],
                      stopwords_path=evidence["stopwords"], output_path=root / "model.joblib")


if __name__ == "__main__":
    unittest.main()
