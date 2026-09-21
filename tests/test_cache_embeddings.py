from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cache_embeddings import CACHE_IMPORT_SCHEMA_VERSION, import_cache
from db import connect, normalized_text_hash, upsert_ads
from preferences import (
    CACHE_HASH_NORMALIZATION,
    EMBEDDING_INPUT_NORMALIZATION,
    FULL_TEXT_CONSTRUCTOR,
    TITLE_TEXT_CONSTRUCTOR,
    full_embedding_text,
    title_embedding_text,
)
from tests.test_workspace_vertical import ad


def metadata_for(rows, kind: str, matrix: np.ndarray) -> dict[str, object]:
    constructor = full_embedding_text if kind == "full" else title_embedding_text
    return {
        "artifact_schema_version": CACHE_IMPORT_SCHEMA_VERSION,
        "kind": kind,
        "text_constructor": FULL_TEXT_CONSTRUCTOR if kind == "full" else TITLE_TEXT_CONSTRUCTOR,
        "cache_hash_normalization": CACHE_HASH_NORMALIZATION,
        "embedding_input_normalization": EMBEDDING_INPUT_NORMALIZATION,
        "model": "fixture-q8",
        "model_sha256": "a" * 64,
        "model_digest_status": "verified-file",
        "cache_model_name": "fixture-q8@sha256:" + "a" * 64 + ":verified-file",
        "count": len(rows),
        "dimensions": matrix.shape[1],
        "ad_ids": [str(row["id"]) for row in rows],
        "text_sha256": [normalized_text_hash(constructor(row)) for row in rows],
    }


class OfflineCacheImportTest(unittest.TestCase):
    def test_full_and_title_import_are_idempotent_and_do_not_change_ads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); db_path = root / "workspace.db"
            rows = [ad("101", "Security Manager", "A", .8), ad("102", "Analyst", "B", .4)]
            upsert_ads(rows, db_path, seen_on="2026-08-23")
            connection = connect(db_path, writable=False)
            try:
                ads_before = [tuple(row) for row in connection.execute("SELECT * FROM ads ORDER BY id")]
            finally:
                connection.close()
            for kind, value in (("full", 1.0), ("title", 2.0)):
                matrix = np.full((2, 4), value, dtype=np.float32)
                matrix_path, metadata_path = root / f"{kind}.npy", root / f"{kind}.json"
                np.save(matrix_path, matrix)
                metadata_path.write_text(json.dumps(metadata_for(rows, kind, matrix)), encoding="utf-8")
                first = import_cache(db_path=db_path, matrix_path=matrix_path,
                                     metadata_path=metadata_path)
                second = import_cache(db_path=db_path, matrix_path=matrix_path,
                                      metadata_path=metadata_path)
                self.assertEqual((first["imported"], first["skipped"]), (2, 0))
                self.assertEqual((second["imported"], second["skipped"]), (0, 2))
            connection = connect(db_path, writable=False)
            try:
                self.assertEqual([tuple(row) for row in connection.execute("SELECT * FROM ads ORDER BY id")], ads_before)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM triage_events").fetchone()[0], 0)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0], 4)
            finally:
                connection.close()

    def test_ids_dimensions_hashes_and_model_provenance_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); db_path = root / "workspace.db"
            rows = [ad("101", "Security Manager", "A", .8)]
            upsert_ads(rows, db_path, seen_on="2026-08-23")
            matrix = np.ones((1, 4), dtype=np.float32)
            matrix_path = root / "matrix.npy"; np.save(matrix_path, matrix)
            base = metadata_for(rows, "full", matrix)
            mutations = {
                "missing ad IDs": lambda meta: meta.update(ad_ids=["999"]),
                "shape disagrees": lambda meta: meta.update(dimensions=3),
                "text hashes disagree": lambda meta: meta.update(text_sha256=["0" * 64]),
                "model provenance": lambda meta: meta.update(cache_model_name="endpoint:fixture-q4"),
                "nonhex digest": lambda meta: meta.update(
                    model_sha256="z" * 64,
                    cache_model_name="fixture-q8@sha256:" + "z" * 64 + ":verified-file",
                ),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    metadata = dict(base); mutate(metadata)
                    metadata_path = root / f"{name.replace(' ', '-')}.json"
                    metadata_path.write_text(json.dumps(metadata))
                    with self.assertRaises(ValueError):
                        import_cache(db_path=db_path, matrix_path=matrix_path,
                                     metadata_path=metadata_path)
            connection = connect(db_path, writable=False)
            try:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0], 0)
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
