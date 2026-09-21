#!/usr/bin/env python3
"""Import validated offline embedding matrices into the workspace cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from db import DB_PATH, connect, migrate, normalized_text_hash, put_cached_vector
from preferences import (
    CACHE_HASH_NORMALIZATION,
    EMBEDDING_INPUT_NORMALIZATION,
    FULL_TEXT_CONSTRUCTOR,
    TITLE_TEXT_CONSTRUCTOR,
    full_embedding_text,
    title_embedding_text,
)

CACHE_IMPORT_SCHEMA_VERSION = 1


def validate_metadata(metadata: dict[str, Any], matrix: np.ndarray) -> None:
    if metadata.get("artifact_schema_version") != CACHE_IMPORT_SCHEMA_VERSION:
        raise ValueError("unsupported cache-import metadata schema")
    kind = metadata.get("kind")
    if kind not in {"full", "title"}:
        raise ValueError("cache metadata kind must be full or title")
    expected_constructor = FULL_TEXT_CONSTRUCTOR if kind == "full" else TITLE_TEXT_CONSTRUCTOR
    if metadata.get("text_constructor") != expected_constructor:
        raise ValueError("cache metadata text constructor does not match production")
    if metadata.get("cache_hash_normalization") != CACHE_HASH_NORMALIZATION:
        raise ValueError("cache metadata hash normalization does not match production")
    if metadata.get("embedding_input_normalization") != EMBEDDING_INPUT_NORMALIZATION:
        raise ValueError("cache metadata embedding input normalization does not match production")
    model = str(metadata.get("model") or "")
    model_hash = str(metadata.get("model_sha256") or "")
    digest_status = metadata.get("model_digest_status")
    expected_cache_name = f"{model}@sha256:{model_hash}:{digest_status}"
    if not model or len(model_hash) != 64 or any(ch not in "0123456789abcdef" for ch in model_hash):
        raise ValueError("cache metadata requires model and lowercase model SHA-256")
    if digest_status not in {"verified-file", "declared-manifest"}:
        raise ValueError("cache metadata model digest status is invalid")
    if metadata.get("cache_model_name") != expected_cache_name:
        raise ValueError("cache model provenance is inconsistent")
    if matrix.ndim != 2 or not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all():
        raise ValueError("embedding matrix must be finite numeric 2-D data")
    count, dimensions = matrix.shape
    if count <= 0 or dimensions <= 0:
        raise ValueError("embedding matrix must contain at least one non-empty vector")
    if metadata.get("count") != count or metadata.get("dimensions") != dimensions:
        raise ValueError("embedding matrix shape disagrees with metadata")
    ids = [str(value) for value in metadata.get("ad_ids", [])]
    hashes = list(metadata.get("text_sha256", []))
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError("cache metadata ad IDs are missing or duplicated")
    if len(hashes) != count:
        raise ValueError("cache metadata text hash count is inconsistent")
    for value in hashes:
        if not isinstance(value, str) or len(value) != 64 or any(
            ch not in "0123456789abcdef" for ch in value
        ):
            raise ValueError("cache metadata contains an invalid text SHA-256")


def import_cache(
    *, db_path: str | Path, matrix_path: str | Path, metadata_path: str | Path
) -> dict[str, int | str]:
    matrix = np.load(Path(matrix_path), allow_pickle=False)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    validate_metadata(metadata, matrix)
    migrate(db_path)
    ids = [str(value) for value in metadata["ad_ids"]]
    connection = connect(db_path)
    imported = skipped = 0
    try:
        placeholders = ",".join("?" for _ in ids)
        rows = {
            str(row["id"]): dict(row)
            for row in connection.execute(
                f"SELECT * FROM ads WHERE id IN ({placeholders})", ids
            )
        }
        missing = [ad_id for ad_id in ids if ad_id not in rows]
        if missing:
            raise ValueError(f"cache metadata references missing ad IDs: {', '.join(missing[:5])}")
        constructor = title_embedding_text if metadata["kind"] == "title" else full_embedding_text
        expected_hashes = [normalized_text_hash(constructor(rows[ad_id])) for ad_id in ids]
        if expected_hashes != list(metadata["text_sha256"]):
            raise ValueError("cache text hashes disagree with current ad text")
        connection.execute("BEGIN IMMEDIATE")
        try:
            for index, text_hash in enumerate(expected_hashes):
                inserted = put_cached_vector(
                    connection,
                    text_hash=text_hash,
                    model_name=metadata["cache_model_name"],
                    vector=matrix[index],
                    dimensions=int(metadata["dimensions"]),
                )
                imported += int(inserted)
                skipped += int(not inserted)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    finally:
        connection.close()
    return {
        "imported": imported,
        "skipped": skipped,
        "kind": metadata["kind"],
        "cache_model_name": metadata["cache_model_name"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = import_cache(db_path=args.db, matrix_path=args.matrix, metadata_path=args.metadata)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
