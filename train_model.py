#!/usr/bin/env python3
"""Train the deterministic production preference artifact from corrected evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize

from preferences import (
    ARTIFACT_SCHEMA_VERSION,
    CACHE_HASH_NORMALIZATION,
    EMBEDDING_INPUT_NORMALIZATION,
    FULL_TEXT_CONSTRUCTOR,
    SOURCE_SCORE_POLICY,
    TITLE_TEXT_CONSTRUCTOR,
    VECTOR_NORMALIZATION,
    full_embedding_text,
    title_embedding_text,
)

ROOT = Path(__file__).parent
DEFAULT_ANALYSIS = ROOT / "oracle-analysis" / "2026-08-23"
DEFAULT_SEED = 20260823
DEFAULT_COHORT = "pre-filtered-first-pass"
DEFAULT_CAMPAIGN = "first-disagreements"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def readonly(path: str | Path) -> sqlite3.Connection:
    resolved = Path(path).expanduser().resolve()
    connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def load_corrected_rows(
    oracle_path: str | Path,
    adjudication_path: str | Path,
    *, cohort: str, campaign: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    with closing(readonly(oracle_path)) as connection:
        rows = [dict(row) for row in connection.execute(
            """SELECT c.*,r.rating original_rating FROM candidates c
               JOIN ratings r ON r.cohort=c.cohort AND r.ad_id=c.ad_id
               WHERE c.cohort=? ORDER BY c.queue_order""", (cohort,))]
        cohort_row = connection.execute(
            "SELECT candidate_count FROM cohorts WHERE name=?", (cohort,)
        ).fetchone()
    if cohort_row is None or int(cohort_row["candidate_count"]) != len(rows):
        raise ValueError("oracle cohort candidate count is inconsistent")
    ids = [str(row["ad_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("oracle cohort contains duplicate ad IDs")
    if any(int(row["original_rating"]) not in range(1, 6) for row in rows):
        raise ValueError("oracle cohort contains an invalid rating")
    with closing(readonly(adjudication_path)) as connection:
        decisions = [dict(row) for row in connection.execute(
            """SELECT ad_id,adjudicated_rating,decision_state,note FROM decisions
               WHERE campaign=?""", (campaign,))]
        cases = {str(row["ad_id"]): int(row["original_rating"]) for row in connection.execute(
            "SELECT ad_id,original_rating FROM cases WHERE campaign=?", (campaign,))}
    decision_ids = [str(row["ad_id"]) for row in decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError("adjudication contains duplicate decision IDs")
    by_id = {str(row["ad_id"]): row for row in rows}
    for decision in decisions:
        ad_id, rating = str(decision["ad_id"]), int(decision["adjudicated_rating"])
        if ad_id not in by_id:
            raise ValueError(f"adjudication references unknown ad ID {ad_id}")
        if rating not in range(1, 6):
            raise ValueError(f"adjudication has invalid rating for {ad_id}")
        if cases.get(ad_id) != int(by_id[ad_id]["original_rating"]):
            raise ValueError(f"adjudication original rating disagrees for {ad_id}")
        by_id[ad_id]["corrected_rating"] = rating
    for row in rows:
        row.setdefault("corrected_rating", int(row["original_rating"]))
        source_score = float(row["source_score"])
        if not math.isfinite(source_score) or not 0.0 <= source_score <= 1.0:
            raise ValueError(f"source score outside 0-1 for {row['ad_id']}")
    return rows, {
        "total": len(rows),
        "adjudicated": len(decisions),
        "changed": sum(int(row["corrected_rating"] != row["original_rating"]) for row in rows),
    }


def _raw_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_embedding_artifact(
    matrix_path: str | Path,
    metadata_path: str | Path,
    rows: Sequence[dict[str, Any]],
    *, kind: str,
) -> tuple[np.ndarray, dict[str, Any], str]:
    matrix = np.load(Path(matrix_path), allow_pickle=False)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if matrix.ndim != 2 or not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all():
        raise ValueError(f"{kind} embedding matrix must be finite numeric 2-D data")
    expected_ids = [str(row["ad_id"]) for row in rows]
    if [str(value) for value in metadata.get("ad_ids", [])] != expected_ids:
        raise ValueError(f"{kind} embedding ad IDs/order disagree with the oracle cohort")
    if metadata.get("count") != len(rows) or matrix.shape[0] != len(rows):
        raise ValueError(f"{kind} embedding row count is inconsistent")
    if metadata.get("dimensions") != matrix.shape[1]:
        raise ValueError(f"{kind} embedding dimensions are inconsistent")
    if not str(metadata.get("model") or "").strip():
        raise ValueError(f"{kind} embedding model is missing")
    texts = ([title_embedding_text(row) for row in rows] if kind == "title"
             else [full_embedding_text(row) for row in rows])
    hashes = metadata.get("text_sha256")
    if hashes is None:
        if kind != "title":
            raise ValueError("full embedding metadata must contain text hashes")
        validation = "metadata-absent; raw headlines reconstructed and IDs/order validated"
    else:
        if list(hashes) != [_raw_sha256(text) for text in texts]:
            raise ValueError(f"{kind} embedding text hashes disagree with source text")
        validation = "validated-against-exact-constructor"
    return np.asarray(matrix, dtype=np.float64), metadata, validation


def load_stopwords(path: str | Path) -> list[str]:
    result: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        word = line.strip().lower()
        if word and not word.startswith("#") and " " not in word:
            result.append(word)
    return result


def title_vectorizer(stopwords: list[str]) -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, stop_words=stopwords,
             ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True,
             max_features=20000)),
        ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb",
             ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=20000)),
    ])


def classifier(*, c_value: float, seed: int) -> LogisticRegression:
    return LogisticRegression(C=c_value, class_weight="balanced", max_iter=5000,
                              solver="liblinear", random_state=seed)


def selected_regularization(report_path: str | Path) -> dict[str, Any]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    selected = report.get("selected_C", {})
    mapping = {"title_tfidf": "title_tfidf", "full_embedding": "full_embedding_1024",
               "title_embedding": "title_embedding_mrl256"}
    result: dict[str, float] = {}
    evidence: dict[str, list[float]] = {}
    for output_name, report_name in mapping.items():
        values = [float(value) for value in selected.get(report_name, [])]
        if not values:
            raise ValueError(f"evaluation report lacks selected C values for {report_name}")
        counts = Counter(values)
        best = max(counts.values())
        result[output_name] = min(value for value, count in counts.items() if count == best)
        evidence[output_name] = values
    return {
        "selection_policy": "modal corrected nested-OOF selected C; smallest C breaks ties",
        "values": result,
        "outer_fold_choices": evidence,
        "evaluation_only": True,
        "training_fit_not_reported_as_validation": True,
    }


def train(
    *, oracle_path: str | Path, adjudication_path: str | Path,
    full_matrix_path: str | Path, full_metadata_path: str | Path,
    title_matrix_path: str | Path, title_metadata_path: str | Path,
    report_path: str | Path, manifest_path: str | Path,
    stopwords_path: str | Path, output_path: str | Path,
    cohort: str = DEFAULT_COHORT, campaign: str = DEFAULT_CAMPAIGN,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    source_paths = {
        "oracle_db": Path(oracle_path), "adjudication_db": Path(adjudication_path),
        "full_matrix": Path(full_matrix_path), "full_metadata": Path(full_metadata_path),
        "title_matrix": Path(title_matrix_path), "title_metadata": Path(title_metadata_path),
        "corrected_oof_report": Path(report_path), "analysis_manifest": Path(manifest_path),
        "stopwords": Path(stopwords_path),
    }
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    rows, label_counts = load_corrected_rows(oracle_path, adjudication_path,
                                             cohort=cohort, campaign=campaign)
    full, full_metadata, full_validation = load_embedding_artifact(
        full_matrix_path, full_metadata_path, rows, kind="full")
    title, title_metadata, title_validation = load_embedding_artifact(
        title_matrix_path, title_metadata_path, rows, kind="title")
    if full_metadata["model"] != title_metadata["model"]:
        raise ValueError("full and title embedding model names disagree")
    if full.shape[1] != title.shape[1]:
        raise ValueError("full and title source embedding dimensions disagree")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    embedding_manifest = manifest.get("embedding", {})
    model_sha256 = str(embedding_manifest.get("model_sha256") or "")
    manifest_model = str(embedding_manifest.get("model") or "")
    source_model = str(full_metadata.get("source_model") or "")
    if len(model_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in model_sha256
    ):
        raise ValueError("analysis manifest embedding SHA-256 must be 64 lowercase hex characters")
    if source_model and Path(source_model).name != manifest_model:
        raise ValueError("full embedding metadata source model disagrees with analysis manifest")
    source_model_path = Path(source_model).expanduser()
    if source_model and not source_model_path.is_absolute():
        source_model_path = Path(full_metadata_path).resolve().parent / source_model_path
    if source_model and source_model_path.is_file():
        if sha256_file(source_model_path) != model_sha256:
            raise ValueError("available embedding model bytes disagree with manifest SHA-256")
        model_digest_status = "verified-file"
    else:
        model_digest_status = "declared-manifest"
    regularization = selected_regularization(report_path)
    clear = [index for index, row in enumerate(rows) if int(row["corrected_rating"]) != 3]
    y = np.asarray([int(int(rows[index]["corrected_rating"]) >= 4) for index in clear], dtype=np.int8)
    if set(y.tolist()) != {0, 1}:
        raise ValueError("clear corrected labels must contain both classes")
    titles = [title_embedding_text(rows[index]) for index in clear]
    vectorizer = title_vectorizer(load_stopwords(stopwords_path))
    title_features = vectorizer.fit_transform(titles)
    title_model = classifier(c_value=regularization["values"]["title_tfidf"], seed=seed).fit(title_features, y)
    full_used, title_used = full.shape[1], min(256, title.shape[1])
    full_model = classifier(c_value=regularization["values"]["full_embedding"], seed=seed).fit(
        normalize(full[clear, :full_used]), y)
    title_embedding_model = classifier(c_value=regularization["values"]["title_embedding"], seed=seed).fit(
        normalize(title[clear, :title_used]), y)
    training_counts = {
        **label_counts, "clear": int(y.size), "positive": int(y.sum()),
        "negative": int(y.size - y.sum()), "excluded_rating_3": len(rows) - int(y.size),
    }
    version_payload = {
        "schema": ARTIFACT_SCHEMA_VERSION, "cohort": cohort, "campaign": campaign,
        "seed": seed, "source_hashes": source_hashes,
        "regularization": regularization, "training_counts": training_counts,
        "constructors": [TITLE_TEXT_CONSTRUCTOR, FULL_TEXT_CONSTRUCTOR],
        "model_digest_status": model_digest_status,
    }
    version_digest = hashlib.sha256(json.dumps(
        version_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    model_version = f"jobsearcher-preference-{version_digest[:16]}"
    cache_model_name = (
        f"{full_metadata['model']}@sha256:{model_sha256}:{model_digest_status}"
    )
    artifact: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_version": model_version, "seed": seed, "cohort": cohort,
        "campaign": campaign,
        "label_policy": "adjudicated rating when present; 1-2 negative, 4-5 positive, 3 excluded",
        "title_vectorizer": vectorizer, "title_classifier": title_model,
        "full_embedding_classifier": full_model,
        "title_embedding_classifier": title_embedding_model,
        "preprocessing": {
            "title_text_constructor": TITLE_TEXT_CONSTRUCTOR,
            "title_embedding_constructor": TITLE_TEXT_CONSTRUCTOR,
            "full_embedding_constructor": FULL_TEXT_CONSTRUCTOR,
            "cache_hash_normalization": CACHE_HASH_NORMALIZATION,
            "embedding_input_normalization": EMBEDDING_INPUT_NORMALIZATION,
            "vector_normalization": VECTOR_NORMALIZATION,
            "source_score_policy": SOURCE_SCORE_POLICY,
            "full_metadata_hash_validation": full_validation,
            "title_metadata_hash_validation": title_validation,
        },
        "dimensions": {"full_source": int(full.shape[1]), "full_used": int(full_used),
                       "title_source": int(title.shape[1]), "title_used": int(title_used)},
        "embedding_model_name": str(full_metadata["model"]),
        "embedding_model_source": str(full_metadata.get("source_model") or embedding_manifest.get("model") or ""),
        "embedding_model_sha256": model_sha256,
        "embedding_model_digest_status": model_digest_status,
        "cache_model_name": cache_model_name,
        "source_hashes": source_hashes, "training_counts": training_counts,
        "regularization": regularization,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=0, protocol=5)
    return {"model_version": model_version, "output": str(output),
            "training_counts": training_counts, "cache_model_name": cache_model_name,
            "title_hash_validation": title_validation}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, default=ROOT / "oracle-ratings.db")
    parser.add_argument("--adjudication", type=Path, default=ROOT / "adjudication.db")
    parser.add_argument("--full-matrix", type=Path, default=DEFAULT_ANALYSIS / "jobsearcher-oracle-arctic-v2-2026-08-23.npy")
    parser.add_argument("--full-metadata", type=Path, default=DEFAULT_ANALYSIS / "jobsearcher-oracle-arctic-v2-2026-08-23.json")
    parser.add_argument("--title-matrix", type=Path, default=DEFAULT_ANALYSIS / "jobsearcher-oracle-title-arctic-v2-2026-08-23.npy")
    parser.add_argument("--title-metadata", type=Path, default=DEFAULT_ANALYSIS / "jobsearcher-oracle-title-arctic-v2-2026-08-23.json")
    parser.add_argument("--report", type=Path, default=DEFAULT_ANALYSIS / "jobsearcher-adjudicated-report.json")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_ANALYSIS / "manifest.json")
    parser.add_argument("--stopwords", type=Path, default=ROOT / "stopwords-sv.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "preference.joblib")
    parser.add_argument("--cohort", default=DEFAULT_COHORT)
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = train(
        oracle_path=args.oracle, adjudication_path=args.adjudication,
        full_matrix_path=args.full_matrix, full_metadata_path=args.full_metadata,
        title_matrix_path=args.title_matrix, title_metadata_path=args.title_metadata,
        report_path=args.report, manifest_path=args.manifest,
        stopwords_path=args.stopwords, output_path=args.output,
        cohort=args.cohort, campaign=args.campaign, seed=args.seed)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
