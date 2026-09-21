"""Local adjudication UI for a frozen campaign of model/human disagreements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit

from flask import Flask, Response, abort, redirect, render_template, request, url_for

APP_DIR = Path(__file__).resolve().parent
DEFAULT_RATINGS_DB = APP_DIR / "oracle-ratings.db"
DEFAULT_PREDICTIONS_CSV = (
    APP_DIR
    / "oracle-analysis"
    / "2026-08-23"
    / "jobsearcher-title-semantic-oof.csv"
)
DEFAULT_STATE_DB = APP_DIR / "adjudication.db"
DEFAULT_CAMPAIGN = "first-disagreements"
DEFAULT_COHORT = "pre-filtered-first-pass"
DEFAULT_PORT = 5001
SELECTION_VERSION = "frozen-disagreement-lanes-v1"
STATE_SCHEMA_VERSION = "1"
LANE_LIMIT = 20

LANES = (
    "model-overestimate",
    "model-underestimate",
    "signal-disagreement",
)

RATING_LABELS = (
    (1, "No, not for me"),
    (2, "Probably not"),
    (3, "Maybe"),
    (4, "Interesting"),
    (5, "Yes, I want to look into this"),
)

REASON_GROUPS = (
    (
        "Negative / constraint",
        (
            ("wrong-role-family", "Wrong role family"),
            ("too-technical", "Too technical"),
            ("too-senior", "Too senior"),
            ("too-junior", "Too junior"),
            ("consultancy-contract", "Consultancy or contract setup"),
            ("network-ot-infra", "Network, OT, or infrastructure focus"),
            ("legal-privacy-mismatch", "Legal or privacy mismatch"),
            (
                "security-protection-mismatch",
                "Security-protection mismatch",
            ),
            ("location-commute", "Location or commute"),
            ("credential-gate", "Credential gate"),
            ("employer-mission-mismatch", "Employer or mission mismatch"),
        ),
    ),
    (
        "Positive / fit",
        (
            ("security-strategy-grc", "Security strategy or GRC"),
            (
                "coordination-delivery-change",
                "Coordination, delivery, or change",
            ),
            (
                "communication-training-human",
                "Communication, training, or human focus",
            ),
            ("meaningful-mission", "Meaningful mission"),
            (
                "interesting-adjacent-wildcard",
                "Interesting adjacent wildcard",
            ),
        ),
    ),
    (
        "Evidence / label",
        (
            ("misleading-vague-ad", "Misleading or vague ad"),
            ("duplicate-repost", "Duplicate or repost"),
            ("changed-mind-context", "Changed mind or context"),
            ("other-unclear", "Other or unclear"),
        ),
    ),
)
REASON_CODES = frozenset(
    code for _group, reasons in REASON_GROUPS for code, _label in reasons
)
REASON_ORDER = {
    code: index
    for index, code in enumerate(
        code for _group, reasons in REASON_GROUPS for code, _label in reasons
    )
}

SOURCE_CASE_COLUMNS = (
    "queue_order",
    "source_score",
    "keyword_score",
    "score_bucket",
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
)
REQUIRED_PREDICTION_COLUMNS = frozenset(
    {
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
    }
)
CORE_SIGNAL_COLUMNS = (
    "oof_title_tfidf",
    "oof_embedding_mrl256",
    "oof_title_embedding",
    "oof_four_way_ensemble",
    "model_spread",
)
OPTIONAL_SIGNAL_COLUMNS = (
    "oof_full_tfidf",
    "oof_embedding_plus_scores",
    "oof_centroid",
    "oof_ensemble",
    "absolute_ensemble_error",
)
ALL_SIGNAL_COLUMNS = CORE_SIGNAL_COLUMNS + OPTIONAL_SIGNAL_COLUMNS

_STATE_SCHEMA = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE campaigns (
    name TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    source_cohort TEXT NOT NULL,
    source_db_path TEXT NOT NULL,
    source_db_sha256 TEXT NOT NULL CHECK (length(source_db_sha256) = 64),
    predictions_csv_path TEXT NOT NULL,
    predictions_csv_sha256 TEXT NOT NULL
        CHECK (length(predictions_csv_sha256) = 64),
    selection_version TEXT NOT NULL,
    candidate_count INTEGER NOT NULL CHECK (candidate_count >= 0),
    model_overestimate_count INTEGER NOT NULL CHECK (model_overestimate_count >= 0),
    model_underestimate_count INTEGER NOT NULL CHECK (model_underestimate_count >= 0),
    signal_disagreement_count INTEGER NOT NULL CHECK (signal_disagreement_count >= 0)
);

CREATE TABLE cases (
    campaign TEXT NOT NULL,
    ad_id TEXT NOT NULL,
    queue_order INTEGER NOT NULL CHECK (queue_order > 0),
    lane TEXT NOT NULL CHECK (
        lane IN (
            'model-overestimate',
            'model-underestimate',
            'signal-disagreement'
        )
    ),
    selection_explanation TEXT NOT NULL,
    frozen_at TEXT NOT NULL,
    source_queue_order INTEGER NOT NULL CHECK (source_queue_order > 0),
    source_score REAL,
    keyword_score INTEGER,
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
    original_rating INTEGER NOT NULL CHECK (original_rating IN (1, 2, 4, 5)),
    oof_title_tfidf REAL NOT NULL,
    oof_embedding_mrl256 REAL NOT NULL,
    oof_title_embedding REAL NOT NULL,
    oof_four_way_ensemble REAL NOT NULL,
    model_spread REAL NOT NULL,
    oof_full_tfidf REAL,
    oof_embedding_plus_scores REAL,
    oof_centroid REAL,
    oof_ensemble REAL,
    absolute_ensemble_error REAL,
    PRIMARY KEY (campaign, ad_id),
    UNIQUE (campaign, queue_order),
    FOREIGN KEY (campaign) REFERENCES campaigns(name) ON DELETE RESTRICT
);

CREATE TABLE decisions (
    campaign TEXT NOT NULL,
    ad_id TEXT NOT NULL,
    adjudicated_rating INTEGER NOT NULL
        CHECK (adjudicated_rating BETWEEN 1 AND 5),
    decision_state TEXT NOT NULL CHECK (decision_state IN ('confirmed', 'revised')),
    note TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    PRIMARY KEY (campaign, ad_id),
    FOREIGN KEY (campaign, ad_id)
        REFERENCES cases(campaign, ad_id) ON DELETE RESTRICT
);

CREATE TABLE decision_reasons (
    campaign TEXT NOT NULL,
    ad_id TEXT NOT NULL,
    reason_order INTEGER NOT NULL CHECK (reason_order >= 0),
    reason_code TEXT NOT NULL,
    PRIMARY KEY (campaign, ad_id, reason_code),
    UNIQUE (campaign, ad_id, reason_order),
    FOREIGN KEY (campaign, ad_id)
        REFERENCES decisions(campaign, ad_id) ON DELETE RESTRICT
);

CREATE TABLE case_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign TEXT NOT NULL,
    ad_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('presented', 'deferred')),
    occurred_at TEXT NOT NULL,
    FOREIGN KEY (campaign, ad_id)
        REFERENCES cases(campaign, ad_id) ON DELETE RESTRICT
);

CREATE INDEX cases_campaign_lane_idx ON cases(campaign, lane, queue_order);
CREATE INDEX case_events_candidate_idx
    ON case_events(campaign, ad_id, event_type, id);
"""


class AdjudicationError(RuntimeError):
    """An input or startup error that should be shown to the local operator."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AdjudicationError(f"cannot hash input file {path}: {exc}") from exc
    return digest.hexdigest()


def _resolve_input_path(
    path: str | os.PathLike[str], *, description: str
) -> Path:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise AdjudicationError(f"{description} is unavailable: {candidate}") from exc
    if not resolved.is_file():
        raise AdjudicationError(f"{description} is not a file: {resolved}")
    return resolved


def _resolve_state_path(path: str | os.PathLike[str]) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.parent.is_dir():
        raise AdjudicationError(
            f"state database directory does not exist: {resolved.parent}"
        )
    if resolved.exists() and not resolved.is_file():
        raise AdjudicationError(f"state database is not a file: {resolved}")
    return resolved


def _same_file(left: Path, right: Path) -> bool:
    if left == right:
        return True
    if not left.exists() or not right.exists():
        return False
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _open_readonly(path: Path, *, description: str) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(
            f"{path.as_uri()}?mode=ro", uri=True, timeout=5
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection
    except sqlite3.Error as exc:
        raise AdjudicationError(
            f"cannot open {description} read-only: {path}: {exc}"
        ) from exc


def _open_state(path: Path) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection
    except sqlite3.Error as exc:
        raise AdjudicationError(f"cannot open state database {path}: {exc}") from exc


def _require_columns(
    connection: sqlite3.Connection,
    table: str,
    required: Iterable[str],
    *,
    description: str,
) -> None:
    try:
        columns = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})")
        }
    except sqlite3.Error as exc:
        raise AdjudicationError(f"cannot inspect {description} table {table}: {exc}") from exc
    missing = sorted(set(required) - columns)
    if missing:
        raise AdjudicationError(
            f"{description} table {table} is missing required columns: "
            + ", ".join(missing)
        )


def _load_source_cases(source_path: Path, cohort: str) -> dict[str, dict[str, Any]]:
    connection = _open_readonly(source_path, description="ratings source database")
    try:
        _require_columns(
            connection,
            "cohorts",
            ("name", "candidate_count"),
            description="ratings source",
        )
        _require_columns(
            connection,
            "candidates",
            ("cohort", "ad_id", *SOURCE_CASE_COLUMNS),
            description="ratings source",
        )
        _require_columns(
            connection,
            "ratings",
            ("cohort", "ad_id", "rating", "rated_at"),
            description="ratings source",
        )

        cohort_rows = connection.execute(
            "SELECT candidate_count FROM cohorts WHERE name = ?", (cohort,)
        ).fetchall()
        if not cohort_rows:
            raise AdjudicationError(
                f"source cohort {cohort!r} does not exist in {source_path}"
            )
        if len(cohort_rows) != 1:
            raise AdjudicationError(f"source cohort {cohort!r} is duplicated")

        candidate_rows = connection.execute(
            f"""
            SELECT ad_id, {', '.join(SOURCE_CASE_COLUMNS)}
            FROM candidates
            WHERE cohort = ?
            ORDER BY queue_order, ad_id
            """,
            (cohort,),
        ).fetchall()
        rating_rows = connection.execute(
            """
            SELECT ad_id, rating
            FROM ratings
            WHERE cohort = ?
            ORDER BY ad_id
            """,
            (cohort,),
        ).fetchall()
    except sqlite3.Error as exc:
        raise AdjudicationError(
            f"cannot read source cohort {cohort!r} from {source_path}: {exc}"
        ) from exc
    finally:
        connection.close()

    declared_count = cohort_rows[0]["candidate_count"]
    if type(declared_count) is not int or declared_count != len(candidate_rows):
        raise AdjudicationError(
            f"source cohort {cohort!r} candidate count is inconsistent: "
            f"metadata says {declared_count!r}, table has {len(candidate_rows)}"
        )

    candidates: dict[str, dict[str, Any]] = {}
    queue_orders: set[int] = set()
    for row in candidate_rows:
        ad_id = str(row["ad_id"])
        if not ad_id:
            raise AdjudicationError(f"source cohort {cohort!r} contains an empty ad_id")
        if ad_id in candidates:
            raise AdjudicationError(
                f"source cohort {cohort!r} contains duplicate candidate ad_id {ad_id!r}"
            )
        queue_order = row["queue_order"]
        if type(queue_order) is not int or queue_order <= 0:
            raise AdjudicationError(
                f"source cohort {cohort!r} has invalid queue_order for ad_id {ad_id!r}"
            )
        if queue_order in queue_orders:
            raise AdjudicationError(
                f"source cohort {cohort!r} has duplicate queue_order {queue_order}"
            )
        queue_orders.add(queue_order)
        candidates[ad_id] = {column: row[column] for column in SOURCE_CASE_COLUMNS}

    ratings: dict[str, int] = {}
    for row in rating_rows:
        ad_id = str(row["ad_id"])
        if ad_id in ratings:
            raise AdjudicationError(
                f"source cohort {cohort!r} contains duplicate rating ad_id {ad_id!r}"
            )
        rating = row["rating"]
        if type(rating) is not int or rating not in {1, 2, 3, 4, 5}:
            raise AdjudicationError(
                f"source cohort {cohort!r} has invalid rating for ad_id {ad_id!r}"
            )
        ratings[ad_id] = rating

    candidate_ids = set(candidates)
    rating_ids = set(ratings)
    if candidate_ids != rating_ids:
        missing_ratings = sorted(candidate_ids - rating_ids)
        unknown_ratings = sorted(rating_ids - candidate_ids)
        details: list[str] = []
        if missing_ratings:
            details.append(
                "candidates without ratings: " + ", ".join(missing_ratings[:5])
            )
        if unknown_ratings:
            details.append(
                "ratings without candidates: " + ", ".join(unknown_ratings[:5])
            )
        raise AdjudicationError(
            f"source cohort {cohort!r} is incomplete or inconsistent ("
            + "; ".join(details)
            + ")"
        )

    for ad_id, rating in ratings.items():
        candidates[ad_id]["rating"] = rating
    return candidates


def _parse_float(
    value: str | None,
    *,
    column: str,
    ad_id: str,
    optional: bool = False,
) -> float | None:
    if value is None or not value.strip():
        if optional:
            return None
        raise AdjudicationError(
            f"prediction column {column!r} is empty for ad_id {ad_id!r}"
        )
    try:
        parsed = float(value)
    except ValueError as exc:
        raise AdjudicationError(
            f"prediction column {column!r} is not numeric for ad_id {ad_id!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise AdjudicationError(
            f"prediction column {column!r} is not finite for ad_id {ad_id!r}"
        )
    return parsed


def _load_predictions(predictions_path: Path) -> dict[str, dict[str, Any]]:
    try:
        input_file = predictions_path.open(newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise AdjudicationError(
            f"cannot open frozen predictions CSV {predictions_path}: {exc}"
        ) from exc

    try:
        with input_file:
            reader = csv.DictReader(input_file)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise AdjudicationError("frozen predictions CSV has no header")
            if len(fieldnames) != len(set(fieldnames)):
                raise AdjudicationError(
                    "frozen predictions CSV contains duplicate column names"
                )
            missing_columns = sorted(REQUIRED_PREDICTION_COLUMNS - set(fieldnames))
            if missing_columns:
                raise AdjudicationError(
                    "frozen predictions CSV is missing required columns: "
                    + ", ".join(missing_columns)
                )

            predictions: dict[str, dict[str, Any]] = {}
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise AdjudicationError(
                        f"frozen predictions CSV row {line_number} has extra fields"
                    )
                raw_ad_id = row.get("ad_id")
                ad_id = "" if raw_ad_id is None else raw_ad_id.strip()
                if not ad_id:
                    raise AdjudicationError(
                        f"frozen predictions CSV row {line_number} has an empty ad_id"
                    )
                if ad_id in predictions:
                    raise AdjudicationError(
                        f"frozen predictions CSV contains duplicate prediction ad_id {ad_id!r}"
                    )

                rating_text = row.get("rating")
                if rating_text not in {"1", "2", "3", "4", "5"}:
                    raise AdjudicationError(
                        f"prediction rating is invalid for ad_id {ad_id!r}"
                    )

                prediction: dict[str, Any] = {
                    "ad_id": ad_id,
                    "headline": row.get("headline", ""),
                    "employer": row.get("employer", ""),
                    "rating": int(rating_text),
                    "keyword_score": _parse_float(
                        row.get("keyword_score"),
                        column="keyword_score",
                        ad_id=ad_id,
                        optional=True,
                    ),
                    "source_score": _parse_float(
                        row.get("source_score"),
                        column="source_score",
                        ad_id=ad_id,
                        optional=True,
                    ),
                }
                for column in CORE_SIGNAL_COLUMNS:
                    prediction[column] = _parse_float(
                        row.get(column), column=column, ad_id=ad_id
                    )
                for column in OPTIONAL_SIGNAL_COLUMNS:
                    prediction[column] = _parse_float(
                        row.get(column),
                        column=column,
                        ad_id=ad_id,
                        optional=True,
                    )
                predictions[ad_id] = prediction
    except csv.Error as exc:
        raise AdjudicationError(
            f"cannot parse frozen predictions CSV {predictions_path}: {exc}"
        ) from exc
    return predictions


def _numbers_match(predicted: float | None, source: Any) -> bool:
    if source is None:
        return predicted is None
    if predicted is None:
        return False
    try:
        source_number = float(source)
    except (TypeError, ValueError):
        return False
    return math.isfinite(source_number) and math.isclose(
        predicted, source_number, rel_tol=1e-12, abs_tol=1e-12
    )


def _join_predictions(
    source_cases: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    clear_source_ids = {
        ad_id for ad_id, case in source_cases.items() if case["rating"] in {1, 2, 4, 5}
    }
    prediction_ids = set(predictions)

    missing_predictions = sorted(clear_source_ids - prediction_ids)
    unknown_predictions = sorted(prediction_ids - clear_source_ids)
    if missing_predictions:
        raise AdjudicationError(
            "frozen predictions CSV is missing clear-label source ad_id values: "
            + ", ".join(missing_predictions[:10])
        )
    if unknown_predictions:
        raise AdjudicationError(
            "prediction ad_id values are missing from the source cohort clear-label join: "
            + ", ".join(unknown_predictions[:10])
        )

    joined: list[dict[str, Any]] = []
    for ad_id in sorted(clear_source_ids):
        source = source_cases[ad_id]
        prediction = predictions[ad_id]
        mismatches: list[str] = []
        if prediction["rating"] != source["rating"]:
            mismatches.append("rating")
        if prediction["headline"] != source["headline"]:
            mismatches.append("headline")
        if prediction["employer"] != source["employer"]:
            mismatches.append("employer")
        if not _numbers_match(prediction["keyword_score"], source["keyword_score"]):
            mismatches.append("keyword_score")
        if not _numbers_match(prediction["source_score"], source["source_score"]):
            mismatches.append("source_score")
        if mismatches:
            raise AdjudicationError(
                f"prediction/source mismatch for ad_id {ad_id!r}: "
                + ", ".join(mismatches)
            )

        joined_case = {"ad_id": ad_id, **source}
        for column in ALL_SIGNAL_COLUMNS:
            joined_case[column] = prediction[column]
        joined.append(joined_case)
    return joined


def _select_cases(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    negative = [case for case in joined if case["rating"] in {1, 2}]
    positive = [case for case in joined if case["rating"] in {4, 5}]

    overestimates = sorted(
        negative,
        key=lambda case: (-case["oof_four_way_ensemble"], case["ad_id"]),
    )[:LANE_LIMIT]
    selected_ids = {case["ad_id"] for case in overestimates}

    underestimates = sorted(
        positive,
        key=lambda case: (case["oof_four_way_ensemble"], case["ad_id"]),
    )[:LANE_LIMIT]
    selected_ids.update(case["ad_id"] for case in underestimates)

    remaining = [case for case in joined if case["ad_id"] not in selected_ids]
    disagreements = sorted(
        remaining,
        key=lambda case: (-case["model_spread"], case["ad_id"]),
    )[:LANE_LIMIT]

    frozen_at = _utc_now()
    selected: list[dict[str, Any]] = []
    lane_groups = (
        ("model-overestimate", overestimates),
        ("model-underestimate", underestimates),
        ("signal-disagreement", disagreements),
    )
    for lane, cases in lane_groups:
        for case in cases:
            frozen = dict(case)
            frozen["source_queue_order"] = case["queue_order"]
            frozen["lane"] = lane
            frozen["frozen_at"] = frozen_at
            if lane == "model-overestimate":
                frozen["selection_explanation"] = (
                    f"Original rating {case['rating']} was negative; selected among "
                    "the highest frozen ensemble preference probabilities "
                    f"({case['oof_four_way_ensemble']:.6f})."
                )
            elif lane == "model-underestimate":
                frozen["selection_explanation"] = (
                    f"Original rating {case['rating']} was positive; selected among "
                    "the lowest frozen ensemble preference probabilities "
                    f"({case['oof_four_way_ensemble']:.6f})."
                )
            else:
                frozen["selection_explanation"] = (
                    f"Original rating {case['rating']} was clear; selected from the "
                    "remaining cases with the widest frozen component spread "
                    f"({case['model_spread']:.6f})."
                )
            frozen["queue_order"] = len(selected) + 1
            selected.append(frozen)
    return selected


def _build_selection(
    source_path: Path, predictions_path: Path, cohort: str
) -> list[dict[str, Any]]:
    source_cases = _load_source_cases(source_path, cohort)
    predictions = _load_predictions(predictions_path)
    return _select_cases(_join_predictions(source_cases, predictions))


def _read_existing_campaign(state_path: Path, campaign: str) -> dict[str, Any] | None:
    if not state_path.exists() or state_path.stat().st_size == 0:
        return None
    connection = _open_readonly(state_path, description="adjudication state database")
    try:
        tables = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        required_tables = {
            "metadata",
            "campaigns",
            "cases",
            "decisions",
            "decision_reasons",
            "case_events",
        }
        if not required_tables.issubset(tables):
            raise AdjudicationError(
                f"existing state database is not a compatible adjudication database: {state_path}"
            )
        schema_row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        if schema_row is None or schema_row["value"] != STATE_SCHEMA_VERSION:
            found = None if schema_row is None else schema_row["value"]
            raise AdjudicationError(
                f"state database schema version is unsupported: {found!r}"
            )
        row = connection.execute(
            "SELECT * FROM campaigns WHERE name = ?", (campaign,)
        ).fetchone()
        return None if row is None else dict(row)
    except sqlite3.Error as exc:
        raise AdjudicationError(
            f"cannot inspect adjudication state database {state_path}: {exc}"
        ) from exc
    finally:
        connection.close()


def _provenance_mismatches(
    existing: dict[str, Any],
    *,
    source_path: Path,
    source_hash: str,
    predictions_path: Path,
    predictions_hash: str,
    cohort: str,
) -> list[str]:
    mismatches: list[str] = []
    expected = (
        ("source cohort", existing["source_cohort"], cohort),
        ("ratings source path", existing["source_db_path"], str(source_path)),
        ("ratings source SHA-256", existing["source_db_sha256"], source_hash),
        (
            "predictions path",
            existing["predictions_csv_path"],
            str(predictions_path),
        ),
        (
            "predictions SHA-256",
            existing["predictions_csv_sha256"],
            predictions_hash,
        ),
    )
    for label, frozen, supplied in expected:
        if frozen != supplied:
            mismatches.append(f"{label} differs (frozen {frozen!r}, current {supplied!r})")
    return mismatches


def _create_campaign(
    state_path: Path,
    campaign: str,
    cohort: str,
    source_path: Path,
    source_hash: str,
    predictions_path: Path,
    predictions_hash: str,
    cases: list[dict[str, Any]],
    before_provenance_commit: Callable[[], None] | None = None,
    after_provenance_verification: Callable[[], None] | None = None,
) -> None:
    connection = _open_state(state_path)
    try:
        existing_tables = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if existing_tables and "metadata" not in existing_tables:
            raise AdjudicationError(
                f"existing state database is not a compatible adjudication database: {state_path}"
            )
        if not existing_tables:
            connection.executescript(_STATE_SCHEMA)
            connection.execute(
                "INSERT INTO metadata (key, value) VALUES ('schema_version', ?)",
                (STATE_SCHEMA_VERSION,),
            )
            connection.commit()

        schema_row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        if schema_row is None or schema_row["value"] != STATE_SCHEMA_VERSION:
            raise AdjudicationError("state database schema version is unsupported")

        counts = {lane: 0 for lane in LANES}
        for case in cases:
            counts[case["lane"]] += 1

        connection.execute("BEGIN IMMEDIATE")
        if connection.execute(
            "SELECT 1 FROM campaigns WHERE name = ?", (campaign,)
        ).fetchone() is not None:
            connection.rollback()
            raise AdjudicationError(
                f"campaign {campaign!r} appeared concurrently; restart to verify it"
            )
        connection.execute(
            """
            INSERT INTO campaigns (
                name, created_at, source_cohort, source_db_path,
                source_db_sha256, predictions_csv_path,
                predictions_csv_sha256, selection_version, candidate_count,
                model_overestimate_count, model_underestimate_count,
                signal_disagreement_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                campaign,
                _utc_now(),
                cohort,
                str(source_path),
                source_hash,
                str(predictions_path),
                predictions_hash,
                SELECTION_VERSION,
                len(cases),
                counts["model-overestimate"],
                counts["model-underestimate"],
                counts["signal-disagreement"],
            ),
        )
        case_columns = (
            "campaign",
            "ad_id",
            "queue_order",
            "lane",
            "selection_explanation",
            "frozen_at",
            "source_queue_order",
            "source_score",
            "keyword_score",
            "score_bucket",
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
            "original_rating",
            *ALL_SIGNAL_COLUMNS,
        )
        placeholders = ", ".join("?" for _column in case_columns)
        connection.executemany(
            f"INSERT INTO cases ({', '.join(case_columns)}) VALUES ({placeholders})",
            (
                tuple(
                    campaign
                    if column == "campaign"
                    else case["rating"]
                    if column == "original_rating"
                    else case[column]
                    for column in case_columns
                )
                for case in cases
            ),
        )

        if before_provenance_commit is not None:
            before_provenance_commit()
        changed_inputs: list[str] = []
        if _sha256(source_path) != source_hash:
            changed_inputs.append("ratings source")
        if _sha256(predictions_path) != predictions_hash:
            changed_inputs.append("frozen predictions")
        if changed_inputs:
            connection.rollback()
            raise AdjudicationError(
                " and ".join(changed_inputs)
                + " changed before the campaign transaction could commit; "
                + "no campaign was created"
            )
        if after_provenance_verification is not None:
            after_provenance_verification()
        connection.commit()
    except sqlite3.IntegrityError as exc:
        connection.rollback()
        raise AdjudicationError(f"cannot freeze campaign {campaign!r}: {exc}") from exc
    except sqlite3.Error as exc:
        connection.rollback()
        raise AdjudicationError(f"cannot create campaign {campaign!r}: {exc}") from exc
    finally:
        connection.close()


def _campaign_summary(state_path: Path, campaign: str) -> dict[str, Any]:
    connection = _open_state(state_path)
    try:
        campaign_row = connection.execute(
            "SELECT * FROM campaigns WHERE name = ?", (campaign,)
        ).fetchone()
        if campaign_row is None:
            raise AdjudicationError(f"campaign {campaign!r} does not exist")
        decision_row = connection.execute(
            """
            SELECT
                count(*) AS decided,
                coalesce(sum(decision_state = 'confirmed'), 0) AS confirmed,
                coalesce(sum(decision_state = 'revised'), 0) AS revised
            FROM decisions
            WHERE campaign = ?
            """,
            (campaign,),
        ).fetchone()
        summary = dict(campaign_row)
        summary.update(
            decided=int(decision_row["decided"]),
            confirmed=int(decision_row["confirmed"]),
            revised=int(decision_row["revised"]),
        )
        summary["remaining"] = summary["candidate_count"] - summary["decided"]
        return summary
    finally:
        connection.close()


def _progress(connection: sqlite3.Connection, campaign: str) -> dict[str, int]:
    row = connection.execute(
        """
        SELECT count(*) AS total, count(d.ad_id) AS decided
        FROM cases AS c
        LEFT JOIN decisions AS d
          ON d.campaign = c.campaign AND d.ad_id = c.ad_id
        WHERE c.campaign = ?
        """,
        (campaign,),
    ).fetchone()
    total = int(row["total"])
    decided = int(row["decided"])
    return {"total": total, "decided": decided, "remaining": total - decided}


def _location(candidate: dict[str, Any]) -> str:
    parts: list[str] = []
    for raw_value in (candidate["municipality"], candidate["region"]):
        value = str(raw_value).strip()
        if value and all(value.casefold() != part.casefold() for part in parts):
            parts.append(value)
    return ", ".join(parts) or "Not specified"


def _safe_link(url: str) -> str | None:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return None
    return url


def _next_case(
    state_path: Path, campaign: str
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    connection = _open_state(state_path)
    try:
        row = connection.execute(
            """
            SELECT c.*, coalesce(events.defer_count, 0) AS defer_count
            FROM cases AS c
            LEFT JOIN decisions AS d
              ON d.campaign = c.campaign AND d.ad_id = c.ad_id
            LEFT JOIN (
                SELECT campaign, ad_id,
                       sum(event_type = 'deferred') AS defer_count
                FROM case_events
                GROUP BY campaign, ad_id
            ) AS events
              ON events.campaign = c.campaign AND events.ad_id = c.ad_id
            WHERE c.campaign = ? AND d.ad_id IS NULL
            ORDER BY coalesce(events.defer_count, 0), c.queue_order
            LIMIT 1
            """,
            (campaign,),
        ).fetchone()
        progress = _progress(connection, campaign)
        if row is None:
            return None, progress

        candidate = dict(row)
        connection.execute(
            """
            INSERT INTO case_events (campaign, ad_id, event_type, occurred_at)
            VALUES (?, ?, 'presented', ?)
            """,
            (campaign, candidate["ad_id"], _utc_now()),
        )
        connection.commit()
    except sqlite3.Error as exc:
        connection.rollback()
        raise AdjudicationError(f"cannot load next adjudication case: {exc}") from exc
    finally:
        connection.close()

    candidate["location"] = _location(candidate)
    candidate["link_url"] = _safe_link(candidate["webpage_url"])
    return candidate, progress


def _completion_counts(state_path: Path, campaign: str) -> list[dict[str, Any]]:
    connection = _open_state(state_path)
    try:
        rows = connection.execute(
            """
            SELECT
                c.lane,
                count(*) AS total,
                count(d.ad_id) AS decided,
                coalesce(sum(d.decision_state = 'confirmed'), 0) AS confirmed,
                coalesce(sum(d.decision_state = 'revised'), 0) AS revised
            FROM cases AS c
            LEFT JOIN decisions AS d
              ON d.campaign = c.campaign AND d.ad_id = c.ad_id
            WHERE c.campaign = ?
            GROUP BY c.lane
            """,
            (campaign,),
        ).fetchall()
    finally:
        connection.close()
    by_lane = {row["lane"]: dict(row) for row in rows}
    output: list[dict[str, Any]] = []
    for lane in LANES:
        counts = by_lane.get(
            lane,
            {"lane": lane, "total": 0, "decided": 0, "confirmed": 0, "revised": 0},
        )
        counts["remaining"] = int(counts["total"]) - int(counts["decided"])
        output.append(counts)
    return output


def _validate_decision_form() -> tuple[str, int, list[str], str]:
    ad_id = request.form.get("ad_id", "")
    if not ad_id:
        abort(400, "ad_id is required")
    rating_text = request.form.get("rating")
    if rating_text not in {"1", "2", "3", "4", "5"}:
        abort(400, "adjudicated rating must be an integer from 1 to 5")
    supplied_reasons = request.form.getlist("reasons")
    if not supplied_reasons:
        abort(400, "select at least one reason tag")
    if len(supplied_reasons) != len(set(supplied_reasons)):
        abort(400, "reason tags must not be duplicated")
    unknown = sorted(set(supplied_reasons) - REASON_CODES)
    if unknown:
        abort(400, "unknown reason tag: " + ", ".join(unknown))
    reasons = sorted(supplied_reasons, key=REASON_ORDER.__getitem__)
    return ad_id, int(rating_text), reasons, request.form.get("note", "")


def _record_decision(
    state_path: Path,
    campaign: str,
    ad_id: str,
    rating: int,
    reasons: list[str],
    note: str,
) -> None:
    connection = _open_state(state_path)
    error: tuple[int, str] | None = None
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT c.original_rating, d.ad_id AS decided_ad_id
            FROM cases AS c
            LEFT JOIN decisions AS d
              ON d.campaign = c.campaign AND d.ad_id = c.ad_id
            WHERE c.campaign = ? AND c.ad_id = ?
            """,
            (campaign, ad_id),
        ).fetchone()
        if row is None:
            error = (404, "candidate is not in this frozen campaign")
        elif row["decided_ad_id"] is not None:
            error = (409, "candidate is already adjudicated")

        if error is not None:
            connection.rollback()
        else:
            decided_at = _utc_now()
            decision_state = (
                "confirmed" if rating == row["original_rating"] else "revised"
            )
            connection.execute(
                """
                INSERT INTO decisions (
                    campaign, ad_id, adjudicated_rating,
                    decision_state, note, decided_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (campaign, ad_id, rating, decision_state, note, decided_at),
            )
            connection.executemany(
                """
                INSERT INTO decision_reasons (
                    campaign, ad_id, reason_order, reason_code
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    (campaign, ad_id, reason_order, reason_code)
                    for reason_order, reason_code in enumerate(reasons)
                ),
            )
            connection.commit()
    except sqlite3.IntegrityError:
        connection.rollback()
        error = (409, "candidate state changed; reload and try again")
    except sqlite3.Error as exc:
        connection.rollback()
        raise AdjudicationError(f"cannot store adjudication decision: {exc}") from exc
    finally:
        connection.close()
    if error is not None:
        abort(error[0], error[1])


def _record_defer(state_path: Path, campaign: str, ad_id: str) -> None:
    if not ad_id:
        abort(400, "ad_id is required")
    connection = _open_state(state_path)
    error: tuple[int, str] | None = None
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT c.ad_id, d.ad_id AS decided_ad_id
            FROM cases AS c
            LEFT JOIN decisions AS d
              ON d.campaign = c.campaign AND d.ad_id = c.ad_id
            WHERE c.campaign = ? AND c.ad_id = ?
            """,
            (campaign, ad_id),
        ).fetchone()
        if row is None:
            error = (404, "candidate is not in this frozen campaign")
        elif row["decided_ad_id"] is not None:
            error = (409, "candidate is already adjudicated")

        if error is not None:
            connection.rollback()
        else:
            connection.execute(
                """
                INSERT INTO case_events (campaign, ad_id, event_type, occurred_at)
                VALUES (?, ?, 'deferred', ?)
                """,
                (campaign, ad_id, _utc_now()),
            )
            connection.commit()
    except sqlite3.Error as exc:
        connection.rollback()
        raise AdjudicationError(f"cannot defer adjudication case: {exc}") from exc
    finally:
        connection.close()
    if error is not None:
        abort(error[0], error[1])


_EXPORT_COLUMNS = (
    "campaign",
    "campaign_created_at",
    "source_cohort",
    "selection_version",
    "source_db_path",
    "source_db_sha256",
    "predictions_csv_path",
    "predictions_csv_sha256",
    "campaign_candidate_count",
    "ad_id",
    "queue_order",
    "lane",
    "selection_explanation",
    "frozen_at",
    "source_queue_order",
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
    "keyword_score",
    "source_score",
    "score_bucket",
    "oof_title_tfidf",
    "oof_full_tfidf",
    "oof_embedding_mrl256",
    "oof_embedding_plus_scores",
    "oof_centroid",
    "oof_ensemble",
    "oof_title_embedding",
    "oof_four_way_ensemble",
    "model_spread",
    "absolute_ensemble_error",
    "original_rating",
    "adjudicated_rating",
    "decision_state",
    "rating_confirmed",
    "rating_revised",
    "reason_codes",
    "note",
    "decided_at",
    "case_status",
    "presentation_count",
    "defer_count",
    "first_presented_at",
    "last_presented_at",
    "first_deferred_at",
    "last_deferred_at",
)


def _export_csv(state_path: Path, campaign: str) -> str:
    connection = _open_state(state_path)
    try:
        rows = connection.execute(
            """
            SELECT
                c.campaign,
                h.created_at AS campaign_created_at,
                h.source_cohort,
                h.selection_version,
                h.source_db_path,
                h.source_db_sha256,
                h.predictions_csv_path,
                h.predictions_csv_sha256,
                h.candidate_count AS campaign_candidate_count,
                c.ad_id,
                c.queue_order,
                c.lane,
                c.selection_explanation,
                c.frozen_at,
                c.source_queue_order,
                c.headline,
                c.employer,
                c.employment_type,
                c.publication_date,
                c.application_deadline,
                c.webpage_url,
                c.description_text,
                c.municipality,
                c.region,
                c.occupation_group,
                c.keyword_score,
                c.source_score,
                c.score_bucket,
                c.oof_title_tfidf,
                c.oof_full_tfidf,
                c.oof_embedding_mrl256,
                c.oof_embedding_plus_scores,
                c.oof_centroid,
                c.oof_ensemble,
                c.oof_title_embedding,
                c.oof_four_way_ensemble,
                c.model_spread,
                c.absolute_ensemble_error,
                c.original_rating,
                d.adjudicated_rating,
                d.decision_state,
                CASE WHEN d.decision_state = 'confirmed' THEN 1
                     WHEN d.decision_state IS NULL THEN NULL ELSE 0 END
                    AS rating_confirmed,
                CASE WHEN d.decision_state = 'revised' THEN 1
                     WHEN d.decision_state IS NULL THEN NULL ELSE 0 END
                    AS rating_revised,
                (
                    SELECT group_concat(reason_code, ';')
                    FROM (
                        SELECT reason_code
                        FROM decision_reasons AS ordered_reasons
                        WHERE ordered_reasons.campaign = c.campaign
                          AND ordered_reasons.ad_id = c.ad_id
                        ORDER BY reason_order
                    )
                ) AS reason_codes,
                d.note,
                d.decided_at,
                CASE
                    WHEN d.ad_id IS NOT NULL THEN 'decided'
                    WHEN coalesce(events.defer_count, 0) > 0 THEN 'deferred'
                    ELSE 'undecided'
                END AS case_status,
                coalesce(events.presentation_count, 0) AS presentation_count,
                coalesce(events.defer_count, 0) AS defer_count,
                events.first_presented_at,
                events.last_presented_at,
                events.first_deferred_at,
                events.last_deferred_at
            FROM cases AS c
            JOIN campaigns AS h ON h.name = c.campaign
            LEFT JOIN decisions AS d
              ON d.campaign = c.campaign AND d.ad_id = c.ad_id
            LEFT JOIN (
                SELECT
                    campaign,
                    ad_id,
                    sum(event_type = 'presented') AS presentation_count,
                    sum(event_type = 'deferred') AS defer_count,
                    min(CASE WHEN event_type = 'presented' THEN occurred_at END)
                        AS first_presented_at,
                    max(CASE WHEN event_type = 'presented' THEN occurred_at END)
                        AS last_presented_at,
                    min(CASE WHEN event_type = 'deferred' THEN occurred_at END)
                        AS first_deferred_at,
                    max(CASE WHEN event_type = 'deferred' THEN occurred_at END)
                        AS last_deferred_at
                FROM case_events
                GROUP BY campaign, ad_id
            ) AS events
              ON events.campaign = c.campaign AND events.ad_id = c.ad_id
            WHERE c.campaign = ?
            ORDER BY c.queue_order
            """,
            (campaign,),
        ).fetchall()
    except sqlite3.Error as exc:
        raise AdjudicationError(f"cannot export adjudication campaign: {exc}") from exc
    finally:
        connection.close()

    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(_EXPORT_COLUMNS)
    writer.writerows(tuple(row[column] for column in _EXPORT_COLUMNS) for row in rows)
    return output.getvalue()


def create_app(
    *,
    ratings_db: str | os.PathLike[str] = DEFAULT_RATINGS_DB,
    predictions_csv: str | os.PathLike[str] = DEFAULT_PREDICTIONS_CSV,
    state_db: str | os.PathLike[str] = DEFAULT_STATE_DB,
    campaign: str = DEFAULT_CAMPAIGN,
    cohort: str = DEFAULT_COHORT,
    _before_provenance_commit: Callable[[], None] | None = None,
    _after_provenance_verification: Callable[[], None] | None = None,
) -> Flask:
    """Build the app, freezing the named campaign only when it is first created."""
    campaign = campaign.strip()
    cohort = cohort.strip()
    if not campaign:
        raise AdjudicationError("campaign name must not be empty")
    if not cohort:
        raise AdjudicationError("cohort name must not be empty")

    source_path = _resolve_input_path(
        ratings_db, description="ratings source database"
    )
    predictions_path = _resolve_input_path(
        predictions_csv, description="frozen predictions CSV"
    )
    state_path = _resolve_state_path(state_db)
    if _same_file(source_path, state_path):
        raise AdjudicationError(
            "state database must be separate from the ratings source database"
        )
    if _same_file(predictions_path, state_path):
        raise AdjudicationError(
            "state database must be separate from the frozen predictions CSV"
        )

    source_hash = _sha256(source_path)
    predictions_hash = _sha256(predictions_path)
    existing = _read_existing_campaign(state_path, campaign)
    if existing is not None:
        mismatches = _provenance_mismatches(
            existing,
            source_path=source_path,
            source_hash=source_hash,
            predictions_path=predictions_path,
            predictions_hash=predictions_hash,
            cohort=cohort,
        )
        if mismatches:
            raise AdjudicationError(
                f"campaign {campaign!r} is an immutable frozen snapshot; "
                + "; ".join(mismatches)
                + ". Use a new --campaign name for different evidence."
            )
    else:
        cases = _build_selection(source_path, predictions_path, cohort)
        if _sha256(source_path) != source_hash:
            raise AdjudicationError(
                "ratings source changed while the campaign was being built; retry with stable input"
            )
        if _sha256(predictions_path) != predictions_hash:
            raise AdjudicationError(
                "frozen predictions changed while the campaign was being built; retry with stable input"
            )
        _create_campaign(
            state_path,
            campaign,
            cohort,
            source_path,
            source_hash,
            predictions_path,
            predictions_hash,
            cases,
            before_provenance_commit=_before_provenance_commit,
            after_provenance_verification=_after_provenance_verification,
        )

    summary = _campaign_summary(state_path, campaign)
    app = Flask(__name__)
    app.config.update(
        ADJUDICATION_STATE_DB=str(state_path),
        ADJUDICATION_CAMPAIGN=campaign,
        ADJUDICATION_SUMMARY=summary,
    )

    @app.get("/")
    def index() -> str:
        candidate, progress = _next_case(state_path, campaign)
        completion = (
            _completion_counts(state_path, campaign) if candidate is None else None
        )
        return render_template(
            "adjudicate.html",
            candidate=candidate,
            progress=progress,
            completion=completion,
            campaign=campaign,
            rating_labels=RATING_LABELS,
            reason_groups=REASON_GROUPS,
        )

    @app.post("/decide")
    def decide() -> Response:
        ad_id, rating, reasons, note = _validate_decision_form()
        _record_decision(state_path, campaign, ad_id, rating, reasons, note)
        return redirect(url_for("index"), code=303)

    @app.post("/defer")
    def defer() -> Response:
        _record_defer(state_path, campaign, request.form.get("ad_id", ""))
        return redirect(url_for("index"), code=303)

    @app.get("/export.csv")
    def export_csv() -> Response:
        safe_campaign = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in campaign
        ).strip("-") or "campaign"
        return Response(
            _export_csv(state_path, campaign),
            content_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{safe_campaign}-adjudication.csv"'
                )
            },
        )

    return app


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local, non-blind adjudication app for a frozen campaign of "
            "human/model disagreements. Inputs are verified read-only snapshots; "
            "adjudications are stored in a separate SQLite database."
        )
    )
    parser.add_argument(
        "--ratings-db",
        default=str(DEFAULT_RATINGS_DB),
        metavar="PATH",
        help=(
            "completed oracle ratings SQLite database, opened read-only "
            f"(default: {DEFAULT_RATINGS_DB})"
        ),
    )
    parser.add_argument(
        "--predictions-csv",
        default=str(DEFAULT_PREDICTIONS_CSV),
        metavar="PATH",
        help=f"frozen out-of-fold predictions CSV (default: {DEFAULT_PREDICTIONS_CSV})",
    )
    parser.add_argument(
        "--state-db",
        default=str(DEFAULT_STATE_DB),
        metavar="PATH",
        help=f"separate adjudication SQLite database (default: {DEFAULT_STATE_DB})",
    )
    parser.add_argument(
        "--campaign",
        default=DEFAULT_CAMPAIGN,
        help=f"immutable campaign name (default: {DEFAULT_CAMPAIGN})",
    )
    parser.add_argument(
        "--cohort",
        default=DEFAULT_COHORT,
        help=f"completed source cohort (default: {DEFAULT_COHORT})",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=DEFAULT_PORT,
        help=f"localhost TCP port (default: {DEFAULT_PORT})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        app = create_app(
            ratings_db=args.ratings_db,
            predictions_csv=args.predictions_csv,
            state_db=args.state_db,
            campaign=args.campaign,
            cohort=args.cohort,
        )
    except AdjudicationError as exc:
        parser.error(str(exc))

    summary = app.config["ADJUDICATION_SUMMARY"]
    shortfall = ""
    if summary["candidate_count"] < LANE_LIMIT * len(LANES):
        shortfall = (
            f"; only {summary['candidate_count']} cases were available under the "
            f"{LANE_LIMIT}-per-lane caps"
        )
    print(
        f"Campaign {summary['name']!r}: {summary['candidate_count']} frozen cases "
        f"(model-overestimate={summary['model_overestimate_count']}, "
        f"model-underestimate={summary['model_underestimate_count']}, "
        f"signal-disagreement={summary['signal_disagreement_count']}), "
        f"{summary['decided']} decisions{shortfall}.",
        flush=True,
    )
    print(f"Open http://127.0.0.1:{args.port}/", flush=True)
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
