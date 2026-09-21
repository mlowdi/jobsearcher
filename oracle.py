"""Private localhost UI for blind first-pass ratings of stored job ads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import os
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, Response, abort, redirect, render_template, request, url_for

APP_DIR = Path(__file__).resolve().parent
DEFAULT_RATINGS_DB = APP_DIR / "oracle-ratings.db"
DEFAULT_COHORT = "pre-filtered-first-pass"
DEFAULT_POOL_SIZE = 500
DEFAULT_MIN_KEYWORD_SCORE = 3

EXPECTED_AD_COLUMNS = frozenset(
    {
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
    }
)

RATING_LABELS = (
    (1, "No, not for me"),
    (2, "Probably not"),
    (3, "Maybe"),
    (4, "Interesting"),
    (5, "Yes, I want to look into this"),
)

_RATINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS cohorts (
    name TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    source_db_path TEXT NOT NULL,
    requested_pool_size INTEGER NOT NULL CHECK (requested_pool_size > 0),
    eligible_count INTEGER NOT NULL CHECK (eligible_count >= 0),
    deduplicated_count INTEGER NOT NULL CHECK (deduplicated_count >= 0),
    candidate_count INTEGER NOT NULL CHECK (candidate_count >= 0),
    selection_version TEXT NOT NULL,
    min_keyword_score INTEGER CHECK (
        min_keyword_score IS NULL OR (
            typeof(min_keyword_score) = 'integer'
            AND min_keyword_score BETWEEN 1 AND 10
        )
    )
);

CREATE TABLE IF NOT EXISTS candidates (
    cohort TEXT NOT NULL,
    ad_id TEXT NOT NULL,
    queue_order INTEGER NOT NULL CHECK (queue_order > 0),
    source_score REAL,
    keyword_score INTEGER CHECK (
        keyword_score IS NULL OR (
            typeof(keyword_score) = 'integer' AND keyword_score BETWEEN 1 AND 10
        )
    ),
    score_bucket INTEGER NOT NULL CHECK (score_bucket BETWEEN 1 AND 5),
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

CREATE TABLE IF NOT EXISTS ratings (
    cohort TEXT NOT NULL,
    ad_id TEXT NOT NULL,
    rating INTEGER NOT NULL
        CHECK (typeof(rating) = 'integer' AND rating BETWEEN 1 AND 5),
    rated_at TEXT NOT NULL,
    PRIMARY KEY (cohort, ad_id),
    FOREIGN KEY (cohort, ad_id)
        REFERENCES candidates(cohort, ad_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS presentations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cohort TEXT NOT NULL,
    ad_id TEXT NOT NULL,
    presented_at TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('rated', 'skip')),
    FOREIGN KEY (cohort, ad_id)
        REFERENCES candidates(cohort, ad_id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS presentations_candidate_idx
    ON presentations(cohort, ad_id, action);
"""


class OracleError(RuntimeError):
    """A startup error that should be shown directly to the local operator."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _resolve_source_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise OracleError(f"source ads database is unavailable: {candidate}") from exc
    if not resolved.is_file():
        raise OracleError(f"source ads database is not a file: {resolved}")
    return resolved


def _resolve_ratings_path(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _paths_refer_to_same_file(source_path: Path, ratings_path: Path) -> bool:
    if source_path == ratings_path:
        return True
    if not ratings_path.exists():
        return False
    try:
        return os.path.samefile(source_path, ratings_path)
    except OSError:
        return False


def _open_source(source_path: Path) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(
            f"{source_path.as_uri()}?mode=ro", uri=True, timeout=5
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
    except sqlite3.Error as exc:
        raise OracleError(
            f"cannot open source ads database read-only: {source_path}: {exc}"
        ) from exc

    try:
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(ads)")
        }
    except sqlite3.Error as exc:
        connection.close()
        raise OracleError(f"cannot inspect source ads table: {exc}") from exc

    missing = sorted(EXPECTED_AD_COLUMNS - columns)
    if missing:
        connection.close()
        raise OracleError(
            "source ads table is missing expected columns: " + ", ".join(missing)
        )
    return connection


def _open_ratings(ratings_path: Path) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(ratings_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection
    except sqlite3.Error as exc:
        raise OracleError(f"cannot open ratings database {ratings_path}: {exc}") from exc


def _migrate_ratings_schema(connection: sqlite3.Connection) -> None:
    cohort_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(cohorts)")
    }
    if "min_keyword_score" not in cohort_columns:
        connection.execute(
            """
            ALTER TABLE cohorts ADD COLUMN min_keyword_score INTEGER CHECK (
                min_keyword_score IS NULL OR (
                    typeof(min_keyword_score) = 'integer'
                    AND min_keyword_score BETWEEN 1 AND 10
                )
            )
            """
        )

    candidate_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(candidates)")
    }
    if "keyword_score" not in candidate_columns:
        connection.execute(
            """
            ALTER TABLE candidates ADD COLUMN keyword_score INTEGER CHECK (
                keyword_score IS NULL OR (
                    typeof(keyword_score) = 'integer'
                    AND keyword_score BETWEEN 1 AND 10
                )
            )
            """
        )


def _initialise_ratings(ratings_path: Path) -> None:
    if not ratings_path.parent.is_dir():
        raise OracleError(
            f"ratings database directory does not exist: {ratings_path.parent}"
        )
    connection = _open_ratings(ratings_path)
    try:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(_RATINGS_SCHEMA)
        connection.execute("BEGIN IMMEDIATE")
        _migrate_ratings_schema(connection)
        connection.commit()
    except sqlite3.Error as exc:
        connection.rollback()
        raise OracleError(
            f"cannot initialise ratings database {ratings_path}: {exc}"
        ) from exc
    finally:
        connection.close()


def _stable_key(cohort: str, purpose: str, ad_id: str) -> bytes:
    value = f"{cohort}\0{purpose}\0{ad_id}".encode("utf-8")
    return hashlib.sha256(value).digest()


def _eligible_ads(
    source: sqlite3.Connection, as_of_date: date, min_keyword_score: int
) -> list[dict[str, Any]]:
    try:
        rows = source.execute(
            """
            SELECT
                id, headline, employer, employment_type, publication_date,
                application_deadline, webpage_url, description_text,
                municipality, region, occupation_group, kw_score, final_score
            FROM ads
            WHERE description_text IS NOT NULL
              AND description_text <> ''
              AND application_deadline IS NOT NULL
              AND length(application_deadline) >= 10
            """
        ).fetchall()
    except sqlite3.Error as exc:
        raise OracleError(f"cannot read source ads table: {exc}") from exc

    eligible: list[dict[str, Any]] = []
    for row in rows:
        keyword_score = row["kw_score"]
        if type(keyword_score) is not int or not min_keyword_score <= keyword_score <= 10:
            continue

        description = _text(row["description_text"])
        if not description.strip():
            continue

        deadline = _text(row["application_deadline"])
        try:
            deadline_date = date.fromisoformat(deadline[:10])
        except ValueError:
            continue
        if deadline_date < as_of_date:
            continue

        ad_id = _text(row["id"])
        if not ad_id:
            raise OracleError("source ads table contains an eligible row with an empty id")

        eligible.append(
            {
                "ad_id": ad_id,
                "headline": _text(row["headline"]),
                "employer": _text(row["employer"]),
                "employment_type": _text(row["employment_type"]),
                "publication_date": _text(row["publication_date"]),
                "application_deadline": deadline,
                "webpage_url": _text(row["webpage_url"]),
                "description_text": description,
                "municipality": _text(row["municipality"]),
                "region": _text(row["region"]),
                "occupation_group": _text(row["occupation_group"]),
                "keyword_score": keyword_score,
                "source_score": _score(row["final_score"]),
            }
        )
    return eligible


def _deduplicate_ads(eligible: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def highest_first(ad: dict[str, Any]) -> tuple[int, float, str]:
        score = ad["source_score"]
        if score is None:
            return (1, 0.0, ad["ad_id"])
        return (0, -score, ad["ad_id"])

    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for ad in sorted(eligible, key=highest_first):
        identity = (
            ad["headline"].strip().casefold(),
            ad["employer"].strip().casefold(),
        )
        if identity in seen:
            continue
        seen.add(identity)
        deduplicated.append(ad)
    return deduplicated


def _assign_buckets(ads: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    buckets: dict[int, list[dict[str, Any]]] = {number: [] for number in range(1, 6)}
    if not ads:
        return buckets

    def lowest_first(ad: dict[str, Any]) -> tuple[int, float, str]:
        score = ad["source_score"]
        if score is None:
            return (0, 0.0, ad["ad_id"])
        return (1, score, ad["ad_id"])

    ordered = sorted(ads, key=lowest_first)
    total = len(ordered)
    for index, ad in enumerate(ordered):
        bucket = min(5, (index * 5) // total + 1)
        frozen = dict(ad)
        frozen["score_bucket"] = bucket
        buckets[bucket].append(frozen)
    return buckets


def _select_pool(
    ads: list[dict[str, Any]], cohort: str, pool_size: int
) -> list[dict[str, Any]]:
    buckets = _assign_buckets(ads)
    for bucket in buckets.values():
        bucket.sort(
            key=lambda ad: (_stable_key(cohort, "sample", ad["ad_id"]), ad["ad_id"])
        )

    target = min(pool_size, len(ads))
    selected: list[dict[str, Any]] = []
    positions = {number: 0 for number in buckets}
    while len(selected) < target:
        added = False
        for number in range(1, 6):
            position = positions[number]
            if position >= len(buckets[number]):
                continue
            selected.append(buckets[number][position])
            positions[number] += 1
            added = True
            if len(selected) == target:
                break
        if not added:
            break

    selected.sort(
        key=lambda ad: (_stable_key(cohort, "queue", ad["ad_id"]), ad["ad_id"])
    )
    for queue_order, ad in enumerate(selected, start=1):
        ad["queue_order"] = queue_order
    return selected


def _ensure_cohort(
    source: sqlite3.Connection,
    source_path: Path,
    ratings_path: Path,
    cohort: str,
    pool_size: int,
    as_of_date: date,
    min_keyword_score: int,
) -> None:
    connection = _open_ratings(ratings_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT min_keyword_score FROM cohorts WHERE name = ?", (cohort,)
        ).fetchone()
        if existing is not None:
            frozen_threshold = existing["min_keyword_score"]
            if (
                frozen_threshold is not None
                and frozen_threshold != min_keyword_score
            ):
                connection.rollback()
                raise OracleError(
                    f"cohort {cohort!r} was created with --min-keyword-score "
                    f"{frozen_threshold}, not {min_keyword_score}; choose a new "
                    "--cohort name"
                )
            connection.commit()
            return

        eligible = _eligible_ads(source, as_of_date, min_keyword_score)
        deduplicated = _deduplicate_ads(eligible)
        selected = _select_pool(deduplicated, cohort, pool_size)
        connection.execute(
            """
            INSERT INTO cohorts (
                name, created_at, as_of_date, source_db_path,
                requested_pool_size, eligible_count, deduplicated_count,
                candidate_count, selection_version, min_keyword_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cohort,
                _utc_now(),
                as_of_date.isoformat(),
                str(source_path),
                pool_size,
                len(eligible),
                len(deduplicated),
                len(selected),
                "sha256-keyword-prefilter-stratified-quintiles-v2",
                min_keyword_score,
            ),
        )
        connection.executemany(
            """
            INSERT INTO candidates (
                cohort, ad_id, queue_order, source_score, keyword_score,
                score_bucket, headline, employer, employment_type,
                publication_date, application_deadline, webpage_url,
                description_text, municipality, region, occupation_group
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    cohort,
                    ad["ad_id"],
                    ad["queue_order"],
                    ad["source_score"],
                    ad["keyword_score"],
                    ad["score_bucket"],
                    ad["headline"],
                    ad["employer"],
                    ad["employment_type"],
                    ad["publication_date"],
                    ad["application_deadline"],
                    ad["webpage_url"],
                    ad["description_text"],
                    ad["municipality"],
                    ad["region"],
                    ad["occupation_group"],
                )
                for ad in selected
            ],
        )
        connection.commit()
    except sqlite3.Error as exc:
        connection.rollback()
        raise OracleError(f"cannot create cohort {cohort!r}: {exc}") from exc
    finally:
        connection.close()


def _next_candidate(
    ratings_path: Path, cohort: str
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    connection = _open_ratings(ratings_path)
    try:
        row = connection.execute(
            """
            SELECT
                c.ad_id, c.headline, c.employer, c.employment_type,
                c.application_deadline, c.webpage_url, c.description_text,
                c.municipality, c.region
            FROM candidates AS c
            LEFT JOIN ratings AS r
              ON r.cohort = c.cohort AND r.ad_id = c.ad_id
            LEFT JOIN (
                SELECT cohort, ad_id, count(*) AS skip_count
                FROM presentations
                WHERE action = 'skip'
                GROUP BY cohort, ad_id
            ) AS p
              ON p.cohort = c.cohort AND p.ad_id = c.ad_id
            WHERE c.cohort = ? AND r.ad_id IS NULL
            ORDER BY coalesce(p.skip_count, 0), c.queue_order
            LIMIT 1
            """,
            (cohort,),
        ).fetchone()
        counts = connection.execute(
            """
            SELECT
                count(*) AS total,
                count(r.ad_id) AS rated
            FROM candidates AS c
            LEFT JOIN ratings AS r
              ON r.cohort = c.cohort AND r.ad_id = c.ad_id
            WHERE c.cohort = ?
            """,
            (cohort,),
        ).fetchone()
    finally:
        connection.close()

    total = int(counts["total"])
    rated = int(counts["rated"])
    progress = {"total": total, "rated": rated, "remaining": total - rated}
    if row is None:
        return None, progress

    candidate = dict(row)
    location_parts: list[str] = []
    for value in (candidate["municipality"], candidate["region"]):
        cleaned = value.strip()
        if cleaned and all(cleaned.casefold() != part.casefold() for part in location_parts):
            location_parts.append(cleaned)
    candidate["location"] = ", ".join(location_parts) or "Not specified"
    return candidate, progress


def _record_action(
    ratings_path: Path,
    cohort: str,
    ad_id: str,
    rating_text: str | None,
    action: str | None,
) -> None:
    if action == "skip":
        if rating_text is not None:
            abort(400, "choose either a rating or skip")
        rating: int | None = None
    elif action is None and rating_text in {"1", "2", "3", "4", "5"}:
        rating = int(rating_text)
    else:
        abort(400, "rating must be an integer from 1 to 5, or use skip")

    connection = _open_ratings(ratings_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        candidate = connection.execute(
            "SELECT 1 FROM candidates WHERE cohort = ? AND ad_id = ?",
            (cohort, ad_id),
        ).fetchone()
        if candidate is None:
            abort(404, "candidate is not in this cohort")

        already_rated = connection.execute(
            "SELECT 1 FROM ratings WHERE cohort = ? AND ad_id = ?",
            (cohort, ad_id),
        ).fetchone()
        if already_rated is not None:
            abort(409, "candidate is already rated")

        timestamp = _utc_now()
        if rating is None:
            connection.execute(
                """
                INSERT INTO presentations (cohort, ad_id, presented_at, action)
                VALUES (?, ?, ?, 'skip')
                """,
                (cohort, ad_id, timestamp),
            )
        else:
            connection.execute(
                """
                INSERT INTO ratings (cohort, ad_id, rating, rated_at)
                VALUES (?, ?, ?, ?)
                """,
                (cohort, ad_id, rating, timestamp),
            )
            connection.execute(
                """
                INSERT INTO presentations (cohort, ad_id, presented_at, action)
                VALUES (?, ?, ?, 'rated')
                """,
                (cohort, ad_id, timestamp),
            )
        connection.commit()
    except sqlite3.IntegrityError:
        connection.rollback()
        abort(409, "candidate state changed; reload and try again")
    finally:
        connection.close()


def _export_csv(ratings_path: Path, cohort: str) -> str:
    connection = _open_ratings(ratings_path)
    try:
        rows = connection.execute(
            """
            SELECT
                c.cohort, c.ad_id, c.queue_order, c.source_score,
                c.keyword_score, c.score_bucket, c.headline, c.employer,
                c.employment_type,
                c.publication_date, c.application_deadline, c.webpage_url,
                c.description_text, c.municipality, c.region,
                c.occupation_group, r.rating, r.rated_at,
                h.created_at AS cohort_created_at, h.as_of_date,
                h.source_db_path, h.requested_pool_size, h.eligible_count,
                h.deduplicated_count, h.candidate_count, h.selection_version,
                h.min_keyword_score,
                coalesce(ps.presentation_count, 0) AS presentation_count,
                coalesce(ps.skip_count, 0) AS skip_count,
                ps.first_presented_at, lp.action AS last_presentation_action,
                lp.presented_at AS last_presented_at
            FROM candidates AS c
            JOIN cohorts AS h ON h.name = c.cohort
            LEFT JOIN ratings AS r
              ON r.cohort = c.cohort AND r.ad_id = c.ad_id
            LEFT JOIN (
                SELECT
                    cohort, ad_id, count(*) AS presentation_count,
                    sum(action = 'skip') AS skip_count,
                    min(presented_at) AS first_presented_at,
                    max(id) AS last_presentation_id
                FROM presentations
                GROUP BY cohort, ad_id
            ) AS ps
              ON ps.cohort = c.cohort AND ps.ad_id = c.ad_id
            LEFT JOIN presentations AS lp ON lp.id = ps.last_presentation_id
            WHERE c.cohort = ?
            ORDER BY c.queue_order
            """,
            (cohort,),
        ).fetchall()
    finally:
        connection.close()

    columns = (
        "cohort",
        "ad_id",
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
        "rating",
        "rated_at",
        "cohort_created_at",
        "as_of_date",
        "source_db_path",
        "requested_pool_size",
        "eligible_count",
        "deduplicated_count",
        "candidate_count",
        "selection_version",
        "min_keyword_score",
        "presentation_count",
        "skip_count",
        "first_presented_at",
        "last_presentation_action",
        "last_presented_at",
    )
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(columns)
    writer.writerows(tuple(row[column] for column in columns) for row in rows)
    return output.getvalue()


def create_app(
    *,
    ads_db: str | os.PathLike[str],
    ratings_db: str | os.PathLike[str] = DEFAULT_RATINGS_DB,
    cohort: str = DEFAULT_COHORT,
    pool_size: int = DEFAULT_POOL_SIZE,
    min_keyword_score: int = DEFAULT_MIN_KEYWORD_SCORE,
    today: date | None = None,
) -> Flask:
    """Build the Flask app and create the named cohort if it does not exist."""
    cohort = cohort.strip()
    if not cohort:
        raise OracleError("cohort name must not be empty")
    if pool_size <= 0:
        raise OracleError("pool size must be greater than zero")
    if type(min_keyword_score) is not int or not 1 <= min_keyword_score <= 10:
        raise OracleError("minimum keyword score must be an integer from 1 through 10")

    source_path = _resolve_source_path(ads_db)
    ratings_path = _resolve_ratings_path(ratings_db)
    if _paths_refer_to_same_file(source_path, ratings_path):
        raise OracleError("ratings database must be separate from the source ads database")

    source = _open_source(source_path)
    try:
        _initialise_ratings(ratings_path)
        _ensure_cohort(
            source,
            source_path,
            ratings_path,
            cohort,
            pool_size,
            today or date.today(),
            min_keyword_score,
        )
    finally:
        source.close()

    app = Flask(__name__)
    app.config.update(
        ORACLE_RATINGS_DB=str(ratings_path),
        ORACLE_COHORT=cohort,
    )

    @app.get("/")
    def index() -> str:
        candidate, progress = _next_candidate(ratings_path, cohort)
        return render_template(
            "oracle.html",
            candidate=candidate,
            progress=progress,
            cohort=cohort,
            rating_labels=RATING_LABELS,
        )

    @app.post("/rate")
    def rate() -> Response:
        ad_id = request.form.get("ad_id")
        if ad_id is None or not ad_id:
            abort(400, "ad_id is required")
        _record_action(
            ratings_path,
            cohort,
            ad_id,
            request.form.get("rating"),
            request.form.get("action"),
        )
        return redirect(url_for("index"), code=303)

    @app.get("/export.csv")
    def export_csv() -> Response:
        filename = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in cohort
        ).strip("-") or "cohort"
        return Response(
            _export_csv(ratings_path, cohort),
            content_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}-ratings.csv"'
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


def _keyword_score_threshold(value: str) -> int:
    try:
        threshold = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "minimum keyword score must be an integer from 1 through 10"
        ) from exc
    if not 1 <= threshold <= 10:
        raise argparse.ArgumentTypeError(
            "minimum keyword score must be an integer from 1 through 10"
        )
    return threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a private localhost web app for blind 1-5 ratings of stored job ads. "
            "The source SQLite database is opened read-only; labels are stored separately."
        )
    )
    parser.add_argument(
        "--ads-db",
        required=True,
        metavar="PATH",
        help="source SQLite database containing the ads table (required, read-only)",
    )
    parser.add_argument(
        "--ratings-db",
        default=str(DEFAULT_RATINGS_DB),
        metavar="PATH",
        help=f"private ratings SQLite database (default: {DEFAULT_RATINGS_DB})",
    )
    parser.add_argument(
        "--cohort",
        default=DEFAULT_COHORT,
        help=f"persistent cohort name (default: {DEFAULT_COHORT})",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=DEFAULT_POOL_SIZE,
        metavar="N",
        help=f"maximum candidates when first creating a cohort (default: {DEFAULT_POOL_SIZE})",
    )
    parser.add_argument(
        "--min-keyword-score",
        type=_keyword_score_threshold,
        default=DEFAULT_MIN_KEYWORD_SCORE,
        metavar="N",
        help=(
            "minimum stored normalized keyword score for a new cohort, "
            f"from 1 through 10 (default: {DEFAULT_MIN_KEYWORD_SCORE})"
        ),
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=5000,
        help="localhost TCP port (default: 5000)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        app = create_app(
            ads_db=args.ads_db,
            ratings_db=args.ratings_db,
            cohort=args.cohort,
            pool_size=args.pool_size,
            min_keyword_score=args.min_keyword_score,
        )
    except OracleError as exc:
        parser.error(str(exc))

    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
