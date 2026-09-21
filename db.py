"""Durable SQLite storage for the ranked job-search workspace."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import unicodedata
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

DB_PATH = Path(__file__).parent / "jobsearcher.db"
TRIAGE_STATES = (
    "unread",
    "saved",
    "dismissed",
    "applied",
    "interview",
    "offer",
    "closed",
)
LANES = ("high", "possible", "wildcard", "low")

_MIGRATIONS: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    (
        1,
        "preserve legacy ads and runs",
        (
            """
            CREATE TABLE IF NOT EXISTS ads (
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
                first_seen TEXT DEFAULT (date('now')),
                last_seen TEXT DEFAULT (date('now')),
                query_source TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TEXT NOT NULL,
                total_fetched INTEGER,
                total_scored INTEGER,
                embedding_available BOOLEAN,
                status TEXT
            )
            """,
        ),
    ),
    (
        2,
        "continuous ranked workspace",
        (
            """
            CREATE TABLE IF NOT EXISTS triage_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ad_id TEXT NOT NULL REFERENCES ads(id),
                state TEXT NOT NULL CHECK (
                    state IN ('unread','saved','dismissed','applied','interview','offer','closed')
                ),
                reason_code TEXT,
                note TEXT,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS preference_scores (
                ad_id TEXT PRIMARY KEY REFERENCES ads(id),
                model_version TEXT NOT NULL,
                scoring_mode TEXT NOT NULL CHECK (scoring_mode IN ('fallback','four-signal')),
                title_tfidf REAL NOT NULL,
                full_embedding REAL,
                title_embedding REAL,
                source_score REAL NOT NULL,
                ensemble_score REAL NOT NULL,
                component_spread REAL NOT NULL,
                title_hash TEXT NOT NULL,
                description_hash TEXT NOT NULL,
                scored_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dimensions INTEGER NOT NULL CHECK (dimensions > 0),
                vector BLOB NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (text_hash, model_name)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS duplicate_groups (
                ad_id TEXT PRIMARY KEY REFERENCES ads(id),
                canonical_ad_id TEXT NOT NULL REFERENCES ads(id),
                normalized_key TEXT NOT NULL,
                method TEXT NOT NULL CHECK (method = 'exact'),
                similarity REAL NOT NULL CHECK (similarity = 1.0),
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS refresh_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_type TEXT NOT NULL CHECK (run_type IN ('refresh','rescore')),
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL CHECK (status IN ('running','success','partial','failure')),
                total_fetched INTEGER NOT NULL DEFAULT 0,
                total_source_scored INTEGER NOT NULL DEFAULT 0,
                new_ads INTEGER NOT NULL DEFAULT 0,
                updated_ads INTEGER NOT NULL DEFAULT 0,
                preference_scored INTEGER NOT NULL DEFAULT 0,
                fallback_count INTEGER NOT NULL DEFAULT 0,
                four_signal_count INTEGER NOT NULL DEFAULT 0,
                active_count INTEGER NOT NULL DEFAULT 0,
                canonical_count INTEGER NOT NULL DEFAULT 0,
                source_embedding_available INTEGER NOT NULL DEFAULT 0,
                endpoint_error TEXT,
                error TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lane_snapshots (
                ad_id TEXT PRIMARY KEY REFERENCES ads(id),
                lane TEXT NOT NULL CHECK (lane IN ('high','possible','wildcard','low')),
                lane_rank INTEGER NOT NULL CHECK (lane_rank >= 1),
                snapshot_at TEXT NOT NULL,
                scoring_run_id INTEGER REFERENCES refresh_runs(id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS ads_active_deadline_idx ON ads(application_deadline, id)",
            "CREATE INDEX IF NOT EXISTS triage_events_latest_idx ON triage_events(ad_id, id DESC)",
            "CREATE INDEX IF NOT EXISTS triage_events_state_idx ON triage_events(state, created_at DESC)",
            "CREATE UNIQUE INDEX IF NOT EXISTS triage_events_import_once_idx ON triage_events(ad_id) WHERE source LIKE 'oracle-import:%'",
            "CREATE INDEX IF NOT EXISTS embedding_cache_lookup_idx ON embedding_cache(model_name, text_hash)",
            "CREATE INDEX IF NOT EXISTS duplicate_groups_canonical_idx ON duplicate_groups(canonical_ad_id, ad_id)",
            "CREATE INDEX IF NOT EXISTS duplicate_groups_key_idx ON duplicate_groups(normalized_key, ad_id)",
            "CREATE UNIQUE INDEX IF NOT EXISTS lane_snapshots_order_idx ON lane_snapshots(lane, lane_rank)",
            "CREATE INDEX IF NOT EXISTS lane_snapshots_run_idx ON lane_snapshots(scoring_run_id)",
            "CREATE INDEX IF NOT EXISTS refresh_runs_latest_idx ON refresh_runs(started_at DESC, id DESC)",
        ),
    ),
    (
        3,
        "preference embedding provenance",
        (
            "ALTER TABLE preference_scores ADD COLUMN embedding_model_name TEXT",
            "ALTER TABLE preference_scores ADD COLUMN embedding_provenance TEXT",
        ),
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(
    db_path: str | Path = DB_PATH,
    *,
    writable: bool = True,
    timeout: float = 5.0,
) -> sqlite3.Connection:
    """Open SQLite with foreign keys, busy handling, and WAL for writers."""
    path = Path(db_path).expanduser().resolve()
    if writable:
        connection = sqlite3.connect(path, timeout=timeout)
    else:
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=timeout)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {max(1, int(timeout * 1000))}")
    if writable:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
    else:
        connection.execute("PRAGMA query_only = ON")
    return connection


def _connect(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Backward-compatible writable connection helper."""
    return connect(db_path)


@contextmanager
def staged_database(db_path: str | Path) -> Iterable[Path]:
    """Yield a private SQLite snapshot and remove it after preparation."""
    source_path = Path(db_path).expanduser().resolve()
    descriptor, name = tempfile.mkstemp(
        prefix=".jobsearcher-refresh-", suffix=".db", dir=source_path.parent
    )
    os.close(descriptor)
    staged_path = Path(name)
    source = connect(source_path, writable=False)
    target = sqlite3.connect(staged_path)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    try:
        yield staged_path
    finally:
        for candidate in (
            staged_path,
            Path(f"{staged_path}-wal"),
            Path(f"{staged_path}-shm"),
        ):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass


def integrate_staged_refresh(
    db_path: str | Path,
    staged_path: str | Path,
    *,
    run_id: int,
    status: str,
    counts: dict[str, int],
    source_embedding_available: bool = False,
    endpoint_error: str | None = None,
) -> None:
    """Publish prepared ads, scores, groups, lanes, and success metadata atomically."""
    if status not in {"success", "partial"}:
        raise ValueError("a staged refresh may only publish a successful terminal status")
    values = {key: 0 for key in (
        "total_fetched", "total_source_scored", "new_ads", "updated_ads",
        "preference_scored", "fallback_count", "four_signal_count",
        "active_count", "canonical_count",
    )}
    values.update(counts)
    connection = connect(db_path)
    attached = False
    try:
        connection.execute("ATTACH DATABASE ? AS staged", (str(Path(staged_path).resolve()),))
        attached = True
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                """
                INSERT INTO ads(
                    id,headline,employer,employment_type,publication_date,
                    application_deadline,webpage_url,description_text,municipality,
                    region,occupation_group,kw_raw,kw_score,similarity,final_score,
                    first_seen,last_seen,query_source
                )
                SELECT id,headline,employer,employment_type,publication_date,
                       application_deadline,webpage_url,description_text,municipality,
                       region,occupation_group,kw_raw,kw_score,similarity,final_score,
                       first_seen,last_seen,query_source
                FROM staged.ads WHERE 1
                ON CONFLICT(id) DO UPDATE SET
                    headline=excluded.headline,
                    employer=excluded.employer,
                    employment_type=excluded.employment_type,
                    publication_date=excluded.publication_date,
                    application_deadline=excluded.application_deadline,
                    webpage_url=excluded.webpage_url,
                    description_text=excluded.description_text,
                    municipality=excluded.municipality,
                    region=excluded.region,
                    occupation_group=excluded.occupation_group,
                    kw_raw=excluded.kw_raw,
                    kw_score=excluded.kw_score,
                    similarity=excluded.similarity,
                    final_score=excluded.final_score,
                    last_seen=excluded.last_seen,
                    query_source=CASE
                        WHEN trim(excluded.query_source)<>'' THEN excluded.query_source
                        ELSE ads.query_source
                    END
                """
            )
            collision = connection.execute(
                """
                SELECT 1 FROM embedding_cache current
                JOIN staged.embedding_cache prepared
                  ON prepared.text_hash=current.text_hash
                 AND prepared.model_name=current.model_name
                WHERE prepared.dimensions<>current.dimensions
                   OR prepared.vector<>current.vector
                LIMIT 1
                """
            ).fetchone()
            if collision is not None:
                raise ValueError("staged embedding cache conflicts with current cache")
            connection.execute(
                """
                INSERT OR IGNORE INTO embedding_cache(
                    text_hash,model_name,dimensions,vector,created_at
                )
                SELECT text_hash,model_name,dimensions,vector,created_at
                FROM staged.embedding_cache
                """
            )
            connection.execute(
                """
                INSERT INTO preference_scores(
                    ad_id,model_version,scoring_mode,title_tfidf,full_embedding,
                    title_embedding,source_score,ensemble_score,component_spread,
                    title_hash,description_hash,scored_at,embedding_model_name,
                    embedding_provenance
                )
                SELECT ad_id,model_version,scoring_mode,title_tfidf,full_embedding,
                       title_embedding,source_score,ensemble_score,component_spread,
                       title_hash,description_hash,scored_at,embedding_model_name,
                       embedding_provenance
                FROM staged.preference_scores WHERE 1
                ON CONFLICT(ad_id) DO UPDATE SET
                    model_version=excluded.model_version,
                    scoring_mode=excluded.scoring_mode,
                    title_tfidf=excluded.title_tfidf,
                    full_embedding=excluded.full_embedding,
                    title_embedding=excluded.title_embedding,
                    source_score=excluded.source_score,
                    ensemble_score=excluded.ensemble_score,
                    component_spread=excluded.component_spread,
                    title_hash=excluded.title_hash,
                    description_hash=excluded.description_hash,
                    scored_at=excluded.scored_at,
                    embedding_model_name=excluded.embedding_model_name,
                    embedding_provenance=excluded.embedding_provenance
                """
            )
            connection.execute("DELETE FROM lane_snapshots")
            connection.execute("DELETE FROM duplicate_groups")
            connection.execute(
                """
                INSERT INTO duplicate_groups(
                    ad_id,canonical_ad_id,normalized_key,method,similarity,updated_at
                )
                SELECT ad_id,canonical_ad_id,normalized_key,method,similarity,updated_at
                FROM staged.duplicate_groups
                """
            )
            connection.execute(
                """
                INSERT INTO lane_snapshots(
                    ad_id,lane,lane_rank,snapshot_at,scoring_run_id
                )
                SELECT ad_id,lane,lane_rank,snapshot_at,scoring_run_id
                FROM staged.lane_snapshots
                """
            )
            cursor = connection.execute(
                """
                UPDATE refresh_runs SET completed_at=?,status=?,total_fetched=?,
                    total_source_scored=?,new_ads=?,updated_ads=?,preference_scored=?,
                    fallback_count=?,four_signal_count=?,active_count=?,canonical_count=?,
                    source_embedding_available=?,endpoint_error=?,error=NULL
                WHERE id=? AND status='running'
                """,
                (utc_now(), status, values["total_fetched"],
                 values["total_source_scored"], values["new_ads"],
                 values["updated_ads"], values["preference_scored"],
                 values["fallback_count"], values["four_signal_count"],
                 values["active_count"], values["canonical_count"],
                 bool(source_embedding_available), endpoint_error, run_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"refresh run {run_id} is missing or already complete")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    finally:
        if attached:
            connection.execute("DETACH DATABASE staged")
        connection.close()


def migrate(db_path: str | Path = DB_PATH) -> list[int]:
    """Apply every missing migration, returning versions applied now."""
    connection = connect(db_path)
    applied: list[int] = []
    try:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
                """
            )
        present = {
            int(row["version"])
            for row in connection.execute("SELECT version FROM schema_migrations")
        }
        for version, name, statements in _MIGRATIONS:
            if version in present:
                continue
            with connection:
                for statement in statements:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations(version, name, applied_at) VALUES (?, ?, ?)",
                    (version, name, utc_now()),
                )
            applied.append(version)
    finally:
        connection.close()
    return applied


def init_db(db_path: str | Path = DB_PATH) -> None:
    migrate(db_path)


def _source_values(ad: dict[str, Any], today: str) -> tuple[Any, ...]:
    return (
        str(ad["id"]), str(ad.get("headline") or ""),
        str(ad.get("employer") or ""), str(ad.get("employment_type") or ""),
        str(ad.get("publication_date") or ""), str(ad.get("application_deadline") or ""),
        str(ad.get("webpage_url") or ""), str(ad.get("description_text") or ""),
        str(ad.get("municipality") or ""), str(ad.get("region") or ""),
        str(ad.get("occupation_group") or ""), int(ad.get("kw_raw") or 0),
        int(ad.get("kw_score") or 0),
        None if ad.get("similarity") is None else float(ad["similarity"]),
        float(ad.get("final_score") or 0.0), today, today,
        str(ad.get("query_source") or ""),
    )


def upsert_ads(
    scored_ads: list[dict[str, Any]],
    db_path: str | Path = DB_PATH,
    *,
    seen_on: str | None = None,
) -> int:
    """Fully refresh mutable ad fields while preserving ``first_seen``."""
    migrate(db_path)
    connection = connect(db_path)
    today = seen_on or date.today().isoformat()
    new_count = 0
    sql = """
        INSERT INTO ads (
            id, headline, employer, employment_type, publication_date,
            application_deadline, webpage_url, description_text,
            municipality, region, occupation_group,
            kw_raw, kw_score, similarity, final_score,
            first_seen, last_seen, query_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            headline = excluded.headline,
            employer = excluded.employer,
            employment_type = excluded.employment_type,
            publication_date = excluded.publication_date,
            application_deadline = excluded.application_deadline,
            webpage_url = excluded.webpage_url,
            description_text = excluded.description_text,
            municipality = excluded.municipality,
            region = excluded.region,
            occupation_group = excluded.occupation_group,
            kw_raw = excluded.kw_raw,
            kw_score = excluded.kw_score,
            similarity = excluded.similarity,
            final_score = excluded.final_score,
            last_seen = excluded.last_seen,
            query_source = CASE
                WHEN trim(excluded.query_source) <> '' THEN excluded.query_source
                ELSE ads.query_source
            END
    """
    try:
        with connection:
            for ad in scored_ads:
                exists = connection.execute(
                    "SELECT 1 FROM ads WHERE id = ?", (str(ad["id"]),)
                ).fetchone()
                connection.execute(sql, _source_values(ad, today))
                if exists is None:
                    new_count += 1
    finally:
        connection.close()
    return new_count


def record_run(
    total_fetched: int,
    total_scored: int,
    embedding_available: bool,
    status: str,
    db_path: str | Path = DB_PATH,
) -> None:
    """Keep the historical ``runs`` interface available to old tooling."""
    migrate(db_path)
    connection = connect(db_path)
    try:
        with connection:
            connection.execute(
                """INSERT INTO runs
                   (run_date,total_fetched,total_scored,embedding_available,status)
                   VALUES (?, ?, ?, ?, ?)""",
                (date.today().isoformat(), total_fetched, total_scored,
                 bool(embedding_available), status),
            )
    finally:
        connection.close()


def start_refresh_run(
    db_path: str | Path = DB_PATH, *, run_type: str = "refresh"
) -> int:
    if run_type not in {"refresh", "rescore"}:
        raise ValueError(f"invalid run type: {run_type}")
    migrate(db_path)
    connection = connect(db_path)
    try:
        with connection:
            now = utc_now()
            connection.execute(
                """
                UPDATE refresh_runs
                SET completed_at=?, status='failure',
                    error=COALESCE(error, 'previous run terminated before completion')
                WHERE run_type=? AND status='running' AND completed_at IS NULL
                """,
                (now, run_type),
            )
            cursor = connection.execute(
                "INSERT INTO refresh_runs(run_type,started_at,status) VALUES (?,?,'running')",
                (run_type, now),
            )
        return int(cursor.lastrowid)
    finally:
        connection.close()


def finish_refresh_run(
    run_id: int,
    db_path: str | Path = DB_PATH,
    *,
    status: str,
    counts: dict[str, int] | None = None,
    source_embedding_available: bool = False,
    endpoint_error: str | None = None,
    error: str | None = None,
) -> None:
    if status not in {"success", "partial", "failure"}:
        raise ValueError(f"invalid terminal refresh status: {status}")
    values = {key: 0 for key in (
        "total_fetched", "total_source_scored", "new_ads", "updated_ads",
        "preference_scored", "fallback_count", "four_signal_count",
        "active_count", "canonical_count",
    )}
    values.update(counts or {})
    connection = connect(db_path)
    try:
        with connection:
            cursor = connection.execute(
                """
                UPDATE refresh_runs SET completed_at=?, status=?, total_fetched=?,
                    total_source_scored=?, new_ads=?, updated_ads=?, preference_scored=?,
                    fallback_count=?, four_signal_count=?, active_count=?, canonical_count=?,
                    source_embedding_available=?, endpoint_error=?, error=?
                WHERE id=? AND status='running'
                """,
                (utc_now(), status, values["total_fetched"], values["total_source_scored"],
                 values["new_ads"], values["updated_ads"], values["preference_scored"],
                 values["fallback_count"], values["four_signal_count"],
                 values["active_count"], values["canonical_count"],
                 bool(source_embedding_available), endpoint_error, error, run_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"refresh run {run_id} is missing or already complete")
    finally:
        connection.close()


def append_triage_event(
    ad_id: str,
    state: str,
    *,
    reason_code: str | None = None,
    note: str | None = None,
    source: str,
    db_path: str | Path = DB_PATH,
    created_at: str | None = None,
) -> int:
    if state not in TRIAGE_STATES:
        raise ValueError(f"invalid triage state: {state}")
    if not source or not source.strip():
        raise ValueError("triage event source must be non-empty")
    connection = connect(db_path)
    try:
        with connection:
            if connection.execute("SELECT 1 FROM ads WHERE id=?", (str(ad_id),)).fetchone() is None:
                raise KeyError(f"unknown ad id: {ad_id}")
            cursor = connection.execute(
                """INSERT INTO triage_events
                   (ad_id,state,reason_code,note,created_at,source)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (str(ad_id), state,
                 reason_code.strip() if reason_code and reason_code.strip() else None,
                 note.strip() if note and note.strip() else None,
                 created_at or utc_now(), source.strip()),
            )
        return int(cursor.lastrowid)
    finally:
        connection.close()


def encode_float32(vector: Sequence[float] | np.ndarray, *, dimensions: int | None = None) -> bytes:
    """Encode a finite one-dimensional vector as exact little-endian float32."""
    array = np.asarray(vector, dtype=np.dtype("<f4"))
    if array.ndim != 1:
        raise ValueError(f"embedding vector must be one-dimensional, got {array.shape}")
    if dimensions is not None and array.size != dimensions:
        raise ValueError(f"embedding dimensions {array.size} != expected {dimensions}")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("embedding vector must be non-empty and finite")
    return array.tobytes(order="C")


def decode_float32(blob: bytes, dimensions: int) -> np.ndarray:
    """Decode and validate the little-endian float32 cache representation."""
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise ValueError("embedding cache vector is not a BLOB")
    if dimensions <= 0 or len(blob) != dimensions * 4:
        raise ValueError(f"embedding BLOB has {len(blob)} bytes; expected {dimensions * 4}")
    vector = np.frombuffer(blob, dtype=np.dtype("<f4")).copy()
    if not np.isfinite(vector).all():
        raise ValueError("embedding cache vector contains non-finite values")
    return vector


def put_cached_vector(
    connection: sqlite3.Connection,
    *, text_hash: str, model_name: str,
    vector: Sequence[float] | np.ndarray,
    dimensions: int, created_at: str | None = None,
) -> bool:
    if len(text_hash) != 64 or any(ch not in "0123456789abcdef" for ch in text_hash):
        raise ValueError(f"invalid SHA-256 text hash: {text_hash!r}")
    if not model_name.strip():
        raise ValueError("embedding model name must be non-empty")
    blob = encode_float32(vector, dimensions=dimensions)
    existing = connection.execute(
        "SELECT dimensions,vector FROM embedding_cache WHERE text_hash=? AND model_name=?",
        (text_hash, model_name),
    ).fetchone()
    if existing is not None:
        old_dimensions = int(existing["dimensions"])
        old = decode_float32(existing["vector"], old_dimensions)
        new = decode_float32(blob, dimensions)
        if old_dimensions != dimensions or not np.array_equal(old, new):
            raise ValueError(f"cache collision for text hash {text_hash} and model {model_name}")
        return False
    connection.execute(
        """INSERT INTO embedding_cache
           (text_hash,model_name,dimensions,vector,created_at) VALUES (?, ?, ?, ?, ?)""",
        (text_hash, model_name, dimensions, blob, created_at or utc_now()),
    )
    return True


def get_cached_vector(
    connection: sqlite3.Connection,
    *, text_hash: str, model_name: str, dimensions: int,
) -> np.ndarray | None:
    row = connection.execute(
        "SELECT dimensions,vector FROM embedding_cache WHERE text_hash=? AND model_name=?",
        (text_hash, model_name),
    ).fetchone()
    if row is None:
        return None
    stored_dimensions = int(row["dimensions"])
    if stored_dimensions != dimensions:
        raise ValueError(f"cached dimensions {stored_dimensions} != model dimensions {dimensions}")
    return decode_float32(row["vector"], dimensions)


def normalize_embedding_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def normalized_text_hash(text: str) -> str:
    return hashlib.sha256(normalize_embedding_text(text).encode("utf-8")).hexdigest()


def normalize_duplicate_part(value: str | None) -> str:
    return " ".join(unicodedata.normalize("NFKC", value or "").casefold().split())


def duplicate_key(headline: str | None, employer: str | None) -> str:
    first = normalize_duplicate_part(headline)
    second = normalize_duplicate_part(employer)
    return f"{len(first)}:{first}{len(second)}:{second}"


def _stable_descending(rows: Iterable[dict[str, Any]], primary: str) -> list[dict[str, Any]]:
    result = sorted(rows, key=lambda row: str(row["id"]))
    result.sort(key=lambda row: str(row.get("publication_date") or ""), reverse=True)
    result.sort(key=lambda row: float(row.get(primary) or 0.0), reverse=True)
    return result


def _build_group_and_lane_rows(
    active_rows: list[dict[str, Any]], *, timestamp: str, run_id: int | None
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in active_rows:
        grouped[duplicate_key(row.get("headline"), row.get("employer"))].append(row)
    duplicate_rows: list[tuple[Any, ...]] = []
    canonical: list[dict[str, Any]] = []
    for key in sorted(grouped):
        chosen = _stable_descending(grouped[key], "ensemble_score")[0]
        canonical.append(chosen)
        for row in sorted(grouped[key], key=lambda item: str(item["id"])):
            duplicate_rows.append((str(row["id"]), str(chosen["id"]), key, "exact", 1.0, timestamp))
    score_order = _stable_descending(canonical, "ensemble_score")
    high, possible, remaining = score_order[:50], score_order[50:200], score_order[200:]
    wildcard = _stable_descending(remaining, "component_spread")[:50]
    wildcard_ids = {str(row["id"]) for row in wildcard}
    low = [row for row in remaining if str(row["id"]) not in wildcard_ids]
    lane_rows: list[tuple[Any, ...]] = []
    for lane, rows in (("high", high), ("possible", possible), ("wildcard", wildcard), ("low", low)):
        for rank, row in enumerate(rows, 1):
            lane_rows.append((str(row["id"]), lane, rank, timestamp, run_id))
    return duplicate_rows, lane_rows


def rebuild_duplicates_and_lanes(
    db_path: str | Path = DB_PATH,
    *, scoring_run_id: int | None,
    active_on: str | None = None,
    snapshot_at: str | None = None,
) -> dict[str, int | str]:
    """Atomically replace exact groups and the complete active lane snapshot."""
    migrate(db_path)
    current_date, timestamp = active_on or date.today().isoformat(), snapshot_at or utc_now()
    connection = connect(db_path)
    try:
        rows = [dict(row) for row in connection.execute(
            """SELECT a.id,a.headline,a.employer,a.publication_date,
                      COALESCE(p.ensemble_score,a.final_score,0.0) ensemble_score,
                      COALESCE(p.component_spread,0.0) component_spread
               FROM ads a LEFT JOIN preference_scores p ON p.ad_id=a.id
               WHERE a.application_deadline>=?""", (current_date,))]
        duplicate_rows, lane_rows = _build_group_and_lane_rows(rows, timestamp=timestamp, run_id=scoring_run_id)
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute("DELETE FROM lane_snapshots")
            connection.execute("DELETE FROM duplicate_groups")
            connection.executemany(
                """INSERT INTO duplicate_groups
                   (ad_id,canonical_ad_id,normalized_key,method,similarity,updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""", duplicate_rows)
            connection.executemany(
                """INSERT INTO lane_snapshots
                   (ad_id,lane,lane_rank,snapshot_at,scoring_run_id)
                   VALUES (?, ?, ?, ?, ?)""", lane_rows)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    finally:
        connection.close()
    return {"active_count": len(duplicate_rows), "canonical_count": len(lane_rows), "snapshot_at": timestamp}
