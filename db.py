"""
SQLite storage for job ads and run history.
"""

import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "jobsearcher.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    conn = _connect()
    conn.executescript("""
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
        );

        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT NOT NULL,
            total_fetched INTEGER,
            total_scored INTEGER,
            embedding_available BOOLEAN,
            status TEXT
        );
    """)
    conn.close()


def upsert_ads(scored_ads: list[dict[str, Any]]) -> int:
    """Insert or update ads. Returns count of newly inserted ads."""
    conn = _connect()
    today = date.today().isoformat()
    new_count = 0

    for ad in scored_ads:
        existing = conn.execute("SELECT id FROM ads WHERE id = ?", (ad["id"],)).fetchone()
        if existing:
            conn.execute("""
                UPDATE ads SET
                    kw_raw = ?, kw_score = ?, similarity = ?, final_score = ?,
                    last_seen = ?, query_source = ?
                WHERE id = ?
            """, (
                ad["kw_raw"], ad["kw_score"], ad.get("similarity"),
                ad["final_score"], today, ad.get("query_source", ""),
                ad["id"],
            ))
        else:
            conn.execute("""
                INSERT INTO ads (
                    id, headline, employer, employment_type, publication_date,
                    application_deadline, webpage_url, description_text,
                    municipality, region, occupation_group,
                    kw_raw, kw_score, similarity, final_score,
                    first_seen, last_seen, query_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ad["id"], ad["headline"], ad.get("employer", ""),
                ad.get("employment_type", ""), ad.get("publication_date", ""),
                ad.get("application_deadline", ""), ad.get("webpage_url", ""),
                ad.get("description_text", ""), ad.get("municipality", ""),
                ad.get("region", ""), ad.get("occupation_group", ""),
                ad["kw_raw"], ad["kw_score"], ad.get("similarity"),
                ad["final_score"], today, today, ad.get("query_source", ""),
            ))
            new_count += 1

    conn.commit()
    conn.close()
    return new_count


def record_run(total_fetched: int, total_scored: int, embedding_available: bool, status: str) -> None:
    conn = _connect()
    conn.execute("""
        INSERT INTO runs (run_date, total_fetched, total_scored, embedding_available, status)
        VALUES (?, ?, ?, ?, ?)
    """, (date.today().isoformat(), total_fetched, total_scored, embedding_available, status))
    conn.commit()
    conn.close()
