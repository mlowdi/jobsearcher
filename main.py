#!/usr/bin/env python3
"""One-shot fetch, store, preference-score, duplicate, and lane refresh."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from db import (
    DB_PATH,
    finish_refresh_run,
    integrate_staged_refresh,
    migrate,
    rebuild_duplicates_and_lanes,
    staged_database,
    start_refresh_run,
    upsert_ads,
)
from fetcher import fetch_jobs
from preferences import OpenAIEmbeddingClient, score_active_ads
from scorer import score_ads

ROOT = Path(__file__).parent
DEFAULT_LOCK = ROOT / ".jobsearcher-refresh.lock"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


class RefreshAlreadyRunning(RuntimeError):
    pass


@contextmanager
def advisory_lock(path: str | Path) -> Iterator[None]:
    """Hold a nonblocking project-local process lock for one refresh."""
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RefreshAlreadyRunning(f"refresh lock is held: {lock_path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_refresh(
    args: argparse.Namespace,
    *,
    fetch: Callable[[], list[dict[str, Any]]] = fetch_jobs,
    source_scorer: Callable[..., tuple[list[dict[str, Any]], bool]] = score_ads,
    preference_scorer: Callable[..., Any] = score_active_ads,
) -> tuple[int, dict[str, Any]]:
    try:
        lock_context = advisory_lock(args.lock_file)
        lock_context.__enter__()
    except RefreshAlreadyRunning as exc:
        return 75, {"status": "locked", "error": str(exc)}
    try:
        migrate(args.db)
        run_id = start_refresh_run(args.db, run_type="refresh")
        counts = {
            "total_fetched": 0,
            "total_source_scored": 0,
            "new_ads": 0,
            "updated_ads": 0,
            "preference_scored": 0,
            "fallback_count": 0,
            "four_signal_count": 0,
            "active_count": 0,
            "canonical_count": 0,
        }
        try:
            log("Fetching JobTech ads")
            ads = fetch()
        except Exception as exc:
            error = f"fetch failed: {exc}"
            finish_refresh_run(run_id, args.db, status="failure", counts=counts, error=error)
            return 2, {"run_id": run_id, "status": "failure", "error": error}
        counts["total_fetched"] = len(ads)
        source_embedding_available = False
        try:
            if ads:
                scored, source_embedding_available = source_scorer(
                    ads,
                    top_n=20,
                    final_n=len(ads),
                    use_embeddings=bool(args.source_embeddings),
                )
                counts["total_source_scored"] = len(scored)
            else:
                scored = []
                log("No new hits; rebuilding from persistent active corpus")

            endpoint_client = None
            if args.enable_embedding_endpoint:
                endpoint_client = OpenAIEmbeddingClient(
                    url=args.embedding_url,
                    model=args.embedding_model,
                    timeout=args.embedding_timeout,
                    batch_size=args.embedding_batch_size,
                    token_budget=args.embedding_token_budget,
                    cache_model_name=f"endpoint:{args.embedding_model}",
                )
            with staged_database(args.db) as staged_path:
                if scored:
                    counts["new_ads"] = upsert_ads(
                        scored, staged_path, seen_on=args.active_on
                    )
                    counts["updated_ads"] = len(scored) - counts["new_ads"]
                preference = preference_scorer(
                    staged_path,
                    args.model,
                    active_on=args.active_on,
                    endpoint_client=endpoint_client,
                    force=False,
                    progress=log,
                )
                counts["preference_scored"] = preference.scored_count
                counts["fallback_count"] = preference.fallback_count
                counts["four_signal_count"] = preference.four_signal_count
                snapshot = rebuild_duplicates_and_lanes(
                    staged_path,
                    scoring_run_id=run_id,
                    active_on=args.active_on,
                )
                counts["active_count"] = int(snapshot["active_count"])
                counts["canonical_count"] = int(snapshot["canonical_count"])
                status = (
                    "partial"
                    if preference.fallback_count
                    or preference.endpoint_error
                    or (bool(ads) and not source_embedding_available)
                    else "success"
                )
                integrate_staged_refresh(
                    args.db,
                    staged_path,
                    run_id=run_id,
                    status=status,
                    counts=counts,
                    source_embedding_available=source_embedding_available,
                    endpoint_error=preference.endpoint_error,
                )
            return 0, {"run_id": run_id, "status": status, **counts,
                       "endpoint_error": preference.endpoint_error}
        except Exception as exc:
            error = f"scoring/storage failed: {exc}"
            finish_refresh_run(
                run_id,
                args.db,
                status="failure",
                counts=counts,
                source_embedding_available=source_embedding_available,
                error=error[:4000],
            )
            return 3, {"run_id": run_id, "status": "failure", "error": error}
    finally:
        if "lock_context" in locals():
            lock_context.__exit__(None, None, None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "preference.joblib")
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--active-on", help="ISO date override for deterministic operations")
    parser.add_argument(
        "--source-embeddings",
        action="store_true",
        help="opt in to the legacy resume/source embedding reranker",
    )
    parser.add_argument(
        "--enable-embedding-endpoint",
        action="store_true",
        default=os.getenv("PREFERENCE_EMBED_ENABLED", "").lower() in {"1", "true", "yes"},
    )
    parser.add_argument(
        "--embedding-url",
        default=os.getenv("PREFERENCE_EMBED_URL", "http://127.0.0.1:9090/v1/embeddings"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("PREFERENCE_EMBED_MODEL", "snowflake-arctic-embed-l-v2.0-q4_k_m.gguf"),
    )
    parser.add_argument(
        "--embedding-timeout",
        type=float,
        default=float(os.getenv("PREFERENCE_EMBED_TIMEOUT", "10")),
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=int(os.getenv("PREFERENCE_EMBED_BATCH_SIZE", "16")),
    )
    parser.add_argument(
        "--embedding-token-budget",
        type=int,
        default=int(os.getenv("PREFERENCE_EMBED_TOKEN_BUDGET", "8192")),
        help="conservative estimated-token budget per llama-server request",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code, result = run_refresh(args)
    print(json.dumps(result, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
