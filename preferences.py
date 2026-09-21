"""Production preference-model loading, embedding cache, and honest scoring modes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import joblib
import numpy as np
import requests
from sklearn.preprocessing import normalize

from db import (
    DB_PATH,
    connect,
    get_cached_vector,
    migrate,
    normalized_text_hash,
    put_cached_vector,
    utc_now,
)

ARTIFACT_SCHEMA_VERSION = 1
TITLE_TEXT_CONSTRUCTOR = "raw-headline-v1"
FULL_TEXT_CONSTRUCTOR = "oracle-structured-v1"
CACHE_HASH_NORMALIZATION = "nfkc-collapsed-whitespace-v1"
EMBEDDING_INPUT_NORMALIZATION = "none-v1"
VECTOR_NORMALIZATION = "l2-v1"
SOURCE_SCORE_POLICY = "stored-final-score-clamped-0-1-v1"

_REQUIRED_ARTIFACT_KEYS = {
    "artifact_schema_version",
    "model_version",
    "seed",
    "title_vectorizer",
    "title_classifier",
    "full_embedding_classifier",
    "title_embedding_classifier",
    "preprocessing",
    "dimensions",
    "embedding_model_name",
    "embedding_model_source",
    "embedding_model_sha256",
    "embedding_model_digest_status",
    "cache_model_name",
    "source_hashes",
    "training_counts",
    "regularization",
}


def title_embedding_text(ad: Mapping[str, Any]) -> str:
    """Reproduce the title artifact's sole raw-headline input."""
    return str(ad.get("headline") or "")


def full_embedding_text(ad: Mapping[str, Any]) -> str:
    """Reproduce the structured full-text constructor used by training vectors."""
    return "\n".join(
        [
            str(ad.get("headline") or ""),
            f"Employer: {ad.get('employer') or ''}",
            f"Location: {ad.get('municipality') or ''}, {ad.get('region') or ''}",
            f"Occupation group: {ad.get('occupation_group') or ''}",
            f"Employment type: {ad.get('employment_type') or ''}",
            "",
            str(ad.get("description_text") or ""),
        ]
    )


def load_model(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(Path(path))
    if not isinstance(artifact, dict):
        raise ValueError("preference model artifact must be a dictionary")
    missing = sorted(_REQUIRED_ARTIFACT_KEYS - artifact.keys())
    if missing:
        raise ValueError(f"preference model is missing fields: {', '.join(missing)}")
    if artifact["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported preference artifact schema {artifact['artifact_schema_version']}"
        )
    preprocessing = artifact["preprocessing"]
    expected = {
        "title_text_constructor": TITLE_TEXT_CONSTRUCTOR,
        "title_embedding_constructor": TITLE_TEXT_CONSTRUCTOR,
        "full_embedding_constructor": FULL_TEXT_CONSTRUCTOR,
        "cache_hash_normalization": CACHE_HASH_NORMALIZATION,
        "embedding_input_normalization": EMBEDDING_INPUT_NORMALIZATION,
        "vector_normalization": VECTOR_NORMALIZATION,
        "source_score_policy": SOURCE_SCORE_POLICY,
    }
    for key, value in expected.items():
        if preprocessing.get(key) != value:
            raise ValueError(
                f"model preprocessing {key}={preprocessing.get(key)!r}; expected {value!r}"
            )
    dimensions = artifact["dimensions"]
    for key in ("full_source", "full_used", "title_source", "title_used"):
        if not isinstance(dimensions.get(key), int) or dimensions[key] <= 0:
            raise ValueError(f"invalid model dimension: {key}")
    if dimensions["full_used"] > dimensions["full_source"]:
        raise ValueError("full embedding used dimensions exceed source dimensions")
    if dimensions["title_used"] > dimensions["title_source"]:
        raise ValueError("title embedding used dimensions exceed source dimensions")
    if dimensions["full_source"] != dimensions["title_source"]:
        raise ValueError("full and title source embedding dimensions disagree")
    digest = str(artifact["embedding_model_sha256"])
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("model embedding SHA-256 must be 64 lowercase hex characters")
    digest_status = artifact["embedding_model_digest_status"]
    if digest_status not in {"verified-file", "declared-manifest"}:
        raise ValueError("model embedding digest status is invalid")
    if not str(artifact["cache_model_name"]).endswith(f":{digest_status}"):
        raise ValueError("model cache provenance does not include digest verification status")
    return artifact


class EmbeddingEndpointError(RuntimeError):
    pass


class EmbeddingInputTooLarge(EmbeddingEndpointError):
    pass


class OpenAIEmbeddingClient:
    """Small bounded OpenAI-compatible embedding client.

    The endpoint model is also the cache provenance. It is deliberately distinct
    from the Q8 training artifact's cache key unless the operator explicitly
    supplies the same provenance string.
    """

    def __init__(
        self,
        *,
        url: str,
        model: str,
        timeout: float = 10.0,
        batch_size: int = 16,
        token_budget: int = 8192,
        cache_model_name: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError("embedding URL must use http or https")
        if not model.strip():
            raise ValueError("embedding endpoint model must be non-empty")
        if not 0.1 <= timeout <= 60.0:
            raise ValueError("embedding timeout must be between 0.1 and 60 seconds")
        if not 1 <= batch_size <= 128:
            raise ValueError("embedding batch size must be between 1 and 128")
        if not 256 <= token_budget <= 131072:
            raise ValueError("embedding token budget must be between 256 and 131072")
        self.url = url
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self.token_budget = token_budget
        self.cache_model_name = cache_model_name or f"endpoint:{model}"
        self.session = session or requests.Session()

    @staticmethod
    def estimated_tokens(text: str) -> int:
        """Conservative tokenizer-free request budgeting for llama-server.

        Snowflake's tokenizer stays comfortably below one token per three UTF-8
        bytes for the Swedish/English job corpus. Counting every byte as a token
        made the initial backfill take more than an hour despite an 8192-token
        server batch.
        """
        return max(1, math.ceil(len(text.encode("utf-8")) / 3))

    def batches(
        self, items: Sequence[tuple[str, str]]
    ) -> Iterable[list[tuple[str, str]]]:
        batch: list[tuple[str, str]] = []
        tokens = 0
        for item in items:
            item_tokens = self.estimated_tokens(item[1])
            if batch and (
                len(batch) >= self.batch_size
                or tokens + item_tokens > self.token_budget
            ):
                yield batch
                batch, tokens = [], 0
            batch.append(item)
            tokens += item_tokens
        if batch:
            yield batch

    def embed(self, texts: Sequence[str], *, dimensions: int) -> list[np.ndarray]:
        if not texts:
            return []
        estimated = sum(self.estimated_tokens(text) for text in texts)
        if estimated > self.token_budget:
            raise EmbeddingInputTooLarge(
                f"embedding request estimate {estimated} exceeds token budget {self.token_budget}"
            )
        try:
            response = self.session.post(
                self.url,
                json={"model": self.model, "input": list(texts)},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload["data"]
            if not isinstance(data, list) or len(data) != len(texts):
                raise ValueError("endpoint returned the wrong embedding count")
            if all(isinstance(item, dict) and "index" in item for item in data):
                data = sorted(data, key=lambda item: int(item["index"]))
            vectors: list[np.ndarray] = []
            for item in data:
                vector = np.asarray(item["embedding"], dtype=np.float32)
                if vector.shape != (dimensions,) or not np.isfinite(vector).all():
                    raise ValueError(
                        f"endpoint embedding shape {vector.shape}; expected ({dimensions},)"
                    )
                vectors.append(vector)
            return vectors
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            raise EmbeddingEndpointError(f"embedding endpoint failed: {exc}") from exc


@dataclass(frozen=True)
class ScoringSummary:
    active_count: int
    scored_count: int
    fallback_count: int
    four_signal_count: int
    endpoint_error: str | None


def _probability(classifier: Any, matrix: Any) -> float:
    probability = float(classifier.predict_proba(matrix)[0, 1])
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"model produced invalid probability {probability!r}")
    return probability


def _embedding_probability(classifier: Any, vector: np.ndarray, used: int) -> float:
    selected = np.asarray(vector[:used], dtype=np.float64).reshape(1, -1)
    return _probability(classifier, normalize(selected))


def _lookup_vector(
    connection: Any,
    *,
    text_hash: str,
    primary_model: str,
    endpoint_model: str | None,
    dimensions: int,
    fetched: dict[str, np.ndarray],
) -> tuple[np.ndarray | None, str | None]:
    vector = get_cached_vector(
        connection,
        text_hash=text_hash,
        model_name=primary_model,
        dimensions=dimensions,
    )
    if vector is not None:
        return vector, primary_model
    if endpoint_model:
        vector = get_cached_vector(
            connection,
            text_hash=text_hash,
            model_name=endpoint_model,
            dimensions=dimensions,
        )
        if vector is not None:
            return vector, endpoint_model
        if text_hash in fetched:
            return fetched[text_hash], endpoint_model
    return None, None


def score_active_ads(
    db_path: str | Path,
    model_path: str | Path,
    *,
    active_on: str | None = None,
    endpoint_client: OpenAIEmbeddingClient | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ScoringSummary:
    """Score active ads from cache and optionally one bounded endpoint attempt.

    Endpoint failure is contained to this call: already cached vectors remain
    usable and every other ad receives the complete two-component fallback.
    """
    migrate(db_path)
    artifact = load_model(model_path)
    dimensions = artifact["dimensions"]
    current_date = active_on or date.today().isoformat()
    connection = connect(db_path)
    fetched: dict[str, np.ndarray] = {}
    endpoint_error: str | None = None
    try:
        ads = [
            dict(row)
            for row in connection.execute(
                """
                SELECT a.*, p.model_version AS previous_model_version,
                       p.scoring_mode AS previous_mode,
                       p.title_hash AS previous_title_hash,
                       p.description_hash AS previous_description_hash,
                       p.source_score AS previous_source_score,
                       p.embedding_provenance AS previous_embedding_provenance
                FROM ads a
                LEFT JOIN preference_scores p ON p.ad_id = a.id
                WHERE a.application_deadline >= ?
                ORDER BY a.id
                """,
                (current_date,),
            )
        ]
        prepared: list[dict[str, Any]] = []
        missing: dict[str, str] = {}
        endpoint_model = endpoint_client.cache_model_name if endpoint_client else None
        primary_model = str(artifact["cache_model_name"])
        for ad in ads:
            title_text = title_embedding_text(ad)
            full_text = full_embedding_text(ad)
            title_hash = normalized_text_hash(title_text)
            full_hash = normalized_text_hash(full_text)
            title_vector, title_provenance = _lookup_vector(
                connection,
                text_hash=title_hash,
                primary_model=primary_model,
                endpoint_model=endpoint_model,
                dimensions=dimensions["title_source"],
                fetched=fetched,
            )
            full_vector, full_provenance = _lookup_vector(
                connection,
                text_hash=full_hash,
                primary_model=primary_model,
                endpoint_model=endpoint_model,
                dimensions=dimensions["full_source"],
                fetched=fetched,
            )
            previous_provenance: dict[str, str] = {}
            if ad.get("previous_mode") == "four-signal":
                try:
                    decoded = json.loads(ad.get("previous_embedding_provenance") or "{}")
                    if isinstance(decoded, dict):
                        previous_provenance = {
                            key: value
                            for key, value in decoded.items()
                            if key in {"full", "title"} and isinstance(value, str)
                        }
                except (TypeError, ValueError):
                    previous_provenance = {}
            if endpoint_client is None:
                previous_title_model = previous_provenance.get("title")
                previous_full_model = previous_provenance.get("full")
                if title_vector is None and previous_title_model:
                    title_vector = get_cached_vector(
                        connection,
                        text_hash=title_hash,
                        model_name=previous_title_model,
                        dimensions=dimensions["title_source"],
                    )
                    if title_vector is not None:
                        title_provenance = previous_title_model
                if full_vector is None and previous_full_model:
                    full_vector = get_cached_vector(
                        connection,
                        text_hash=full_hash,
                        model_name=previous_full_model,
                        dimensions=dimensions["full_source"],
                    )
                    if full_vector is not None:
                        full_provenance = previous_full_model
            if endpoint_client:
                if title_vector is None:
                    missing.setdefault(title_hash, title_text)
                if full_vector is None:
                    missing.setdefault(full_hash, full_text)
            prepared.append(
                {
                    "ad": ad,
                    "title_text": title_text,
                    "full_text": full_text,
                    "title_hash": title_hash,
                    "full_hash": full_hash,
                    "title_vector": title_vector,
                    "full_vector": full_vector,
                    "title_provenance": title_provenance,
                    "full_provenance": full_provenance,
                }
            )

        if endpoint_client and missing:
            items = list(missing.items())
            embedded = 0
            oversized_errors: list[str] = []
            for batch in endpoint_client.batches(items):
                if progress:
                    progress(
                        f"Embedding {embedded + 1}-{embedded + len(batch)} "
                        f"of {len(items)} missing texts"
                    )
                try:
                    vectors = endpoint_client.embed(
                        [text for _, text in batch],
                        dimensions=dimensions["full_source"],
                    )
                except EmbeddingInputTooLarge as exc:
                    oversized_errors.append(str(exc))
                    embedded += len(batch)
                    continue
                except EmbeddingEndpointError as exc:
                    endpoint_error = str(exc)
                    break
                for (text_hash, _), vector in zip(batch, vectors, strict=True):
                    fetched[text_hash] = vector
                embedded += len(batch)
            if oversized_errors and endpoint_error is None:
                endpoint_error = (
                    f"{len(oversized_errors)} embedding input(s) exceeded the "
                    f"configured token budget; first: {oversized_errors[0]}"
                )

        score_rows: list[tuple[Any, ...]] = []
        now = utc_now()
        for item in prepared:
            ad = item["ad"]
            if item["title_vector"] is None:
                item["title_vector"], item["title_provenance"] = _lookup_vector(
                    connection,
                    text_hash=item["title_hash"],
                    primary_model=primary_model,
                    endpoint_model=endpoint_model,
                    dimensions=dimensions["title_source"],
                    fetched=fetched,
                )
            if item["full_vector"] is None:
                item["full_vector"], item["full_provenance"] = _lookup_vector(
                    connection,
                    text_hash=item["full_hash"],
                    primary_model=primary_model,
                    endpoint_model=endpoint_model,
                    dimensions=dimensions["full_source"],
                    fetched=fetched,
                )
            source_score = min(1.0, max(0.0, float(ad.get("final_score") or 0.0)))
            stale = (
                force
                or ad.get("previous_model_version") != artifact["model_version"]
                or ad.get("previous_title_hash") != item["title_hash"]
                or ad.get("previous_description_hash") != item["full_hash"]
                or ad.get("previous_source_score") is None
                or abs(float(ad.get("previous_source_score") or 0.0) - source_score) > 1e-12
                or (
                    ad.get("previous_mode") == "fallback"
                    and item["title_vector"] is not None
                    and item["full_vector"] is not None
                )
                or (
                    ad.get("previous_mode") == "four-signal"
                    and (
                        item["title_vector"] is None
                        or item["full_vector"] is None
                    )
                )
            )
            if not stale:
                continue
            title_matrix = artifact["title_vectorizer"].transform([item["title_text"]])
            title_probability = _probability(artifact["title_classifier"], title_matrix)
            full_probability: float | None = None
            title_embedding_probability: float | None = None
            embedding_name: str | None = None
            provenance: str | None = None
            if item["title_vector"] is not None and item["full_vector"] is not None:
                mode = "four-signal"
                full_probability = _embedding_probability(
                    artifact["full_embedding_classifier"],
                    item["full_vector"],
                    dimensions["full_used"],
                )
                title_embedding_probability = _embedding_probability(
                    artifact["title_embedding_classifier"],
                    item["title_vector"],
                    dimensions["title_used"],
                )
                components = [
                    title_probability,
                    full_probability,
                    title_embedding_probability,
                    source_score,
                ]
                provenance_map = {
                    "full": item["full_provenance"],
                    "title": item["title_provenance"],
                }
                embedding_name = (
                    item["full_provenance"]
                    if item["full_provenance"] == item["title_provenance"]
                    else "mixed"
                )
                provenance = json.dumps(provenance_map, sort_keys=True, separators=(",", ":"))
            else:
                mode = "fallback"
                components = [title_probability, source_score]
            ensemble = sum(components) / len(components)
            spread = max(components) - min(components)
            score_rows.append(
                (
                    str(ad["id"]), artifact["model_version"], mode,
                    title_probability, full_probability, title_embedding_probability,
                    source_score, ensemble, spread, item["title_hash"], item["full_hash"],
                    embedding_name, provenance, now,
                )
            )

        with connection:
            if endpoint_client:
                for text_hash, vector in fetched.items():
                    put_cached_vector(
                        connection,
                        text_hash=text_hash,
                        model_name=endpoint_client.cache_model_name,
                        vector=vector,
                        dimensions=dimensions["full_source"],
                        created_at=now,
                    )
            connection.executemany(
                """
                INSERT INTO preference_scores(
                    ad_id,model_version,scoring_mode,title_tfidf,full_embedding,
                    title_embedding,source_score,ensemble_score,component_spread,
                    title_hash,description_hash,embedding_model_name,
                    embedding_provenance,scored_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    embedding_model_name=excluded.embedding_model_name,
                    embedding_provenance=excluded.embedding_provenance,
                    scored_at=excluded.scored_at
                """,
                score_rows,
            )
        counts = connection.execute(
            """
            SELECT SUM(scoring_mode='fallback') fallback_count,
                   SUM(scoring_mode='four-signal') four_signal_count
            FROM preference_scores p JOIN ads a ON a.id=p.ad_id
            WHERE a.application_deadline>=?
            """,
            (current_date,),
        ).fetchone()
        return ScoringSummary(
            active_count=len(ads),
            scored_count=len(score_rows),
            fallback_count=int(counts["fallback_count"] or 0),
            four_signal_count=int(counts["four_signal_count"] or 0),
            endpoint_error=endpoint_error,
        )
    finally:
        connection.close()
