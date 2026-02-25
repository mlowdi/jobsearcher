"""
Job fetcher — queries jobtechdev API for relevant job postings.

Returns a list of raw ad dicts from the API.
"""

import requests
from typing import Any

QUERY_URL = "https://jobsearch.api.jobtechdev.se/search"

GEOGRAPHY = [
    ("region", "01"),           # Stockholm
    ("municipality", "1980"),   # Västerås
    ("municipality", "1880"),   # Örebro
    ("municipality", "0380"),   # Uppsala
    ("municipality", "0480"),   # Nyköping
    ("municipality", "0484"),   # Eskilstuna
    ("municipality", "0580"),   # Linköping
    ("municipality", "0581"),   # Norrköping
]

OCCUPATION_GROUPS = [
    ("occupation-group", "2516"),   # IT-säkerhetsspecialister
    ("occupation-group", "1335"),   # Driftschefer inom IT
    ("occupation-group", "2421"),   # Organisations- och systemanalytiker
    ("occupation-group", "2422"),   # IT-strateger
]

FREETEXT_QUERIES = [
    "säkerhetssamordnare",
    "IT-säkerhet",
    "cybersäkerhet",
    "CISO",
    "SOC manager",
    "security operations",
    "informationssäkerhet",
]


def _fetch(extra_params: list[tuple[str, str]], headers: dict | None = None) -> list[dict[str, Any]]:
    hdrs = {"accept": "application/json"}
    if headers:
        hdrs.update(headers)
    params = extra_params + [("published-after", "1440"), ("limit", "50")]
    response = requests.get(QUERY_URL, headers=hdrs, params=params)
    response.raise_for_status()
    return response.json()["hits"]


def fetch_jobs() -> list[dict[str, Any]]:
    """Fetch jobs from all query sources, deduplicated by ad ID.

    Returns list of raw API hit dicts, each tagged with 'query_source'.
    """
    hits_by_id: dict[str, dict[str, Any]] = {}

    # 1. Occupation group query (geography-filtered)
    for hit in _fetch(GEOGRAPHY + OCCUPATION_GROUPS):
        hit["query_source"] = "occupation_group"
        hits_by_id[hit["id"]] = hit

    # 2. Freetext queries (geography-filtered, precise matching)
    freetext_headers = {
        "x-feature-freetext-bool-method": "and",
        "x-feature-disable-smart-freetext": "true",
    }
    for query in FREETEXT_QUERIES:
        for hit in _fetch(GEOGRAPHY + [("q", query)], headers=freetext_headers):
            if hit["id"] not in hits_by_id:
                hit["query_source"] = f"freetext:{query}"
            hits_by_id[hit["id"]] = hit

    # 3. Remote jobs (no geography filter) — catches nationwide remote positions
    for query in FREETEXT_QUERIES:
        for hit in _fetch([("q", query), ("remote", "true")], headers=freetext_headers):
            if hit["id"] not in hits_by_id:
                hit["query_source"] = f"remote:{query}"
            hits_by_id[hit["id"]] = hit

    return list(hits_by_id.values())
