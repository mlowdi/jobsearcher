"""
Job scoring — keyword matching + embedding reranking.

Accepts raw API ad dicts and returns scored/ranked results.
"""

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

# Embedding server config
EMBED_URL = "http://toybox:9090/v1/embeddings"
EMBED_MODEL = "snowflake-arctic-embed-l-v2.0-q4_k_m.gguf"

RESUME_FILE = Path(__file__).parent / "resume.md"
STOPWORDS_FILE = Path(__file__).parent / "stopwords-sv.txt"

EMBED_MAX_CHARS = 2000


# ---------------------------------------------------------------------------
# Stopword filtering
# ---------------------------------------------------------------------------

def _load_stopwords() -> set[str]:
    if not STOPWORDS_FILE.exists():
        return set()
    return set(STOPWORDS_FILE.read_text(encoding="utf-8").splitlines())

_STOPWORDS: set[str] = _load_stopwords()


def strip_stopwords(text: str) -> str:
    if not _STOPWORDS:
        return text
    words = re.split(r"(\s+)", text)
    return "".join(w for w in words if w.strip() == "" or w.lower() not in _STOPWORDS)


# ---------------------------------------------------------------------------
# Keyword scoring
# ---------------------------------------------------------------------------

PROFILE_KEYWORDS = {
    "high": [
        # Security roles
        "cybersecurity", "cybersäkerhet", "it-säkerhet", "informationssäkerhet",
        "information security", "säkerhetsspecialist", "säkerhetschef", "ciso",
        "säkerhetsskydd", "skyddssäkerhet", "säkerhetssamordnare",
        "security architect", "säkerhetsarkitekt", "security officer",
        # MSSP/SOC
        "mssp", "soc", "security operations", "soc analyst", "soc manager",
        # Microsoft stack
        "microsoft 365", "m365", "microsoft defender", "defender", "sentinel",
        "entra", "intune", "purview", "microsoft security",
        # Tools from resume
        "kql", "powershell", "fortisiem", "sentinelone", "tenable",
        # Incident & compliance
        "incident", "incidenthantering", "incident management", "incident response",
        "nis2", "dora", "gdpr", "compliance", "efterlevnad",
        # Role match
        "technical delivery", "delivery manager",
    ],
    "medium": [
        "risk", "riskhantering", "riskanalys", "risk management",
        "säkerhetsincident", "security incident",
        "hotbild", "threat", "threat intelligence", "underrättelse",
        "säkerhetsstrateg", "säkerhetsstrategi", "security strategy",
        "verksamhetsskydd",
        "kontinuitet", "continuity", "beredskap", "crisis", "krisberedskap",
        "säkerhetspolicy", "security policy",
        "ledning", "ledarskap", "leadership", "manager", "chef", "lead",
        "zero trust", "xdr", "edr", "siem",
        "säkerhetsrevision", "audit", "penetrationstest", "penetration testing",
        "exchange",
    ],
    "low": [
        "strategi", "strategy", "strategisk",
        "samarbete", "collaboration", "samverkan",
        "projekt", "project", "projektledning",
        "kommunikation", "communication",
        "process", "processutveckling",
        "kvalitet", "quality",
        "offentlig sektor", "myndighet", "statlig",
    ],
}

NEGATIVE_KEYWORDS = [
    "läkemedel", "pharmaceutical", "apotek",
    "fastighet", "lokalstrateg", "lokaler", "hyresvärd",
    "data center", "datacenter", "facility manager",
    "energi", "kraft", "elbolag",
    "bygg", "construction", "tunnelbana", "spårväg",
    "bemanning", "rekrytering", "recruitment", "consultant manager",
    "lss", "omsorg", "hemtjänst", "äldreomsorg",
    "gruv", "mineral", "metall",
    "handläggare", "stiftelserätt",
    "sjuksköterska", "läkare", "tandläkare",
]


def score_job_keywords(text: str) -> int:
    """Raw keyword score (not normalised)."""
    text = text.lower()
    score = 0
    for kw in PROFILE_KEYWORDS["high"]:
        if kw in text:
            score += 3
    for kw in PROFILE_KEYWORDS["medium"]:
        if kw in text:
            score += 2
    for kw in PROFILE_KEYWORDS["low"]:
        if kw in text:
            score += 1

    penalty = 0
    for kw in NEGATIVE_KEYWORDS:
        if kw in text:
            if "säkerhet" in text or "security" in text:
                penalty += 1
            else:
                penalty += 3

    return max(0, score - penalty)


def normalise_score(raw: int) -> int:
    """Map raw keyword score to 1-10."""
    thresholds = [(15, 10), (12, 9), (10, 8), (8, 7), (6, 6),
                  (5, 5), (4, 4), (3, 3), (2, 2)]
    for threshold, rating in thresholds:
        if raw >= threshold:
            return rating
    return 1


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> list[float]:
    data = json.dumps({"model": EMBED_MODEL, "input": text}).encode()
    req = urllib.request.Request(
        EMBED_URL, data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.URLError as e:
        raise RuntimeError(f"Embedding server unavailable at {EMBED_URL}: {e}") from e
    return resp["data"][0]["embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------

def _extract_text(ad: dict[str, Any]) -> str:
    """Build searchable text from a raw API ad dict."""
    parts = [
        ad.get("headline", ""),
        ad.get("employer", {}).get("name", "") if isinstance(ad.get("employer"), dict) else "",
        ad.get("description", {}).get("text", "") if isinstance(ad.get("description"), dict) else "",
    ]
    return "\n".join(parts)


def score_ads(
    ads: list[dict[str, Any]],
    top_n: int = 20,
    final_n: int = 8,
    use_embeddings: bool = True,
) -> tuple[list[dict[str, Any]], bool]:
    """Score and rank ads. Returns (ranked_results, embedding_available).

    Each result dict has keys: id, headline, employer, employment_type,
    publication_date, application_deadline, webpage_url, description_text,
    municipality, region, occupation_group, query_source,
    kw_raw, kw_score, similarity, final_score.
    """
    # Build scored list
    scored = []
    for ad in ads:
        full_text = _extract_text(ad)
        filtered_text = strip_stopwords(full_text.lower())
        kw_raw = score_job_keywords(filtered_text)
        kw_score = normalise_score(kw_raw)

        scored.append({
            "id": ad["id"],
            "headline": ad.get("headline", ""),
            "employer": ad.get("employer", {}).get("name", "") if isinstance(ad.get("employer"), dict) else "",
            "employment_type": ad.get("employment_type", {}).get("label", "") if isinstance(ad.get("employment_type"), dict) else "",
            "publication_date": ad.get("publication_date", ""),
            "application_deadline": ad.get("application_deadline", ""),
            "webpage_url": ad.get("webpage_url", ""),
            "description_text": ad.get("description", {}).get("text", "") if isinstance(ad.get("description"), dict) else "",
            "municipality": ad.get("workplace_address", {}).get("municipality", "") if isinstance(ad.get("workplace_address"), dict) else "",
            "region": ad.get("workplace_address", {}).get("region", "") if isinstance(ad.get("workplace_address"), dict) else "",
            "occupation_group": ad.get("occupation_group", {}).get("label", "") if isinstance(ad.get("occupation_group"), dict) else "",
            "query_source": ad.get("query_source", ""),
            "kw_raw": kw_raw,
            "kw_score": kw_score,
            "similarity": None,
            "final_score": kw_score / 10.0,
            "_full_text": full_text,
        })

    if not scored:
        return [], False

    # Embedding reranking
    embedding_available = False
    if use_embeddings and RESUME_FILE.exists():
        try:
            resume_text = RESUME_FILE.read_text(encoding="utf-8")
            resume_vec = get_embedding(strip_stopwords(resume_text)[:EMBED_MAX_CHARS])
            embedding_available = True
        except RuntimeError as e:
            print(f"Warning: {e}\nFalling back to keyword-only scoring.")

    if embedding_available:
        # Take top-N by keyword for embedding
        candidates = sorted(scored, key=lambda j: -j["kw_raw"])[:top_n]
        candidate_ids = {j["id"] for j in candidates}

        for i, job in enumerate(candidates):
            snippet = strip_stopwords(f"{job['headline']}\n{job['_full_text']}")[:EMBED_MAX_CHARS]
            try:
                job_vec = get_embedding(snippet)
                job["similarity"] = cosine_similarity(resume_vec, job_vec)
            except RuntimeError as e:
                print(f"  Warning: embedding failed for job {i + 1}: {e}")
                job["similarity"] = 0.0
            job["final_score"] = 0.4 * (job["kw_score"] / 10.0) + 0.6 * job["similarity"]
            print(f"  [{i + 1}/{len(candidates)}] {job['headline'][:60]} "
                  f"(kw={job['kw_score']}, sim={job['similarity']:.3f})")

        for job in scored:
            if job["id"] not in candidate_ids:
                job["final_score"] = job["kw_score"] / 10.0 * 0.4

    # Clean up internal field
    for job in scored:
        del job["_full_text"]

    ranked = sorted(scored, key=lambda j: -j["final_score"])
    return ranked[:final_n], embedding_available
