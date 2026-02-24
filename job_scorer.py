#!/usr/bin/env python3
"""
Job Scoring Utility

Parses job ads from jobs.txt, scores them with keyword matching, then
re-ranks the top candidates using embedding similarity against resume.md.
Optionally passes the final shortlist to Claude for a qualitative analysis.

Usage:
    uv run job_scorer.py [options]

Options:
    --output FILENAME   Output file (default: out/YYYY-MM-DD-results.md)
    --top-n N           Jobs to pass to embedding step after keyword filter (default: 20)
    --final N           Jobs to include in final output (default: 8)
    --no-embed          Skip embedding, use keyword score only
    --claude            Run Claude analysis on final shortlist
"""

import argparse
import json
import re
import subprocess
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Embedding server config (same as engram)
EMBED_URL = "http://toybox:9090/v1/embeddings"
EMBED_MODEL = "snowflake-arctic-embed-l-v2.0-q4_k_m.gguf"

RESUME_FILE = Path("resume.md")
JOBS_FILE = Path("jobs.txt")
STOPWORDS_FILE = Path(__file__).parent / "stopwords-sv.txt"

# Embedding server has a ~2100 char limit per request
EMBED_MAX_CHARS = 2000


# ---------------------------------------------------------------------------
# Stopword filtering
# ---------------------------------------------------------------------------

def load_stopwords() -> set:
    if not STOPWORDS_FILE.exists():
        return set()
    return set(STOPWORDS_FILE.read_text(encoding='utf-8').splitlines())

_STOPWORDS: set = load_stopwords()

def strip_stopwords(text: str) -> str:
    """Remove stopwords from text, preserving word boundaries."""
    if not _STOPWORDS:
        return text
    words = re.split(r'(\s+)', text)
    return ''.join(
        w for w in words
        if w.strip() == '' or w.lower() not in _STOPWORDS
    )


# ---------------------------------------------------------------------------
# Keyword scoring
# ---------------------------------------------------------------------------

PROFILE_KEYWORDS = {
    # High value — core competencies (3 points each)
    'high': [
        # Security roles
        'cybersecurity', 'cybersäkerhet', 'it-säkerhet', 'informationssäkerhet',
        'information security', 'säkerhetsspecialist', 'säkerhetschef', 'ciso',
        'säkerhetsskydd', 'skyddssäkerhet',
        # MSSP/SOC
        'mssp', 'soc', 'security operations',
        # Microsoft stack
        'microsoft 365', 'm365', 'microsoft defender', 'defender', 'sentinel',
        'entra', 'intune', 'purview', 'microsoft security',
        # Incident & compliance
        'incident', 'incidenthantering', 'incident management', 'incident response',
        'nis2', 'dora', 'gdpr', 'compliance', 'efterlevnad',
        # Role match
        'technical delivery', 'delivery manager',
    ],
    # Medium value — related experience (2 points each)
    'medium': [
        'risk', 'riskhantering', 'riskanalys', 'risk management',
        'säkerhetsincident', 'security incident',
        'hotbild', 'threat', 'threat intelligence', 'underrättelse',
        'säkerhetsstrateg', 'säkerhetsstrategi', 'security strategy', 'säkerhetsarkitekt',
        'verksamhetsskydd',
        'kontinuitet', 'continuity', 'beredskap', 'crisis', 'krisberedskap',
        'säkerhetspolicy', 'security policy',
        'ledning', 'ledarskap', 'leadership', 'manager', 'chef', 'lead',
        'zero trust', 'xdr', 'edr', 'siem',
        'säkerhetsrevision', 'audit', 'penetrationstest',
    ],
    # Low value — transferable skills (1 point each)
    'low': [
        'strategi', 'strategy', 'strategisk',
        'samarbete', 'collaboration', 'samverkan',
        'projekt', 'project', 'projektledning',
        'kommunikation', 'communication',
        'process', 'processutveckling',
        'kvalitet', 'quality',
        'offentlig sektor', 'myndighet', 'statlig',
    ]
}

# Negative indicators — penalise clearly irrelevant roles
NEGATIVE_KEYWORDS = [
    'läkemedel', 'pharmaceutical', 'apotek',
    'fastighet', 'lokalstrateg', 'lokaler', 'hyresvärd',
    'data center', 'datacenter', 'facility manager',
    'energi', 'kraft', 'elbolag',
    'bygg', 'construction', 'tunnelbana', 'spårväg',
    'bemanning', 'rekrytering', 'recruitment', 'consultant manager',
    'lss', 'omsorg', 'hemtjänst', 'äldreomsorg',
    'gruv', 'mineral', 'metall',
    'handläggare', 'stiftelserätt',
    'sjuksköterska', 'läkare', 'tandläkare',
]


def score_job_keywords(job: Dict[str, str]) -> int:
    """Raw keyword score (not normalised)."""
    text = job['text']
    score = 0

    for kw in PROFILE_KEYWORDS['high']:
        if kw in text:
            score += 3
    for kw in PROFILE_KEYWORDS['medium']:
        if kw in text:
            score += 2
    for kw in PROFILE_KEYWORDS['low']:
        if kw in text:
            score += 1

    penalty = 0
    for kw in NEGATIVE_KEYWORDS:
        if kw in text:
            if 'säkerhet' in text or 'security' in text:
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

def get_embedding(text: str) -> List[float]:
    data = json.dumps({"model": EMBED_MODEL, "input": text}).encode()
    req = urllib.request.Request(
        EMBED_URL, data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.URLError as e:
        raise RuntimeError(f"Embedding server unavailable at {EMBED_URL}: {e}") from e
    return resp["data"][0]["embedding"]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_jobs(filepath: Path) -> List[Dict[str, str]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    jobs = []
    for block in content.split('\n---\n'):
        block = block.strip()
        if not block:
            continue

        lines = block.split('\n')
        if not lines:
            continue

        headline = lines[0].strip()
        company = url = ""

        for line in lines:
            if line.startswith('Företag:'):
                company = line.removeprefix('Företag:').strip()
            elif line.startswith('URL:'):
                url = line.removeprefix('URL:').strip()

        if not headline or not company or not url:
            continue

        jobs.append({
            'headline': headline,
            'company': company,
            'url': url,
            'raw_text': block,
            'text': strip_stopwords(block.lower()),
        })

    return jobs


# ---------------------------------------------------------------------------
# Claude analysis
# ---------------------------------------------------------------------------

def analyze_with_claude(jobs: List[Dict], resume_text: str) -> str:
    job_summaries = []
    for i, job in enumerate(jobs, 1):
        snippet = job['raw_text'][:2000]
        job_summaries.append(f"## Job {i}: {job['headline']} at {job['company']}\n{snippet}")

    prompt = f"""You are helping a job seeker evaluate job postings against their profile.

# Candidate Profile (resume.md)
{resume_text}

# Job Postings to Evaluate
{chr(10).join(job_summaries)}

Please analyse each job posting in the context of the candidate's profile. For each job:
1. How well does it match the candidate's skills and experience?
2. What are the key reasons to apply or skip?
3. Give a suitability rating: Excellent / Good / Marginal / Skip

Be concise. Format as a list, one job per entry."""

    result = subprocess.run(
        ["claude", "--print", prompt],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return f"Claude analysis failed: {result.stderr}"
    return result.stdout


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Score and rank job postings')
    parser.add_argument('--output', '-o', help='Output file path (overrides --out-dir)')
    parser.add_argument('--out-dir', default='out', help='Output directory (default: out/)')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Jobs to embed after keyword filter (default: 20)')
    parser.add_argument('--final', type=int, default=8,
                        help='Jobs in final output (default: 8)')
    parser.add_argument('--no-embed', action='store_true',
                        help='Skip embedding step, rank by keyword score only')
    parser.add_argument('--claude', action='store_true',
                        help='Run Claude analysis on final shortlist')
    args = parser.parse_args()

    # Output file
    if args.output:
        output_file = Path(args.output)
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = out_dir / f'{date.today().strftime("%Y-%m-%d")}-results.md'

    # Parse jobs
    if not JOBS_FILE.exists():
        print(f"Error: {JOBS_FILE} not found. Run main.py first.")
        return 1

    print(f"Parsing {JOBS_FILE}...")
    jobs = parse_jobs(JOBS_FILE)
    print(f"Found {len(jobs)} jobs")
    if not jobs:
        print("No valid jobs found.")
        return 1

    # Step 1: keyword scoring
    for job in jobs:
        job['kw_raw'] = score_job_keywords(job)
        job['kw_score'] = normalise_score(job['kw_raw'])

    # Step 2: embedding similarity (optional)
    if args.no_embed:
        for job in jobs:
            job['similarity'] = None
            job['final_score'] = job['kw_score'] / 10.0
        print(f"Skipping embedding step (--no-embed)")
    else:
        # Take top-N by keyword score for embedding
        candidates = sorted(jobs, key=lambda j: -j['kw_raw'])[:args.top_n]
        rest = [j for j in jobs if j not in candidates]

        print(f"Embedding {len(candidates)} candidates + resume...")

        if not RESUME_FILE.exists():
            print(f"Warning: {RESUME_FILE} not found, falling back to keyword-only scoring")
            for job in jobs:
                job['similarity'] = None
                job['final_score'] = job['kw_score'] / 10.0
        else:
            resume_text = RESUME_FILE.read_text(encoding='utf-8')
            try:
                resume_vec = get_embedding(strip_stopwords(resume_text)[:EMBED_MAX_CHARS])
            except RuntimeError as e:
                print(f"Warning: {e}\nFalling back to keyword-only scoring.")
                for job in jobs:
                    job['similarity'] = None
                    job['final_score'] = job['kw_score'] / 10.0
            else:
                for i, job in enumerate(candidates):
                    snippet = strip_stopwords(f"{job['headline']}\n{job['raw_text']}")[:EMBED_MAX_CHARS]
                    try:
                        job_vec = get_embedding(snippet)
                        job['similarity'] = cosine_similarity(resume_vec, job_vec)
                    except RuntimeError as e:
                        print(f"  Warning: embedding failed for job {i+1}: {e}")
                        job['similarity'] = 0.0
                    # Combined: 40% keyword, 60% semantic similarity
                    job['final_score'] = 0.4 * (job['kw_score'] / 10.0) + 0.6 * job['similarity']
                    print(f"  [{i+1}/{len(candidates)}] {job['headline'][:60]} "
                          f"(kw={job['kw_score']}, sim={job['similarity']:.3f})")

                for job in rest:
                    job['similarity'] = None
                    job['final_score'] = job['kw_score'] / 10.0 * 0.4  # no embedding bonus

    # Step 3: sort and take final N
    ranked = sorted(jobs, key=lambda j: -j['final_score'])
    final_jobs = ranked[:args.final]

    # Step 4: write markdown output
    lines = [f"# Job search results — {date.today()}\n"]

    if not args.no_embed:
        lines.append("Ranked by combined keyword + embedding similarity score.\n")
    else:
        lines.append("Ranked by keyword score only (embedding disabled).\n")

    lines += [
        "| Rank | Headline | Company | KW | Sim | URL |",
        "|------|----------|---------|-----|-----|-----|",
    ]

    for i, job in enumerate(final_jobs, 1):
        headline = job['headline'].replace('|', '\\|')
        company = job['company'].replace('|', '\\|')
        sim_str = f"{job['similarity']:.3f}" if job['similarity'] is not None else "—"
        lines.append(f"| {i} | {headline} | {company} | {job['kw_score']} | {sim_str} | {job['url']} |")

    # Step 5: optional Claude analysis
    if args.claude:
        print(f"\nRunning Claude analysis on top {len(final_jobs)} jobs...")
        resume_text = RESUME_FILE.read_text(encoding='utf-8') if RESUME_FILE.exists() else ""
        analysis = analyze_with_claude(final_jobs, resume_text)
        lines += ["\n---\n", "# Claude Analysis\n", analysis]

    output_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"\nResults written to {output_file}")

    print(f"\nTop {min(5, len(final_jobs))} matches:")
    for job in final_jobs[:5]:
        sim_str = f", sim={job['similarity']:.3f}" if job['similarity'] is not None else ""
        print(f"  kw={job['kw_score']}{sim_str} — {job['headline']}")

    return 0


if __name__ == '__main__':
    exit(main())
