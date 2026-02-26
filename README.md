# jobsearcher

Automated job search tool that fetches IT security job postings from [Arbetsförmedlingen's JobTech API](https://data.arbetsformedlingen.se/), scores them against a resume using keyword matching and embedding similarity, and outputs a ranked shortlist.

## How it works

`python main.py` runs the full pipeline:

1. **Fetch** — Queries the jobtechdev API using SSYK occupation groups (IT security specialists, IT managers, system analysts, IT strategists), expanded freetext queries (cybersäkerhet, CISO, SOC manager, etc.), and remote job variants. Ads are deduplicated by ID.
2. **Score** — Each ad is keyword-scored against a two-tier profile (high-value core security terms at 3 points, medium-value related terms at 2 points) with negative keyword penalties. Short keywords like `soc` and `xdr` use word-boundary matching to avoid substring false positives. The top candidates are then reranked using cosine similarity between ad and resume embeddings (snowflake-arctic-embed via a local inference server). If the embedding server is unavailable, scoring degrades gracefully to keyword-only.
3. **Store** — All ads and scores are upserted into a SQLite database (`jobsearcher.db`) with `first_seen`/`last_seen` tracking. Run metadata is logged to a `runs` table.
4. **Output** — A ranked markdown table is written to the results directory.
5. **Commit** — Results are committed to the results git repo for easy access via `git pull`.

## Setup

```
uv sync
```

Requires a `resume.md` file in the project root (gitignored for privacy). The resume should include a keyword-dense matching preamble for optimal embedding coverage.

## Usage

```
uv run main.py
```

Designed to run daily via cron. Set `RESULTS_DIR` to control where output goes (default: `~/job-results`).

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success (or no jobs found — not an error) |
| 2 | API fetch failure |
| 3 | Scoring failure |
| 4 | Git commit failure |

## Project structure

```
main.py          — CLI entrypoint, orchestration
fetcher.py       — API queries, returns raw ad dicts
scorer.py        — Keyword scoring, embedding reranking
db.py            — SQLite storage (ads + run history)
resume.md        — Matching reference document (gitignored)
stopwords-sv.txt — Swedish stopword list for text filtering
```
