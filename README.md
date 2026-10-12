# jobsearcher

Self-hosted, persistent job-search workspace built on JobTech ads and a corrected preference model. A one-shot refresh process owns fetching, source scoring, storage, preference scoring, exact-duplicate grouping, and frozen lane snapshots. A separate server-rendered Flask/Waitress process presents the durable state.

The workspace keeps every active canonical ad in exactly one ranked lane:

- **high** — top 50 by ensemble score;
- **possible** — next 150;
- **wildcard** — next pool's 50 widest component disagreements;
- **low** — the complete remaining active corpus.

Exact `(headline, employer)` copies are grouped for navigation, but each ad keeps its own append-only triage history. Scores rank ads; they never set human state or remove low-ranked ads.

## Setup

```console
uv sync
uv run python train_model.py --output models/preference.joblib
uv run python import_labels.py --db jobsearcher.db
uv run python rescore.py --db jobsearcher.db --model models/preference.joblib
```

The trainer reads the completed oracle, adjudication, corrected OOF report, and existing NumPy embedding artifacts without modifying them. Missing production embeddings are honest: scoring uses the title-TF-IDF plus stored source-score fallback, with no zero-filled embedding signals.

The repository intentionally excludes personal data and generated state: resumes, job databases, human-rating databases, analysis matrices, trained models, backups, and local environment files. Training therefore requires your own local evidence artifacts; an existing compatible model can instead be placed at `models/preference.joblib`.

## Run

One refresh, with preference endpoint access disabled by default:

```console
uv run python main.py --db jobsearcher.db --model models/preference.joblib
```

Private local web process:

```console
uv run python web.py --db jobsearcher.db --host 127.0.0.1 --port 8080
```

The production server is Waitress, not Flask's development server. The application has no authentication and must only be bound to loopback or a trusted private-network address; it is not suitable for public exposure. Optional local embeddings use an OpenAI-compatible endpoint, defaulting to `http://127.0.0.1:9090/v1/embeddings`, and an 8192-token request budget. Endpoint vectors retain distinct `endpoint:` cache provenance from the training artifact.

Refresh exit codes are `0` for a completed success/partial refresh (including a no-hit day), `2` for JobTech fetch failure, `3` for scoring/storage failure, and `75` when another refresh holds the advisory lock. Failure leaves the prior complete lane snapshot readable.

## Workspace interface

The production workspace uses IBM Carbon-inspired styling: a dark UI shell, Gray 10 surfaces, blue actions, productive typography, ranked job rows, and a responsive navigation menu. The dashboard separates active ads and ranked lanes from all-time human-state counts. Job detail pages keep state updates beside the description, with source metadata, a compact two-column preference-score table with full provenance, append-only history, and duplicate navigation below.

The interface remains server-rendered and works without JavaScript. CSS and IBM Plex Sans fonts are served locally; there are no CDN requests or frontend build steps. The bundled fonts come from [`@ibm/plex-sans` 1.1.0](https://github.com/IBM/plex) and retain their [SIL Open Font License](static/fonts/OFL.txt). Styling follows [Carbon’s UI shell](https://www.carbondesignsystem.com/building-blocks/core/components/ui-shell-header/specifications) and [productive typography](https://v10.carbondesignsystem.com/guidelines/typography/productive/), without introducing a React dependency.

## Commands

- `train_model.py` — deterministic corrected-label production trainer;
- `import_labels.py` — idempotent corrected oracle/adjudication state import;
- `cache_embeddings.py` — validated offline full/title cache import;
- `rescore.py` — cache-only active-corpus rescore and lane freeze;
- `main.py` — operational one-shot refresh;
- `web.py` — Waitress workspace server;
- `oracle.py` and `adjudicate.py` — retained historical evidence tools, not the production UI.

See [OPERATIONS.md](OPERATIONS.md) for initial cutover, cache metadata, deployment, backup, rollback, logs, and retirement of the old cron job. Historical Markdown results and analysis artifacts remain untouched; Markdown generation and git commits are no longer part of the operational refresh.

## Test

```console
uv run python -m unittest discover -s tests -v
```
