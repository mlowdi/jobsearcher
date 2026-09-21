# Frozen disagreement adjudicator

This is a one-off localhost instrument for revisiting a frozen set of disagreements between the operator's completed oracle ratings and frozen out-of-fold ranking signals. It is intentionally non-blind: the disagreement and its component signals are part of the evidence being adjudicated.

## Quick start

From the repository root:

```sh
cd /path/to/jobsearcher
uv run python adjudicate.py
```

Then open exactly:

```text
http://127.0.0.1:5001/
```

The defaults are:

- ratings source: `./oracle-ratings.db`
- frozen predictions: `./oracle-analysis/2026-08-23/jobsearcher-title-semantic-oof.csv`
- adjudication state: `./adjudication.db`
- campaign: `first-disagreements`
- source cohort: `pre-filtered-first-pass`
- port: `5001`

Run `uv run python adjudicate.py --help` for the corresponding override flags. The host is fixed to `127.0.0.1`; only the port is configurable.

## What the lanes mean

A new campaign freezes at most 20 cases per lane, in this precedence order. An ad can occur only once.

1. **`model-overestimate`** — original rating 1–2, ordered from highest to lowest frozen `oof_four_way_ensemble` preference probability.
2. **`model-underestimate`** — original rating 4–5, ordered from lowest to highest frozen `oof_four_way_ensemble` preference probability.
3. **`signal-disagreement`** — remaining clear labels (1–2 or 4–5), ordered from largest to smallest frozen `model_spread`.

Ties use `ad_id` order. If a lane has fewer than 20 eligible cases, the campaign contains the available cases only; it never duplicates or backfills a case from an earlier lane.

The displayed model values are frozen out-of-fold decision-support signals. The primary panel shows the stored keyword/source scores, the three components of the final four-way ensemble, its preference probability, and model spread. A secondary panel shows every additional analysis signal supplied by the prediction artifact; optional values absent from an artifact are marked `Not available` and export as empty fields. These signals estimate prior preference, not hiring probability, job quality, or the likelihood of an offer.

## Ratings, reasons, and deferral

For every decision:

- leave the preselected original rating unchanged to **confirm** it, or choose 1–5 to **revise** it;
- select every reason tag that materially explains the rating; at least one is required;
- use the optional note only for evidence the structured tags do not capture.

The negative/constraint tags cover role family, technicality, seniority, contract/consultancy, network/OT/infrastructure, legal/privacy, security-protection, location, credentials, and employer/mission mismatch. The positive/fit tags cover security strategy/GRC, coordination/delivery/change, communication/training/human work, meaningful mission, and adjacent wildcards. The evidence/label tags cover vague ads, duplicates/reposts, changed context, and other uncertainty. Multiple tags are expected when several causes apply; they are evidence, not a forced single diagnosis.

**Defer this case** does not decide or discard it. A deferred case moves behind every never-deferred undecided case. After those cases are exhausted, deferred cases return; another defer rotates a case behind cases with fewer deferrals.

## State, export, and resume semantics

Decisions and events are stored separately in:

```text
./adjudication.db
```

Download the enriched campaign export from:

```text
http://127.0.0.1:5001/export.csv
```

The CSV includes every frozen case, including undecided and deferred cases, with input provenance, frozen metadata and signals, decisions, reason codes, notes, and presentation/defer timestamps.

A campaign is an immutable snapshot. Restarting the same command resumes its stored queue, decisions, reasons, and deferrals without rebuilding or reordering it. On every reopen, the app requires the same canonical source and prediction paths and the same current SHA-256 hashes recorded when the campaign was created. If either path or file hash changes, startup stops instead of mixing evidence. Choose a new campaign name for changed inputs, for example:

```sh
uv run python adjudicate.py --campaign second-disagreements
```

The ratings source is opened with SQLite `mode=ro` and `query_only`. This app never changes prior oracle ratings, the frozen prediction artifact, the production ad snapshot, embeddings, or analysis outputs. It makes no external requests and does not call or start an embedding service.
