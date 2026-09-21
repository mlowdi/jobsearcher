# Jobsearcher operations

## Security and ownership assumptions

The web application has no login. Bind it only to loopback or a trusted private-network address; never bind it to a public interface or publish it through a public reverse proxy. The shipped systemd user services run as the current user, keep a restrictive umask, and allow writes only inside `%h/projects/jobsearcher`.

Oracle/adjudication databases, corrected analysis files, embedding training matrices, and OOF reports are evidence inputs. All production commands open those inputs read-only. Back them up, do not move the defaults mid-campaign, and never point a writable workspace command at them.

## Initial cutover

The service templates assume the checkout is at `$HOME/projects/jobsearcher`. If it is elsewhere, update the `%h/projects/jobsearcher` paths before installing the units. Run from the repository root:

```console
uv sync --frozen
mkdir -p models backups
sqlite3 jobsearcher.db ".backup 'backups/jobsearcher-before-workspace.db'"
uv run python -c "from db import migrate; print(migrate('jobsearcher.db'))"
uv run python train_model.py --output models/preference.joblib
uv run python import_labels.py --db jobsearcher.db
uv run python rescore.py --db jobsearcher.db --model models/preference.joblib
```

`migrate` is idempotent and preserves the legacy `ads` and `runs` tables. The import always selects `pre-filtered-first-pass`, applies `first-disagreements` adjudicated ratings when present, and reports `imported`, `skipped`, and `missing`. Re-running it does not append another import event.

The production trainer fits all corrected clear labels and uses the corrected nested-OOF report only to choose fixed regularization; it does not report training-fit performance as validation. Its joblib artifact records source hashes, model digest, exact text-constructor versions, dimensions, counts, and the title-metadata boundary (the historical title metadata has ordered IDs but no text hashes).

## Daily refresh

Manual refresh:

```console
uv run python main.py --db jobsearcher.db --model models/preference.joblib
```

The process holds `.jobsearcher-refresh.lock`, fetches and fully upserts mutable source fields, scores active ads, then atomically replaces exact groups and lanes. A no-hit day still rebuilds the active snapshot. Fetch/scoring failure records a failed refresh but does not replace the prior complete snapshot.

Preference endpoint use is opt-in; an absent environment file makes no inference request. Copy `jobsearcher.env.example` to the gitignored `jobsearcher.env`, replace the secret, and enable the endpoint only when it is available:

```ini
# jobsearcher.env
JOBSEARCHER_SECRET_KEY=replace-with-a-long-random-value
JOBSEARCHER_HOST=127.0.0.1
PREFERENCE_EMBED_ENABLED=1
PREFERENCE_EMBED_URL=http://127.0.0.1:9090/v1/embeddings
PREFERENCE_EMBED_MODEL=snowflake-arctic-embed-l-v2.0-q4_k_m.gguf
PREFERENCE_EMBED_TIMEOUT=10
PREFERENCE_EMBED_BATCH_SIZE=16
PREFERENCE_EMBED_TOKEN_BUDGET=8192
```

The configured llama-server endpoint serves that explicit Q4 model through an OpenAI-compatible API with an 8192-token context/request budget. The client preserves the exact text constructor, uses UTF-8 byte length as a conservative token-count upper bound to keep each request within `PREFERENCE_EMBED_TOKEN_BUDGET`, and falls back rather than truncating an over-budget input. The operator should exercise the live endpoint during deployment; implementation tests never call it.

When enabled, an endpoint error is recorded once for that refresh and unresolved ads use the complete title/source fallback. Endpoint vectors are cached under `endpoint:snowflake-arctic-embed-l-v2.0-q4_k_m.gguf`. This is deliberately distinct from the Q8 training/cache provenance; Q4 endpoint vectors are never represented as byte-identical to the local Q8 training vectors.

## Offline full-corpus embedding cache

The repository supplies the importer, not the local embedding-generation job. Generate full and title `.npy` matrices locally, then provide one JSON metadata file per matrix with this contract:

```json
{
  "artifact_schema_version": 1,
  "kind": "full",
  "text_constructor": "oracle-structured-v1",
  "cache_hash_normalization": "nfkc-collapsed-whitespace-v1",
  "embedding_input_normalization": "none-v1",
  "model": "jobsearcher-arctic-embed",
  "model_sha256": "<64 lowercase hex characters>",
  "model_digest_status": "verified-file",
  "cache_model_name": "jobsearcher-arctic-embed@sha256:<same digest>:verified-file",
  "count": 5442,
  "dimensions": 1024,
  "ad_ids": ["..."],
  "text_sha256": ["..."]
}
```

For title metadata use `kind: "title"` and `text_constructor: "raw-headline-v1"`. The title input is the raw headline. The full input is exactly:

```text
<headline>
Employer: <employer>
Location: <municipality>, <region>
Occupation group: <occupation_group>
Employment type: <employment_type>

<description_text>
```

Embed that exact constructor output (`embedding_input_normalization: none-v1`). Cache hashes are SHA-256 of its UTF-8 text after NFKC normalization and whitespace collapse, as implemented by `db.normalized_text_hash`. IDs must be unique and refer to current `ads` rows. Model name, lowercase-hex digest, digest status, matrix dimensions, row count, ID order, and every text hash are validated before any cache row is inserted. Use `verified-file` only when the named local model bytes were hashed; use `declared-manifest` when the file was unavailable and the digest is carried solely from immutable analysis metadata. The trainer records which case applied, and the status is part of the cache key. Use the exact Q8 digest and status recorded by `models/preference.joblib` when importing vectors produced by that model.

```console
uv run python cache_embeddings.py --db jobsearcher.db --matrix /path/full.npy --metadata /path/full.json
uv run python cache_embeddings.py --db jobsearcher.db --matrix /path/title.npy --metadata /path/title.json
uv run python rescore.py --db jobsearcher.db --model models/preference.joblib
```

The importer changes only `embedding_cache`; rescore never fetches JobTech or calls an embedding endpoint. Approximate semantic duplicate grouping is a future extension. Production grouping remains exact and deterministic.

## Local web verification

```console
uv run python web.py --db jobsearcher.db --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/`, then exercise dashboard → high lane → detail → save with a note → saved view → duplicate sibling → low lane. Confirm the event appears in history and that the sibling remains independently unread. The application uses Waitress, strict SameSite session cookies, per-session CSRF tokens, autoescaping, and short SQLite write transactions.

## systemd user-service deployment

The repository only ships templates. The parent operator owns installation and cutover:

```console
install -d "$HOME/.config/systemd/user"
install -m 0644 deploy/jobsearcher-web.service "$HOME/.config/systemd/user/"
install -m 0644 deploy/jobsearcher-refresh.service "$HOME/.config/systemd/user/"
install -m 0644 deploy/jobsearcher-refresh.timer "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user start jobsearcher-refresh.service
systemctl --user enable --now jobsearcher-web.service jobsearcher-refresh.timer
systemctl --user status jobsearcher-web.service jobsearcher-refresh.timer
systemctl --user list-timers jobsearcher-refresh.timer
```

The timer runs at 07:00, is persistent across downtime, and adds up to five minutes of randomized delay. Do not enable these units until the model, initial import/rescore, private bind, and web flow have been verified. To run the user units before login after a reboot, enable lingering for the service account with `loginctl enable-linger "$USER"`.

## Logging and health

```console
journalctl --user -u jobsearcher-refresh.service -n 200 --no-pager
journalctl --user -u jobsearcher-web.service -n 200 --no-pager
journalctl --user -u jobsearcher-refresh.service -f
```

The dashboard shows the most recent refresh status/time, endpoint error, counts, active/unread totals, lane totals, and current-state totals. A `partial` refresh means persistent lanes were rebuilt but one or more ads used fallback scoring or the legacy source embedding reranker was unavailable.

## Backup and restore

SQLite WAL is enabled. Use SQLite's online backup rather than copying only the main file while services are live:

```console
mkdir -p backups
sqlite3 jobsearcher.db ".backup 'backups/jobsearcher-$(date +%F-%H%M%S).db'"
cp -a models/preference.joblib jobsearcher.env backups/
```

Triage history lives in the database; include it in every backup. Retain the model artifact that produced the stored model version.

For rollback, stop both units, preserve the failed database for diagnosis, restore the complete pre-cutover backup as `jobsearcher.db`, restore the matching model/source revision, and start the web service first. Do not try to reverse schema migrations in place. A failed refresh alone does not require database rollback because it leaves the prior lane snapshot intact.

## Old cron retirement

Any previous cron entry remains the operator's responsibility. Disable it only after all of the following are true: the initial refresh succeeds, the web service is reachable only over the intended private interface, the browser triage flow persists an event, the timer is enabled and scheduled, and one timer-triggered refresh is visible in both the dashboard and journal. The new operational path neither writes daily Markdown nor commits/pushes a results repository; historical result files are retained.
