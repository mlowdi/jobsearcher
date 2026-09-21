# Stored-ad oracle

A private localhost tool for quickly rating stored job ads without seeing jobsearcher's current ranker signals. It is a first-pass labelling pass: trust the human response first, then use the ratings later to inspect where the matcher agrees, misses, or surfaces interesting disagreements.

## Quick start

```bash
cd /path/to/jobsearcher
uv run python oracle.py --ads-db /path/to/jobsearcher-production.db
```

Open http://127.0.0.1:5000/ in a browser.

This command creates or reopens the `pre-filtered-first-pass` cohort. By default, it includes only stored ads whose existing normalized keyword score is at least 3, then builds the frozen score-stratified sample from those ads. Use `--min-keyword-score N` to choose another threshold from 1 through 10; the oracle uses the score already stored by jobsearcher and does not rescore the ads.

The old `first-pass` cohort remains a separate, unfiltered snapshot with its original membership, order, ratings, and presentation history. The rating page for either cohort hides all score, threshold, bucket, queue, keyword, matcher, ranker, similarity, and model information. It resumes at the next unrated ad.

Choose the answer that best captures “do I want to look into this?”

1. No, not for me
2. Probably not
3. Maybe
4. Interesting
5. Yes, I want to look into this

**Skip for now** records that the ad was shown without rating or discarding it. Use it when an ad needs more brain than is available.

## Data and export

The oracle reads the supplied ad database read-only. It stores the cohort, ratings, and presentation history in `oracle-ratings.db` beside the app; that private database and its SQLite sidecars are gitignored.

`http://127.0.0.1:5000/export.csv` exports the frozen candidate metadata, each candidate's stored keyword-score snapshot, the cohort threshold, ratings, timestamps, original ranker fields, and skip history for later analysis. Those private selection fields stay out of the rating UI on purpose.

## Starting another pass

A cohort freezes its membership, order, and keyword threshold when created. To make a separate pass from a later snapshot, with another pool size, or with another threshold, give it a new name:

```bash
uv run python oracle.py --ads-db /path/to/new-snapshot.db --cohort september-review --pool-size 500 --min-keyword-score 4
```

Reopening a filtered cohort with a different threshold fails rather than rebuilding it. Use `uv run python oracle.py --help` for all ratings-database, threshold, and localhost-port options.
