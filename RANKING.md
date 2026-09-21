# How jobsearcher ranks ads

## Executive summary

Jobsearcher ranks ads by estimating one narrow thing: **how likely the operator is to want to investigate the role further**. It does not estimate job quality, eligibility, probability of getting an interview, or whether an employer is good.

The production score combines four deliberately different signals:

1. lexical patterns in the job title;
2. semantic meaning of the full ad;
3. semantic meaning of the title;
4. the original hand-written keyword/resume score.

Each learned component is a logistic-regression classifier trained from the operator's ratings. The four outputs are averaged without learned ensemble weights. If embeddings are unavailable, jobsearcher falls back to the title model plus the original score and labels the result as fallback rather than pretending missing components are zero.

The score produces four complete, mutually exclusive lanes over active canonical ads:

- **High:** the top 50 preference scores;
- **Possible:** the next 150;
- **Wildcard:** 50 remaining ads on which the component models disagree most;
- **Low:** every remaining active ad.

Nothing is deleted or hidden because of its score. Human state such as saved, dismissed, applied, or interview is separate and append-only.

## Where the labels came from

The first oracle presented 444 ads without showing any ranking information. The operator rated each ad from 1 to 5 in response to one question: “Does this look like something I want to investigate?”

The initial corpus was prefiltered at the existing normalized keyword score of at least 3. This removed obvious collector noise while retaining a broad range of plausibly adjacent roles. Ads were sampled across five strata of the old score so the labels would not simply validate its top results.

A second adjudication pass selected 60 informative disagreements:

- 20 ads the model liked much more than the operator did;
- 20 ads the operator liked much more than the model did;
- 20 ads where the component models disagreed strongly.

The operator confirmed or revised each rating and added reason tags. Thirteen ratings changed. The corrected distribution was:

| Rating | Meaning | Count |
|---:|---|---:|
| 1 | No, not for me | 297 |
| 2 | Probably not | 58 |
| 3 | Maybe | 33 |
| 4 | Interesting | 33 |
| 5 | Yes, investigate | 23 |

For model training and evaluation, ratings 1–2 are negative, ratings 4–5 are positive, and rating 3 is excluded. This leaves 411 clear examples: 355 negative and 56 positive. Adjudicated ratings take precedence over the original blind rating.

The reason tags showed why semantic similarity alone is insufficient. `wrong-role-family`, consultancy arrangements, credential gates, and network/OT focus were strongly negative. Coordination/delivery/change work and security strategy/GRC were strongly positive. In other words, an ad can discuss all the right security concepts while describing the wrong working life.

## The four production components

### 1. Title TF-IDF

The first learned model works directly on the raw title. It combines:

- lower-cased word unigrams and bigrams;
- character n-grams of length 3–5;
- Swedish stop-word filtering for the word features;
- sublinear term frequency;
- a class-balanced logistic-regression classifier.

The character features help with Swedish compounds, spelling variants, inflection, abbreviations, and near-equivalent titles. The title is often unusually informative because it is the employer's compressed statement of what the role is. It also contains less recruitment boilerplate than the full description.

The component produces `title_tfidf`, a value between 0 and 1.

### 2. Full-ad semantic embedding

The full semantic representation uses Snowflake Arctic Embed 2. The exact text constructor is versioned and includes:

- headline;
- employer;
- location;
- occupation group;
- employment type;
- full description.

The 1,024-dimensional embedding is L2-normalized and passed to a class-balanced logistic-regression classifier. This lets the model learn broader combinations of duties, organizational context, technology, mission, and requirements that do not share literal wording.

The component produces `full_embedding`.

### 3. Title semantic embedding

The title is also embedded independently. The classifier uses the first 256 dimensions of Arctic Embed 2's Matryoshka representation, L2-normalized. This provides a compact semantic title signal alongside the lexical title model.

This component catches relationships such as different titles describing similar role families, while the TF-IDF component remains sensitive to exact phrases and morphology.

The component produces `title_embedding`.

### 4. Original source score

The source score is the earlier jobsearcher ranking heuristic. It contains:

- weighted security, SOC, Microsoft security, incident, compliance, risk, and delivery keywords;
- penalties for common irrelevant domains;
- a normalized keyword score from 1–10;
- for the strongest newly fetched keyword candidates, cosine similarity between the ad and the operator's resume embedding.

When resume embeddings are available, the old score is:

```text
0.4 × normalized keyword score + 0.6 × resume similarity
```

On embedding-enabled source runs, other candidates retain a reduced keyword-only score. If the resume embedding endpoint is unavailable for that source run, every candidate uses its normalized keyword score directly. This signal is less accurate than the learned components, but it encodes explicit domain knowledge and contributes useful independent information.

The component is stored as `source_score`.

## Ensemble and fallback formulas

When both title and full-description embeddings are available, production uses an unweighted four-way average:

```text
preference_score = (
    title_tfidf
  + full_embedding_1024
  + title_embedding_256
  + source_score
) / 4
```

Equal weights are intentional. With only 411 clear labels, learning a second model over the component predictions would add another overfitting surface. Averaging makes every component inspectable and gives no single representation automatic sovereignty.

The component spread is:

```text
component_spread = max(components) - min(components)
```

A large spread means the signals disagree. That disagreement is useful rather than erroneous: it is how the wildcard lane finds roles that deserve a human look despite a middling combined score.

If either embedding is unavailable, jobsearcher uses an explicit two-signal fallback:

```text
preference_score = (title_tfidf + source_score) / 2
```

The missing embedding components remain `NULL`; they are never substituted with zero. Every score records its mode, model version, text hashes, time, and embedding provenance.

## Training and evaluation method

The evaluation uses stratified nested cross-validation:

- five shuffled outer folds produce out-of-fold predictions;
- four inner folds select logistic-regression regularization strength by average precision;
- candidate `C` values are 0.01, 0.1, 1, 10, and 100;
- class balancing is enabled because only 13.6% of clear labels are positive;
- the random seed is fixed.

Each reported learned prediction therefore comes from a model that did not train on that ad. The production artifact is subsequently fitted on all clear corrected labels. For each component it uses the modal regularization value selected across outer folds, with the smaller value breaking ties.

The ensemble itself has no fitted weights. The production four-signal combination uses the stronger 1,024-dimensional full-ad component and the 256-dimensional title component. That exact combination was calculated from the saved out-of-fold component predictions. The earlier fixed analysis report used the 256-dimensional full-ad component; it reached ROC-AUC 0.934 and average precision 0.738. Production's 1,024-dimensional substitution reached 0.936 and 0.749. This is a small post-analysis design choice, so performance on future chronological ads remains the important next validation.

## Corrected out-of-fold results

Positive prevalence is 13.6%, so random average precision would be approximately 0.136.

| Signal | ROC-AUC | Average precision | Precision@25 | Precision@50 | Recall@100 |
|---|---:|---:|---:|---:|---:|
| Original source score | 0.851 | 0.487 | 0.56 | 0.50 | 0.750 |
| Title TF-IDF | 0.911 | 0.689 | 0.76 | 0.66 | 0.875 |
| Full embedding, 1,024 dimensions | 0.919 | 0.676 | 0.80 | 0.68 | 0.911 |
| Title embedding, 256 dimensions | 0.911 | 0.665 | 0.76 | 0.70 | 0.857 |
| Two-signal fallback | 0.923 | 0.677 | 0.76 | 0.68 | 0.929 |
| Production four-signal ensemble | **0.936** | **0.749** | **0.84** | **0.78** | **0.929** |

Interpretation: among the 25 highest out-of-fold ensemble scores, 21 were roles the operator had rated 4–5. Among the top 100, the model recovered 52 of the 56 positive examples.

ROC-AUC measures how often a random positive is ranked above a random negative. Average precision is more informative for the actual inbox because it emphasizes precision and recall among the relatively rare positives. Neither metric says anything about interview success or employer quality.

## From score to inbox lanes

After a successful refresh, jobsearcher considers ads active when their application deadline is today or later.

It first creates exact duplicate groups using a Unicode-normalized, case-folded, whitespace-collapsed `(headline, employer)` key. The canonical representative is selected deterministically by:

1. highest preference score;
2. newest publication date;
3. ad ID.

Duplicate siblings remain visible and independently triageable. Saving or dismissing one never changes another.

Canonical active ads are then sorted by preference score, with publication date and ad ID as deterministic tie-breakers:

1. positions 1–50 become **High**;
2. positions 51–200 become **Possible**;
3. from the remainder, the 50 highest component spreads become **Wildcard**;
4. all remaining ads become **Low**.

This assignment is written as one atomic snapshot. A failed fetch, embedding request, or scoring pass cannot publish half a new ranking beside half an old one. The prior complete snapshot remains available until a new one commits successfully.

The wildcard lane is not random. It contains ads for which the component views disagree most after removing the top 200 combined scores. This preserves a route to unusual titles, adjacent roles, and cases where keyword matching and semantic similarity tell different stories.

Triage state is applied as a view filter, not as part of ranking. An ad can be unread, saved, dismissed, applied, interview, offer, or closed. Every state transition is appended to history. The model does not silently retrain from clicks, propagate state across duplicates, submit applications, or remove low-scoring ads.

## Runtime behavior and provenance

Embeddings are cached by normalized-text SHA-256 and model provenance. The initial corpus was embedded locally using the Q8 Arctic model. New ads may be embedded by a Q4 llama-server endpoint. These vectors use the same model family and dimensions but are not claimed to be byte-identical; the cache and each score preserve which representation supplied each component.

If the endpoint is unavailable or an individual description exceeds the configured safe budget, affected ads receive fallback scores and the dashboard reports a partial refresh. Later successful runs can fill the missing cache entries and upgrade those ads to four-signal mode. A single oversized input does not abort every later ad.

The database stores each component, the ensemble, component spread, scoring mode, model version, input hashes, embedding provenance, and scoring time. The web detail page exposes these values so the ranking remains inspectable.

## What the model does not know

The current model is personal and narrow. It does not yet reliably represent every practical constraint behind a rating. A low rating may mean wrong role family, consultancy arrangement, excessive technical depth, seniority mismatch, credential gate, location, or simply a bad employer. The reason-tag data demonstrates these distinctions but is not yet large enough to train separate constraint models.

The dataset comes from one collector configuration, one period, and one person's preferences. Cross-validation measures interpolation within that corpus. It does not establish temporal generalization to future ads, robustness after the operator's preferences change, or transfer to another person.

The safest operating policy is therefore:

- rank, never automatically reject;
- preserve the complete low lane;
- keep a disagreement lane;
- show score mode and components;
- let human state remain authoritative;
- evaluate later on genuinely future ads before making stronger claims.

## Reproducibility

The production trainer, corrected out-of-fold predictions, reports, source hashes, embedding metadata, and experiment scripts are retained in the project and its ignored analysis artifact directory. The main relevant files are:

- `train_model.py` — deterministic production artifact;
- `preferences.py` — component and ensemble scoring;
- `db.py` — duplicate grouping and lane snapshots;
- `scorer.py` — original source score;
- `oracle-analysis/2026-08-23/jobsearcher-adjudicated-report.json` — corrected evaluation;
- `oracle-analysis/2026-08-23/jobsearcher-adjudicated-oof.csv` — per-ad out-of-fold predictions;
- `models/preference.joblib` — deployed production artifact.

The ranking is intentionally boring at its core: several different representations, simple regularized classifiers, honest out-of-fold evaluation, an unweighted average, and a complete human-review path. The sophistication lies less in the formula than in preserving disagreement, provenance, fallback behavior, and the operator's right to overrule it.
