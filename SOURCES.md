# Third-party sources

## stopwords-sv.txt

Merged Swedish stopword list built from two sources:

### 1. codelucas/newspaper
- File: `newspaper/resources/text/stopwords-sv.txt`
- Repo: https://github.com/codelucas/newspaper
- License: **MIT** — Copyright (c) 2013 Lucas Ou-Yang
- ~455 words

### 2. peterdalle/svensktext
- File: `stoppord/stoppord.csv`
- Repo: https://github.com/peterdalle/svensktext
- License: Described as "fria att använda" (free to use) — no explicit OSI license found
  in the repo at time of inclusion. Check upstream before publishing.
- ~331 single-word entries used (multi-word phrases excluded as machine-translation noise)

**Merge notes:** Lists were lowercased, deduplicated, and sorted. Only single-word entries
were taken from the peterdalle list. Combined total: 627 words.

If publishing jobsearcher, consider replacing `stopwords-sv.txt` with a list you source
and license independently, or verify peterdalle's license terms directly with the author.
