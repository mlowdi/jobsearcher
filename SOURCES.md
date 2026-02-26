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
- Dahlgren, P. M. (2018). Svensk text. Svensk nationell datatjänst. https://snd.gu.se/sv/catalogue/study/ext0278
- ~331 single-word entries used (multi-word phrases excluded as machine-translation noise)

**Merge notes:** Lists were lowercased, deduplicated, and sorted. Only single-word entries
were taken from the peterdalle list. Combined total: 627 words.