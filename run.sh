#!/usr/bin/bash
set -e

# Git repo that serves as the remote — clone/pull from this machine to get results
RESULTS_DIR="${RESULTS_DIR:-$HOME/job-results}"

# Fetch last 24h of job postings
uv run main.py

# Keyword filter + embedding rerank → ranked markdown
uv run job_scorer.py --out-dir "$RESULTS_DIR"

# Commit so the new file is reachable via git pull
cd "$RESULTS_DIR"
git add -A
git commit -m "job results $(date +%Y-%m-%d)"
