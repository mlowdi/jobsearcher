#!/usr/sbin/zsh

# Let's just be really clear so Claude understands the assignment
QUERY="Read the job ads from @jobs.txt and match against my @resume.md. Write a table to [date]-results.md with a suitability rating from 1-10, the headline, company and the URL. Write only the table and nothing else, make sure it contains all the jobs you read from the file and that each one has a rating and URL. Sort the table from highest rating to lowest."

# Script will exit with exit code 1 if no jobs are returned
uv run main.py > jobs.txt && claude --model haiku --add-dir "$(pwd)" --allowed-tools "Read,Edit,Write" --print $QUERY || echo "No jobs found!"