import json
import requests
import sys

QUERY_URL = "https://jobsearch.api.jobtechdev.se/search"
OUTFILE = "jobs.txt"

GEOGRAPHY = [
    ("region", "01"),           # Stockholm
    ("municipality", "1980"),   # Västerås
    ("municipality", "1880"),   # Örebro
    ("municipality", "0380"),   # Uppsala
    ("municipality", "0480"),   # Nyköping
    ("municipality", "0484"),   # Eskilstuna
    ("municipality", "0580"),   # Linköping
    ("municipality", "0581"),   # Norrköping
]

OCCUPATION_GROUPS = [
    ("occupation-group", "2516"),   # IT-säkerhetsspecialister
    ("occupation-group", "1335"),   # Driftschefer inom IT
    ("occupation-group", "2421"),   # Organisations- och systemanalytiker
    ("occupation-group", "2422"),   # IT-strateger
]

def fetch(extra_params):
    headers = {"accept": "application/json"}
    params = GEOGRAPHY + extra_params + [("published-after", "1440"), ("limit", "50")]
    response = requests.get(QUERY_URL, headers=headers, params=params)
    response.raise_for_status()
    return response.json()["hits"]

def run_query():
    # Two queries: occupation groups + freetext säkerhetssamordnare, deduplicated by id
    hits_by_id = {}
    for hit in fetch(OCCUPATION_GROUPS):
        hits_by_id[hit["id"]] = hit
    for hit in fetch([("q", "säkerhetssamordnare")]):
        hits_by_id[hit["id"]] = hit
    return list(hits_by_id.values())

def format_hit(item):
    pb_url = item["webpage_url"]
    headline = item["headline"]
    deadline = item["application_deadline"]
    text = item["description"]["text"]
    employer = item["employer"]["name"]
    employment_type = item["employment_type"]["label"]
    publish_date = item["publication_date"]
    string = f"""{headline}
Företag: {employer}
Typ: {employment_type}
Publicerad: {publish_date}
Deadline: {deadline}
URL: {pb_url}
Annons: {text}"""
    return string

def main():
    hits = run_query()

    # If the query returns no jobs, just exit with exit code 1. Otherwise, write the jobs.txt file
    if not hits:
        sys.exit(1)
    else:
        out_array = []
        for hit in hits:
            out_array.append(format_hit(hit))
        with open(OUTFILE,"w") as f:
            f.write("\n---\n".join(out_array))
        sys.exit(0)

if __name__ == "__main__":
    main()
