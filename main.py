import json
import requests
import sys

QUERY_URL = "https://jobsearch.api.jobtechdev.se/search"
OUTFILE = "jobs.txt"

def run_query():
    headers = {"accept": "application/json"}
    params = {"occupation-group": "2516",   # Yrkesgrupp IT-säkerhetsspecialister
              "region": "01",               # Region Stockholm
              "published-after": "1440",    # Last 24 hours (in minutes)
              "limit": 30}                  # Nice and high limit
    response = requests.get(QUERY_URL, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

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
    json_response = run_query()

    # If the query returns no jobs, just exit with exit code 1. Otherwise, write the jobs.txt file
    if json_response["total"]["value"] == 0:
        sys.exit(1)
    else:
        hits = json_response["hits"]
        out_array = []
        for hit in hits:
            out_array.append(format_hit(hit))
        with open(OUTFILE,"w") as f:
            f.write("\n---\n".join(out_array))
        sys.exit(0)

if __name__ == "__main__":
    main()
