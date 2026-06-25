"""
This script links UKRI circular economy projects to their associated research outputs in OpenAlex
using grant reference IDs.

It reads a CSV file of UKRI-funded projects, queries the OpenAlex Works API using each project's
grant reference, and retrieves all publications linked to that grant ID.

This version includes some errors as its currently matching works with the project id anywhere in
their id. It also collects no outputs for many of the grants, so next steps is looking into adding 
different searches to capture more.

Test run example:  python3 openalex_api.py --test-limit 30 --verbose
"""

import requests
import pandas as pd
import argparse
from tqdm import tqdm

API_KEY = "k4XSonjxeAF7OthnZ8qfzV"
HEADERS = {"User-Agent": "DurhamMDS-CE-ResearchProject/1.0 (academic use)"}

grant_cache = {}

def search_openalex_by_grant(grant_id):
    """
    Query OpenAlex for research works linked to a specific grant ID
    and return allmatching publications.
    """
    if grant_id in grant_cache:
        return grant_cache[grant_id]
    
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"awards.funder_award_id:{grant_id}",
        "per-page": 200,
        "api_key": API_KEY
    }

    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    data = r.json()

    if "results" not in data:
        print("Error:", data)
        return []
    
    grant_cache[grant_id] = data["results"]
    return data["results"]


def main():
    parser = argparse.ArgumentParser(description="OpenAlex UKRI grant matching pipeline")
    parser.add_argument("--test-limit", type=int, default=None, help = "Limit number of rows processed (for testing)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress messages")
    args = parser.parse_args()
    
    df = pd.read_csv("gtr_ce_projects_enriched.csv")
    data = df.head(args.test_limit) if args.test_limit else df

    results = []

    iterator = tqdm(data.itertuples(index=False), total=len(data))
    for row in iterator:
        grant_id = row.grant_reference
        if pd.isna(grant_id) or str(grant_id).strip() == "":
            continue
        if args.verbose: 
            print("\nGRANT:", grant_id)
        
        works = search_openalex_by_grant(grant_id)
        if args.verbose: 
            print("RESULTS:", len(works))
        
        for w in works:
            authors = "; ".join(a["author"]["display_name"]
                for a in w.get("authorships", []) if a.get("author"))
            institutions = "; ".join(inst["display_name"]
                for a in w.get("authorships", [])
                for inst in a.get("institutions", []) if inst.get("display_name"))
            concepts = "; ".join(c["display_name"]
                for c in w.get("concepts", []) if c.get("display_name"))
            topics = "; ".join(t["display_name"]
                for t in w.get("topics", []) if t.get("display_name"))
            keywords = "; ".join(k["display_name"]
                for k in w.get("keywords", []) if k.get("display_name"))
            primary_topic = (w.get("primary_topic", {}).get("display_name")
                if w.get("primary_topic") else None)

            results.append({
                "project_id": row.project_id,
                "project_title": row.csv_Title,
                "grant_id": grant_id,
                "output_title": w.get("title"),
                "openalex_url": w.get("id"),
                "doi": w.get("doi"),
                "publication_date": w.get("publication_date"),
                "authors": authors,
                "institutions": institutions,
                "cited_by": w.get("cited_by_count"),
                "fwci": w.get("fwci"),
                "concepts": concepts,
                "topics": topics,
                "keywords": keywords,
                "primary_topic": primary_topic
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(
        "openalex_matches.csv",
        index=False
    )
    print("Saved:", len(out_df), "matches")

if __name__ == "__main__":
    main()