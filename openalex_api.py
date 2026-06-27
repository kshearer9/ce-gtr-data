"""
Matches UKRI project titles to OpenAlex awards, retrieves funded outputs,
and enriches them with full OpenAlex work metadata.

Pipeline:
- Load UKRI GtR project dataset
- Query OpenAlex Awards API using cleaned project titles
- Match awards to UKRI grant references
- Extract funded OpenAlex works from matched awards
- Batch fetch full work metadata
- Cache results to avoid repeat API calls
- Export enriched outputs + optional skipped-project log

Exported Outputs:
- openalex_outputs.csv - all works associated with UKRI CE projects and metadata
- Optional: openalex_missing_outputs.csv - logs projects that have no matches and the reason

Note:
- The script currently saves the same project multiple times if it appears under
several projects. Can decide what we want to do with this later.
"""

import requests
import pandas as pd
import argparse
from tqdm import tqdm
import json
import os
import time

# ---------------------------------------------------------------------------
# API CONFIG
# ---------------------------------------------------------------------------

API_KEY = "k4XSonjxeAF7OthnZ8qfzV"
HEADERS = {
    "User-Agent": "DurhamMDS-CE-ResearchProject/1.0 (academic use)"
}

# ---------------------------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------------------------

API_DOWN = False
SKIPPED_LOOKUP = {}

# ---------------------------------------------------------------------------
# CACHE
# ---------------------------------------------------------------------------

# Local caching to avoid repeated API calls
_AWARD_CACHE = "award_cache.json"
_WORK_CACHE = "work_cache.json"

def load_cache(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache, path):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)

award_cache = load_cache(_AWARD_CACHE)
work_cache = load_cache(_WORK_CACHE)

# ---------------------------------------------------------------------------
# API FETCHERS
# ---------------------------------------------------------------------------

def safe_get(url, params=None, headers=None, timeout=15, retries=3, backoff=0.5, project_title=None):
    """
    Wrapper around requests.get with:
    - retry + exponential backoff
    - basic HTTP error handling
    - optional tracking of skipped queries
    """
    global API_DOWN
    last_exception = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            time.sleep(0.3)
            return r
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            if 400 <= status < 500:
                if project_title:
                    SKIPPED_LOOKUP[project_title] = {
                        "status": status,
                        "search": params.get("search") if params else "",
                    }
                print(f"[HTTP {status}] Skipping project:")
                print(f"    Title: {repr(project_title)}")
                print(f"    Search: {params.get('search') if params else ''}")
                return None
            last_exception = e
        except requests.exceptions.ConnectionError as e:
            last_exception = e
        except requests.exceptions.Timeout as e:
            last_exception = e
        except requests.exceptions.RequestException as e:
            last_exception = e
        print(f"[Attempt {attempt}/{retries}] Error fetching {url}: {last_exception}")
        if attempt < retries:
            time.sleep(backoff * (2 ** (attempt - 1)))
    print(f"[Failed after {retries} retries] {url}: {last_exception}")
    # Only stop the script if the API appears genuinely unreachable
    if isinstance(last_exception, requests.exceptions.ConnectionError):
        API_DOWN = True
    return None

def find_award(raw_project_title):
    """
    Search the OpenAlex Awards API using a UKRI project title and
    return the top 5 matching award records.
    """
    # Used cached result if available
    if raw_project_title in award_cache:
        return award_cache[raw_project_title]
    
    # Clean title for search (API can't search "|")
    search_title = raw_project_title.replace(" | ", " ")
    url = "https://api.openalex.org/awards"
    params = {"search": search_title, "per-page": 50, "api_key": API_KEY}
    r = safe_get(url, params=params, headers=HEADERS, project_title=search_title)
    if r is None:
        return []
    data = r.json()
    results = data.get("results", [])

    award_cache[raw_project_title] = results
    save_cache(award_cache, _AWARD_CACHE)
    return results

def fetch_works_batch(work_ids):
    """
    Batch fetch OpenAlex works. Updates work_cache and returns nothing.
    """
    uncached = [wid for wid in work_ids if wid not in work_cache]
    if not uncached:
        return

    url = "https://api.openalex.org/works"
    # OpenAlex supports OR-style filtering via |
    params = {"filter": f"openalex:{'|'.join(uncached)}", "per-page": len(uncached), "api_key": API_KEY}

    r = safe_get(url, params=params, headers=HEADERS)
    if r is None:
        print(f"[Batch fetch failed] {len(uncached)} works")
        return
    data = r.json()
    for work in data.get("results", []):
        work_id = work["id"].split("/")[-1]
        work_cache[work_id] = work
    save_cache(work_cache, _WORK_CACHE)

# ---------------------------------------------------------------------------
# TRANSFORM / PARSING
# ---------------------------------------------------------------------------

def reconstruct_abstract(inverted_index):
    """
    OpenAlex stores abstracts as an inverted index. 
    This function reconstructs a readable abstract string.
    """
    if not inverted_index:
        return None
    # Convert inverted index into position-based dictionary
    word_positions = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions[pos] = word
    # Rebuild OpenAlex inverted-index abstracts into readable text
    return " ".join(word_positions[i] for i in sorted(word_positions.keys()))

def extract_work_metadata(w):
    """
    Extracts and standardises metadata fields from OpenAlex works.
    """
    authors = "; ".join(dict.fromkeys(a["author"]["display_name"]
                for a in w.get("authorships", [])
                if a.get("author") and a["author"].get("display_name")))
    institutions = "; ".join(dict.fromkeys(inst["display_name"]
            for a in w.get("authorships", [])
            for inst in a.get("institutions", []) if inst.get("display_name")))
    topics = "; ".join(t["display_name"] for t in w.get("topics", []) if t.get("display_name"))
    primary_topic = w.get("primary_topic", {}) or {}
    domain = primary_topic.get("domain", {}).get("display_name")
    field = primary_topic.get("field", {}).get("display_name")
    subfield = primary_topic.get("subfield", {}).get("display_name")
    abstract = reconstruct_abstract(w.get("abstract_inverted_index"))

    return {
        "output_title": w.get("title"),
        "output_type": w.get("type"),
        "publication_date": w.get("publication_date"),
        "authors": authors,
        "institutions": institutions,
        "cited_by": w.get("cited_by_count"),
        "fwci": w.get("fwci"),
        "abstract": abstract,
        "topics": topics,
        "domain": domain,
        "field": field,
        "subfield": subfield,
        "doi": w.get("doi"),
        "openalex_url": w.get("id"),
    }

# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-limit", type=int, default=None, help = "Limit number of rows processed (for testing)")
    parser.add_argument("--save-skipped", action="store_true", help="Save skipped projects to CSV (for evaluation)")
    args = parser.parse_args()

    df = pd.read_csv("gtr_ce_projects_clean.csv", encoding="latin1")

    if args.test_limit:
        df = df.head(args.test_limit)

    # Iterate through projects and resolve OpenAlex awards + outputs
    results = []
    skipped_projects = []
    all_work_ids = set()
    award_lookup = {}
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        if API_DOWN:
            print("Stopping run: OpenAlex API is unreachable")
            break
        
        # Skip invalid or missing titles
        project_title = row.title
        if pd.isna(project_title) or not str(project_title).strip():
            print(f"Skipping project {row.project_id}: blank title")
            continue
        project_title = str(project_title).strip()

        grant_reference = row.grant_reference
        if pd.isna(grant_reference):
            continue
        grant_reference = str(grant_reference).strip()

        # Match OpenAlex award to UKRI grant reference
        awards = find_award(project_title)
        award = None
        for candidate in awards:
            award_ref = str(candidate.get("funder_award_id", "")).strip()
            if award_ref == grant_reference:
                award = candidate
                break
        if award is None:
            # Track unmatched projects for evaluation/debugging
            if project_title in SKIPPED_LOOKUP:
                reason = "search error"
            elif len(awards) == 0:
                reason = "no title match"
            else:
                reason = "no title and grant reference match"

            skipped_projects.append({
                "project_id": row.project_id,
                "project_title": project_title,
                "grant_reference": grant_reference,
                "reason": reason,
            })
            continue
        
        # Collect all unique work IDs for batch retrieval
        work_ids = [
            url.split("/")[-1]
            for url in award.get("funded_outputs", [])
        ]
        if not work_ids:
            skipped_projects.append({
                "project_id": row.project_id,
                "project_title": project_title,
                "grant_reference": grant_reference,
                "reason": "no outputs",
            })
            continue
        all_work_ids.update(work_ids)
        award_lookup[row.project_id] = {
            "row": row,
            "grant_reference": grant_reference,
            "award": award,
            "work_ids": work_ids
        }
    
    # Batch fetch all works
    print(f"\nFetching {len(all_work_ids)} unique works in batches...")
    BATCH_SIZE = 50
    work_ids_list = list(all_work_ids)
    for i in tqdm(range(0, len(work_ids_list), BATCH_SIZE)):
        batch = work_ids_list[i:i + BATCH_SIZE]
        fetch_works_batch(batch)

    # Build output using cache
    for item in award_lookup.values():
        row = item["row"]
        grant_reference = item["grant_reference"]
        award = item["award"]

        retrieved = 0
        for work_id in item["work_ids"]:
            w = work_cache.get(work_id)
            if not w:
                continue
            retrieved += 1
            metadata = extract_work_metadata(w)
            results.append({
                "project_id": row.project_id,
                "project_title": row.title,
                "grant_reference": grant_reference,
                "project_openalex_url": award["id"],
                **metadata
            })
        if retrieved == 0:
            skipped_projects.append({
                "project_id": row.project_id,
                "project_title": row.title,
                "grant_reference": grant_reference,
                "reason": "no_outputs",
            })

    out = pd.DataFrame(results)
    out.to_csv("openalex_outputs.csv", index=False)
    print(f"\nSaved {len(out)} funded output(s).")
    
    # Optionally export skipped projects for evaluation
    if args.save_skipped:
        skipped_df = pd.DataFrame(skipped_projects)
        skipped_df.to_csv("openalex_missing_outputs.csv", index=False)
        print(f"Saved {len(skipped_df)} skipped project(s).")

if __name__ == "__main__":
    main()
