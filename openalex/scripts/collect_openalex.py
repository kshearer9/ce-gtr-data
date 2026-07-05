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
import time
import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# DIRECTORY CONFIG
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
OPENALEX_DIR = SCRIPT_DIR.parent    
ROOT_DIR = OPENALEX_DIR.parent   
DATA_DIR = OPENALEX_DIR / "data"
GTR_DATA_DIR = ROOT_DIR / "gtr" / "data" / "processed"

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

# Cache to disk to avoid repeated API calls
CACHE_DB = DATA_DIR / "openalex_cache.db"
CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(str(CACHE_DB))
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS award_cache (
    project_title TEXT PRIMARY KEY,
    response TEXT NOT NULL
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS work_cache (
    work_id TEXT PRIMARY KEY,
    response TEXT NOT NULL
)
""")
conn.commit()


# ---------------------------------------------------------------------------
# CACHE HELPERS
# ---------------------------------------------------------------------------

def get_award(title):
    cursor.execute(
        "SELECT response FROM award_cache WHERE project_title = ?",
        (title,)
    )
    row = cursor.fetchone()
    return json.loads(row[0]) if row else None


def save_award(title, results):
    cursor.execute(
        """
        INSERT OR REPLACE INTO award_cache
        (project_title, response)
        VALUES (?, ?)
        """,
        (title, json.dumps(results))
    )
    conn.commit()


def get_work(work_id):
    cursor.execute(
        "SELECT response FROM work_cache WHERE work_id = ?",
        (work_id,)
    )
    row = cursor.fetchone()
    return json.loads(row[0]) if row else None


def save_work(work_id, work):
    cursor.execute(
        """
        INSERT OR REPLACE INTO work_cache
        (work_id, response)
        VALUES (?, ?)
        """,
        (work_id, json.dumps(work))
    )

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

def find_award(search_title, raw_title):
    """
    Search the OpenAlex Awards API using a UKRI project title and
    return the top 5 matching award records.
    """
    # Use cached result if available
    cached = get_award(raw_title)
    if cached is not None:
        return cached
    
    # Clean title for search (API can't search "|")
    search_title = raw_title.replace("|", " ")
    url = "https://api.openalex.org/awards"
    params = {"search": search_title, "per-page": 50, "api_key": API_KEY}
    r = safe_get(url, params=params, headers=HEADERS, project_title=search_title)
    if r is None:
        return []
    data = r.json()
    results = data.get("results", [])
    save_award(raw_title, results)
    return results

def fetch_works_batch(work_ids):
    """
    Batch fetch OpenAlex works. Updates work_cache and returns nothing.
    """
    if not work_ids:
        return
    placeholders = ",".join("?" * len(work_ids))
    cursor.execute(
        f"SELECT work_id FROM work_cache WHERE work_id IN ({placeholders})",
        work_ids,
    )
    cached = {row[0] for row in cursor.fetchall()}
    uncached = [wid for wid in work_ids if wid not in cached]
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
        save_work(work_id, work)
    conn.commit()

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
    topics = "; ".join(dict.fromkeys(t["display_name"]
                for t in w.get("topics", []) if t.get("display_name")))
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
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-limit", type=int, default=None, help = "Limit number of rows processed (for testing)")
    parser.add_argument("--save-skipped", action="store_true", help="Save skipped projects to CSV (for evaluation)")
    args = parser.parse_args()

    df = pd.read_csv(GTR_DATA_DIR / "gtr_ce_projects_latest.csv", encoding = "latin1")

    if args.test_limit:
        df = df.head(args.test_limit)

    # Iterate through projects and resolve OpenAlex awards + outputs
    results = []
    award_results = []
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
        search_title = project_title.replace(" | ", " ")
        awards = find_award(search_title, project_title)
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

        # Save all award metadata
        award_row = {
            "project_id": row.project_id,
            "project_title": row.title,
            "grant_reference": grant_reference
        }

        award_row.update(award)
        award_results.append(award_row)
        
        # Collect all unique work IDs for batch retrieval
        work_ids = [
            url.split("/")[-1]
            for url in award.get("funded_outputs", [])
        ]

        award_lookup[row.project_id] = {
            "row": row,
            "grant_reference": grant_reference,
            "award": award,
            "work_ids": work_ids
        }

        all_work_ids.update(work_ids)

    
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

        if not item["work_ids"]:
            continue

        retrieved = 0
        for work_id in item["work_ids"]:
            w = get_work(work_id)
            if w is None:
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
                "reason": "no outputs",
            })

    # Save outputs
    out = pd.DataFrame(results)
    out.to_csv(DATA_DIR / "openalex_outputs.csv", index=False)

    awards_df = pd.json_normalize(award_results)
    awards_df.to_csv(DATA_DIR / "openalex_awards.csv", index=False)

    print(f"\nSaved {len(out)} funded output(s).")
    print(f"Saved {len(awards_df)} award(s).")
    
    # Optionally export skipped projects for evaluation
    if args.save_skipped:
        skipped_df = pd.DataFrame(skipped_projects)
        skipped_df.to_csv(DATA_DIR / "openalex_missing_outputs.csv", index=False)
        print(f"Saved {len(skipped_df)} skipped project(s).")

    conn.close()

if __name__ == "__main__":
    main()
