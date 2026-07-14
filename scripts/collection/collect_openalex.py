"""
Matches UKRI Gateway to Research (GtR) projects to OpenAlex awards using
project titles and grant references, then retrieves both award metadata
and funded research outputs.

Pipeline:
- Load cleaned UKRI GtR project dataset
- Search the OpenAlex Awards API using cleaned project titles
- Match awards to UKRI grant references
- Extract and save OpenAlex award (project) metadata
- Collect funded OpenAlex work IDs from matched awards
- Batch fetch full work metadata
- Cache award searches and work metadata in SQLite to avoid repeated API calls
- Export project metadata, funded outcomes, and (optionally) skipped projects

Exported Outputs:
- openalex_projects_latest.csv - OpenAlex award metadata for matched UKRI projects
- openalex_outcomes_latest.csv - metadata for funded OpenAlex works linked to matched projects

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
import re
from datetime import datetime

# ---------------------------------------------------------------------------
# DIRECTORY CONFIG
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_DIR = ROOT_DIR / "data" / "cleaned"

DATA_DIR = ROOT_DIR / "data" / "processed" / "openalex"
CACHE_DIR = ROOT_DIR / "cache"
    
for d in (DATA_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API CONFIG
# ---------------------------------------------------------------------------

API_KEY = "k4XSonjxeAF7OthnZ8qfzV"
HEADERS = {
    "User-Agent": "DurhamMDS-CE-ResearchProject/1.0 (academic use)",
    "Accept": "application/json"
}

# ---------------------------------------------------------------------------
# CACHE SETUP
# ---------------------------------------------------------------------------

# Cache to disk to avoid repeated API calls
CACHE_DB = CACHE_DIR / "openalex_cache.db"
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
    conn.commit()


# ---------------------------------------------------------------------------
# SEARCH HELPERS
# ---------------------------------------------------------------------------


def clean_search_title(title):
    """
    Normalise project titles before sending them to OpenAlex.

    OpenAlex search can struggle with some punctuation, so we:
    - replace "|" with a space
    - collapse multiple spaces
    - strip leading/trailing whitespace
    """
    title = str(title)
    title = title.replace("|", " ")
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def fallback_search_title(title):
    """
    Produce a simplified version of a project title for a second search
    if the original title returns no useful matches.
    """
    title = clean_search_title(title)
    # Remove anything after a colon
    title = title.split(":")[0]
    # Remove punctuation
    title = re.sub(r"[^\w\s]", " ", title)
    # Collapse spaces again
    title = re.sub(r"\s+", " ", title)
    return title.strip()


# ---------------------------------------------------------------------------
# API FETCHERS
# ---------------------------------------------------------------------------

def safe_get(url, params=None, headers=None, timeout=15, retries=5, 
             backoff=0.8, session=None, failed=None, key=None):
    """
    OpenAlex request wrapper.
    Handles:
    - retries for transient failures
    - exponential backoff
    - rate limiting
    - permanent client errors
    Failed requests are stored in `failed`
    so they can be retried later.
    """
    if session is None:
        session = requests.Session()
    if failed is None:
        failed = {}
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params=params, headers=headers,timeout=timeout)
            status = r.status_code

            # Rate limit
            if status == 429:
                retry_after = r.headers.get("Retry-After")
                wait = (
                    float(retry_after)
                    if retry_after
                    else backoff * (2 ** (attempt - 1))
                )
                print(f"[429] Rate limited. " 
                      f"Waiting {wait:.1f}s")
                time.sleep(wait)
                continue

            # Temporary server errors
            if status >= 500:
                wait = backoff * (2 ** (attempt - 1))
                print(f"[{status}] Server error. "
                      f"Retrying in {wait:.1f}s")
                time.sleep(wait)
                continue

            # Permanent client errors
            if 400 <= status < 500:
                return None
            r.raise_for_status()
            return r

        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as e:
            wait = backoff * (2 ** (attempt - 1))
            print(f"[Connection issue] "
                f"Retry {attempt}/{retries}")
            time.sleep(wait)

        except requests.RequestException as e:
            wait = backoff * (2 ** (attempt - 1))
            print(f"[Request error] "
                f"Retry {attempt}/{retries}")
            time.sleep(wait)

    # Failed after all retries
    if key:
        failed[key] = {
            "url": url,
            "params": params,
            "headers": headers,
            "type": "awards" if "awards" in url else "works",
            "error": f"Failed after {retries} retries"
        }
    return None


def find_award(raw_title, session, failed):    
    """
    Search the OpenAlex Awards API using a cleaned UKRI project title.
    Results are cached using the cleaned title so repeated searches
    for equivalent titles do not trigger another API request.
    """
    search_title = clean_search_title(raw_title)
    cached = get_award(search_title)
    if cached is not None:
        return cached
    url = "https://api.openalex.org/awards"
    params = {"search": search_title, "per-page": 50, "api_key": API_KEY}
    r = safe_get(url, params=params, headers=HEADERS, session=session, 
                 failed=failed, key=raw_title)
    if r is None:
        return []
    results = r.json().get("results", [])
    fallback = None
    if not results:
        fallback = fallback_search_title(raw_title)
        if fallback != search_title:
            params["search"] = fallback
            r = safe_get(url, params=params, headers=HEADERS, session=session,
                         failed=failed, key=raw_title)
            if r is not None:
                results = r.json().get("results", [])
    if results:
        save_award(search_title, results)
    if fallback and fallback != search_title:
        if results:
            save_award(fallback, results)
    return results


def fetch_works_batch(work_ids, session, failed):
    """
    Fetch a batch of OpenAlex works.
    Returns:
        missing work IDs
    """
    if not work_ids:
        return []
    
    placeholders = ",".join("?" * len(work_ids))
    cursor.execute(
        f"""
        SELECT work_id
        FROM work_cache
        WHERE work_id IN ({placeholders})
        """,
        work_ids,
    )

    cached = {row[0] for row in cursor.fetchall()}
    uncached = [wid for wid in work_ids if wid not in cached]
    if not uncached:
        return []

    url = "https://api.openalex.org/works"
    params = {"filter": f"openalex:{'|'.join(uncached)}",
              "per-page": len(uncached), "api_key": API_KEY}
    r = safe_get(url, params=params, headers=HEADERS, session=session,
                 failed=failed, key=",".join(uncached))
    if r is None:
        failed["works_" + ",".join(uncached)] = {
            "url": url,
            "params": params,
            "headers": HEADERS,
            "type": "works",
            "work_ids": uncached
        }
        return uncached

    data = r.json()
    returned_ids = []
    for work in data.get("results", []):
        work_id = work["id"].split("/")[-1]
        save_work(work_id, work)
        returned_ids.append(work_id)

    missing = [wid for wid in uncached if wid not in returned_ids]
    if missing:
        failed["works_" + ",".join(missing)] = {
            "url": url,
            "params": {
                "filter": f"openalex:{'|'.join(missing)}",
                "per-page": len(missing),
                "api_key": API_KEY
            },
            "headers": HEADERS,
            "type": "works",
            "work_ids": missing
        }
    return missing

def retry_failed_requests(failed, session, max_attempts=3):
    remaining = failed.copy()
    for attempt in range(1, max_attempts + 1):
        if not remaining:
            break
        print(f"\nRetry round {attempt}/{max_attempts}: "
              f"{len(remaining)} requests")
        still_failed = {}
        for key, request in tqdm(remaining.items(), desc="Retrying failed requests"):
            r = safe_get(request["url"], params=request["params"],
                         headers=request["headers"], session=session,
                         failed=still_failed, key=key)
            if r is not None:
                if request["type"] == "awards":
                    results = r.json().get("results", [])
                    save_award(key, results)

                elif request["type"] == "works":
                    for work in r.json().get("results", []):
                        work_id = work["id"].split("/")[-1]
                        save_work(work_id, work)
        remaining = still_failed
    return remaining

# ---------------------------------------------------------------------------
# TRANSFORM / PARSING
# ---------------------------------------------------------------------------

def extract_award_metadata(a):
    """
    Extracts metadata fields from OpenAlex awards.
    """
    return {
        "openalex_url": a.get("id", ""),
        "description": a.get("description", ""),
        "funding_amount": a.get("amount", ""),
        "currency": a.get("currency", ""),
        "funding_type": a.get("funding_type", ""),
        "start_date": a.get("start_date", ""),
        "end_date": a.get("end_date", ""),
        "ukri_url": a.get("landing_page_url", ""),
        "primary_topic": (a.get("primary_topic") or {}).get("display_name", ""),
        "primary_topic_score": (a.get("primary_topic") or {}).get("score", ""),
        "subfield": (a.get("primary_topic") or {}).get("subfield", {}).get("display_name", ""),
        "field": (a.get("primary_topic") or {}).get("field", {}).get("display_name", ""),
        "domain": (a.get("primary_topic") or {}).get("domain", {}).get("display_name", "")
    }

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


def extract_work_url(w):
    """
    Returns the best available URL for an OpenAlex work.

    Priority:
    1. DOI
    2. Landing page URL
    3. PDF URL
    4. None if no URL exists
    """
    # 1. DOI
    if w.get("doi"):
        return w["doi"]
    # 2. Landing page URL
    for loc in w.get("locations", []):
        if loc.get("landing_page_url"):
            return loc["landing_page_url"]
    # 3. PDF URL
    for loc in w.get("locations", []):
        if loc.get("pdf_url"):
            return loc["pdf_url"]
    # Nothing found
    return None

def extract_doi(w):
    """
    Extract DOI without the https://doi.org/ prefix.
    """
    doi = w.get("doi")
    if not doi:
        return None
    return doi.replace("https://doi.org/", "")


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
        "title": w.get("title"),
        "outcome_type": w.get("type"),
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
        "url": extract_work_url(w),
        "doi": extract_doi(w),
        "openalex_url": w.get("id")
    }

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-limit", type=int, default=None, help = "Limit number of rows processed (for testing)")
    args = parser.parse_args()

    # Load GtR project data
    cleaned_file = INPUT_DIR / "gtr_projects_clean.csv"
    processed_file = ROOT_DIR / "data" / "processed" / "gtr" / "gtr_projects_latest.csv"
    # If data has not been cleaned yet, use data from processed folder
    if cleaned_file.exists():
        input_file = cleaned_file
        print("Using cleaned GtR projects dataset.")
    elif processed_file.exists():
        input_file = processed_file
        print("Cleaned dataset not found. Using processed GtR projects dataset.")
    else:
        raise FileNotFoundError(
            "Could not find either gtr_projects_clean.csv or gtr_projects_latest.csv")

    df = pd.read_csv(input_file, encoding="utf-8")
    df = pd.read_csv(INPUT_DIR / "gtr_projects_clean.csv", encoding = "utf-8")

    if args.test_limit:
        df = df.head(args.test_limit)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = requests.Session()

    # Iterate through projects and resolve OpenAlex awards + works
    results = []
    award_results = []
    all_work_ids = set()
    award_lookup = {}
    failed = {}
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        
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
        awards = find_award(
            project_title,
            session,
            failed
        )
        award = None
        for candidate in awards:
            award_ref = str(candidate.get("funder_award_id", "")).strip()
            if award_ref == grant_reference:
                award = candidate
                break

        if award is None:
            continue

        # Save all award metadata
        award_row = {
            "project_id": row.project_id,
            "project_title": row.title,
            "grant_reference": grant_reference,
            **extract_award_metadata(award)
        }

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
        try:
            fetch_works_batch(
                batch,
                session,
                failed
            )
        except Exception as e:
            print(f"Batch exception: {e}")
    
    if failed:
        print(f"\nRetrying {len(failed)} failed requests...")
        failed = retry_failed_requests(failed, session)
        print(f"Still failed: {len(failed)}")
    
    # Build outcome using cache
    for item in award_lookup.values():
        row = item["row"]
        grant_reference = item["grant_reference"]
        award = item["award"]

        if not item["work_ids"]:
            continue

        for work_id in item["work_ids"]:
            w = get_work(work_id)
            if w is None:
                continue
            metadata = extract_work_metadata(w)
            results.append({
                "project_id": row.project_id,
                "project_title": row.title,
                "grant_reference": grant_reference,
                "project_openalex_url": award["id"],
                **metadata
            })

    # Save outcomes
    out = pd.DataFrame(results)
    out.to_csv(DATA_DIR / f"openalex_outcomes_{timestamp}.csv", index=False, encoding="utf-8")
    out.to_csv(DATA_DIR / "openalex_outcomes_latest.csv", index=False, encoding="utf-8")

    projects_df = pd.json_normalize(award_results)
    projects_df.to_csv(DATA_DIR / f"openalex_projects_{timestamp}.csv", index=False, encoding="utf-8")
    projects_df.to_csv(DATA_DIR / "openalex_projects_latest.csv", index=False, encoding="utf-8")

    print(f"\nSaved {len(out)} funded outcomes.")
    print(f"Saved {len(projects_df)} projects.")
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()