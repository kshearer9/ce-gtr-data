"""
Search UKRI Circular Economy projects in Scopus.

Current pipeline:
- Load UKRI project dataset
- Search Scopus using either:
    * grant reference (default)
    * project title
- Save every returned Scopus record
- Cache search responses
- Export raw search results

This script deliberately DOES NOT try to decide whether a paper
belongs to a project. It simply records what Scopus returns.
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data" 

INPUT_DIR = DATA_DIR / "processed" / "gtr"

RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed" / "scopus"

CACHE_DIR = ROOT_DIR / "cache"

for d in (INPUT_DIR, RAW_DIR, PROC_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

API_KEY = "f5b9f62fba19244ad19f2f614a3863b5"

HEADERS = {
    "X-ELS-APIKey": API_KEY,
    "Accept": "application/json",
    "User-Agent": "DurhamMDS CE project"
}


# ---------------------------------------------------------------------------
# CACHE
# ---------------------------------------------------------------------------

CACHE_DB = CACHE_DIR / "scopus_cache.db"

SEARCH_PREFIX = "SEARCH::"
RECORD_PREFIX = "RECORD::"

conn = sqlite3.connect(CACHE_DB)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS search_cache (
    query TEXT PRIMARY KEY,
    response TEXT
)
""")

conn.commit()


# ---------------------------------------------------------------------------
# CACHE HELPERS
# ---------------------------------------------------------------------------

def get_cache(query, expected_type=None):
    cursor.execute(
        "SELECT response FROM search_cache WHERE query=?",
        (query,)
    )
    row = cursor.fetchone()
    if not row:
        return None
    data = json.loads(row[0])
    if isinstance(data, dict) and "type" in data and "value" in data:
        if expected_type and data["type"] != expected_type:
            return None
        return data["value"]
    return data


def save_cache(query, response, cache_type):
    payload = {
        "type": cache_type,
        "value": response
    }
    cursor.execute(
        """
        INSERT OR REPLACE INTO search_cache
        VALUES (?,?)
        """,
        (query, json.dumps(payload))
    )
    conn.commit()


KEEP_COLUMNS = {
    # Your project
    "project_id": "project_id",
    "project_title": "project_title",
    "grant_reference": "grant_reference",

    # Core publication metadata
    "abstracts-retrieval-response.coredata.dc:title": "title",
    "abstracts-retrieval-response.coredata.dc:description": "abstract",
    "abstracts-retrieval-response.coredata.prism:doi": "doi",
    "abstracts-retrieval-response.coredata.eid": "eid",
    "abstracts-retrieval-response.coredata.dc:identifier": "scopus_id",

    "abstracts-retrieval-response.coredata.prism:publicationName": "journal",
    "abstracts-retrieval-response.coredata.prism:coverDate": "publication_date",
    "abstracts-retrieval-response.coredata.prism:volume": "volume",
    "abstracts-retrieval-response.coredata.prism:issueIdentifier": "issue",
    "abstracts-retrieval-response.coredata.prism:startingPage": "start_page",
    "abstracts-retrieval-response.coredata.prism:endingPage": "end_page",
    "abstracts-retrieval-response.coredata.prism:pageRange": "page_range",

    "abstracts-retrieval-response.coredata.subtypeDescription": "publication_type",
    "abstracts-retrieval-response.coredata.prism:aggregationType": "aggregation_type",

    "abstracts-retrieval-response.coredata.citedby-count": "citation_count",

    "abstracts-retrieval-response.coredata.openaccess": "open_access",
    "abstracts-retrieval-response.coredata.openaccessFlag": "open_access_flag",

    "abstracts-retrieval-response.coredata.prism:url": "scopus_url",

    # Authors
    "abstracts-retrieval-response.authors.author": "authors",

    "abstracts-retrieval-response.affiliation": "affiliations",

    # Subject classifications
    "abstracts-retrieval-response.subject-areas.subject-area":
        "subject_areas",

    "abstracts-retrieval-response.idxterms.mainterm":
        "indexed_keywords",

    # Journal metadata
    "abstracts-retrieval-response.coredata.prism:issn":
        "issn",

    "abstracts-retrieval-response.coredata.dc:publisher":
        "publisher",

    "abstracts-retrieval-response.coredata.source-id":
        "source_id",

    # Other identifiers
    "abstracts-retrieval-response.coredata.pubmed-id":
        "pubmed_id",

    # Reference count
    "abstracts-retrieval-response.item.bibrecord.tail.bibliography.@refcount":
        "reference_count",
}


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def safe_get(url, params):
    for attempt in range(3):
        try:
            r = requests.get(
                url,
                headers=HEADERS,
                params=params,
                timeout=20)
            if r.status_code != 200:
                print(r.text)
                r.raise_for_status()
            time.sleep(0.3)
            return r
        except Exception as e:
            print(e)
            time.sleep(2)
    return None


# ---------------------------------------------------------------------------
# SEARCH
# ---------------------------------------------------------------------------


def search_scopus(query):
    cache_key = f"{SEARCH_PREFIX}{query}"
    cached = get_cache(cache_key, expected_type="search")
    if cached is not None:
        return cached

    url = "https://api.elsevier.com/content/search/scopus"
    params = {"query": query, "count": 25}

    r = safe_get(url, params)
    if r is None:
        return []
    
    data = r.json()
    entries = (data.get("search-results", {}).get("entry", []))
    if isinstance(entries, dict):
        entries = [entries]

    save_cache(cache_key, entries, "search")
    return entries

def build_project_query(grant_reference):
    if not grant_reference:
        return None
    return f'FUND-NO("{grant_reference.replace("/", "\\/")}")'


# ---------------------------------------------------------------------------
# QUERY BUILDERS
# ---------------------------------------------------------------------------

def retrieve_record(eid):
    cache_key = f"{RECORD_PREFIX}{eid}"
    cached = get_cache(cache_key)
    if cached is not None:
        return cached
    url = f"https://api.elsevier.com/content/abstract/eid/{eid}"
    r = safe_get(url, {"view": "FULL"})
    if r is None:
        return None
    data = r.json()
    save_cache(cache_key, data, "record")
    return data

def save_references(df):
    citations = []
    reference_col = (
        "abstracts-retrieval-response.item.bibrecord.tail.bibliography.reference")

    for _, row in df.iterrows():
        references = row.get(reference_col)
        if not isinstance(references, (list, dict)):
            continue
        if isinstance(references, dict):
            references = [references]

        for ref in references:
            ref_info = ref.get("ref-info", {})

            # DOI
            doi = None
            itemids = (ref_info.get("refd-itemidlist", {}).get("itemid", []))
            if isinstance(itemids, dict):
                itemids = [itemids]
            for item in itemids:
                if item.get("@idtype") == "DOI":
                    doi = item.get("$")

            # Authors
            authors = []
            ref_authors = (ref_info.get("ref-authors", {}))
            if "author" in ref_authors:
                author_list = ref_authors["author"]
                if isinstance(author_list, dict):
                    author_list = [author_list]
                for author in author_list:
                    name = (author.get("ce:indexed-name") or author.get("ce:surname"))
                    if name:
                        authors.append(name)

            elif "collaboration" in ref_authors:
                collaborations = ref_authors["collaboration"]
                if isinstance(collaborations, dict):
                    collaborations = [collaborations]
                for collaboration in collaborations:
                    name = collaboration.get("ce:text")
                    if name:
                        authors.append(name)

            # Year
            year = (ref_info.get("ref-publicationyear", {}).get("@first"))

            # Title
            title = (ref_info.get("ref-title", {}).get("ref-titletext"))

            # Journal/source
            source = (ref_info.get("ref-sourcetitle"))

            citations.append({
                # citing paper
                "citing_project_id": row.get("project_id"),
                "citing_grant_reference": row.get("grant_reference"),
                "citing_eid": row.get(
                    "abstracts-retrieval-response.coredata.eid"),
                "citing_doi": row.get(
                    "abstracts-retrieval-response.coredata.prism:doi"),
                "citing_title":row.get(
                    "abstracts-retrieval-response.coredata.dc:title"),

                # cited paper
                "cited_title": title,
                "cited_doi": doi,
                "cited_year": year,
                "cited_source": source,
                "cited_authors":"; ".join([a for a in authors if a]),
                "reference_text":ref.get("ce:source-text")
            })
    return citations


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--save-references", action="store_true", 
                        help="Extract and save outcome references.")
    args = parser.parse_args()

    df = pd.read_csv(INPUT_DIR / "gtr_projects_latest.csv", encoding="utf-8")
    if args.test_limit:
        df = df.head(args.test_limit)

    rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        query = build_project_query(row.grant_reference)
        records = search_scopus(query)
        for record in records:
            eid = record.get("eid")
            if not eid:
                continue

            full_record = retrieve_record(eid)
            if full_record:
                flat = pd.json_normalize(full_record, sep=".")
                result = flat.iloc[0].to_dict()
                rows.append({
                    "project_id": row.project_id,
                    "project_title": row.title,
                    "grant_reference": row.grant_reference,
                    **result
                })

    if rows:
        raw_out = pd.DataFrame(rows)
        raw_out.to_csv(RAW_DIR / "scopus_raw_outcomes.csv", index=False, encoding="utf-8")

        cols = [c for c in KEEP_COLUMNS if c in raw_out.columns]
        proc_out = raw_out[cols].rename(columns=KEEP_COLUMNS)
        proc_out.to_csv(PROC_DIR / "scopus_outcomes.csv", index=False, encoding="utf-8")

        print(f"\nSaved {len(proc_out)} search results.")

    if args.save_references:
        refs = save_references(raw_out)
        if refs:
            refs_out = pd.DataFrame(refs)
            refs_out.to_csv(PROC_DIR / "scopus_outcomes_references.csv", 
                            index=False, encoding="utf-8")
            print(f"\nSaved {len(refs_out)} corresponding references.")

    conn.close()


if __name__ == "__main__":
    main()