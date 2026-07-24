"""
Retrieve publication outcomes for UKRI Gateway to Research projects using the
Scopus API.

Pipeline:
    1. Builds a Scopus search query from the project grant reference.
    2. Searches Scopus for matching outcomes.
    3. Retrieves the full metadata for each outcome.
    4. Extracts outcome, author, institution and subject metadata.
    5. Saves raw API responses and processed datasets.
    6. Optionally extracts cited references into a separate table.

Exported Outputs:
- scopus_outcomes_latest.csv - Outcome metadata for Scopus records linked to UKRI projects.
- scopus_outcomes_institutions_latest.csv - Author affiliation institutions associated with each matched outcome.
- Optional: scopus_outcomes_references_latest.csv - Bibliographic metadata for references cited by each matched outcome.
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv
import os

# ---------------------------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data" 

CLEAN_INPUT_DIR = DATA_DIR / "cleaned"
PROC_INPUT_DIR = DATA_DIR / "processed" / "gtr"

RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed" / "scopus"

CACHE_DIR = ROOT_DIR / "cache"

for d in (CLEAN_INPUT_DIR, PROC_INPUT_DIR, RAW_DIR, PROC_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API CONFIG
# ---------------------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("SCOPUS_API_KEY")

HEADERS = {
    "X-ELS-APIKey": API_KEY,
    "Accept": "application/json",
    "User-Agent": "DurhamMDS CE project"
}


# ---------------------------------------------------------------------------
# CACHE SETUP
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


# ---------------------------------------------------------------------------
# FIELD MAPPING
# ---------------------------------------------------------------------------

# Map nested Scopus JSON fields to simplified output column names
KEEP_COLUMNS = {
    # Project identifiers
    "project_id": "project_id",
    "project_title": "project_title",
    "grant_reference": "grant_reference",

    # Core pulication metadata
    "abstracts-retrieval-response.coredata.dc:title": "title",
    "abstracts-retrieval-response.coredata.dc:description": "abstract",
    "abstracts-retrieval-response.coredata.prism:doi": "doi",
    "abstracts-retrieval-response.coredata.eid": "eid",
    "abstracts-retrieval-response.coredata.dc:identifier": "scopus_id",
    "abstracts-retrieval-response.coredata.pubmed-id": "pubmed_id",
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

    # Authors and institutions
    "abstracts-retrieval-response.authors.author": "authors",
    "abstracts-retrieval-response.affiliation": "institutions",

    # Subject classifications
    "abstracts-retrieval-response.subject-areas.subject-area":
        "subject_areas",
    "abstracts-retrieval-response.idxterms.mainterm":
        "indexed_keywords",

    # Journal metadata
    "abstracts-retrieval-response.coredata.prism:issn": "issn",
    "abstracts-retrieval-response.coredata.dc:publisher": "publisher",
    "abstracts-retrieval-response.coredata.source-id": "source_id",

    # Reference count
    "abstracts-retrieval-response.item.bibrecord.tail.bibliography.@refcount":
        "reference_count"
}


# ---------------------------------------------------------------------------
# API REQUESTS
# ---------------------------------------------------------------------------

def safe_get(url, params=None, headers=HEADERS, timeout=15, retries=5, 
             backoff=0.8, session=None):
    if session is None:
        session = requests.Session()

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
    return None


# ---------------------------------------------------------------------------
# SCOPUS SEARCH
# ---------------------------------------------------------------------------

def build_project_query(grant_reference):
    if not grant_reference:
        return None
    return f'FUND-NO("{grant_reference.replace("/", "\\/")}")'


def search_scopus(query, session):
    cache_key = f"{SEARCH_PREFIX}{query}"
    # Return cached search results when available
    cached = get_cache(cache_key, expected_type="search")
    if cached is not None:
        return cached

    url = "https://api.elsevier.com/content/search/scopus"
    params = {"query": query, "count": 25}

    r = safe_get(url, params=params, session=session)
    if r is None:
        return []
    
    data = r.json()
    entries = (data.get("search-results", {}).get("entry", []))
    if isinstance(entries, dict):
        entries = [entries]

    save_cache(cache_key, entries, "search")
    return entries

def retrieve_record(eid, session):
    cache_key = f"{RECORD_PREFIX}{eid}"
    # Return cached search results when available
    cached = get_cache(cache_key)
    if cached is not None:
        return cached
    # Retrieve the full outcome record for a Scopus EID
    url = f"https://api.elsevier.com/content/abstract/eid/{eid}"
    r = safe_get(url, session = session, params={"view": "FULL"})
    if r is None:
        return None
    data = r.json()
    save_cache(cache_key, data, "record")
    return data


# ---------------------------------------------------------------------------
# RECORD PARSING
# ---------------------------------------------------------------------------

def parse_authors(authors):
    if not isinstance(authors, list):
        return {"authors": None}
    
    names = []
    for author in authors:
        if not isinstance(author, dict):
            continue
        name = (author.get("ce:indexed-name") or author.get("ce:surname"))
        if name:
            names.append(name)
    return {"authors": "; ".join(names)}


def parse_institutions(institutions, eid=None, scopus_id=None, doi=None, project_id=None):
    """
    Returns institution names as a semicolon-separated string.
    Also saves a separate scopus_institutions.csv table.
    """
    if not isinstance(institutions, list):
        if isinstance(institutions, dict):
            institutions = [institutions]
        else:
            return {"institutions": None,
                    "institution_rows": []}

    aff_names = []
    institution_rows = []
    for org in institutions:
        if not isinstance(org, dict):
            continue
        institution = org.get("affilname")
        if institution:
            aff_names.append(institution)
        institution_rows.append({
            "project_id": project_id,
            "scopus_id": scopus_id,
            "eid": eid,
            "doi": doi,
            "institution": institution,
            "city": org.get("affiliation-city"),
            "country": org.get("affiliation-country"),
            "institution_id": org.get("@id")
        })

    return {"institutions": "; ".join(sorted(set(aff_names)))
            if aff_names else None,
            "institution_rows": institution_rows}


def parse_subject_areas(subject_areas):
    if not isinstance(subject_areas, list):
        return {"subject_areas": None}
    
    subjects = []
    for item in subject_areas:
        if not isinstance(item, dict):
            continue
        subject = item.get("$")
        if subject:
            subjects.append(subject)
    return {"subject_areas": "; ".join(subjects)}


def parse_keywords(indexed_keywords):
    if not isinstance(indexed_keywords, list):
        return {"keywords": None}
    
    keywords = []
    for item in indexed_keywords:
        if not isinstance(item, dict):
            continue
        keyword = item.get("$")
        if keyword:
            keywords.append(keyword)
    return {"keywords": "; ".join(keywords)}


# ---------------------------------------------------------------------------
# OUTPUT GENERATION
# ---------------------------------------------------------------------------

def clean_df(df, timestamp):
    # Standardise Scopus identifier format
    df["scopus_id"] = df["scopus_id"].str.replace("SCOPUS_ID:", "", regex=False)

    # Flatten nested author information
    if "authors" in df.columns:
        author_data = (df["authors"].apply(parse_authors).apply(pd.Series))
        df = pd.concat([df.drop(columns=["authors"]), author_data], axis=1)

    # Extract affiliations into a semicolon-separated column and institutions lookup table
    institution_rows = []
    if "institutions" in df.columns:
        parsed_aff = (df.apply(
            lambda row: parse_institutions(row["institutions"],
                                           eid=row["eid"],
                                           scopus_id=row["scopus_id"],
                                           doi=row["doi"], 
                                           project_id=row["project_id"]), axis=1))
        df["institutions"] = parsed_aff.apply(lambda x: x["institutions"])
        for x in parsed_aff:
            institution_rows.extend(x["institution_rows"])

    if institution_rows:
        pd.DataFrame(institution_rows).to_csv(
            PROC_DIR / f"scopus_outcomes_institutions_{timestamp}.csv",
            index=False, encoding="utf-8")
        pd.DataFrame(institution_rows).to_csv(
            PROC_DIR / "scopus_outcomes_institutions_latest.csv",
            index=False, encoding="utf-8")
        
    # Flatten subject area information
    if "subject_areas" in df.columns:
        subject_data = (df["subject_areas"].apply(parse_subject_areas)
                        .apply(pd.Series))
        df = pd.concat([df.drop(columns=["subject_areas"]), subject_data], axis=1)

    # Flatten indexed keywords
    if "indexed_keywords" in df.columns:
        keyword_data = (df["indexed_keywords"].apply(parse_keywords)
                        .apply(pd.Series))
        df = pd.concat([df.drop(columns=["indexed_keywords"]), keyword_data], axis=1)
    return df

def save_references(df):
    """
    Extract cited reference metadata from each outcome into a 
    outcome-reference table."""
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
                "citing_scopus_id": row.get("scopus_id"),
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

    # Load GtR project data
    cleaned_file = CLEAN_INPUT_DIR / "gtr_projects_clean.csv"
    processed_file = PROC_INPUT_DIR / "gtr_projects_latest.csv"
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

    if args.test_limit:
        df = df.head(args.test_limit)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = requests.Session()

    # Search scopus for outcomes
    rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        query = build_project_query(row.grant_reference)
        records = search_scopus(query, session)
        for record in records:
            eid = record.get("eid")
            if not eid:
                continue

            full_record = retrieve_record(eid, session)
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
        proc_out = clean_df(proc_out, timestamp)
        proc_out.to_csv(PROC_DIR / f"scopus_outcomes_{timestamp}.csv", index=False, encoding="utf-8")
        proc_out.to_csv(PROC_DIR / "scopus_outcomes_latest.csv", index=False, encoding="utf-8")

        print(f"\nSaved {len(proc_out)} search results.")

        if args.save_references:
            refs = save_references(raw_out)
            if refs:
                refs_out = pd.DataFrame(refs)
                refs_out.to_csv(PROC_DIR / f"scopus_outcomes_references_{timestamp}.csv", 
                                index=False, encoding="utf-8")
                refs_out.to_csv(PROC_DIR / "scopus_outcomes_references_latest.csv", 
                                index=False, encoding="utf-8")
                print(f"\nSaved {len(refs_out)} corresponding references.")

    conn.close()


if __name__ == "__main__":
    main()