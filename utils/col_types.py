import pandas as pd

PROJECT_COLUMN_TYPES = {
    # Strings
    "discipline_primary": "string",
    "fund_end": "string",
    "fund_start": "string",
    "grant_reference": "string",
    "gtr_url": "string",
    "lead_organisation": "string",
    "participant_organisations": "string",
    "principal_investigator": "string",
    "project_id": "string",
    "project_openalex_url": "string",
    "project_title": "string",
    "research_subjects": "string",
    "research_topics": "string",
    "sectors": "string",
    "ukri_url": "string",

    # Numeric
    "funding_amount": "float64",
    "primary_topic_score": "float64",
    "value_gbp": "float64",
}


OUTCOME_COLUMN_TYPES = {
    # Strings
    "abstract": "string",
    "author": "string",
    "author_keywords": "string",
    "categories_extended": "string",
    "categories_traditional": "string",
    "chapter_num": "string",
    "chapter_title": "string",
    "citation_topic_macro": "string",
    "citation_topic_meso": "string",
    "citation_topic_micro": "string",
    "company_name": "string",
    "conference": "string",
    "conference_location": "string",
    "conference_num": "string",
    "description": "string",
    "doi": "string",
    "early_access_year": "string",
    "eid": "string",
    "eissn": "string",
    "edition": "string",
    "end_page": "string",
    "funding_agencies": "string",
    "funding_grant_ids": "string",
    "further_funding_id": "string",
    "grant_reference": "string",
    "guideline_title": "string",
    "impact": "string",
    "influence": "string",
    "institutions": "string",
    "ip_exploited": "string",
    "isbn": "string",
    "issue": "string",
    "issn": "string",
    "journal": "string",
    "keywords": "string",
    "methods": "string",
    "narrative": "string",
    "open_source_license": "string",
    "organisation": "string",
    "organisations": "string",
    "orcids": "string",
    "outcome_id": "string",
    "page_range": "string",
    "page_ref": "string",
    "parent_org": "string",
    "partner_contribution": "string",
    "patent_id": "string",
    "patent_url": "string",
    "pi_contribution": "string",
    "project_id": "string",
    "project_openalex_url": "string",
    "project_title": "string",
    "publisher": "string",
    "pubmed_id": "string",
    "reg_num": "string",
    "researcher_ids": "string",
    "results": "string",
    "sdg_categories": "string",
    "series_num": "string",
    "series_title": "string",
    "software_open_sourced": "string",
    "source_id": "string",
    "source_title": "string",
    "start_page": "string",
    "subject_areas": "string",
    "sub_title": "string",
    "supporting_url": "string",
    "title": "string",
    "topics": "string",
    "typeOfPresentation": "string",
    "url": "string",
    "vol_num": "string",
    "volume": "string",
    "volume_title": "string",
    "wos_title": "string",
    "wos_uid": "string",
    "year": "string",
    "yearsOfDissemination": "string",
    "yearFirstProvided": "string",
    "yearEstablished": "string",
    "yearDevelopmentCompleted": "string",
    "yearProtectionGranted": "string",

    # Numeric
    "amount": "float64",
    "cited_by": "Int64",
    "end": "Int64",
    "fwci": "float64",
    "n_addresses": "Int64",
    "reference_count": "Int64",
    "start": "Int64",
    "times_cited_all_db": "Int64",
    "times_cited_core": "Int64",
    "total_pages": "Int64"
}


def get_base_column_name(col, col_types):
    """
    Convert a column name to its canonical name.

    Examples:
        title               -> title
        title_clean         -> title
        gtr_title           -> title
        gtr_title_clean     -> title
        openalex_abstract   -> abstract
        wos_doi             -> doi
    """
    # Remove _clean suffix
    if col.endswith("_clean"):
        col = col[:-6]
    # Already canonical
    if col in col_types:
        return col
    # Remove source prefix
    prefix, separator, remainder = col.partition("_")
    if separator and remainder in col_types:
        return remainder
    return col


def get_dtypes(cols, col_types):
    """
    Return pandas dtypes for recognised columns.
    """
    dtypes = {}
    for col in cols:
        base_col = get_base_column_name(col, col_types)
        if base_col in col_types:
            dtypes[col] = col_types[base_col]
    return dtypes


def read_csv(path, col_types=OUTCOME_COLUMN_TYPES):
    """
    Read a CSV using the shared column type rules.
    """
    # Read only the header
    cols = pd.read_csv(path, nrows=0).columns
    # Build dtypes for columns actually present
    dtypes = get_dtypes(cols, col_types)
    # Read the  data
    df = pd.read_csv(path, encoding="utf-8", 
                     dtype=dtypes, low_memory=False)
    return df