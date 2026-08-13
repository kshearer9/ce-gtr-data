GTR_TYPE_MAP = {
    # Publications
    "Journal Article/Review": {
        "type": "publication",
        "subtype": "journal_article"},
    "Conference/Paper/Proceeding/Abstract": {
        "type": "conference",
        "subtype": "conference_paper"},
    "Preprint": {
        "type": "publication",
        "subtype": "preprint"},
    "Working Paper": {
        "type": "publication",
        "subtype": "working_paper"},
    "Systematic review": {
        "type": "publication",
        "subtype": "systematic_review"},
    # Books
    "Book Chapter": {
        "type": "book",
        "subtype": "book_chapter"},
    "Book": {
        "type": "book",
        "subtype": "book"},
    "Book edited": {
        "type": "book",
        "subtype": "edited_book"},
    "Monograph": {
        "type": "book",
        "subtype": "monograph"},
    # Database
    "Database/Collection of data": {
        "type": "dataset",
        "subtype": "dataset"},
    # Reports
    "Technical Report": {
        "type": "report",
        "subtype": "technical_report"},
    "Policy briefing/Report": {
        "type": "report",
        "subtype": "policy_report"},
    "Consultancy Report": {
        "type": "report",
        "subtype": "consultancy_report"},
    "Manual/Guide": {
        "type": "report",
        "subtype": "manual"},
    "Technical Standard": {
        "type": "report",
        "subtype": "standard"},
    # Thesis
    "Thesis": {
        "type": "thesis",
        "subtype": "thesis"},
    "Software": {
        "type": "software",
        "subtype": "software"},
    "Webtool/Application": {
        "type": "software",
        "subtype": "web_application"},
    "Computer model/algorithm": {
        "type": "software",
        "subtype": "model_algorithm"},
    "New/Improved Technique/Technology": {
        "type": "technology",
        "subtype": "technique_or_technology"}}

OPENALEX_TYPE_MAP = {
    "article": ("publication", "journal_article"),
    "review": ("publication", "review"),
    "preprint": ("publication", "preprint"),
    "editorial": ("publication", "editorial"),
    "letter": ("publication", "letter"),
    "conference-paper": ("conference", "conference_paper"),
    "conference-abstract": ("conference", "conference_abstract"),
    "book": ("book", "book"),
    "book-chapter": ("book", "book_chapter"),
    "dataset": ("dataset", "dataset"),
    "report": ("report", "report"),
    "dissertation": ("thesis", "thesis")}