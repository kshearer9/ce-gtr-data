from pathlib import Path
import pandas as pd
from utils.merge_type_map import GTR_TYPE_MAP, SCOPUS_TYPE_MAP, WOS_TYPE_MAP, OPENALEX_TYPE_MAP


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
PROC_DIR = DATA_DIR / "processed"

PROJECT_INPUT_DIR = DATA_DIR / "cleaned"
OUTCOME_INPUT_DIR = PROJECT_INPUT_DIR / "outcomes"

OUTPUT_DIR = PROJECT_INPUT_DIR / "merged"

DISAGREEMENT_DIR = OUTPUT_DIR / "disagreements"
PROJECT_DISAGREEMENT_DIR = DISAGREEMENT_DIR / "projects"
OUTCOME_DISAGREEMENT_DIR = DISAGREEMENT_DIR / "outcomes"

# ---------------------------------------------------------------------------
# DIRECTORY SETUP
# ---------------------------------------------------------------------------

DIRECTORIES = (
    PROJECT_INPUT_DIR,
    OUTCOME_INPUT_DIR,
    OUTPUT_DIR,
    DISAGREEMENT_DIR,
    PROJECT_DISAGREEMENT_DIR,
    OUTCOME_DISAGREEMENT_DIR,
)

def ensure_directories():
    """Create required data directories if they do not already exist."""
    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# SOURCES
# ---------------------------------------------------------------------------

SOURCE_PRIORITY = ["gtr", "scopus", "wos", "openalex"]

TYPE_MAPS = {
    "gtr": GTR_TYPE_MAP,
    "scopus": SCOPUS_TYPE_MAP,
    "wos": WOS_TYPE_MAP,
    "openalex": OPENALEX_TYPE_MAP,
}
