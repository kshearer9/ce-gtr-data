"""Embed project and publication texts for the discipline classification stage.

Run from the repo root:
    /opt/anaconda3/bin/python scripts/classification/embed_texts.py

Requires (one-off):
    /opt/anaconda3/bin/python -m pip install sentence-transformers

Outputs to data/classification/:
    project_embeddings.npy      (1,380 x 384, rows aligned to project_embedding_index.csv)
    project_embedding_index.csv
    publication_embeddings.npy  (field-labelled OpenAlex publications, aligned to publication_embedding_index.csv)
    publication_embedding_index.csv

Model: sentence-transformers all-MiniLM-L6-v2, embeddings L2-normalised.
Deterministic: no sampling, fixed model revision, order follows the source CSVs.
"""
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "classification"
PROJECTS = ROOT / "data" / "cleaned" / "merged" / "projects.csv"
PUBS = ROOT / "data" / "cleaned" / "outcomes" / "openalex_all_outcomes_clean.csv"
MODEL = "all-MiniLM-L6-v2"


def main() -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit(
            "sentence-transformers is not installed. Run:\n"
            "  /opt/anaconda3/bin/python -m pip install sentence-transformers"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL)
    print(f"Model {MODEL} loaded (device: {model.device})")

    # --- projects ---
    p = pd.read_csv(PROJECTS)
    texts = (p["title_clean"].fillna("") + ". " + p["abstract_text_clean"].fillna("")).str.strip()
    t0 = time.time()
    emb = model.encode(texts.tolist(), batch_size=64, normalize_embeddings=True,
                       show_progress_bar=True)
    np.save(OUT / "project_embeddings.npy", emb)
    p[["project_id"]].to_csv(OUT / "project_embedding_index.csv", index=False)
    print(f"Projects: {emb.shape} in {time.time() - t0:.0f}s")

    # --- field-labelled publications (training corpus for set-up B) ---
    o = pd.read_csv(PUBS)
    o = o[o["field"].notna()].reset_index(drop=True)
    ptexts = (o["title_clean"].fillna("") + ". " + o["abstract_clean"].fillna("")).str.strip()
    t0 = time.time()
    pemb = model.encode(ptexts.tolist(), batch_size=64, normalize_embeddings=True,
                        show_progress_bar=True)
    np.save(OUT / "publication_embeddings.npy", pemb)
    o[["project_id", "field", "openalex_url"]].to_csv(
        OUT / "publication_embedding_index.csv", index=False)
    print(f"Publications: {pemb.shape} in {time.time() - t0:.0f}s")
    print("Done. Files written to data/classification/")


if __name__ == "__main__":
    main()
