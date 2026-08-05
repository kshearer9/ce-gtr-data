"""Embed the wider GtR training corpus for the discipline classifier.

Companion to embed_texts.py, using the same model and settings so the
project and corpus embeddings live in one vector space.

Run from the repo root:
    /opt/anaconda3/bin/python scripts/classification/embed_corpus.py

Input:  data/classification/gtr_corpus_labelled.csv
Output: data/classification/corpus_embeddings.npy
        data/classification/corpus_embedding_index.csv
"""
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "classification"
SRC = DIR / "gtr_corpus_labelled.csv"
MODEL = "all-MiniLM-L6-v2"


def main() -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("sentence-transformers not installed; see embed_texts.py header.")

    if not SRC.exists():
        sys.exit(f"{SRC} not found. Copy gtr_corpus_labelled.csv into data/classification/ first.")

    df = pd.read_csv(SRC)
    df = df[df["primary_field"].notna()].reset_index(drop=True)
    texts = (df["title"].fillna("") + ". " + df["abstract_text"].fillna("")).str.strip()
    print(f"Embedding {len(df)} labelled corpus projects with {MODEL}")

    model = SentenceTransformer(MODEL)
    t0 = time.time()
    emb = model.encode(texts.tolist(), batch_size=128, normalize_embeddings=True,
                       show_progress_bar=True)
    np.save(DIR / "corpus_embeddings.npy", emb)
    df[["project_id", "primary_field", "lead_funder"]].to_csv(
        DIR / "corpus_embedding_index.csv", index=False)
    print(f"Done: {emb.shape} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
