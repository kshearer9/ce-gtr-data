"""Re-embed with a larger sentence transformer, as a sensitivity check.

All results so far use all-MiniLM-L6-v2 (6 layers, 384 dimensions), chosen for
speed. This script produces embeddings from all-mpnet-base-v2 (12 layers, 768
dimensions), generally the stronger general-purpose sentence-transformer, so the
bake-off can be repeated and the choice of encoder tested rather than assumed.

Writes to a parallel set of filenames so the MiniLM results are not overwritten
and the two can be compared directly.

Run from the repo root (after the MiniLM embeddings exist):
    /opt/anaconda3/bin/python scripts/classification/embed_texts_mpnet.py

Outputs to data/classification/:
    project_embeddings_mpnet.npy
    corpus_embeddings_mpnet.npy      (only if the labelled corpus is present)
"""
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "classification"
PROJECTS = ROOT / "data" / "cleaned" / "merged" / "projects.csv"
CORPUS = DIR / "gtr_corpus_labelled.csv"
MODEL = "all-mpnet-base-v2"


def main() -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("sentence-transformers not installed; see embed_texts.py header.")

    model = SentenceTransformer(MODEL)
    print(f"Model {MODEL} loaded (device: {model.device}). "
          "This is ~3x slower than MiniLM.")

    projects = pd.read_csv(PROJECTS)
    texts = (projects.title_clean.fillna("") + ". "
             + projects.abstract_text_clean.fillna("")).str.strip()
    t0 = time.time()
    emb = model.encode(texts.tolist(), batch_size=32, normalize_embeddings=True,
                       show_progress_bar=True)
    np.save(DIR / "project_embeddings_mpnet.npy", emb)
    print(f"Projects: {emb.shape} in {time.time() - t0:.0f}s")

    pubs = ROOT / "data" / "cleaned" / "outcomes" / "openalex_all_outcomes_clean.csv"
    if pubs.exists():
        o = pd.read_csv(pubs)
        o = o[o.field.notna()].reset_index(drop=True)
        ptexts = (o.title_clean.fillna("") + ". " + o.abstract_clean.fillna("")).str.strip()
        t0 = time.time()
        pemb = model.encode(ptexts.tolist(), batch_size=32, normalize_embeddings=True,
                            show_progress_bar=True)
        np.save(DIR / "publication_embeddings_mpnet.npy", pemb)
        print(f"Publications: {pemb.shape} in {time.time() - t0:.0f}s")

    if CORPUS.exists():
        corpus = pd.read_csv(CORPUS)
        corpus = corpus[corpus.primary_field.notna()].reset_index(drop=True)
        ctexts = (corpus.title.fillna("") + ". "
                  + corpus.abstract_text.fillna("")).str.strip()
        t0 = time.time()
        cemb = model.encode(ctexts.tolist(), batch_size=32,
                            normalize_embeddings=True, show_progress_bar=True)
        np.save(DIR / "corpus_embeddings_mpnet.npy", cemb)
        print(f"Corpus: {cemb.shape} in {time.time() - t0:.0f}s")
    else:
        print(f"{CORPUS} not found; skipped the corpus (projects only).")

    print("\nDone. To compare encoders, rerun run_bakeoff.py against these files "
          "by pointing project_embeddings.npy at the _mpnet version, or keep both "
          "and report the comparison as a sensitivity check.")


if __name__ == "__main__":
    main()
