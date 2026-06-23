"""
collect_subjects.py
Query every project in the full GtR CSV download for its research subjects.
Saves the actual subject values (not just a count), checkpoints every 250
projects, and resumes automatically if interrupted.
"""

import time
import pandas as pd
import requests
from pathlib import Path

HEADERS = {"Accept": "application/vnd.rcuk.gtr.json-v7"}
DELAY = 0.5   # seconds between calls (0.5 = ~4hrs; set 1.0 to be gentler/slower)
CSV_PATH = "full_project_search-1781609996900.csv"
OUT_PATH = Path("data") / "subject_coverage_full.csv"
CHECKPOINT_EVERY = 250

# --- Load all project IDs from the full download ---
df = pd.read_csv(CSV_PATH, low_memory=False)
all_ids = df["ProjectId"].dropna().astype(str).tolist()
print(f"Total projects in CSV: {len(all_ids)}")

# --- Resume: skip IDs already collected in a previous run ---
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
done_ids = set()
results = []
if OUT_PATH.exists():
    prev = pd.read_csv(OUT_PATH)
    done_ids = set(prev["project_id"].astype(str))
    results = prev.to_dict("records")
    print(f"Resuming: {len(done_ids)} already done, {len(all_ids) - len(done_ids)} to go")

todo = [pid for pid in all_ids if pid not in done_ids]

def save():
    pd.DataFrame(results).to_csv(OUT_PATH, index=False, encoding="utf-8")

# --- Main loop ---
for i, pid in enumerate(todo, start=1):
    url = f"https://gtr.ukri.org/gtr/api/projects/{pid}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        data = r.json()
        subjects = data.get("researchSubjects", {}).get("researchSubject", [])
        # Build a readable string of subject (percentage) pairs
        parts = []
        for s in subjects:
            text = (s.get("text") or "").strip()
            pct = s.get("percentage")
            if text and pct is not None:
                parts.append(f"{text} ({pct}%)")
            elif text:
                parts.append(text)
        results.append({
            "project_id": pid,
            "has_subjects": bool(subjects),
            "n_subjects": len(subjects),
            "research_subjects": "; ".join(parts),
        })
    except Exception as e:
        results.append({
            "project_id": pid,
            "has_subjects": "ERROR",
            "n_subjects": "",
            "research_subjects": str(e)[:100],
        })
    time.sleep(DELAY)

    if i % CHECKPOINT_EVERY == 0:
        save()
        print(f"  checkpoint saved at {i}/{len(todo)}        ")
    print(f"  processed {i}/{len(todo)}", end="\r")

# --- Final save + summary ---
save()
final = pd.DataFrame(results)
valid = final[final["has_subjects"] != "ERROR"]
has = (valid["has_subjects"] == True).sum()
n = len(valid)
print(f"\n\n=== DONE ===")
print(f"Total processed:   {len(final)}")
print(f"Errors:            {(final['has_subjects'] == 'ERROR').sum()}")
print(f"Has subjects:      {has}/{n} ({100*has/max(n,1):.0f}%)")
print(f"Saved to:          {OUT_PATH}")