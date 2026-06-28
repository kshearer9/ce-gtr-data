"""Diagnose how GtR search counts change when terms are quoted as phrases."""
import requests, time

HEADERS = {"Accept": "application/vnd.rcuk.gtr.json-v7"}
BASE = "https://gtr.ukri.org/gtr/api/projects"
TERMS = ["circular economy", "industrial symbiosis", "closed-loop",
         "cradle to cradle", "remanufacturing", "circular bioeconomy"]

def total_for(q):
    r = requests.get(BASE, headers=HEADERS, params={"q": q, "p": 1, "s": 10}, timeout=60)
    r.raise_for_status()
    return r.json().get("totalSize", "?")

print(f"{'term':28s} {'unquoted':>10} {'quoted':>10}")
print("-" * 50)
for t in TERMS:
    unq = total_for(t)
    time.sleep(1)
    quo = total_for(f'"{t}"')   # wrap in double quotes for phrase search
    time.sleep(1)
    print(f"{t:28s} {str(unq):>10} {str(quo):>10}")