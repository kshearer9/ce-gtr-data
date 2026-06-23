"""Test candidate replacement terms against existing terms (unquoted volumes)."""
import requests, time

HEADERS = {"Accept": "application/vnd.rcuk.gtr.json-v7"}
BASE = "https://gtr.ukri.org/gtr/api/projects"

EXISTING = [
    "circular economy",
    "industrial symbiosis",
    "closed-loop",
    "remanufacturing",
    "circular bioeconomy",
]

CANDIDATES = [
    "industrial ecology",
    "material flow",
    "material flows",
    "resource recovery",
    "secondary raw materials",
    "end-of-life",
    "waste valorisation",
    "waste valorization",
    "urban mining",
    "closed material cycle",
    "technical cycle",
]

def total_for(t):
    try:
        r = requests.get(BASE, headers=HEADERS, params={"q": t, "p": 1, "s": 10}, timeout=60)
        r.raise_for_status()
        return r.json().get("totalSize", "?")
    except Exception as e:
        return f"ERR {e}"

print(f"{'term':30s} {'totalSize':>10}")
print("=== EXISTING (kept) ===")
for t in EXISTING:
    print(f"{t:30s} {str(total_for(t)):>10}")
    time.sleep(1)

print("\n=== CANDIDATES ===")
for t in CANDIDATES:
    print(f"{t:30s} {str(total_for(t)):>10}")
    time.sleep(1)