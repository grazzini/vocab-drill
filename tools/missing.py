#!/usr/bin/env python3
"""List words short of 5 sentences: tools/missing.py LEVEL [START] [COUNT], easiest first."""
import json
import sys
from pathlib import Path

from sentences import load_words

ROOT = Path(__file__).resolve().parent.parent
candidates = json.loads((ROOT / "data" / "candidates.json").read_text(encoding="utf8"))
have = {}
for file in (ROOT / "data" / "written").glob("*.tsv"):
    for line in file.read_text(encoding="utf8").splitlines():
        if line.strip():
            wid = line.split("\t")[0]
            have[wid] = have.get(wid, 0) + 1
level = int(sys.argv[1])
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
count = int(sys.argv[3]) if len(sys.argv) > 3 else 10**6
short = [(w, 5 - len(candidates[w["id"]]) - have.get(w["id"], 0)) for w in load_words() if w["level"] == level]
short = [(w, n) for w, n in short if n > 0]
print(f"N{level}: {len(short)} words short, {sum(n for _, n in short)} sentences", file=sys.stderr)
for w, n in short[start:start + count]:
    print(f"{n}\t{w['id']}\t{w['gloss'][:60]}")
