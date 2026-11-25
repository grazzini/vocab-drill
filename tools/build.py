#!/usr/bin/env python3
"""Build index.html from src/app.html plus the JLPT and frequency word lists.

Usage: python3 -I tools/build.py [source-dir]

source-dir defaults to data/ and must hold n1.csv..n5.csv from jamsinclair/open-anki-jlpt-decks
(MIT) and freq.txt from hingston/japanese (44998-japanese-words.txt). Output
order is N5 to N1, most frequent first within each level, so it is the order
the diagnostic walks.
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAPSULE_MAX = 24
UNRANKED = 999999


def forms(value):
    return [f.strip() for f in re.split(r"[;；]", value) if f.strip()]


def capsule(gloss):
    """First sense of the gloss, cut to a few words."""
    text = re.sub(r"^-+\s*|\s*-+$", "", gloss).strip()
    # Split on separators outside parentheses.
    depth, first = 0, ""
    for ch in text:
        depth += ch in "(（"
        depth -= ch in ")）"
        if depth <= 0 and ch in ",;；、":
            if first.strip():
                break
        first += ch
    first = first.strip()
    bare = re.sub(r"\s*[(（][^)）]*[)）]\s*", " ", first).strip()
    result = bare if len(bare) >= 2 else first
    if len(result) > CAPSULE_MAX:
        result = result[:CAPSULE_MAX].rsplit(" ", 1)[0].rstrip(" ,;-")
    return result


def load_rank(path):
    rank = {}
    with open(path, encoding="utf8") as f:
        for i, line in enumerate(f):
            rank.setdefault(line.strip(), i + 1)
    return rank


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
    rank = load_rank(src / "freq.txt")
    merged = {}
    for level in range(5, 0, -1):
        with open(src / f"n{level}.csv", encoding="utf8", newline="") as f:
            for row in list(csv.reader(f))[1:]:
                if len(row) < 3 or not row[0].strip():
                    continue
                expr, reading, gloss = (c.strip() for c in row[:3])
                reading = reading or expr
                key = (forms(expr)[0], forms(reading)[0])
                if key in merged:  # Easiest level wins; keep any new sense.
                    entry = merged[key]
                    if gloss.lower() not in entry["gloss"].lower():
                        entry["gloss"] += "; " + gloss
                    continue
                merged[key] = dict(expr=expr, reading=reading, gloss=gloss, level=level)

    words = []
    for entry in merged.values():
        # Written forms only: a reading like に would match the particle.
        names = [f.strip("～~") for f in forms(entry["expr"])]
        best = min((rank.get(n, UNRANKED) for n in names), default=UNRANKED)
        words.append((-entry["level"], best, entry))
    words.sort(key=lambda w: (w[0], w[1]))

    slug_file = ROOT / "src" / "slugs.json"  # From tools/slugs.py.
    slugs = json.loads(slug_file.read_text(encoding="utf8")) if slug_file.exists() else {}

    def slug(e):
        clean = lambda text: re.sub(r"[～~\s]", "", forms(text)[0])
        return slugs.get(clean(e["expr"]) + "|" + clean(e["reading"]), "")

    data = [
        [e["expr"], e["reading"], e["gloss"], capsule(e["gloss"]), e["level"], best if best != UNRANKED else 0, slug(e)]
        for _, best, e in words
    ]
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    html = (ROOT / "src" / "app.html").read_text(encoding="utf8").replace("/*WORDS*/[]", blob)
    html = html.replace("/*BUILD*/", hashlib.sha1(html.encode()).hexdigest()[:7])   # same page, same stamp
    (ROOT / "index.html").write_text(html, encoding="utf8")

    counts = {n: sum(1 for w in data if w[4] == n) for n in range(5, 0, -1)}
    ranked = sum(1 for w in data if w[5])
    linked = sum(1 for w in data if w[6])
    print(f"{len(data)} words {counts}; {ranked} ranked; {linked} slugs; {len(html) // 1024} KB")


if __name__ == "__main__":
    main()
