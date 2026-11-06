#!/usr/bin/env python3
"""Resolve each word's Jisho page slug into src/slugs.json.

Usage: python3 -I tools/slugs.py [word|reading ...]

Plain /word/<text> URLs 404 for some words and pick the wrong entry for
homographs (中 as ちゅう lands on なか), so slugs are matched on word and
reading through Jisho's API. The API has no CORS headers, so this runs at
build time. Results are saved as it goes; rerun to resume. Words with no
match stay "" and the app links them to a Jisho search.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "slugs.json"
PAUSE = 0.25
SAVE_EVERY = 50
STOP = {"to", "a", "an", "the", "of", "in", "on", "be", "or", "and", "for", "at", "as", "with", "something", "someone", "one"}


def clean(text):
    return re.sub(r"[～~\s]", "", re.split("[;；]", text)[0])


def all_forms(text):
    return [clean(f) for f in re.split("[;；]", text) if clean(f)]


def is_kana(text):
    return re.fullmatch(r"[぀-ヿー]+", text) is not None


def tokens(text):
    return set(re.findall(r"[a-z]+", text.lower())) - STOP


def score(entry, gloss, kana_word):
    """Prefer the entry whose definitions share words with our gloss."""
    senses = entry.get("senses", [])
    defs = " ".join(d for sense in senses for d in sense.get("english_definitions", []))
    points = len(tokens(gloss) & tokens(defs))
    if kana_word and senses and "Usually written using kana alone" in senses[0].get("tags", []):
        points += 0.5
    return points


def choose(entries, forms, readings, gloss):
    def spelled(j):
        return j.get("word") in forms

    def read(j):
        return j.get("reading") in readings

    kana_word = all(is_kana(f) for f in forms)
    tests = [
        lambda j: spelled(j) and read(j),
        lambda j: kana_word and read(j),
        spelled,
    ]
    for test in tests:
        matches = [e for e in entries if any(test(j) for j in e["japanese"])]
        if matches:
            return max(matches, key=lambda e: score(e, gloss, kana_word))["slug"]
    return ""


def fetch(keyword):
    url = "https://jisho.org/api/v1/search/words?keyword=" + urllib.parse.quote(keyword)
    request = urllib.request.Request(url, headers={"User-Agent": "vocab-drill build"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.load(response)["data"]
        except Exception as error:
            print("retry", keyword, error, flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def main():
    html = (ROOT / "index.html").read_text(encoding="utf8")
    blob = re.search(r'id="words" type="application/json">(.*?)</script>', html, re.S).group(1)
    words = json.loads(blob)
    slugs = json.loads(OUT.read_text(encoding="utf8")) if OUT.exists() else {}
    only = set(sys.argv[1:])
    failed = 0
    for i, row in enumerate(words):
        key = clean(row[0]) + "|" + clean(row[1])
        if (only and key not in only) or (not only and key in slugs):
            continue
        forms, readings = all_forms(row[0]), all_forms(row[1])
        entries = fetch(forms[0])
        if entries is None:
            failed += 1
            continue
        slugs[key] = choose(entries, forms, readings, row[2])
        if only:
            print(key, "->", slugs[key], flush=True)
        if i % SAVE_EVERY == 0:
            OUT.write_text(json.dumps(slugs, ensure_ascii=False, indent=0), encoding="utf8")
            print(i, len(slugs), flush=True)
        time.sleep(PAUSE)
    OUT.write_text(json.dumps(slugs, ensure_ascii=False, indent=0), encoding="utf8")
    print("done", len(slugs), "unresolved", sum(1 for v in slugs.values() if not v), "failed", failed, flush=True)


if __name__ == "__main__":
    main()
