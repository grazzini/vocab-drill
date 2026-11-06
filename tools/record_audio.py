#!/usr/bin/env python3
"""Record audio/<hash>.m4a for each level's words and sentences: tools/record_audio.py 5 4 [--jobs 6] [--limit N]

Uses `say`, which follows the Japanese voice in System Settings > Spoken Content, and
stops if that isn't VOICE. Finished clips are skipped, so an interrupted run can resume.
The app computes the same hash from the text it is about to speak (audioUrl in src/app.html).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "audio"
VOICE = "gryphon-neural_Hiro_ja-JP_premium"   # shown as "Voice 1"


def clip_name(text):
    """FNV-1a over UTF-16 code units, matching audioUrl in the app."""
    h = 0x811c9dc5
    data = text.encode("utf-16-le")
    for i in range(0, len(data), 2):
        h = ((h ^ int.from_bytes(data[i:i + 2], "little")) * 0x01000193) & 0xffffffff
    return f"{h:08x}.m4a"


def load_words():
    page = (ROOT / "index.html").read_text(encoding="utf8")
    raw = re.search(r'id="words"[^>]*>(.*?)</script>', page, re.S).group(1)
    for row in json.loads(raw):
        form = re.split(r"[;；]", row[0])[0].strip()
        reading = re.split(r"[;；]", row[1])[0].strip()
        sound = re.sub(r"[～~]", "", re.sub(r"\s*[(（][^)）]*[)）]", "", reading)).strip()
        yield form + "|" + reading, sound, row[4]


def texts_for(levels):
    js = (ROOT / "sentences.js").read_text(encoding="utf8")
    data = json.loads(js[js.index("{"):js.rindex("}") + 1])
    texts = set()
    for word_id, sound, level in load_words():
        if level in levels:
            texts.add(sound)
            texts.update(data["s"][ref[0]][1] for ref in data["w"].get(word_id, []))
    texts.discard("")
    return sorted(texts)


def record(text):
    target = OUT / clip_name(text)
    if target.exists():
        return True
    with tempfile.TemporaryDirectory() as tmp:
        source, aiff, m4a = (Path(tmp) / n for n in ("t.txt", "t.aiff", "t.m4a"))
        source.write_text(text, encoding="utf8")
        try:
            subprocess.run(["say", "-f", str(source), "-o", str(aiff)], check=True, capture_output=True, timeout=60)
            subprocess.run(["afconvert", str(aiff), str(m4a), "-d", "aac", "-f", "m4af", "-b", "32000"],
                           check=True, capture_output=True, timeout=60)
        except (subprocess.SubprocessError, OSError):
            return False
        os.replace(m4a, target)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("levels", nargs="+", type=int)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    prefs = subprocess.run(["defaults", "read", "com.apple.Accessibility", "SpokenContentDefaultVoiceSelectionsByLanguage"],
                           capture_output=True, text=True).stdout
    if VOICE not in prefs:
        sys.exit(f"Spoken Content's Japanese voice isn't {VOICE}")
    texts = texts_for(set(args.levels))
    names = {}
    for text in texts:
        if names.setdefault(clip_name(text), text) != text:
            sys.exit(f"hash collision: {text} / {names[clip_name(text)]}")
    texts = texts[:args.limit]
    OUT.mkdir(exist_ok=True)
    todo = [t for t in texts if not (OUT / clip_name(t)).exists()]
    print(f"{len(texts)} clips, {len(todo)} to record", flush=True)
    failed = []
    with ThreadPoolExecutor(args.jobs) as pool:
        for i, (text, ok) in enumerate(zip(todo, pool.map(record, todo)), 1):
            if not ok:
                failed.append(text)
            if i % 250 == 0:
                print(f"{i}/{len(todo)}, {len(failed)} failed", flush=True)
    print(f"done: {len(todo) - len(failed)} recorded, {len(failed)} failed", flush=True)
    for text in failed:
        print("FAILED", text)


if __name__ == "__main__":
    main()
