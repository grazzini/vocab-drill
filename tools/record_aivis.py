#!/usr/bin/env python3
"""Record audio-<voice>/<hash>.m4a with AivisSpeech: tools/record_aivis.py 5 4 3 2 1 [--voice morioki] [--jobs 4] [--limit N]

Needs the AivisSpeech app open (its engine at 127.0.0.1:10101). Speaks the same texts as
record_audio.py, under the same names, and fixes misread kanji the way the app does live
(respell in src/app.html), so a recording sounds like what the Mac says. Clips are made at
speed 1; the app sets the playback rate. Finished clips are skipped, so a stopped run resumes.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_audio import ROOT, clip_name, texts_for  # noqa: E402

ENGINE = "http://127.0.0.1:10101"
VOICES = {"morioki": ("morioki", "ノーマル")}   # folder name: (engine speaker name, style name)


def post(path, body=None):
    req = urllib.request.Request(ENGINE + path, data=body, method="POST",
                                 headers={"content-type": "application/json"} if body else {})
    with urllib.request.urlopen(req, timeout=120) as res:
        return res.read()


def style_id(speaker, style):
    with urllib.request.urlopen(ENGINE + "/speakers", timeout=10) as res:
        for s in json.load(res):
            for st in s["styles"] if s["name"] == speaker else []:
                if st["name"] == style:
                    return st["id"]
    sys.exit(f"AivisSpeech has no {speaker} ({style})")


# ---- reading fixes, as in the app (groupsFor, loose, respell) ----

def load_readings():
    js = (ROOT / "sentences.js").read_text(encoding="utf8")
    data = json.loads(js[js.index("{"):js.rindex("}") + 1])
    rows = {text: {"transcription": tr, "trust_all": sid == 0, "spans": []} for sid, text, tr, *_ in data["s"]}
    for refs in data["w"].values():
        for index, at, length in refs:
            rows[data["s"][index][1]]["spans"].append((at, at + length))
    return rows


READINGS = load_readings()
GROUP = re.compile(r"\[([^|\]]+)\|([^\]]*)\]|([^\[]+|\[)")


def groups_for(text):
    row = READINGS.get(text)
    if not row:
        return None
    groups, pos = [], 0
    for m in GROUP.finditer(row["transcription"]):
        base = m.group(3) if m.group(3) is not None else m.group(1)
        start = pos
        pos += len(base)
        if m.group(3) is not None:
            groups.append({"base": base})
        else:
            groups.append({"base": base, "rt": m.group(2).replace("|", ""),
                           "trusted": row["trust_all"] or any(start < end and pos > at for at, end in row["spans"])})
    return groups if "".join(g["base"] for g in groups) == text else None


LOOSE = {"ヂ": "シ", "ヅ": "ス", "ウ": "オ", "イ": "エ", "ハ": "ワ", "ヘ": "エ", "ヲ": "オ"}


def loose(s):
    s = re.sub(r"[ぁ-ゖ]", lambda m: chr(ord(m.group()) + 0x60), s)
    out = []
    for c in s:
        if re.match(r"[ァ-ヶ]", c):
            c = LOOSE.get(c) or unicodedata.normalize("NFD", c)[0]
            out.append(LOOSE.get(c) or c)
    return out


def respell(text, query):
    groups = groups_for(text)
    if not groups:
        return None
    heard = loose("".join(m["text"] for p in query["accent_phrases"] for m in p["moras"]))
    ours = [(c, i) for i, g in enumerate(groups) for c in loose(g.get("rt") or g["base"])]
    if "".join(c for c, _ in ours) == "".join(heard):
        return None
    n, m = len(ours), len(heard)
    d = [[a] + [0 if a else b for b in range(1, m + 1)] for a in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(1, m + 1):
            d[a][b] = min(d[a - 1][b] + 1, d[a][b - 1] + 1, d[a - 1][b - 1] + (ours[a - 1][0] != heard[b - 1]))
    wrong, a, b = set(), n, m
    while a > 0 or b > 0:
        if a and b and d[a][b] == d[a - 1][b - 1] + (ours[a - 1][0] != heard[b - 1]):
            if ours[a - 1][0] != heard[b - 1]:
                wrong.add(ours[a - 1][1])
            a, b = a - 1, b - 1
        elif a and d[a][b] == d[a - 1][b] + 1:
            wrong.add(ours[a - 1][1])
            a -= 1
        else:
            if a:
                wrong.add(ours[a - 1][1])
            b -= 1
    fix = {i for i in wrong if groups[i].get("trusted")}
    if not fix:
        return None
    return "".join(g["rt"] if i in fix else g["base"] for i, g in enumerate(groups))


# ---- recording ----

def query_for(text, style):
    return json.loads(post(f"/audio_query?speaker={style}&text={urllib.parse.quote(text)}"))


def record(text, style, out):
    target = out / clip_name(text)
    if target.exists():
        return True
    try:
        query = query_for(text, style)
        spoken = respell(text, query)
        if spoken:
            query = query_for(spoken, style)
        query["speedScale"] = 1
        wav_bytes = post(f"/synthesis?speaker={style}", json.dumps(query).encode())
        with tempfile.TemporaryDirectory() as tmp:
            wav, m4a = Path(tmp) / "t.wav", Path(tmp) / "t.m4a"
            wav.write_bytes(wav_bytes)
            subprocess.run(["afconvert", str(wav), str(m4a), "-d", "aac", "-f", "m4af", "-b", "32000"],
                           check=True, capture_output=True, timeout=60)
            os.replace(m4a, target)
        return True
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("levels", nargs="+", type=int)
    parser.add_argument("--voice", default="morioki", choices=VOICES)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    style = style_id(*VOICES[args.voice])
    out = ROOT / f"audio-{args.voice}"
    out.mkdir(exist_ok=True)
    texts = texts_for(set(args.levels))[:args.limit]
    todo = [t for t in texts if not (out / clip_name(t)).exists()]
    print(f"{len(texts)} clips, {len(todo)} to record", flush=True)
    failed = []
    with ThreadPoolExecutor(args.jobs) as pool:
        for i, (text, ok) in enumerate(zip(todo, pool.map(lambda t: record(t, style, out), todo)), 1):
            if not ok:
                failed.append(text)
            if i % 250 == 0:
                print(f"{i}/{len(todo)}, {len(failed)} failed", flush=True)
    print(f"done: {len(todo) - len(failed)} recorded, {len(failed)} failed", flush=True)
    for text in failed:
        print("FAILED", text)


if __name__ == "__main__":
    main()
