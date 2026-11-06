#!/usr/bin/env python3
"""Write sentences.js, which index.html loads, from data/candidates.json and data/llm.json.

Words the LLM has seen get its choices; other words get the top candidates. Any word still short of
5 is topped up from data/written/*.tsv (word id, Japanese, target substring, English), checked here:
the target must occur in the sentence. Sentences without usable furigana from
Tatoeba get it from Sudachi. Sentences are stored once and referenced by each word that uses them.
"""
import json
import re
from pathlib import Path

from sudachipy import Dictionary, SplitMode

from sentences import char_readings, hira, load_words

ROOT = Path(__file__).resolve().parent.parent
WANTED = 5
KANJI = re.compile(r"[㐀-鿿々]")
tokenizer = Dictionary(dict="core").create()


def markup(text):
    """[漢字|かんじ]-style furigana, leaving okurigana outside the brackets."""
    out = []
    for m in tokenizer.tokenize(text, SplitMode.C):
        surface, reading = m.surface(), hira(m.reading_form())
        if not KANJI.search(surface):
            out.append(surface)
            continue
        tail = ""
        while surface and reading and surface[-1] == reading[-1] and not KANJI.search(surface[-1]):
            tail = surface[-1] + tail
            surface, reading = surface[:-1], reading[:-1]
        out.append(f"[{surface}|{reading}]{tail}")
    return "".join(out)


def markup_target(text, at, target, word):
    """Like markup, but the target takes the word's own reading: Sudachi would read 音 as おと for
    the word ね. Inflected targets don't equal a form, and keep Sudachi's reading. So does a target
    that is only part of a token: forcing 前 in 午前 would leave 午 to be read on its own."""
    bare = re.sub(r"[～~]", "", target)
    if target not in [re.sub(r"[～~]", "", f) for f in word["forms"]] or not KANJI.search(target):
        return markup(text)
    edges = {edge for m in tokenizer.tokenize(text, SplitMode.C) for edge in (m.begin(), m.end())}
    if at not in edges or at + len(target) not in edges:
        return markup(text)
    reading = hira(re.sub(r"\s*[(（][^)）]*[)）]|[～~]", "", word["readings"][0]).strip())
    tail = ""
    while bare and reading and bare[-1] == reading[-1] and not KANJI.search(bare[-1]):
        tail, bare, reading = bare[-1] + tail, bare[:-1], reading[:-1]
    return markup(text[:at]) + f"[{bare}|{reading}]{tail}" + markup(text[at + len(target):])


def main():
    candidates = json.loads((ROOT / "data" / "candidates.json").read_text(encoding="utf8"))
    path = ROOT / "data" / "llm.json"
    llm = json.loads(path.read_text(encoding="utf8")) if path.exists() else {}
    extra = {}
    for file in sorted((ROOT / "data" / "written").glob("*.tsv")):
        for line in file.read_text(encoding="utf8").splitlines():
            if line.strip():
                wid, japanese, target, english = line.split("\t")
                extra.setdefault(wid, []).append(dict(japanese=japanese, target=target, english=english))
    table, index, words = [], {}, {}
    written = rejected = 0

    def add(sid, text, hrkt, english, marked=None):
        key = (sid, text)
        if key not in index:
            index[key] = len(table)
            table.append([sid or 0, text, marked or (hrkt if char_readings(text, hrkt) else markup(text)), english])
        return index[key]

    for w in load_words():
        pool, chosen = candidates[w["id"]], []
        result = llm.get(w["id"])
        picks = [pool[i] for i in result["keep"] if 0 <= i < len(pool)][:WANTED] if result else pool[:WANTED]
        for c in picks:
            chosen.append([add(c["id"], c["text"], c["hrkt"], c["english"][0]), c["at"], c["end"] - c["at"]])
        for s in (result["written"] if result else []) + extra.get(w["id"], []):
            if len(chosen) >= WANTED:
                break
            at = s["japanese"].find(s["target"])
            if at < 0 or not s["target"]:
                rejected += 1
                continue
            marked = markup_target(s["japanese"], at, s["target"], w)
            chosen.append([add(None, s["japanese"], None, s["english"], marked), at, len(s["target"])])
            written += 1
        if chosen:
            words[w["id"]] = chosen

    body = json.dumps({"s": table, "w": words}, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    (ROOT / "sentences.js").write_text("window.SENTENCES=" + body + ";\n", encoding="utf8")
    full = sum(len(v) >= WANTED for v in words.values())
    print(f"{len(words)} words with sentences, {full} with {WANTED}; {len(table)} distinct sentences; "
          f"{written} LLM-written, {rejected} rejected; {(ROOT / 'sentences.js').stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
