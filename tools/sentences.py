#!/usr/bin/env python3
"""Find real example sentences for every word, using morphological analysis.

Usage: .venv/bin/python tools/sentences.py

Reads the words from index.html and Tatoeba's exports from data/tatoeba/ (see README in
tools/requirements.txt for the packages), tokenizes every Japanese sentence with Sudachi, and
writes data/candidates.json: up to MAX_CANDIDATES ranked sentences per word. Unlike a substring
search this knows 見つける is not つける, 部屋 is not ～屋, and 温かい is not 温い. It cannot tell
senses apart (気をつける vs 電気をつける), which is what the LLM pass is for.
"""
import bz2
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from sudachipy import Dictionary, SplitMode

ROOT = Path(__file__).resolve().parent.parent
TATOEBA = ROOT / "data" / "tatoeba"
MIN_LEN, MAX_LEN = 6, 42
MAX_CANDIDATES = 30
IRREGULAR = {"来る", "くる", "する", "為る"}
CONTENT = {"名詞", "動詞", "形容詞", "副詞", "形状詞"}
NOT_ATTACHABLE = {"助詞", "補助記号", "空白", "助動詞"}
GLOSS_NOISE = re.compile(r"^(the|and|for|with|from|into|that|this|very|you|your|one|something|someone|somebody|etc)$")
NAMES = re.compile(r"トム|メアリー|ジョン|メアリ")


def rows(name):
    with bz2.open(TATOEBA / name, "rt", encoding="utf8") as f:
        for line in f:
            yield line.rstrip("\n").split("\t")


def hira(s):
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def clean(s):
    return re.sub(r"\s*[(（][^)）]*[)）]", "", s).replace("～", "").replace("~", "").strip()


def load_words():
    html = (ROOT / "index.html").read_text(encoding="utf8")
    start = html.index('<script id="words" type="application/json">') + len('<script id="words" type="application/json">')
    data, _ = json.JSONDecoder().raw_decode(html, start)
    words = []
    for expr, reading, gloss, capsule, level, rank, slug in data:
        forms = [s.strip() for s in re.split(r"[;；]", expr) if s.strip()]
        readings = [s.strip() for s in re.split(r"[;；]", reading) if s.strip()]
        words.append(dict(id=forms[0] + "|" + readings[0], word=forms[0], forms=forms, readings=readings,
                          gloss=gloss, level=level, rank=rank))
    return words


def load_sentences():
    english = {}
    for sid, lang, text in rows("eng_sentences.tsv.bz2"):
        english[sid] = text
    links = defaultdict(list)
    for jpn, eng in rows("jpn-eng_links.tsv.bz2"):
        if eng in english:
            links[jpn].append(english[eng])
    hrkt = {}
    for row in rows("jpn_transcriptions.tsv.bz2"):
        if row[2] == "Hrkt":
            hrkt[row[0]] = row[4]
    out = {}
    for row in rows("jpn_sentences_detailed.tsv.bz2"):
        sid, _, text, owner = row[:4]
        if owner == "\\N" or sid not in links or not MIN_LEN <= len(text) <= MAX_LEN:
            continue
        if any(ord(c) > 0xFFFF for c in text):   # offsets are shared with JavaScript strings
            continue
        out[sid] = dict(text=text, english=links[sid][:3], hrkt=hrkt.get(sid))
    return out


def analyze(tokenizer, text, mode):
    return [(m.begin(), m.end(), m.surface(), m.dictionary_form(), m.normalized_form(),
             hira(m.reading_form()), m.part_of_speech()[0], m.part_of_speech()[1], m.part_of_speech()[2])
            for m in tokenizer.tokenize(text, mode)]


def char_readings(text, transcription):
    """Reading per character from "[学校|がっ|こう]" markup, or None if it doesn't line up.

    A compound with a single reading puts it all on its first character and "" on the rest.
    """
    if not transcription:
        return None
    out = []
    for m in re.finditer(r"\[([^|\]]+)\|([^\]]*)\]|([^\[]+|\[)", transcription):
        if m.group(3) is not None:
            out.extend(hira(c) for c in m.group(3))
            continue
        chars, rts = list(m.group(1)), m.group(2).split("|")
        if len(rts) == len(chars):
            out.extend(rts)
        else:
            out.extend(["".join(rts)] + [""] * (len(chars) - 1))
    return out if len(out) == len(text) else None


def unvoiced(s):
    table = str.maketrans("がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ", "かきくけこさしすせそたちつてとはひふへほはひふへほ")
    return s[:1].translate(table) + s[1:]


def bare_reading(reading, form):
    """Hiragana reading without the する the list adds to 生活 (せいかつする)."""
    r = hira(clean(reading))
    return r[:-2] if r.endswith("する") and not form.endswith("する") and len(r) > 2 else r


def sense_test(word):
    if not re.fullmatch(r"[぀-ゟ]+", clean(word["word"])):
        return None
    text = re.sub(r"\be\.g\.|\bi\.e\.|\bv\.[it]\.", " ", word["gloss"].lower())
    terms = {t.removesuffix("e") for t in re.findall(r"[a-z]{3,}", text) if not GLOSS_NOISE.match(t)}
    return re.compile(r"\b(" + "|".join(sorted(terms)) + ")", re.I) if terms else None


class Matcher:
    def __init__(self, tokenizer, sentences):
        self.tokenizer, self.sentences = tokenizer, sentences
        self.parsed = {mode: {} for mode in "CA"}
        self.chars = {}
        self.index = {mode: defaultdict(list) for mode in "CA"}
        for count, (sid, s) in enumerate(sentences.items()):
            for name, mode in (("C", SplitMode.C), ("A", SplitMode.A)):
                morphs = analyze(tokenizer, s["text"], mode)
                self.parsed[name][sid] = morphs
                for i, m in enumerate(morphs):
                    self.index[name][m[3]].append((sid, i))
                    if m[4] != m[3]:
                        self.index[name][m[4]].append((sid, i))
            if count % 50000 == 0:
                print(f"  tokenized {count}", file=sys.stderr)

    def span_reading(self, sid, morphs):
        """Tatoeba's furigana when it has any: Sudachi knows only one reading per spelling."""
        begin, end = morphs[0][0], morphs[-1][1]
        if sid not in self.chars:
            s = self.sentences[sid]
            self.chars[sid] = char_readings(s["text"], s["hrkt"])
        chars = self.chars[sid]
        return "".join(chars[begin:end]) if chars else "".join(m[5] for m in morphs)

    def reads_as(self, word, morphs, sid):
        if morphs[-1][3] in IRREGULAR:
            return True
        reading = self.span_reading(sid, morphs)
        inflecting = morphs[-1][6] in ("動詞", "形容詞", "助動詞")
        for form in word["forms"]:
            for r in word["readings"]:
                r = bare_reading(r, form)
                if reading.startswith(r[:-1] if inflecting and len(r) > 1 else r):
                    return True
        return False

    def plain(self, word):
        """(sid, begin, end) of every token run whose lemmas equal the word's."""
        forms = {f for f in (clean(x) for x in word["forms"]) if f}
        spoken = {bare_reading(r, "") for r in word["readings"]}
        hits = self.lemma_hits(word, forms)
        # A headword the dictionary can't parse as one word is a typo or a list (楽む, 回る、回す), so
        # fall back on its reading. A real word must not: 磁器 would match the じき of もうじき.
        unparsed = all(len(analyze(self.tokenizer, f, SplitMode.C)) > 1 for f in forms)
        return hits or (self.lemma_hits(word, spoken) if unparsed else {})

    def lemma_hits(self, word, forms):
        hits = {}
        for mode, split in (("C", SplitMode.C), ("A", SplitMode.A)):
            for form in forms:
                pattern = [{m[3], m[4]} for m in analyze(self.tokenizer, form, split)]
                if not pattern:
                    continue
                for key in pattern[0]:
                    for sid, i in self.index[mode].get(key, ()):
                        morphs = self.parsed[mode][sid][i:i + len(pattern)]
                        if len(morphs) == len(pattern) and all({m[3], m[4]} & p for m, p in zip(morphs, pattern)) \
                                and self.reads_as(word, morphs, sid):
                            hits.setdefault(sid, (morphs[0][0], morphs[-1][1]))
            if hits:   # the long (C) reading wins; A only rescues compounds C merged away
                break
        return hits

    def affix(self, word):
        suffix = word["word"][0] in "～~"
        name = clean(word["word"])
        kanji = re.search(r"[㐀-鿿]", name)
        grammar = "(" in word["word"] or "（" in word["word"]   # ～(て) しまう: て is part of it
        readings = {unvoiced(bare_reading(r, name)) for r in word["readings"]}
        hits = {}
        pattern = [m[3] for m in analyze(self.tokenizer, name, SplitMode.A)]
        if not pattern:
            return hits
        for sid, i in self.index["A"].get(pattern[0], ()):
            parsed = self.parsed["A"][sid]
            run = parsed[i:i + len(pattern)]
            if [m[3] for m in run] != pattern and [m[2] for m in run] != pattern:
                continue
            neighbour = parsed[i - 1] if suffix and i else parsed[i + len(pattern)] if not suffix and i + len(pattern) < len(parsed) else None
            if not grammar and (neighbour is None or neighbour[6] in NOT_ATTACHABLE):
                continue
            wanted = "接尾辞" if suffix else "接頭辞"
            counter = suffix and run[0][8] == "助数詞可能" and neighbour and neighbour[7] == "数詞"
            verb = suffix and run[0][6] == "動詞" and neighbour and neighbour[6] == "動詞"
            if kanji and not (run[0][6] == wanted or counter or verb):
                continue
            sentence = self.sentences[sid]
            if char_readings(sentence["text"], sentence["hrkt"]) and not (verb or grammar) \
                    and unvoiced(self.span_reading(sid, run)) not in readings:
                continue
            hits.setdefault(sid, (run[0][0], run[-1][1]))
        return hits


def main():
    words = load_words()
    level_of = {}
    for w in words:
        for f in w["forms"]:
            level_of[clean(f)] = max(level_of.get(clean(f), 0), w["level"])
    print(f"{len(words)} words", file=sys.stderr)
    sentences = load_sentences()
    print(f"{len(sentences)} usable sentences", file=sys.stderr)
    tokenizer = Dictionary(dict="core").create()
    matcher = Matcher(tokenizer, sentences)

    def score(word, sid, sense):
        s = sentences[sid]
        content = [m for m in matcher.parsed["C"][sid] if m[6] in CONTENT]
        hard = sum(1 for m in content if level_of.get(m[3], 0) < word["level"])
        length = abs(len(s["text"]) - 18) / 18
        matches = 1 if sense and any(sense.search(e) for e in s["english"]) else 0
        return (-matches, hard / (len(content) or 1) + length * .5 + (.3 if NAMES.search(s["text"]) else 0)), matches

    result = {}
    for w in words:
        is_affix = w["word"][0] in "～~" or w["word"][-1] in "～~"
        hits = matcher.affix(w) if is_affix else matcher.plain(w)
        sense = sense_test(w)
        ranked, seen = [], set()
        for key, sid in sorted(((score(w, sid, sense), sid) for sid in hits), key=lambda x: x[0][0]):
            bare = re.sub(r"[。、！？!?.,\s]", "", sentences[sid]["text"])   # ですか。 and ですか？ are one sentence
            if bare not in seen:
                seen.add(bare)
                ranked.append((key, sid))
        result[w["id"]] = [dict(id=int(sid), text=sentences[sid]["text"], english=sentences[sid]["english"],
                                hrkt=sentences[sid]["hrkt"], at=hits[sid][0], end=hits[sid][1], sense=key[1])
                           for key, sid in ranked[:MAX_CANDIDATES]]
    (ROOT / "data" / "candidates.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf8")

    print("candidates per word: " + str({n: sum(1 for w in words if (len(result[w["id"]]) >= n if n else not result[w["id"]]))
                                         for n in (0, 1, 3, 5, 15, 30)}))
    for level in range(5, 0, -1):
        group = [w for w in words if w["level"] == level]
        print(f"N{level}: {len(group)} words, {sum(len(result[w['id']]) >= 5 for w in group)} with 5+, "
              f"{sum(not result[w['id']] for w in group)} with none")


if __name__ == "__main__":
    main()
