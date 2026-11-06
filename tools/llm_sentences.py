#!/usr/bin/env python3
"""LLM pass over data/candidates.json: pick the 5 best sentences for each word's sense, and
generate sentences for words Tatoeba can't cover. Runs through the Batch API.

  .venv/bin/python tools/llm_sentences.py estimate      # counts and cost, no API calls
  .venv/bin/python tools/llm_sentences.py submit [N]    # send (first N words, spread across levels)
  .venv/bin/python tools/llm_sentences.py collect       # wait, then write data/llm.json

Needs ANTHROPIC_API_KEY. Words already in data/llm.json are skipped, so a sample can be followed
by a full run without paying twice. tools/export_sentences.py applies the results.
"""
import json
import sys
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from sentences import load_words

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES, RESULTS, BATCH = ROOT / "data" / "candidates.json", ROOT / "data" / "llm.json", ROOT / "data" / "llm_batch.txt"
MODEL = "claude-opus-5-5"
INPUT_PER_M, OUTPUT_PER_M = 2.00, 10.00   # Batch API: half of $4 / $20
OFFERED = 20
WANTED = 5

SYSTEM = """You choose example sentences for a Japanese vocabulary drill used by an English-speaking learner.

For the given word you receive numbered candidate sentences from Tatoeba, with the target marked 【like this】. Candidates were matched by a morphological analyzer, so the word form is usually right but the meaning may not be.

Keep a candidate only if the marked text really is this word, in the sense given by the gloss, used naturally. Reject other words that share the spelling, set phrases where the meaning has drifted (気をつける for つける "turn on", やる気 for やる "give"), and compounds. Prefer short, everyday sentences whose other vocabulary is easier than the word's JLPT level, and whose context makes the meaning guessable. If the gloss lists several senses, cover more than one when the candidates allow.

List the kept candidates best first, at most 5. If fewer than 5 are good, write new sentences to reach 5 in total: natural Japanese of 8 to 30 characters at or below the word's JLPT level, each with a natural English translation, and `target` set to the exact substring of the sentence that is the word (inflected if needed). Never keep a weak candidate just to avoid writing a new one."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["keep", "written"],
    "properties": {
        "keep": {"type": "array", "items": {"type": "integer"}},
        "written": {"type": "array", "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["japanese", "english", "target"],
            "properties": {"japanese": {"type": "string"}, "english": {"type": "string"}, "target": {"type": "string"}},
        }},
    },
}


def prompt(word, candidates):
    lines = [f"Word: {word['word']} ({word['readings'][0]})", f"Gloss: {word['gloss']}", f"JLPT level: N{word['level']}", ""]
    for i, c in enumerate(candidates[:OFFERED]):
        marked = c["text"][:c["at"]] + "【" + c["text"][c["at"]:c["end"]] + "】" + c["text"][c["end"]:]
        lines.append(f"{i}. {marked} — {c['english'][0]}")
    if not candidates:
        lines.append("(no candidates: write all 5)")
    return "\n".join(lines)


def pending():
    candidates = json.loads(CANDIDATES.read_text(encoding="utf8"))
    done = json.loads(RESULTS.read_text(encoding="utf8")) if RESULTS.exists() else {}
    return [(w, candidates[w["id"]]) for w in load_words() if w["id"] not in done], done


def request(index, word, candidates):
    return Request(custom_id=f"w{index}", params=MessageCreateParamsNonStreaming(
        model=MODEL,
        max_tokens=4000,   # thinking counts toward this
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt(word, candidates)}],
    ))


def estimate():
    work, done = pending()
    chars = sum(len(prompt(w, c)) + len(SYSTEM) for w, c in work)
    missing = sum(max(0, WANTED - len(c)) for _, c in work)
    # Rough: mixed Japanese and English runs near 2 characters per token. Output is thinking at
    # low effort, plus the JSON; 100 tokens per written sentence.
    input_tokens = chars / 2
    output_tokens = len(work) * 400 + missing * 100
    cost = input_tokens / 1e6 * INPUT_PER_M + output_tokens / 1e6 * OUTPUT_PER_M
    print(f"{len(work)} words to do, {len(done)} done; at least {missing} sentences to write")
    print(f"about {input_tokens / 1e6:.1f}M input and {output_tokens / 1e6:.1f}M output tokens")
    print(f"about ${cost:.0f} on {MODEL} via Batch, before prompt caching")


def submit(limit):
    work, _ = pending()
    if limit:
        step = max(1, len(work) // limit)
        work = work[::step][:limit]
    ids = {f"w{i}": w["id"] for i, (w, _) in enumerate(work)}
    batch = anthropic.Anthropic().messages.batches.create(requests=[request(i, w, c) for i, (w, c) in enumerate(work)])
    BATCH.write_text(json.dumps({"batch": batch.id, "ids": ids}), encoding="utf8")
    print(f"batch {batch.id}: {len(work)} words")


def collect():
    state = json.loads(BATCH.read_text(encoding="utf8"))
    client = anthropic.Anthropic()
    while True:
        batch = client.messages.batches.retrieve(state["batch"])
        if batch.processing_status == "ended":
            break
        print(f"{batch.request_counts.processing} still processing", flush=True)
        time.sleep(60)
    done = json.loads(RESULTS.read_text(encoding="utf8")) if RESULTS.exists() else {}
    failed = 0
    for result in client.messages.batches.results(state["batch"]):
        message = result.result.message if result.result.type == "succeeded" else None
        text = next((b.text for b in message.content if b.type == "text"), "") if message else ""
        if not message or message.stop_reason != "end_turn" or not text:
            failed += 1
            continue
        done[state["ids"][result.custom_id]] = json.loads(text)
    RESULTS.write_text(json.dumps(done, ensure_ascii=False), encoding="utf8")
    print(f"{len(done)} words in {RESULTS.name}; {failed} failed this batch (run submit again to retry)")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "estimate"
    if command == "estimate":
        estimate()
    elif command == "submit":
        submit(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    elif command == "collect":
        collect()
    else:
        sys.exit(__doc__)
