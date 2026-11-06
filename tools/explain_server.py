#!/usr/bin/env python3
"""Grammar explainer for the vocab drill: a tiny local server that asks `claude -p`.

    python3 tools/explain_server.py          # listens on 127.0.0.1:10102

The page POSTs {"sentence": ..., "start": i, "end": j} (offsets optional) and gets the
explanation back as plain text, streamed as it is written. The prompt is built here, so a
caller can only ask about a short Japanese sentence; Claude runs with every tool turned off,
in an empty directory, without saving the session.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 10102
MAX_SENTENCE = 300
# file:// pages send Origin "null"; localhost covers the page served by a dev server.
ORIGINS = {"null"} | {f"http://{h}{p}" for h in ("localhost", "127.0.0.1") for p in ("", ":8000", ":8771")}
SYSTEM = (
    "You help a Japanese learner around JLPT N2 read sentences. Explain concisely, in English, "
    "what the marked words are (grammar pattern, conjugation or word) and what they mean in this sentence. "
    "Name the dictionary form or pattern (e.g. 〜うちに) with its reading, give a natural translation of the "
    "sentence, and mention a likely confusion only if there is one. Plain text, no markdown, no preamble, "
    "at most about 80 words."
)
CLAUDE = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
WORKDIR = tempfile.mkdtemp(prefix="vocab-explain-")   # no CLAUDE.md or project settings to pick up
cache, cache_lock = {}, threading.Lock()


def question(body):
    sentence = body.get("sentence")
    if not isinstance(sentence, str) or not sentence.strip() or len(sentence) > MAX_SENTENCE:
        raise ValueError("sentence missing or too long")
    start, end = body.get("start"), body.get("end")
    if start is None:
        return "Sentence: " + sentence + "\n\nExplain the grammar of this sentence."
    if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(sentence)):
        raise ValueError("bad selection")
    marked = sentence[:start] + "<<" + sentence[start:end] + ">>" + sentence[end:]
    return "Sentence: " + marked + "\n\nExplain the part marked <<like this>>."


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"   # stream until close; no chunking needed

    def cors(self):
        origin = self.headers.get("Origin")
        if origin in ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.send_header("Access-Control-Allow-Private-Network", "true")
        return origin in ORIGINS or origin is None

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.end_headers()

    def fail(self, code, why):
        self.send_response(code)
        self.cors()
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(why.encode())

    def do_POST(self):
        if self.path != "/explain":
            return self.fail(404, "not found")
        if self.headers.get("Origin") not in ORIGINS and self.headers.get("Origin") is not None:
            return self.fail(403, "origin not allowed")
        try:
            body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 4096)))
            prompt = question(body)
        except (ValueError, TypeError, AttributeError) as e:
            return self.fail(400, str(e))
        with cache_lock:
            hit = cache.get(prompt)
        self.send_response(200)
        self.cors()
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if hit:
            self.wfile.write(hit.encode())
            return
        proc = subprocess.Popen(
            [CLAUDE, "-p", "--tools", "", "--strict-mcp-config", "--no-session-persistence",
             "--effort", "low", "--system-prompt", SYSTEM,
             "--output-format", "stream-json", "--include-partial-messages", "--verbose"],
            cwd=WORKDIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        proc.stdin.write(prompt)
        proc.stdin.close()
        text = ""
        try:
            for line in proc.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("type") == "stream_event":
                    e = event["event"]
                    if e.get("type") == "content_block_delta" and e["delta"].get("type") == "text_delta":
                        text += e["delta"]["text"]
                        self.wfile.write(e["delta"]["text"].encode())
                        self.wfile.flush()
                elif event.get("type") == "result" and event.get("is_error"):
                    self.wfile.write(("\n[claude: " + str(event.get("result")) + "]").encode())
            if proc.wait() == 0 and text:
                with cache_lock:
                    cache[prompt] = text
        except (BrokenPipeError, ConnectionResetError):
            pass   # the page moved on
        finally:
            if proc.poll() is None:
                proc.kill()

    def log_message(self, fmt, *args):
        sys.stderr.write("explain: " + fmt % args + "\n")


if __name__ == "__main__":
    print(f"explainer on http://127.0.0.1:{PORT} using {CLAUDE}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
