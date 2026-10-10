"""
Bhasha Setu: Hindi to Santali translation with teacher review.

Put these next to this file:
    index.html          the web page
    lora_santali/       the folder you downloaded from Kaggle (adapter_config.json + adapter_model.safetensors)
    vocab/              optional: ho_hindi_pairs_raw.csv and mundari_pairs_raw.csv

Run:
    python app.py              start the app with the real model
    python app.py --selftest   load the model, translate two sentences, and exit
    python app.py --mock       start with fake translations (to try the interface without the model)

Then open http://localhost:8000
"""
import csv
import difflib
import json
import os
import re
import sqlite3
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "bhasha_setu.db"
ADAPTER_DIR = HERE / "lora_santali"
VOCAB_DIR = HERE / "vocab"
BASE_MODEL = "facebook/nllb-200-distilled-600M"
PORT = 8000
MOCK = "--mock" in sys.argv or os.environ.get("BS_MOCK") == "1"

# Update these two numbers after you retrain. Keep the note honest about what was measured.
BENCHMARK = {
    "base": 28.67,
    "tuned": 32.22,
    "note": "chrF on 100 FLORES-200 test sentences that the model never trained on",
}


# ---------------------------------------------------------------- translation model
class Translator:
    """Hindi -> Santali using NLLB-200 plus your LoRA adapter, with a confidence score."""

    def __init__(self, mock=False):
        self.mock = mock
        self.lock = threading.Lock()
        if mock:
            self.label = "demo mode (fake translations)"
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL, src_lang="hin_Deva")
        model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
        if ADAPTER_DIR.exists():
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(ADAPTER_DIR)).merge_and_unload()
            self.label = "fine-tuned"
        else:
            print("WARNING: folder 'lora_santali' not found next to app.py. Using the base model without fine-tuning.")
            self.label = "base model (not fine-tuned)"
        self.model = model.eval()
        # Language tags are placed by hand, exactly as in training.
        self.src_id = self.tok.convert_tokens_to_ids("hin_Deva")
        self.tgt_id = self.tok.convert_tokens_to_ids("sat_Beng")
        self.eos = self.tok.eos_token_id
        unk = self.tok.unk_token_id
        if unk in (self.src_id, self.tgt_id):
            raise RuntimeError("Tokenizer does not know the hin_Deva / sat_Beng language tags.")

    def translate(self, text):
        """Return (santali_text, confidence between 0 and 1)."""
        if self.mock:
            return "[demo] " + text, 0.35 + (sum(map(ord, text)) % 55) / 100
        torch = self.torch
        body = self.tok(text, add_special_tokens=False, truncation=True, max_length=126)["input_ids"]
        x = torch.tensor([[self.src_id] + body + [self.eos]])
        with self.lock, torch.no_grad():
            out = self.model.generate(
                input_ids=x,
                attention_mask=torch.ones_like(x),
                forced_bos_token_id=self.tgt_id,
                max_length=160,
                num_beams=1,
                no_repeat_ngram_size=3,
                output_scores=True,
                return_dict_in_generate=True,
            )
            steps = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
        santali = self.tok.decode(out.sequences[0], skip_special_tokens=True)
        steps = steps[1:]  # the first step is the forced language tag, which tells us nothing
        conf = float(torch.exp(steps.mean())) if len(steps) else 0.0
        if conf != conf:  # NaN guard
            conf = 0.0
        return santali, conf


_translator = None
_translator_lock = threading.Lock()


def get_translator():
    global _translator
    with _translator_lock:
        if _translator is None:
            _translator = Translator(mock=MOCK)
    return _translator


# ---------------------------------------------------------------- database
SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    hindi      TEXT NOT NULL UNIQUE,
    draft      TEXT NOT NULL,
    final      TEXT,
    confidence REAL NOT NULL,
    model      TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'pending',
    edited     INTEGER NOT NULL DEFAULT 0,
    edit_ratio REAL NOT NULL DEFAULT 0,
    created    REAL NOT NULL,
    approved   REAL
);
"""


@contextmanager
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


with db() as _con:
    _con.executescript(SCHEMA)


class ApiError(Exception):
    def __init__(self, status, msg):
        super().__init__(msg)
        self.status, self.msg = status, msg


def add_translation(text):
    hindi = " ".join(str(text or "").split())
    if not hindi:
        raise ApiError(400, "Enter a Hindi sentence to translate.")
    if len(hindi) > 400:
        raise ApiError(400, "That sentence is too long. Keep each line under 400 characters.")
    with db() as con:
        row = con.execute("SELECT * FROM items WHERE hindi=?", (hindi,)).fetchone()
    if row:  # already translated: reuse it, which also makes repeat demos instant
        return {**dict(row), "cached": True}
    tr = get_translator()
    draft, conf = tr.translate(hindi)
    with db() as con:
        con.execute(
            "INSERT OR IGNORE INTO items (hindi, draft, confidence, model, created) VALUES (?,?,?,?,?)",
            (hindi, draft, conf, tr.label, time.time()),
        )
        row = con.execute("SELECT * FROM items WHERE hindi=?", (hindi,)).fetchone()
    return {**dict(row), "cached": False}


def get_queue():
    with db() as con:
        rows = con.execute(
            "SELECT * FROM items WHERE status='pending' ORDER BY confidence ASC, id ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def approve_item(item_id, final):
    final = str(final or "").strip()
    if not final:
        raise ApiError(400, "The Santali text is empty. Add a translation before approving.")
    with db() as con:
        row = con.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
        if not row:
            raise ApiError(404, "That sentence no longer exists. Reload the page.")
        edited = int(final != row["draft"].strip())
        ratio = 0.0
        if edited:
            ratio = 1 - difflib.SequenceMatcher(None, row["draft"], final).ratio()
        con.execute(
            "UPDATE items SET final=?, status='approved', edited=?, edit_ratio=?, approved=? WHERE id=?",
            (final, edited, ratio, time.time(), item_id),
        )
    return {"id": item_id, "edited": bool(edited), "edit_ratio": ratio}


def get_approved():
    with db() as con:
        rows = con.execute(
            "SELECT id, hindi, final FROM items WHERE status='approved' ORDER BY approved DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_stats():
    with db() as con:
        one = lambda sql: con.execute(sql).fetchone()[0]
        return {
            "total": one("SELECT COUNT(*) FROM items"),
            "pending": one("SELECT COUNT(*) FROM items WHERE status='pending'"),
            "approved": one("SELECT COUNT(*) FROM items WHERE status='approved'"),
            "edited": one("SELECT COUNT(*) FROM items WHERE status='approved' AND edited=1"),
            "avg_conf": one("SELECT AVG(confidence) FROM items"),
            "conf_edited": one("SELECT AVG(confidence) FROM items WHERE status='approved' AND edited=1"),
            "conf_kept": one("SELECT AVG(confidence) FROM items WHERE status='approved' AND edited=0"),
            "avg_edit_ratio": one("SELECT AVG(edit_ratio) FROM items WHERE status='approved' AND edited=1"),
            "benchmark": BENCHMARK,
            "model": get_translator().label,
        }


def export_jsonl():
    with db() as con:
        rows = con.execute("SELECT hindi, final, edited FROM items WHERE status='approved' ORDER BY id").fetchall()
    lines = [
        json.dumps({"hindi": r["hindi"], "santali": r["final"], "teacher_edited": bool(r["edited"])}, ensure_ascii=False)
        for r in rows
    ]
    return ("\n".join(lines) + "\n") if lines else ""


# ---------------------------------------------------------------- Ho / Mundari word lists
def load_vocab():
    entries = []
    for filename, lang in (("ho_hindi_pairs_raw.csv", "Ho"), ("mundari_pairs_raw.csv", "Mundari")):
        path = VOCAB_DIR / filename
        if not path.exists():
            continue
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                word, hindi = (row.get("target") or "").strip(), (row.get("hindi") or "").strip()
                if len(hindi) < 2:
                    continue
                if len(word) < 2 and len(hindi) <= 3:  # alphabet-chart debris like इ / ना; real one-letter words have a longer Hindi meaning
                    continue
                if re.search(r"[A-Za-z0-9\u0966-\u096F\u0964]", word + hindi):  # latin, digits, danda: not a word
                    continue
                if " " in word or " " in hindi:  # sentence fragments, not single words
                    continue
                entries.append({"lang": lang, "word": word, "hindi": hindi})
    return entries


VOCAB = load_vocab()


def lookup(q):
    q = (q or "").strip()
    results = [e for e in VOCAB if q in e["hindi"] or q in e["word"]][:30] if q else []
    return {"total_words": len(VOCAB), "results": results}


# ---------------------------------------------------------------- web server
# The service worker keeps the page itself available when the server is unreachable.
SW_JS = """
const CACHE = 'bhasha-setu-v1';
self.addEventListener('install', e => e.waitUntil(
  caches.open(CACHE).then(c => c.add('/')).then(() => self.skipWaiting())));
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', e => {
  const u = new URL(e.request.url);
  if (e.request.method !== 'GET' || u.pathname !== '/') return;
  e.respondWith(fetch(e.request)
    .then(r => { const copy = r.clone(); caches.open(CACHE).then(c => c.put('/', copy)); return r; })
    .catch(() => caches.match('/')));
});
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "BhashaSetu/1.0"

    def log_message(self, *args):  # keep the terminal quiet
        pass

    def _send(self, status, body, ctype, headers=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 100_000:
            raise ApiError(413, "That request is too large.")
        raw = self.rfile.read(n) if n else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except ValueError:
            raise ApiError(400, "The request body is not valid JSON.")
        if not isinstance(data, dict):
            raise ApiError(400, "The request body must be a JSON object.")
        return data

    def _route(self, method):
        url = urlparse(self.path)
        path, qs = url.path, parse_qs(url.query)
        try:
            if method == "GET":
                if path == "/":
                    return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
                if path == "/sw.js":
                    return self._send(200, SW_JS.encode("utf-8"), "application/javascript; charset=utf-8")
                if path == "/favicon.ico":
                    return self._send(204, b"", "image/x-icon")
                if path == "/api/health":
                    return self._json({"model": get_translator().label})
                if path == "/api/queue":
                    return self._json(get_queue())
                if path == "/api/approved":
                    return self._json(get_approved())
                if path == "/api/stats":
                    return self._json(get_stats())
                if path == "/api/lookup":
                    return self._json(lookup((qs.get("q") or [""])[0]))
                if path == "/api/export":
                    return self._send(
                        200, export_jsonl().encode("utf-8"), "application/x-ndjson; charset=utf-8",
                        {"Content-Disposition": "attachment; filename=corrections.jsonl"},
                    )
            elif method == "POST":
                if path == "/api/translate":
                    return self._json(add_translation(self._read_json().get("text")))
                m = re.fullmatch(r"/api/approve/(\d+)", path)
                if m:
                    return self._json(approve_item(int(m.group(1)), self._read_json().get("final")))
            raise ApiError(404, "Not found.")
        except ApiError as e:
            self._json({"detail": e.msg}, e.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            traceback.print_exc()
            self._json({"detail": "Something went wrong on the server. The terminal window shows the details."}, 500)

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")


def selftest():
    tr = get_translator()
    for sentence in ["बच्चे स्कूल जाते हैं", "मेरा नाम राम है"]:
        t0 = time.time()
        santali, conf = tr.translate(sentence)
        print(f"{sentence}\n  -> {santali}\n  confidence {conf:.2f}   {time.time() - t0:.1f}s")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("Loading the model. The first run downloads about 2.4 GB, so this can take a few minutes..." if not MOCK
          else "Starting in demo mode (no model).")
    tr = get_translator()
    print("Model ready:", tr.label)
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Open http://localhost:{PORT}   (press Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
