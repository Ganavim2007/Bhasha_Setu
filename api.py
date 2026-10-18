"""FastAPI API and persistent document-translation worker for Bhasha Setu."""
import json
import difflib
import asyncio
import hashlib
import ipaddress
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import threading
import shutil
import tempfile
import time
import zipfile
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from functools import wraps
from io import BytesIO
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field

import app as core

MAX_DOCUMENT_BYTES = 20 * 1024 * 1024
MAX_DOCUMENT_PAGES = 120
MAX_DOCUMENT_SEGMENTS = 2000
MAX_DOCUMENT_UNPACKED_BYTES = 120 * 1024 * 1024
MAX_TRANSLATED_PDF_PAGES = 500
MAX_AUDIO_BYTES = 5 * 1024 * 1024
MAX_GENERATED_AUDIO_BYTES = 24 * 1024 * 1024
MAX_VISUAL_BYTES = 5 * 1024 * 1024
HF_VISUAL_MODEL = os.environ.get("BS_VISUAL_MODEL", "black-forest-labs/FLUX.1-schnell")
HF_VISUAL_PROVIDER = os.environ.get("BS_VISUAL_PROVIDER", "nscale")
SUPPORTED_LANGUAGE = "Santali"
TEACHER_PIN = os.environ.get("BS_TEACHER_PIN", "")
EXPERT_PIN = os.environ.get("BS_EXPERT_PIN", "")
_reviewer_tokens = {}
_login_attempts = {}
_auth_lock = threading.Lock()
_student_document_uploads = {}
_student_document_upload_lock = threading.Lock()
_stop_worker = threading.Event()
_worker_thread = None
_santali_tts_process = None
_santali_tts_stderr_thread = None
_santali_tts_lock = threading.Lock()
_model_resource_lock = threading.RLock()
_santali_tts_stderr_lock = threading.Lock()
_santali_tts_stderr_tail = []
DB_PATH = core.HERE / "bhasha_setu.db"
AUDIO_DIR = core.HERE / "audio_recordings"
VISUAL_DIR = core.HERE / "lesson_visuals"
PDF_STAGING_DIR = core.HERE / ".pdf_staging"
STUDENT_PDF_CACHE_DIR = core.HERE / ".student_pdf_cache"
_student_pdf_generation_lock = threading.Lock()
DEMO_MODEL = "demo mode (fake translations)"
try:
    REQUIRED_REVIEWERS = int(os.environ.get("BS_REQUIRED_REVIEWERS", "1"))
except ValueError as exc:
    raise RuntimeError("BS_REQUIRED_REVIEWERS must be 1 or 2.") from exc
if REQUIRED_REVIEWERS not in (1, 2):
    raise RuntimeError("BS_REQUIRED_REVIEWERS must be 1 or 2.")


def required_reviewers_for_language(language):
    return REQUIRED_REVIEWERS if str(language or "").casefold() == SUPPORTED_LANGUAGE.casefold() else 2


def _translation_resource_guard(function):
    """Serialize translation against TTS and free the TTS process before translating."""
    @wraps(function)
    def guarded(*args, **kwargs):
        with _model_resource_lock:
            _stop_santali_tts_helper()
            return function(*args, **kwargs)
    return guarded
SERVICE_WORKER_JS = r"""
const CACHE = 'bhasha-setu-v22';
const AUDIO_CACHE = 'bhasha-approved-audio-v1';
const VISUAL_CACHE = 'bhasha-approved-visual-v1';
const STUDENT_PDF_CACHE = 'bhasha-student-pdf-v1';
const SHELL = ['/', '/manifest.webmanifest', '/icon.svg'];
async function cachedRangeResponse(request, fullResponse) {
  const range = /^bytes=(\d*)-(\d*)$/.exec(request.headers.get('range') || '');
  if (!range) return fullResponse.clone();
  const bytes = await fullResponse.arrayBuffer();
  const size = bytes.byteLength;
  let start, end;
  if (!range[1]) {
    const suffixLength = Number(range[2]);
    if (!suffixLength || !size) return new Response(null, {status: 416, headers: {'Content-Range': 'bytes */' + size}});
    start = Math.max(size - suffixLength, 0);
    end = size - 1;
  } else {
    start = Number(range[1]);
    end = range[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
  }
  if (!size || start > end || start >= size) {
    return new Response(null, {status: 416, headers: {'Content-Range': 'bytes */' + size}});
  }
  const headers = new Headers(fullResponse.headers);
  headers.set('Accept-Ranges', 'bytes');
  headers.set('Content-Range', 'bytes ' + start + '-' + end + '/' + size);
  headers.set('Content-Length', String(end - start + 1));
  headers.delete('Content-Encoding');
  return new Response(bytes.slice(start, end + 1), {status: 206, headers});
}
self.addEventListener('install', event => event.waitUntil(
  caches.open(CACHE).then(cache => Promise.allSettled(SHELL.map(path => cache.add(path))))
    .then(() => self.skipWaiting())));
self.addEventListener('activate', event => event.waitUntil((async () => {
  for (const key of await caches.keys()) if (key.startsWith('bhasha-setu-') && key !== CACHE) await caches.delete(key);
  await self.clients.claim();
})()));
self.addEventListener('message', event => {
  const data = event.data || {};
  if (data.type !== 'SYNC_STUDENT_DOCUMENTS' && data.type !== 'DELETE_STUDENT_PDF') return;
  event.waitUntil((async () => {
    const cache = await caches.open(STUDENT_PDF_CACHE);
    const keep = new Set((data.ids || []).map(Number).filter(Number.isFinite));
    for (const request of await cache.keys()) {
      const match = new URL(request.url).pathname.match(/^\/api\/student-documents\/(\d+)\/pdf$/);
      if (!match) continue;
      if (data.type === 'DELETE_STUDENT_PDF' || !keep.has(Number(match[1]))) await cache.delete(request);
    }
  })());
});
self.addEventListener('fetch', event => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== 'GET' || url.origin !== self.location.origin) return;
  const approvedAudio = /^\/api\/(?:approved-audio|language-approved-audio)\/\d+$/.test(url.pathname);
  const approvedVisual = /^\/api\/approved-visuals\/\d+$/.test(url.pathname);
  const studentPdf = /^\/api\/student-documents\/\d+\/pdf$/.test(url.pathname);
  if (studentPdf) {
    const cacheKey = new Request(url.href);
    if (request.headers.has('range')) {
      event.respondWith((async () => {
        try {
          const response = await fetch(request);
          if (response.status === 404) await (await caches.open(STUDENT_PDF_CACHE)).delete(cacheKey);
          return response;
        } catch (_) {
          const cached = await (await caches.open(STUDENT_PDF_CACHE)).match(cacheKey);
          return cached ? cachedRangeResponse(request, cached) : Response.error();
        }
      })());
      event.waitUntil((async () => {
        const cache = await caches.open(STUDENT_PDF_CACHE);
        if (await cache.match(cacheKey)) return;
        try {
          const response = await fetch(cacheKey);
          if (response.status === 200 && response.ok) await cache.put(cacheKey, response.clone());
          else if (response.status === 404) await cache.delete(cacheKey);
        } catch (_) {}
      })());
      return;
    }
    event.respondWith(caches.open(STUDENT_PDF_CACHE).then(async cache => {
      const cached = await cache.match(cacheKey);
      try {
        const response = await fetch(request);
        if (response.ok) {
          try { await cache.put(cacheKey, response.clone()); } catch (_) {}
        }
        else if (response.status === 404) await cache.delete(cacheKey);
        return response;
      } catch (_) { return cached || Response.error(); }
    }));
    return;
  }
  if (approvedAudio || approvedVisual) {
    const cacheName = approvedAudio ? AUDIO_CACHE : VISUAL_CACHE;
    if (approvedAudio && request.headers.has('range')) {
      const cacheKey = new Request(url.href);
      event.respondWith((async () => {
        try {
          const response = await fetch(request);
          if (response.status === 404) await (await caches.open(cacheName)).delete(cacheKey);
          return response;
        } catch (_) {
          const cached = await (await caches.open(cacheName)).match(cacheKey);
          return cached ? cachedRangeResponse(request, cached) : Response.error();
        }
      })());
      event.waitUntil((async () => {
        const cache = await caches.open(cacheName);
        if (await cache.match(cacheKey)) return;
        try {
          const response = await fetch(cacheKey);
          if (response.status === 200 && response.ok) await cache.put(cacheKey, response.clone());
          else if (response.status === 404) await cache.delete(cacheKey);
        } catch (_) {}
      })());
      return;
    }
    event.respondWith(caches.open(cacheName).then(async cache => {
      const cached = await cache.match(request);
      try {
        const response = await fetch(request);
        if (response.ok) await cache.put(request, response.clone());
        else if (response.status === 404) await cache.delete(request);
        return response;
      } catch (_) { return cached || Response.error(); }
    }));
    return;
  }
  if (!SHELL.includes(url.pathname)) return;
  event.respondWith(fetch(request).then(response => {
    if (response.ok) caches.open(CACHE).then(cache => cache.put(url.pathname, response.clone()));
    return response;
  }).catch(async () => (await caches.match(url.pathname)) || Response.error()));
});
self.addEventListener('sync', event => {
  if (event.tag !== 'bhasha-pending-uploads') return;
  event.waitUntil(self.clients.matchAll({type: 'window', includeUncontrolled: true}).then(clients => {
    for (const client of clients) client.postMessage({type: 'BS_FLUSH_UPLOADS'});
  }));
});
"""


@contextmanager
def db():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


with db() as con:
    con.executescript("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT, hindi TEXT NOT NULL UNIQUE,
            draft TEXT NOT NULL, final TEXT, confidence REAL NOT NULL,
            model TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
            edited INTEGER NOT NULL DEFAULT 0, edit_ratio REAL NOT NULL DEFAULT 0,
            created REAL NOT NULL, approved REAL,
            review_round INTEGER NOT NULL DEFAULT 0,
            explanation TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT NOT NULL,
            language TEXT NOT NULL DEFAULT 'Santali', status TEXT NOT NULL DEFAULT 'queued',
            writing_system TEXT NOT NULL DEFAULT '', page_count INTEGER NOT NULL DEFAULT 0,
            ocr_pages INTEGER NOT NULL DEFAULT 0, source_language TEXT NOT NULL DEFAULT 'Hindi',
            file_type TEXT NOT NULL DEFAULT 'pdf',
            error TEXT, created REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS document_segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT, document_id INTEGER NOT NULL,
            page_number INTEGER NOT NULL, segment_number INTEGER NOT NULL,
            hindi TEXT NOT NULL, source_text TEXT NOT NULL DEFAULT '',
            item_id INTEGER, language_example_id INTEGER,
            UNIQUE(document_id, page_number, segment_number)
        );
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, payload TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued', error TEXT,
            created REAL NOT NULL, updated REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS document_segments_doc
            ON document_segments(document_id, page_number, segment_number);
        CREATE TABLE IF NOT EXISTS review_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, item_id INTEGER NOT NULL,
            reviewer_role TEXT NOT NULL, final TEXT NOT NULL, edited INTEGER NOT NULL,
            edit_ratio REAL NOT NULL, reviewed REAL NOT NULL,
            review_round INTEGER NOT NULL DEFAULT 0,
            explanation TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS review_events_item ON review_events(item_id, id);
        CREATE TABLE IF NOT EXISTS upload_requests (
            upload_id TEXT PRIMARY KEY, document_id INTEGER NOT NULL,
            created REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS lesson_correction_flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id INTEGER NOT NULL,
            review_round INTEGER NOT NULL,
            reviewer_role TEXT NOT NULL,
            note TEXT NOT NULL,
            created REAL NOT NULL,
            UNIQUE(item_id, review_round)
        );
        CREATE TABLE IF NOT EXISTS language_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lang TEXT NOT NULL CHECK(lang IN ('ho','mun')),
            hindi TEXT NOT NULL,
            target TEXT NOT NULL,
            writing_system TEXT NOT NULL DEFAULT 'unspecified',
            source_note TEXT NOT NULL DEFAULT '',
            contributor_role TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            review_round INTEGER NOT NULL DEFAULT 0,
            review_note TEXT,
            created REAL NOT NULL,
            approved REAL
        );
        CREATE TABLE IF NOT EXISTS language_example_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            example_id INTEGER NOT NULL,
            reviewer_role TEXT NOT NULL,
            hindi TEXT NOT NULL,
            target TEXT NOT NULL,
            writing_system TEXT NOT NULL DEFAULT 'unspecified',
            source_note TEXT NOT NULL DEFAULT '',
            review_round INTEGER NOT NULL DEFAULT 0,
            action TEXT NOT NULL,
            note TEXT,
            reviewed REAL NOT NULL,
            UNIQUE(example_id, review_round, reviewer_role)
        );
    """)
    document_columns = {row["name"] for row in con.execute("PRAGMA table_info(documents)").fetchall()}
    if "ocr_pages" not in document_columns:
        con.execute("ALTER TABLE documents ADD COLUMN ocr_pages INTEGER NOT NULL DEFAULT 0")
    if "writing_system" not in document_columns:
        con.execute("ALTER TABLE documents ADD COLUMN writing_system TEXT NOT NULL DEFAULT ''")
    if "source_language" not in document_columns:
        con.execute("ALTER TABLE documents ADD COLUMN source_language TEXT NOT NULL DEFAULT 'Hindi'")
    if "file_type" not in document_columns:
        con.execute("ALTER TABLE documents ADD COLUMN file_type TEXT NOT NULL DEFAULT 'pdf'")
    segment_columns = {row["name"] for row in con.execute("PRAGMA table_info(document_segments)").fetchall()}
    if "language_example_id" not in segment_columns:
        con.execute("ALTER TABLE document_segments ADD COLUMN language_example_id INTEGER")
    if "source_text" not in segment_columns:
        con.execute("ALTER TABLE document_segments ADD COLUMN source_text TEXT NOT NULL DEFAULT ''")
    con.execute("UPDATE document_segments SET source_text=hindi WHERE source_text='' AND hindi!=''")
    item_columns = {row["name"] for row in con.execute("PRAGMA table_info(items)").fetchall()}
    for name, declaration in (
        ("review_round", "INTEGER NOT NULL DEFAULT 0"),
        ("explanation", "TEXT NOT NULL DEFAULT ''"),
        ("audio_filename", "TEXT"),
        ("audio_mime", "TEXT"),
        ("audio_transcript", "TEXT"),
        ("audio_author", "TEXT"),
        ("audio_status", "TEXT NOT NULL DEFAULT 'none'"),
        ("audio_reviewer", "TEXT"),
        ("audio_reviewed", "REAL"),
        ("audio_generated", "INTEGER NOT NULL DEFAULT 0"),
        ("audio_generation_status", "TEXT NOT NULL DEFAULT 'idle'"),
        ("audio_generation_error", "TEXT"),
        ("audio_generation_author", "TEXT"),
        ("visual_filename", "TEXT"),
        ("visual_alt", "TEXT"),
        ("visual_credit", "TEXT"),
        ("visual_author", "TEXT"),
        ("visual_status", "TEXT NOT NULL DEFAULT 'none'"),
        ("visual_reviewer", "TEXT"),
        ("visual_reviewed", "REAL"),
    ):
        if name not in item_columns:
            con.execute(f"ALTER TABLE items ADD COLUMN {name} {declaration}")
    event_columns = {row["name"] for row in con.execute("PRAGMA table_info(review_events)").fetchall()}
    if "review_round" not in event_columns:
        con.execute("ALTER TABLE review_events ADD COLUMN review_round INTEGER NOT NULL DEFAULT 0")
    if "explanation" not in event_columns:
        con.execute("ALTER TABLE review_events ADD COLUMN explanation TEXT NOT NULL DEFAULT ''")
    con.execute("CREATE INDEX IF NOT EXISTS review_events_round ON review_events(item_id, review_round, id)")
    con.execute("CREATE INDEX IF NOT EXISTS items_audio_status ON items(status,audio_status,id)")
    con.execute("CREATE INDEX IF NOT EXISTS items_visual_status ON items(status,visual_status,id)")
    con.execute("CREATE INDEX IF NOT EXISTS lesson_correction_item ON lesson_correction_flags(item_id,review_round,id)")
    language_example_columns = {row["name"] for row in con.execute("PRAGMA table_info(language_examples)").fetchall()}
    if "writing_system" not in language_example_columns:
        con.execute("ALTER TABLE language_examples ADD COLUMN writing_system TEXT NOT NULL DEFAULT 'unspecified'")
    for name, declaration in (
        ("audio_filename", "TEXT"),
        ("audio_mime", "TEXT"),
        ("audio_transcript", "TEXT"),
        ("audio_author", "TEXT"),
        ("audio_status", "TEXT NOT NULL DEFAULT 'none'"),
        ("audio_reviewer", "TEXT"),
        ("audio_reviewed", "REAL"),
    ):
        if name not in language_example_columns:
            con.execute(f"ALTER TABLE language_examples ADD COLUMN {name} {declaration}")
    language_review_columns = {row["name"] for row in con.execute("PRAGMA table_info(language_example_reviews)").fetchall()}
    if "writing_system" not in language_review_columns:
        con.execute("ALTER TABLE language_example_reviews ADD COLUMN writing_system TEXT NOT NULL DEFAULT 'unspecified'")
    con.execute("CREATE INDEX IF NOT EXISTS language_example_work ON language_examples(lang,status,id)")
    con.execute("CREATE INDEX IF NOT EXISTS language_example_audio ON language_examples(status,audio_status,id)")
    con.execute("CREATE INDEX IF NOT EXISTS language_example_review_round ON language_example_reviews(example_id,review_round,reviewer_role)")
    # Preserve truly legacy single-review approvals as the teacher's first
    # review event. Never invent a teacher review when another role already has
    # recorded review history.
    con.execute(
        """INSERT INTO review_events (item_id, reviewer_role, final, edited, edit_ratio, reviewed, explanation)
           SELECT i.id, 'teacher', COALESCE(i.final, i.draft), i.edited, i.edit_ratio,
                  COALESCE(i.approved, i.created), i.explanation
           FROM items i WHERE i.status='approved' AND i.model!=? AND NOT EXISTS (
             SELECT 1 FROM review_events e WHERE e.item_id=i.id
           )""", (DEMO_MODEL,),
    )
    if REQUIRED_REVIEWERS == 2:
        con.execute(
            """UPDATE items
               SET status='reviewing', approved=NULL,
                   review_round=review_round + CASE WHEN EXISTS (
                       SELECT 1 FROM review_events e WHERE e.item_id=items.id
                         AND e.review_round=items.review_round
                         AND e.final IS NOT items.final
                   ) THEN 1 ELSE 0 END
               WHERE status='approved' AND model!=? AND (
                 NOT EXISTS (SELECT 1 FROM review_events e WHERE e.item_id=items.id
                             AND e.review_round=items.review_round AND e.reviewer_role='teacher'
                             AND e.final=items.final) OR
                 NOT EXISTS (SELECT 1 FROM review_events e WHERE e.item_id=items.id
                             AND e.review_round=items.review_round AND e.reviewer_role='community_expert'
                             AND e.final=items.final)
               )""", (DEMO_MODEL,),
        )
    else:
        con.execute(
            """UPDATE items
               SET status='reviewing', approved=NULL,
                   review_round=review_round + CASE WHEN EXISTS (
                       SELECT 1 FROM review_events e WHERE e.item_id=items.id
                         AND e.review_round=items.review_round AND e.final IS NOT items.final
                   ) THEN 1 ELSE 0 END
               WHERE status='approved' AND model!=? AND NOT EXISTS (
                 SELECT 1 FROM review_events e WHERE e.item_id=items.id
                   AND e.review_round=items.review_round AND e.final=items.final
               )""", (DEMO_MODEL,),
        )


@_translation_resource_guard
def add_translation(text, translated_result=None):
    hindi = " ".join(str(text or "").split())
    if not hindi:
        raise core.ApiError(400, "Enter a Hindi sentence to translate.")
    if len(hindi) > 2000:
        raise core.ApiError(400, "That text section is too long to translate. Split it into shorter sentences.")
    translator = core.get_translator()
    if not translator.mock and translator.count_tokens(hindi) > 120:
        raise core.ApiError(400, "That sentence is too long for one model input. Split it into shorter Hindi sentences.")
    with db() as con:
        row = con.execute(
            "SELECT * FROM items WHERE hindi=? AND model!=? AND status!='cancelled'", (hindi, DEMO_MODEL),
        ).fetchone()
    if row:
        return {**dict(row), "cached": True}
    draft, confidence = translated_result if translated_result is not None else translator.translate(hindi)
    now = time.time()
    with db() as con:
        stale = con.execute("SELECT id FROM items WHERE hindi=? AND model=?", (hindi, DEMO_MODEL)).fetchone()
        cancelled = con.execute(
            "SELECT id FROM items WHERE hindi=? AND model!=? AND status='cancelled'", (hindi, DEMO_MODEL),
        ).fetchone()
        if stale:
            con.execute(
                "UPDATE items SET draft=?, final=NULL, explanation='', confidence=?, model=?, status='pending', edited=0, "
                "edit_ratio=0, created=?, approved=NULL, review_round=0 WHERE id=?",
                (draft, confidence, translator.label, now, stale["id"]),
            )
        elif cancelled:
            con.execute(
                """UPDATE items SET draft=?,final=NULL,explanation='',confidence=?,model=?,status='pending',
                   edited=0,edit_ratio=0,created=?,approved=NULL,review_round=review_round+1,
                   audio_filename=NULL,audio_mime=NULL,audio_transcript=NULL,audio_author=NULL,
                   audio_status='none',audio_reviewer=NULL,audio_reviewed=NULL,audio_generated=0,
                   audio_generation_status='idle',audio_generation_error=NULL,audio_generation_author=NULL,
                   visual_filename=NULL,visual_alt=NULL,visual_credit=NULL,visual_author=NULL,
                   visual_status='none',visual_reviewer=NULL,visual_reviewed=NULL WHERE id=?""",
                (draft, confidence, translator.label, now, cancelled["id"]),
            )
        else:
            con.execute(
                "INSERT OR IGNORE INTO items (hindi,draft,confidence,model,created) VALUES (?,?,?,?,?)",
                (hindi, draft, confidence, translator.label, now),
            )
        row = con.execute("SELECT * FROM items WHERE hindi=?", (hindi,)).fetchone()
    return {**dict(row), "cached": False}


@_translation_resource_guard
def add_translations(texts):
    """Translate uncached PDF sections in small batches and keep repeated text deduplicated."""
    hindi_texts = [" ".join(str(text or "").split()) for text in texts]
    translator = core.get_translator()
    results = [None] * len(hindi_texts)
    missing = {}
    with db() as con:
        for index, hindi in enumerate(hindi_texts):
            if not hindi:
                raise core.ApiError(400, "Enter a Hindi sentence to translate.")
            if len(hindi) > 2000:
                raise core.ApiError(400, "That text section is too long to translate. Split it into shorter sentences.")
            if not translator.mock and translator.count_tokens(hindi) > 120:
                raise core.ApiError(400, "That sentence is too long for one model input. Split it into shorter Hindi sentences.")
            row = con.execute(
                "SELECT * FROM items WHERE hindi=? AND model!=? AND status!='cancelled'", (hindi, DEMO_MODEL),
            ).fetchone()
            if row:
                results[index] = {**dict(row), "cached": True}
            else:
                missing.setdefault(hindi, []).append(index)

    new_texts = list(missing)
    drafts = translator.translate_many(new_texts, batch_size=core.PDF_TRANSLATION_BATCH_SIZE)
    for hindi, translated in zip(new_texts, drafts):
        item = add_translation(hindi, translated_result=translated)
        for index in missing[hindi]:
            results[index] = item
    return results


def get_queue(reviewer_role, limit=20):
    where = """i.model!=? AND (
        i.status='pending' OR (i.status='reviewing' AND NOT EXISTS (
          SELECT 1 FROM review_events e WHERE e.item_id=i.id
            AND e.review_round=i.review_round AND e.reviewer_role=?
        )))"""
    params = (DEMO_MODEL, reviewer_role)
    with db() as con:
        total = con.execute(
            f"SELECT COUNT(*) FROM items i WHERE {where}", params,
        ).fetchone()[0]
        rows = con.execute(
            f"""SELECT i.*,
                       (SELECT COUNT(DISTINCT e.reviewer_role) FROM review_events e
                        WHERE e.item_id=i.id AND e.review_round=i.review_round) AS review_count,
                       (SELECT f.reviewer_role FROM lesson_correction_flags f
                        WHERE f.item_id=i.id AND f.review_round<=i.review_round
                        ORDER BY f.review_round DESC,f.id DESC LIMIT 1) AS correction_role,
                       (SELECT f.note FROM lesson_correction_flags f
                        WHERE f.item_id=i.id AND f.review_round<=i.review_round
                        ORDER BY f.review_round DESC,f.id DESC LIMIT 1) AS correction_note,
                       CASE WHEN i.status='reviewing' THEN i.final ELSE i.draft END AS review_text
                FROM items i WHERE {where} ORDER BY i.confidence,i.id LIMIT ?""",
            (*params, max(1, min(int(limit), 50))),
        ).fetchall()
        result = [dict(row) for row in rows]
        for item in result:
            sources = con.execute(
                """SELECT d.filename,d.file_type,ds.page_number,ds.segment_number,d.source_language,ds.source_text FROM document_segments ds
                   JOIN documents d ON d.id=ds.document_id WHERE ds.item_id=?
                   ORDER BY d.id,ds.page_number""",
                (item["id"],),
            ).fetchall()
            item["sources"] = [dict(source) for source in sources]
    return {"items": result, "total": total, "required_reviewers": REQUIRED_REVIEWERS}


def approved_media_url(kind, item_id, filename):
    if kind == "audio":
        route = "approved-audio"
    elif kind == "visual":
        route = "approved-visuals"
    else:
        raise ValueError("Unknown approved media type")
    return f"/api/{route}/{item_id}?v={filename}"


def get_my_reviews(reviewer_role, limit=20):
    """Return recent review submissions so reviewers can track what they sent."""
    with db() as con:
        rows = con.execute(
            """SELECT e.item_id,e.final AS submitted_final,e.reviewed,e.review_round,
                      i.hindi,i.final AS current_final,i.status AS current_status,
                      i.review_round AS current_round
               FROM review_events e JOIN items i ON i.id=e.item_id
               WHERE e.reviewer_role=? AND e.id=(
                 SELECT MAX(e2.id) FROM review_events e2
                 WHERE e2.item_id=e.item_id AND e2.reviewer_role=e.reviewer_role
               )
               ORDER BY e.reviewed DESC,e.id DESC LIMIT ?""",
            (reviewer_role, max(1, min(int(limit), 50))),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            is_current = item["review_round"] == item["current_round"] and \
                item["submitted_final"] == item["current_final"]
            if is_current and item["current_status"] == "approved":
                item["submission_status"] = "approved"
            elif is_current and item["current_status"] == "reviewing":
                item["submission_status"] = "awaiting_review"
            else:
                item["submission_status"] = "superseded"
            result.append(item)
    return {"items": result, "total": len(result)}


def get_approved():
    with db() as con:
        rows = con.execute(
            """SELECT id,hindi,final,audio_status,audio_generated,
                      visual_status,visual_alt,visual_credit,
                      audio_filename AS _audio_version,visual_filename AS _visual_version FROM items
               WHERE status='approved' AND final IS NOT NULL AND model!=?
                 AND (SELECT COUNT(DISTINCT e.reviewer_role) FROM review_events e WHERE e.item_id=items.id
                      AND e.review_round=items.review_round AND e.final=items.final)>=?
               ORDER BY id DESC""", (DEMO_MODEL, REQUIRED_REVIEWERS),
        ).fetchall()
    sources_by_item = {}
    with db() as con:
        item_ids = [int(row["id"]) for row in rows]
        for start in range(0, len(item_ids), 500):
            batch = item_ids[start:start + 500]
            placeholders = ",".join("?" for _ in batch)
            sources = con.execute(
                f"""SELECT ds.item_id,d.filename,d.file_type,ds.page_number,ds.segment_number,d.source_language,ds.source_text
                    FROM document_segments ds JOIN documents d ON d.id=ds.document_id
                    WHERE ds.item_id IN ({placeholders})
                    ORDER BY ds.item_id,d.id,ds.page_number,ds.segment_number""",
                batch,
            ).fetchall()
            for source in sources:
                sources_by_item.setdefault(source["item_id"], []).append(dict(source))
    lessons = []
    for row in rows:
        item = dict(row)
        audio_version = item.pop("_audio_version")
        visual_version = item.pop("_visual_version")
        item["audio_url"] = approved_media_url("audio", item["id"], audio_version) \
            if item["audio_status"] == "approved" and audio_version else None
        item["visual_url"] = approved_media_url("visual", item["id"], visual_version) \
            if item["visual_status"] == "approved" and visual_version else None
        if item["visual_status"] != "approved":
            item["visual_alt"] = None
            item["visual_credit"] = None
        item["sources"] = sources_by_item.get(item["id"], [])
        lessons.append(item)
    return lessons


def get_audio_work(reviewer_role, limit=20, offset=0):
    limit = max(1, min(int(limit), 50))
    offset = max(0, int(offset))
    where = """status='approved' AND final IS NOT NULL AND model!=? AND
        (SELECT COUNT(DISTINCT e.reviewer_role) FROM review_events e WHERE e.item_id=items.id
         AND e.review_round=items.review_round AND e.final=items.final)>=? AND (
        audio_status='none' OR
        audio_status='awaiting_review' OR
        (audio_status='rejected' AND audio_author=?)
    )"""
    params = (DEMO_MODEL, REQUIRED_REVIEWERS, reviewer_role)
    with db() as con:
        total = con.execute(f"SELECT COUNT(*) FROM items WHERE {where}", params).fetchone()[0]
        rows = con.execute(
            f"""SELECT id,hindi,final,audio_status,audio_filename,audio_mime,
                       audio_transcript,audio_author,audio_reviewer,audio_generated,
                       audio_generation_status,audio_generation_error
                FROM items WHERE {where}
                ORDER BY CASE audio_status WHEN 'awaiting_review' THEN 0 WHEN 'rejected' THEN 1 ELSE 2 END,
                         id LIMIT ? OFFSET ?""",
            (*params, limit, offset),
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        generating = item["audio_generation_status"] in ("queued", "processing")
        item["can_record"] = item["audio_status"] in ("none", "rejected") and not generating
        item["can_generate"] = item["can_record"]
        item["can_review"] = item["audio_status"] == "awaiting_review"
        item["audio_url"] = f"/api/audio/{item['id']}" if item["can_review"] else None
        items.append(item)
    return {"items": items, "total": total, "offset": offset, "limit": limit}


def get_my_audio(reviewer_role, limit=30):
    where = "audio_author=? AND audio_filename IS NOT NULL"
    with db() as con:
        total = con.execute(f"SELECT COUNT(*) FROM items WHERE {where}", (reviewer_role,)).fetchone()[0]
        rows = con.execute(
            f"""SELECT id,hindi,final,audio_status,audio_reviewer,audio_reviewed,audio_generated
                FROM items WHERE {where} ORDER BY id DESC LIMIT ?""",
            (reviewer_role, max(1, min(int(limit), 50))),
        ).fetchall()
    return {"items": [dict(row) for row in rows], "total": total}


def get_language_audio_work(reviewer_role, limit=30):
    where = "e.status='approved' AND (e.audio_status='none' OR " \
        "(e.audio_status='awaiting_review' AND e.audio_author!=?) OR " \
        "(e.audio_status='rejected' AND e.audio_author=?))"
    params = (reviewer_role, reviewer_role)
    with db() as con:
        total = con.execute(
            f"SELECT COUNT(*) FROM language_examples e WHERE {where}", params,
        ).fetchone()[0]
        rows = con.execute(
            f"""SELECT e.id,e.lang,e.hindi,e.target,e.writing_system,e.source_note,
                       e.audio_status,e.audio_filename,e.audio_mime,e.audio_transcript,e.audio_author,e.audio_reviewer
                FROM language_examples e WHERE {where}
                ORDER BY CASE e.audio_status WHEN 'awaiting_review' THEN 0 WHEN 'rejected' THEN 1 ELSE 2 END,
                         e.id DESC LIMIT ?""",
            (*params, max(1, min(int(limit), 50))),
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["final"] = item["target"]
        item["language"] = "Ho" if item["lang"] == "ho" else "Mundari"
        item["audio_base"] = "/api/language-audio"
        item["can_record"] = item["audio_status"] in ("none", "rejected")
        item["can_review"] = item["audio_status"] == "awaiting_review"
        item["audio_url"] = f"/api/language-audio/{item['id']}" if item["can_review"] else None
        items.append(item)
    return {"items": items, "total": total}


def get_my_language_audio(reviewer_role, limit=30):
    with db() as con:
        total = con.execute(
            "SELECT COUNT(*) FROM language_examples WHERE audio_author=? AND audio_filename IS NOT NULL",
            (reviewer_role,),
        ).fetchone()[0]
        rows = con.execute(
            """SELECT id,lang,hindi,target,writing_system,audio_status,audio_reviewer,audio_reviewed
               FROM language_examples WHERE audio_author=? AND audio_filename IS NOT NULL
               ORDER BY id DESC LIMIT ?""",
            (reviewer_role, max(1, min(int(limit), 50))),
        ).fetchall()
    items = [dict(row) for row in rows]
    for item in items:
        item["final"] = item["target"]
        item["language"] = "Ho" if item["lang"] == "ho" else "Mundari"
        item["audio_base"] = "/api/language-audio"
    return {"items": items, "total": total}


def remove_audio_file(filename):
    if not filename or not re.fullmatch(r"(?:\d+|lang-\d+)-[a-f0-9]{32}\.(webm|ogg|wav|mp4|mp3)", filename):
        return
    try:
        (AUDIO_DIR / filename).unlink(missing_ok=True)
    except OSError:
        pass


def normalize_visual_image(raw):
    """Validate and re-encode a lesson illustration, stripping metadata."""
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise core.ApiError(503, "Install the Pillow package on the school server to add lesson visuals, then restart it.") from exc
    try:
        with Image.open(BytesIO(raw)) as source:
            if source.format not in ("JPEG", "PNG", "WEBP"):
                raise core.ApiError(415, "Use a JPEG, PNG, or WebP image for a lesson visual.")
            if getattr(source, "is_animated", False):
                raise core.ApiError(415, "Animated images are not supported. Upload a still image.")
            if source.width < 1 or source.height < 1 or source.width * source.height > 20_000_000:
                raise core.ApiError(413, "Images must have at most 20 million pixels.")
            oriented = ImageOps.exif_transpose(source)
            oriented.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            rgba = oriented.convert("RGBA")
            background = Image.new("RGB", rgba.size, "white")
            background.paste(rgba, mask=rgba.getchannel("A"))
            output = BytesIO()
            background.save(output, format="JPEG", quality=88, optimize=True, progressive=True)
            return output.getvalue()
    except core.ApiError:
        raise
    except Exception as exc:
        raise core.ApiError(400, "The image could not be opened. Choose a clear JPEG, PNG, or WebP image.") from exc


def remove_visual_file(filename):
    if not filename or not re.fullmatch(r"\d+-[a-f0-9]{32}\.jpg", filename):
        return
    try:
        (VISUAL_DIR / filename).unlink(missing_ok=True)
    except OSError:
        pass


def get_visual_work(reviewer_role, limit=20):
    where = """status='approved' AND model!=? AND (
        visual_status='none' OR
        visual_status='awaiting_review' OR
        (visual_status='rejected' AND visual_author=?)
    )"""
    params = (DEMO_MODEL, reviewer_role)
    with db() as con:
        total = con.execute(f"SELECT COUNT(*) FROM items WHERE {where}", params).fetchone()[0]
        rows = con.execute(
            f"""SELECT id,hindi,final,explanation,visual_status,visual_filename,
                       visual_alt,visual_credit,visual_author,visual_reviewer
                FROM items WHERE {where}
                ORDER BY CASE visual_status WHEN 'awaiting_review' THEN 0
                             WHEN 'rejected' THEN 1 ELSE 2 END,id LIMIT ?""",
            (*params, max(1, min(int(limit), 50))),
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["can_attach"] = item["visual_status"] in ("none", "rejected")
        item["can_review"] = item["visual_status"] == "awaiting_review"
        item["visual_url"] = f"/api/visual/{item['id']}" if item["can_review"] else None
        items.append(item)
    return {"items": items, "total": total}


def get_my_visuals(reviewer_role, limit=30):
    with db() as con:
        total = con.execute(
            "SELECT COUNT(*) FROM items WHERE visual_author=? AND visual_filename IS NOT NULL",
            (reviewer_role,),
        ).fetchone()[0]
        rows = con.execute(
            """SELECT id,final,visual_alt,visual_credit,visual_status,visual_reviewer,visual_reviewed
               FROM items WHERE visual_author=? AND visual_filename IS NOT NULL ORDER BY id DESC LIMIT ?""",
            (reviewer_role, max(1, min(int(limit), 50))),
        ).fetchall()
    return {"items": [dict(row) for row in rows], "total": total}


def get_stats():
    with db() as con:
        row = con.execute(
            """SELECT COUNT(*) AS total,
                      COALESCE(SUM(status='pending' OR (status='reviewing' AND NOT EXISTS (
                        SELECT 1 FROM review_events e WHERE e.item_id=items.id
                          AND e.review_round=items.review_round
                      ))),0) AS pending,
                      COALESCE(SUM(status='reviewing' AND EXISTS (
                        SELECT 1 FROM review_events e WHERE e.item_id=items.id
                          AND e.review_round=items.review_round
                      )),0) AS awaiting_second_review,
                      SUM(status='approved') AS approved,
                      SUM(status='approved' AND edited=1) AS edited,
                      SUM(audio_status='awaiting_review') AS audio_awaiting_review,
                      SUM(audio_status='approved' AND status='approved') AS audio_approved,
                      SUM(visual_status='awaiting_review') AS visual_awaiting_review,
                      SUM(visual_status='approved' AND status='approved') AS visual_approved,
                      AVG(confidence) AS avg_conf,
                      AVG(CASE WHEN status='approved' AND edited=1 THEN confidence END) AS conf_edited,
                      AVG(CASE WHEN status='approved' AND edited=0 THEN confidence END) AS conf_kept
               FROM items WHERE model!=? AND status!='cancelled'""",
            (DEMO_MODEL,),
        ).fetchone()
        reviewer_rows = con.execute(
            "SELECT reviewer_role, COUNT(*) AS approvals, SUM(edited) AS edits "
            "FROM review_events GROUP BY reviewer_role"
        ).fetchall()
    reviewers = {r["reviewer_role"]: {"approvals": r["approvals"], "edits": r["edits"] or 0}
                 for r in reviewer_rows}
    reviewers.setdefault("teacher", {"approvals": 0, "edits": 0})
    reviewers.setdefault("community_expert", {"approvals": 0, "edits": 0})
    with core.db() as con:
        glossary = {row["status"]: row["n"] for row in con.execute(
            "SELECT status,COUNT(*) AS n FROM glossary GROUP BY status").fetchall()}
    for status in ("pending", "reviewing", "approved", "rejected"):
        glossary.setdefault(status, 0)
    stats = dict(row)
    stats["required_reviewers"] = REQUIRED_REVIEWERS
    if REQUIRED_REVIEWERS == 1:
        stats["awaiting_second_review"] = 0
    return {**stats, "reviewers": reviewers,
            "glossary": glossary,
            "model": core.get_translator_label(), "benchmark": None}


def lookup_words(q):
    # Raw OCR entries stay out of the learner search until a reviewer approves them.
    result = core.list_approved_glossary() if not str(q or "").strip() else core.search_glossary(q, "")
    return {"total_words": sum(result["counts"].values()), "results": result["results"],
            "complete": result.get("complete", False)}


def get_glossary_work(reviewer_role, limit=20):
    where = "status IN ('pending','reviewing') AND NOT EXISTS (" \
        "SELECT 1 FROM glossary_review_events e WHERE e.glossary_id=glossary.id " \
        "AND e.review_round=glossary.review_round AND e.reviewer_role=?)"
    with core.db() as con:
        total = con.execute(f"SELECT COUNT(*) FROM glossary WHERE {where}", (reviewer_role,)).fetchone()[0]
        rows = con.execute(
            f"""SELECT id,lang,hindi,word,source,status,review_round,rejection_note FROM glossary
                WHERE {where}
                ORDER BY CASE status WHEN 'reviewing' THEN 0 ELSE 1 END,id LIMIT ?""",
            (reviewer_role, max(1, min(int(limit), 50))),
        ).fetchall()
    return {"items": [dict(row) for row in rows], "total": total}


def add_glossary_entry(lang, hindi, word, reviewer_role):
    if lang not in ("ho", "mun"):
        raise core.ApiError(400, "Choose Ho or Mundari.")
    hindi = " ".join(str(hindi or "").split())
    word = " ".join(str(word or "").split())
    if not hindi or not word:
        raise core.ApiError(400, "Fill in both the Hindi word and the Ho or Mundari word.")
    if " " in hindi:
        raise core.ApiError(400, "The Hindi side must be a single word.")
    if len(hindi) > 60 or len(word) > 60:
        raise core.ApiError(400, "Each word must be 60 characters or fewer.")
    source = "Submitted by the community language expert" if reviewer_role == "community_expert" \
        else "Submitted by the teacher"
    with core.db() as con:
        old = con.execute(
            "SELECT id,status FROM glossary WHERE lang=? AND hindi=? AND word=?",
            (lang, hindi, word),
        ).fetchone()
        if old and old["status"] != "rejected":
            raise core.ApiError(409, "That word pair already exists. Review it in the queue instead of adding a duplicate.")
        if old:
            con.execute(
                "UPDATE glossary SET source=?,status='pending',review_round=review_round+1,rejection_note=NULL WHERE id=?",
                (source, old["id"]),
            )
            return {"id": old["id"], "status": "pending", "reopened": True}
        try:
            cursor = con.execute(
                "INSERT INTO glossary (lang,hindi,word,source,status) VALUES (?,?,?,?,'pending')",
                (lang, hindi, word, source),
            )
        except sqlite3.IntegrityError as exc:
            raise core.ApiError(409, "That word pair was added by another reviewer. Reload the review queue.") from exc
    return {"id": cursor.lastrowid, "status": "pending", "reopened": False}


def review_glossary_entry(entry_id, body, reviewer_role):
    hindi = " ".join(str(body.hindi or "").split())
    word = " ".join(str(body.word or "").split())
    note = " ".join(str(body.note or "").split())
    if body.action == "approve":
        if not hindi or not word:
            raise core.ApiError(400, "Enter both sides of the word pair before submitting it for review.")
        if " " in hindi or len(hindi) > 60 or len(word) > 60:
            raise core.ApiError(400, "Use one Hindi word and keep each word to 60 characters or fewer.")
    elif len(note) < 5:
        raise core.ApiError(400, "Add a short reason so the rejected OCR entry can be corrected at its source.")

    try:
        with core.db() as con:
            row = con.execute(
                "SELECT id,lang,hindi,word,status,review_round FROM glossary WHERE id=?",
                (entry_id,),
            ).fetchone()
            if not row or row["status"] not in ("pending", "reviewing"):
                raise core.ApiError(409, "That word pair is no longer waiting for review.")
            prior = con.execute(
                "SELECT 1 FROM glossary_review_events WHERE glossary_id=? AND review_round=? AND reviewer_role=?",
                (entry_id, row["review_round"], reviewer_role),
            ).fetchone()
            if prior:
                raise core.ApiError(409, "This reviewer has already checked the word pair in this review round.")
            reviewed = time.time()
            if body.action == "reject":
                con.execute(
                    "UPDATE glossary SET status='rejected',rejection_note=? WHERE id=?",
                    (note[:500], entry_id),
                )
                con.execute(
                    """INSERT INTO glossary_review_events
                       (glossary_id,reviewer_role,hindi,word,review_round,reviewed,action,note)
                       VALUES (?,?,?,?,?,?,'reject',?)""",
                    (entry_id, reviewer_role, row["hindi"], row["word"], row["review_round"], reviewed, note[:500]),
                )
                return {"id": entry_id, "status": "rejected", "message": "Entry rejected and kept out of learner search."}

            changed = hindi != row["hindi"] or word != row["word"]
            restarted = row["status"] == "reviewing" and changed
            next_round = row["review_round"] + 1 if restarted else row["review_round"]
            approved = True
            status = "approved"
            con.execute(
                """UPDATE glossary SET hindi=?,word=?,status=?,review_round=?,rejection_note=NULL
                   WHERE id=?""",
                (hindi, word, status, next_round, entry_id),
            )
            con.execute(
                """INSERT INTO glossary_review_events
                   (glossary_id,reviewer_role,hindi,word,review_round,reviewed,action)
                   VALUES (?,?,?,?,?,?,'approve')""",
                (entry_id, reviewer_role, hindi, word, next_round, reviewed),
            )
    except sqlite3.IntegrityError as exc:
        raise core.ApiError(409, "That corrected pair duplicates an existing word. Choose the existing row instead.") from exc

    message = "Word pair approved for learner search."
    return {"id": entry_id, "status": status, "approved": approved, "revision_restarted": restarted,
            "message": message}


def reopen_glossary_entry(entry_id, note, reviewer_role):
    note = " ".join(str(note or "").split())
    if len(note) < 5:
        raise core.ApiError(400, "Explain why this approved word pair needs correction.")
    with core.db() as con:
        row = con.execute("SELECT status,review_round FROM glossary WHERE id=?", (entry_id,)).fetchone()
        if not row:
            raise core.ApiError(404, "That word pair was not found.")
        if row["status"] != "approved":
            raise core.ApiError(409, "Only an approved pair can be reopened. Check the review queue instead.")
        con.execute(
            "UPDATE glossary SET status='pending',review_round=review_round+1,rejection_note=? WHERE id=?",
            (f"Correction requested by {reviewer_role}: {note[:450]}", entry_id),
        )
    return {"id": entry_id, "status": "pending",
            "message": "The pair is hidden from learners until a reviewer approves it again."}


def get_language_example_work(reviewer_role, lang=None, limit=20):
    if lang not in (None, "ho", "mun"):
        raise core.ApiError(400, "Choose Ho or Mundari.")
    where = "e.status IN ('pending','reviewing') AND NOT EXISTS (" \
        "SELECT 1 FROM language_example_reviews r WHERE r.example_id=e.id " \
        "AND r.review_round=e.review_round AND r.reviewer_role=?)"
    args = [reviewer_role]
    if lang:
        where += " AND e.lang=?"
        args.append(lang)
    limit = max(1, min(int(limit), 100))
    with db() as con:
        total = con.execute(f"SELECT COUNT(*) FROM language_examples e WHERE {where}", args).fetchone()[0]
        rows = con.execute(
            f"""SELECT e.id,e.lang,e.hindi,e.target,e.writing_system,e.source_note,e.contributor_role,
                       e.status,e.review_round,e.review_note,e.created
                FROM language_examples e WHERE {where}
                ORDER BY CASE e.status WHEN 'reviewing' THEN 0 ELSE 1 END,e.created,e.id LIMIT ?""",
            [*args, limit],
        ).fetchall()
    return {"items": [dict(row) for row in rows], "total": total}


def get_language_corpus_state():
    counts = {lang: {"pending": 0, "reviewing": 0, "approved": 0, "rejected": 0}
              for lang in ("ho", "mun")}
    with db() as con:
        rows = con.execute(
            "SELECT lang,status,COUNT(*) AS n FROM language_examples GROUP BY lang,status"
        ).fetchall()
        for row in rows:
            if row["lang"] in counts and row["status"] in counts[row["lang"]]:
                counts[row["lang"]][row["status"]] = row["n"]
        approved = con.execute(
            """SELECT e.id,e.lang,e.hindi,e.target,e.writing_system,e.source_note,e.contributor_role,
                      e.review_round,e.approved
               FROM language_examples e
               WHERE e.status='approved'
                 AND EXISTS (SELECT 1 FROM language_example_reviews r WHERE r.example_id=e.id
                             AND r.review_round=e.review_round
                             AND r.action='approve' AND r.hindi=e.hindi AND r.target=e.target
                             AND r.writing_system=e.writing_system AND r.source_note=e.source_note)
               ORDER BY e.approved DESC,e.id DESC LIMIT 30"""
        ).fetchall()
    return {"languages": counts, "approved_examples": [dict(row) for row in approved],
            "approved_examples_shown": len(approved), "approved_examples_limit": 30}


def get_language_student_lessons(lang):
    if lang not in ("ho", "mun"):
        raise core.ApiError(400, "Choose Ho or Mundari.")
    with db() as con:
        rows = con.execute(
            """SELECT e.id,e.lang,e.hindi,e.target,e.writing_system,e.source_note,
                      e.audio_status,e.audio_filename AS _audio_version
               FROM language_examples e
               WHERE e.lang=? AND e.status='approved' AND TRIM(e.target)!=''
                 AND EXISTS (SELECT 1 FROM language_example_reviews r WHERE r.example_id=e.id
                             AND r.review_round=e.review_round
                             AND r.action='approve' AND r.hindi=e.hindi AND r.target=e.target
                             AND r.writing_system=e.writing_system AND r.source_note=e.source_note)
               ORDER BY e.id DESC""", (lang,),
        ).fetchall()
    lessons = []
    for row in rows:
        lesson = dict(row)
        audio_version = lesson.pop("_audio_version")
        lesson["audio_url"] = f"/api/language-approved-audio/{lesson['id']}" \
            if lesson["audio_status"] == "approved" and audio_version else None
        lessons.append(lesson)
    return {"language": lang, "lessons": lessons}


def get_my_language_reviews(reviewer_role, limit=20):
    with db() as con:
        rows = con.execute(
            """SELECT r.example_id,r.hindi AS submitted_hindi,r.target AS submitted_target,
                      r.writing_system AS submitted_writing_system,
                      r.source_note AS submitted_source_note,r.reviewed,r.review_round,
                      e.lang,e.hindi AS current_hindi,e.target AS current_target,
                      e.writing_system AS current_writing_system,e.source_note AS current_source_note,
                      e.status AS current_status,e.review_round AS current_round,e.review_note
               FROM language_example_reviews r JOIN language_examples e ON e.id=r.example_id
               WHERE r.reviewer_role=? AND r.id=(
                 SELECT MAX(r2.id) FROM language_example_reviews r2
                 WHERE r2.example_id=r.example_id AND r2.reviewer_role=r.reviewer_role
               )
               ORDER BY r.reviewed DESC,r.id DESC LIMIT ?""",
            (reviewer_role, max(1, min(int(limit), 50))),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            is_current = item["review_round"] == item["current_round"] and all((
                item["submitted_hindi"] == item["current_hindi"],
                item["submitted_target"] == item["current_target"],
                item["submitted_writing_system"] == item["current_writing_system"],
                item["submitted_source_note"] == item["current_source_note"],
            ))
            if is_current and item["current_status"] == "approved":
                item["submission_status"] = "approved"
            elif item["current_status"] == "rejected":
                item["submission_status"] = "rejected"
            elif is_current and item["current_status"] == "reviewing":
                item["submission_status"] = "awaiting_review"
            else:
                item["submission_status"] = "superseded"
            result.append(item)
    return {"items": result, "total": len(result)}


def queue_language_translation(lang, hindi, writing_system, source_note, reviewer_role):
    if lang not in ("ho", "mun"):
        raise core.ApiError(400, "Choose Ho or Mundari.")
    hindi = " ".join(str(hindi or "").split())
    writing_system = " ".join(str(writing_system or "").split())
    source_note = " ".join(str(source_note or "").split())
    if not hindi or not writing_system:
        raise core.ApiError(400, "Enter the Hindi curriculum text and target writing system.")
    if len(hindi) > 1200 or len(writing_system) > 80 or len(source_note) > 300:
        raise core.ApiError(400, "Keep the Hindi text under 1,200 characters, script under 80, and source under 300.")
    with db() as con:
        duplicate = con.execute(
            """SELECT id,status FROM language_examples
               WHERE lang=? AND hindi=? AND writing_system=? AND target=''
                 AND status IN ('pending','reviewing','approved') ORDER BY id DESC LIMIT 1""",
            (lang, hindi, writing_system),
        ).fetchone()
        if duplicate:
            raise core.ApiError(409, "That source text already has a human translation task or approved result.")
        cur = con.execute(
            """INSERT INTO language_examples
               (lang,hindi,target,writing_system,source_note,contributor_role,status,created)
               VALUES (?,?,'',?,?,?,'pending',?)""",
            (lang, hindi, writing_system, source_note, reviewer_role, time.time()),
        )
    return {"id": cur.lastrowid, "language": lang, "status": "pending",
            "message": "Added to the human translation queue. A reviewer must write and approve the translation before students can see it."}


def add_language_example(lang, hindi, target, writing_system, source_note, reviewer_role):
    if lang not in ("ho", "mun"):
        raise core.ApiError(400, "Choose Ho or Mundari.")
    hindi = " ".join(str(hindi or "").split())
    target = " ".join(str(target or "").split())
    writing_system = " ".join(str(writing_system or "").split())
    source_note = " ".join(str(source_note or "").split())
    if not hindi or not target or not writing_system:
        raise core.ApiError(400, "Enter the Hindi sentence, target translation, and its writing system.")
    if len(hindi) > 1200 or len(target) > 2000 or len(writing_system) > 80 or len(source_note) > 300:
        raise core.ApiError(400, "Keep the Hindi sentence under 1,200 characters, translation under 2,000, and source note under 300.")
    with db() as con:
        duplicate = con.execute(
            """SELECT id,status FROM language_examples
               WHERE lang=? AND hindi=? AND target=? AND writing_system=? AND status!='rejected'
               ORDER BY id DESC LIMIT 1""",
            (lang, hindi, target, writing_system),
        ).fetchone()
        if duplicate:
            raise core.ApiError(409, "That exact pair already exists. Review the existing entry or flag it for correction.")
        cur = con.execute(
            """INSERT INTO language_examples
               (lang,hindi,target,writing_system,source_note,contributor_role,status,created)
               VALUES (?,?,?,?,?,?,'pending',?)""",
            (lang, hindi, target, writing_system, source_note, reviewer_role, time.time()),
        )
    return {"id": cur.lastrowid, "status": "pending",
            "message": "Sentence pair submitted for one reviewer to check and approve."}


def review_language_example(example_id, body, reviewer_role):
    hindi = " ".join(str(body.hindi or "").split())
    target = " ".join(str(body.target or "").split())
    writing_system = " ".join(str(body.writing_system or "").split())
    source_note = " ".join(str(body.source_note or "").split())
    note = " ".join(str(body.note or "").split())
    if body.action == "approve":
        if not hindi or not target or not writing_system:
            raise core.ApiError(400, "Enter the Hindi sentence, target translation, and writing system before approving it.")
        if len(hindi) > 1200 or len(target) > 2000 or len(writing_system) > 80 or len(source_note) > 300:
            raise core.ApiError(400, "The sentence or source note is too long.")
    elif len(note) < 5:
        raise core.ApiError(400, "Give a short reason for rejecting this sentence pair.")

    invalidated_audio = None
    try:
        with db() as con:
            row = con.execute(
                """SELECT id,lang,hindi,target,writing_system,source_note,status,review_round,
                          audio_filename,audio_transcript
                   FROM language_examples WHERE id=?""", (example_id,),
            ).fetchone()
            if not row or row["status"] not in ("pending", "reviewing"):
                raise core.ApiError(409, "That sentence pair is no longer waiting for review.")
            prior = con.execute(
                """SELECT 1 FROM language_example_reviews WHERE example_id=?
                   AND review_round=? AND reviewer_role=?""",
                (example_id, row["review_round"], reviewer_role),
            ).fetchone()
            if prior:
                raise core.ApiError(409, "This reviewer has already checked the sentence pair in this review round.")
            reviewed = time.time()
            if body.action == "reject":
                con.execute(
                    "UPDATE language_examples SET status='rejected',review_note=?,approved=NULL WHERE id=?",
                    (note[:500], example_id),
                )
                con.execute(
                    """INSERT INTO language_example_reviews
                       (example_id,reviewer_role,hindi,target,writing_system,source_note,review_round,action,note,reviewed)
                       VALUES (?,?,?,?,?,?,?,'reject',?,?)""",
                    (example_id, reviewer_role, row["hindi"], row["target"], row["writing_system"], row["source_note"],
                     row["review_round"], note[:500], reviewed),
                )
                return {"id": example_id, "status": "rejected",
                        "message": "Pair rejected and excluded from the reviewed corpus."}

            changed = (hindi != row["hindi"] or target != row["target"]
                       or writing_system != row["writing_system"]
                       or source_note != row["source_note"])
            restarted = row["status"] == "reviewing" and changed
            next_round = row["review_round"] + 1 if restarted else row["review_round"]
            approved = True
            status = "approved"
            if (row["audio_filename"] and
                    (target != row["target"] or writing_system != row["writing_system"]) and
                    row["audio_transcript"] != target):
                invalidated_audio = row["audio_filename"]
            con.execute(
                """UPDATE language_examples SET hindi=?,target=?,writing_system=?,source_note=?,status=?,review_round=?,
                   approved=? WHERE id=?""",
                (hindi, target, writing_system, source_note, status, next_round, reviewed if approved else None, example_id),
            )
            if invalidated_audio:
                con.execute(
                    """UPDATE language_examples SET audio_filename=NULL,audio_mime=NULL,audio_transcript=NULL,
                       audio_author=NULL,audio_status='none',audio_reviewer=NULL,audio_reviewed=NULL WHERE id=?""",
                    (example_id,),
                )
            con.execute(
                """INSERT INTO language_example_reviews
                   (example_id,reviewer_role,hindi,target,writing_system,source_note,review_round,action,reviewed)
                   VALUES (?,?,?,?,?,?,?,'approve',?)""",
                (example_id, reviewer_role, hindi, target, writing_system, source_note, next_round, reviewed),
            )
    except sqlite3.IntegrityError as exc:
        raise core.ApiError(409, "Your review was already recorded. Reload the review queue.") from exc

    remove_audio_file(invalidated_audio)

    message = "One reviewer approved this exact sentence pair for the Ho/Mundari collection."
    return {"id": example_id, "status": status, "approved": approved,
            "revision_restarted": restarted, "message": message}


def reopen_language_example(example_id, note, reviewer_role):
    note = " ".join(str(note or "").split())
    if len(note) < 5:
        raise core.ApiError(400, "Explain why this approved sentence pair needs correction.")
    with db() as con:
        row = con.execute("SELECT status,review_round FROM language_examples WHERE id=?", (example_id,)).fetchone()
        if not row:
            raise core.ApiError(404, "That sentence pair was not found.")
        if row["status"] not in ("approved", "rejected"):
            raise core.ApiError(409, "Only an approved or rejected sentence pair can be reopened.")
        con.execute(
            """UPDATE language_examples SET status='pending',approved=NULL,review_round=review_round+1,
               review_note=? WHERE id=?""",
            (f"Correction requested by {reviewer_role}: {note[:450]}", example_id),
        )
    return {"id": example_id, "status": "pending",
            "message": "The translation task is back in the queue; one reviewer must approve its exact text before students can see it."}


def export_language_examples(lang):
    if lang not in ("ho", "mun"):
        raise core.ApiError(400, "Choose Ho or Mundari.")
    with db() as con:
        rows = con.execute(
            """SELECT e.id,e.lang,e.hindi,e.target,e.writing_system,e.source_note,e.contributor_role,
                      e.review_round,
                      (SELECT MAX(r.reviewed) FROM language_example_reviews r
                       WHERE r.example_id=e.id AND r.review_round=e.review_round
                         AND r.action='approve' AND r.hindi=e.hindi AND r.target=e.target
                         AND r.writing_system=e.writing_system AND r.source_note=e.source_note) AS reviewed_at
               FROM language_examples e
               WHERE e.lang=? AND e.status='approved'
                 AND EXISTS (SELECT 1 FROM language_example_reviews r WHERE r.example_id=e.id
                   AND r.review_round=e.review_round AND r.action='approve' AND r.hindi=e.hindi
                   AND r.target=e.target AND r.writing_system=e.writing_system AND r.source_note=e.source_note)
               ORDER BY e.id""", (lang,),
        ).fetchall()
        result = []
        for row in rows:
            roles = [r["reviewer_role"] for r in con.execute(
                """SELECT DISTINCT reviewer_role FROM language_example_reviews
                   WHERE example_id=? AND review_round=? AND action='approve' AND hindi=? AND target=?
                     AND writing_system=? AND source_note=? ORDER BY reviewer_role""",
                (row["id"], row["review_round"], row["hindi"], row["target"],
                 row["writing_system"], row["source_note"]),
            ).fetchall()]
            result.append(json.dumps({
                "id": row["id"], "hindi": row["hindi"], "target": row["target"],
                "language": row["lang"], "writing_system": row["writing_system"],
                "source_note": row["source_note"], "contributor_role": row["contributor_role"],
                "reviewers": roles, "reviewer_roles": roles,
                "human_validated": True, "synthetic": False,
                "reviewed_at": row["reviewed_at"],
            }, ensure_ascii=False))
    return result


def export_jsonl():
    with db() as con:
        rows = con.execute(
            """SELECT i.id, i.hindi, i.final, i.edited, i.model,
                      (SELECT f.note FROM lesson_correction_flags f
                       WHERE f.item_id=i.id AND f.review_round<=i.review_round
                       ORDER BY f.review_round DESC,f.id DESC LIMIT 1) AS correction_note,
                      (SELECT f.reviewer_role FROM lesson_correction_flags f
                       WHERE f.item_id=i.id AND f.review_round<=i.review_round
                       ORDER BY f.review_round DESC,f.id DESC LIMIT 1) AS correction_role,
                      (SELECT f.review_round FROM lesson_correction_flags f
                       WHERE f.item_id=i.id AND f.review_round<=i.review_round
                       ORDER BY f.review_round DESC,f.id DESC LIMIT 1) AS correction_round,
                      (SELECT f.created FROM lesson_correction_flags f
                       WHERE f.item_id=i.id AND f.review_round<=i.review_round
                       ORDER BY f.review_round DESC,f.id DESC LIMIT 1) AS correction_requested_at,
                      e.reviewer_role, e.reviewed
               FROM items i LEFT JOIN review_events e
                 ON e.id=(SELECT MAX(e2.id) FROM review_events e2 WHERE e2.item_id=i.id
                          AND e2.review_round=i.review_round AND e2.final=i.final)
               WHERE i.status='approved' AND i.final IS NOT NULL AND i.model!=?
                 AND EXISTS (SELECT 1 FROM review_events e WHERE e.item_id=i.id
                             AND e.review_round=i.review_round AND e.reviewer_role='teacher'
                             AND e.final=i.final)
                 AND EXISTS (SELECT 1 FROM review_events e WHERE e.item_id=i.id
                             AND e.review_round=i.review_round AND e.reviewer_role='community_expert'
                             AND e.final=i.final)
               ORDER BY i.id""",
            (DEMO_MODEL,),
        ).fetchall()
        reviewer_roles = {
            row["id"]: [r["reviewer_role"] for r in con.execute(
                """SELECT DISTINCT reviewer_role FROM review_events
                   WHERE item_id=? AND review_round=(SELECT review_round FROM items WHERE id=?)
                     AND final=(SELECT final FROM items WHERE id=?)
                     ORDER BY reviewer_role""",
                (row["id"], row["id"], row["id"]),
            ).fetchall()]
            for row in rows
        }
    return "".join(json.dumps({"hindi": row["hindi"], "target": row["final"],
                               "correction_note": row["correction_note"],
                               "correction_role": row["correction_role"],
                               "correction_round": row["correction_round"],
                               "correction_requested_at": row["correction_requested_at"],
                               "language": "sat", "teacher_edited": bool(row["edited"]),
                               "reviewer_role": row["reviewer_role"] or "legacy",
                               "reviewer_roles": reviewer_roles[row["id"]],
                               "reviewed_at": row["reviewed"], "source": row["model"]},
                              ensure_ascii=False) + "\n" for row in rows)


def export_glossary_jsonl(lang):
    if lang not in ("ho", "mun"):
        raise core.ApiError(400, "Choose Ho or Mundari.")
    with core.db() as con:
        rows = con.execute(
            """SELECT g.id,g.hindi,g.word,g.source,
                      (SELECT MAX(e.reviewed) FROM glossary_review_events e
                       WHERE e.glossary_id=g.id AND e.review_round=g.review_round
                         AND e.action='approve' AND e.hindi=g.hindi AND e.word=g.word) AS reviewed_at
                      ,(SELECT GROUP_CONCAT(DISTINCT e.reviewer_role) FROM glossary_review_events e
                        WHERE e.glossary_id=g.id AND e.review_round=g.review_round
                          AND e.action='approve' AND e.hindi=g.hindi AND e.word=g.word) AS reviewer_roles
               FROM glossary g WHERE g.lang=? AND g.status='approved'
                 AND EXISTS (SELECT 1 FROM glossary_review_events e WHERE e.glossary_id=g.id
                   AND e.review_round=g.review_round AND e.action='approve'
                   AND e.hindi=g.hindi AND e.word=g.word)
               ORDER BY g.hindi,g.word""",
            (lang,),
        ).fetchall()
    return "".join(json.dumps({
        "hindi": row["hindi"], "target": row["word"], "language": lang,
        "data_type": "reviewed_word_pair",
        "reviewer_roles": (row["reviewer_roles"] or "").split(",") if row["reviewer_roles"] else [],
        "reviewed_at": row["reviewed_at"], "source": row["source"],
    }, ensure_ascii=False) + "\n" for row in rows)


def approve_item(item_id, final, reviewer_role):
    final = " ".join(str(final or "").split())
    if not final:
        raise core.ApiError(400, "The translation is empty. Type the translation before approving.")
    stale_audio = None
    stale_visual = None
    with db() as con:
        row = con.execute("SELECT * FROM items WHERE id=? AND model!=?", (item_id, DEMO_MODEL)).fetchone()
        if not row:
            raise core.ApiError(404, "That translation was not found. Reload the review queue.")
        if row["status"] == "cancelled":
            raise core.ApiError(409, "This sentence was removed from review. Upload its source again to create a new draft.")
        draft = row["draft"] or ""
        candidate = (row["final"] or draft) if row["status"] == "reviewing" else draft
        if row["status"] not in ("pending", "reviewing"):
            raise core.ApiError(409, "This translation is already approved or is no longer waiting for review.")
        prior = con.execute(
            "SELECT reviewer_role FROM review_events WHERE item_id=? AND review_round=? AND reviewer_role=? LIMIT 1",
            (item_id, row["review_round"], reviewer_role),
        ).fetchone()
        if prior:
            raise core.ApiError(409, "A different reviewer role must check this draft next.")
        cumulative_edited = int(final != draft)
        reviewer_changed = int(final != candidate)
        revision_changed = reviewer_changed
        if row["visual_filename"] and revision_changed:
            stale_visual = row["visual_filename"]
        ratio = 1 - difflib.SequenceMatcher(None, draft, final).ratio() if cumulative_edited else 0.0
        other_role_matches = con.execute(
            """SELECT 1 FROM review_events WHERE item_id=? AND review_round=?
               AND reviewer_role!=? AND final=? LIMIT 1""",
            (item_id, row["review_round"], reviewer_role, final),
        ).fetchone()
        revision_restarted = row["status"] == "reviewing" and revision_changed
        next_round = row["review_round"] + 1 if revision_restarted else row["review_round"]
        fully_approved = REQUIRED_REVIEWERS == 1 or (
            row["status"] == "reviewing" and bool(other_role_matches) and not revision_restarted
        )
        next_status = "approved" if fully_approved else "reviewing"
        reviewed = time.time()
        if row["audio_filename"] and row["audio_transcript"] != final:
            stale_audio = row["audio_filename"]
        updates = [
            "final=?", "explanation=''", "status=?", "edited=?", "edit_ratio=?", "approved=?", "review_round=?",
        ]
        values = [final, next_status, cumulative_edited, ratio,
                  reviewed if fully_approved else None, next_round]
        if stale_audio:
            updates.append(
                "audio_filename=NULL,audio_mime=NULL,audio_transcript=NULL,audio_author=NULL,"
                "audio_status='none',audio_reviewer=NULL,audio_reviewed=NULL,audio_generated=0,"
                "audio_generation_status='idle',audio_generation_error=NULL,audio_generation_author=NULL"
            )
        if stale_visual:
            updates.append(
                "visual_filename=NULL,visual_alt=NULL,visual_credit=NULL,visual_author=NULL,"
                "visual_status='none',visual_reviewer=NULL,visual_reviewed=NULL"
            )
        con.execute(f"UPDATE items SET {','.join(updates)} WHERE id=?", (*values, item_id))
        con.execute(
            "INSERT INTO review_events (item_id, reviewer_role, final, edited, edit_ratio, reviewed, review_round, explanation) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (item_id, reviewer_role, final, reviewer_changed, ratio, reviewed, next_round, ""),
        )
    remove_audio_file(stale_audio)
    remove_visual_file(stale_visual)
    return {"id": item_id, "edited": bool(reviewer_changed), "approved": fully_approved,
            "required_reviewers": REQUIRED_REVIEWERS,
            "awaiting_role": None if fully_approved or REQUIRED_REVIEWERS == 1 else
                ("community_expert" if reviewer_role == "teacher" else "teacher"),
            "revision_restarted": revision_restarted, "edit_ratio": ratio}


def reopen_lesson(item_id, note, reviewer_role):
    note = " ".join(str(note or "").split())
    if len(note) < 5:
        raise core.ApiError(400, "Explain what needs to be checked again.")
    with db() as con:
        row = con.execute(
            "SELECT status,review_round FROM items WHERE id=? AND model!=?",
            (item_id, DEMO_MODEL),
        ).fetchone()
        if not row:
            raise core.ApiError(404, "That approved lesson was not found.")
        if row["status"] != "approved":
            raise core.ApiError(409, "That lesson is no longer published. Reload the student view.")
        next_round = row["review_round"] + 1
        changed = con.execute(
            """UPDATE items SET status='reviewing',approved=NULL,review_round=?
               WHERE id=? AND status='approved'""",
            (next_round, item_id),
        )
        if changed.rowcount != 1:
            raise core.ApiError(409, "That lesson changed while it was being reopened. Reload the student view.")
        con.execute(
            """INSERT INTO lesson_correction_flags
               (item_id,review_round,reviewer_role,note,created) VALUES (?,?,?,?,?)""",
            (item_id, next_round, reviewer_role, note[:500], time.time()),
        )
    return {"id": item_id, "status": "reviewing",
            "message": "Lesson hidden from students and returned to review with your correction note."}


class TranslationRequest(BaseModel):
    text: str


class ApprovalRequest(BaseModel):
    final: str


class AudioReviewRequest(BaseModel):
    action: Literal["approve", "reject"]
    listened: bool = False


class VisualReviewRequest(BaseModel):
    action: Literal["approve", "reject"]
    viewed: bool = False


class VisualGenerationRequest(BaseModel):
    prompt: str = Field(min_length=5, max_length=1000)
    alt_text: str = Field(min_length=5, max_length=400)


class GlossarySubmissionRequest(BaseModel):
    lang: Literal["ho", "mun"]
    hindi: str = Field(max_length=60)
    word: str = Field(max_length=60)


class GlossaryReviewRequest(BaseModel):
    action: Literal["approve", "reject"]
    hindi: str = Field(default="", max_length=60)
    word: str = Field(default="", max_length=60)
    note: str = Field(default="", max_length=500)


class GlossaryFlagRequest(BaseModel):
    note: str = Field(min_length=5, max_length=500)


class LanguageExampleSubmissionRequest(BaseModel):
    lang: Literal["ho", "mun"]
    hindi: str = Field(max_length=1200)
    target: str = Field(max_length=2000)
    writing_system: str = Field(max_length=80)
    source_note: str = Field(default="", max_length=300)


class LanguageTranslationTaskRequest(BaseModel):
    lang: Literal["ho", "mun"]
    hindi: str = Field(max_length=1200)
    writing_system: str = Field(max_length=80)
    source_note: str = Field(default="", max_length=300)


class LanguageExampleReviewRequest(BaseModel):
    action: Literal["approve", "reject"]
    hindi: str = Field(default="", max_length=1200)
    target: str = Field(default="", max_length=2000)
    writing_system: str = Field(default="", max_length=80)
    source_note: str = Field(default="", max_length=300)
    note: str = Field(default="", max_length=500)


class LessonCorrectionRequest(BaseModel):
    note: str = Field(min_length=5, max_length=500)


class TeacherLoginRequest(BaseModel):
    pin: str = Field(min_length=8, max_length=256)


def teacher_token(authorization: str = Header(default="")):
    scheme, _, token = authorization.partition(" ")
    if scheme.casefold() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Sign in as a teacher or community language expert to continue.")
    with _auth_lock:
        role = _reviewer_tokens.get(token)
    if not role:
        raise HTTPException(status_code=401, detail="Reviewer session expired. Sign in again.")
    return {"token": token, "role": role}


def is_loopback_address(value):
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return value.casefold() == "localhost"


def split_page_text(text, tokenizer=None, max_tokens=110):
    """Join PDF line wraps, then group complete Hindi sentences within model limits."""
    text = text.replace("\u00ad", "").replace("\r", "\n")
    paragraphs = []
    for block in re.split(r"\n\s*\n+", text):
        # PDF extraction commonly inserts a newline at every printed line. Join those
        # wraps so the translator sees a sentence instead of disconnected fragments.
        lines = [" ".join(line.split()) for line in block.splitlines() if line.strip()]
        if lines:
            paragraphs.append(" ".join(lines))

    def size(value):
        if tokenizer is not None:
            return len(tokenizer(value, add_special_tokens=False, truncation=False)["input_ids"])
        return len(value)

    chunks = []
    current = []

    def flush():
        if current:
            chunks.append(" ".join(current))
            current.clear()

    sentence_pattern = re.compile(r"(?<=[।.!?])\s+")
    for paragraph in paragraphs:
        for sentence in (part.strip() for part in sentence_pattern.split(paragraph)):
            if not sentence:
                continue
            if size(sentence) > max_tokens:
                flush()
                # Split an unusually long sentence only at word boundaries.
                words = sentence.split()
                piece = []
                for word in words:
                    candidate = " ".join(piece + [word])
                    if piece and size(candidate) > max_tokens:
                        chunks.append(" ".join(piece))
                        piece = [word]
                    else:
                        piece.append(word)
                if piece:
                    chunks.append(" ".join(piece))
                continue
            candidate = " ".join(current + [sentence])
            if current and size(candidate) > max_tokens:
                flush()
            current.append(sentence)
        flush()  # keep paragraph boundaries intact
    flush()
    return chunks


def _hindi_letter_ratio(text):
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 0.0
    hindi_letters = sum(0x0900 <= ord(character) <= 0x097F for character in letters)
    return hindi_letters / len(letters)


def _hindi_text_quality(text):
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return 0.0
    letters = [character for character in visible if character.isalpha()]
    if not letters:
        return 0.0
    # Penalize pages where extraction is mostly numbers or symbols, while
    # giving Hindi script a modest preference for Hindi-source documents.
    return (len(letters) / len(visible)) * (0.7 + 0.3 * _hindi_letter_ratio(text))


def _looks_like_legacy_hindi_text_layer(text, source_language):
    """Flag Hindi PDF pages whose selectable text is mostly encoded outside Devanagari."""
    if source_language != "Hindi":
        return False
    letters = sum(character.isalpha() for character in text)
    return letters >= 20 and (
        _hindi_letter_ratio(text) < 0.50 or _hindi_text_quality(text) < 0.52
    )


def extract_pdf_sections(raw, tokenizer=None, source_language="Hindi"):
    """Extract page text and OCR blank or likely unreadable Hindi text layers."""
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(raw), strict=False)
    if reader.is_encrypted and not reader.decrypt(""):
        raise core.ApiError(422, "This PDF is password-protected. Upload an unlocked copy.")
    if len(reader.pages) > MAX_DOCUMENT_PAGES:
        raise core.ApiError(413, f"PDFs must have {MAX_DOCUMENT_PAGES} pages or fewer.")

    page_texts = []
    pages_needing_ocr = []
    pages_with_suspicious_hindi_text = set()
    for page_index, page in enumerate(reader.pages):
        # pypdf's extract_text raises KeyError('/Contents') on a valid blank
        # page. A page without a content stream cannot be a scanned image, so
        # leave it empty and let the final no-readable-text check explain it.
        if page.get("/Contents") is None:
            page_texts.append("")
            continue
        try:
            text = page.extract_text(extraction_mode="layout") or ""
        except TypeError:  # support pypdf versions without layout mode
            text = page.extract_text() or ""
        page_texts.append(text)
        if not text.strip():
            pages_needing_ocr.append(page_index)
        elif _looks_like_legacy_hindi_text_layer(text, source_language):
            pages_needing_ocr.append(page_index)
            pages_with_suspicious_hindi_text.add(page_index)

    ocr_pages = 0
    if pages_needing_ocr:
        try:
            import pypdfium2
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise core.ApiError(
                422,
                "This PDF has scanned or unreadable text pages. Install requirements-ocr.txt and Tesseract with the matching Hindi (hin) or English (eng) language data, restart the school server, then upload this file again.",
            ) from exc

        tesseract_cmd = os.environ.get("BS_TESSERACT_CMD", "").strip()
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        try:
            available_languages = set(pytesseract.get_languages(config=""))
        except Exception as exc:
            raise core.ApiError(
                422,
                "The local Tesseract OCR program was not found. Install Tesseract with the matching Hindi (hin) or English (eng) language data, set BS_TESSERACT_CMD if needed, restart the school server, then upload this file again.",
            ) from exc
        configured_ocr_language = os.environ.get("BS_OCR_LANGUAGE", "").strip()
        if configured_ocr_language:
            ocr_language = configured_ocr_language
        elif source_language == "Hindi" and {"hin", "eng"}.issubset(available_languages):
            ocr_language = "hin+eng"
        else:
            ocr_language = "hin" if source_language == "Hindi" else "eng"
        if not re.fullmatch(r"[A-Za-z0-9_+.\-]{2,32}", ocr_language):
            raise core.ApiError(500, "BS_OCR_LANGUAGE must be a Tesseract language code such as hin or eng.")
        requested_languages = set(ocr_language.split("+"))
        if not requested_languages.issubset(available_languages):
            missing = ", ".join(sorted(requested_languages - available_languages))
            raise core.ApiError(
                422,
                f"Tesseract is missing OCR language data ({missing}). Install it, restart the school server, then upload this PDF again.",
            )

        try:
            pdf = pypdfium2.PdfDocument(raw)
            try:
                for page_index in pages_needing_ocr:
                    page = pdf[page_index]
                    try:
                        bitmap = page.render(scale=300 / 72)
                        try:
                            image = bitmap.to_pil().convert("L")
                            with image:
                                ocr_text = pytesseract.image_to_string(
                                    image, lang=ocr_language, config="--psm 3", timeout=45,
                                ) or ""
                            previous_text = page_texts[page_index]
                            if not previous_text.strip():
                                page_texts[page_index] = ocr_text
                                ocr_pages += bool(ocr_text.strip())
                            elif page_index in pages_with_suspicious_hindi_text and ocr_text.strip():
                                old_quality = _hindi_text_quality(previous_text)
                                new_quality = _hindi_text_quality(ocr_text)
                                script_improved = (
                                    _hindi_letter_ratio(ocr_text) >=
                                    _hindi_letter_ratio(previous_text) + 0.10
                                )
                                if new_quality >= old_quality + 0.05 or (
                                    script_improved and new_quality >= old_quality
                                ):
                                    page_texts[page_index] = ocr_text
                                    ocr_pages += 1
                        finally:
                            bitmap.close()
                    finally:
                        page.close()
            finally:
                pdf.close()
        except RuntimeError as exc:
            raise core.ApiError(
                422,
                "OCR took too long on a scanned page. Try fewer pages or a clearer, lower-resolution scan.",
            ) from exc
        except core.ApiError:
            raise
        except Exception as exc:
            raise core.ApiError(
                422,
                f"Local OCR failed ({type(exc).__name__}). Check the Tesseract installation and requested language data.",
            ) from exc

    sections = []
    for page_number, text in enumerate(page_texts, 1):
        sections.extend(
            (page_number, sequence, chunk)
            for sequence, chunk in enumerate(split_page_text(text, tokenizer=tokenizer), 1)
        )
    return sections, len(reader.pages), ocr_pages


def extract_document_sections(raw, file_type, source_language, tokenizer=None):
    """Read text from supported classroom files and retain page/slide order."""
    sections = []
    per_page = {}

    def add_block(page_number, text):
        text = " ".join(str(text or "").split())
        if not text:
            return
        for chunk in split_page_text(text, tokenizer=tokenizer):
            per_page[page_number] = per_page.get(page_number, 0) + 1
            sections.append((page_number, per_page[page_number], chunk))

    if file_type == "pdf":
        return extract_pdf_sections(raw, tokenizer=tokenizer, source_language=source_language)
    if file_type == "txt":
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise core.ApiError(422, "Save the text file as UTF-8, then upload it again.") from exc
        for paragraph in re.split(r"\n\s*\n+", text):
            add_block(1, paragraph)
        return sections, 1, 0
    if file_type == "docx":
        try:
            from docx import Document as WordDocument
        except ImportError as exc:
            raise core.ApiError(503, "Word file support is missing. Run `python -m pip install -r requirements.txt`, restart the school server, then upload again.") from exc
        document = WordDocument(BytesIO(raw))
        for paragraph in core._docx_paragraphs(document):
            add_block(1, paragraph.text)
        return sections, 1, 0
    if file_type == "pptx":
        try:
            from pptx import Presentation
        except ImportError as exc:
            raise core.ApiError(503, "PowerPoint file support is missing. Run `python -m pip install -r requirements.txt`, restart the school server, then upload again.") from exc
        presentation = Presentation(BytesIO(raw))
        if len(presentation.slides) > MAX_DOCUMENT_PAGES:
            raise core.ApiError(413, f"PowerPoint files must have {MAX_DOCUMENT_PAGES} slides or fewer.")
        for slide_number, slide in enumerate(presentation.slides, 1):
            for text_frame in core._pptx_frames(slide.shapes):
                for paragraph in text_frame.paragraphs:
                    add_block(slide_number, paragraph.text)
        return sections, len(presentation.slides), 0
    raise core.ApiError(415, "Upload a PDF, DOCX, PPTX, or UTF-8 TXT file.")


def claim_next_job():
    with db() as con:
        row = con.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY id LIMIT 1"
        ).fetchone()
        if not row:
            return None
        con.execute(
            "UPDATE jobs SET status='processing', updated=? WHERE id=? AND status='queued'",
            (time.time(), row["id"]),
        )
        return dict(row)


def requeue_demo_documents(tokenizer):
    """Replace old fake drafts and rebuild their short line-based chunks."""
    now = time.time()
    with db() as con:
        document_rows = con.execute(
            """SELECT DISTINCT ds.document_id
               FROM document_segments ds JOIN items i ON i.id=ds.item_id
               WHERE i.model=? ORDER BY ds.document_id""",
            (DEMO_MODEL,),
        ).fetchall()
        for row in document_rows:
            document_id = int(row["document_id"])
            segments = con.execute(
                """SELECT ds.page_number, ds.segment_number, ds.hindi, ds.source_text, i.model
                   FROM document_segments ds LEFT JOIN items i ON i.id=ds.item_id
                   WHERE ds.document_id=? ORDER BY ds.page_number, ds.segment_number""",
                (document_id,),
            ).fetchall()
            if all(segment["model"] in (None, DEMO_MODEL) for segment in segments):
                # Earlier code split at every printed line. Join each page back
                # together, then split again on complete sentences and model tokens.
                pages = {}
                for segment in segments:
                    pages.setdefault(segment["page_number"], []).append(
                        (segment["hindi"], segment["source_text"] or segment["hindi"]),
                    )
                rebuilt = []
                for page_number, page_parts in pages.items():
                    original_text = " ".join(source for _hindi, source in page_parts)
                    for number, chunk in enumerate(
                        split_page_text(" ".join(hindi for hindi, _source in page_parts), tokenizer=tokenizer), 1,
                    ):
                        rebuilt.append((document_id, page_number, number, chunk, original_text))
                con.execute("DELETE FROM document_segments WHERE document_id=?", (document_id,))
                con.executemany(
                    """INSERT INTO document_segments
                       (document_id, page_number, segment_number, hindi, source_text) VALUES (?,?,?,?,?)""",
                    rebuilt,
                )
            else:
                con.execute(
                    """UPDATE document_segments SET item_id=NULL
                       WHERE document_id=? AND item_id IN (SELECT id FROM items WHERE model=?)""",
                    (document_id, DEMO_MODEL),
                )
            con.execute(
                "UPDATE documents SET status='queued', error=NULL WHERE id=?",
                (document_id,),
            )
            payload = json.dumps({"document_id": document_id})
            job = con.execute(
                "SELECT id FROM jobs WHERE kind='translate_document' AND payload=? ORDER BY id DESC LIMIT 1",
                (payload,),
            ).fetchone()
            if job:
                con.execute(
                    "UPDATE jobs SET status='queued', error=NULL, updated=? WHERE id=?",
                    (now, job["id"]),
                )
            else:
                con.execute(
                    "INSERT INTO jobs (kind, payload, status, created, updated) VALUES (?,?,?,?,?)",
                    ("translate_document", payload, "queued", now, now),
                )
    return len(document_rows)


def staged_document_path(staging_name):
    if not isinstance(staging_name, str) or not re.fullmatch(r"[a-f0-9]{32}\.(?:pdf|docx|pptx|txt)", staging_name):
        return None
    return PDF_STAGING_DIR / staging_name


def remove_staged_document(staging_name):
    path = staged_document_path(staging_name)
    if path:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def clean_orphan_staged_documents():
    if not PDF_STAGING_DIR.is_dir():
        return
    with db() as con:
        jobs = con.execute(
            "SELECT payload FROM jobs WHERE kind='prepare_document' AND status='queued'",
        ).fetchall()
    retained = set()
    for job in jobs:
        try:
            retained.add(json.loads(job["payload"]).get("staging_name"))
        except (TypeError, json.JSONDecodeError):
            continue
    for path in PDF_STAGING_DIR.iterdir():
        if re.fullmatch(r"[a-f0-9]{32}\.(?:pdf|docx|pptx|txt)", path.name) and path.name not in retained:
            remove_staged_document(path.name)


# Keep these names for deployments that have queued legacy PDF jobs in SQLite.
staged_pdf_path = staged_document_path
remove_staged_pdf = remove_staged_document
clean_orphan_staged_pdfs = clean_orphan_staged_documents


class CancelledDocumentJob(Exception):
    """Raised when a reviewer removes a document while a worker is handling it."""


def _document_job_is_cancelled(document_id, job_id):
    with db() as con:
        row = con.execute(
            """SELECT d.status AS document_status,j.status AS job_status
               FROM documents d LEFT JOIN jobs j ON j.id=? WHERE d.id=?""",
            (job_id, document_id),
        ).fetchone()
    return not row or row["document_status"] == "cancelled" or row["job_status"] == "cancelled"


def _remove_student_pdf_cache(document_id):
    with _student_pdf_generation_lock:
        try:
            STUDENT_PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            for path in STUDENT_PDF_CACHE_DIR.glob(f"{int(document_id)}-*.pdf"):
                if path.parent.resolve() == STUDENT_PDF_CACHE_DIR.resolve():
                    path.unlink(missing_ok=True)
        except OSError:
            pass


def _purge_cancelled_document(document_id):
    """Remove a cancelled upload and cancel unshared, unfinished sentence tasks."""
    with db() as con:
        document = con.execute("SELECT status FROM documents WHERE id=?", (document_id,)).fetchone()
        if not document or document["status"] != "cancelled":
            return
        segments = con.execute(
            "SELECT item_id,language_example_id FROM document_segments WHERE document_id=?", (document_id,),
        ).fetchall()
        item_ids = [int(row["item_id"]) for row in segments if row["item_id"] is not None]
        example_ids = [int(row["language_example_id"]) for row in segments
                       if row["language_example_id"] is not None]
        for item_id in item_ids:
            other_source = con.execute(
                "SELECT 1 FROM document_segments WHERE item_id=? AND document_id!=? LIMIT 1",
                (item_id, document_id),
            ).fetchone()
            if not other_source:
                changed = con.execute(
                    "UPDATE items SET status='cancelled',approved=NULL WHERE id=? AND status IN ('pending','reviewing')",
                    (item_id,),
                )
                if changed.rowcount:
                    con.execute("DELETE FROM review_events WHERE item_id=?", (item_id,))
                    con.execute("DELETE FROM lesson_correction_flags WHERE item_id=?", (item_id,))
        for example_id in example_ids:
            other_source = con.execute(
                "SELECT 1 FROM document_segments WHERE language_example_id=? AND document_id!=? LIMIT 1",
                (example_id, document_id),
            ).fetchone()
            if not other_source:
                changed = con.execute(
                    "UPDATE language_examples SET status='cancelled',approved=NULL WHERE id=? AND status IN ('pending','reviewing')",
                    (example_id,),
                )
                if changed.rowcount:
                    con.execute("DELETE FROM language_example_reviews WHERE example_id=?", (example_id,))
        con.execute("DELETE FROM document_segments WHERE document_id=?", (document_id,))
        con.execute("DELETE FROM upload_requests WHERE document_id=?", (document_id,))
        con.execute("DELETE FROM documents WHERE id=? AND status='cancelled'", (document_id,))
    _remove_student_pdf_cache(document_id)


def _ensure_document_job_active(document_id, job_id):
    if _document_job_is_cancelled(document_id, job_id):
        raise CancelledDocumentJob("Study material removal requested.")


@_translation_resource_guard
def prepare_document(document_id, job_id, staging_name):
    path = staged_document_path(staging_name)
    if not path or not path.is_file():
        raise core.ApiError(500, "The staged lesson file is missing. Upload the document again.")
    try:
        raw = path.read_bytes()
        _ensure_document_job_active(document_id, job_id)
        with db() as con:
            document = con.execute(
                "SELECT language,source_language,status FROM documents WHERE id=?", (document_id,),
            ).fetchone()
        if not document or document["status"] == "cancelled":
            raise CancelledDocumentJob("Study material removal requested.")
        if document["language"].casefold() == SUPPORTED_LANGUAGE.casefold():
            translator = core.get_translator()
            tokenizer = None if translator.mock else translator.tok
        else:
            # Ho and Mundari currently use a Hindi source for human translation.
            tokenizer = None
        file_type = path.suffix.lower().lstrip(".")
        sections, page_count, ocr_pages = extract_document_sections(
            raw, file_type, document["source_language"], tokenizer,
        )
        _ensure_document_job_active(document_id, job_id)
        if not sections:
            raise core.ApiError(
                422,
                "No text could be extracted. Check that the document contains selectable text or that scanned pages are clear.",
            )
        if len(sections) > MAX_DOCUMENT_SEGMENTS:
            raise core.ApiError(413, f"This file has too many text sections. Keep it under {MAX_DOCUMENT_SEGMENTS} sections.")
        now = time.time()
        with db() as con:
            active_document = con.execute(
                "SELECT status FROM documents WHERE id=?", (document_id,),
            ).fetchone()
            active_job = con.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not active_document or active_document["status"] == "cancelled" or not active_job or active_job["status"] == "cancelled":
                raise CancelledDocumentJob("Study material removal requested.")
            con.executemany(
                "INSERT OR IGNORE INTO document_segments (document_id,page_number,segment_number,hindi,source_text) VALUES (?,?,?,?,?)",
                [(document_id, page_number, sequence, text if document["source_language"] == "Hindi" else "", text)
                 for page_number, sequence, text in sections],
            )
            con.execute(
                "UPDATE documents SET status='queued', page_count=?, ocr_pages=?, error=NULL WHERE id=?",
                (page_count, ocr_pages, document_id),
            )
            payload = json.dumps({"document_id": document_id})
            con.execute(
                "INSERT INTO jobs (kind, payload, status, created, updated) VALUES (?,?,?,?,?)",
                ("translate_document", payload, "queued", now, now),
            )
            con.execute("UPDATE jobs SET status='done', error=NULL, updated=? WHERE id=?", (now, job_id))
    finally:
        remove_staged_document(staging_name)


def create_pdf_translation_tasks(document_id, job_id):
    now = time.time()
    with db() as con:
        document = con.execute(
            "SELECT filename,language,writing_system,status FROM documents WHERE id=?", (document_id,),
        ).fetchone()
        job = con.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not document or document["status"] == "cancelled" or not job or job["status"] == "cancelled":
            raise CancelledDocumentJob("Study material removal requested.")
        lang = "ho" if document["language"].casefold() == "ho" else "mun"
        segments = con.execute(
            """SELECT id,page_number,segment_number,hindi FROM document_segments
               WHERE document_id=? AND language_example_id IS NULL
               ORDER BY page_number,segment_number""", (document_id,),
        ).fetchall()
        for segment in segments:
            source_note = f"PDF: {document['filename']} · page {segment['page_number']} · section {segment['segment_number']}"
            cur = con.execute(
                """INSERT INTO language_examples
                   (lang,hindi,target,writing_system,source_note,contributor_role,status,created)
                   VALUES (?,?, '', ?, ?, 'system', 'pending', ?)""",
                (lang, segment["hindi"], document["writing_system"], source_note, now),
            )
            con.execute(
                "UPDATE document_segments SET language_example_id=? WHERE id=?",
                (cur.lastrowid, segment["id"]),
            )
        con.execute("UPDATE documents SET status='reviewing',error=NULL WHERE id=?", (document_id,))
        con.execute("UPDATE jobs SET status='done',error=NULL,updated=? WHERE id=?", (now, job_id))


def process_document(document_id, job_id):
    _ensure_document_job_active(document_id, job_id)
    with db() as con:
        document = con.execute(
            "SELECT language,source_language,status FROM documents WHERE id=?", (document_id,),
        ).fetchone()
        if not document or document["status"] == "cancelled":
            raise CancelledDocumentJob("Study material removal requested.")
        language = document["language"]
        source_language = document["source_language"]
        if language.casefold() != SUPPORTED_LANGUAGE.casefold():
            con.execute("UPDATE documents SET status='processing', error=NULL WHERE id=?", (document_id,))
            human_translation = True
        else:
            human_translation = False
            con.execute("UPDATE documents SET status='processing', error=NULL WHERE id=?", (document_id,))
        segments = con.execute(
            "SELECT id,hindi,source_text FROM document_segments WHERE document_id=? AND item_id IS NULL ORDER BY page_number,segment_number",
            (document_id,),
        ).fetchall()

    if human_translation:
        create_pdf_translation_tasks(document_id, job_id)
        return

    batch_size = core.PDF_TRANSLATION_BATCH_SIZE
    translator = core.get_translator() if source_language == "English" else None
    for start in range(0, len(segments), batch_size):
        _ensure_document_job_active(document_id, job_id)
        batch = [dict(segment) for segment in segments[start:start + batch_size]]
        if translator is not None:
            hindi_bridge = translator.translate_tagged_many(
                [segment["source_text"] for segment in batch], "eng_Latn", "hin_Deva",
            )
            _ensure_document_job_active(document_id, job_id)
            for segment, hindi in zip(batch, hindi_bridge):
                segment["hindi"] = hindi
            with db() as con:
                con.executemany(
                    "UPDATE document_segments SET hindi=? WHERE id=?",
                    [(segment["hindi"], segment["id"]) for segment in batch],
                )
        items = add_translations([segment["hindi"] for segment in batch])
        with db() as con:
            con.executemany(
                "UPDATE document_segments SET item_id=? WHERE id=?",
                [(item["id"], segment["id"]) for segment, item in zip(batch, items)],
            )
        try:
            _ensure_document_job_active(document_id, job_id)
        except CancelledDocumentJob:
            with db() as con:
                for item in items:
                    if item.get("cached"):
                        continue
                    con.execute(
                        """UPDATE items SET status='cancelled',approved=NULL WHERE id=?
                           AND status IN ('pending','reviewing')
                           AND NOT EXISTS (SELECT 1 FROM document_segments WHERE item_id=items.id)""",
                        (item["id"],),
                    )
            raise

    _ensure_document_job_active(document_id, job_id)
    with db() as con:
        active_document = con.execute("SELECT status FROM documents WHERE id=?", (document_id,)).fetchone()
        active_job = con.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not active_document or active_document["status"] == "cancelled" or not active_job or active_job["status"] == "cancelled":
            raise CancelledDocumentJob("Study material removal requested.")
        con.execute("UPDATE documents SET status='ready' WHERE id=?", (document_id,))
        con.execute("UPDATE jobs SET status='done', updated=? WHERE id=?", (time.time(), job_id))

    if language.casefold() == SUPPORTED_LANGUAGE.casefold():
        try:
            draft_document = _santali_draft_whole_document(document_id)
        except core.ApiError:
            if _document_job_is_cancelled(document_id, job_id):
                raise CancelledDocumentJob("Study material removal requested.")
            raise
        if draft_document:
            try:
                cached_student_document_pdf(
                    draft_document["document"], draft_document["translated_sections"], "ai_draft",
                    draft_document["source_sections"],
                )
            except Exception as exc:
                print(f"Could not prebuild the student PDF for document {document_id}: {exc}")


def _drain_santali_tts_stderr(process):
    """Drain helper logs so a verbose model cannot block on a full stderr pipe."""
    while True:
        try:
            line = process.stderr.readline()
        except (OSError, ValueError):
            return
        if not line:
            return
        decoded = line.decode("utf-8", errors="replace").rstrip()
        if decoded:
            with _santali_tts_stderr_lock:
                _santali_tts_stderr_tail.append(decoded[:500])
                del _santali_tts_stderr_tail[:-20]


def _recent_santali_tts_log(max_chars=260):
    with _santali_tts_stderr_lock:
        recent_output = "\n".join(_santali_tts_stderr_tail[-8:])
    recent_output = re.sub(r"hf_[A-Za-z0-9]{16,}", "hf_[redacted]", recent_output)
    return recent_output[-max_chars:].strip()


def _get_santali_tts_process():
    global _santali_tts_process, _santali_tts_stderr_thread
    process = _santali_tts_process
    if process is not None and process.poll() is None:
        return process
    script = core.HERE / "santali_tts.py"
    if not script.is_file():
        raise RuntimeError("The Santali speech helper file santali_tts.py is missing.")
    try:
        process = subprocess.Popen(
            [sys.executable, str(script), "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            cwd=str(core.HERE),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as exc:
        raise RuntimeError(f"Could not start the isolated Santali speech helper: {exc}") from exc
    _santali_tts_process = process
    with _santali_tts_stderr_lock:
        _santali_tts_stderr_tail.clear()
    _santali_tts_stderr_thread = threading.Thread(
        target=_drain_santali_tts_stderr, args=(process,),
        name="bhasha-santali-tts-log", daemon=True,
    )
    _santali_tts_stderr_thread.start()
    return process


def _generate_santali_audio_file(text, output_path):
    """Ask the isolated TTS process to write a WAV, keeping model crashes out of Uvicorn."""
    global _santali_tts_process
    with _santali_tts_lock:
        process = _get_santali_tts_process()
        request = json.dumps({"text": text, "output_path": str(output_path)}, ensure_ascii=False)
        try:
            process.stdin.write((request + "\n").encode("utf-8"))
            process.stdin.flush()
            response_line = process.stdout.readline()
        except (BrokenPipeError, OSError, ValueError) as exc:
            response_line = b""
            failure = str(exc).strip()
        else:
            failure = ""
        if not response_line:
            try:
                exit_code = process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    process.terminate()
                    exit_code = process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        process.kill()
                        exit_code = process.wait(timeout=2)
                    except (OSError, subprocess.TimeoutExpired):
                        exit_code = process.poll()
            if _santali_tts_stderr_thread is not None:
                _santali_tts_stderr_thread.join(timeout=1)
            _santali_tts_process = None
            reason = f" (exit code {exit_code})" if exit_code is not None else ""
            recent_output = _recent_santali_tts_log()
            diagnostic = f" Helper log tail: {recent_output}" if recent_output else " No helper log output was captured."
            raise RuntimeError(
                "Santali speech helper stopped before returning audio" + reason + ". " +
                (failure + ". " if failure else "") + diagnostic
            )
        try:
            response = json.loads(response_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("The Santali speech process returned an unreadable result; retry the audio job.") from exc
        if not response.get("ok"):
            error = str(response.get("error") or "Santali audio generation failed.")[:260]
            recent_output = _recent_santali_tts_log(180)
            diagnostic = f" Helper log tail: {recent_output}" if recent_output else ""
            raise RuntimeError(error + diagnostic)
        timings = response.get("timings") or {}
        if timings:
            print(
                "[Santali TTS] model load: "
                f"{float(timings.get('model_load_seconds', 0)):.1f}s; synthesis: "
                f"{float(timings.get('synthesis_seconds', 0)):.1f}s; "
                f"text: {int(timings.get('characters', 0))} characters",
                flush=True,
            )


def _stop_santali_tts_helper():
    global _santali_tts_process
    process = _santali_tts_process
    _santali_tts_process = None
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            pass


def process_santali_audio_generation(item_id, job_id, reviewer_role, transcript):
    with db() as con:
        row = con.execute(
            """SELECT status,final,audio_status,audio_author,audio_filename,
                      audio_generation_status,audio_generation_author
               FROM items WHERE id=? AND model!=?""",
            (item_id, DEMO_MODEL),
        ).fetchone()
        if (not row or row["status"] != "approved" or row["final"] != transcript or
                row["audio_generation_author"] != reviewer_role or
                row["audio_status"] not in ("none", "rejected") or
                (row["audio_status"] == "rejected" and row["audio_author"] != reviewer_role)):
            raise core.ApiError(409, "This lesson changed or is no longer eligible for speech generation. Reload Audio and try again.")
        con.execute(
            "UPDATE items SET audio_generation_status='processing',audio_generation_error=NULL WHERE id=?",
            (item_id,),
        )

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    temp_audio = AUDIO_DIR / f".tts-pending-{item_id}-{secrets.token_hex(12)}.wav"
    try:
        with _model_resource_lock:
            # The translation model is already resident after startup. Release it
            # before loading Parler-TTS to lower Windows' committed memory demand.
            core.release_translator()
            _generate_santali_audio_file(transcript, temp_audio)
            if not temp_audio.is_file():
                raise RuntimeError("The Santali speech process completed without creating its WAV recording.")
            if temp_audio.stat().st_size > MAX_GENERATED_AUDIO_BYTES:
                raise RuntimeError("Generated audio is over 24 MB. Split this lesson into shorter sentences and retry.")
            audio = temp_audio.read_bytes()
    finally:
        try:
            temp_audio.unlink(missing_ok=True)
        except OSError:
            pass
    if not audio.startswith(b"RIFF") or audio[8:12] != b"WAVE":
        raise RuntimeError("The speech model did not return a valid WAV recording.")

    filename = f"{item_id}-{secrets.token_hex(16)}.wav"
    destination = AUDIO_DIR / filename
    destination.write_bytes(audio)
    old_filename = None
    try:
        with db() as con:
            current = con.execute(
                """SELECT status,final,audio_status,audio_author,
                          audio_generation_status,audio_generation_author
                   FROM items WHERE id=? AND model!=?""",
                (item_id, DEMO_MODEL),
            ).fetchone()
            if (not current or current["status"] != "approved" or current["final"] != transcript or
                    current["audio_generation_status"] != "processing" or
                    current["audio_generation_author"] != reviewer_role or
                    current["audio_status"] not in ("none", "rejected") or
                    (current["audio_status"] == "rejected" and current["audio_author"] != reviewer_role)):
                raise core.ApiError(409, "This lesson changed while its audio was being generated. Reload Audio and try again.")
            old = con.execute("SELECT audio_filename FROM items WHERE id=?", (item_id,)).fetchone()
            old_filename = old["audio_filename"] if old else None
            con.execute(
                """UPDATE items SET audio_filename=?,audio_mime='audio/wav',audio_transcript=?,audio_author=?,
                   audio_status='awaiting_review',audio_reviewer=NULL,audio_reviewed=NULL,audio_generated=1,
                   audio_generation_status='done',audio_generation_error=NULL,audio_generation_author=NULL
                   WHERE id=?""",
                (filename, transcript, reviewer_role, item_id),
            )
            con.execute("UPDATE jobs SET status='done',error=NULL,updated=? WHERE id=?", (time.time(), job_id))
    except Exception:
        remove_audio_file(filename)
        raise
    remove_audio_file(old_filename)


def worker_loop():
    while not _stop_worker.is_set():
        job = claim_next_job()
        if not job:
            _stop_worker.wait(0.75)
            continue
        payload = {}
        try:
            payload = json.loads(job["payload"])
            if job["kind"] == "prepare_document":
                prepare_document(int(payload["document_id"]), job["id"], payload["staging_name"])
            elif job["kind"] == "translate_document":
                process_document(int(payload["document_id"]), job["id"])
            elif job["kind"] == "generate_santali_audio":
                process_santali_audio_generation(
                    int(payload["item_id"]), job["id"], payload["reviewer_role"], payload["transcript"],
                )
            else:
                raise ValueError("Unknown background job type.")
        except CancelledDocumentJob:
            document_id = payload.get("document_id") if isinstance(payload, dict) else None
            if isinstance(payload, dict):
                remove_staged_document(payload.get("staging_name"))
            with db() as con:
                con.execute(
                    "UPDATE jobs SET status='cancelled',error=NULL,updated=? WHERE id=?",
                    (time.time(), job["id"]),
                )
            if document_id is not None:
                _purge_cancelled_document(int(document_id))
        except Exception as exc:
            cancelled_document_id = payload.get("document_id") if isinstance(payload, dict) else None
            if cancelled_document_id is not None and _document_job_is_cancelled(
                    int(cancelled_document_id), int(job["id"])):
                if isinstance(payload, dict):
                    remove_staged_document(payload.get("staging_name"))
                with db() as con:
                    con.execute(
                        "UPDATE jobs SET status='cancelled',error=NULL,updated=? WHERE id=?",
                        (time.time(), job["id"]),
                    )
                _purge_cancelled_document(int(cancelled_document_id))
                continue
            error_limit = 900 if isinstance(payload, dict) and payload.get("item_id") is not None else 400
            message = str(exc).strip()[:error_limit] or "Document processing failed."
            document_id = payload.get("document_id") if isinstance(payload, dict) else None
            if isinstance(payload, dict):
                remove_staged_pdf(payload.get("staging_name"))
            with db() as con:
                con.execute(
                    "UPDATE documents SET status='error', error=? WHERE id=? AND status!='cancelled'",
                    (message, document_id),
                )
                con.execute(
                    "UPDATE jobs SET status='error', error=?, updated=? WHERE id=? AND status!='cancelled'",
                    (message, time.time(), job["id"]),
                )
                if isinstance(payload, dict) and payload.get("item_id") is not None:
                    con.execute(
                        """UPDATE items SET audio_generation_status='error',audio_generation_error=?
                           WHERE id=? AND audio_generation_author=?
                             AND audio_generation_status IN ('queued','processing')""",
                        (message, int(payload["item_id"]), payload.get("reviewer_role")),
                    )


@asynccontextmanager
async def lifespan(_app):
    global _worker_thread
    if len(TEACHER_PIN) < 8:
        raise RuntimeError("Set BS_TEACHER_PIN to a teacher passcode with at least 8 characters before starting Bhasha Setu.")
    if len(EXPERT_PIN) < 8:
        raise RuntimeError("Set BS_EXPERT_PIN to a community expert passcode with at least 8 characters before starting Bhasha Setu.")
    if secrets.compare_digest(TEACHER_PIN.encode("utf-8"), EXPERT_PIN.encode("utf-8")):
        raise RuntimeError("Use different passcodes for BS_TEACHER_PIN and BS_EXPERT_PIN.")
    # Load the model once before accepting classroom requests.
    core.get_translator()
    with db() as con:
        # Fail interrupted TTS work instead of loading its large model at startup;
        # unfinished PDF work is requeued and completed segments stay linked.
        interrupted_audio = "Santali audio generation stopped when the server exited. Retry it from the Audio dashboard."
        con.execute(
            """UPDATE jobs SET status='error',error=?,updated=?
               WHERE kind='generate_santali_audio' AND status IN ('queued','processing')""",
            (interrupted_audio, time.time()),
        )
        con.execute(
            """UPDATE items SET audio_generation_status='error',audio_generation_error=?,
               audio_generation_author=NULL WHERE audio_generation_status IN ('queued','processing')""",
            (interrupted_audio,),
        )
        con.execute("UPDATE jobs SET status='queued', error=NULL WHERE status='processing'")
        con.execute("UPDATE documents SET status='queued' WHERE status='processing'")
        cancelled_documents = [row["id"] for row in con.execute(
            "SELECT id FROM documents WHERE status='cancelled'",
        ).fetchall()]
    for document_id in cancelled_documents:
        _purge_cancelled_document(int(document_id))
    clean_orphan_staged_pdfs()
    if not core.MOCK:
        # Older demo runs wrote fake drafts into the main database. Detach them so
        # the worker replaces those records using the real translator.
        requeued = requeue_demo_documents(core.get_translator().tok)
        if requeued:
            print(f"Re-queued {requeued} PDF lesson(s) that previously contained fake demo output.")
    _stop_worker.clear()
    _worker_thread = threading.Thread(target=worker_loop, name="bhasha-job-worker", daemon=True)
    _worker_thread.start()
    yield
    _stop_worker.set()
    if _worker_thread:
        _worker_thread.join(timeout=3)
    _stop_santali_tts_helper()


app = FastAPI(title="Bhasha Setu", version="0.2.0", lifespan=lifespan)


@app.middleware("http")
async def require_https_for_classroom_devices(request: Request, call_next):
    client_host = request.client.host if request.client else "127.0.0.1"
    if not is_loopback_address(client_host) and request.url.scheme != "https":
        return JSONResponse(
            {"detail": "Classroom devices must connect over HTTPS. Configure a trusted TLS certificate on the school server."},
            status_code=426,
        )
    return await call_next(request)


@app.exception_handler(core.ApiError)
async def api_error_handler(_request, exc):
    return JSONResponse({"detail": exc.msg}, status_code=exc.status)


@app.get("/", response_class=FileResponse)
def home():
    return FileResponse(core.HERE / "index.html", media_type="text/html; charset=utf-8")


@app.get("/manifest.webmanifest")
def manifest():
    return JSONResponse({
        "name": "Bhasha Setu",
        "short_name": "Bhasha Setu",
        "description": "Offline lessons in students' mother tongues",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#f4f7fb",
        "theme_color": "#183d68",
        "icons": [{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any"}],
    }, media_type="application/manifest+json")


@app.get("/icon.svg")
def icon():
    return FileResponse(core.HERE / "icon.svg", media_type="image/svg+xml")


@app.get("/learning-resources/{filename}")
def learning_resource(filename: str):
    if filename not in {"ho_hindi.pdf", "ho_hindi_compressed.pdf", "mundari_hundi.pdf"}:
        raise HTTPException(status_code=404, detail="Learning resource not found.")
    path = core.HERE / "learning_resources" / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Learning resource not found.")
    return FileResponse(path, media_type="application/pdf", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/sw.js")
def service_worker():
    return Response(SERVICE_WORKER_JS, media_type="application/javascript; charset=utf-8",
                    headers={"Service-Worker-Allowed": "/"})


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/api/health")
def health():
    return {
        "model": core.get_translator_label(),
        "languages": [SUPPORTED_LANGUAGE],
        "required_reviewers": REQUIRED_REVIEWERS,
        "capabilities": {
            "curriculum_translation": [SUPPORTED_LANGUAGE],
            "student_document_pdfs": True,
            "student_document_audio_text": True,
            "reviewed_word_lookup": ["Ho", "Mundari"],
            "human_sentence_corpus": ["Ho", "Mundari"],
            "reviewed_language_lessons": ["Ho", "Mundari"],
            "human_translation_queue": ["Ho", "Mundari"],
            "community_recorded_audio": [SUPPORTED_LANGUAGE],
            "generated_speech": [SUPPORTED_LANGUAGE],
        },
    }


@app.post("/api/teacher/login")
@app.post("/api/reviewer/login")
def teacher_login(body: TeacherLoginRequest, request: Request):
    client_host = request.client.host if request.client else "unknown"
    now = time.time()
    with _auth_lock:
        attempts, locked_until = _login_attempts.get(client_host, (0, 0))
        if now < locked_until:
            raise HTTPException(status_code=429, detail="Too many incorrect attempts. Try again in one minute.")
        pin = body.pin.encode("utf-8")
        role = None
        if secrets.compare_digest(pin, TEACHER_PIN.encode("utf-8")):
            role = "teacher"
        elif secrets.compare_digest(pin, EXPERT_PIN.encode("utf-8")):
            role = "community_expert"
        if role is None:
            attempts += 1
            _login_attempts[client_host] = (0, now + 60) if attempts >= 5 else (attempts, 0)
            raise HTTPException(status_code=401, detail="That reviewer passcode is not correct.")
        _login_attempts.pop(client_host, None)
        token = secrets.token_urlsafe(32)
        _reviewer_tokens[token] = role
    return {"access_token": token, "token_type": "bearer", "role": role}


@app.post("/api/teacher/logout")
def teacher_logout(token: dict = Depends(teacher_token)):
    with _auth_lock:
        _reviewer_tokens.pop(token["token"], None)
    return {"ok": True}


@app.get("/api/queue")
def queue(reviewer: dict = Depends(teacher_token)):
    return get_queue(reviewer["role"])


@app.get("/api/reviews/mine")
def my_reviews(reviewer: dict = Depends(teacher_token)):
    return get_my_reviews(reviewer["role"])


@app.get("/api/approved")
def approved():
    return get_approved()


@app.post("/api/approved/{item_id}/reopen")
def reopen_approved_lesson(item_id: int, body: LessonCorrectionRequest,
                           reviewer: dict = Depends(teacher_token)):
    return reopen_lesson(item_id, body.note, reviewer["role"])


@app.get("/api/audio/work")
def audio_work(offset: int = 0, limit: int = 20, reviewer: dict = Depends(teacher_token)):
    return get_audio_work(reviewer["role"], limit, offset)


@app.get("/api/audio/mine")
def audio_mine(reviewer: dict = Depends(teacher_token)):
    return get_my_audio(reviewer["role"])


@app.get("/api/language-audio/work")
def language_audio_work(reviewer: dict = Depends(teacher_token)):
    return get_language_audio_work(reviewer["role"])


@app.get("/api/language-audio/mine")
def language_audio_mine(reviewer: dict = Depends(teacher_token)):
    return get_my_language_audio(reviewer["role"])


@app.get("/api/visual/work")
def visual_work(reviewer: dict = Depends(teacher_token)):
    return get_visual_work(reviewer["role"])


@app.get("/api/visual/mine")
def visual_mine(reviewer: dict = Depends(teacher_token)):
    return get_my_visuals(reviewer["role"])


def audio_response(item_id, require_approved=False):
    with db() as con:
        row = con.execute(
            "SELECT status,audio_status,audio_filename,audio_mime FROM items WHERE id=?",
            (item_id,),
        ).fetchone()
    if not row or not row["audio_filename"]:
        raise core.ApiError(404, "That recording is not available.")
    if require_approved and (row["status"] != "approved" or row["audio_status"] != "approved"):
        raise core.ApiError(404, "That approved recording is not available.")
    if not re.fullmatch(r"\d+-[a-f0-9]{32}\.(webm|ogg|wav|mp4|mp3)", row["audio_filename"]):
        raise core.ApiError(404, "That recording is not available.")
    path = AUDIO_DIR / row["audio_filename"]
    if not path.is_file():
        raise core.ApiError(404, "That recording file is missing from the school server.")
    return FileResponse(path, media_type=row["audio_mime"] or "application/octet-stream",
                        headers={"Cache-Control": "private, no-cache"})


@app.get("/api/audio/{item_id}")
def get_review_audio(item_id: int, _reviewer: dict = Depends(teacher_token)):
    return audio_response(item_id)


@app.get("/api/approved-audio/{item_id}")
def get_approved_audio(item_id: int):
    return audio_response(item_id, require_approved=True)


def language_audio_response(example_id, require_approved=False):
    with db() as con:
        row = con.execute(
            "SELECT status,audio_status,audio_filename,audio_mime FROM language_examples WHERE id=?",
            (example_id,),
        ).fetchone()
    if not row or not row["audio_filename"]:
        raise core.ApiError(404, "That community recording is not available.")
    if require_approved and (row["status"] != "approved" or row["audio_status"] != "approved"):
        raise core.ApiError(404, "That approved community recording is not available.")
    if not re.fullmatch(rf"lang-{example_id}-[a-f0-9]{{32}}\.(webm|ogg|wav|mp4|mp3)", row["audio_filename"]):
        raise core.ApiError(404, "That community recording is not available.")
    path = AUDIO_DIR / row["audio_filename"]
    if not path.is_file():
        raise core.ApiError(404, "That community recording file is missing from the school server.")
    return FileResponse(path, media_type=row["audio_mime"] or "application/octet-stream",
                        headers={"Cache-Control": "private, no-cache"})


@app.get("/api/language-audio/{example_id}")
def get_language_review_audio(example_id: int, reviewer: dict = Depends(teacher_token)):
    with db() as con:
        row = con.execute(
            "SELECT status,audio_status,audio_author FROM language_examples WHERE id=?", (example_id,),
        ).fetchone()
    if not row or row["status"] != "approved" or row["audio_status"] != "awaiting_review":
        raise core.ApiError(404, "That community recording is not waiting for review.")
    if row["audio_author"] == reviewer["role"]:
        raise core.ApiError(403, "A different reviewer role must listen to this recording.")
    return language_audio_response(example_id)


@app.get("/api/language-approved-audio/{example_id}")
def get_approved_language_audio(example_id: int):
    return language_audio_response(example_id, require_approved=True)


def visual_file_response(filename):
    if not filename or not re.fullmatch(r"\d+-[a-f0-9]{32}\.jpg", filename):
        raise core.ApiError(404, "That lesson visual is not available.")
    path = VISUAL_DIR / filename
    if not path.is_file():
        raise core.ApiError(404, "That lesson visual is missing from the school server.")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, no-cache"})


@app.get("/api/visual/{item_id}")
def get_review_visual(item_id: int, reviewer: dict = Depends(teacher_token)):
    with db() as con:
        row = con.execute(
            "SELECT status,visual_status,visual_author,visual_filename FROM items WHERE id=? AND model!=?",
            (item_id, DEMO_MODEL),
        ).fetchone()
    if not row or row["status"] != "approved" or row["visual_status"] != "awaiting_review":
        raise core.ApiError(404, "That lesson visual is not waiting for review.")
    return visual_file_response(row["visual_filename"])


@app.get("/api/approved-visuals/{item_id}")
def get_approved_visual(item_id: int):
    with db() as con:
        row = con.execute(
            "SELECT status,visual_status,visual_filename FROM items WHERE id=? AND model!=?",
            (item_id, DEMO_MODEL),
        ).fetchone()
    if not row or row["status"] != "approved" or row["visual_status"] != "approved":
        raise core.ApiError(404, "That approved lesson visual is not available.")
    return visual_file_response(row["visual_filename"])


@app.get("/api/stats")
def stats(_token: dict = Depends(teacher_token)):
    return get_stats()


@app.get("/api/lookup")
def lookup(q: str = ""):
    return lookup_words(q)


@app.get("/api/glossary/work")
def glossary_work(reviewer: dict = Depends(teacher_token)):
    return get_glossary_work(reviewer["role"])


@app.post("/api/glossary", status_code=201)
def glossary_add(body: GlossarySubmissionRequest, reviewer: dict = Depends(teacher_token)):
    return add_glossary_entry(body.lang, body.hindi, body.word, reviewer["role"])


@app.post("/api/glossary/{entry_id}/review")
def glossary_review(entry_id: int, body: GlossaryReviewRequest,
                    reviewer: dict = Depends(teacher_token)):
    return review_glossary_entry(entry_id, body, reviewer["role"])


@app.post("/api/glossary/{entry_id}/reopen")
def glossary_reopen(entry_id: int, body: GlossaryFlagRequest,
                    reviewer: dict = Depends(teacher_token)):
    return reopen_glossary_entry(entry_id, body.note, reviewer["role"])


@app.get("/api/language-corpus")
def language_corpus_state(_reviewer: dict = Depends(teacher_token)):
    return get_language_corpus_state()


@app.get("/api/language-lessons")
def language_lessons(lang: Literal["ho", "mun"]):
    return get_language_student_lessons(lang)


@app.get("/api/language-corpus/work")
def language_corpus_work(lang: Literal["ho", "mun"] | None = None,
                         reviewer: dict = Depends(teacher_token)):
    return get_language_example_work(reviewer["role"], lang)


@app.get("/api/language-corpus/mine")
def language_corpus_mine(reviewer: dict = Depends(teacher_token)):
    return get_my_language_reviews(reviewer["role"])


@app.post("/api/language-corpus", status_code=201)
def language_corpus_add(body: LanguageExampleSubmissionRequest,
                        reviewer: dict = Depends(teacher_token)):
    return add_language_example(body.lang, body.hindi, body.target, body.writing_system,
                                body.source_note, reviewer["role"])


@app.post("/api/language-corpus/lesson-request", status_code=201)
def language_corpus_lesson_request(body: LanguageTranslationTaskRequest,
                                   reviewer: dict = Depends(teacher_token)):
    return queue_language_translation(body.lang, body.hindi, body.writing_system,
                                      body.source_note, reviewer["role"])


@app.post("/api/language-corpus/{example_id}/review")
def language_corpus_review(example_id: int, body: LanguageExampleReviewRequest,
                           reviewer: dict = Depends(teacher_token)):
    return review_language_example(example_id, body, reviewer["role"])


@app.post("/api/language-corpus/{example_id}/reopen")
def language_corpus_reopen(example_id: int, body: GlossaryFlagRequest,
                           reviewer: dict = Depends(teacher_token)):
    return reopen_language_example(example_id, body.note, reviewer["role"])


@app.get("/api/language-corpus/export")
def language_corpus_export(lang: Literal["ho", "mun"], _reviewer: dict = Depends(teacher_token)):
    lines = export_language_examples(lang)
    return Response(
        "\n".join(lines) + ("\n" if lines else ""),
        media_type="application/x-ndjson; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="reviewed_{lang}_sentences.jsonl"'},
    )


@app.get("/api/export")
def export(_token: dict = Depends(teacher_token)):
    return Response(
        export_jsonl(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=corrections.jsonl"},
    )


@app.get("/api/admin/backup")
def download_school_backup(_reviewer: dict = Depends(teacher_token)):
    """Build a consistent, role-protected copy of classroom data and lesson media."""
    created = datetime.now(timezone.utc)
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    work_dir = tempfile.mkdtemp(prefix="bhasha-setu-backup-")
    archive_path = os.path.join(work_dir, f"bhasha-setu-backup-{stamp}.zip")
    databases = ((DB_PATH, "bhasha_setu.db"), (core.DB_PATH, "bhasha_setu_v2.db"))
    media_summary = {}

    try:
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for source_path, archive_name in databases:
                if not os.path.isfile(source_path):
                    raise HTTPException(
                        status_code=503,
                        detail=f"The school data backup is missing {archive_name}; no backup was created.",
                    )
                snapshot_path = os.path.join(work_dir, f"snapshot-{archive_name}")
                source = sqlite3.connect(str(source_path), timeout=30)
                snapshot = sqlite3.connect(snapshot_path, timeout=30)
                try:
                    # SQLite's backup API includes committed WAL data safely while the app is running.
                    source.backup(snapshot, pages=256, sleep=0.05)
                finally:
                    snapshot.close()
                    source.close()
                archive.write(snapshot_path, arcname=archive_name)

            for directory_name, source_dir in (
                ("audio_recordings", AUDIO_DIR),
                ("lesson_visuals", VISUAL_DIR),
            ):
                archive.writestr(directory_name + "/", "")
                count = 0
                total_bytes = 0
                if os.path.isdir(source_dir):
                    with os.scandir(source_dir) as entries:
                        for entry in entries:
                            if entry.name.startswith(".") or not entry.is_file(follow_symlinks=False):
                                continue
                            archive.write(entry.path, arcname=f"{directory_name}/{entry.name}")
                            count += 1
                            total_bytes += entry.stat(follow_symlinks=False).st_size
                media_summary[directory_name] = {"files": count, "bytes": total_bytes}

            manifest = {
                "format_version": 1,
                "created_utc": created.isoformat(),
                "included": ["bhasha_setu.db", "bhasha_setu_v2.db", "audio_recordings/", "lesson_visuals/"],
                "media": media_summary,
                "excluded": [
                    "model weights and caches",
                    "passcodes and environment settings",
                    "temporary PDF uploads and OCR staging files",
                    "student-device offline caches",
                ],
                "restore_note": "Stop the server before replacing the two database files and media folders.",
            }
            archive.writestr("backup_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    except HTTPException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except (OSError, sqlite3.Error, zipfile.BadZipFile) as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail="Could not create the school backup. Check server disk space and database access, then retry.",
        ) from exc
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename=os.path.basename(archive_path),
        background=BackgroundTask(shutil.rmtree, work_dir, ignore_errors=True),
    )


@app.get("/api/glossary/export")
def glossary_export(lang: Literal["ho", "mun"], _token: dict = Depends(teacher_token)):
    return Response(
        export_glossary_jsonl(lang),
        media_type="application/x-ndjson; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=reviewed_words_{lang}.jsonl"},
    )


@app.post("/api/translate")
def translate(body: TranslationRequest, _token: dict = Depends(teacher_token)):
    return add_translation(body.text)


@app.post("/api/approve/{item_id}")
def approve(item_id: int, body: ApprovalRequest, reviewer: dict = Depends(teacher_token)):
    return approve_item(item_id, body.final, reviewer["role"])


@app.delete("/api/review/{item_id}")
def remove_review_item(item_id: int, _reviewer: dict = Depends(teacher_token)):
    with db() as con:
        row = con.execute(
            "SELECT id,status,audio_filename,visual_filename FROM items WHERE id=? AND model!=?",
            (item_id, DEMO_MODEL),
        ).fetchone()
        if not row:
            raise core.ApiError(404, "That review sentence was not found. Reload the review queue.")
        if row["status"] not in ("pending", "reviewing"):
            raise core.ApiError(409, "Only a sentence still waiting for review can be removed here.")
        source_ids = [int(source["document_id"]) for source in con.execute(
            "SELECT DISTINCT document_id FROM document_segments WHERE item_id=?", (item_id,),
        ).fetchall()]
        con.execute("UPDATE items SET status='cancelled',approved=NULL WHERE id=?", (item_id,))
        con.execute("DELETE FROM review_events WHERE item_id=?", (item_id,))
        con.execute("DELETE FROM lesson_correction_flags WHERE item_id=?", (item_id,))
    for document_id in source_ids:
        _remove_student_pdf_cache(document_id)
    remove_audio_file(row["audio_filename"])
    remove_visual_file(row["visual_filename"])
    return {"id": item_id, "removed": True, "linked_documents": len(source_ids)}


@app.post("/api/audio/{item_id}/generate")
def queue_santali_audio_generation(item_id: int, reviewer: dict = Depends(teacher_token)):
    role = reviewer["role"]
    with db() as con:
        active_generation = con.execute(
            "SELECT 1 FROM jobs WHERE kind='generate_santali_audio' AND status IN ('queued','processing') LIMIT 1"
        ).fetchone()
        if active_generation:
            raise core.ApiError(409, "Another Santali audio job is already queued or generating. Wait for it to finish before starting another.")
        row = con.execute(
            """SELECT status,final,audio_status,audio_author,audio_generation_status
               FROM items WHERE id=? AND model!=?""",
            (item_id, DEMO_MODEL),
        ).fetchone()
        if not row or row["status"] != "approved" or not row["final"]:
            raise core.ApiError(409, "A reviewer must approve this Santali lesson before generating speech.")
        approved_count = con.execute(
            """SELECT COUNT(DISTINCT reviewer_role) FROM review_events
               WHERE item_id=? AND review_round=(SELECT review_round FROM items WHERE id=?)
                 AND final=(SELECT final FROM items WHERE id=?)""",
            (item_id, item_id, item_id),
        ).fetchone()[0]
        if approved_count < REQUIRED_REVIEWERS:
            raise core.ApiError(409, f"This Santali lesson needs {REQUIRED_REVIEWERS} text reviewer approval(s) before generating speech.")
        if row["audio_status"] not in ("none", "rejected"):
            raise core.ApiError(409, "This lesson already has audio waiting for review or approved for students.")
        if row["audio_status"] == "rejected" and row["audio_author"] != role:
            raise core.ApiError(403, "Only the reviewer who submitted the rejected clip can generate its replacement.")
        if row["audio_generation_status"] in ("queued", "processing"):
            raise core.ApiError(409, "Santali audio is already being generated for this lesson.")
        transcript = row["final"]
        now = time.time()
        payload = json.dumps({"item_id": item_id, "reviewer_role": role, "transcript": transcript},
                             ensure_ascii=False)
        cursor = con.execute(
            "INSERT INTO jobs (kind,payload,status,created,updated) VALUES (?,?, 'queued', ?, ?)",
            ("generate_santali_audio", payload, now, now),
        )
        job_id = cursor.lastrowid
        con.execute(
            """UPDATE items SET audio_generation_status='queued',audio_generation_error=NULL,
               audio_generation_author=? WHERE id=?""",
            (role, item_id),
        )
    return {"job_id": job_id, "status": "queued", "message": "Santali speech generation was queued on the school server."}


@app.get("/api/audio/generation/{job_id}")
def santali_audio_generation_status(job_id: int, reviewer: dict = Depends(teacher_token)):
    with db() as con:
        row = con.execute(
            "SELECT kind,payload,status,error FROM jobs WHERE id=? AND kind='generate_santali_audio'",
            (job_id,),
        ).fetchone()
    if not row:
        raise core.ApiError(404, "That Santali audio generation task was not found.")
    try:
        payload = json.loads(row["payload"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise core.ApiError(500, "The Santali audio task details could not be read.") from exc
    if payload.get("reviewer_role") != reviewer["role"]:
        raise core.ApiError(403, "Only the reviewer who started generation can view this task.")
    return {"job_id": job_id, "status": row["status"], "error": row["error"]}


@app.post("/api/audio/{item_id}", status_code=201)
async def upload_audio(item_id: int, file: UploadFile = File(...), transcript: str = Form(...),
                       reviewer: dict = Depends(teacher_token)):
    transcript = " ".join(str(transcript or "").split())
    if not transcript or len(transcript) > 2000:
        raise core.ApiError(400, "Record a reading of the approved Santali text.")
    mime = (file.content_type or "").split(";", 1)[0].strip().lower()
    file_types = {
        "audio/webm": (".webm", lambda b: b.startswith(b"\x1aE\xdf\xa3")),
        "audio/ogg": (".ogg", lambda b: b.startswith(b"OggS")),
        "audio/wav": (".wav", lambda b: b.startswith(b"RIFF") and b[8:12] == b"WAVE"),
        "audio/x-wav": (".wav", lambda b: b.startswith(b"RIFF") and b[8:12] == b"WAVE"),
        "audio/mp4": (".mp4", lambda b: len(b) > 12 and b[4:8] == b"ftyp"),
        "audio/mpeg": (".mp3", lambda b: b.startswith(b"ID3") or b[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")),
    }
    if mime not in file_types:
        raise core.ApiError(415, "Use a browser recording in WebM, Ogg, WAV, MP4, or MP3 format.")
    raw = await file.read(MAX_AUDIO_BYTES + 1)
    if len(raw) > MAX_AUDIO_BYTES:
        raise core.ApiError(413, "Recordings must be 5 MB or smaller. Try a shorter reading.")
    extension, valid_signature = file_types[mime]
    if not valid_signature(raw):
        raise core.ApiError(400, "The uploaded file does not match its audio format.")

    with db() as con:
        row = con.execute(
            "SELECT status,audio_status,audio_author,audio_filename,audio_generation_status FROM items WHERE id=? AND model!=?",
            (item_id, DEMO_MODEL),
        ).fetchone()
        if not row or row["status"] != "approved":
            raise core.ApiError(409, "A reviewer must approve this Santali lesson before recording lesson audio.")
        if row["audio_generation_status"] in ("queued", "processing"):
            raise core.ApiError(409, "Santali audio is still being generated for this lesson.")
        if row["audio_status"] == "awaiting_review":
            raise core.ApiError(409, "This recording is already waiting for audio review.")
        if row["audio_status"] == "rejected" and row["audio_author"] != reviewer["role"]:
            raise core.ApiError(403, "The reviewer who recorded this clip must make the requested replacement.")
        old_filename = row["audio_filename"]

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{item_id}-{secrets.token_hex(16)}{extension}"
    destination = AUDIO_DIR / filename
    try:
        destination.write_bytes(raw)
        with db() as con:
            current = con.execute(
                "SELECT status,audio_status,audio_author FROM items WHERE id=? AND model!=?",
                (item_id, DEMO_MODEL),
            ).fetchone()
            if not current or current["status"] != "approved" or current["audio_status"] == "awaiting_review":
                raise core.ApiError(409, "This lesson changed while the recording was uploading. Reload the audio page.")
            if current["audio_status"] == "rejected" and current["audio_author"] != reviewer["role"]:
                raise core.ApiError(403, "The reviewer who recorded this clip must make the requested replacement.")
            con.execute(
                """UPDATE items SET audio_filename=?,audio_mime=?,audio_transcript=?,audio_author=?,
                   audio_status='awaiting_review',audio_reviewer=NULL,audio_reviewed=NULL,audio_generated=0,
                   audio_generation_status='idle',audio_generation_error=NULL,audio_generation_author=NULL WHERE id=?""",
                (filename, mime, transcript, reviewer["role"], item_id),
            )
    except Exception:
        remove_audio_file(filename)
        raise
    remove_audio_file(old_filename)
    return {"id": item_id, "audio_status": "awaiting_review", "reviewer_role": reviewer["role"]}


@app.delete("/api/audio/{item_id}")
def delete_audio(item_id: int, reviewer: dict = Depends(teacher_token)):
    with db() as con:
        row = con.execute(
            "SELECT audio_filename,audio_author FROM items WHERE id=? AND model!=?",
            (item_id, DEMO_MODEL),
        ).fetchone()
        if not row or not row["audio_filename"]:
            raise core.ApiError(404, "That recording is not available.")
        if row["audio_author"] != reviewer["role"]:
            raise core.ApiError(403, "Only the reviewer role that submitted this recording can withdraw it.")
        filename = row["audio_filename"]
        con.execute(
            """UPDATE items SET audio_filename=NULL,audio_mime=NULL,audio_transcript=NULL,audio_author=NULL,
               audio_status='none',audio_reviewer=NULL,audio_reviewed=NULL,audio_generated=0,
               audio_generation_status='idle',audio_generation_error=NULL,audio_generation_author=NULL WHERE id=?""",
            (item_id,),
        )
    remove_audio_file(filename)
    return {"id": item_id, "deleted": True}


@app.post("/api/audio/{item_id}/review")
def review_audio(item_id: int, body: AudioReviewRequest, reviewer: dict = Depends(teacher_token)):
    if not body.listened:
        raise core.ApiError(400, "Listen to the recording before approving or requesting a replacement.")
    with db() as con:
        row = con.execute(
            "SELECT status,final,audio_transcript,audio_author,audio_status FROM items WHERE id=? AND model!=?",
            (item_id, DEMO_MODEL),
        ).fetchone()
        if not row or row["status"] != "approved" or row["audio_status"] != "awaiting_review":
            raise core.ApiError(409, "That recording is no longer waiting for review.")
        transcript_matches = row["audio_transcript"] == row["final"]
        accepted = body.action == "approve" and transcript_matches
        con.execute(
            "UPDATE items SET audio_status=?,audio_reviewer=?,audio_reviewed=? WHERE id=?",
            ("approved" if accepted else "rejected", reviewer["role"], time.time(), item_id),
        )
    if body.action == "approve" and not transcript_matches:
        return {"id": item_id, "approved": False, "rejected": True,
                "message": "The translation changed after this clip was recorded. The recording was rejected; record the approved text again."}
    return {"id": item_id, "approved": accepted, "rejected": not accepted,
            "message": "Recording approved for student lessons." if accepted
            else "Recording rejected. Ask the recording reviewer to submit a replacement."}


@app.post("/api/language-audio/{example_id}", status_code=201)
async def upload_language_audio(example_id: int, file: UploadFile = File(...), transcript: str = Form(...),
                                reviewer: dict = Depends(teacher_token)):
    transcript = " ".join(str(transcript or "").split())
    if not transcript or len(transcript) > 2000:
        raise core.ApiError(400, "Record a reading of the approved Ho or Mundari text.")
    mime = (file.content_type or "").split(";", 1)[0].strip().lower()
    file_types = {
        "audio/webm": (".webm", lambda b: b.startswith(b"\x1aE\xdf\xa3")),
        "audio/ogg": (".ogg", lambda b: b.startswith(b"OggS")),
        "audio/wav": (".wav", lambda b: b.startswith(b"RIFF") and b[8:12] == b"WAVE"),
        "audio/x-wav": (".wav", lambda b: b.startswith(b"RIFF") and b[8:12] == b"WAVE"),
        "audio/mp4": (".mp4", lambda b: len(b) > 12 and b[4:8] == b"ftyp"),
        "audio/mpeg": (".mp3", lambda b: b.startswith(b"ID3") or b[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")),
    }
    try:
        if mime not in file_types:
            raise core.ApiError(415, "Use a recording in WebM, Ogg, WAV, MP4, or MP3 format.")
        raw = await file.read(MAX_AUDIO_BYTES + 1)
    finally:
        await file.close()
    if len(raw) > MAX_AUDIO_BYTES:
        raise core.ApiError(413, "Recordings must be 5 MB or smaller. Try a shorter reading.")
    extension, valid_signature = file_types[mime]
    if not valid_signature(raw):
        raise core.ApiError(400, "The uploaded file does not match its audio format.")

    with db() as con:
        row = con.execute(
            "SELECT status,target,audio_status,audio_author,audio_filename FROM language_examples WHERE id=?",
            (example_id,),
        ).fetchone()
        if not row or row["status"] != "approved" or not row["target"]:
            raise core.ApiError(409, "Both text reviewer roles must approve the translation before recording its audio.")
        if transcript != row["target"]:
            raise core.ApiError(409, "Use the exact approved Ho or Mundari text shown on this card.")
        if row["audio_status"] == "awaiting_review":
            raise core.ApiError(409, "This recording is already waiting for the other reviewer role.")
        if row["audio_status"] == "rejected" and row["audio_author"] != reviewer["role"]:
            raise core.ApiError(403, "The reviewer who recorded this clip must make the requested replacement.")
        old_filename = row["audio_filename"]

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"lang-{example_id}-{secrets.token_hex(16)}{extension}"
    destination = AUDIO_DIR / filename
    try:
        destination.write_bytes(raw)
        with db() as con:
            current = con.execute(
                "SELECT status,target,audio_status,audio_author FROM language_examples WHERE id=?",
                (example_id,),
            ).fetchone()
            if (not current or current["status"] != "approved" or current["target"] != transcript
                    or current["audio_status"] == "awaiting_review"):
                raise core.ApiError(409, "The lesson changed while the recording was uploading. Reload the audio page.")
            if current["audio_status"] == "rejected" and current["audio_author"] != reviewer["role"]:
                raise core.ApiError(403, "The reviewer who recorded this clip must make the requested replacement.")
            con.execute(
                """UPDATE language_examples SET audio_filename=?,audio_mime=?,audio_transcript=?,audio_author=?,
                   audio_status='awaiting_review',audio_reviewer=NULL,audio_reviewed=NULL WHERE id=?""",
                (filename, mime, transcript, reviewer["role"], example_id),
            )
    except Exception:
        remove_audio_file(filename)
        raise
    remove_audio_file(old_filename)
    return {"id": example_id, "audio_status": "awaiting_review", "reviewer_role": reviewer["role"]}


@app.delete("/api/language-audio/{example_id}")
def delete_language_audio(example_id: int, reviewer: dict = Depends(teacher_token)):
    with db() as con:
        row = con.execute(
            "SELECT audio_filename,audio_author FROM language_examples WHERE id=?", (example_id,),
        ).fetchone()
        if not row or not row["audio_filename"]:
            raise core.ApiError(404, "That community recording is not available.")
        if row["audio_author"] != reviewer["role"]:
            raise core.ApiError(403, "Only the reviewer role that submitted this recording can withdraw it.")
        filename = row["audio_filename"]
        con.execute(
            """UPDATE language_examples SET audio_filename=NULL,audio_mime=NULL,audio_transcript=NULL,
               audio_author=NULL,audio_status='none',audio_reviewer=NULL,audio_reviewed=NULL WHERE id=?""",
            (example_id,),
        )
    remove_audio_file(filename)
    return {"id": example_id, "deleted": True}


@app.post("/api/language-audio/{example_id}/review")
def review_language_audio(example_id: int, body: AudioReviewRequest, reviewer: dict = Depends(teacher_token)):
    if not body.listened:
        raise core.ApiError(400, "Listen to the recording before approving or requesting a replacement.")
    with db() as con:
        row = con.execute(
            """SELECT status,target,audio_transcript,audio_author,audio_status
               FROM language_examples WHERE id=?""", (example_id,),
        ).fetchone()
        if not row or row["status"] != "approved" or row["audio_status"] != "awaiting_review":
            raise core.ApiError(409, "That community recording is no longer waiting for review.")
        if row["audio_author"] == reviewer["role"]:
            raise core.ApiError(403, "A different reviewer role must check this recording.")
        transcript_matches = row["audio_transcript"] == row["target"]
        accepted = body.action == "approve" and transcript_matches
        con.execute(
            "UPDATE language_examples SET audio_status=?,audio_reviewer=?,audio_reviewed=? WHERE id=?",
            ("approved" if accepted else "rejected", reviewer["role"], time.time(), example_id),
        )
    if body.action == "approve" and not transcript_matches:
        return {"id": example_id, "approved": False, "rejected": True,
                "message": "The Ho/Mundari translation changed after recording. Record the approved text again."}
    return {"id": example_id, "approved": accepted, "rejected": not accepted,
            "message": "Community recording approved for students." if accepted
            else "Recording rejected. Ask the recording reviewer to submit a replacement."}


@app.post("/api/visual/{item_id}", status_code=201)
async def upload_visual(item_id: int, file: UploadFile = File(...), alt_text: str = Form(...),
                        credit: str = Form(...), reviewer: dict = Depends(teacher_token)):
    alt_text = " ".join(str(alt_text or "").split())
    credit = " ".join(str(credit or "").split())
    if len(alt_text) < 5 or len(alt_text) > 400:
        raise core.ApiError(400, "Add an accessible Santali description between 5 and 400 characters.")
    if not credit or len(credit) > 300:
        raise core.ApiError(400, "Add an image source or credit, up to 300 characters.")
    try:
        raw = await file.read(MAX_VISUAL_BYTES + 1)
    finally:
        await file.close()
    if len(raw) > MAX_VISUAL_BYTES:
        raise core.ApiError(413, "Lesson images must be 5 MB or smaller.")
    jpeg = await asyncio.to_thread(normalize_visual_image, raw)
    if len(jpeg) > MAX_VISUAL_BYTES:
        raise core.ApiError(413, "The image is still over 5 MB after resizing. Choose a simpler image.")

    with db() as con:
        row = con.execute(
            "SELECT status,visual_status,visual_author FROM items WHERE id=? AND model!=?",
            (item_id, DEMO_MODEL),
        ).fetchone()
        if not row or row["status"] != "approved":
            raise core.ApiError(409, "The lesson text must be approved before attaching a visual.")
        if row["visual_status"] == "awaiting_review":
            raise core.ApiError(409, "This visual is already waiting for a reviewer decision.")
        if row["visual_status"] in ("rejected", "approved") and row["visual_author"] != reviewer["role"]:
            raise core.ApiError(403, "Only the reviewer role that submitted this visual can replace it.")

    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{item_id}-{secrets.token_hex(16)}.jpg"
    destination = VISUAL_DIR / filename
    try:
        destination.write_bytes(jpeg)
        with db() as con:
            current = con.execute(
                "SELECT status,visual_status,visual_author,visual_filename FROM items WHERE id=? AND model!=?",
                (item_id, DEMO_MODEL),
            ).fetchone()
            if not current or current["status"] != "approved" or current["visual_status"] == "awaiting_review":
                raise core.ApiError(409, "This lesson changed while the visual was uploading. Reload the visuals page.")
            if current["visual_status"] in ("rejected", "approved") and current["visual_author"] != reviewer["role"]:
                raise core.ApiError(403, "Only the reviewer role that submitted this visual can replace it.")
            old_filename = current["visual_filename"]
            con.execute(
                """UPDATE items SET visual_filename=?,visual_alt=?,visual_credit=?,visual_author=?,
                   visual_status='awaiting_review',visual_reviewer=NULL,visual_reviewed=NULL WHERE id=?""",
                (filename, alt_text, credit, reviewer["role"], item_id),
            )
    except Exception:
        remove_visual_file(filename)
        raise
    remove_visual_file(old_filename)
    return {"id": item_id, "visual_status": "awaiting_review", "reviewer_role": reviewer["role"]}


@app.post("/api/visual/{item_id}/generate", status_code=201)
async def generate_visual(item_id: int, body: VisualGenerationRequest,
                          reviewer: dict = Depends(teacher_token)):
    image_prompt = " ".join(body.prompt.split())
    alt_text = " ".join(body.alt_text.split())
    with db() as con:
        row = con.execute(
            "SELECT status,visual_status,visual_author,hindi,final FROM items WHERE id=? AND model!=?",
            (item_id, DEMO_MODEL),
        ).fetchone()
    if not row or row["status"] != "approved":
        raise core.ApiError(409, "The lesson text must be approved before generating a visual.")
    if row["visual_status"] == "awaiting_review":
        raise core.ApiError(409, "This visual is already waiting for a reviewer decision.")
    if row["visual_status"] in ("rejected", "approved") and row["visual_author"] != reviewer["role"]:
        raise core.ApiError(403, "Only the reviewer role that submitted this visual can replace it.")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise core.ApiError(503, "Set HF_TOKEN with Hugging Face Inference Providers access before generating an AI visual.")

    prompt = (
        "Create one clear, child-friendly educational illustration for primary-school students. "
        "Accurately and respectfully depict the core idea in the lesson. Use a simple, culturally neutral setting; "
        "avoid stereotypes. Do not include text, letters, numbers, signage, labels, logos, or watermarks. "
        "Do not depict a named or identifiable person. The reviewer description may be written in Hindi or English.\n"
        f"Reviewer image description: {image_prompt[:1000]}\n"
        f"Hindi source: {row['hindi'][:1000]}\n"
        f"Approved Santali lesson: {row['final'][:1000]}"
    )

    def request_image():
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(
                model=HF_VISUAL_MODEL, provider=HF_VISUAL_PROVIDER, token=token, timeout=180,
            )
            image = client.text_to_image(prompt, width=768, height=768, num_inference_steps=4)
            output = BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()
        except ImportError as exc:
            raise core.ApiError(503, "Hugging Face image generation requires huggingface_hub in the server environment.") from exc
        except Exception as exc:
            detail = re.sub(r"hf_[A-Za-z0-9]{16,}", "hf_[redacted]", str(exc))
            folded = detail.casefold()
            if any(term in folded for term in ("401", "403", "permission", "unauthorized", "forbidden", "token")):
                message = "Hugging Face rejected the request. Use a token with Inference Providers permission and confirm your account can use the selected provider. "
            elif any(term in folded for term in ("402", "credit", "billing", "payment")):
                message = "Hugging Face reports that inference credits or billing are unavailable for this account. "
            else:
                message = f"Hugging Face provider '{HF_VISUAL_PROVIDER}' could not generate the image. Check provider availability, model access, and account credits. "
            raise core.ApiError(
                502,
                message + detail[:240],
            ) from exc

    raw = await asyncio.to_thread(request_image)
    generated_file = UploadFile(file=BytesIO(raw), filename="ai-generated.png")
    credit = f"AI-generated with {HF_VISUAL_MODEL} via Hugging Face provider {HF_VISUAL_PROVIDER}"
    result = await upload_visual(item_id, generated_file, alt_text, credit, reviewer)
    result["generated"] = True
    result["model"] = HF_VISUAL_MODEL
    result["provider"] = HF_VISUAL_PROVIDER
    return result


@app.delete("/api/visual/{item_id}")
def delete_visual(item_id: int, reviewer: dict = Depends(teacher_token)):
    with db() as con:
        row = con.execute(
            "SELECT visual_filename,visual_author FROM items WHERE id=? AND model!=?",
            (item_id, DEMO_MODEL),
        ).fetchone()
        if not row or not row["visual_filename"]:
            raise core.ApiError(404, "That lesson visual is not available.")
        if row["visual_author"] != reviewer["role"]:
            raise core.ApiError(403, "Only the reviewer role that submitted this visual can withdraw it.")
        filename = row["visual_filename"]
        con.execute(
            """UPDATE items SET visual_filename=NULL,visual_alt=NULL,visual_credit=NULL,
               visual_author=NULL,visual_status='none',visual_reviewer=NULL,visual_reviewed=NULL WHERE id=?""",
            (item_id,),
        )
    remove_visual_file(filename)
    return {"id": item_id, "deleted": True}


@app.post("/api/visual/{item_id}/review")
def review_visual(item_id: int, body: VisualReviewRequest, reviewer: dict = Depends(teacher_token)):
    if not body.viewed:
        raise core.ApiError(400, "Load and inspect the image and its description before responding.")
    with db() as con:
        row = con.execute(
            "SELECT status,visual_status,visual_author FROM items WHERE id=? AND model!=?",
            (item_id, DEMO_MODEL),
        ).fetchone()
        if not row or row["status"] != "approved" or row["visual_status"] != "awaiting_review":
            raise core.ApiError(409, "That visual is no longer waiting for review.")
        accepted = body.action == "approve"
        con.execute(
            "UPDATE items SET visual_status=?,visual_reviewer=?,visual_reviewed=? WHERE id=?",
            ("approved" if accepted else "rejected", reviewer["role"], time.time(), item_id),
        )
    return {"id": item_id, "approved": accepted, "rejected": not accepted,
            "message": "Visual approved for student lessons." if accepted
            else "Visual rejected. Ask the submitting reviewer to provide a replacement."}


@app.post("/api/documents", status_code=202)
async def upload_document(file: UploadFile = File(...), language: str = Form("Santali"),
                          writing_system: str = Form(""), source_language: str = Form("Hindi"),
                          upload_id: str | None = Header(default=None, alias="X-Upload-Id"),
                          _token: dict = Depends(teacher_token)):
    filename = re.split(r"[\\/]", file.filename or "lesson.pdf")[-1].strip() or "lesson.pdf"
    extension = os.path.splitext(filename)[1].casefold()
    file_type = {".pdf": "pdf", ".docx": "docx", ".pptx": "pptx", ".txt": "txt"}.get(extension)
    try:
        if upload_id and not re.fullmatch(r"[A-Za-z0-9_-]{16,100}", upload_id):
            raise core.ApiError(400, "The upload retry identifier is invalid.")
        if upload_id:
            with db() as con:
                previous = con.execute(
                    "SELECT d.id,d.filename,d.language,d.writing_system,d.source_language,d.status,d.page_count,d.ocr_pages "
                    "FROM upload_requests u JOIN documents d ON d.id=u.document_id WHERE u.upload_id=?",
                    (upload_id,),
                ).fetchone()
            if previous:
                result = dict(previous)
                result["segment_count"] = 0
                result["duplicate"] = True
                return result
        language_key = " ".join(str(language or "").split()).casefold()
        if language_key in ("santali", "sat"):
            language = SUPPORTED_LANGUAGE
            writing_system = "Bengali script"
        elif language_key == "ho":
            language = "Ho"
        elif language_key in ("mundari", "mun"):
            language = "Mundari"
        else:
            raise core.ApiError(400, "Choose Santali, Ho, or Mundari.")
        source_key = " ".join(str(source_language or "").split()).casefold()
        if source_key in ("hindi", "hin"):
            source_language = "Hindi"
        elif source_key in ("english", "eng"):
            source_language = "English"
        else:
            raise core.ApiError(400, "Choose Hindi or English as the source language.")
        if source_language == "English" and language != SUPPORTED_LANGUAGE:
            raise core.ApiError(400, "English source files currently use the Santali translation workflow. Choose Hindi for Ho or Mundari human-translation tasks.")
        writing_system = " ".join(str(writing_system or "").split())
        if language != SUPPORTED_LANGUAGE and not writing_system:
            raise core.ApiError(400, "Enter the Ho or Mundari writing system confirmed for this class.")
        if len(writing_system) > 80:
            raise core.ApiError(400, "Keep the writing system under 80 characters.")
        if not file_type:
            if extension in (".doc", ".ppt"):
                raise core.ApiError(415, "Save older .doc or .ppt files as .docx or .pptx, then upload them again.")
            raise core.ApiError(415, "Upload a PDF, DOCX, PPTX, or UTF-8 TXT file.")

        raw = await file.read(MAX_DOCUMENT_BYTES + 1)
        if len(raw) > MAX_DOCUMENT_BYTES:
            raise core.ApiError(413, "Uploaded files must be 20 MB or smaller.")
        if not raw:
            raise core.ApiError(400, "The selected file is empty.")

        if file_type == "pdf":
            if not raw.startswith(b"%PDF-"):
                raise core.ApiError(400, "The uploaded file does not look like a PDF.")
            from pypdf import PdfReader
            reader = PdfReader(BytesIO(raw), strict=False)
            if reader.is_encrypted and not reader.decrypt(""):
                raise core.ApiError(422, "This PDF is password-protected. Upload an unlocked copy.")
            page_count = len(reader.pages)
            if page_count > MAX_DOCUMENT_PAGES:
                raise core.ApiError(413, f"PDFs must have {MAX_DOCUMENT_PAGES} pages or fewer.")
        elif file_type in ("docx", "pptx"):
            try:
                with zipfile.ZipFile(BytesIO(raw)) as package:
                    infos = package.infolist()
                    names = {info.filename for info in infos}
                    expanded_bytes = sum(info.file_size for info in infos)
                    if len(infos) > 5000 or expanded_bytes > MAX_DOCUMENT_UNPACKED_BYTES:
                        raise core.ApiError(413, "The office document expands beyond the 120 MB processing limit.")
                    if file_type == "docx":
                        if "word/document.xml" not in names:
                            raise core.ApiError(400, "This is not a readable .docx Word document.")
                        page_count = 1  # Word pagination depends on the installed fonts and layout engine.
                    else:
                        if "ppt/presentation.xml" not in names:
                            raise core.ApiError(400, "This is not a readable .pptx PowerPoint presentation.")
                        page_count = sum(bool(re.fullmatch(r"ppt/slides/slide\d+\.xml", name)) for name in names)
                        if page_count > MAX_DOCUMENT_PAGES:
                            raise core.ApiError(413, f"PowerPoint files must have {MAX_DOCUMENT_PAGES} slides or fewer.")
            except zipfile.BadZipFile as exc:
                raise core.ApiError(400, "The office document is damaged or is not a valid DOCX/PPTX package.") from exc
        else:
            try:
                raw.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise core.ApiError(422, "Save the text file as UTF-8, then upload it again.") from exc
            page_count = 1
    except core.ApiError:
        raise
    except Exception as exc:
        raise core.ApiError(400, "Could not read this file. Export it again as a standard PDF, DOCX, PPTX, or UTF-8 TXT file.") from exc
    finally:
        await file.close()

    now = time.time()
    staging_name = secrets.token_hex(16) + extension
    PDF_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    staging_path = staged_document_path(staging_name)
    try:
        staging_path.write_bytes(raw)
        with db() as con:
            cur = con.execute(
                "INSERT INTO documents (filename,language,writing_system,status,page_count,source_language,file_type,created) VALUES (?,?,?,?,?,?,?,?)",
                (filename, language, writing_system, "extracting", page_count, source_language, file_type, now),
            )
            document_id = cur.lastrowid
            payload = json.dumps({"document_id": document_id, "staging_name": staging_name})
            con.execute(
                "INSERT INTO jobs (kind, payload, status, created, updated) VALUES (?,?,?,?,?)",
                ("prepare_document", payload, "queued", now, now),
            )
            if upload_id:
                con.execute("INSERT INTO upload_requests (upload_id, document_id, created) VALUES (?,?,?)",
                            (upload_id, document_id, now))
    except Exception:
        remove_staged_document(staging_name)
        raise
    return {"id": document_id, "filename": filename, "language": language,
            "source_language": source_language, "file_type": file_type,
            "writing_system": writing_system,
            "status": "extracting", "page_count": page_count, "ocr_pages": 0, "segment_count": 0}


@app.post("/api/student-documents/upload", status_code=202)
async def student_upload_document(request: Request,
                                  file: UploadFile = File(...),
                                  source_language: str = Form("Hindi"),
                                  upload_id: str | None = Header(default=None, alias="X-Upload-Id")):
    """Let a student submit a PDF to the existing translation and review queue."""
    origin = request.headers.get("origin", "").rstrip("/") if request else ""
    expected_origin = str(request.base_url).rstrip("/") if request else ""
    if not origin or origin != expected_origin:
        await file.close()
        raise core.ApiError(403, "Submit PDFs from the Bhasha Setu page on this school server.")
    filename = re.split(r"[\\/]", file.filename or "lesson.pdf")[-1].strip()
    if os.path.splitext(filename)[1].casefold() != ".pdf":
        await file.close()
        raise core.ApiError(415, "Student listening uploads currently accept PDF files. Choose a PDF and try again.")
    if upload_id and not re.fullmatch(r"[A-Za-z0-9_-]{16,100}", upload_id):
        await file.close()
        raise core.ApiError(400, "The upload retry identifier is invalid.")
    if upload_id:
        with db() as con:
            previous = con.execute(
                "SELECT d.id,d.filename,d.language,d.writing_system,d.source_language,d.status,d.page_count,d.ocr_pages "
                "FROM upload_requests u JOIN documents d ON d.id=u.document_id WHERE u.upload_id=?",
                (upload_id,),
            ).fetchone()
        if previous:
            await file.close()
            result = dict(previous)
            result["segment_count"] = 0
            result["duplicate"] = True
            return result

    client_host = request.client.host if request.client else "unknown"
    now = time.time()
    cutoff = now - 15 * 60
    limited = False
    retry_after = 0
    with _student_document_upload_lock:
        for host, stamps in list(_student_document_uploads.items()):
            fresh = [stamp for stamp in stamps if stamp > cutoff]
            if fresh:
                _student_document_uploads[host] = fresh
            else:
                _student_document_uploads.pop(host, None)
        recent = [stamp for stamp in _student_document_uploads.get(client_host, []) if stamp > cutoff]
        if len(recent) >= 3:
            retry_after = max(1, int(recent[0] + 15 * 60 - now))
            _student_document_uploads[client_host] = recent
            limited = True
        else:
            recent.append(now)
            _student_document_uploads[client_host] = recent
    if limited:
        await file.close()
        raise HTTPException(status_code=429,
                            detail=f"This device has submitted three PDFs recently. Try again in about {max(1, retry_after // 60)} minute(s).")

    return await upload_document(
        file=file,
        language="Santali",
        writing_system="Bengali script",
        source_language=source_language,
        upload_id=upload_id,
        _token=None,
    )


def document_detail(document_id):
    with db() as con:
        doc = con.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        if not doc:
            raise core.ApiError(404, "That PDF lesson was not found.")
        segments = con.execute(
            """SELECT ds.id,ds.page_number,ds.segment_number,ds.hindi,ds.source_text,ds.item_id,
                      ds.language_example_id,e.target,e.writing_system,e.source_note,
                      e.status AS language_review_status,
                      (SELECT COUNT(DISTINCT lr.reviewer_role) FROM language_example_reviews lr
                       WHERE lr.example_id=e.id AND lr.review_round=e.review_round) AS language_review_roles_done,
                      e.audio_status AS language_audio_status,e.audio_filename AS language_audio_version,
                      i.draft, i.final, i.explanation, i.confidence, i.model,
                      i.status AS review_status,
                      (SELECT COUNT(DISTINCT e.reviewer_role) FROM review_events e
                       WHERE e.item_id=i.id AND e.review_round=i.review_round) AS review_roles_done,
                      i.audio_status,i.audio_generated,i.visual_status,i.visual_alt,i.visual_credit,
                      i.audio_filename AS _audio_version,i.visual_filename AS _visual_version
               FROM document_segments ds
               LEFT JOIN language_examples e ON e.id=ds.language_example_id
               LEFT JOIN items i ON i.id=ds.item_id
               WHERE ds.document_id=? ORDER BY ds.page_number, ds.segment_number""",
            (document_id,),
        ).fetchall()
    result = dict(doc)
    result["required_reviewers"] = required_reviewers_for_language(result["language"])
    result["segments"] = []
    for row in segments:
        segment = dict(row)
        audio_version = segment.pop("_audio_version")
        visual_version = segment.pop("_visual_version")
        item_id = segment["item_id"]
        segment["audio_url"] = approved_media_url("audio", item_id, audio_version) \
            if segment["audio_status"] == "approved" and audio_version else None
        language_audio_version = segment.pop("language_audio_version", None)
        if segment.get("language_audio_status") != "approved" or not language_audio_version:
            segment["language_audio_url"] = None
        else:
            segment["language_audio_url"] = f"/api/language-approved-audio/{segment['language_example_id']}"
        segment["visual_url"] = approved_media_url("visual", item_id, visual_version) \
            if segment["visual_status"] == "approved" and visual_version else None
        if segment["visual_status"] != "approved":
            segment["visual_alt"] = None
            segment["visual_credit"] = None
        result["segments"].append(segment)
    result["segment_count"] = len(segments)
    result["approved_count"] = sum(1 for row in segments
                                   if row["review_status"] == "approved" or row["language_review_status"] == "approved")
    return result


def _approved_whole_document(document_id, verify_review_events=False, detail=None):
    """Return the document and ordered target text only when every section is approved."""
    detail = detail or document_detail(document_id)
    if not detail["segments"]:
        return None
    translated = []
    community_language = detail["language"] != SUPPORTED_LANGUAGE
    role_counts = {}
    if verify_review_events:
        with db() as con:
            if community_language:
                reviewed_rows = con.execute(
                    """SELECT ds.language_example_id AS reviewed_id,
                              COUNT(DISTINCT lr.reviewer_role) AS role_count
                       FROM document_segments ds
                       JOIN language_examples le ON le.id=ds.language_example_id
                       JOIN language_example_reviews lr ON lr.example_id=le.id
                         AND lr.review_round=le.review_round AND lr.action='approve'
                         AND lr.hindi=le.hindi AND lr.target=le.target
                         AND lr.writing_system=le.writing_system AND lr.source_note=le.source_note
                       WHERE ds.document_id=? AND lr.reviewer_role IN ('teacher','community_expert')
                       GROUP BY ds.language_example_id""",
                    (document_id,),
                ).fetchall()
            else:
                reviewed_rows = con.execute(
                    """SELECT ds.item_id AS reviewed_id,
                              COUNT(DISTINCT re.reviewer_role) AS role_count
                       FROM document_segments ds JOIN items i ON i.id=ds.item_id
                       JOIN review_events re ON re.item_id=i.id
                         AND re.review_round=i.review_round AND re.final=i.final
                       WHERE ds.document_id=? AND re.reviewer_role IN ('teacher','community_expert')
                       GROUP BY ds.item_id""",
                    (document_id,),
                ).fetchall()
        role_counts = {row["reviewed_id"]: row["role_count"] for row in reviewed_rows}

    for segment in detail["segments"]:
        if community_language:
            target = str(segment.get("target") or "").strip()
            review_id = segment.get("language_example_id")
            if segment.get("language_review_status") != "approved" or not target or segment.get("language_review_roles_done", 0) < 1:
                return None
        else:
            target = str(segment.get("final") or "").strip()
            review_id = segment.get("item_id")
            required = REQUIRED_REVIEWERS
            if segment.get("review_status") != "approved" or not target or segment.get("review_roles_done", 0) < required:
                return None
        required = 1 if community_language else REQUIRED_REVIEWERS
        if verify_review_events and role_counts.get(review_id, 0) < required:
            return None
        translated.append(target)
    source_sections = [
        str(segment.get("source_text") or segment.get("hindi") or "").strip()
        for segment in detail["segments"]
    ]
    return {"document": detail, "source_sections": source_sections, "translated_sections": translated}


def _santali_draft_whole_document(document_id, detail=None):
    """Expose one complete machine-drafted Santali PDF while it awaits review."""
    detail = detail or document_detail(document_id)
    if detail["language"] != SUPPORTED_LANGUAGE or detail["status"] not in ("ready", "reviewing"):
        return None
    if not detail["segments"]:
        return None
    translated = []
    for segment in detail["segments"]:
        draft = str(segment.get("draft") or "").strip()
        if not draft or segment.get("model") == DEMO_MODEL or segment.get("review_status") == "cancelled":
            return None
        translated.append(draft)
    source_sections = [
        str(segment.get("source_text") or segment.get("hindi") or "").strip()
        for segment in detail["segments"]
    ]
    return {"document": detail, "source_sections": source_sections, "translated_sections": translated}


def student_document_list(language):
    language_name = {"sat": SUPPORTED_LANGUAGE, "ho": "Ho", "mun": "Mundari"}[language]
    with db() as con:
        live_document_ids = [int(row["id"]) for row in con.execute(
            "SELECT id FROM documents WHERE status IN ('ready','reviewing')",
        ).fetchall()]
        rows = con.execute(
            "SELECT id,filename,language,source_language,file_type,page_count FROM documents "
            "WHERE language=? AND status IN ('ready','reviewing') ORDER BY id DESC LIMIT 300",
            (language_name,),
        ).fetchall()
    available = []
    for row in rows:
        document_id = int(row["id"])
        detail = document_detail(document_id)
        result = _approved_whole_document(document_id, verify_review_events=True, detail=detail)
        publication_status = "reviewed"
        if not result:
            result = _santali_draft_whole_document(document_id, detail=detail)
            publication_status = "ai_draft"
        if not result:
            continue
        document = result["document"]
        version_data = json.dumps(
            [publication_status, result["source_sections"], result["translated_sections"]],
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        pdf_version = hashlib.sha256(version_data).hexdigest()[:16]
        available.append({
            "id": document["id"], "filename": document["filename"],
            "language": document["language"], "source_language": document["source_language"],
            "file_type": document["file_type"], "page_count": document["page_count"],
            "required_reviewers": required_reviewers_for_language(document["language"]),
            "segment_count": document["segment_count"],
            "publication_status": publication_status,
            "download_url": f"/api/student-documents/{document['id']}/pdf?v={pdf_version}",
        })
    return {"language": language, "required_reviewers": required_reviewers_for_language(language_name),
            "live_document_ids": live_document_ids, "documents": available}


def build_student_document_pdf(document, translated_sections, publication_status="reviewed", source_sections=None):
    try:
        import pymupdf
    except ImportError as exc:
        raise core.ApiError(503, "Whole-document PDF export needs PyMuPDF. Run `python -m pip install -r requirements.txt`, restart the school server, then retry.") from exc
    from html import escape

    title = escape(os.path.basename(document["filename"]))
    language = escape(document["language"])
    source_language = escape(document["source_language"])
    source_html_language = "hi" if document["source_language"].casefold() == "hindi" else "en"
    target_html_language = {"Santali": "sat", "Ho": "hoc", "Mundari": "unr"}.get(document["language"], "und")
    source_sections = source_sections or [
        str(segment.get("source_text") or segment.get("hindi") or "")
        for segment in document.get("segments", [])
    ]
    paired_sections = []
    for index, translated in enumerate(translated_sections):
        original = source_sections[index] if index < len(source_sections) else ""
        original = " ".join(str(original or "").split())
        translated = " ".join(str(translated or "").split())
        if original:
            paired_sections.append(
                "<div class='lesson-pair'>"
                f"<p class='source-label'>Original {source_language}</p>"
                f"<p class='source-text' lang='{source_html_language}'>{escape(original)}</p>"
                f"<p class='translation-label'>{language} translation</p>"
                f"<p class='lesson-text' lang='{target_html_language}'>{escape(translated)}</p>"
                "</div>"
            )
        elif translated:
            paired_sections.append(f"<div class='lesson-pair'><p class='lesson-text' lang='{target_html_language}'>{escape(translated)}</p></div>")
    translated_body = "\n".join(paired_sections)
    publication_note = (
        f"AI-generated Santali draft - not yet approved by {REQUIRED_REVIEWERS} reviewer(s)"
        if publication_status == "ai_draft"
        else (f"Reviewed Santali translation - approved by {REQUIRED_REVIEWERS} reviewer(s)"
              if language == SUPPORTED_LANGUAGE else f"Reviewed {language} translation - approved by one reviewer")
    )
    html = (
        f"<!doctype html><html lang='{target_html_language}'><head><meta charset='utf-8'></head><body>"
        f"<h1>{title}</h1><p class='meta'>{escape(publication_note)} · Source: {source_language}</p>"
        f"{translated_body}</body></html>"
    )
    css = """
        body { font-family: sans-serif; font-size: 12pt; color: #1d2d3d; }
        h1 { font-size: 20pt; margin: 0 0 8pt; color: #163b63; }
        .meta { font-size: 10pt; color: #163b63; margin: 0 0 20pt; border: 1px solid #a83b48; padding: 7pt; }
        p { margin: 0; }
        .lesson-pair { margin: 0 0 15pt; padding: 0 0 10pt; border-bottom: 1px solid #dbe3ed; }
        .source-label, .translation-label { font-size: 9pt; font-weight: bold; color: #617188; margin: 0 0 3pt; }
        .source-text { font-size: 11pt; line-height: 1.45; margin: 0 0 8pt; }
        .lesson-text { font-size: 13pt; line-height: 1.65; text-align: left; }
    """
    writer_output = BytesIO()
    writer = pymupdf.DocumentWriter(writer_output)
    media_box = pymupdf.paper_rect("a4")
    content_box = media_box + (48, 52, -48, -48)
    story = pymupdf.Story(html=html, user_css=css)
    more = True
    page_count = 0
    while more:
        page_count += 1
        if page_count > MAX_TRANSLATED_PDF_PAGES:
            writer.close()
            raise core.ApiError(413, f"The translated document would exceed {MAX_TRANSLATED_PDF_PAGES} pages. Split the source into smaller study documents.")
        device = writer.begin_page(media_box)
        more, _filled = story.place(content_box)
        story.draw(device)
        writer.end_page()
    writer.close()
    pdf = pymupdf.open(stream=writer_output.getvalue(), filetype="pdf")
    metadata = pdf.metadata or {}
    metadata.update({
        "title": os.path.basename(document["filename"]),
        "subject": publication_note,
        "creator": "Bhasha Setu",
    })
    pdf.set_metadata(metadata)
    return pdf.tobytes()


def cached_student_document_pdf(document, translated_sections, publication_status="reviewed", source_sections=None):
    """Build each student document PDF once per text and publication status."""
    cache_data = json.dumps(
        [document["id"], document["filename"], document["language"],
         document["source_language"], publication_status, source_sections, translated_sections],
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(cache_data).hexdigest()
    cache_path = STUDENT_PDF_CACHE_DIR / f"{int(document['id'])}-{digest}.pdf"
    with _student_pdf_generation_lock:
        try:
            latest_detail = document_detail(int(document["id"]))
        except core.ApiError:
            raise core.ApiError(404, "This translated document was removed or is no longer available.")
        latest = (_santali_draft_whole_document(int(document["id"]), detail=latest_detail)
                  if publication_status == "ai_draft" else
                  _approved_whole_document(int(document["id"]), verify_review_events=True, detail=latest_detail))
        if not latest or latest["translated_sections"] != translated_sections or \
                latest["source_sections"] != (source_sections or latest["source_sections"]):
            raise core.ApiError(409, "This translated document changed. Reload the student document list and try again.")
        try:
            if cache_path.is_file():
                return cache_path.read_bytes()
        except OSError:
            pass
        pdf = build_student_document_pdf(document, translated_sections, publication_status, source_sections)
        temp_path = STUDENT_PDF_CACHE_DIR / f".{int(document['id'])}-{secrets.token_hex(8)}.tmp"
        try:
            STUDENT_PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            temp_path.write_bytes(pdf)
            os.replace(temp_path, cache_path)
            for stale_path in STUDENT_PDF_CACHE_DIR.glob(f"{int(document['id'])}-*.pdf"):
                if stale_path != cache_path:
                    stale_path.unlink(missing_ok=True)
        except OSError:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        return pdf


@app.get("/api/documents")
def documents(_token: dict = Depends(teacher_token)):
    with db() as con:
        rows = con.execute("SELECT * FROM documents WHERE status!='cancelled' ORDER BY id DESC").fetchall()
    return [document_detail(row["id"]) for row in rows]


@app.get("/api/documents/{document_id}")
def get_document(document_id: int, _token: dict = Depends(teacher_token)):
    return document_detail(document_id)


@app.delete("/api/documents/{document_id}")
def remove_document(document_id: int, _reviewer: dict = Depends(teacher_token)):
    active_job = False
    staged_to_remove = []
    with db() as con:
        document = con.execute("SELECT id,status FROM documents WHERE id=?", (document_id,)).fetchone()
        if not document:
            raise core.ApiError(404, "That study material has already been removed.")
        if document["status"] == "cancelled":
            return {
                "id": document_id, "removed": False, "stopping": True,
                "message": "Removal is already in progress. Current processing will finish its active step before cleanup.",
            }
        jobs = con.execute(
            """SELECT id,kind,payload,status FROM jobs
               WHERE kind IN ('prepare_document','translate_document')
                 AND status IN ('queued','processing')""",
        ).fetchall()
        job_ids = []
        for job in jobs:
            try:
                payload = json.loads(job["payload"])
            except (TypeError, json.JSONDecodeError):
                continue
            if int(payload.get("document_id", -1)) != document_id:
                continue
            job_ids.append(int(job["id"]))
            active_job = active_job or job["status"] == "processing"
            if job["status"] == "queued" and job["kind"] == "prepare_document":
                staged_to_remove.append(payload.get("staging_name"))
        con.execute("UPDATE documents SET status='cancelled',error=NULL WHERE id=?", (document_id,))
        for job_id in job_ids:
            con.execute(
                "UPDATE jobs SET status='cancelled',error=NULL,updated=? WHERE id=? AND status IN ('queued','processing')",
                (time.time(), job_id),
            )
    for staging_name in staged_to_remove:
        remove_staged_document(staging_name)
    if not active_job:
        _purge_cancelled_document(document_id)
    return {
        "id": document_id,
        "removed": not active_job,
        "stopping": active_job,
        "message": "Removal requested. The current extraction or translation step will finish, then its lesson data and PDF will be removed."
        if active_job else "Study material and its generated PDF were removed.",
    }


@app.get("/api/student-documents")
def get_student_documents(language: Literal["sat", "ho", "mun"] = "sat"):
    return student_document_list(language)


@app.get("/api/student-documents/{document_id}/pdf")
def get_student_document_pdf(document_id: int):
    detail = document_detail(document_id)
    result = _approved_whole_document(document_id, verify_review_events=True, detail=detail)
    publication_status = "reviewed"
    if not result:
        result = _santali_draft_whole_document(document_id, detail=detail)
        publication_status = "ai_draft"
    if not result:
        raise core.ApiError(404, "The complete translated PDF will appear after text extraction and translation finish.")
    try:
        pdf = cached_student_document_pdf(
            result["document"], result["translated_sections"], publication_status,
            result["source_sections"],
        )
    except core.ApiError:
        raise
    except Exception as exc:
        raise core.ApiError(500, "Could not prepare the student PDF. Check that the school server has the current document dependencies and script fonts.") from exc
    stem = os.path.splitext(os.path.basename(result["document"]["filename"]))[0]
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-_")[:60] or "translated-lesson"
    draft_suffix = "-ai-draft" if publication_status == "ai_draft" else ""
    filename = f"{safe_stem}-{result['document']['language'].casefold()}{draft_suffix}.pdf"
    return Response(
        pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "X-Bhasha-Setu-Publication-Status": publication_status,
            "Cache-Control": "private, no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/api/student-documents/{document_id}/listen")
def get_student_document_listen_text(document_id: int):
    """Return ordered Santali text for the student read-aloud player.

    This follows the same publication rules as the student PDF: a complete
    Santali AI draft may be heard with its draft label, while reviewed text is
    returned only after every section passes the configured approval checks.
    """
    detail = document_detail(document_id)
    if detail["language"] != SUPPORTED_LANGUAGE:
        raise core.ApiError(404, "Read-aloud is currently available for Santali documents only.")
    result = _approved_whole_document(document_id, verify_review_events=True, detail=detail)
    publication_status = "reviewed"
    if not result:
        result = _santali_draft_whole_document(document_id, detail=detail)
        publication_status = "ai_draft"
    if not result:
        raise core.ApiError(404, "The complete translated text will be available for listening after translation finishes.")

    sections = []
    for index, text in enumerate(result["translated_sections"]):
        text = str(text).strip()
        if not text:
            continue
        segment = detail["segments"][index]
        # Only expose an audio clip when the corresponding reviewed text is
        # exactly the text being published in this document.
        audio_url = None
        if (publication_status == "reviewed"
                and str(segment.get("final") or "").strip() == text):
            audio_url = segment.get("audio_url")
        sections.append({"number": len(sections) + 1, "text": text, "audio_url": audio_url})
    version_data = json.dumps(
        [publication_status, sections], ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "id": int(document_id),
        "filename": result["document"]["filename"],
        "language": SUPPORTED_LANGUAGE,
        "publication_status": publication_status,
        "required_reviewers": required_reviewers_for_language(SUPPORTED_LANGUAGE),
        "content_version": hashlib.sha256(version_data).hexdigest()[:20],
        "sections": sections,
    }


if __name__ == "__main__":
    host = os.environ.get("BS_HOST", "127.0.0.1")
    certfile = os.environ.get("BS_SSL_CERTFILE")
    keyfile = os.environ.get("BS_SSL_KEYFILE")
    try:
        is_loopback = host.casefold() == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback and not (certfile and keyfile):
        raise SystemExit("Network access needs HTTPS. Set BS_SSL_CERTFILE and BS_SSL_KEYFILE to the school's trusted TLS certificate and key.")
    import uvicorn
    uvicorn.run(app, host=host, port=core.PORT, ssl_certfile=certfile, ssl_keyfile=keyfile)
