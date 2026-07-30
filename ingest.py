#!/usr/bin/env python3
"""
ingest.py — Build a RAG index in Supabase from the corpus/ markdown files.

For each markdown file in corpus/ (epf.md, socso.md, eis.md, pcb.md):
  - read it and split it into chunks on blank-line paragraph boundaries
    (one paragraph/block per chunk; the H1 title and heading lines are skipped;
     the top "Source:" line is kept as metadata, not emitted as a content chunk)
  - embed each chunk with the local Ollama model `nomic-embed-text`
  - insert a row into kira_chunks (content, scheme, source_label, embedding)

  scheme        is derived from the filename stem (epf / socso / eis / pcb)
  source_label  is derived from the "Source:" line at the top of each file

The Supabase/Postgres connection is read from the environment (see db_connect()).
Put the values in a local .env file (see .env.example) or export them, so the
password never has to be hard-coded in this script.
"""

import argparse
import os
import re
import sys
import time

import psycopg2
import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "corpus")
FILES = ["epf.md", "socso.md", "eis.md", "pcb.md"]

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "nomic-embed-text")
EMBED_ENDPOINT = f"{OLLAMA_URL}/api/embeddings"
EMBED_DIM = 768  # nomic-embed-text produces 768-dimensional embeddings

# nomic-embed-text task prefixes. PAIRED: chunks embed with 'search_document:',
# queries with 'search_query:'. DEFAULT ON (the model's trained convention). Set
# EMBED_TASK_PREFIXES=0 only for the no-prefix A/B baseline — and if you do, both the
# ingest and the query side must use the same setting so the vector spaces align.
EMBED_TASK_PREFIXES = os.environ.get("EMBED_TASK_PREFIXES", "1").lower() not in ("0", "false", "no", "")
TASK_PREFIXES = {"document": "search_document: ", "query": "search_query: "}

MAX_RETRIES = 5  # simple retry in case Ollama is briefly busy

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS kira_chunks (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    content      text NOT NULL,
    scheme       text NOT NULL,
    source_label text,
    embedding    vector(%d)
);
""" % EMBED_DIM


# --------------------------------------------------------------------------- #
# Environment / connection
# --------------------------------------------------------------------------- #
def load_dotenv(path=".env"):
    """Minimal .env loader (KEY=VALUE lines); no external dependency."""
    p = os.path.join(HERE, path)
    if not os.path.exists(p):
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def db_connect():
    """
    Connect using either:
      * SUPABASE_DB_URL / DATABASE_URL  (a full Postgres URI), or
      * SUPABASE_DB_HOST / _PORT / _NAME / _USER / _PASSWORD  (individual parts)
    """
    url = os.environ.get("SUPABASE_DB_URL") or os.environ.get("DATABASE_URL")
    if url:
        return psycopg2.connect(url)

    host = os.environ.get("SUPABASE_DB_HOST")
    if not host:
        sys.exit(
            "ERROR: no database configuration found.\n"
            "Set SUPABASE_DB_URL (a full Postgres connection string) or the parts\n"
            "SUPABASE_DB_HOST / SUPABASE_DB_PORT / SUPABASE_DB_NAME / "
            "SUPABASE_DB_USER / SUPABASE_DB_PASSWORD — in a .env file or the shell."
        )
    return psycopg2.connect(
        host=host,
        port=os.environ.get("SUPABASE_DB_PORT", "5432"),
        dbname=os.environ.get("SUPABASE_DB_NAME", "postgres"),
        user=os.environ.get("SUPABASE_DB_USER", "postgres"),
        password=os.environ.get("SUPABASE_DB_PASSWORD"),
        sslmode=os.environ.get("SUPABASE_DB_SSLMODE", "require"),
    )


# --------------------------------------------------------------------------- #
# Parsing / chunking
# --------------------------------------------------------------------------- #
def parse_file(path):
    """Return (source_label, [chunk, ...]) for one markdown file."""
    with open(path, encoding="utf-8") as f:
        text = f.read()

    # source_label: text after the first line beginning with "Source:"
    source_label = None
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith("source:"):
            source_label = s[len("source:"):].strip()
            break

    chunks = []
    for block in re.split(r"\n\s*\n", text):  # split on blank-line boundaries
        block = block.strip()
        if not block:
            continue
        if block.lstrip().startswith("#"):  # skip H1 title and any heading line
            continue
        if block.lower().startswith("source:"):  # kept as metadata, not content
            continue
        chunks.append(block)
    return source_label, chunks


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def embed(text, role="query"):
    """Get an embedding from Ollama, retrying if it is briefly busy/unavailable.

    role selects the nomic task prefix when EMBED_TASK_PREFIXES is on (the default):
    'document' at ingestion, 'query' at search time. Set EMBED_TASK_PREFIXES=0 for
    the raw-text (no-prefix) baseline."""
    if EMBED_TASK_PREFIXES:
        text = TASK_PREFIXES.get(role, "") + text
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                EMBED_ENDPOINT,
                json={"model": OLLAMA_MODEL, "prompt": text},
                timeout=120,
            )
            resp.raise_for_status()
            emb = resp.json().get("embedding")
            if not emb:
                raise ValueError(f"no 'embedding' field in response: {resp.text[:200]}")
            return emb
        except Exception as e:  # network error, 5xx, model loading, etc.
            last_err = e
            if attempt < MAX_RETRIES:
                wait = 2 ** (attempt - 1)  # 1s, 2s, 4s, 8s ...
                print(
                    f"    embed attempt {attempt}/{MAX_RETRIES} failed ({e}); "
                    f"retrying in {wait}s...",
                    file=sys.stderr,
                )
                time.sleep(wait)
    raise RuntimeError(f"embedding failed after {MAX_RETRIES} attempts: {last_err}")


def to_vector_literal(emb):
    """Format a list of floats as a pgvector literal, e.g. '[0.1,0.2,...]'."""
    return "[" + ",".join(str(float(x)) for x in emb) + "]"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Ingest corpus/*.md into Supabase kira_chunks.")
    ap.add_argument(
        "--reset",
        action="store_true",
        help="delete existing rows for the schemes being ingested before inserting",
    )
    ap.add_argument(
        "--no-create",
        action="store_true",
        help="do not attempt to create the vector extension / kira_chunks table",
    )
    args = ap.parse_args()

    load_dotenv()

    # Parse all files up front so we fail fast on missing corpus before touching the DB.
    parsed = []
    for fname in FILES:
        path = os.path.join(CORPUS_DIR, fname)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping", file=sys.stderr)
            continue
        scheme = os.path.splitext(fname)[0]
        source_label, chunks = parse_file(path)
        parsed.append((scheme, source_label, chunks))
    if not parsed:
        sys.exit(f"ERROR: no corpus files found in {CORPUS_DIR}")

    conn = db_connect()
    conn.autocommit = False
    cur = conn.cursor()

    if not args.no_create:
        cur.execute(DDL)
        conn.commit()

    if args.reset:
        schemes = [scheme for scheme, _, _ in parsed]
        cur.execute("DELETE FROM kira_chunks WHERE scheme = ANY(%s)", (schemes,))
        print(f"reset: deleted {cur.rowcount} existing row(s) for schemes {schemes}")
        conn.commit()

    total = 0
    for scheme, source_label, chunks in parsed:
        print(f"\n== {scheme}.md — {len(chunks)} chunk(s) — source_label: {source_label}")
        for chunk in chunks:
            emb = embed(chunk, role="document")
            if total == 0 and len(emb) != EMBED_DIM:
                print(
                    f"NOTE: embedding dimension is {len(emb)}, but the table expects "
                    f"{EMBED_DIM}. Adjust EMBED_DIM (and the table) if using another model.",
                    file=sys.stderr,
                )
            cur.execute(
                "INSERT INTO kira_chunks (content, scheme, source_label, embedding) "
                "VALUES (%s, %s, %s, %s::vector)",
                (chunk, scheme, source_label, to_vector_literal(emb)),
            )
            total += 1
            preview = " ".join(chunk[:60].split())
            print(f"  [{total}] {scheme}: {preview}...")
        conn.commit()  # commit per file

    cur.close()
    conn.close()
    print(f"\nDone. Total chunks inserted: {total}")


if __name__ == "__main__":
    main()
