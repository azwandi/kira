#!/usr/bin/env python3
"""
search.py — Semantic search over the kira_chunks table in Supabase.

Takes a question from the command line, embeds it with the same local Ollama
model used by ingest.py (nomic-embed-text), and returns the top 5 chunks ranked
by cosine similarity using pgvector's cosine distance operator (<=>).

This is retrieval only: no answer generation, no LLM call beyond the embedding.
No index is created or required — a sequential scan is fine at this scale.

Usage:
    ./.venv/bin/python search.py "how much does an employer contribute to EPF?"
    ./.venv/bin/python search.py --top-k 5 "socso wage ceiling"
"""

import argparse
import os
import sys

# Reuse ingest.py's helpers so the embedding model + DB connection are identical.
# (ingest.py's run logic is guarded by `if __name__ == "__main__"`, so importing
#  it here has no side effects.)
import ingest
import llm   # only used for optional query translation (see translate_to_english)

# --- optional query translation (cross-language retrieval), gated by a flag ---
TRANSLATE_BACKEND = os.environ.get("TRANSLATE_BACKEND", "openrouter")
TRANSLATE_MODEL = os.environ.get("TRANSLATE_MODEL", "openai/gpt-4o-mini")
TRANSLATE_SYSTEM = (
    "You translate a short search query into English for a Malaysian statutory-payroll "
    "search engine (EPF/KWSP, SOCSO/PERKESO, EIS, PCB). If the query is already in English, "
    "return it unchanged. Reply with ONLY the English query — no quotes, notes, or preamble."
)


def translate_to_english(text, backend=None, model=None, api_key=None):
    """Translate a (possibly non-English) query to English via a cheap LLM call.

    Returns the English text (unchanged if already English). Used ONLY for the
    embedding step; the caller keeps the original query for display. This adds one
    short chat call per query, so it is opt-in (see --translate-queries).
    """
    if backend is None:
        backend, model = TRANSLATE_BACKEND, TRANSLATE_MODEL
        api_key, err = llm._key_for(backend)
        if err:
            raise RuntimeError(err)
    out = llm.chat(TRANSLATE_SYSTEM, text, backend=backend, model=model, api_key=api_key,
                   temperature=0.0, max_tokens=120)
    return out.strip().strip('"').strip()


def search(question, top_k=5, translate=False):
    """Return a list of (similarity, scheme, source_label, content) rows.

    If translate=True, the query is translated to English before embedding (retrieval
    runs on the English text); the original `question` is not otherwise used."""
    query_text = translate_to_english(question) if translate else question
    embedding = ingest.embed(query_text)  # same model + retry logic as ingest.py

    conn = ingest.db_connect()
    try:
        cur = conn.cursor()
        # <=> is pgvector's cosine DISTANCE; smaller = closer, so ORDER BY ASC.
        # cosine similarity = 1 - cosine distance.
        cur.execute(
            """
            SELECT scheme,
                   source_label,
                   content,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM kira_chunks
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s
            """,
            (ingest.to_vector_literal(embedding),
             ingest.to_vector_literal(embedding),
             top_k),
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    # rows come back as (scheme, source_label, content, similarity)
    return [(sim, scheme, label, content) for scheme, label, content, sim in rows]


def main():
    ap = argparse.ArgumentParser(
        description="Semantic search over kira_chunks (retrieval only)."
    )
    ap.add_argument("question", nargs="+", help="the question / query text")
    ap.add_argument("--top-k", type=int, default=5, help="number of results (default: 5)")
    ap.add_argument("--translate-queries", action="store_true",
                    help="translate a non-English query to English before embedding")
    args = ap.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        sys.exit("ERROR: empty query.")

    ingest.load_dotenv()

    if args.translate_queries:
        english = translate_to_english(question)
        print(f'\nQuery (original):   {question}')
        print(f'Query (translated): {english}')
        results = search(english, top_k=args.top_k)
    else:
        print(f'\nQuery: {question}')
        results = search(question, top_k=args.top_k)

    if not results:
        print("\n(no results — is kira_chunks populated?)")
        return

    for rank, (similarity, scheme, source_label, content) in enumerate(results, start=1):
        print("\n" + "-" * 78)
        print(f"#{rank}  similarity {similarity:.4f}  [{scheme}]  {source_label}")
        print("-" * 78)
        print(content)
    print()


if __name__ == "__main__":
    main()
