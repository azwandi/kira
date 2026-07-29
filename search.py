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
import re
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


# Common Malay marker words used to detect a non-English query cheaply (no model
# call). The corpus/queries here are English or Malay; a query containing any of
# these is treated as non-English. None of these are English words, so English
# queries never match. Swap for a proper language detector if the languages widen.
_MALAY_MARKERS = frozenset({
    "berapa", "kadar", "caruman", "majikan", "adakah", "apakah", "bilakah", "mengapa",
    "untuk", "pekerja", "bawah", "atas", "membayar", "dibayar", "dikenakan", "siling",
    "gaji", "bayaran", "perlu", "tahun", "kerja", "lebih", "masa", "yang", "dengan",
    "adalah", "ialah", "atau", "kepada", "daripada", "ini", "itu", "syarikat", "faedah",
})


def is_english_query(text):
    """Cheap heuristic: a query is non-English if it contains any Malay marker word.
    Returns True for English (so the pipeline can skip translation with no model call)."""
    tokens = re.findall(r"[a-z]+", (text or "").lower())
    return not any(t in _MALAY_MARKERS for t in tokens)


def search(question, top_k=5, translate="auto"):
    """Return a list of (similarity, scheme, source_label, content) rows.

    translate controls query preprocessing before embedding:
      "auto" (default) — translate to English ONLY if the query is not English
                         (English queries make no model call and are used as-is);
      True             — always translate;
      False            — never translate.
    The original `question` is only used for the retrieval query text here."""
    do_translate = translate is True or (translate == "auto" and not is_english_query(question))
    query_text = translate_to_english(question) if do_translate else question
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
                    help="force translation even if the query looks English")
    ap.add_argument("--no-translate", action="store_true",
                    help="disable conditional translation (retrieve on the raw query)")
    args = ap.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        sys.exit("ERROR: empty query.")

    ingest.load_dotenv()

    mode = False if args.no_translate else (True if args.translate_queries else "auto")
    do_translate = mode is True or (mode == "auto" and not is_english_query(question))
    if do_translate:
        english = translate_to_english(question)
        print(f'\nQuery (original):   {question}')
        print(f'Query (translated): {english}')
        results = search(english, top_k=args.top_k, translate=False)
    else:
        print(f'\nQuery: {question}')
        results = search(question, top_k=args.top_k, translate=False)

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
