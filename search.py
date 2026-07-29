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
import sys

# Reuse ingest.py's helpers so the embedding model + DB connection are identical.
# (ingest.py's run logic is guarded by `if __name__ == "__main__"`, so importing
#  it here has no side effects.)
import ingest


def search(question, top_k=5):
    """Return a list of (similarity, scheme, source_label, content) rows."""
    embedding = ingest.embed(question)  # same model + retry logic as ingest.py

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
    args = ap.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        sys.exit("ERROR: empty query.")

    ingest.load_dotenv()

    results = search(question, top_k=args.top_k)

    print(f'\nQuery: {question}')
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
