#!/usr/bin/env python3
"""
experiment_translate.py — Retrieval-only A/B: does translating a non-English query
to English BEFORE embedding improve cross-language retrieval?

Isolates query translation as the single variable. For each case it retrieves twice
(baseline = original query; translated = English translation) and reports hit / gold
rank / top-1 similarity for both, then the recall delta. No answer generation, no
judging — the only extra call vs. plain retrieval is the translation itself.

    ./.venv/bin/python evals/experiment_translate.py            # cross_language, top-k 5
    ./.venv/bin/python evals/experiment_translate.py --type cross_language --top-k 5
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import ingest      # noqa: E402  (load_dotenv, embeddings, DB)
import search      # noqa: E402  (search.search + translate_to_english)
import score_run   # noqa: E402  (reuse the exact retrieval-hit match: contains/norm)

EVAL_SET = os.path.join(HERE, "eval_set.json")


def hit_rank_top(rows, gold_hint):
    """rows: [(sim, scheme, label, content), ...] ordered best-first.
    Return (hit_bool, gold_rank_or_None, top1_similarity_or_None)."""
    top = rows[0][0] if rows else None
    gold_rank = None
    if gold_hint:
        for i, (_sim, _scheme, _label, content) in enumerate(rows, start=1):
            if score_run.contains(content, gold_hint):
                gold_rank = i
                break
    return (gold_rank is not None), gold_rank, top


def main():
    ap = argparse.ArgumentParser(description="Retrieval-only A/B for query translation.")
    ap.add_argument("--type", default="cross_language", help="eval type to test")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    ingest.load_dotenv()

    with open(EVAL_SET, encoding="utf-8") as f:
        cases = [c for c in json.load(f)["cases"] if c.get("type") == args.type]
    if not cases:
        sys.exit(f"no cases of type {args.type!r}")

    print(f"Retrieval-only A/B — {len(cases)} '{args.type}' case(s), top_k={args.top_k}")
    print(f"translation model: {search.TRANSLATE_BACKEND}:{search.TRANSLATE_MODEL}")

    base_hits = trans_hits = 0
    for c in cases:
        q, hint = c["question"], c.get("gold_chunk_hint")

        base_rows = search.search(q, top_k=args.top_k, translate=False)   # no translation
        bh, brank, btop = hit_rank_top(base_rows, hint)

        english = search.translate_to_english(q)
        tr_rows = search.search(english, top_k=args.top_k, translate=False)  # on translation
        th, trank, ttop = hit_rank_top(tr_rows, hint)

        base_hits += bh
        trans_hits += th
        flip = ""
        if th and not bh:
            flip = "  <-- FIXED by translation"
        elif bh and not th:
            flip = "  <-- REGRESSED"
        print("\n" + "=" * 80)
        print(f"{c['id']}  {q}")
        print(f"    EN: {english}")
        print(f"    baseline   : hit={str(bh):<5} gold@rank={str(brank):<4} "
              f"top1_sim={btop:.4f}")
        print(f"    translated : hit={str(th):<5} gold@rank={str(trank):<4} "
              f"top1_sim={ttop:.4f}{flip}")

    print("\n" + "=" * 80)
    print(f"RECALL@{args.top_k}:  baseline {base_hits}/{len(cases)}  ->  "
          f"translated {trans_hits}/{len(cases)}   "
          f"(delta {trans_hits - base_hits:+d})")


if __name__ == "__main__":
    main()
