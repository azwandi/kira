#!/usr/bin/env python3
"""
run_evals.py — Capture-only eval runner for the Kira RAG system.

Loads evals/eval_set.json and runs every case through the EXISTING answer.py
pipeline (retrieval via search.search + grounded generation via answer.py's
configured backend, GEN_BACKEND). It does NO scoring — it only records what
happened, so a run is a frozen artifact you can score (and re-score) later
without touching the model again.

For each case it saves:
  - id, type, question
  - the full expected_* fields copied from the eval set (so score_run.py needs
    only the run file, not the eval set)
  - the retrieved chunks (rank, similarity, scheme, source_label, content)
  - the generated answer
  - the cited source (the text after the final 'Source:' line)

Output: evals/runs/run_{timestamp}.json, CHECKPOINTED after every case so an
interruption (quota, Ctrl-C, crash) never loses completed work.

  Surviving free-tier rate limits
  -------------------------------
  Free tiers (Gemini, Groq) cap requests/tokens per MINUTE and per DAY. This runner:
    * writes the run file after each case (no lost progress);
    * stops cleanly the moment it sees a quota / rate-limit error, instead of
      burning retries on every remaining case;
    * can --resume a partial run, reusing already-completed cases.
  If you hit a per-minute cap, add --delay 8 and re-run/--resume; if you hit a
  daily cap, --resume once it resets (or switch GEN_BACKEND / GEN_MODEL in .env).

Requirements at run time: Ollama (embeddings), Supabase (kira_chunks), and a
generation backend configured via GEN_BACKEND (default groq → needs GROQ_API_KEY;
gemini → GEMINI_API_KEY; ollama → local) — see llm.py.

Usage:
    ./.venv/bin/python evals/run_evals.py
    ./.venv/bin/python evals/run_evals.py --delay 8            # pace under RPM
    ./.venv/bin/python evals/run_evals.py --resume evals/runs/run_XXXX.json
    ./.venv/bin/python evals/run_evals.py --only F02 X01 A01   # smoke subset
"""

import argparse
import datetime
import json
import os
import sys
import time

# The pipeline modules live in the project root (one level up from evals/).
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import ingest   # noqa: E402  (load_dotenv, embed, db helpers)
import search   # noqa: E402  (search.search -> retrieval)
import answer   # noqa: E402  (build_user_content, call_gemini, SYSTEM_INSTRUCTION)

EVAL_SET_DEFAULT = os.path.join(HERE, "eval_set.json")
RUNS_DIR = os.path.join(HERE, "runs")

# Fields copied verbatim from each eval case so the run file is self-contained.
EXPECTED_FIELDS = [
    "expected_behavior",
    "expected_facts",
    "forbidden_facts",
    "expected_source",
    "gold_chunk_hint",
    "seeded_failure",
    "notes",
]

# Substrings that mark a Gemini quota / rate-limit failure (case-insensitive).
QUOTA_MARKERS = ("429", "resource_exhausted", "quota", "rate limit", "ratelimit",
                 "too many requests")


def is_quota_error(msg):
    m = (msg or "").lower()
    return any(k in m for k in QUOTA_MARKERS)


def extract_source_line(answer_text):
    """Return the text after the final 'Source:' line, or None if absent."""
    for line in reversed(answer_text.splitlines()):
        s = line.strip()
        if s.lower().startswith("source:"):
            return s[len("source:"):].strip()
    return None


def run_case(case, top_k, gen):
    """Run one case through retrieval + generation. Returns a result dict.

    `gen` is a resolved {backend, model, api_key} dict so we don't re-read the
    environment on every case.
    """
    question = case["question"]

    # --- retrieval: identical embedding + pgvector path as search.py / answer.py
    retrieved = search.search(question, top_k=top_k)  # (sim, scheme, label, content)

    # --- generation: identical prompt + backend as answer.py
    user_content = answer.build_user_content(question, retrieved)
    generated = answer.generate(answer.SYSTEM_INSTRUCTION, user_content,
                                gen["backend"], gen["model"], gen["api_key"])
    cited_source = extract_source_line(generated)

    retrieved_out = [
        {
            "rank": i,
            "similarity": float(sim),
            "scheme": scheme,
            "source_label": source_label,
            "content": content,
        }
        for i, (sim, scheme, source_label, content) in enumerate(retrieved, start=1)
    ]

    return {
        "id": case.get("id"),
        "type": case.get("type"),
        "question": question,
        "expected": {k: case.get(k) for k in EXPECTED_FIELDS},
        "retrieved": retrieved_out,
        "answer": generated,
        "cited_source": cited_source,
    }


def error_result(case, message, elapsed_s):
    return {
        "id": case.get("id"),
        "type": case.get("type"),
        "question": case.get("question"),
        "expected": {k: case.get(k) for k in EXPECTED_FIELDS},
        "retrieved": [],
        "answer": None,
        "cited_source": None,
        "elapsed_s": elapsed_s,
        "error": message,
    }


def save_run(out_path, base_meta, results_by_id, order, stopped_early):
    """Atomically write the run file, ordered by the eval set, with fresh counts."""
    results = [results_by_id[i] for i in order if i in results_by_id]
    completed = sum(1 for r in results if not r.get("error"))
    meta = dict(base_meta)
    meta.update({
        "num_results": len(results),
        "num_completed": completed,
        "num_errors": len(results) - completed,
        "num_pending": len(order) - completed,
        "stopped_early": stopped_early,
    })
    run_obj = {"meta": meta, "results": results}
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(run_obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)  # atomic: a crash mid-write can't corrupt the file


def main():
    ap = argparse.ArgumentParser(description="Capture-only eval runner for Kira.")
    ap.add_argument("--eval-set", default=EVAL_SET_DEFAULT,
                    help="path to eval_set.json (default: evals/eval_set.json)")
    ap.add_argument("--top-k", type=int, default=5,
                    help="chunks to retrieve per case (default: 5)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these case ids (e.g. --only F02 X01)")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N cases (after --only filtering)")
    ap.add_argument("--resume", default=None, metavar="RUN_JSON",
                    help="continue a partial run file; completed cases are reused, "
                         "and results are written back to that same file")
    ap.add_argument("--delay", type=float, default=0.0, metavar="SECONDS",
                    help="sleep between cases to stay under the per-minute rate limit "
                         "(e.g. --delay 8)")
    args = ap.parse_args()

    ingest.load_dotenv()

    gen_backend, gen_model, gen_key, gen_err = answer.gen_config()
    if gen_err:
        sys.exit(f"ERROR: {gen_err}")
    gen = {"backend": gen_backend, "model": gen_model, "api_key": gen_key}

    with open(args.eval_set, encoding="utf-8") as f:
        eval_data = json.load(f)
    all_cases = eval_data.get("cases", eval_data if isinstance(eval_data, list) else [])
    if not all_cases:
        sys.exit(f"ERROR: no cases found in {args.eval_set}")
    order = [c.get("id") for c in all_cases]

    # --- resume: reuse already-completed cases from a prior partial run ---
    results_by_id = {}
    out_path = None
    base_meta = {
        "eval_set": os.path.relpath(args.eval_set, PROJECT_ROOT),
        "eval_set_version": eval_data.get("version"),
        "top_k": args.top_k,
        "gen_model": f"{gen_backend}:{gen_model}",
        "embed_model": ingest.OLLAMA_MODEL,
        "num_cases_in_set": len(all_cases),
    }
    if args.resume:
        out_path = args.resume
        if not os.path.exists(out_path):
            sys.exit(f"ERROR: --resume file not found: {out_path}")
        with open(out_path, encoding="utf-8") as f:
            prev = json.load(f)
        for r in prev.get("results", []):
            results_by_id[r.get("id")] = r
        base_meta = {**prev.get("meta", {}), **base_meta}
        base_meta["timestamp"] = prev.get("meta", {}).get("timestamp")  # keep original
        completed = {i for i, r in results_by_id.items() if not r.get("error")}
        print(f"Resuming {out_path}: {len(completed)} case(s) already completed "
              f"and will be reused.")
    else:
        completed = set()

    # --- select which cases still need running ---
    cases = all_cases
    if args.only:
        wanted = set(args.only)
        cases = [c for c in cases if c.get("id") in wanted]
        missing = wanted - {c.get("id") for c in cases}
        if missing:
            print(f"WARNING: --only ids not found: {sorted(missing)}", file=sys.stderr)
    if args.limit is not None:
        cases = cases[:args.limit]
    to_run = [c for c in cases if c.get("id") not in completed]
    if not to_run:
        print("Nothing to run: every selected case is already completed.")
        if out_path:
            print(f"Score it with:\n  ./.venv/bin/python evals/score_run.py {out_path}")
        return

    if out_path is None:
        os.makedirs(RUNS_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_meta["timestamp"] = timestamp
        out_path = os.path.join(RUNS_DIR, f"run_{timestamp}.json")

    started = time.time()
    print(f"Running {len(to_run)} case(s) at top_k={args.top_k} "
          f"(gen: {gen_backend}:{gen_model}, embed: {ingest.OLLAMA_MODEL})"
          f"{f', delay {args.delay}s' if args.delay else ''}")
    print(f"Checkpointing to: {out_path}\n")

    stopped_early = False
    for i, case in enumerate(to_run, start=1):
        cid = case.get("id", f"#{i}")
        t0 = time.time()
        print(f"[{i}/{len(to_run)}] {cid}: {case.get('question','')[:66]}", flush=True)
        try:
            res = run_case(case, args.top_k, gen)
            res["elapsed_s"] = round(time.time() - t0, 2)
            res["error"] = None
            print(f"      -> ok ({res['elapsed_s']}s, {len(res['retrieved'])} chunks, "
                  f"cited: {res['cited_source']!r})")
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            res = error_result(case, msg, round(time.time() - t0, 2))
            if is_quota_error(msg):
                # Daily/per-minute cap hit: don't burn retries on the rest.
                results_by_id[cid] = res
                save_run(out_path, base_meta, results_by_id, order, stopped_early=True)
                print(f"      -> QUOTA/RATE LIMIT hit: {msg}", file=sys.stderr)
                stopped_early = True
                break
            print(f"      -> ERROR: {msg}", file=sys.stderr)
        results_by_id[cid] = res
        save_run(out_path, base_meta, results_by_id, order, stopped_early=False)  # checkpoint
        if args.delay and i < len(to_run):
            time.sleep(args.delay)

    # Final save reflects total elapsed for this invocation.
    base_meta["last_run_elapsed_s"] = round(time.time() - started, 2)
    save_run(out_path, base_meta, results_by_id, order, stopped_early=stopped_early)

    done = sum(1 for r in results_by_id.values() if not r.get("error"))
    pending = len(order) - done
    print(f"\nSaved to: {out_path}")
    print(f"Completed {done}/{len(order)} case(s); {pending} still pending.")
    if stopped_early or pending:
        print("\nTo continue where you left off (after the quota window resets):")
        print(f"  ./.venv/bin/python evals/run_evals.py --resume {out_path}"
              f"{'  --delay 8' if stopped_early else ''}")
    if done:
        print("\nScore what you have (partial is fine; errored cases are skipped):")
        print(f"  ./.venv/bin/python evals/score_run.py {out_path}")


if __name__ == "__main__":
    main()
