#!/usr/bin/env python3
"""
score_run.py — Score a saved eval run WITHOUT re-running generation.

Reads one evals/runs/run_*.json file (the frozen output of run_evals.py) and
computes metrics over it. Generation is never re-invoked; the only model call is
the LLM-as-judge, and only for answer cases the system actually attempted.

The judge backend is independent of the generator (answer.py) and is selected by
JUDGE_BACKEND (groq | gemini | ollama; default groq). Keeping the judge on a
DIFFERENT model family than the generator both dodges Gemini's daily quota and
avoids self-preference bias in grading.

Deterministic checks (no LLM), per case:
  refusal_correct     answer matches the expected refusal type by string-matching
                      the distinct wording baked into answer.py's prompt, OR
                      correctly does NOT refuse when expected_behavior == 'answer'
  citation_correct    the final Source: line contains the expected_source label
                      (skipped when expected_source is null)
  forbidden_violation any forbidden_facts string appears in the answer
  retrieval_hit       the gold_chunk_hint substring appears in some retrieved chunk
                      (skipped when gold_chunk_hint is null) -> retrieval recall@k
  facts_present       every expected_facts substring appears in the answer
  false_refusal       expected 'answer' AND the reply used the not-covered wording
                      AND the gold content was retrievable (retrieval_hit or
                      facts_present) -> scored accuracy 0 here; never sent to the judge

LLM-as-judge (Groq / Gemini / Ollama; default Groq) grades ONLY answer cases the
system actually ATTEMPTED (refusals are never sent to it — it only sees the retrieved
chunks and so cannot tell a warranted refusal from a false one). Rubric: 2 = correct &
supported by the chunks; 1 = partial; 0 = incorrect/unsupported. The ACC column is the
mean over attempted answers (0/1/2) plus deterministic false refusals (0).

Prints a scorecard broken down by type and overall (mean accuracy, retrieval
recall@k, refusal correctness, false-refusal rate, forbidden violations),
highlights every seeded_failure case, and lists every case where the
deterministic facts_present check disagrees with the judge's accuracy.

  Refusal-wording coupling
  ------------------------
  The three markers below are the DISTINCT refusal wordings that answer.py's
  SYSTEM_INSTRUCTION tells the model to use. If that prompt's wording changes,
  update these markers to match, or refusal scoring will silently drift.
  The eval set files calculation cases under 'refuse_advice' (there is no
  separate 'refuse_calculation' behavior), so a 'refuse_advice' case is scored
  correct when EITHER the calculation OR the advice guardrail wording fired --
  what must NOT happen is the not-covered wording (that is the collapse bug this
  project fixed).

Usage:
    ./.venv/bin/python evals/score_run.py                        # newest run
    ./.venv/bin/python evals/score_run.py evals/runs/run_X.json  # a specific run
    ./.venv/bin/python evals/score_run.py --no-judge             # deterministic only
    ./.venv/bin/python evals/score_run.py --judge-backend ollama # judge locally
"""

import argparse
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import ingest   # noqa: E402  (load_dotenv)
import llm      # noqa: E402  (multi-backend chat client for the judge)

RUNS_DIR = os.path.join(HERE, "runs")

# --- distinct refusal wordings from answer.py's SYSTEM_INSTRUCTION (normalized) ---
CALC_MARKERS = ["can't calculate your exact figure", "cannot calculate your exact figure"]
ADVICE_MARKERS = ["can't advise you on a personal decision",
                  "cannot advise you on a personal decision"]
NOT_COVERED_MARKERS = ["not covered in the official documents available"]

# Order matters for by-type printing; unknown types are appended in first-seen order.
TYPE_ORDER = ["factual", "cross_language", "out_of_scope", "advice_calculation",
              "adversarial"]


# --------------------------------------------------------------------------- #
# Normalization + matching helpers
# --------------------------------------------------------------------------- #
def norm(s):
    """Lowercase, unify apostrophes, and collapse whitespace for robust matching."""
    if not s:
        return ""
    s = s.replace("’", "'").replace("‘", "'")
    return " ".join(s.lower().split())


def contains(haystack, needle):
    return norm(needle) in norm(haystack)


def detect_refusals(text):
    """Which distinct refusal wordings are present in the answer."""
    n = norm(text)
    return {
        "calc": any(m in n for m in CALC_MARKERS),
        "advice": any(m in n for m in ADVICE_MARKERS),
        "not_covered": any(m in n for m in NOT_COVERED_MARKERS),
    }


def check_citation(cited_source, expected_source):
    """None = not scored (expected_source is null); else the label must be present."""
    if expected_source is None:
        return None
    return contains(cited_source or "", expected_source)


# --------------------------------------------------------------------------- #
# Deterministic scoring
# --------------------------------------------------------------------------- #
def score_case_deterministic(res):
    exp = res.get("expected", {})
    behavior = exp.get("expected_behavior")
    ans = res.get("answer") or ""
    retrieved = res.get("retrieved", [])

    detected = detect_refusals(ans)
    guardrail_fired = detected["calc"] or detected["advice"]
    any_refusal = guardrail_fired or detected["not_covered"]

    is_answer = (behavior == "answer")
    is_refusal = bool(behavior) and behavior.startswith("refuse")

    # refusal_correct
    if is_answer:
        refusal_correct = not any_refusal            # a correct answer must not refuse
    elif behavior == "refuse_not_covered":
        refusal_correct = detected["not_covered"]
    elif behavior == "refuse_advice":
        refusal_correct = guardrail_fired            # calc OR advice wording (not not-covered)
    else:
        refusal_correct = None                       # unknown behavior -> not scored

    # citation
    citation_correct = check_citation(res.get("cited_source"), exp.get("expected_source"))

    # forbidden
    forbidden = exp.get("forbidden_facts") or []
    forbidden_hits = [f for f in forbidden if contains(ans, f)]
    forbidden_violation = len(forbidden_hits) > 0

    # retrieval hit (recall@k)
    hint = exp.get("gold_chunk_hint")
    if hint:
        retrieval_hit = any(contains(c.get("content", ""), hint) for c in retrieved)
    else:
        retrieval_hit = None

    # facts present
    facts = exp.get("expected_facts") or []
    if is_answer and facts:
        missing = [f for f in facts if not contains(ans, f)]
        facts_present = (len(missing) == 0)
    else:
        facts_present = None
        missing = []

    # FALSE REFUSAL — owned by this deterministic layer, NOT the judge. The judge
    # only sees the retrieved chunks, so it structurally cannot tell a false refusal
    # from a correct one (given those chunks, 'not covered' looks correct). A false
    # refusal is an answerable case the system refused anyway: expected an answer, the
    # reply used the not-covered wording, yet the gold content WAS retrievable (the
    # gold chunk came back, or the expected facts are present). These are scored
    # accuracy 0 here and never sent to the judge.
    false_refusal = (is_answer and detected["not_covered"]
                     and (retrieval_hit is True or facts_present is True))
    if false_refusal:
        judge_accuracy = 0
        judge_reason = ("false refusal (deterministic): gold content was retrievable "
                        "but the reply said 'not covered'")
    else:
        judge_accuracy = None
        judge_reason = None

    return {
        "id": res.get("id"),
        "type": res.get("type"),
        "question": res.get("question"),
        "expected_behavior": behavior,
        "is_answer": is_answer,
        "is_refusal": is_refusal,
        "seeded_failure": bool(exp.get("seeded_failure")),
        "detected": detected,
        "refusal_correct": refusal_correct,
        "false_refusal": false_refusal,
        "citation_correct": citation_correct,
        "cited_source": res.get("cited_source"),
        "forbidden_violation": forbidden_violation,
        "forbidden_hits": forbidden_hits,
        "retrieval_hit": retrieval_hit,
        "facts_present": facts_present,
        "missing_facts": missing,
        "error": res.get("error"),
        # 0 preset for deterministic false refusals; the LLM judge fills the rest
        # (attempted-answer cases) later.
        "judge_accuracy": judge_accuracy,
        "judge_reason": judge_reason,
    }


# --------------------------------------------------------------------------- #
# LLM-as-judge (backend-agnostic; default Groq)
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = """\
You are a strict grader for a retrieval-augmented QA system about Malaysian \
statutory payroll (EPF/KWSP, SOCSO/PERKESO, EIS, PCB/LHDNM). You are given a \
QUESTION, the CONTEXT PASSAGES that were retrieved for it, the system's ANSWER, \
and a list of EXPECTED FACTS.

You are only ever shown genuine answer attempts — never refusals. (False refusals are \
detected separately and never reach you, so do not reason about whether a refusal was \
warranted.)

Grade ONLY whether the ANSWER is accurate AND supported by the CONTEXT PASSAGES. \
Judge support-by-chunks: do NOT use your own outside knowledge, and do NOT reward \
a claim that is true in the world but absent from the passages. The EXPECTED FACTS \
describe what a correct answer should convey; use them as a guide, not as a \
string-matching checklist (a paraphrase that is supported by the passages counts).

Assign an integer accuracy score:
  2 = correct: conveys the expected fact(s) and every claim is supported by the passages.
  1 = partial: some of the expected content is present and supported, but incomplete,
      or it adds a minor unsupported detail.
  0 = incorrect: one or more claims are wrong or not supported by the passages.

Respond with ONLY a compact JSON object and nothing else:
{"accuracy": 0|1|2, "reason": "<one short sentence>"}"""


def build_judge_content(res):
    lines = [f"QUESTION:\n{res.get('question','')}\n", "CONTEXT PASSAGES:"]
    retrieved = res.get("retrieved", [])
    if not retrieved:
        lines.append("(no passages retrieved)")
    for c in retrieved:
        lines.append(f"[{c.get('rank')}] (source_label: {c.get('source_label')})")
        lines.append(c.get("content", ""))
        lines.append("")
    lines.append("SYSTEM ANSWER:")
    lines.append(res.get("answer") or "(empty)")
    lines.append("")
    facts = (res.get("expected", {}) or {}).get("expected_facts") or []
    lines.append("EXPECTED FACTS:")
    lines.extend(f"- {f}" for f in facts) if facts else lines.append("- (none listed)")
    lines.append("\nGrade now. Respond with only the JSON object.")
    return "\n".join(lines)


def parse_judge_json(text):
    """Return (accuracy:int|None, reason:str)."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            obj = json.loads(t[i:j + 1])
            acc = obj.get("accuracy")
            acc = int(acc)
            if acc in (0, 1, 2):
                return acc, str(obj.get("reason", "")).strip()
        except (ValueError, TypeError):
            pass
    return None, "unparseable judge output: " + t[:160]


# --------------------------------------------------------------------------- #
# Judge model selection (transport delegated to llm.py).
# The judge is instrumentation, not the system under test, so keep it OFF the
# generator (answer.py) — and ideally on a different model to avoid self-
# preference bias. Configure with JUDGE_BACKEND / JUDGE_MODEL.
# --------------------------------------------------------------------------- #
DEFAULT_JUDGE_BACKEND = "groq"
# Groq's free tier is token-per-minute limited, so pace judge calls by default;
# override with --judge-delay. llm.py honors Retry-After, and the sidecar cache
# below means a limit hit is never lost work.
GROQ_DEFAULT_DELAY_S = 8.0


def _attempted_answer(scored_case):
    """True if the reply is a genuine answer attempt (no refusal wording present)."""
    d = scored_case["detected"]
    return not (d["calc"] or d["advice"] or d["not_covered"])


def call_judge(system, user, backend, model, api_key):
    """Grade one case via the chosen backend (deterministic, JSON-constrained)."""
    return llm.chat(system, user, backend=backend, model=model, api_key=api_key,
                    temperature=0.0, max_tokens=512, json_mode=True)


def _save_judge_cache(path, cache):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)  # atomic: never leave a half-written cache


def run_judge(scored, results_by_id, backend, model, api_key, run_file,
              delay=0.0, rejudge=False):
    """Grade every answer case, caching each score to a <run>.judge.json sidecar.

    Because Groq's free tier can cut you off mid-run, every grade is checkpointed
    immediately and reused on the next invocation, so scoring resumes instead of
    restarting. On a hard rate/quota cap it stops cleanly rather than thrashing.
    Fills judge_accuracy / judge_reason in place.
    """
    # Judge ONLY answer cases where the system actually attempted an answer. Refusals
    # never reach the judge — deterministic false refusals are already scored 0, and
    # other refusals (e.g. retrieval misses) are left unjudged (they show up in
    # recall@k, not ACC). The judge cannot tell a warranted refusal from a false one.
    judgeable = [s for s in scored
                 if s["is_answer"] and not s["error"] and _attempted_answer(s)]
    if not judgeable:
        return

    cache_path = run_file + ".judge.json"
    cache = {}
    if os.path.exists(cache_path) and not rejudge:
        try:
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
        except (ValueError, OSError):
            cache = {}

    # Apply cached grades; queue only the ungraded (or previously-failed) cases.
    pending = []
    for s in judgeable:
        c = cache.get(s["id"])
        if c and c.get("accuracy") is not None:
            s["judge_accuracy"], s["judge_reason"] = c["accuracy"], c.get("reason", "")
        else:
            pending.append(s)
    reused = len(judgeable) - len(pending)

    note = []
    if reused:
        note.append(f"{reused} reused from {os.path.basename(cache_path)}")
    if delay:
        note.append(f"pacing {delay:g}s")
    print(f"\nJudging with {backend}:{model} — {len(pending)} to grade"
          + (f" ({', '.join(note)})" if note else "") + "...")

    graded = 0
    for k, s in enumerate(pending, start=1):
        res = results_by_id[s["id"]]
        try:
            raw = call_judge(JUDGE_SYSTEM, build_judge_content(res), backend, model, api_key)
            acc, reason = parse_judge_json(raw)
        except Exception as e:
            acc, reason = None, f"judge call failed: {type(e).__name__}: {e}"
        s["judge_accuracy"], s["judge_reason"] = acc, reason
        cache[s["id"]] = {"accuracy": acc, "reason": reason}
        _save_judge_cache(cache_path, cache)  # checkpoint after every call
        graded = k
        print(f"  [{k}/{len(pending)}] {s['id']}: accuracy={'?' if acc is None else acc}")
        if acc is None and llm.looks_rate_limited(reason):
            done = reused + graded - 1
            print(f"  -> judge hit a rate/quota limit; stopping. {done}/"
                  f"{len(judgeable)} graded and cached. Re-run score_run.py to "
                  "resume (or wait for the daily window to reset).", file=sys.stderr)
            break
        if delay and k < len(pending):
            time.sleep(delay)


# --------------------------------------------------------------------------- #
# Aggregation + printing
# --------------------------------------------------------------------------- #
def frac(part, whole):
    if not whole:
        return "  -  "
    return f"{part}/{whole} {round(100 * part / whole)}%"


def mean_or_dash(values):
    return f"{sum(values) / len(values):.2f}" if values else "  -  "


def aggregate(group):
    answer_cases = [s for s in group if s["is_answer"]]
    refusal_cases = [s for s in group if s["is_refusal"]]
    acc_vals = [s["judge_accuracy"] for s in answer_cases if s["judge_accuracy"] is not None]
    recall_cases = [s for s in group if s["retrieval_hit"] is not None]
    return {
        "n": len(group),
        "acc_mean": mean_or_dash(acc_vals),
        "acc_n": len(acc_vals),
        "recall": frac(sum(1 for s in recall_cases if s["retrieval_hit"]), len(recall_cases)),
        "refuse_ok": frac(sum(1 for s in refusal_cases if s["refusal_correct"]), len(refusal_cases)),
        "false_refuse": frac(sum(1 for s in answer_cases if s["false_refusal"]), len(answer_cases)),
        "forbidden": sum(1 for s in group if s["forbidden_violation"]),
    }


def print_scorecard(scored):
    types = list(TYPE_ORDER)
    for s in scored:
        if s["type"] not in types:
            types.append(s["type"])
    groups = [(t, [s for s in scored if s["type"] == t]) for t in types]
    groups = [(t, g) for t, g in groups if g]

    header = (f"{'TYPE':<18}{'N':>4}  {'ACC/2':>7}  {'RECALL@k':>11}  "
              f"{'REFUSE-OK':>11}  {'FALSE-REFUSE':>13}  {'FORBID':>7}")
    print("\n" + "=" * len(header))
    print("SCORECARD  (ACC = mean over attempted answers + false-refusals@0; "
          "RECALL@k = gold chunk in top-k)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for t, g in groups:
        a = aggregate(g)
        print(f"{t:<18}{a['n']:>4}  {a['acc_mean']:>7}  {a['recall']:>11}  "
              f"{a['refuse_ok']:>11}  {a['false_refuse']:>13}  {a['forbidden']:>7}")
    print("-" * len(header))
    a = aggregate(scored)
    print(f"{'OVERALL':<18}{a['n']:>4}  {a['acc_mean']:>7}  {a['recall']:>11}  "
          f"{a['refuse_ok']:>11}  {a['false_refuse']:>13}  {a['forbidden']:>7}")
    print("=" * len(header))
    print("(‘-’ = not applicable for that group; ACC counts attempted answers and "
          "deterministic false-refusals (=0), not retrieval-miss refusals.)")


def print_seeded(scored):
    seeded = [s for s in scored if s["seeded_failure"]]
    print("\n" + "-" * 78)
    print(f"SEEDED FAILURES ({len(seeded)}) — cases known to fail at baseline; "
          "improvements must move these")
    print("-" * 78)
    if not seeded:
        print("(none marked in this run)")
        return
    for s in seeded:
        bits = []
        if s["retrieval_hit"] is not None:
            bits.append("retrieval " + ("HIT" if s["retrieval_hit"] else "MISS"))
        if s["is_answer"]:
            if s["false_refusal"]:
                bits.append("FALSE-REFUSAL")
            elif not _attempted_answer(s):
                d = s["detected"]
                kind = ("not-covered" if d["not_covered"]
                        else "calc/advice-guardrail" if (d["calc"] or d["advice"])
                        else "refusal")
                retr = ("retrievable" if (s["retrieval_hit"] or s["facts_present"])
                        else "not retrievable")
                bits.append(f"refused[{kind}; {retr}]")
            else:
                bits.append("answered")
            if s["judge_accuracy"] is not None:
                bits.append(f"acc={s['judge_accuracy']}")
            if s["facts_present"] is not None:
                bits.append("facts " + ("ok" if s["facts_present"] else "missing"))
        elif s["is_refusal"]:
            bits.append("refusal " + ("ok" if s["refusal_correct"] else "WRONG"))
        # still-failing heuristic: answer case not scored a clean 2, or a retrieval miss
        still_failing = (s["retrieval_hit"] is False) or \
                        (s["is_answer"] and s["judge_accuracy"] not in (2,)) or \
                        (s["is_refusal"] and not s["refusal_correct"])
        flag = "✗ still failing" if still_failing else "✓ now passing"
        print(f"  {s['id']:<5} [{s['type']}] {flag}")
        print(f"        {s['question']}")
        print(f"        {' | '.join(bits)}")


def print_disagreements(scored):
    """Cases where facts_present (deterministic) and judge accuracy disagree."""
    rows = []
    for s in scored:
        fp, acc = s["facts_present"], s["judge_accuracy"]
        if fp is None or acc is None:
            continue
        if fp != (acc == 2):  # agreement iff (all facts present) <-> (judge says 2)
            rows.append(s)
    print("\n" + "-" * 78)
    print(f"FACTS ↔ JUDGE DISAGREEMENTS ({len(rows)}) — spot-check these by hand")
    print("-" * 78)
    if not rows:
        print("(none — deterministic facts check and judge agree on every answer case)")
        return
    for s in rows:
        kind = ("strings present but judge < 2" if s["facts_present"]
                else "judge says 2 but a required string is missing")
        print(f"  {s['id']:<5} [{s['type']}] facts_present={s['facts_present']} "
              f"judge={s['judge_accuracy']}  ({kind})")
        print(f"        {s['question']}")
        if not s["facts_present"] and s["missing_facts"]:
            print(f"        missing: {s['missing_facts']}")
        if s["judge_reason"]:
            print(f"        judge: {s['judge_reason']}")


def print_extras(scored):
    violations = [s for s in scored if s["forbidden_violation"]]
    if violations:
        print("\n" + "-" * 78)
        print(f"FORBIDDEN-FACT VIOLATIONS ({len(violations)})")
        print("-" * 78)
        for s in violations:
            print(f"  {s['id']:<5} [{s['type']}] hit: {s['forbidden_hits']}")
            print(f"        {s['question']}")

    citation_scored = [s for s in scored if s["citation_correct"] is not None]
    cit_ok = sum(1 for s in citation_scored if s["citation_correct"])
    print("\n" + "-" * 78)
    print(f"CITATION CORRECTNESS: {frac(cit_ok, len(citation_scored))} "
          "(scored only where expected_source is set)")
    bad_cit = [s for s in citation_scored if not s["citation_correct"]]
    for s in bad_cit:
        print(f"  {s['id']:<5} expected vs cited: cited={s['cited_source']!r}")

    errors = [s for s in scored if s["error"]]
    if errors:
        print("\n" + "-" * 78)
        print(f"CASES WITH RUN ERRORS ({len(errors)}) — excluded from generation-based metrics")
        print("-" * 78)
        for s in errors:
            print(f"  {s['id']:<5} {s['error']}")


def find_latest_run():
    files = sorted(glob.glob(os.path.join(RUNS_DIR, "run_*.json")))
    return files[-1] if files else None


def main():
    ap = argparse.ArgumentParser(description="Score a saved Kira eval run (no re-generation).")
    ap.add_argument("run_file", nargs="?", default=None,
                    help="path to a runs/run_*.json file (default: the newest one)")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the LLM-as-judge (deterministic checks only)")
    ap.add_argument("--judge-backend", default=None, choices=["groq", "gemini", "ollama"],
                    help="judge model backend; overrides JUDGE_BACKEND (default: groq)")
    ap.add_argument("--judge-delay", type=float, default=None, metavar="SECONDS",
                    help="pause between judge calls to respect tokens/min limits "
                         "(default: 8s for groq, 0 otherwise)")
    ap.add_argument("--rejudge", action="store_true",
                    help="ignore cached judge scores (<run>.judge.json) and grade afresh")
    args = ap.parse_args()

    ingest.load_dotenv()

    run_file = args.run_file or find_latest_run()
    if not run_file:
        sys.exit(f"ERROR: no run file given and none found in {RUNS_DIR}. "
                 "Run evals/run_evals.py first.")
    if not os.path.exists(run_file):
        sys.exit(f"ERROR: run file not found: {run_file}")

    with open(run_file, encoding="utf-8") as f:
        run = json.load(f)
    results = run.get("results", [])
    meta = run.get("meta", {})
    if not results:
        sys.exit(f"ERROR: no results in {run_file}")

    print(f"Scoring run: {run_file}")
    print(f"  captured {meta.get('timestamp')} | top_k={meta.get('top_k')} | "
          f"gen={meta.get('gen_model')} | cases={meta.get('num_cases')} | "
          f"errors={meta.get('num_errors')}")

    scored = [score_case_deterministic(r) for r in results]
    results_by_id = {r.get("id"): r for r in results}

    if not args.no_judge:
        if args.judge_backend:  # CLI overrides the env for this run
            os.environ["JUDGE_BACKEND"] = args.judge_backend
        b, model, key, err = llm.resolve_role("JUDGE_BACKEND", "JUDGE_MODEL",
                                              DEFAULT_JUDGE_BACKEND)
        if err:
            print(f"\nWARNING: {err}\n         Skipping the judge (deterministic "
                  "checks only). Use --no-judge to silence this.", file=sys.stderr)
        else:
            delay = (args.judge_delay if args.judge_delay is not None
                     else (GROQ_DEFAULT_DELAY_S if b == "groq" else 0.0))
            run_judge(scored, results_by_id, b, model, key, run_file,
                      delay=delay, rejudge=args.rejudge)

    print_scorecard(scored)
    print_seeded(scored)
    print_disagreements(scored)
    print_extras(scored)
    print()


if __name__ == "__main__":
    main()
