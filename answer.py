#!/usr/bin/env python3
"""
answer.py — Grounded RAG generation over the kira_chunks index.

Takes a question from the command line, retrieves the top 5 chunks with the same
embedding + pgvector approach as search.py, then asks an LLM (generation backend
selected by GEN_BACKEND: groq | gemini | ollama, default groq) to answer
*strictly* from those chunks.

The model is instructed to:
  1. answer ONLY from the provided chunks, never from its own knowledge;
  2. say the answer is not covered in the official documents available, and not
     guess, when the chunks don't contain it;
  3. for a specific calculation or personalized tax/financial advice, explain the
     concept from the chunks but decline to compute an exact figure, and point to
     the official LHDN PCB Calculator / KWSP / PERKESO resources;
  4. end with a 'Source:' line listing the source_label(s) it actually used.

The backend, model, and API key are resolved from the environment via llm.py
(GEN_BACKEND / GEN_MODEL; GROQ_API_KEY or GEMINI_API_KEY as needed). Keys never
appear in code.

Usage:
    ./.venv/bin/python answer.py "how much does an employer contribute to EPF?"
    ./.venv/bin/python answer.py --dry-run "compute my PCB on RM6000 salary"
    GEN_BACKEND=gemini ./.venv/bin/python answer.py "..."   # override the backend
"""

import argparse
import sys

import ingest   # DB + .env helpers
import search   # search.search() -> retrieval (reuses the same embedding model)
import llm      # multi-backend chat client (groq | gemini | ollama)

GEN_BACKEND_DEFAULT = "groq"  # override with GEN_BACKEND in .env

SYSTEM_INSTRUCTION = """\
You are a careful assistant answering questions about Malaysian statutory payroll: \
EPF (KWSP), SOCSO (PERKESO), EIS, and PCB (Potongan Cukai Bulanan / Monthly Tax Deduction). \
You are given numbered context passages. Work through the following steps IN ORDER.

STEP 1 — Classify the request FIRST, before considering what the passages contain.
Decide whether the question is asking for either:
  (a) a PERSONAL CALCULATION — a request to compute a specific ringgit figure from the \
user's OWN numbers (their salary, wage, number of children, or other personal inputs): \
their PCB, their EPF/SOCSO/EIS deduction, their take-home pay, their contribution amount; or
  (b) a PERSONAL RECOMMENDATION or tax/financial ADVICE — asking what the user should do or \
which option is better for them, e.g. "should I...", "is it worth it...", "which is better \
for me...", "do you recommend...".

Asking what a statutory RATE, THRESHOLD, DEFINITION, RULE, or ELIGIBILITY fact IS is NOT a \
calculation and NOT advice — it is a FACTUAL question that you MUST answer in Step 2 from the \
passages. Naming a category of person (an age, residency status, or wage band) does not make \
the question personal; that is just selecting which published rate applies. Do NOT fire the \
calculation/advice refusal on a factual question.

Worked examples —
  ANSWER (factual → Step 2); do NOT refuse these:
    - "what's the SOCSO rate" / "berapa kadar caruman PERKESO"
    - "what is the EPF contribution rate for someone under 60" / "berapa kadar caruman KWSP"
    - "what is the employer EPF rate for wages of RM5,000 or below"
    - "what is the SOCSO wage ceiling" / "who is exempt from EIS" / "when is PCB due"
  DECLINE (personal calculation/advice → use the ADVICE/CALCULATION refusal below):
    - "what's my SOCSO deduction on an RM3,200 salary"
    - "calculate my PCB with two kids" / "how much is my take-home pay on RM4,500"
    - "should I opt for voluntary EPF" / "should I hire under-60s to save on SOCSO"

Rule of thumb: if the question asks for a rate, threshold, definition, rule, or eligibility \
fact, ANSWER it (Step 2). Only DECLINE when it asks for a specific ringgit figure computed \
from the user's OWN numbers, or for a personal recommendation.

If the request is (a) or (b), you MUST use the ADVICE/CALCULATION refusal, and you MUST do \
so REGARDLESS of whether the passages contain the relevant information. This takes precedence \
over everything else. Do NOT use the "not covered in the official documents" wording for \
these requests — that wording is only ever for Step 2.

  ADVICE/CALCULATION refusal — the response must:
   - OPEN by making the boundary explicit, distinct from a corpus miss, e.g. "I can explain \
how this works, but I can't calculate your exact figure" (for a calculation) or "...but I \
can't advise you on a personal decision like this" (for a recommendation);
   - then explain the relevant concept USING ONLY the passages; if the passages don't cover \
it, say that specific detail isn't in the documents you have — but still do NOT compute a \
figure or give a recommendation;
   - direct them to the official resource: the LHDN PCB Calculator on the MyTax / ezHasil \
portal (hasil.gov.my) for PCB, KWSP / EPF (kwsp.gov.my) for EPF, or PERKESO \
(perkeso.gov.my) for SOCSO and EIS.

STEP 2 — Only if the request is NOT (a) or (b): answer it.
   - Answer ONLY using the numbered context passages. Do NOT use outside or prior knowledge, \
and do NOT introduce any rate, figure, threshold, date, or fact not explicitly stated in the \
passages.
   - If the passages do not contain what is needed, use the NOT-COVERED refusal: say plainly \
that this is not covered in the official documents available, and do not guess, infer, or \
speculate. Use this wording ONLY here — never for an (a)/(b) request.

ALWAYS end your reply with a single line beginning 'Source:'. List the exact \
source_label VALUE(S) of the passages you actually used — copy the text that appears \
after 'source_label:' in the header of each passage you relied on (for example: \
'Source: PERKESO — Contributions'). Do NOT cite passages by their bracketed [number] \
index. If you relied on more than one passage, list each DISTINCT source_label once \
(deduplicated), separated by '; '. If you used no passage, write 'Source: (none)'.

Be concise, factual, and neutral."""


def build_user_content(question, retrieved):
    """Assemble the context passages (clearly labeled) plus the question."""
    lines = ["Context passages:\n"]
    for i, (sim, scheme, source_label, content) in enumerate(retrieved, start=1):
        lines.append(f"[{i}] (scheme: {scheme} | source_label: {source_label})")
        lines.append(content)
        lines.append("")  # blank line between passages
    lines.append(f"Question: {question}")
    return "\n".join(lines)


def gen_config():
    """Resolve (backend, model, api_key, error) for generation from the environment."""
    return llm.resolve_role("GEN_BACKEND", "GEN_MODEL", GEN_BACKEND_DEFAULT)


def generate(system_instruction, user_content, backend=None, model=None, api_key=None):
    """Generate an answer with the configured backend. Returns the answer text.

    Pass backend/model/api_key to reuse a config resolved once by the caller
    (e.g. the eval runner); omit them to resolve from the environment here.
    """
    if backend is None:
        backend, model, api_key, err = gen_config()
        if err:
            raise RuntimeError(err)
    return llm.chat(system_instruction, user_content, backend=backend, model=model,
                    api_key=api_key, temperature=0.0, max_tokens=2048)


def call_gemini(system_instruction, user_content, api_key):
    """Backward-compatible Gemini-only helper (kept for callers that pass a key)."""
    return llm.chat(system_instruction, user_content, backend="gemini",
                    model=llm.default_model("gemini"), api_key=api_key,
                    temperature=0.2, max_tokens=2048)


def main():
    ap = argparse.ArgumentParser(
        description="Grounded RAG answer over kira_chunks via the configured LLM backend.")
    ap.add_argument("question", nargs="+", help="the question / query text")
    ap.add_argument("--top-k", type=int, default=5, help="chunks to retrieve (default: 5)")
    ap.add_argument("--dry-run", action="store_true",
                    help="retrieve and build the prompt but do NOT call the model")
    args = ap.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        sys.exit("ERROR: empty query.")

    ingest.load_dotenv()

    # --- retrieval (same embedding + pgvector as search.py) ---
    retrieved = search.search(question, top_k=args.top_k)

    print(f"\nQuestion: {question}")
    print("\n--- Retrieved chunks (debug) ---")
    if not retrieved:
        print("(none — is kira_chunks populated?)")
    for i, (sim, scheme, source_label, content) in enumerate(retrieved, start=1):
        preview = " ".join(content.split())[:140]
        print(f"[{i}] sim={sim:.4f}  {scheme:6s}  {source_label}")
        print(f"     {preview}...")

    user_content = build_user_content(question, retrieved)

    if args.dry_run:
        print("\n--- Prompt (dry run; model NOT called) ---")
        print("[system instruction]")
        print(SYSTEM_INSTRUCTION)
        print("\n[user content]")
        print(user_content)
        return

    backend, model, api_key, err = gen_config()
    if err:
        sys.exit(f"ERROR: {err}")

    answer = generate(SYSTEM_INSTRUCTION, user_content, backend, model, api_key)

    print(f"\n--- Answer ({backend}:{model}) ---")
    print(answer)
    print()


if __name__ == "__main__":
    main()
