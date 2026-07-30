#!/usr/bin/env python3
"""
serve.py — a small, LOCAL-ONLY FastAPI wrapper around the committed Kira pipeline.

It does not reimplement anything: it calls the exact same code the eval harness uses
(search.search for retrieval — prefixes default-on, conditional translation, acronym
expansion — and answer.generate for grounded generation at temperature 0), then shapes
the result for a thin web client. One endpoint (POST /ask) plus the static page.

Run (localhost only):
    ./.venv/bin/uvicorn serve:app --host 127.0.0.1 --port 8000
    # or: ./.venv/bin/python serve.py

Prereqs (same as the pipeline): Ollama running with nomic-embed-text, Supabase
reachable, and OPENROUTER_API_KEY in .env. See README "Local web client".
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "evals")):
    if p not in sys.path:
        sys.path.insert(0, p)

import ingest       # noqa: E402  load_dotenv + DB/embed helpers
import search       # noqa: E402  retrieval (prefixes, translation, acronym expansion)
import answer       # noqa: E402  grounded generation (temp 0)
import score_run    # noqa: E402  classify_refusal — canonical refusal-type detection

from fastapi import FastAPI, HTTPException          # noqa: E402
from fastapi.responses import FileResponse          # noqa: E402
from pydantic import BaseModel                       # noqa: E402

ingest.load_dotenv()
app = FastAPI(title="Kira — local", docs_url=None, redoc_url=None)
INDEX = os.path.join(HERE, "static", "index.html")


class AskRequest(BaseModel):
    question: str


def _extract_source(text):
    """Text after the final 'Source:' line (the pipeline always ends with one)."""
    for line in reversed((text or "").splitlines()):
        s = line.strip()
        if s.lower().startswith("source:"):
            return s[len("source:"):].strip()
    return None


def _refusal_type(text):
    """none | not_covered | advice — reuses the eval scorer's classifier so the UI
    reflects exactly the grounding behavior the harness measures. Advice/calc takes
    precedence (it's the Step-1 guardrail in the generation prompt)."""
    d = score_run.classify_refusal(text)
    if d["type_b"]:
        return "advice"
    if d["type_a"]:
        return "not_covered"
    return "none"


@app.get("/")
def index():
    return FileResponse(INDEX)


@app.post("/ask")
def ask(req: AskRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    backend, model, api_key, err = answer.gen_config()
    if err:
        raise HTTPException(status_code=500, detail=err)

    try:
        retrieved = search.search(question, top_k=5)  # (sim, scheme, label, content)
        generated = answer.generate(answer.SYSTEM_INSTRUCTION,
                                    answer.build_user_content(question, retrieved),
                                    backend, model, api_key)
    except Exception as e:  # Ollama/Supabase/OpenRouter hiccup → surface to the client
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")

    rtype = _refusal_type(generated)
    chunks = [{
        "rank": i,
        "score": round(float(sim), 4),
        "scheme": scheme,
        "source_label": source_label,
        "snippet": " ".join((content or "").split())[:240],
    } for i, (sim, scheme, source_label, content) in enumerate(retrieved, start=1)]

    return {
        "answer": generated,
        "source": _extract_source(generated),
        "refused": rtype != "none",
        "refusal_type": rtype,
        "model": f"{backend}:{model}",
        "chunks": chunks,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
