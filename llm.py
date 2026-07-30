#!/usr/bin/env python3
"""
llm.py — Minimal multi-backend chat client shared by generation (answer.py) and
the eval judge (evals/score_run.py). Backends: gemini | groq | openrouter | ollama,
all reached with plain requests (no SDKs), matching this project's dependency-light
style. Groq and OpenRouter share one OpenAI-compatible caller.

It centralises transient-error retries and rate-limit handling: HTTP 429 is
retried honoring the server's Retry-After for short (per-minute) waits, and a long
(daily / token) cap is surfaced immediately so callers can stop, checkpoint, and
resume instead of blocking for minutes.

Each *role* resolves its backend / model / key from the environment, so generation
and the judge can run on DIFFERENT models (keep the judge off the generator to
avoid self-preference bias):

  generation : GEN_BACKEND   (default groq)   + optional GEN_MODEL override
  judge      : JUDGE_BACKEND (default groq)   + optional JUDGE_MODEL override

Backend default models (used when the per-role override is unset):
  gemini     : GEMINI_MODEL       (default gemini-2.5-flash)
  groq       : GROQ_MODEL         (default llama-3.3-70b-versatile)
  openrouter : OPENROUTER_MODEL   (default meta-llama/llama-3.3-70b-instruct)
  ollama     : OLLAMA_CHAT_MODEL  (default qwen2.5:7b-instruct)

Because the model is chosen per role (GEN_MODEL / JUDGE_MODEL), a single provider
key can drive two different models — e.g. generation and the judge on two different
OpenRouter models under one OPENROUTER_API_KEY.

Keys: GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY; none for ollama (local).
"""

import os
import sys
import time

import requests

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_ENDPOINT_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")

BACKENDS = ("gemini", "groq", "ollama", "openrouter")
MAX_RETRIES = 5
MAX_RATELIMIT_WAIT_S = 65  # honor short (per-minute) waits; bail on long (daily) caps
RATE_LIMIT_MARKERS = ("429", "rate limit", "ratelimit", "too many requests",
                      "quota", "resource_exhausted", "tokens per minute", "tpm")


class RateLimited(Exception):
    """Raised on HTTP 429 so retries can honor the server's Retry-After hint."""
    def __init__(self, wait_s, detail):
        super().__init__(detail)
        self.wait_s = wait_s


def parse_retry_after(resp):
    ra = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
    try:
        return float(ra)
    except (TypeError, ValueError):
        return None


def looks_rate_limited(msg):
    m = (msg or "").lower()
    return any(x in m for x in RATE_LIMIT_MARKERS)


def _with_retries(fn, what):
    """Run fn(), honoring Retry-After on 429 and backing off on transient errors."""
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except RateLimited as e:
            last = e
            wait = e.wait_s if e.wait_s is not None else 2 ** (attempt - 1)
            if wait > MAX_RATELIMIT_WAIT_S or attempt == MAX_RETRIES:
                raise RuntimeError(f"{what} rate-limited (retry-after={e.wait_s}s): {e}")
            print(f"    {what} rate-limited; waiting {wait:.0f}s (Retry-After)...",
                  file=sys.stderr)
            time.sleep(wait)
        except Exception as e:
            last = e
            if attempt < MAX_RETRIES:
                wait = 2 ** (attempt - 1)
                print(f"    {what} attempt {attempt}/{MAX_RETRIES} failed ({e}); "
                      f"retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
    raise RuntimeError(f"{what} failed after {MAX_RETRIES} attempts: {last}")


# --------------------------------------------------------------------------- #
# Backend callers
# --------------------------------------------------------------------------- #
def _gemini_chat(system, user, model, api_key, temperature, max_tokens, json_mode):
    endpoint = GEMINI_ENDPOINT_TMPL.format(model=model)
    gen_cfg = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if json_mode:
        gen_cfg["responseMimeType"] = "application/json"
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": gen_cfg,
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    def do():
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=120)
        if resp.status_code == 429:
            raise RateLimited(parse_retry_after(resp), f"HTTP 429: {resp.text[:200]}")
        if resp.status_code in (500, 503):
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(
                f"no candidates returned (promptFeedback={data.get('promptFeedback', {})})")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            raise RuntimeError(
                f"empty answer (finishReason={candidates[0].get('finishReason', 'UNKNOWN')})")
        return text
    return _with_retries(do, "Gemini")


def _openai_compatible_chat(what, endpoint, api_key, system, user, model,
                            temperature, max_tokens, json_mode, extra_headers=None):
    """Shared OpenAI-style /chat/completions caller (Groq, OpenRouter, ...)."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    def do():
        resp = requests.post(endpoint, headers=headers, json=body, timeout=120)
        if resp.status_code == 429:
            raise RateLimited(parse_retry_after(resp), f"HTTP 429: {resp.text[:200]}")
        if resp.status_code in (500, 502, 503):
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    return _with_retries(do, what)


def _groq_chat(system, user, model, api_key, temperature, max_tokens, json_mode):
    return _openai_compatible_chat("Groq", GROQ_ENDPOINT, api_key, system, user, model,
                                   temperature, max_tokens, json_mode)


def _openrouter_chat(system, user, model, api_key, temperature, max_tokens, json_mode):
    # Optional OpenRouter ranking headers — harmless, help attribute usage.
    extra = {"HTTP-Referer": "https://github.com/azwandi/kira-payroll-rag", "X-Title": "Kira"}
    return _openai_compatible_chat("OpenRouter", OPENROUTER_ENDPOINT, api_key, system, user,
                                   model, temperature, max_tokens, json_mode,
                                   extra_headers=extra)


def _ollama_chat(system, user, model, temperature, json_mode):
    endpoint = f"{OLLAMA_URL}/api/chat"
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        body["format"] = "json"

    def do():
        resp = requests.post(endpoint, json=body, timeout=300)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")
    return _with_retries(do, "Ollama")


def chat(system, user, *, backend, model, api_key=None,
         temperature=0.2, max_tokens=2048, json_mode=False):
    """Dispatch a single chat completion to the chosen backend. Returns the text."""
    if backend == "gemini":
        return _gemini_chat(system, user, model, api_key, temperature, max_tokens, json_mode)
    if backend == "groq":
        return _groq_chat(system, user, model, api_key, temperature, max_tokens, json_mode)
    if backend == "openrouter":
        return _openrouter_chat(system, user, model, api_key, temperature, max_tokens, json_mode)
    if backend == "ollama":
        return _ollama_chat(system, user, model, temperature, json_mode)
    raise RuntimeError(f"unknown backend {backend!r} (expected: {' | '.join(BACKENDS)})")


# --------------------------------------------------------------------------- #
# Role / config resolution
# --------------------------------------------------------------------------- #
def default_model(backend):
    if backend == "gemini":
        return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    if backend == "groq":
        return os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    if backend == "openrouter":
        return os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
    if backend == "ollama":
        return os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:7b-instruct")
    return None


def _key_for(backend):
    """Return (api_key_or_None, error_or_None)."""
    if backend == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        return key, (None if key else
                     "GEMINI_API_KEY not set — free key: https://aistudio.google.com/apikey")
    if backend == "groq":
        key = os.environ.get("GROQ_API_KEY")
        return key, (None if key else
                     "GROQ_API_KEY not set — free key: https://console.groq.com/keys")
    if backend == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        return key, (None if key else
                     "OPENROUTER_API_KEY not set — get one at https://openrouter.ai/keys")
    if backend == "ollama":
        return None, None  # local, no key
    return None, f"unknown backend {backend!r}"


def resolve_role(backend_env, model_env, default_backend):
    """Resolve (backend, model, api_key, error) for a role from the environment."""
    backend = os.environ.get(backend_env, default_backend).lower()
    if backend not in BACKENDS:
        return backend, None, None, (f"unknown {backend_env}={backend!r} "
                                     f"(expected: {' | '.join(BACKENDS)})")
    model = os.environ.get(model_env) or default_model(backend)
    key, err = _key_for(backend)
    return backend, model, key, err
