"""
llm_client.py — shared LLM caller for all pipeline modules.

Centralises DeepSeek and Ollama HTTP calls so nl_to_ir.py and
baseline_translator.py do not duplicate the same urllib logic.

All callers return:
    {"text": str, "latency_s": float, "error": str | None}
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

_RETRY_STATUSES = {429, 500, 502, 503, 504}
_RETRY_DELAYS   = [5, 15, 30]

BACKEND_API_KEY_ENV = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def default_model_for_backend(backend: str) -> str:
    """Return the project default model name for a supported backend."""
    return DEFAULT_BACKEND_MODELS.get(backend.strip().lower(), DEFAULT_BACKEND_MODELS["deepseek"])


def api_key_for_backend(backend: str, explicit_key: str | None = None) -> str | None:
    """Resolve the API key for a backend from an explicit key or matching env var."""
    if explicit_key:
        return explicit_key
    env_var = BACKEND_API_KEY_ENV.get(backend.strip().lower())
    return os.environ.get(env_var) if env_var else None


def _urlopen_with_retry(req: urllib.request.Request, timeout: int) -> bytes:
    """Open a urllib Request, retrying on rate-limit and server errors."""
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in _RETRY_STATUSES and attempt < len(_RETRY_DELAYS):
                continue
            raise
        except urllib.error.URLError:
            if attempt < len(_RETRY_DELAYS):
                continue
            raise

TEMPERATURE = 0.0
MAX_TOKENS  = 2048
TIMEOUT_S   = 300

DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL    = "deepseek-chat"
OLLAMA_URL        = "http://localhost:11434"
OLLAMA_MODEL      = "qwen3:8b"
OPENAI_API_BASE   = "https://api.openai.com/v1"
OPENAI_MODEL      = "gpt-4o"
ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_MODEL   = "claude-sonnet-4-6"
ANTHROPIC_VERSION = "2023-06-01"

DEFAULT_BACKEND_MODELS = {
    "deepseek": DEEPSEEK_MODEL,
    "openai": OPENAI_MODEL,
    "anthropic": ANTHROPIC_MODEL,
    "ollama": OLLAMA_MODEL,
}


def _load_project_env() -> None:
    """Load simple KEY=VALUE pairs from repo-root .env without extra deps."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def call_ollama(
    prompt: str,
    model: str = OLLAMA_MODEL,
    timeout: int = TIMEOUT_S,
    ollama_url: str = OLLAMA_URL,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
) -> dict:
    """Call a local Ollama instance. Returns {text, latency_s, error}.

    `max_tokens` overrides the module default (MAX_TOKENS=2048); useful for
    DIN-SQL §5.2 which sets 350 for self-correction and 600 for other stages.
    `stop` is a list of sequences that terminate generation; DIN-SQL uses
    "Q:" for most stages and "#;\\n\\n" for self-correction.
    """
    options = {
        "temperature": TEMPERATURE,
        "num_predict": max_tokens if max_tokens is not None else MAX_TOKENS,
    }
    if stop:
        options["stop"] = stop
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": options,
    }).encode("utf-8")
    t0 = time.time()
    try:
        req = urllib.request.Request(
            f"{ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return {"text": data.get("response", ""), "latency_s": round(time.time() - t0, 3), "error": None}
    except Exception as exc:
        return {"text": "", "latency_s": round(time.time() - t0, 3), "error": str(exc)}


def call_deepseek(
    prompt: str,
    model: str = DEEPSEEK_MODEL,
    api_key: str | None = None,
    timeout: int = 60,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
) -> dict:
    """Call the DeepSeek chat completions API.

    See `call_ollama` for the meaning of `max_tokens` and `stop`. DeepSeek's
    OpenAI-compatible API supports both; we pass them straight through.
    """
    _load_project_env()
    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        return {"text": "", "latency_s": 0.0, "error": "DEEPSEEK_API_KEY not set"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens if max_tokens is not None else MAX_TOKENS,
        "stream": False,
    }
    if stop:
        body["stop"] = stop
    payload = json.dumps(body).encode("utf-8")
    t0 = time.time()
    try:
        req = urllib.request.Request(
            f"{DEEPSEEK_API_BASE}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        # Keep DeepSeek as a single measured request here so latency and failure
        # counts remain comparable to the saved evaluation artifacts.
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        text = data["choices"][0]["message"]["content"]
        return {"text": text, "latency_s": round(time.time() - t0, 3), "error": None}
    except Exception as exc:
        return {"text": "", "latency_s": round(time.time() - t0, 3), "error": str(exc)}


def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    api_key: str | None = None,
    timeout: int = 60,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
) -> dict:
    """Call OpenAI chat completions API. Returns {text, latency_s, error}."""
    _load_project_env()
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return {"text": "", "latency_s": 0.0, "error": "OPENAI_API_KEY not set"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens if max_tokens is not None else MAX_TOKENS,
        "stream": False,
    }
    if stop:
        body["stop"] = stop
    payload = json.dumps(body).encode("utf-8")
    t0 = time.time()
    try:
        req = urllib.request.Request(
            f"{OPENAI_API_BASE}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        data = json.loads(_urlopen_with_retry(req, timeout))
        text = data["choices"][0]["message"]["content"]
        return {"text": text, "latency_s": round(time.time() - t0, 3), "error": None}
    except Exception as exc:
        return {"text": "", "latency_s": round(time.time() - t0, 3), "error": str(exc)}


def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    api_key: str | None = None,
    timeout: int = 60,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
) -> dict:
    """Call Anthropic Messages API. Returns {text, latency_s, error}."""
    _load_project_env()
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return {"text": "", "latency_s": 0.0, "error": "ANTHROPIC_API_KEY not set"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens if max_tokens is not None else MAX_TOKENS,
    }
    if stop:
        body["stop_sequences"] = stop
    payload = json.dumps(body).encode("utf-8")
    t0 = time.time()
    try:
        req = urllib.request.Request(
            f"{ANTHROPIC_API_BASE}/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            method="POST",
        )
        data = json.loads(_urlopen_with_retry(req, timeout))
        text = data["content"][0]["text"]
        return {"text": text, "latency_s": round(time.time() - t0, 3), "error": None}
    except Exception as exc:
        return {"text": "", "latency_s": round(time.time() - t0, 3), "error": str(exc)}


def call_llm(
    prompt: str,
    backend: str,
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
) -> dict:
    """
    Unified dispatcher.  Returns {"text", "latency_s", "error"}.
    Supports DIN-SQL §5.2 hyperparams via the optional `max_tokens` and
    `stop` arguments. Raises ValueError for unknown backend.
    """
    if backend == "ollama":
        return call_ollama(
            prompt, model=model or OLLAMA_MODEL,
            max_tokens=max_tokens, stop=stop,
        )
    if backend == "deepseek":
        return call_deepseek(
            prompt, model=model or DEEPSEEK_MODEL, api_key=api_key,
            max_tokens=max_tokens, stop=stop,
        )
    if backend == "openai":
        return call_openai(
            prompt, model=model or OPENAI_MODEL, api_key=api_key,
            max_tokens=max_tokens, stop=stop,
        )
    if backend == "anthropic":
        return call_anthropic(
            prompt, model=model or ANTHROPIC_MODEL, api_key=api_key,
            max_tokens=max_tokens, stop=stop,
        )
    raise ValueError(f"Unknown LLM backend '{backend}'. Choose 'deepseek', 'openai', 'anthropic', or 'ollama'.")
