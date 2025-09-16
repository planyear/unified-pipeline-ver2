import logging, requests
from typing import List, Dict, Any, Optional
from ..config import settings

logger = logging.getLogger("pipeline")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def _headers():
    return {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://planyear.tools",
        "X-Title": "PlanYear Unified Pipeline",
    }

def _usage_from_headers(h: requests.structures.CaseInsensitiveDict) -> Dict[str, Any]:
    """Parse usage/caching hints from headers that some providers return."""
    # OpenRouter sometimes exposes usage/cost in headers for certain providers.
    # We check multiple common keys to be robust.
    keys = {
        "prompt_tokens": [
            "x-openrouter-usage-prompt-tokens",
            "x-openai-meta-usage-input-tokens",
        ],
        "completion_tokens": [
            "x-openrouter-usage-completion-tokens",
            "x-openai-meta-usage-output-tokens",
        ],
        "total_tokens": [
            "x-openrouter-usage-total-tokens",
            "x-openai-meta-usage-total-tokens",
        ],
        "cost": [
            "x-openrouter-credits-consumed",
            "x-openrouter-usage-cost",
        ],
        "model": ["openrouter-model", "x-openrouter-model"],
        "provider": ["openrouter-provider", "x-openrouter-provider"],
        "gen_id": ["openrouter-id", "x-openrouter-id"],
        "cache_read": ["openrouter-cache-read", "x-openrouter-cache-read"],
        "cache_write": ["openrouter-cache-write", "x-openrouter-cache-write"],
        "cache_status": ["x-cache-proxy", "cf-cache-status", "x-served-from-cache"],
    }
    out: Dict[str, Any] = {}
    for field, names in keys.items():
        for name in names:
            if name in h:
                out[field] = h.get(name)
                break
    return out

def _log_usage(resp_json: Dict[str, Any], resp_headers: requests.structures.CaseInsensitiveDict) -> None:
    usage = resp_json.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    cost = usage.get("cost")
    model = resp_json.get("model")

    logger.info(
        "LLM USAGE\nmodel=%s "
        "prompt_tokens=%s completion_tokens=%s total_tokens=%s cached_prompt=%s cost=%s ",
        model, prompt_tokens, completion_tokens, total_tokens,
        cached_tokens, cost,
        extra={"job_id": "-", "broker_id": "-", "employer_id": "-"},
    )

def chat_completion(
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    request_overrides: Optional[Dict[str, Any]] = None,
    return_full: bool = True,
):
    body: Dict[str, Any] = {
        "model": settings.OPENROUTER_MODEL,
        "messages": messages,
        "temperature": temperature,
        "usage": {"include": True},     # ask OpenRouter to include usage in the JSON
    }
    if request_overrides:
        body.update(request_overrides)

    r = requests.post(OPENROUTER_URL, headers=_headers(), json=body, timeout=180)
    r.raise_for_status()
    out = r.json()

    # Log usage either from JSON or headers
    _log_usage(out, r.headers)

    if return_full:
        return out  # caller can inspect out["usage"], out["id"], etc.

    return out["choices"][0]["message"]["content"]

from typing import Dict, Any  # ensure this exists in the file

def run_prompt_with_context(
    context_markdown: str,
    prompt_text: str,
    *,
    cache: bool = True,           # give it a default and place before return_full
    return_full: bool = False,
):
    """
    Multipart message; cache the LARGE document chunk when cache=True.
    """
    doc_part: Dict[str, Any] = {
        "type": "text",
        "text": f"<Document>\n{context_markdown}\n</Document>",
    }
    if cache:
        doc_part["cache_control"] = {"type": "ephemeral"}

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a precise benefits-document extraction assistant."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                doc_part,  # <-- separate list element; don't wrap inside another dict
            ],
        },
    ]
    return chat_completion(messages, temperature=0.0, return_full=return_full)
