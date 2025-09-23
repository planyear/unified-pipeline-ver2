# app/services/llm.py
import os, json, time, hashlib, logging, requests
from typing import List, Dict, Any, Optional, Tuple
from ..config import settings

logger = logging.getLogger("pipeline")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://planyear.tools",
        "X-Title": "PlanYear Unified Pipeline",
    }

# ---------- canonical cached part ----------
def _canonical_doc(md: str) -> str:
    return (md or "").replace("\r\n", "\n").strip()

def build_cachable_doc_part(md: str, enable_cache: bool) -> Dict[str, Any]:
    """
    ONE stable content block we reuse across *all* steps.
    It already includes <Document> tags, so we never inline doc in templates.
    """
    text = f"<Document>\n{_canonical_doc(md)}\n</Document>"
    part: Dict[str, Any] = {"type": "text", "text": text}
    if enable_cache:
        part["cache_control"] = {"type": "ephemeral"}
    return part
def _maybe_text_part(s: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return a valid text part only if non-empty after stripping."""
    if s is None:
        return None
    t = s.strip()
    if not t:
        return None
    return {"type": "text", "text": t}


def compose_messages_with_document(
    *,
    template: str,
    document_md: str,
    enable_cache: bool,
    system_text: str = "You are a precise benefits-document extraction assistant.",
    extra_user_texts: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build messages with a single cached doc part (that includes <Document> tags).
    If the template contains <Document></Document>, strip the placeholder, and
    DO NOT send an empty text block.
    """
    extra_user_texts = extra_user_texts or []

    # Strip the empty document slot (it often is the whole template)
    template_no_slot = (template or "").replace("<Document></Document>", "")

    # Build the cacheable doc part (<Document> ... </Document>)
    doc_part = build_cachable_doc_part(document_md, enable_cache=enable_cache)

    user_content: List[Dict[str, Any]] = []

    # Only append the template text part if it’s actually non-empty
    maybe_tpl = _maybe_text_part(template_no_slot)
    if maybe_tpl:
        user_content.append(maybe_tpl)

    # Always append the cached document part
    user_content.append(doc_part)

    # Append any extra user texts, but only if non-empty
    for txt in extra_user_texts:
        maybe_extra = _maybe_text_part(txt)
        if maybe_extra:
            user_content.append(maybe_extra)

    messages = [
        {"role": "system", "content": [_maybe_text_part(system_text) or {"type": "text", "text": ""}]},
        {"role": "user", "content": user_content},
    ]
    return messages

# ---------- prompt save (already in your code path) ----------
def _save_prompt(messages: List[Dict[str, Any]], label: str) -> None:
    if not getattr(settings, "LOG_PROMPTS", False):
        return
    try:
        os.makedirs("/tmp/llm_prompts", exist_ok=True)
        doc_parts = 0
        chars = 0
        # count only user content chars for a quick sanity check
        for m in messages:
            if m.get("role") == "user":
                for c in (m.get("content") or []):
                    if isinstance(c, dict):
                        s = c.get("text") or ""
                        chars += len(s)
                        if c.get("cache_control"):
                            doc_parts += 1
        path = f"/tmp/llm_prompts/{int(time.time())}_{label}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"messages": messages}, f, ensure_ascii=False, indent=2)
        logger.info("LLM PROMPT saved: %s (content_parts=%s, chars=%s) -> %s",
                    label, doc_parts + 2, chars, path)
    except Exception as e:
        logger.warning("Failed to save LLM prompt: %s", e)

# ---------- usage logging ----------
def _usage_from_headers(h: requests.structures.CaseInsensitiveDict) -> Dict[str, Any]:
    keys = {
        "prompt_tokens": ["x-openrouter-usage-prompt-tokens","x-openai-meta-usage-input-tokens"],
        "completion_tokens": ["x-openrouter-usage-completion-tokens","x-openai-meta-usage-output-tokens"],
        "total_tokens": ["x-openrouter-usage-total-tokens","x-openai-meta-usage-total-tokens"],
        "cost": ["x-openrouter-credits-consumed","x-openrouter-usage-cost"],
        "model": ["openrouter-model","x-openrouter-model"],
        "provider": ["openrouter-provider","x-openrouter-provider"],
        "gen_id": ["openrouter-id","x-openrouter-id"],
        "cache_read": ["openrouter-cache-read","x-openrouter-cache-read"],
        "cache_write": ["openrouter-cache-write","x-openrouter-cache-write"],
        "cache_status": ["x-cache-proxy","cf-cache-status","x-served-from-cache"],
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
    cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    data = {
        "model": resp_json.get("model"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cost": usage.get("cost"),
        "cached_prompt": cached_tokens,
        **_usage_from_headers(resp_headers),
    }
    logger.info(
        "LLM USAGE | model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s "
        "cached_prompt=%s cost=%s cache_read=%s cache_write=%s cache_status=%s",
        data.get("model"), data.get("prompt_tokens"), data.get("completion_tokens"),
        data.get("total_tokens"), data.get("cached_prompt"), data.get("cost"),
        data.get("cache_read"), data.get("cache_write"), data.get("cache_status"),
        extra={"job_id": "-", "broker_id": "-", "employer_id": "-"},
    )
def chat_completion(
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    request_overrides: Optional[Dict[str, Any]] = None,
    return_full: bool = True,
    log_label: Optional[str] = None,
):
    body: Dict[str, Any] = {
        "model": settings.OPENROUTER_MODEL,
        "messages": messages,
        "temperature": temperature,
        "usage": {"include": True},
    }
    if request_overrides:
        body.update(request_overrides)

    # Optional: debug sizes (helps spot oversize requests)
    try:
        user_content = next((m.get("content", []) for m in messages if m.get("role") == "user"), [])
        approx_chars = sum(len(c.get("text", "")) for c in user_content if isinstance(c, dict))
        logger.info("LLM payload approx chars=%s %s", approx_chars, f"({log_label})" if log_label else "")
    except Exception:
        pass

    r = requests.post(OPENROUTER_URL, headers=_headers(), json=body, timeout=180)

    # ---- NEW: print server error text BEFORE raising ----
    if r.status_code >= 400:
        try:
            logger.error("OpenRouter error %s %s | %s", r.status_code, r.reason, r.text)
        except Exception:
            pass
        # rethrow with body attached so FastAPI shows it in the 500
        raise requests.HTTPError(f"{r.status_code} {r.reason}: {r.text}", response=r)

    r.raise_for_status()
    out = r.json()

    _log_usage(out, r.headers)

    if return_full:
        return out
    return out["choices"][0]["message"]["content"]
