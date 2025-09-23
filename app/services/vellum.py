import requests
import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException
from ..config import settings

logger = logging.getLogger("pipeline")

_PROMPT_CACHE: Dict[str, str] = {}

def _join_vellum_message_text(payload: Dict[str, Any]) -> Optional[str]:
    """
    Extract plain prompt text from Vellum provider-payload response:
      payload -> messages[] -> content[] -> {type: "text", text: "..."}
    If multiple text parts exist, join them with newlines.
    """
    try:
        msgs = payload.get("messages") or []
        if not msgs:
            return None
        contents = msgs[0].get("content") or []
        parts = []
        for c in contents:
            if isinstance(c, dict) and c.get("type") == "text":
                t = c.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t)
        return "\n".join(parts) if parts else None
    except Exception:
        return None

def get_prompt_from_vellum(deployment_name: str, version: Optional[str] = None) -> Optional[str]:
    """
    Fetch the *deployed* prompt template from Vellum's provider-payload API.
    """
    if not settings.VELLUM_API_KEY:
        raise HTTPException(status_code=500, detail="VELLUM_API_KEY not set.")
    base = (settings.VELLUM_BASE_URL or "https://api.vellum.ai").rstrip("/")
    url = f"{base}/v1/deployments/provider-payload"
    headers = {
        "X-API-KEY": settings.VELLUM_API_KEY,
        "Content-Type": "application/json",
    }
    body: Dict[str, Any] = {"deployment_name": deployment_name, "inputs": []}
    if version:
        body["tag"] = version

    try:
        resp = requests.post(url, json=body, headers=headers, timeout=30)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Vellum provider-payload request failed: {e}")

    if resp.status_code >= 300:
        # Some deployments insist on at least one input; retry with a dummy "document".
        if resp.status_code == 400:
            try:
                body_retry = dict(body)
                body_retry["inputs"] = [{"type": "STRING", "name": "document", "value": ""}]
                resp2 = requests.post(url, json=body_retry, headers=headers, timeout=30)
                if 200 <= resp2.status_code < 300:
                    data2 = resp2.json()
                    payload2 = (data2 or {}).get("payload") or {}
                    txt2 = _join_vellum_message_text(payload2)
                    if txt2:
                        return txt2
            except requests.exceptions.RequestException:
                pass

        raise HTTPException(
            status_code=502,
            detail=f"Vellum prompt fetch failed for '{deployment_name}': {resp.status_code} {resp.text[:300]}",
        )

    data = resp.json()
    payload = (data or {}).get("payload") or {}
    text = _join_vellum_message_text(payload)
    if text:
        return text

    # Fallback: sometimes prompt text lives under other keys
    for k in ("compiled_prompt", "prompt", "template", "text", "content", "body"):
        v = (data.get(k) or payload.get(k))
        if isinstance(v, str) and v.strip():
            return v

    raise HTTPException(
        status_code=502,
        detail=f"Vellum provider-payload returned no text for '{deployment_name}'. Body keys: {list(data.keys())}",
    )

def get_prompt(slug_or_deployment: str, version: Optional[str] = None) -> str:
    """
    Cached accessor that uses provider-payload by deployment name (slug).
    """
    key = f"{slug_or_deployment}:{version or 'latest'}"
    cached = _PROMPT_CACHE.get(key)
    if cached:
        return cached
    remote = get_prompt_from_vellum(slug_or_deployment, version)
    if not remote:
        raise HTTPException(status_code=500, detail=f"Missing prompt '{slug_or_deployment}' from Vellum.")
    _PROMPT_CACHE[key] = remote
    return remote
