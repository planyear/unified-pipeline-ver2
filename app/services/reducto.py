# app/services/reducto.py
import time
import json
import logging
from pathlib import Path
from typing import Optional, Union, List, Tuple

import requests
from fastapi import HTTPException

from ..config import settings

logger = logging.getLogger("pipeline")

# -----------------------------
# Reducto: PDF → Markdown/Text
# -----------------------------

def _log_reducto_summary(pj: dict) -> None:
    """Log a compact summary of the Reducto response (no full JSON)."""
    data = pj.get("data") or {}
    usage = pj.get("usage") or data.get("usage") or {}
    logger.info(
        "Reducto summary | job_id=%s duration=%s pages=%s credits=%s",
        pj.get("job_id") or data.get("job_id"),
        pj.get("duration") or data.get("duration"),
        usage.get("num_pages"),
        usage.get("credits"),
    )

def _result_to_page_marked_text(pj: dict) -> Optional[str]:
    """
    Turn result.chunks into:
      [START OF PAGE 1]
      ...page text...
      [END OF PAGE 1]
    Assumes chunking by page. Uses 'embed' first, then 'content'.
    """
    res = pj.get("result") or (pj.get("data") or {}).get("result")
    if not isinstance(res, dict):
        return None
    chunks = res.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return None

    parts = []
    for idx, ch in enumerate(chunks, start=1):
        if not isinstance(ch, dict):
            continue
        text = (ch.get("embed") or ch.get("content") or "").strip()
        if not text:
            continue
        parts.append(f"[START OF PAGE {idx}]")
        parts.append(text)
        parts.append(f"[END OF PAGE {idx}]")
    if not parts:
        return None
    return "\n\n".join(parts)

def _download_text_from_url(url: str, timeout: int = 180) -> str:
    r = requests.get(url, timeout=timeout)
    if r.status_code >= 300:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download text from {url}: {r.status_code} {r.text[:200]}",
        )
    # If the signed URL returns JSON with text fields, use them; else raw body.
    try:
        j = r.json()
        for k in ("markdown", "text", "content"):
            v = j.get(k)
            if isinstance(v, str) and v.strip():
                return v
    except Exception:
        pass
    return r.text

def _extract_text_from_payload(pj: dict) -> Optional[str]:
    """
    Try inline fields; then page-marked chunks; then URL downloads.
    """
    # 1) Inline
    for k in ("markdown", "text", "content"):
        v = pj.get(k)
        if isinstance(v, str) and v.strip():
            return v

    # 2) Nested (data.*)
    data = pj.get("data")
    if isinstance(data, dict):
        for k in ("markdown", "text", "content"):
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                return v

    # 3) Structured chunks → page markers
    v = _result_to_page_marked_text(pj)
    if v:
        return v

    # 4) Downloadable URLs (last resort)
    url_keys = ("markdown_url", "text_url", "content_url", "md_url", "result_url", "url")
    for k in url_keys:
        url = pj.get(k)
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            return _download_text_from_url(url)
    if isinstance(data, dict):
        for k in url_keys:
            url = data.get(k)
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                return _download_text_from_url(url)

    return None

def _poll_job_for_markdown(job_id: str, max_wait_s: int = 180) -> Optional[str]:
    """
    Poll a few likely job endpoints until text materializes.
    Different tenants expose different paths; we try several.
    """
    headers = {"Authorization": f"Bearer {settings.REDUCTO_API_KEY}"}
    base = settings.REDUCTO_BASE_URL
    candidates = [
        f"{base}/jobs/{job_id}",
        f"{base}/job/{job_id}",
        f"{base}/parse/jobs/{job_id}",
        f"{base}/results/{job_id}",
    ]
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        for url in candidates:
            try:
                r = requests.get(url, headers=headers, timeout=20)
            except requests.exceptions.RequestException:
                continue
            if r.status_code >= 300:
                continue
            try:
                j = r.json()
            except Exception:
                if r.text.strip():
                    return r.text
                continue

            txt = _extract_text_from_payload(j)
            if txt:
                return txt

            status = (j.get("status") or (j.get("data") or {}).get("status") or "").lower()
            if status in {"queued", "processing", "running", "in_progress"}:
                continue
        time.sleep(2)
    return None

def pdf_to_markdown(
    pdf_path: Union[str, Path],
    *,
    log_payload: bool = False,
    return_meta: bool = False,
) -> Union[str, Tuple[str, dict]]:
    """
    Robust Reducto flow:
      1) POST /upload  -> returns document_url or file_id (reducto://...)
      2) POST /parse   -> asks for markdown/text; normalize any response shape
      3) If needed, poll job endpoints or try /extract

    Returns:
      - markdown string (default), or
      - (markdown, full_json) when return_meta=True

    If log_payload=True, pretty-prints the full Reducto JSON to logs.
    """
    if not (settings.REDUCTO_API_KEY and settings.REDUCTO_BASE_URL):
        raise HTTPException(status_code=500, detail="Reducto not configured (REDUCTO_API_KEY/REDUCTO_BASE_URL).")

    path = Path(pdf_path)

    # --- Step 1: upload ---
    upload_url = f"{settings.REDUCTO_BASE_URL}/upload"
    up_headers = {"Authorization": f"Bearer {settings.REDUCTO_API_KEY}"}
    files = {"file": (path.name, path.open("rb"), "application/pdf")}
    try:
        up_resp = requests.post(upload_url, headers=up_headers, files=files, timeout=180)
    finally:
        files["file"][1].close()

    if up_resp.status_code >= 300:
        raise HTTPException(
            status_code=502,
            detail=f"Reducto upload failed ({up_resp.status_code}) at {upload_url}: {up_resp.text}",
        )

    try:
        up_json = up_resp.json()
    except Exception:
        up_json = {}

    # Prefer document_url; accept file_id (reducto://...pdf) as a handle.
    document_url = (
        up_json.get("document_url")
        or up_json.get("url")
        or (up_json.get("data") or {}).get("document_url")
        or up_json.get("file_id")
    )
    if not document_url:
        raise HTTPException(
            status_code=502,
            detail=f"Reducto upload succeeded but no document_url found. Body: {up_resp.text[:300]}",
        )

    # --- Step 2: parse ---
    parse_url = f"{settings.REDUCTO_BASE_URL}/parse"
    parse_headers = {"Authorization": f"Bearer {settings.REDUCTO_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "document_url": document_url,
        "options": {
            "force_url_result": False,
            "ocr_mode": "standard",
            "extraction_mode": "ocr",
            "chunking": {"chunk_mode": "page"},
            "table_summary": {"enabled": False},
            "figure_summary": {"enabled": False},
        },
        "advanced_options": {
            "enable_change_tracking": False,
            "ocr_system": "highres",
            "table_output_format": "html",
            "merge_tables": False,
            "include_color_information": False,
            "continue_hierarchy": False,
            "keep_line_breaks": True,
            "large_table_chunking": {"enabled": False},
            "add_page_markers": True,
            "exclude_hidden_sheets": True,
            "exclude_hidden_rows_cols": True,
        },
        "experimental_options": {"danger_filter_wide_boxes": False, "rotate_pages": True, "enable_scripts": True},
        "priority": True,
    }
    try:
        p_resp = requests.post(parse_url, headers=parse_headers, json=payload, timeout=300)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Reducto request to {parse_url} failed: {e}")

    if p_resp.status_code >= 300:
        raise HTTPException(status_code=502, detail=f"Reducto parse failed: {p_resp.status_code} — {p_resp.text}")

    # --- Normalize response into JSON (pj) ---
    try:
        pj = p_resp.json()
    except Exception:
        # Fallback: treat body as text
        txt = p_resp.text
        if txt and txt.strip():
            logger.info("Reducto Conversion Finished", extra={"job_id":"-", "broker_id":"-", "employer_id":"-"})
            return (txt, {"raw": txt}) if return_meta else txt
        raise HTTPException(status_code=502, detail="Reducto parse returned an empty body.")

    # Optional compact summary
    _log_reducto_summary(pj)

    # Optional full payload logging
    if log_payload:
        try:
            logger.info("Reducto full response:\n%s", json.dumps(pj, indent=2, ensure_ascii=False))
        except Exception:
            logger.info("Reducto full response (non-JSON)")

    # Build page-marked text
    txt = _extract_text_from_payload(pj)
    if txt:
        usage = pj.get("usage") or (pj.get("data") or {}).get("usage") or {}
        logger.info(
            "Reducto Conversion Finished (pages=%s, credits=%s)",
            usage.get("num_pages"), usage.get("credits"),
            extra={"job_id": "-", "broker_id": "-", "employer_id": "-"},
        )
        return (txt, pj) if return_meta else txt

    # --- If only a job handle was returned, poll for completion ---
    job_id = pj.get("job_id") or (pj.get("data") or {}).get("job_id")
    if job_id:
        txt2 = _poll_job_for_markdown(job_id, max_wait_s=180)
        if txt2:
            logger.info("Reducto Conversion Finished", extra={"job_id":"-", "broker_id":"-", "employer_id":"-"})
            return (txt2, pj) if return_meta else txt2

    # --- Step 3: fallback to /extract if tenant requires it ---
    try:
        extract_url = f"{settings.REDUCTO_BASE_URL}/extract"
        ex_headers = {"Authorization": f"Bearer {settings.REDUCTO_API_KEY}", "Content-Type": "application/json"}
        ex_payload = {"document_url": document_url, "output": {"format": "markdown"}, "priority": True}
        ex_resp = requests.post(extract_url, headers=ex_headers, json=ex_payload, timeout=300)
        if ex_resp.status_code < 300:
            try:
                exj = ex_resp.json()
                txt3 = _extract_text_from_payload(exj)
            except Exception:
                txt3 = ex_resp.text
                exj = {"raw": txt3}
            if txt3 and txt3.strip():
                logger.info("Reducto Conversion Finished", extra={"job_id":"-", "broker_id":"-", "employer_id":"-"})
                return (txt3, exj) if return_meta else txt3
    except Exception:
        pass

    raise HTTPException(
        status_code=502,
        detail=f"Reducto parse returned no markdown/text. Body: {str(pj)[:400]}",
    )
