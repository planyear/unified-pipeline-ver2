import json, time, os, logging
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any

import requests
from fastapi import HTTPException

from ..config import settings

logger = logging.getLogger("pipeline")

# -----------------------------
# Reducto: PDF → Cleaned JSON
# -----------------------------

def _save_cleaned(obj, label: str = "cleaned", out_dir: str = "/tmp/reducto_cleaned") -> str:
    """
    Save a cleaned variable to disk.
    - If obj is str -> writes .txt
    - If obj is dict/list -> writes .json (pretty)
    Returns the full path.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    safe_label = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in label)[:80]

    if isinstance(obj, (dict, list)):
        path = os.path.join(out_dir, f"{ts}_{safe_label}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    else:
        path = os.path.join(out_dir, f"{ts}_{safe_label}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(obj))

    logger.info("Saved cleaned output -> %s", path)
    return path

def _log_reducto_summary(pj: dict) -> None:
    """Compact usage/summary log (no full JSON)."""
    data = pj.get("data") or {}
    usage = pj.get("usage") or data.get("usage") or {}
    logger.info(
        "Reducto summary | job_id=%s duration=%s pages=%s credits=%s",
        pj.get("job_id") or data.get("job_id"),
        pj.get("duration") or data.get("duration"),
        usage.get("num_pages"),
        usage.get("credits"),
    )

def _download_signed_result_json(url: str, timeout: int = 180) -> Optional[dict]:
    """Fetch JSON from Reducto's signed result URL when chunks aren't inline."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("Failed to download signed result JSON: %s", e)
        return None

def _extract_chunks(pj: dict) -> List[dict]:
    """
    Get result.chunks from the immediate payload; if empty, try result.url (signed S3).
    Returns [] if nothing usable.
    """
    res = pj.get("result") or (pj.get("data") or {}).get("result") or {}
    chunks = res.get("chunks")
    if isinstance(chunks, list) and chunks:
        return chunks

    # Fallback: some tenants put chunks behind a signed URL
    for key in ("url", "result_url", "md_url", "content_url"):
        signed = res.get(key)
        if isinstance(signed, str) and signed.startswith(("http://", "https://")):
            j = _download_signed_result_json(signed)
            if isinstance(j, dict):
                ch = j.get("chunks")
                if isinstance(ch, list) and ch:
                    return ch
    return []

def _first_page_from_blocks(blocks: Any) -> str:
    """
    Company logic: take page number from blocks[1].bbox.page when present,
    else fall back to the first block with a bbox.page.
    """
    try:
        if isinstance(blocks, list) and len(blocks) > 1:
            bbox = blocks[1].get("bbox") or {}
            page = bbox.get("page")
            if page is not None:
                return str(page)
    except Exception:
        pass

    # Fallback: first block with a page number
    if isinstance(blocks, list):
        for b in blocks:
            try:
                bbox = b.get("bbox") or {}
                page = bbox.get("page")
                if page is not None:
                    return str(page)
            except Exception:
                continue
    return ""

def _clean_reducto_payload(pj: dict) -> Dict[str, Any]:
    """
    Company-cleaned structure:
      {"pages":[ {"page":"<n>","content":"<text>"}, ... ]}
    Uses chunk.content (falls back to chunk.embed) and block-derived page number.
    """
    chunks = _extract_chunks(pj)
    pages: List[Dict[str, str]] = []

    for ch in chunks or []:
        if not isinstance(ch, dict):
            continue
        blocks = ch.get("blocks") or []
        if not blocks:
            # keep behavior consistent with the provided spec: skip truly empty pages
            continue
        content = ch.get("content")
        if not (isinstance(content, str) and content.strip()):
            content = ch.get("embed") or ""
        page_num = _first_page_from_blocks(blocks)
        pages.append({
            "content": content or "",
            "page": page_num,
        })

    return {"pages": pages}

def _poll_job_for_cleaned(job_id: str, max_wait_s: int = 180) -> Optional[Dict[str, Any]]:
    """
    Poll typical job endpoints until a usable cleaned payload can be produced.
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
                if r.status_code >= 300:
                    continue
                j = r.json()
            except Exception:
                continue

            cleaned = _clean_reducto_payload(j)
            if cleaned.get("pages"):
                return cleaned

            status = (j.get("status") or (j.get("data") or {}).get("status") or "").lower()
            if status in {"queued", "processing", "running", "in_progress"}:
                continue
        time.sleep(2)
    return None

def pdf_to_markdown(
    pdf_path: Union[str, Path],
    *,
    log_content: bool = True,     # if True, log cleaned_data
) -> Union[str, Tuple[str, dict]]:
    """
    Upload → Parse with your exact payload → CLEAN the response to {"pages":[...]}.
    Returns the CLEANED **JSON string** so the LLM receives <Document>{...}</Document>.
    """
    if not (settings.REDUCTO_API_KEY and settings.REDUCTO_BASE_URL):
        raise HTTPException(status_code=500, detail="Reducto not configured (REDUCTO_API_KEY/REDUCTO_BASE_URL).")

    path = Path(pdf_path)

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

    # --- Step 2: parse (your exact payload preserved) ---
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

    try:
        pj = p_resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Reducto parse returned a non-JSON body.")

    _log_reducto_summary(pj)

    # clean_and_save_json()
    cleaned = _clean_reducto_payload(pj)

    if not cleaned.get("pages"):
        job_id = pj.get("job_id") or (pj.get("data") or {}).get("job_id")
        if job_id:
            cleaned = _poll_job_for_cleaned(job_id, max_wait_s=180) or {"pages": []}

    if not cleaned.get("pages"):
        try:
            extract_url = f"{settings.REDUCTO_BASE_URL}/extract"
            ex_headers = {"Authorization": f"Bearer {settings.REDUCTO_API_KEY}", "Content-Type": "application/json"}
            ex_payload = {"document_url": document_url, "output": {"format": "markdown"}, "priority": True}
            ex_resp = requests.post(extract_url, headers=ex_headers, json=ex_payload, timeout=300)
            if ex_resp.status_code < 300:
                ex_json = ex_resp.json()
                cleaned = _clean_reducto_payload(ex_json)
        except Exception:
            pass

    if not cleaned.get("pages"):
        raise HTTPException(status_code=502, detail="Reducto parse produced no usable content.")

    if log_content: 
        logger.info("Cleaned Reducto Output:\n%s", cleaned)
        _save_cleaned(cleaned, label="reducto_page_marked")

    # Return a JSON string (stable, LLM-ready) so <Document>{...}</Document> gets the cleaned data
    cleaned_json_str = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))
    logger.info(
        "Reducto Conversion Finished (cleaned_pages=%d)",
        len(cleaned.get("pages", [])),
        extra={"job_id": "-", "broker_id": "-", "employer_id": "-"},
    )

    return cleaned_json_str
