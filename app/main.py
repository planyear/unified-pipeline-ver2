# app/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional, Literal, Dict
import uuid
import traceback

from .logging_setup import setup_logging
from .processing.pipeline import run_pipeline
from .models import ProcessingOption  # "Auto-Read" | "Search" | "All Plans"
from .utils.files import save_upload_to_tmp

logger = setup_logging()

app = FastAPI(
    title="PlanYear Unified Pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Optional: redirect root to Swagger UI
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

# -------- In-memory job store (replace with Redis/DB in prod) --------
JOBS: Dict[str, Dict] = {}   # job_id -> {"status": "...", "result": {...}} or {"status":"error","error":"..."}

# -------- Optional request model (only used for JSON endpoints) ------
class ProcessRequest(BaseModel):
    job_id: str
    broker_id: str
    employer_id: str
    option: Literal["Auto-Read", "Search", "All Plans"]
    plan_name: Optional[str] = ""
    prompt_cache: bool = True

# Synchronous processing. May hit Cloudflare 502 timeouts.
@app.post("/v1/process")
async def process_document(
    document: UploadFile = File(...),
    job_id: str = Form(...),
    broker_id: str = Form(...),
    employer_id: str = Form(...),
    option: ProcessingOption = Form(...),
    plan_name: Optional[str] = Form(None),
    prompt_cache: bool = Form(True),
):
    extra = {"job_id": job_id, "broker_id": broker_id, "employer_id": employer_id}

    if option.value == "Search" and not (plan_name and plan_name.strip()):
        raise HTTPException(status_code=422, detail="plan_name is required when option == 'Search'.")

    try:
        tmp = save_upload_to_tmp(document)
        out = run_pipeline(
            input_path=tmp,
            job_id=job_id,
            broker_id=broker_id,
            employer_id=employer_id,
            option=option.value,
            search_plan_name=(plan_name or "").strip(),
            prompt_cache=prompt_cache,
        )
        return JSONResponse(content=out, status_code=200)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error", extra=extra)
        raise HTTPException(status_code=500, detail=str(e))

# Asynchronous processing. Returns 202 + job_id immediately; poll /v1/jobs/{job_id}. Use this to avoid Cloudflare 520 timeouts.
@app.post("/v1/process_async")
async def process_document_async(
    background: BackgroundTasks,
    document: UploadFile = File(...),
    job_id: str = Form(...),
    broker_id: str = Form(...),
    employer_id: str = Form(...),
    option: ProcessingOption = Form(...),
    plan_name: Optional[str] = Form(None),
    prompt_cache: bool = Form(True),
):
    if option.value == "Search" and not (plan_name and plan_name.strip()):
        raise HTTPException(status_code=422, detail="plan_name is required when option == 'Search'.")

    rid = str(uuid.uuid4())
    JOBS[rid] = {"status": "queued"}

    # Save upload now; background task will read it
    tmp = save_upload_to_tmp(document)

    def _work():
        extra = {"job_id": job_id, "broker_id": broker_id, "employer_id": employer_id, "rid": rid}
        JOBS[rid] = {"status": "running"}
        try:
            out = run_pipeline(
                input_path=tmp,
                job_id=job_id,
                broker_id=broker_id,
                employer_id=employer_id,
                option=option.value,
                search_plan_name=(plan_name or "").strip(),
                prompt_cache=prompt_cache,
            )
            JOBS[rid] = {"status": "done", "result": out}
        except Exception as e:
            logger.error("Async job failed: %s\n%s", e, traceback.format_exc(), extra=extra)
            JOBS[rid] = {"status": "error", "error": str(e)}

    background.add_task(_work)
    return JSONResponse({"job_id": rid, "status": "queued"}, status_code=202)

@app.get("/v1/jobs/{rid}")
async def get_job_status(rid: str):
    data = JOBS.get(rid)
    if not data:
        return JSONResponse({"message": "not found"}, status_code=404)
    return JSONResponse(data, status_code=200)
