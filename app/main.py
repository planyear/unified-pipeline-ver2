from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from .models import ProcessRequest, ProcessResponse, ProcessingOption
from .logging_setup import setup_logging
from .processing.pipeline import run_pipeline
from .utils.files import save_upload_to_tmp
from pydantic import BaseModel, StrictBool
from typing import Literal, Optional

logger = setup_logging()
app = FastAPI(title="PlanYear Unified Pipeline")

class ProcessRequest(BaseModel):
    job_id: str
    broker_id: str
    employer_id: str
    option: Literal["Auto-Read", "Search", "All Plans"]
    plan_name: Optional[str] = ""
    prompt_cache: bool = True

@app.post("/v1/process", response_model=ProcessResponse)
async def process_document(
    document: UploadFile = File(...),
    job_id: str = Form(...),
    broker_id: str = Form(...),
    employer_id: str = Form(...),
    option: ProcessingOption = Form(...),
    plan_name: str = Form(""),
    prompt_cache: bool = Form(False),
):
    extra = {"job_id": job_id, "broker_id": broker_id, "employer_id": employer_id}
    try:
        tmp = save_upload_to_tmp(document)
        out = run_pipeline(
            input_path=tmp,
            job_id=job_id,
            broker_id=broker_id,
            employer_id=employer_id,
            option=option.value,
            search_plan_name=plan_name or "",
            prompt_cache=prompt_cache,
        )
        return JSONResponse(content=out, status_code=200 if out.get("message") == "OK" or "SBC" in out.get("message","") else 200)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error", extra=extra)
        raise HTTPException(status_code=500, detail=str(e))
