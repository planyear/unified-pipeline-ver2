# app/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional, Literal

from .logging_setup import setup_logging
from .processing.pipeline import run_pipeline
from .models import ProcessingOption  # your Enum with .value matching "Auto-Read" | "Search" | "All Plans"
from .utils.files import save_upload_to_tmp

logger = setup_logging()

app = FastAPI(
    title="PlanYear Unified Pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Optional: make root redirect to Swagger UI.
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

# NOTE: This Pydantic class is not used by the endpoint below (since we accept multipart/form-data).
# Keep it only if you also expose a JSON body endpoint. Otherwise you can delete it.
class ProcessRequest(BaseModel):
    job_id: str
    broker_id: str
    employer_id: str
    option: Literal["Auto-Read", "Search", "All Plans"]
    plan_name: Optional[str] = ""
    prompt_cache: bool = True

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
    """
    Multipart form endpoint:
      - `document` (file): PDF or Office doc
      - `job_id`, `broker_id`, `employer_id`: strings (required)
      - `option`: "Auto-Read" | "Search" | "All Plans" (required)
      - `plan_name`: required iff option == "Search"
      - `prompt_cache`: enable OpenRouter prompt caching for big document chunk
    """
    extra = {"job_id": job_id, "broker_id": broker_id, "employer_id": employer_id}

    if option.value == "Search":
        if not plan_name or not plan_name.strip():
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

        return JSONResponse(
            content=out,
            status_code=200
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error", extra=extra)
        raise HTTPException(status_code=500, detail=str(e))
