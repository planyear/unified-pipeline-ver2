# app/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional, Literal, Dict
import uuid
import traceback
import os

from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth

from .logging_setup import setup_logging
from .processing.pipeline import run_pipeline
from .models import ProcessingOption  # "Auto-Read" | "Search" | "All Plans"
from .utils.files import save_upload_to_tmp

# ----------------- Auth / Config -----------------
logger = setup_logging()

ALLOWED_DOMAIN = "planyear.com"

# Required env vars (set these in Render → Environment)
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
SECRET_KEY = os.environ.get("SECRET_KEY")
BASE_URL = os.environ.get("BASE_URL")
if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET and SECRET_KEY and BASE_URL):
    logger.warning("Missing one of GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET/SECRET_KEY/BASE_URL")

REDIRECT_URI = f"{BASE_URL}/auth/callback"

oauth = OAuth()
oauth.register(
    name="google",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    client_kwargs={"scope": "openid email profile", "prompt": "select_account"},
)

def require_user(domain_enforced: bool = True):
    async def _dep(request: Request):
        user = request.session.get("user")
        if not user:
            raise HTTPException(status_code=401, detail="Login required")
        email = user.get("email")
        if domain_enforced and (not email or not email.endswith(f"@{ALLOWED_DOMAIN}")):
            raise HTTPException(status_code=403, detail="Email domain not allowed")
        return user
    return _dep

# ----------------- App & Middleware -----------------
app = FastAPI(
    title="PlanYear Unified Pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY or "dev-secret-change-me")

# Optional: redirect root to Swagger UI
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

# ----------------- Auth Routes -----------------
@app.get("/login", include_in_schema=False)
async def login(request: Request):
    return await oauth.google.authorize_redirect(request, REDIRECT_URI)

@app.get("/auth/callback", include_in_schema=False)
async def auth_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)

    # Try to parse ID token first (preferred)
    userinfo = None
    try:
        if token and "id_token" in token:
            userinfo = await oauth.google.parse_id_token(request, token)
    except Exception:
        userinfo = None

    # Fallback to UserInfo endpoint if no id_token
    if not userinfo:
        userinfo = await oauth.google.userinfo(token=token)

    email = (userinfo or {}).get("email")
    email_verified = (userinfo or {}).get("email_verified", False)

    # Enforce @planyear.com
    if not email or not email_verified or not email.endswith(f"@{ALLOWED_DOMAIN}"):
        request.session.clear()
        raise HTTPException(status_code=403, detail="Only @planyear.com allowed")

    request.session["user"] = {"email": email, "name": userinfo.get("name")}
    return RedirectResponse(url="/docs")

@app.get("/logout", include_in_schema=False)
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/")

@app.get("/me")
async def me(request: Request):
    # small helper to see who you are
    return request.session.get("user") or {}

# ----------------- Jobs Store -----------------
JOBS: Dict[str, Dict] = {}   # job_id -> {"status": "...", "result": {...}} or {"status":"error","error":"..."}

# ----------------- Request Model -----------------
class ProcessRequest(BaseModel):
    job_id: str
    broker_id: str
    employer_id: str
    option: Literal["Auto-Read", "Search", "All Plans"]
    plan_name: Optional[str] = ""
    prompt_cache: bool = True

# ----------------- Protected API -----------------
# NOTE: the only change to your endpoints is adding `user=Depends(require_user())`

@app.post("/v1/process")
async def process_document(
    document: UploadFile = File(...),
    job_id: str = Form(...),
    broker_id: str = Form(...),
    employer_id: str = Form(...),
    option: ProcessingOption = Form(...),
    plan_name: Optional[str] = Form(None),
    prompt_cache: bool = Form(True),
    user=Depends(require_user()),
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
    user=Depends(require_user()),
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
async def get_job_status(rid: str, user=Depends(require_user())):
    data = JOBS.get(rid)
    if not data:
        return JSONResponse({"message": "not found"}, status_code=404)
    return JSONResponse(data, status_code=200)
