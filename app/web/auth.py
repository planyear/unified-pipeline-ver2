import os
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from authlib.integrations.starlette_client import OAuth

router = APIRouter()

GOOGLE_CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]
GOOGLE_CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]
BASE_URL = os.environ["BASE_URL"]
REDIRECT_URI = os.environ["REDIRECT_URI"]
ALLOWED_DOMAIN = "planyear.com"

oauth = OAuth()
oauth.register(
    name="google",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    client_kwargs={"scope": "openid email profile", "prompt": "select_account"},
)

@router.get("/login")
async def login(request: Request):
    return await oauth.google.authorize_redirect(request, REDIRECT_URI)

@router.get("/auth/callback")
async def auth_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    userinfo = await oauth.google.parse_id_token(request, token)
    email = userinfo.get("email")
    if not userinfo.get("email_verified") or not email or not email.endswith(f"@{ALLOWED_DOMAIN}"):
        request.session.clear()
        raise HTTPException(status_code=403, detail="Only @planyear.com allowed")
    request.session["user"] = {"email": email, "name": userinfo.get("name")}
    return RedirectResponse(url="/")

@router.get("/me")
async def me(request: Request):
    return JSONResponse(request.session.get("user") or {})

@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/")
