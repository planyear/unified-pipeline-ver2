from fastapi import Request, HTTPException

ALLOWED_DOMAIN = "planyear.com"

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
