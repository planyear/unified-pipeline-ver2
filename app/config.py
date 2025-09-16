import os

# Try to load .env locally; ignore if package is missing (e.g., on Render)
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

if load_dotenv:
    load_dotenv()  # only runs locally where python-dotenv is installed

class Settings:
    OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
    CLOUDCONVERT_API_KEY: str | None = os.getenv("CLOUDCONVERT_API_KEY")
    REDUCTO_API_KEY: str | None = os.getenv("REDUCTO_API_KEY")
    REDUCTO_BASE_URL: str = os.getenv("REDUCTO_BASE_URL", "https://platform.reducto.ai")
    VELLUM_API_KEY: str | None = os.getenv("VELLUM_API_KEY")
    VELLUM_BASE_URL: str = os.getenv("VELLUM_BASE_URL", "https://api.vellum.ai")
    GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    TOKEN_HARD_LIMIT: int = int(os.getenv("TOKEN_HARD_LIMIT", "50000"))

settings = Settings()
