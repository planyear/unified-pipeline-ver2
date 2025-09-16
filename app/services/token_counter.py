import logging
from ..config import settings

logger = logging.getLogger("pipeline")

def count_tokens_google(markdown_text: str) -> int:
    try:
        import google.generativeai as genai
    except Exception as e:
        logger.error("Token counter import failed: %s", e, extra={"job_id":"-", "broker_id":"-", "employer_id":"-"})
        raise

    genai.configure(api_key=settings.GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    res = model.count_tokens(markdown_text)
    count = int(res.total_tokens if hasattr(res, "total_tokens") else res.get("total_tokens", 0))
    logger.info("Token counting finished: %s tokens", count, extra={"job_id":"-", "broker_id":"-", "employer_id":"-"})
    return count
