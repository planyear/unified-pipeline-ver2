import logging
from ..services import vellum, llm

logger = logging.getLogger("pipeline")

CLASSIFICATION_DEPLOYMENT = "classify-document-and-identify-carrier-loc-and-plan-names-v-1-0-variant-1"

def run_classification(markdown_text: str, cache: bool = True) -> str:
    prompt = vellum.get_prompt(CLASSIFICATION_DEPLOYMENT)
    logger.info("Classification: fetched Vellum deployment", extra={"job_id":"-", "broker_id":"-", "employer_id":"-"})
    out = llm.run_prompt_with_context(markdown_text, prompt, cache=cache)
    logger.info("Classification Finished", extra={"job_id":"-", "broker_id":"-", "employer_id":"-"})
    return out
