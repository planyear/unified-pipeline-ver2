# app/processing/classification.py
import logging
from ..services import vellum
from ..services.llm import compose_messages_with_document, chat_completion

logger = logging.getLogger("pipeline")
CLASSIFICATION_DEPLOYMENT = "classify-document-and-identify-carrier-loc-and-plan-names-v-1-0-variant-1"

def run_classification(markdown_text: str, cache: bool = True) -> str:
    prompt = vellum.get_prompt(CLASSIFICATION_DEPLOYMENT)
    logger.info("Classification: fetched Vellum deployment", extra={"job_id":"-", "broker_id":"-", "employer_id":"-"})
    messages = compose_messages_with_document(
        template=prompt,
        document_md=markdown_text,
        enable_cache=cache,
        system_text="You are a precise benefits-document extraction assistant.",
    )
    out = chat_completion(messages, return_full=False, log_label="classification")
    logger.info("Classification Finished", extra={"job_id":"-", "broker_id":"-", "employer_id":"-"})
    return out
