# app/processing/key_params.py
import logging
from typing import Dict
from ..services import vellum
from ..services.llm import compose_messages_with_document, chat_completion

logger = logging.getLogger("pipeline")

KEY_PARAMETER_PROMPT_MAP: Dict[str, str] = {
    "Medical": "key-parameter-extraction-prompt-medical-v-12-0-variant-1",
    "Dental": "key-parameter-extraction-prompt-dental-v-9-0",
    "Vision": "key-parameter-extraction-prompt-vision-v-5-0-variant-1",
    "LifeADD": "key-parameter-extraction-prompt-life-and-or-add-v-6-0-variant-1",
    "STD": "key-parameter-extraction-prompt-std-v-5-0-variant-1",
    "LTD": "key-parameter-extraction-prompt-ltd-v-5-0-variant-1",
    "VL": "key-parameter-extraction-prompt-vol-life-add-or-life-add-v-2-0-variant-1",
    "VSTD": "key-parameter-extraction-prompt-vol-std-v-2-0-variant-1",
    "VLTD": "key-parameter-extraction-prompt-vol-ltd-v-2-0-variant-1",
    "VA": "key-parameter-extraction-prompt-vol-accident-v-2-0-variant-1",
    "VHI": "key-parameter-extraction-prompt-vol-hospital-indemnity-v-2-0-variant-1",
    "VCI": "key-parameter-extraction-prompt-vol-critical-illness-v-2-0-variant-1",
}

def run_key_param_extractor(markdown_text: str, loc: str, cache: bool = True) -> str:
    slug = KEY_PARAMETER_PROMPT_MAP.get(loc)
    if not slug:
        logger.warning("Key Params: no Vellum deployment for LOC=%s; skipping", loc, extra={"job_id":"-", "broker_id":"-", "employer_id":"-"})
        return ""
    prompt = vellum.get_prompt(slug)
    logger.info("Key Params: fetched Vellum deployment (%s) for LOC=%s", slug, loc, extra={"job_id":"-", "broker_id":"-", "employer_id":"-"})
    messages = compose_messages_with_document(
        template=prompt,
        document_md=markdown_text,
        enable_cache=cache,
        system_text="You are a precise benefits-document extraction assistant.",
    )
    out = chat_completion(messages, return_full=False, log_label=f"key_params::{loc}")
    logger.info("Key Params Finished for LOC=%s", loc, extra={"job_id":"-", "broker_id":"-", "employer_id":"-"})
    return out
