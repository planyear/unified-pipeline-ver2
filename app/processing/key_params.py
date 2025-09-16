import logging
from typing import Dict
from ..services import vellum, llm

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
    # Normalize keys (e.g., 'Life' vs 'LifeADD'); prefer exact keys present in map
    slug = KEY_PARAMETER_PROMPT_MAP.get(loc)
    if not slug:
        logger.warning("Key Params: no Vellum deployment for LOC=%s; skipping", loc, extra={"job_id":"-", "broker_id":"-", "employer_id":"-"})
        return ""
    prompt = vellum.get_prompt(slug)
    logger.info("Key Params: fetched Vellum deployment (%s)", slug, extra={"job_id":"-", "broker_id":"-", "employer_id":"-"})
    out = llm.run_prompt_with_context(markdown_text, prompt, cache=cache)
    logger.info("Key Params Finished for LOC=%s", loc, extra={"job_id":"-", "broker_id":"-", "employer_id":"-"})
    return out
