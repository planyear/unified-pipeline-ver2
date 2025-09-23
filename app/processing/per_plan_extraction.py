# app/processing/per_plan_extraction.py
import logging
from typing import Dict
from ..services import vellum
from ..services.llm import compose_messages_with_document, chat_completion

logger = logging.getLogger("pipeline")

PROMPT_MAP: Dict[str, str] = {
    "Medical": "medical-unified-refiner-v-5-0-variant-1",
    "Dental": "dental-unified-refiner-v-3-0-variant-1",
    "Vision": "vision-unified-refiner-v-3-0-variant-1",
    "LifeADD": "vol-life-and-add-generic-unified-refiner-reducto-v-7-0-variant-1",
    "STD": "std-unified-refiner-v-3-0-variant-1",
    "LTD": "ltd-unified-refiner-v-3-0-variant-1",
    "VL": "vol-life-and-add-generic-unified-refiner-reducto-v-7-0-variant-1",
    "VSTD": "vol-std-generic-unified-refiner-reducto-v-8-0-variant-1",
    "VLTD": "vol-ltd-generic-unified-refiner-reducto-v-8-0-variant-1",
    "VA": "vol-accident-generic-unified-refiner-reducto-v-7-0-variant-1",
    "VHI": "vol-hospital-indemnity-generic-unified-refiner-reducto-v-7-0-variant-1",
    "VCI": "vol-critical-illness-generic-unified-refiner-reducto-v-7-0-variant-1",
}

def run_per_plan_extraction(markdown_text: str, loc: str, plan_name: str, cache: bool = True) -> str:
    slug = PROMPT_MAP.get(loc, PROMPT_MAP["Medical"])
    prompt = vellum.get_prompt(slug)
    logger.info("Per-plan: fetched Vellum deployment (%s) for LOC=%s", slug, loc, extra={"job_id":"-", "broker_id":"-", "employer_id":"-"})

    # inject plan name ONLY; never touch the doc
    for token in ("{{plan_name}}", "{{PLAN_NAME}}", "<PlanName></PlanName>", "<Plan Name></Plan Name>"):
        if token in prompt:
            prompt = prompt.replace(token, plan_name if token.startswith("{{") else f"<PlanName>{plan_name}</PlanName>")
            break

    messages = compose_messages_with_document(
        template=prompt,
        document_md=markdown_text,
        enable_cache=cache,
        system_text="You are a precise, deterministic parser. Return ONLY the extraction text for this plan. No extra commentary.",
    )
    out = chat_completion(messages, return_full=False, log_label=f"per_plan::{loc}::{plan_name}")
    logger.info("Per-plan extraction finished for %s / %s", loc, plan_name, extra={"job_id":"-", "broker_id":"-", "employer_id":"-"})
    return out
