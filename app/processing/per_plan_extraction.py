import logging
from typing import Dict
from ..services import vellum, llm

logger = logging.getLogger("pipeline")

PROMPT_MAP: Dict[str, str] = {
    "Medical": "medical-unified-refiner-v-5-0-variant-1",
    "Dental": "dental-unified-refiner-v-3-0-variant-1",
    "Vision": "vision-unified-refiner-v-3-0-variant-1",
    "LifeADD": "life-add-life-add-unified-refiner-v-2-0-variant-1",
    "STD": "std-unified-refiner-v-3-0-variant-1",
    "LTD": "ltd-unified-refiner-v-3-0-variant-1",
    "VL": "vol-life-and-add-generic-unified-refiner-reducto-v-7-0-variant-1",
    "VSTD": "vol-std-generic-unified-refiner-reducto-v-8-0-variant-1",
    "VLTD": "vol-ltd-generic-unified-refiner-reducto-v-8-0-variant-1",
    "VA": "vol-accident-generic-unified-refiner-reducto-v-7-0-variant-1",
    "VHI": "vol-hospital-indemnity-generic-unified-refiner-reducto-v-7-0-variant-1",
    "VCI": "vol-critical-illness-generic-unified-refiner-reducto-v-7-0-variant-1",
}

def _inject_tag_once(prompt: str, open_tag: str, close_tag: str, content: str) -> str:
    sentinel = f"{open_tag}{close_tag}"
    if sentinel in prompt:
        return prompt.replace(sentinel, f"{open_tag}{content}{close_tag}", 1)
    return prompt

def run_per_plan_extraction(markdown_text: str, loc: str, plan_name: str, cache: bool = True) -> str:
    """
    Fetch the correct per-LOC Vellum deployment and run it with the full document.
    Inject the plan name if the template expects {{plan_name}} (or common variants).
    """
    slug = PROMPT_MAP.get(loc, PROMPT_MAP["Medical"])
    prompt = vellum.get_prompt(slug)
    logger.info("Per-plan: fetched Vellum deployment (%s) for LOC=%s", slug, loc, extra={"job_id":"-", "broker_id":"-", "employer_id":"-"})

    used_prompt = _inject_tag_once(prompt, "<Document>", "</Document>", markdown_text)

    replacements = ["{{plan_name}}", "{{PLAN_NAME}}", "<PlanName></PlanName>", "<Plan Name></Plan Name>"]
    for token in replacements:
        if token in used_prompt:
            if token.startswith("{{"):
                used_prompt = used_prompt.replace(token, plan_name)
            else:
                used_prompt = used_prompt.replace(token, f"<PlanName>{plan_name}</PlanName>")
            break

    system = (
        "You are a precise, deterministic parser. Return ONLY the extraction text for this plan. "
        "No explanations, no extra commentary."
    )
    out = llm.run_prompt_with_context(markdown_text, used_prompt, cache=cache)
    logger.info("Per-plan extraction finished for %s / %s", loc, plan_name, extra={"job_id":"-", "broker_id":"-", "employer_id":"-"})
    return out
