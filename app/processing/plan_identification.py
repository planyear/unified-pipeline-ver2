# app/processing/plan_identification.py
import logging
from typing import List, Dict, Optional
from ..services import vellum, llm
from ..utils.parse import parse_line_of_coverage  # keep

logger = logging.getLogger("pipeline")

PLAN_IDENT_DEPLOYMENT = "plan-name-identification-prompt-v-18-0-variant-1"

def _inject(tag_open: str, tag_close: str, doc: str, content: str) -> str:
    token = f"{tag_open}{tag_close}"
    if token in doc:
        return doc.replace(token, f"{tag_open}{content}{tag_close}", 1)
    return doc

def run_plan_identification(
    markdown_text: str,
    classification_output: str,
    step7_output: str,
    locs_for_prompt: Optional[List[str]] = None,
    cache: bool = True
) -> str:
    """
    If locs_for_prompt is provided, we use those LOC(s) in {{LOC}}; otherwise we
    derive from the classification output.
    """
    prompt = vellum.get_prompt(PLAN_IDENT_DEPLOYMENT)

    used = _inject("<Document>", "</Document>", prompt, markdown_text)
    used = _inject("<Classification Output>", "</Classification Output>", used, classification_output)

    if "<Key Parameter Output></Key Parameter Output>" in used:
        used = used.replace(
            "<Key Parameter Output></Key Parameter Output>",
            f"<Key Parameter Output>{step7_output}</Key Parameter Output>",
            1,
        )

    # choose LOCs
    locs = locs_for_prompt if locs_for_prompt is not None else parse_line_of_coverage(classification_output)
    used = used.replace("{{LOC}}", ", ".join(locs))

    system = "You are a precise, deterministic parser. Output only <index>::Plans::<LOC>::<Plan Name>::$0.00::<page_ref> lines."
    out = llm.run_prompt_with_context(markdown_text, used, cache=cache)
    logger.info("Plan Identification Finished", extra={"job_id":"-", "broker_id":"-", "employer_id":"-"})
    return out
