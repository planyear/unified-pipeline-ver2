# app/processing/plan_identification.py
import logging
from typing import List, Optional
from ..services import vellum
from ..services.llm import compose_messages_with_document, chat_completion
from ..utils.parse import parse_line_of_coverage

logger = logging.getLogger("pipeline")

PLAN_IDENT_DEPLOYMENT = "plan-name-identification-prompt-v-18-0-variant-1"

def run_plan_identification(
    markdown_text: str,
    classification_output: str,
    step7_output: str,
    locs_for_prompt: Optional[List[str]] = None,
    cache: bool = True,
) -> str:
    """
    Build a cache-friendly message:
      - Small dynamic text (classification, key params) stays in the template
      - Big document is attached as its own cached content part
    """
    prompt = vellum.get_prompt(PLAN_IDENT_DEPLOYMENT)

    # inline small dynamic blocks
    if "<Classification Output></Classification Output>" in prompt:
        prompt = prompt.replace(
            "<Classification Output></Classification Output>",
            f"<Classification Output>{classification_output}</Classification Output>",
            1,
        )
    if "<Key Parameter Output></Key Parameter Output>" in prompt:
        prompt = prompt.replace(
            "<Key Parameter Output></Key Parameter Output>",
            f"<Key Parameter Output>{step7_output}</Key Parameter Output>",
            1,
        )

    # LOCs for {{LOC}}
    locs = locs_for_prompt if locs_for_prompt is not None else parse_line_of_coverage(classification_output)
    prompt = prompt.replace("{{LOC}}", ", ".join(locs))

    messages = compose_messages_with_document(
        template=prompt,
        document_md=markdown_text,
        enable_cache=cache,
        system_text="You are a precise, deterministic parser. Output only <index>::Plans::<LOC>::<Plan Name>::$0.00::<page_ref> lines.",
    )

    out = chat_completion(messages, temperature=0.0, return_full=False, log_label="plan_identification")
    logger.info("Plan Identification Finished", extra={"job_id": "-", "broker_id": "-", "employer_id": "-"})
    return out
