# app/processing/key_params.py
import logging
from typing import Dict, List, Optional

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

def run_key_param_extractor(
    markdown_text: str,
    loc: str,
    *,
    plan_names_for_loc: Optional[List[str]] = None,   # <-- NEW
    cache: bool = True
) -> str:
    """
    Run the Key Parameter extraction for a given LOC.

    If the Vellum template contains <Plan_Name_List></Plan_Name_List> (or the spaced variant),
    we inject a newline-separated list of plan names for this LOC.
    """
    slug = KEY_PARAMETER_PROMPT_MAP.get(loc)
    if not slug:
        logger.warning(
            "Key Params: no Vellum deployment for LOC=%s; skipping",
            loc,
            extra={"job_id": "-", "broker_id": "-", "employer_id": "-"},
        )
        return ""

    prompt = vellum.get_prompt(slug)
    logger.info(
        "Key Params: fetched Vellum deployment (%s) for LOC=%s",
        slug, loc,
        extra={"job_id": "-", "broker_id": "-", "employer_id": "-"},
    )

    # Inject plan list if the template expects it and we were given names
    if plan_names_for_loc:
        plan_list_text = "\n".join(sorted({p.strip() for p in plan_names_for_loc if p and p.strip()}))
        if "<Plan_Name_List></Plan_Name_List>" in prompt:
            prompt = prompt.replace(
                "<Plan_Name_List></Plan_Name_List>",
                f"<Plan_Name_List>{plan_list_text}</Plan_Name_List>",
                1,
            )
        if "<Plan Name List></Plan Name List>" in prompt:
            prompt = prompt.replace(
                "<Plan Name List></Plan Name List>",
                f"<Plan Name List>{plan_list_text}</Plan Name List>",
                1,
            )

    messages = compose_messages_with_document(
        template=prompt,
        document_md=markdown_text,
        enable_cache=cache,
        system_text="You are a precise benefits-document extraction assistant.",
    )

    out = chat_completion(
        messages,
        return_full=False,
        log_label=f"key_params::{loc}",
    )
    logger.info(
        "Key Params Finished for LOC=%s",
        loc,
        extra={"job_id": "-", "broker_id": "-", "employer_id": "-"},
    )
    return out
