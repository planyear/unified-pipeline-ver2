import logging, os
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import settings
from ..utils.files import get_ext
from ..services import cloudconvert, reducto, token_counter
from ..processing import classification as clf
from ..processing import key_params as kp
from ..processing import per_plan_extraction as ppe
from ..processing import plan_identification as pid
from ..services import deterministic
from ..utils.parse import parse_line_of_coverage, parse_plan_listing

logger = logging.getLogger("pipeline")

def _maybe_pdf(input_path: str) -> str:
    ext = get_ext(input_path)
    if ext == "pdf":
        return input_path
    if ext in {"doc", "docx", "xls", "xlsx", "xlsm"}:
        return cloudconvert.convert_to_pdf(settings.CLOUDCONVERT_API_KEY, input_path)
    raise ValueError(f"Unsupported file type: .{ext}")

def semantic_match_plan_name(plan_names: List[str], plan_name: str) -> str:
    # normalize dashes & spaces for both sides before asking the LLM
    def norm(s: str) -> str:
        return (s or "").replace("–", "-").replace("—", "-").strip()
    plan_names_norm = [norm(p) for p in plan_names]
    plan_name_norm = norm(plan_name)

    prompt = f"""
Given this list {plan_names_norm}.

Question: Is {plan_name_norm} in the list?

Guidelines to answer Question:
- Search for semantic meaning (treat punctuation/dashes/extra spaces as equivalent).
- If it is in the list, return exactly the matching plan from the list.
- Otherwise return "NA".
Your response must be a single string: either the plan name from the list or "NA".
""".strip()

    from ..services import llm
    out = llm.run_prompt_with_context("\n".join(plan_names_norm), prompt).strip().strip('"').strip("'")
    return out

def run_pipeline(
    *,
    input_path: str,
    job_id: str,
    broker_id: str,
    employer_id: str,
    option: str,
    search_plan_name: str = "",
    prompt_cache: bool = True,
) -> Dict:
    base_extra = {"job_id": job_id, "broker_id": broker_id, "employer_id": employer_id}
    logger.info("Pipeline: Start (prompt_cache=%s)", prompt_cache, extra=base_extra)

    logger.info("Pipeline: Start", extra=base_extra)
    pdf_path = _maybe_pdf(input_path)
    logger.info("File normalization finished (PDF ready)", extra=base_extra)

    markdown = reducto.pdf_to_markdown(pdf_path)
    token_count = token_counter.count_tokens_google(markdown)
    if token_count > settings.TOKEN_HARD_LIMIT:
        logger.warning("Token limit exceeded: %s", token_count, extra=base_extra)
        return {
            "job_id": job_id,
            "broker_id": broker_id,
            "employer_id": employer_id,
            "message": "Document exceeds 50,000 tokens. Pipeline stopped processing. Please try again.",
            "classification_output": "",
            "kp_extract_output": "",
            "plan_name_identification_output": "",
            "plans": [],
        }

    # Step 6: classification
    classification_out = clf.run_classification(markdown)

    # Special SBC handling
    if "Basic Info::Document Type::SBC" in classification_out:
        logger.info("SBC document detected; running SBC-only handling", extra=base_extra)
        return {
            "job_id": job_id,
            "broker_id": broker_id,
            "employer_id": employer_id,
            "message": "SBC document processed (SBC path).",
            "classification_output": classification_out,
            "kp_extract_output": "",
            "plan_name_identification_output": "",
            "plans": [],
        }

    # Step 7: LOCs + plan listing
    locs = parse_line_of_coverage(classification_out)
    plan_listing = parse_plan_listing(classification_out)
    logger.info("LOCs parsed: %s", locs, extra=base_extra)

    # Option narrowing (build a working set of {loc: [plans...]})
    selected: Dict[str, List[Tuple[str, List[int]]]] = {}
    need_pid_search = False            # <— NEW
    initial_search_match = None        # <— NEW

    if option == "Auto-Read":
        for loc in locs:
            plans = plan_listing.get(loc, [])
            selected[loc] = plans[:4]

    elif option == "Search":
        # flatten all plan names from CLASSIFIER listing
        all_plans = []
        for loc in locs:
            for (p, _pg) in plan_listing.get(loc, []):
                all_plans.append(p)

        initial_search_match = semantic_match_plan_name(all_plans, search_plan_name) if all_plans else "NA"
        logger.info("Search (classifier list) match: %s", initial_search_match, extra=base_extra)

        if initial_search_match != "NA":
            # find loc/pages for this plan from classifier listing
            for loc in locs:
                for (p, pages) in plan_listing.get(loc, []):
                    if p.strip() == initial_search_match.strip():
                        selected[loc] = [(p, pages)]
                        break
                if loc in selected:
                    break
        else:
            # defer, we'll try again after Plan Identification
            need_pid_search = True

    elif option == "All Plans":
        for loc in locs:
            selected[loc] = plan_listing.get(loc, [])

    else:
        raise ValueError(f"Unknown option: {option}")

    logger.info("Selected plans per LOC prepared", extra=base_extra)

    # Key Parameter Extraction per LOC
    key_param_outputs: Dict[str, str] = {}
    for loc in locs:
        key_param_outputs[loc] = kp.run_key_param_extractor(markdown, loc, cache=prompt_cache)

    step7_joined = "\n\n".join(
        [key_param_outputs[loc] for loc in selected.keys() if key_param_outputs[loc]]
    )

    # --- Plan Identification (per-LOC for Auto-Read; single run otherwise) ---
    plan_id_output = ""      # for API response
    pid_listing: Dict[str, List[Tuple[str, List[int]]]] = {}

    if option == "Auto-Read":
        # Run PID once per LOC so each LOC yields its own plans
        pid_outputs = []
        for loc in locs:  # 'locs' is from parse_line_of_coverage(classification_out)
            step7_loc = key_param_outputs.get(loc, "")
            out_loc = pid.run_plan_identification(
                markdown, classification_out, step7_loc, [loc], cache=prompt_cache  # <-- pass [loc]
            )
            pid_outputs.append(f"### {loc}\n{out_loc}")

            # merge listing for this LOC
            listing_loc = parse_plan_listing(out_loc)
            for L, items in listing_loc.items():
                pid_listing.setdefault(L, []).extend(items)

        plan_id_output = "\n\n".join(pid_outputs)

    else:
        # Search / All Plans: keep single-shot behavior
        plan_id_output = pid.run_plan_identification(
            markdown, classification_out, step7_joined, cache=prompt_cache
        )
        pid_listing = parse_plan_listing(plan_id_output)

    if option == "Auto-Read":

        all_locs = set(locs) | set(pid_listing.keys())

        new_selected: Dict[str, List[Tuple[str, List[int]]]] = {}
        for loc in sorted(all_locs):
            combined: List[Tuple[str, List[int]]] = []
            seen = set()

            for (p, pages) in (plan_listing.get(loc, []) + pid_listing.get(loc, [])):
                key = p.strip()
                if key in seen:
                    continue
                seen.add(key)
                combined.append((p, pages))
                if len(combined) >= 4:
                    break

            new_selected[loc] = combined

        selected = new_selected
        logger.info(
            "Auto-Read: merged classifier + plan-id, taking first 4 per LOC. LOCs=%s",
            list(selected.keys()), extra=base_extra
        )
    
    if option == "Search" and need_pid_search:
        pid_listing = parse_plan_listing(plan_id_output)
        all_plans_pid = []
        for loc, items in pid_listing.items():
            for (p, _pg) in items:
                all_plans_pid.append(p)

        retry_match = semantic_match_plan_name(all_plans_pid, search_plan_name) if all_plans_pid else "NA"
        logger.info("Search (plan-ident list) match: %s", retry_match, extra=base_extra)

        if retry_match == "NA":
            return {
                "job_id": job_id,
                "broker_id": broker_id,
                "employer_id": employer_id,
                "message": f'Plan "{search_plan_name}" not found.',
                "classification_output": classification_out,
                "kp_extract_output": step7_joined,
                "plan_name_identification_output": plan_id_output,
                "plans": [],
            }

        selected = {}
        for loc, items in pid_listing.items():
            for (p, pages) in items:
                if p.strip() == retry_match.strip():
                    selected[loc] = [(p, pages)]
                    break
            if loc in selected:
                break

    def _count_selected(sel: Dict[str, List[Tuple[str, List[int]]]]) -> int:
        return sum(len(v) for v in sel.values())


    if _count_selected(selected) == 0:
        pid_listing = parse_plan_listing(plan_id_output)
        pid_count = sum(len(v) for v in pid_listing.values())
        if pid_count > 0:
            logger.info(
                "No plans from classification; rebuilding selection from Plan Identification output (%d plans).",
                pid_count,
                extra=base_extra,
            )
            selected = {}
            locs_pid = list(pid_listing.keys())
            if option == "Auto-Read":
                for loc in locs_pid:
                    selected[loc] = pid_listing.get(loc, [])[:4]
            elif option == "Search":
                all_plans_pid = []
                for loc in locs_pid:
                    for p, _pg in pid_listing.get(loc, []):
                        all_plans_pid.append(p)
                match = semantic_match_plan_name(all_plans_pid, search_plan_name)
                logger.info(
                    "Search match (from Plan Identification list): %s",
                    match,
                    extra=base_extra,
                )
                if match == "NA":
                    return {
                        "job_id": job_id,
                        "broker_id": broker_id,
                        "employer_id": employer_id,
                        "message": f'Plan "{search_plan_name}" not found.',
                        "classification_output": classification_out,
                        "kp_extract_output": step7_joined,
                        "plan_name_identification_output": plan_id_output,
                        "plans": [],
                    }
                for loc in locs_pid:
                    for p, pages in pid_listing.get(loc, []):
                        if p.strip() == match.strip():
                            selected[loc] = [(p, pages)]
                            break
                    if loc in selected:
                        break
            elif option == "All Plans":
                for loc in locs_pid:
                    selected[loc] = pid_listing.get(loc, [])

    # Fan out per-plan extraction
    results = []

    def process_one(loc: str, plan_name: str, pages: List[int]):
        # Use the full markdown and per-LOC deployment, inject plan_name, return raw LLM output
        raw_output = ppe.run_per_plan_extraction(markdown, loc, plan_name, cache=prompt_cache)
        return {"loc": loc, "plan_name": plan_name, "output": raw_output}

    total_plans = sum(len(v) for v in selected.values())
    logger.info("Total plans to process: %d", total_plans, extra=base_extra)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = []
        for loc, plans in selected.items():
            for (plan, pages) in plans:
                futs.append(ex.submit(process_one, loc, plan, pages))
        for fut in as_completed(futs):
            results.append(fut.result())

    logger.info("Pipeline: Finished", extra=base_extra)
    return {
        "job_id": job_id,
        "broker_id": broker_id,
        "employer_id": employer_id,
        "message": "OK",
        "classification_output": classification_out,
        "kp_extract_output": step7_joined,
        "plan_name_identification_output": plan_id_output,
        "plans": results,
    }
