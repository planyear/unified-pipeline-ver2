import re
from typing import List, Dict, Tuple

__all__ = ["parse_line_of_coverage", "parse_plan_listing"]

# Accept "Line of Coverage" or "LOC_Temp"
LOC_RE = re.compile(r"Basic Info::(?:Line of Coverage|LOC_Temp)::\[(.*?)\]", re.IGNORECASE | re.DOTALL)

def parse_line_of_coverage(text: str) -> List[str]:
    if not text:
        return []
    m = LOC_RE.search(text)
    if not m:
        return []
    raw = m.group(1)
    parts = [p.strip() for p in raw.split(",")]
    # filter out placeholders like "Other"
    return [p for p in parts if p and p.lower() != "other"]

# Plan listing lines look like:
#  1::Plans::Medical::Blue Shield HMO 20::$511.00::[1, 29, 49]
# Be flexible: rate is optional; pages block is optional; allow spaces.
PLAN_RE = re.compile(
    r"""
    (?P<idx>\d+)\s*::\s*Plans\s*::\s*
    (?P<loc>[^:]+?)\s*::\s*
    (?P<plan>[^:]+?)\s*::\s*
    (?P<rate>\$?[0-9][^:\[]*)?      # optional rate up to next '::' or '['
    (?:\s*::\s*\[(?P<pages>[^\]]*)\])?   # optional [1, 2, 3]
    """,
    re.IGNORECASE | re.VERBOSE,
)

def parse_plan_listing(text: str) -> Dict[str, List[Tuple[str, List[int]]]]:
    out: Dict[str, List[Tuple[str, List[int]]]] = {}
    if not text:
        return out
    for m in PLAN_RE.finditer(text):
        loc = (m.group("loc") or "").strip()
        plan = (m.group("plan") or "").strip()
        pages_raw = (m.group("pages") or "")
        pages = [int(n) for n in re.findall(r"\d+", pages_raw)]
        if loc and plan:
            out.setdefault(loc, []).append((plan, pages))
    return out
