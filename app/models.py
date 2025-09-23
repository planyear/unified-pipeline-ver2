# app/models.py
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class ProcessingOption(str, Enum):
    AUTO_READ = "Auto-Read"
    SEARCH = "Search"
    ALL_PLANS = "All Plans"

class ProcessRequest(BaseModel):
    job_id: str
    broker_id: str
    employer_id: str
    option: ProcessingOption
    plan_name: Optional[str] = ""

class PlanResult(BaseModel):
    loc: str
    plan_name: str
    output: str

class ProcessResponse(BaseModel):
    job_id: str
    broker_id: str
    employer_id: str
    message: str
    classification_output: str = ""
    kp_extract_output: str = ""
    plan_name_identification_output: str = ""
    plans: List[PlanResult] = []
