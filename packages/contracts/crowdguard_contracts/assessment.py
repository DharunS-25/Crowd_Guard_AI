"""Risk levels, assessment status, reason codes, and the fused zone assessment."""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


SEVERITY_LEVELS = (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH)


class AssessmentStatus(str, Enum):
    DEMONSTRATION = "DEMONSTRATION"
    EXPERIMENTAL = "EXPERIMENTAL"
    VALIDATED_FOR_PILOT = "VALIDATED_FOR_PILOT"


# Least to most validated; a fused result takes the least validated contributing input.
ASSESSMENT_STATUS_ORDER = [
    AssessmentStatus.DEMONSTRATION,
    AssessmentStatus.EXPERIMENTAL,
    AssessmentStatus.VALIDATED_FOR_PILOT,
]


def least_validated(*statuses: AssessmentStatus) -> AssessmentStatus:
    return min(statuses, key=ASSESSMENT_STATUS_ORDER.index)


class ReasonCode(str, Enum):
    WARMUP_INCOMPLETE = "WARMUP_INCOMPLETE"
    SEQUENCE_GAP = "SEQUENCE_GAP"
    FEATURE_INVALID = "FEATURE_INVALID"
    FEATURE_STALE = "FEATURE_STALE"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_ERROR = "MODEL_ERROR"
    NO_FEATURE_PRODUCER_FOR_SCHEMA = "NO_FEATURE_PRODUCER_FOR_SCHEMA"
    NO_ELIGIBLE_SEVERITY_EVIDENCE = "NO_ELIGIBLE_SEVERITY_EVIDENCE"
    POLICY_THRESHOLDS_UNCONFIGURED = "POLICY_THRESHOLDS_UNCONFIGURED"
    MODEL_DISAGREEMENT = "MODEL_DISAGREEMENT"
    NO_WINDOW = "NO_WINDOW"
    POLICY_RULE_MATCHED = "POLICY_RULE_MATCHED"
    MODEL_EVIDENCE_SELECTED = "MODEL_EVIDENCE_SELECTED"


class WarningCode(str, Enum):
    CAPACITY_UNCONFIGURED = "CAPACITY_UNCONFIGURED"
    FPS_FALLBACK_USED = "FPS_FALLBACK_USED"
    NO_DETECTIONS = "NO_DETECTIONS"
    FRAMES_DROPPED = "FRAMES_DROPPED"
    ZONE_CROP_UNVALIDATED = "ZONE_CROP_UNVALIDATED"


class PolicyRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    version: str


class RuleSignal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    signal_id: str
    status: Literal["OK", "NOT_CONFIGURED", "INSUFFICIENT_DATA"]
    value: Optional[float] = None
    finding: Optional[RiskLevel] = None
    threshold_source: Optional[str] = None


class ZoneRiskAssessment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    level: RiskLevel
    assessment_status: AssessmentStatus
    reason_codes: List[ReasonCode] = Field(default_factory=list)
    policy: PolicyRef
    evidence_refs: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _insufficient_needs_reason(self) -> "ZoneRiskAssessment":
        if self.level is RiskLevel.INSUFFICIENT_DATA and not self.reason_codes:
            raise ValueError("INSUFFICIENT_DATA requires at least one reason code")
        return self
