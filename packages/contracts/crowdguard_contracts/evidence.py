"""Model adapter descriptor and per-window model evidence (risk-assessment.md §4)."""

from __future__ import annotations

import math
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .assessment import AssessmentStatus
from .features import DataQuality, FeatureSchema, Window


class InferenceStatus(str, Enum):
    OK = "OK"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    ERROR = "ERROR"
    SKIPPED_INELIGIBLE = "SKIPPED_INELIGIBLE"


class Sampling(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    unit: Literal["window", "frame"]
    step: float = Field(gt=0)


class ModelDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())
    model_id: str
    model_version: str
    artifact_sha256: Optional[str] = None
    scaler_sha256: Optional[str] = None
    task: str
    label_space: str
    class_labels: List[str] = Field(min_length=2)
    feature_schema: FeatureSchema
    sequence_length: int = Field(ge=1)
    sampling: Sampling
    scope: Literal["frame", "zone"]
    calibration_status: str = "UNCALIBRATED"
    validation_status: AssessmentStatus = AssessmentStatus.EXPERIMENTAL
    eligible_for_severity: bool = False

    @model_validator(mode="after")
    def _unique_labels(self) -> "ModelDescriptor":
        if len(set(self.class_labels)) != len(self.class_labels):
            raise ValueError("class_labels must be unique")
        return self

    @property
    def ref(self) -> str:
        return f"{self.model_id}@{self.model_version}"


class EvidenceInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    feature_schema: str
    rows_used: int = Field(ge=0)
    first_row_s: Optional[float] = None
    last_row_s: Optional[float] = None


class ModelEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())
    evidence_id: str
    model_id: str
    model_version: str
    label_space: str
    class_labels: List[str]
    camera_id: str
    zone_id: str
    window: Window
    input: Optional[EvidenceInput] = None
    inference_status: InferenceStatus
    probabilities: Optional[Dict[str, float]] = None
    data_quality: DataQuality
    calibration_status: str
    validation_status: AssessmentStatus
    reason: Optional[str] = None
    latency_ms: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _probabilities_consistent(self) -> "ModelEvidence":
        if self.inference_status is InferenceStatus.OK:
            if self.probabilities is None:
                raise ValueError("OK evidence requires probabilities")
            if set(self.probabilities) != set(self.class_labels):
                raise ValueError("probability keys must equal class_labels")
            values = list(self.probabilities.values())
            if any((not math.isfinite(p)) or p < 0 or p > 1 for p in values):
                raise ValueError("probabilities must be finite and within [0, 1]")
            if abs(sum(values) - 1.0) > 1e-3:
                raise ValueError("probabilities must sum to 1")
        elif self.probabilities is not None:
            raise ValueError("probabilities must be null unless inference_status is OK")
        return self
