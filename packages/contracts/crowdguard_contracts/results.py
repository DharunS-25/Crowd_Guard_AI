"""Result and artifact contracts (system-architecture.md §7.2-§7.3)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .analysis import SourceMetadata, ZoneConfigRef
from .assessment import AssessmentStatus, PolicyRef, RiskLevel, RuleSignal, ZoneRiskAssessment
from .evidence import ModelDescriptor, ModelEvidence
from .features import DataQuality, FeatureSchema, FeatureValue, Window


class ArtifactName(str, Enum):
    RESULTS_JSON = "results.json"
    FEATURES_CSV = "features.csv"
    ZONE_RISK_CSV = "zone_risk.csv"
    OVERLAY_MANIFEST = "overlay_manifest.json"
    ANNOTATED_VIDEO = "annotated.mp4"
    MANIFEST = "manifest.json"


ARTIFACT_CONTENT_TYPES = {
    ArtifactName.RESULTS_JSON: "application/json",
    ArtifactName.FEATURES_CSV: "text/csv",
    ArtifactName.ZONE_RISK_CSV: "text/csv",
    ArtifactName.OVERLAY_MANIFEST: "application/json",
    ArtifactName.ANNOTATED_VIDEO: "video/mp4",
    ArtifactName.MANIFEST: "application/json",
}


class ArtifactLink(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: ArtifactName
    url: str
    expires_at: datetime


class FeatureBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_id: str
    schema_version: str
    values: Dict[str, FeatureValue]
    units: Dict[str, str] = Field(default_factory=dict)


class ZoneWindowResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analysis_id: str
    run_id: str
    camera_id: str
    zone_id: str
    window: Window
    features: FeatureBlock
    data_quality: DataQuality
    rule_signals: List[RuleSignal] = Field(default_factory=list)
    model_evidence: List[ModelEvidence] = Field(default_factory=list)
    assessment: ZoneRiskAssessment


class PolicyInfo(PolicyRef):
    assessment_status: AssessmentStatus
    sha256: str


class ResultsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "results.v1"
    analysis_id: str
    run_id: str
    source: SourceMetadata
    zone_config: ZoneConfigRef
    feature_schema: FeatureSchema
    models: List[ModelDescriptor]
    policy: PolicyInfo
    windows: List[ZoneWindowResult]
    next_cursor: Optional[str] = None


class OverlayManifestEntry(BaseModel):
    """One contiguous frame range for one zone, as drawn on the annotated video."""

    model_config = ConfigDict(extra="forbid")
    zone_id: str
    frame_start: int = Field(ge=0)
    frame_end: int = Field(ge=0)  # inclusive
    window_index: Optional[int] = None
    level: RiskLevel
    assessment_status: AssessmentStatus
    palette_token: str


class OverlayManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "overlay_manifest.v1"
    analysis_id: str
    run_id: str
    fps: float
    frame_count: int
    entries: List[OverlayManifestEntry]
