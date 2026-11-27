"""Deterministic fake model adapters (risk-assessment.md §4). No weights, no network."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from crowdguard.models.base import BaseAdapter, FeatureSequence, InferenceError
from crowdguard.models.orchestrator import ProjectionProducer
from crowdguard_contracts.assessment import AssessmentStatus
from crowdguard_contracts.evidence import ModelDescriptor, Sampling
from crowdguard_contracts.features import FeatureSchema

COUNTS_SCHEMA = FeatureSchema(
    id="zone_window_counts",
    version="1.0.0",
    columns=["people_count_avg", "people_count_end", "image_density_proxy_per_10000_px", "mean_speed_px_per_s"],
)
COUNTS_PRODUCER = ProjectionProducer(COUNTS_SCHEMA)

SEVERITY_LABELS = ["LOW", "MEDIUM", "HIGH"]


def _descriptor(model_id: str, label_space: str, labels: List[str], *, eligible: bool, seq_len: int = 2,
                schema: FeatureSchema = COUNTS_SCHEMA, status=AssessmentStatus.DEMONSTRATION) -> ModelDescriptor:
    return ModelDescriptor(
        model_id=model_id,
        model_version="fixture-1",
        task="venue_risk_3class" if label_space == "venue_risk_v1" else "binary_anomaly",
        label_space=label_space,
        class_labels=labels,
        feature_schema=schema,
        sequence_length=seq_len,
        sampling=Sampling(unit="window", step=1.0),
        scope="zone",
        validation_status=status,
        eligible_for_severity=eligible,
    )


class FakeVenueRiskAdapter(BaseAdapter):
    """Deterministic 3-class model: each zone gets a fixed level (default LOW)."""

    def __init__(self, levels_by_zone: Dict[str, str], model_id: str = "fake-venue-risk", seq_len: int = 2,
                 per_window: Optional[Dict[tuple, str]] = None):
        super().__init__(_descriptor(model_id, "venue_risk_v1", SEVERITY_LABELS, eligible=True, seq_len=seq_len))
        self.levels_by_zone = levels_by_zone
        self.per_window = per_window or {}

    def _infer(self, seq: FeatureSequence) -> Dict[str, float]:
        level = self.per_window.get((seq.zone_id, seq.target_window.index), self.levels_by_zone.get(seq.zone_id, "LOW"))
        return {label: (0.8 if label == level else 0.1) for label in SEVERITY_LABELS}


class FakeBinaryAnomalyAdapter(BaseAdapter):
    """Stands in for the UCSD GRU/LSTM: binary anomaly label space, never eligible for severity."""

    def __init__(self):
        super().__init__(_descriptor("fake-binary-anomaly", "ucsd_binary_anomaly", ["Normal", "Anomalous"],
                                     eligible=False, status=AssessmentStatus.EXPERIMENTAL))

    def _infer(self, seq: FeatureSequence) -> Dict[str, float]:
        return {"Normal": 0.1, "Anomalous": 0.9}  # would be "HIGH" if anyone dared to map it


class RaisingAdapter(FakeVenueRiskAdapter):
    def __init__(self, fail_on_index: int):
        super().__init__({}, model_id="raising")
        self.fail_on_index = fail_on_index

    def _infer(self, seq: FeatureSequence) -> Dict[str, float]:
        if seq.target_window.index == self.fail_on_index:
            raise InferenceError("synthetic failure")
        return super()._infer(seq)


class MissingArtifactAdapter(FakeVenueRiskAdapter):
    def __init__(self, artifact: Path, expected_sha256: str = "0" * 64):
        super().__init__({}, model_id="missing-artifact")
        self.require_artifact(artifact, expected_sha256)


class SchemaMismatchAdapter(FakeVenueRiskAdapter):
    """Declares the counts schema id/version but with columns in a different order."""

    def __init__(self):
        super().__init__({}, model_id="schema-mismatch")
        reordered = FeatureSchema(id=COUNTS_SCHEMA.id, version=COUNTS_SCHEMA.version,
                                  columns=list(reversed(COUNTS_SCHEMA.columns)))
        self._descriptor = self._descriptor.model_copy(update={"feature_schema": reordered})


class UnknownSchemaAdapter(FakeVenueRiskAdapter):
    def __init__(self):
        super().__init__({}, model_id="unknown-schema")
        other = FeatureSchema(id="ucsd_flow_features", version="1.0.0", columns=["crowd_density", "motion_magnitude"])
        self._descriptor = self._descriptor.model_copy(update={"feature_schema": other})
