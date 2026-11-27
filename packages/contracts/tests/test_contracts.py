import pytest
from pydantic import BaseModel, ValidationError

from crowdguard_contracts import analysis, assessment, evidence, features, results, upload
from crowdguard_contracts.assessment import AssessmentStatus, PolicyRef, ReasonCode, RiskLevel, ZoneRiskAssessment
from crowdguard_contracts.errors import ErrorCode
from crowdguard_contracts.evidence import InferenceStatus, ModelEvidence
from crowdguard_contracts.features import DataQuality, Window

POLICY = PolicyRef(id="p", version="0.1.0")


def _evidence(**overrides):
    base = dict(
        evidence_id="m@v/A/0",
        model_id="m",
        model_version="v",
        label_space="venue_risk_v1",
        class_labels=["LOW", "MEDIUM", "HIGH"],
        camera_id="cam-1",
        zone_id="A",
        window=Window(index=0, start_s=0.0, end_s=1.0),
        inference_status=InferenceStatus.OK,
        probabilities={"LOW": 0.1, "MEDIUM": 0.2, "HIGH": 0.7},
        data_quality=DataQuality(state="OK"),
        calibration_status="UNCALIBRATED",
        validation_status=AssessmentStatus.DEMONSTRATION,
    )
    base.update(overrides)
    return ModelEvidence(**base)


def test_insufficient_data_requires_reason_code():
    with pytest.raises(ValidationError):
        ZoneRiskAssessment(
            level=RiskLevel.INSUFFICIENT_DATA,
            assessment_status=AssessmentStatus.EXPERIMENTAL,
            reason_codes=[],
            policy=POLICY,
        )
    ok = ZoneRiskAssessment(
        level=RiskLevel.INSUFFICIENT_DATA,
        assessment_status=AssessmentStatus.EXPERIMENTAL,
        reason_codes=[ReasonCode.WARMUP_INCOMPLETE],
        policy=POLICY,
    )
    assert ok.level is RiskLevel.INSUFFICIENT_DATA


def test_probabilities_only_when_ok():
    with pytest.raises(ValidationError):
        _evidence(inference_status=InferenceStatus.SCHEMA_MISMATCH)
    with pytest.raises(ValidationError):
        _evidence(probabilities=None)
    failed = _evidence(inference_status=InferenceStatus.MODEL_UNAVAILABLE, probabilities=None)
    assert failed.probabilities is None


def test_probability_keys_must_equal_class_labels_and_sum_to_one():
    with pytest.raises(ValidationError):
        _evidence(probabilities={"Normal": 0.5, "Anomalous": 0.5})
    with pytest.raises(ValidationError):
        _evidence(probabilities={"LOW": 0.5, "MEDIUM": 0.5, "HIGH": 0.5})


def test_window_start_before_end():
    with pytest.raises(ValidationError):
        Window(index=0, start_s=2.0, end_s=2.0)
    with pytest.raises(ValidationError):
        Window(index=-1, start_s=0.0, end_s=1.0)


def test_unknown_enum_values_rejected():
    with pytest.raises(ValidationError):
        ZoneRiskAssessment(level="CRITICAL", assessment_status="EXPERIMENTAL", reason_codes=[], policy=POLICY)
    with pytest.raises(ValueError):
        ErrorCode("NOT_A_CODE")


def test_api_responses_expose_no_storage_internals():
    forbidden = {"key", "object_key", "s3_key", "bucket", "s3_upload_id", "multipart_upload_id", "path"}
    response_models = [
        upload.CreateUploadResponse,
        upload.UploadSessionResponse,
        upload.PartUrlsResponse,
        upload.CompleteUploadResponse,
        analysis.AnalysisStatusResponse,
        results.ArtifactLink,
        results.ResultsResponse,
    ]

    def walk(model: type[BaseModel], seen=None):
        seen = seen or set()
        if model in seen:
            return set()
        seen.add(model)
        names = set(model.model_fields)
        for field in model.model_fields.values():
            for arg in getattr(field.annotation, "__args__", ()) + (field.annotation,):
                inner = getattr(arg, "__args__", (arg,))
                for candidate in inner:
                    if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                        names |= walk(candidate, seen)
        return names

    for model in response_models:
        assert not (walk(model) & forbidden), model.__name__


def test_create_upload_requires_attestation_and_positive_size():
    with pytest.raises(ValidationError):
        upload.CreateUploadRequest(display_name="a.mp4", size_bytes=10, content_type="video/mp4", authorization_attested=False)
    with pytest.raises(ValidationError):
        upload.CreateUploadRequest(display_name="a.mp4", size_bytes=0, content_type="video/mp4", authorization_attested=True)
    req = upload.CreateUploadRequest(display_name="a.mp4", size_bytes=10, content_type="video/mp4", authorization_attested=True)
    assert req.size_bytes == 10


def test_part_checksum_must_be_base64_sha256():
    with pytest.raises(ValidationError):
        upload.PartUrlRequestItem(part_number=1, size_bytes=5, checksum_sha256="not-base64")
    with pytest.raises(ValidationError):
        upload.PartUrlRequestItem(part_number=0, size_bytes=5, checksum_sha256="A" * 43 + "=")
    upload.PartUrlRequestItem(part_number=1, size_bytes=5, checksum_sha256="A" * 43 + "=")


def test_zone_config_validation():
    square = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]
    bowtie = [[0.1, 0.1], [0.4, 0.4], [0.4, 0.1], [0.1, 0.4]]
    analysis.ZoneConfigCreate(name="c", zones=[analysis.ZoneDefinition(zone_id="A", points=square)])
    with pytest.raises(ValidationError):
        analysis.ZoneDefinition(zone_id="A", points=bowtie)
    with pytest.raises(ValidationError):
        analysis.ZoneDefinition(zone_id="A", points=[[0.1, 0.1], [1.2, 0.1], [0.4, 0.4]])
    with pytest.raises(ValidationError):
        analysis.ZoneDefinition(zone_id="A", points=square, capacity=40)  # capacity without a source
    with pytest.raises(ValidationError):
        analysis.ZoneConfigCreate(
            name="c",
            zones=[analysis.ZoneDefinition(zone_id="A", points=square), analysis.ZoneDefinition(zone_id="A", points=square)],
        )
    sourced = analysis.ZoneDefinition(zone_id="A", points=square, capacity=40, capacity_source="Venue plan rev 3 (J. Doe)")
    assert sourced.capacity == 40


def test_zone_window_features_schema_constant():
    schema = features.ZONE_WINDOW_FEATURES_V1
    assert schema.id == "zone_window_features"
    assert schema.columns[0] == "people_count_avg"
    assert "occupancy_ratio" in schema.columns
    assert schema.units["mean_speed_px_per_s"].startswith("px/s")


def test_assessment_and_evidence_modules_export_enums():
    assert {e.value for e in RiskLevel} == {"LOW", "MEDIUM", "HIGH", "INSUFFICIENT_DATA"}
    assert InferenceStatus.SKIPPED_INELIGIBLE.value == "SKIPPED_INELIGIBLE"
    assert assessment.ReasonCode.NO_ELIGIBLE_SEVERITY_EVIDENCE
    assert evidence.ModelDescriptor
