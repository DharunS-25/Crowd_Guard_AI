"""Decision policy (risk-assessment.md §6): MD-01, MD-05, MD-06, MD-08 at unit level."""

from pathlib import Path

import pytest

from crowdguard.models.orchestrator import Orchestrator
from crowdguard.risk.policy import DEFAULT_POLICY_PATH, DecisionPolicy, PolicyError
from crowdguard_contracts.assessment import AssessmentStatus, ReasonCode, RiskLevel
from crowdguard_contracts.errors import ErrorCode
from tests.fixtures.models.fakes import (
    COUNTS_PRODUCER,
    FakeBinaryAnomalyAdapter,
    FakeVenueRiskAdapter,
    UnknownSchemaAdapter,
)
from tests.fixtures.models.windows import make_windows

POLICIES = Path(__file__).resolve().parents[1] / "fixtures" / "policies"


def _assess(policy, adapters, windows):
    policy = policy.bind([a.descriptor() for a in adapters])
    evidence = Orchestrator(adapters, producers=[COUNTS_PRODUCER]).run(windows)
    return {(w.zone_id, w.window.index): policy.assess(w, evidence.get((w.zone_id, w.window.index), [])) for w in windows}


def test_demo_policy_yields_expected_levels_after_warmup():
    out = _assess(DecisionPolicy.load(POLICIES / "demo.yaml"), [FakeVenueRiskAdapter({"A": "HIGH", "B": "LOW", "C": "MEDIUM"})], make_windows())
    assert out[("A", 3)].assessment.level is RiskLevel.HIGH
    assert out[("B", 3)].assessment.level is RiskLevel.LOW
    assert out[("C", 3)].assessment.level is RiskLevel.MEDIUM
    assert out[("A", 3)].assessment.assessment_status is AssessmentStatus.DEMONSTRATION
    assert ReasonCode.MODEL_EVIDENCE_SELECTED in out[("A", 3)].assessment.reason_codes
    # window 0 is warm-up (sequence_length 2): never LOW, always explained
    first = out[("B", 0)].assessment
    assert first.level is RiskLevel.INSUFFICIENT_DATA
    assert ReasonCode.WARMUP_INCOMPLETE in first.reason_codes


def test_default_policy_without_models_is_insufficient_everywhere():
    out = _assess(DecisionPolicy.load(DEFAULT_POLICY_PATH), [], make_windows())
    for result in out.values():
        a = result.assessment
        assert a.level is RiskLevel.INSUFFICIENT_DATA
        assert ReasonCode.NO_ELIGIBLE_SEVERITY_EVIDENCE in a.reason_codes
        assert ReasonCode.POLICY_THRESHOLDS_UNCONFIGURED in a.reason_codes
        assert a.assessment_status is AssessmentStatus.EXPERIMENTAL
        assert {s.status for s in result.rule_signals} == {"NOT_CONFIGURED"}


def test_binary_anomaly_evidence_never_changes_the_level():
    windows = make_windows()
    demo = DecisionPolicy.load(POLICIES / "demo.yaml")
    venue = FakeVenueRiskAdapter({"A": "LOW", "B": "LOW", "C": "LOW"})
    with_binary = _assess(demo, [venue, FakeBinaryAnomalyAdapter()], windows)
    without = _assess(demo, [venue], windows)
    for key in without:
        assert with_binary[key].assessment.level is without[key].assessment.level
    # default policy + only the binary model: still INSUFFICIENT_DATA (no mapping Normal/Anomalous -> severity)
    only_binary = _assess(DecisionPolicy.load(DEFAULT_POLICY_PATH), [FakeBinaryAnomalyAdapter()], windows)
    assert all(r.assessment.level is RiskLevel.INSUFFICIENT_DATA for r in only_binary.values())


def test_policy_mapping_ucsd_label_space_is_refused():
    with pytest.raises(PolicyError) as err:
        DecisionPolicy.load(POLICIES / "maps_ucsd.yaml")
    assert err.value.code is ErrorCode.POLICY_INCOMPLETE


def test_disagreement_requires_an_explicit_rule():
    a = FakeVenueRiskAdapter({"A": "HIGH"}, model_id="venue-a")
    b = FakeVenueRiskAdapter({"A": "LOW"}, model_id="venue-b")
    with pytest.raises(PolicyError):
        DecisionPolicy.load(POLICIES / "demo_no_disagreement_rule.yaml").bind([a.descriptor(), b.descriptor()])
    out = _assess(DecisionPolicy.load(POLICIES / "demo.yaml"), [a, b], make_windows())
    result = out[("A", 3)].assessment
    assert result.level is RiskLevel.HIGH  # highest_severity rule
    assert ReasonCode.MODEL_DISAGREEMENT in result.reason_codes
    assert len(result.evidence_refs) == 2


def test_invalid_window_quality_is_insufficient_data():
    out = _assess(DecisionPolicy.load(POLICIES / "demo.yaml"), [FakeVenueRiskAdapter({"A": "HIGH"})], make_windows(invalid=[("A", 3)]))
    assert out[("A", 3)].assessment.level is RiskLevel.INSUFFICIENT_DATA
    assert ReasonCode.FEATURE_INVALID in out[("A", 3)].assessment.reason_codes


def test_unknown_feature_schema_is_skipped_ineligible():
    evidence = Orchestrator([UnknownSchemaAdapter()], producers=[COUNTS_PRODUCER]).run(make_windows())
    ev = evidence[("A", 3)][0]
    assert ev.inference_status.value == "SKIPPED_INELIGIBLE"
    assert ev.reason == "NO_FEATURE_PRODUCER_FOR_SCHEMA"


def test_rule_signals_use_sourced_thresholds_only():
    windows = make_windows(counts={"A": 6, "B": 3, "C": 0})
    out = _assess(DecisionPolicy.load(POLICIES / "demo_rules.yaml"), [], windows)
    assert out[("A", 0)].assessment.level is RiskLevel.HIGH
    assert out[("B", 0)].assessment.level is RiskLevel.MEDIUM
    assert out[("C", 0)].assessment.level is RiskLevel.LOW
    assert out[("A", 0)].rule_signals[0].threshold_source.startswith("synthetic test fixture")
    assert ReasonCode.POLICY_RULE_MATCHED in out[("A", 0)].assessment.reason_codes


def test_thresholds_without_source_fail_to_load(tmp_path):
    bad = (POLICIES / "demo_rules.yaml").read_text().replace('threshold_source: "synthetic test fixture (tests/fixtures/policies/demo_rules.yaml)"', "threshold_source: null")
    path = tmp_path / "bad.yaml"
    path.write_text(bad)
    with pytest.raises(PolicyError):
        DecisionPolicy.load(path)


def test_policy_identity_and_hash():
    policy = DecisionPolicy.load(DEFAULT_POLICY_PATH)
    assert policy.ref.id == "crowdguard-default" and policy.ref.version == "0.1.0"
    assert len(policy.sha256) == 64
