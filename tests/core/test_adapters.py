"""Adapter contract checks (risk-assessment.md §4.2), MD-02/03/04/07 at unit level."""

import math

from crowdguard.models.base import FeatureSequence, sha256_file
from crowdguard.models.orchestrator import Orchestrator
from crowdguard_contracts.evidence import InferenceStatus
from tests.fixtures.models.fakes import (
    COUNTS_PRODUCER,
    COUNTS_SCHEMA,
    FakeVenueRiskAdapter,
    MissingArtifactAdapter,
    SchemaMismatchAdapter,
)
from tests.fixtures.models.windows import make_windows


def _sequence(windows, zone="A", upto=3, columns=None):
    rows = [w for w in windows if w.zone_id == zone and w.window.index <= upto]
    seq = COUNTS_PRODUCER.produce(rows)
    if columns is not None:
        seq = FeatureSequence(**{**seq.__dict__, "columns": columns,
                                 "rows": [[r[seq.columns.index(c)] for c in columns] for r in seq.rows]})
    return seq


def test_ok_evidence_has_probabilities_and_input_provenance():
    ev = FakeVenueRiskAdapter({"A": "HIGH"}).predict(_sequence(make_windows()))
    assert ev.inference_status is InferenceStatus.OK
    assert max(ev.probabilities, key=ev.probabilities.get) == "HIGH"
    assert ev.input.rows_used == 2 and ev.input.feature_schema == "zone_window_counts@1.0.0"
    assert ev.window.index == 3 and ev.zone_id == "A"


def test_warmup_incomplete_is_insufficient_data_not_a_prediction():
    ev = FakeVenueRiskAdapter({"A": "HIGH"}, seq_len=3).predict(_sequence(make_windows(), upto=1))
    assert ev.inference_status is InferenceStatus.INSUFFICIENT_DATA
    assert ev.reason == "WARMUP_INCOMPLETE" and ev.probabilities is None


def test_sequence_gap_detected():
    windows = [w for w in make_windows() if not (w.zone_id == "A" and w.window.index == 2)]
    ev = FakeVenueRiskAdapter({"A": "HIGH"}).predict(_sequence(windows))
    assert ev.inference_status is InferenceStatus.INSUFFICIENT_DATA and ev.reason == "SEQUENCE_GAP"


def test_nan_inf_or_missing_values_are_never_filled():
    for bad in (float("nan"), math.inf, None):
        windows = make_windows(override={("A", 3): {"mean_speed_px_per_s": bad}})
        ev = FakeVenueRiskAdapter({"A": "HIGH"}).predict(_sequence(windows))
        assert ev.inference_status is InferenceStatus.INSUFFICIENT_DATA, bad
        assert ev.reason == "FEATURE_INVALID" and ev.probabilities is None


def test_invalid_quality_row_rejected():
    ev = FakeVenueRiskAdapter({"A": "HIGH"}).predict(_sequence(make_windows(invalid=[("A", 3)])))
    assert ev.inference_status is InferenceStatus.INSUFFICIENT_DATA and ev.reason == "FEATURE_INVALID"


def test_reordered_columns_are_a_schema_mismatch_not_reordered():
    seq = _sequence(make_windows(), columns=list(reversed(COUNTS_SCHEMA.columns)))
    ev = FakeVenueRiskAdapter({"A": "HIGH"}).predict(seq)
    assert ev.inference_status is InferenceStatus.SCHEMA_MISMATCH and ev.probabilities is None
    ev2 = SchemaMismatchAdapter().predict(_sequence(make_windows()))
    assert ev2.inference_status is InferenceStatus.SCHEMA_MISMATCH


def test_missing_or_tampered_artifact_is_model_unavailable(tmp_path):
    ev = MissingArtifactAdapter(tmp_path / "nope.pt").predict(_sequence(make_windows()))
    assert ev.inference_status is InferenceStatus.MODEL_UNAVAILABLE
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"weights")
    ev_bad_hash = MissingArtifactAdapter(artifact, expected_sha256="f" * 64).predict(_sequence(make_windows()))
    assert ev_bad_hash.inference_status is InferenceStatus.MODEL_UNAVAILABLE
    ok = MissingArtifactAdapter(artifact, expected_sha256=sha256_file(artifact)).predict(_sequence(make_windows()))
    assert ok.inference_status is InferenceStatus.OK


def test_orchestrator_isolates_adapter_exceptions_per_window():
    from tests.fixtures.models.fakes import RaisingAdapter

    evidence = Orchestrator([RaisingAdapter(fail_on_index=2)], producers=[COUNTS_PRODUCER]).run(make_windows())
    statuses = {(e.zone_id, e.window.index): e.inference_status for e in evidence[("A", 2)] + evidence[("A", 3)]}
    assert statuses[("A", 2)] is InferenceStatus.ERROR
    assert statuses[("A", 3)] is InferenceStatus.OK
    assert evidence[("A", 2)][0].reason == "MODEL_ERROR"
