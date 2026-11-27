"""Overlay manifest and rendering (OV-01..OV-04 at unit level; D-07 containing-window semantics)."""

import numpy as np

from crowdguard.overlay import PALETTE, ManifestIndex, OverlayRenderer, build_overlay_manifest
from crowdguard.processing import pixel_polygon
from crowdguard_contracts.assessment import AssessmentStatus, PolicyRef, ReasonCode, RiskLevel, ZoneRiskAssessment

POLICY = PolicyRef(id="fixture-demo", version="1.0.0")
ZONES = {"A": [(0.05, 0.1), (0.45, 0.1), (0.45, 0.9), (0.05, 0.9)], "B": [(0.55, 0.1), (0.95, 0.1), (0.95, 0.9), (0.55, 0.9)]}


def _a(level, status=AssessmentStatus.DEMONSTRATION):
    reasons = [ReasonCode.WARMUP_INCOMPLETE] if level is RiskLevel.INSUFFICIENT_DATA else [ReasonCode.MODEL_EVIDENCE_SELECTED]
    return ZoneRiskAssessment(level=level, assessment_status=status, reason_codes=reasons, policy=POLICY)


def _manifest():
    # 2 windows of 10 frames, 25 frames total -> frames 20..24 are a tail without a window.
    assessments = {
        ("A", 0): _a(RiskLevel.INSUFFICIENT_DATA), ("A", 1): _a(RiskLevel.HIGH),
        ("B", 0): _a(RiskLevel.INSUFFICIENT_DATA), ("B", 1): _a(RiskLevel.LOW),
    }
    return build_overlay_manifest("an_1", "run_1", fps=10.0, frame_count=25, interval_frames=10, window_count=2,
                                  zone_order=["A", "B"], assessments=assessments, banner_status=AssessmentStatus.DEMONSTRATION)


def test_manifest_covers_every_frame_and_matches_assessments():
    manifest = _manifest()
    index = ManifestIndex(manifest)
    for zone in ("A", "B"):
        covered = sorted(f for e in manifest.entries if e.zone_id == zone for f in range(e.frame_start, e.frame_end + 1))
        assert covered == list(range(25))
    assert index.entry("A", 10).level is RiskLevel.HIGH and index.entry("A", 10).window_index == 1
    assert index.entry("B", 19).level is RiskLevel.LOW


def test_window_boundaries_have_no_off_by_one():
    index = ManifestIndex(_manifest())
    assert index.entry("A", 9).window_index == 0  # last frame of window 0 (t = 0.9 s)
    assert index.entry("A", 10).window_index == 1  # first frame of window 1 (t = 1.0 s)


def test_tail_frames_render_unknown():
    entry = ManifestIndex(_manifest()).entry("A", 22)
    assert entry.window_index is None
    assert entry.level is RiskLevel.INSUFFICIENT_DATA
    assert entry.palette_token == "risk-unknown"


def _blend(token, base=(90, 90, 90), alpha=0.35):
    color = np.array(PALETTE[token], dtype=float)
    return alpha * color + (1 - alpha) * np.array(base, dtype=float)


def test_rendered_interior_pixel_matches_palette_and_high_is_never_low():
    frame = np.full((200, 400, 3), 90, dtype=np.uint8)
    polygons = {z: pixel_polygon(p, 400, 200) for z, p in ZONES.items()}
    renderer = OverlayRenderer(_manifest(), polygons)
    out = renderer.render(frame, frame_index=12, tracks=[])
    a_px, b_px = out[100, 100].astype(float), out[100, 300].astype(float)  # interior samples away from labels
    assert np.linalg.norm(a_px - _blend("risk-high")) < 12
    assert np.linalg.norm(a_px - _blend("risk-low")) > 40
    assert np.linalg.norm(b_px - _blend("risk-low")) < 12


def test_banner_is_always_drawn():
    frame = np.full((200, 400, 3), 90, dtype=np.uint8)
    polygons = {z: pixel_polygon(p, 400, 200) for z, p in ZONES.items()}
    out = OverlayRenderer(_manifest(), polygons).render(frame, frame_index=22, tracks=[])
    assert not np.array_equal(out[0:16, :], frame[0:16, :])  # banner strip at the top
