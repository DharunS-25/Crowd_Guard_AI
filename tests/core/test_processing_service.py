from pathlib import Path

import pytest

from crowdguard.processing import (
    AnalysisInput,
    ProcessingCancelled,
    VideoOpenError,
    ZoneSpec,
    process_video,
)
from crowdguard_contracts.features import ZONE_WINDOW_FEATURES_V1
from tests.fixtures.synthetic import FakeDetector, Scenario, write_synthetic_video

SQUARES = {
    "STAGE": [(0.2, 0.02), (0.8, 0.02), (0.8, 0.22), (0.2, 0.22)],
    "MAIN": [(0.08, 0.23), (0.92, 0.23), (0.92, 0.72), (0.08, 0.72)],
    "ENTRY": [(0.02, 0.74), (0.48, 0.74), (0.48, 0.98), (0.02, 0.98)],
    "EXIT": [(0.52, 0.74), (0.98, 0.74), (0.98, 0.98), (0.52, 0.98)],
}


def _zones(capacity_for=None):
    capacity_for = capacity_for or {}
    return [ZoneSpec(zone_id=z, points=p, capacity=capacity_for.get(z)) for z, p in SQUARES.items()]


@pytest.fixture()
def clip(tmp_path):
    scenario = Scenario()
    return scenario, write_synthetic_video(tmp_path / "clip.mp4", scenario)


def test_windows_are_ordered_per_zone_and_aligned_to_frames(clip):
    scenario, path = clip
    result = process_video(AnalysisInput(source=path, zones=_zones(), interval_s=1.0), FakeDetector(scenario))

    assert result.meta.fps == pytest.approx(scenario.fps)
    assert result.interval_frames == 10
    assert result.meta.frames_processed == scenario.n_frames
    # 45 frames, 10-frame windows -> 4 complete windows; the 5 tail frames have no window.
    assert result.window_count == 4
    by_zone = {}
    for w in result.windows:
        assert w.feature_schema == ZONE_WINDOW_FEATURES_V1
        assert list(w.values) == ZONE_WINDOW_FEATURES_V1.columns
        by_zone.setdefault(w.zone_id, []).append(w.window)
    assert set(by_zone) == set(SQUARES)
    for windows in by_zone.values():
        assert [w.index for w in windows] == [0, 1, 2, 3]
        assert [w.start_s for w in windows] == pytest.approx([0.0, 1.0, 2.0, 3.0])
        assert [w.end_s for w in windows] == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert result.window_index_for_frame(0) == 0
    assert result.window_index_for_frame(39) == 3
    assert result.window_index_for_frame(40) is None


def test_zone_assignment_and_counts(clip):
    scenario, path = clip
    result = process_video(AnalysisInput(source=path, zones=_zones(), interval_s=1.0), FakeDetector(scenario))
    get = {(w.zone_id, w.window.index): w for w in result.windows}
    assert get[("MAIN", 0)].values["people_count_end"] == 1
    assert get[("STAGE", 0)].values["people_count_avg"] == 0  # track 3 appears at frame 10
    assert get[("STAGE", 1)].values["people_count_avg"] == 1
    # Legacy behaviour (frozen): line_side() treats the flow line as an infinite line, so both the ENTRY->EXIT walker
    # and the MAIN walker crossing x = 0.5 are counted. This is the documented "illustrative line" limitation.
    assert get[("EXIT", 3)].values["virtual_line_crossings_total"] == 2


def test_capacity_requires_source_and_occupancy_is_null_otherwise(clip):
    scenario, path = clip
    result = process_video(AnalysisInput(source=path, zones=_zones({"MAIN": 10.0})), FakeDetector(scenario))
    main0 = next(w for w in result.windows if w.zone_id == "MAIN" and w.window.index == 0)
    entry0 = next(w for w in result.windows if w.zone_id == "ENTRY" and w.window.index == 0)
    assert main0.values["occupancy_ratio"] == pytest.approx(0.1)
    assert entry0.values["occupancy_ratio"] is None
    assert entry0.values["configured_capacity"] is None
    assert "CAPACITY_UNCONFIGURED" in entry0.data_quality.warnings
    assert "CAPACITY_UNCONFIGURED" not in main0.data_quality.warnings


def test_no_detections_is_an_observation_with_a_warning(clip):
    scenario, path = clip
    result = process_video(AnalysisInput(source=path, zones=_zones()), FakeDetector(scenario))
    stage0 = next(w for w in result.windows if w.zone_id == "STAGE" and w.window.index == 0)
    assert stage0.values["people_count_avg"] == 0
    assert stage0.data_quality.state.value == "OK"
    assert "NO_DETECTIONS" in stage0.data_quality.warnings
    assert stage0.data_quality.frames_expected == stage0.data_quality.frames_decoded == 10


def test_fps_fallback_is_recorded(clip, monkeypatch):
    scenario, path = clip
    import cv2

    class NoFps:
        def __init__(self, src):
            self._cap = cv2.VideoCapture(src)

        def get(self, prop):
            return 0.0 if prop == cv2.CAP_PROP_FPS else self._cap.get(prop)

        def __getattr__(self, name):
            return getattr(self._cap, name)

    result = process_video(
        AnalysisInput(source=path, zones=_zones(), fps_fallback=5.0), FakeDetector(scenario), open_capture=NoFps
    )
    assert result.meta.fps == 5.0 and result.meta.fps_fallback_used
    assert all("FPS_FALLBACK_USED" in w.data_quality.warnings for w in result.windows)


def test_progress_and_cancellation(clip):
    scenario, path = clip
    seen = []
    process_video(AnalysisInput(source=path, zones=_zones()), FakeDetector(scenario), progress=lambda done, total: seen.append((done, total)))
    assert seen and seen[-1][0] == scenario.n_frames

    calls = {"n": 0}

    def cancel_after_5():
        calls["n"] += 1
        return calls["n"] > 5

    with pytest.raises(ProcessingCancelled):
        process_video(AnalysisInput(source=path, zones=_zones()), FakeDetector(scenario), is_cancelled=cancel_after_5)


def test_frame_tracks_recorded_for_render_pass(clip):
    scenario, path = clip
    result = process_video(AnalysisInput(source=path, zones=_zones()), FakeDetector(scenario))
    assert len(result.frame_tracks) == scenario.n_frames
    assert {t.track_id for t in result.frame_tracks[15]} == {1, 2, 3}
    assert result.frame_tracks[36] == []


def test_unreadable_source_raises(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(VideoOpenError):
        process_video(AnalysisInput(source=bad, zones=_zones()), FakeDetector(Scenario()))


def test_invalid_interval_rejected(clip):
    _, path = clip
    with pytest.raises(ValueError):
        AnalysisInput(source=path, zones=_zones(), interval_s=0)
    with pytest.raises(ValueError):
        AnalysisInput(source=path, zones=[])
