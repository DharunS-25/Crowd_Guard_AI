"""Deterministic test fixtures: synthetic videos and a fake YOLO/ByteTrack.

Nothing here downloads weights or touches the network. Videos are generated at test time and never committed.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Normalized bottom-centre tracks: track_id -> function(frame_index) -> (x, y_bottom) or None.
Track = Callable[[int], Optional[Tuple[float, float]]]

BOX_W, BOX_H = 0.06, 0.15


def _linear(x0: float, x1: float, y: float, n: int) -> Track:
    return lambda f: (x0 + (x1 - x0) * min(f, n - 1) / max(1, n - 1), y)


def default_tracks(n_frames: int) -> Dict[int, Track]:
    """Tracks laid out against config/zones.json (STAGE, MAIN, ENTRY, EXIT; flow line at x=0.5)."""
    return {
        1: _linear(0.20, 0.70, 0.50, n_frames),  # MAIN, walking east
        2: _linear(0.30, 0.70, 0.85, n_frames),  # ENTRY -> EXIT, crosses the flow line
        3: lambda f: (0.50, 0.15) if 10 <= f < 30 else None,  # STAGE, stationary, frames 10-29
    }


@dataclass
class Scenario:
    n_frames: int = 45
    width: int = 320
    height: int = 240
    fps: float = 10.0
    blank_frames: Tuple[int, int] = (35, 40)  # [start, end): detector returns no tracks
    tracks: Optional[Dict[int, Track]] = None

    def boxes_for(self, frame_index: int) -> List[Tuple[float, float, float, float, int, float]]:
        if self.blank_frames[0] <= frame_index < self.blank_frames[1]:
            return []
        tracks = self.tracks if self.tracks is not None else default_tracks(self.n_frames)
        out = []
        for track_id, fn in tracks.items():
            pos = fn(frame_index)
            if pos is None:
                continue
            x, yb = pos
            x1, x2 = (x - BOX_W / 2) * self.width, (x + BOX_W / 2) * self.width
            y1, y2 = (yb - BOX_H) * self.height, yb * self.height
            out.append((x1, y1, x2, y2, track_id, 0.8 + 0.01 * track_id))
        return out


def write_synthetic_video(path: Path, scenario: Scenario, seed: int = 7) -> Path:
    """Write an mp4v video whose frames are deterministic noise with drawn 'people'."""
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), scenario.fps, (scenario.width, scenario.height))
    assert writer.isOpened(), "OpenCV could not open a VideoWriter for the fixture"
    try:
        for f in range(scenario.n_frames):
            frame = rng.integers(40, 80, size=(scenario.height, scenario.width, 3), dtype=np.uint8)
            for x1, y1, x2, y2, _, _ in scenario.boxes_for(f):
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 200), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def write_noise_video(path: Path, seconds: float, fps: float = 25.0, size: Tuple[int, int] = (640, 480), seed: int = 3) -> Path:
    """High-entropy video, useful for multipart-sized uploads."""
    rng = np.random.default_rng(seed)
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for _ in range(int(seconds * fps)):
            writer.write(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8))
    finally:
        writer.release()
    return path


# ------------------------------------------------------------------ fake ultralytics (YOLO + ByteTrack)


class _T:
    """Minimal tensor stand-in supporting .cpu().numpy(), .int().cpu().tolist()."""

    def __init__(self, arr):
        self._a = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    def int(self):
        return _T(self._a.astype(np.int64))

    def tolist(self):
        return self._a.tolist()

    def __len__(self):
        return len(self._a)


class _Boxes:
    def __init__(self, rows):
        self._n = len(rows)
        self.xyxy = _T(np.array([r[:4] for r in rows], dtype=np.float32).reshape(-1, 4))
        self.id = _T(np.array([r[4] for r in rows], dtype=np.float32)) if rows else None
        self.conf = _T(np.array([r[5] for r in rows], dtype=np.float32))

    def __len__(self):
        return self._n


class _Result:
    def __init__(self, rows):
        self.boxes = _Boxes(rows)


def fake_ultralytics_module(scenario: Scenario) -> types.ModuleType:
    """Build a module named `ultralytics` whose YOLO.track replays the scenario frame by frame."""

    class YOLO:  # noqa: D401 - mimics ultralytics.YOLO
        def __init__(self, *_args, **_kwargs):
            self.calls = 0

        def track(self, frame, **kwargs):
            assert kwargs.get("persist") is True and kwargs.get("tracker") == "bytetrack.yaml"
            assert kwargs.get("classes") == [0]
            rows = scenario.boxes_for(self.calls)
            self.calls += 1
            return [_Result(rows)]

    module = types.ModuleType("ultralytics")
    module.YOLO = YOLO
    return module


class FakeDetector:
    """crowdguard.processing.Detector implementation replaying a Scenario."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.calls = 0

    def track(self, frame) -> Sequence:
        from crowdguard.processing import TrackedBox

        rows = self.scenario.boxes_for(self.calls)
        self.calls += 1
        return [
            TrackedBox(float(np.float32(x1)), float(np.float32(y1)), float(np.float32(x2)), float(np.float32(y2)), int(t), float(np.float32(c)))
            for x1, y1, x2, y2, t, c in rows
        ]


def install_fake_ultralytics(monkeypatch, scenario: Scenario) -> None:
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics_module(scenario))
