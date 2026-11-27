"""Golden test: the refactored CLI must produce the same CSV and annotated frames as the legacy CLI.

`tests/fixtures/legacy_cli_main.py` is a verbatim snapshot of crowdguard/main.py taken before Task 2.
"""

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from tests.fixtures.synthetic import Scenario, install_fake_ultralytics, write_synthetic_video

ROOT = Path(__file__).resolve().parents[2]
ZONES = ROOT / "config" / "zones.json"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(module, monkeypatch, video: Path, out_dir: Path, extra=()):
    argv = ["crowdguard.main", "--source", str(video), "--zones", str(ZONES), "--output-dir", str(out_dir), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()


def _frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


@pytest.mark.parametrize("extra", [(), ("--interval", "0.5"), ("--no-save-video",)])
def test_refactored_cli_matches_legacy(tmp_path, monkeypatch, extra):
    scenario = Scenario()
    video = write_synthetic_video(tmp_path / "clip.mp4", scenario)

    install_fake_ultralytics(monkeypatch, scenario)
    legacy = _load_module("legacy_cli_main", ROOT / "tests" / "fixtures" / "legacy_cli_main.py")
    _run(legacy, monkeypatch, video, tmp_path / "legacy", extra)

    install_fake_ultralytics(monkeypatch, scenario)  # fresh call counter
    monkeypatch.delitem(sys.modules, "crowdguard.main", raising=False)
    import crowdguard.main as current

    _run(current, monkeypatch, video, tmp_path / "current", extra)

    legacy_csv = pd.read_csv(tmp_path / "legacy" / "clip_features.csv", keep_default_na=False)
    current_csv = pd.read_csv(tmp_path / "current" / "clip_features.csv", keep_default_na=False)
    assert len(legacy_csv) > 0
    pd.testing.assert_frame_equal(legacy_csv, current_csv)

    if "--no-save-video" in extra:
        assert not (tmp_path / "current" / "clip_annotated.mp4").exists()
        return
    a, b = _frames(tmp_path / "legacy" / "clip_annotated.mp4"), _frames(tmp_path / "current" / "clip_annotated.mp4")
    assert len(a) == len(b) == scenario.n_frames
    assert max(int(np.abs(x.astype(int) - y.astype(int)).max()) for x, y in zip(a, b)) == 0


def test_cli_still_exports_legacy_helpers(monkeypatch):
    install_fake_ultralytics(monkeypatch, Scenario())
    monkeypatch.delitem(sys.modules, "crowdguard.main", raising=False)
    import crowdguard.main as current

    for name in ["CSV_COLUMNS", "load_config", "pixel_polygon", "assign_zone", "direction_label", "line_side", "parse_args", "main"]:
        assert hasattr(current, name), name
