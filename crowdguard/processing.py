"""Reusable video-processing service: decode → detect/track → zones → ordered per-zone feature windows.

This module holds the logic that used to live inside `crowdguard/main.py::main()`. The CLI is now a thin adapter
(see main.py) and must keep producing byte-identical CSV rows and annotated frames (tests/core/test_cli_compat.py).

The service knows nothing about HTTP, S3, or databases. The detector is injectable so tests use a fake.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import cv2
import numpy as np

from crowdguard_contracts.assessment import WarningCode
from crowdguard_contracts.features import DataQuality, DataQualityState, Window, ZoneWindowFeatures

from .feature_schema import legacy_row_to_values, schema

# ============================================================ legacy CSV contract (frozen)

CSV_COLUMNS = [
    "timestamp_start_s",
    "timestamp_end_s",
    "zone",
    "people_count_avg",
    "people_count_end",
    "image_density_proxy_per_10000_px",
    "mean_speed_px_per_s",
    "dominant_direction",
    "virtual_line_crossings_total",
    "flow_crossings_per_s_interval",
    "configured_capacity",
    "occupancy_ratio",
    "elapsed_video_s",
]

DEFAULT_FLOW_LINE = ((0.5, 0.72), (0.5, 0.98))
UNASSIGNED = "UNASSIGNED"


# ============================================================ geometry helpers (moved from main.py, unchanged)


def load_config(path: Path) -> dict:
    """Load zone polygons and capacity settings from JSON."""

    if not path.exists():
        raise FileNotFoundError(f"Zone configuration file was not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if "zones" not in config or not config["zones"]:
        raise ValueError("zones.json must contain a non-empty 'zones' object.")

    return config


def pixel_polygon(points, width: int, height: int) -> np.ndarray:
    """Convert normalized polygon points into image coordinates."""

    polygon = np.array(
        [[round(float(x) * width), round(float(y) * height)] for x, y in points],
        dtype=np.int32,
    )

    if len(polygon) < 3:
        raise ValueError("Each zone polygon must contain at least three points.")

    return polygon


def point_in_polygon(point, polygon) -> bool:
    """Check whether a point is inside or on a zone boundary."""

    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


def assign_zone(point, polygons) -> str:
    """Assign a person to the first polygon containing the point."""

    for name, polygon in polygons.items():
        if point_in_polygon(point, polygon):
            return name

    return UNASSIGNED


def line_side(point, a, b) -> float:
    """Return which side of a directed line a point is on."""

    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


def direction_label(dx: float, dy: float, threshold: float = 2.0) -> str:
    """Convert image-coordinate displacement to 8-way direction."""

    if abs(dx) < threshold and abs(dy) < threshold:
        return "STATIONARY"

    # Image y-coordinates increase downwards.
    angle = math.degrees(math.atan2(-dy, dx))

    if -22.5 <= angle < 22.5:
        return "E"
    if 22.5 <= angle < 67.5:
        return "NE"
    if 67.5 <= angle < 112.5:
        return "N"
    if 112.5 <= angle < 157.5:
        return "NW"
    if angle >= 157.5 or angle < -157.5:
        return "W"
    if -157.5 <= angle < -112.5:
        return "SW"
    if -112.5 <= angle < -67.5:
        return "S"

    return "SE"


# ============================================================ detector boundary


@dataclass(frozen=True)
class TrackedBox:
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int
    confidence: float


class Detector(Protocol):
    """Detect people and return camera-local track IDs for one frame (frames arrive in order)."""

    def track(self, frame: np.ndarray) -> Sequence[TrackedBox]: ...


class UltralyticsDetector:
    """YOLO person detection + ByteTrack via Ultralytics. Imported lazily so tests never need the weights."""

    def __init__(self, model_path: str = "yolo11n.pt", conf: float = 0.25, imgsz: int = 640):
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._conf = conf
        self._imgsz = imgsz

    def track(self, frame: np.ndarray) -> List[TrackedBox]:
        result = self._model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=self._conf,
            imgsz=self._imgsz,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().tolist()
        confs = boxes.conf.cpu().numpy()
        return [
            TrackedBox(float(b[0]), float(b[1]), float(b[2]), float(b[3]), int(i), float(c))
            for b, i, c in zip(xyxy, ids, confs)
        ]


# ============================================================ inputs / outputs


class VideoOpenError(RuntimeError):
    pass


class ProcessingCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class ZoneSpec:
    """A zone in normalized coordinates. `capacity` must come from a sourced venue value (or legacy CLI config)."""

    zone_id: str
    points: Sequence[Tuple[float, float]]
    capacity: Optional[float] = None


@dataclass(frozen=True)
class AnalysisInput:
    source: Union[str, int, Path]
    zones: Sequence[ZoneSpec]
    camera_id: str = "cam-1"
    interval_s: float = 1.0
    flow_line: Tuple[Tuple[float, float], Tuple[float, float]] = DEFAULT_FLOW_LINE
    fps_fallback: float = 25.0
    progress_every_frames: int = 10

    def __post_init__(self):
        if not self.interval_s or self.interval_s <= 0:
            raise ValueError("interval_s must be greater than 0")
        if not self.zones:
            raise ValueError("at least one zone is required")
        if len({z.zone_id for z in self.zones}) != len(self.zones):
            raise ValueError("zone ids must be unique")
        if self.fps_fallback <= 0:
            raise ValueError("fps_fallback must be greater than 0")


@dataclass(frozen=True)
class SourceMeta:
    fps: float
    fps_fallback_used: bool
    width: int
    height: int
    frames_processed: int
    reported_frame_count: Optional[int]

    @property
    def duration_s(self) -> float:
        return self.frames_processed / self.fps


@dataclass(frozen=True)
class FrameTrack:
    """What the render pass needs to redraw a detection without re-running the detector."""

    x1: int
    y1: int
    x2: int
    y2: int
    track_id: int
    confidence: float
    zone: str
    speed_px_s: float


@dataclass
class FrameObservation:
    """Per-frame state handed to an optional observer (the CLI uses it to draw its legacy annotations)."""

    frame: np.ndarray
    frame_index: int
    timestamp: float
    tracks: List[FrameTrack]
    total_people_in_zones: int
    total_line_crossings: int


@dataclass
class StartInfo:
    fps: float
    width: int
    height: int
    polygons: Dict[str, np.ndarray]
    flow_a: Tuple[int, int]
    flow_b: Tuple[int, int]


@dataclass
class ProcessingResult:
    meta: SourceMeta
    interval_frames: int
    windows: List[ZoneWindowFeatures]
    legacy_rows: List[dict]
    frame_tracks: List[List[FrameTrack]]
    polygons: Dict[str, np.ndarray]
    zone_order: List[str] = field(default_factory=list)

    @property
    def window_count(self) -> int:
        return len(self.legacy_rows) // max(1, len(self.zone_order))

    def window_index_for_frame(self, frame_index: int) -> Optional[int]:
        """Index of the complete window containing this frame, or None (e.g. tail frames)."""
        index = frame_index // self.interval_frames
        return index if index < self.window_count else None


# ============================================================ the service


def _fresh_acc():
    return {"count_sum": 0, "speed_sum": 0.0, "speed_n": 0, "directions": defaultdict(int), "frames": 0}


def process_video(
    request: AnalysisInput,
    detector: Detector,
    progress: Optional[Callable[[int, Optional[int]], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    on_start: Optional[Callable[[StartInfo], None]] = None,
    on_frame: Optional[Callable[[FrameObservation], bool]] = None,
    open_capture: Callable = cv2.VideoCapture,
) -> ProcessingResult:
    """Run detection/tracking and build ordered per-zone feature windows.

    `on_frame` may return True to stop early (legacy CLI preview 'q'). Windows are aligned to frames:
    window k covers frames [k*N, (k+1)*N) with N = round(fps * interval_s); incomplete tail frames get no window.
    """

    source = request.source
    cap = open_capture(str(source) if isinstance(source, Path) else source)
    if not cap.isOpened():
        raise VideoOpenError(f"Cannot open video source: {source}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_fallback_used = False
        if not math.isfinite(fps) or fps <= 0:
            fps = request.fps_fallback
            fps_fallback_used = True

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise VideoOpenError("Could not read the input video's dimensions.")
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None

        polygons = {z.zone_id: pixel_polygon(z.points, width, height) for z in request.zones}
        capacities = {z.zone_id: z.capacity for z in request.zones}
        zone_order = list(polygons)

        line_from, line_to = request.flow_line
        flow_a = (int(float(line_from[0]) * width), int(float(line_from[1]) * height))
        flow_b = (int(float(line_to[0]) * width), int(float(line_to[1]) * height))
        if flow_a == flow_b:
            raise ValueError("The virtual flow line must have two different endpoints.")

        if on_start is not None:
            on_start(StartInfo(fps=fps, width=width, height=height, polygons=polygons, flow_a=flow_a, flow_b=flow_b))

        track_history = defaultdict(lambda: deque(maxlen=8))
        previous_line_side: Dict[int, float] = {}
        last_counted_crossing: Dict[int, float] = {}
        total_line_crossings = 0
        interval_frames = max(1, int(round(fps * request.interval_s)))
        zone_accum = defaultdict(_fresh_acc)
        zone_last_count = defaultdict(int)

        rows: List[dict] = []
        windows: List[ZoneWindowFeatures] = []
        frame_tracks: List[List[FrameTrack]] = []
        frame_index = 0

        while True:
            if is_cancelled is not None and is_cancelled():
                raise ProcessingCancelled("processing cancelled")

            ok, frame = cap.read()
            if not ok:
                break

            timestamp = frame_index / fps
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

            detections = detector.track(frame)

            current_by_zone = defaultdict(set)
            current_speed_by_zone = defaultdict(list)
            current_direction_by_zone = defaultdict(list)
            tracks_this_frame: List[FrameTrack] = []

            for det in detections:
                x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
                track_id = det.track_id
                # Bottom-centre approximates the ground contact point.
                point = (int((x1 + x2) / 2), int(y2))
                zone = assign_zone(point, polygons)

                hist = track_history[track_id]
                hist.append((timestamp, point[0], point[1]))

                speed_px_s = 0.0
                direction = "UNKNOWN"
                if len(hist) >= 2:
                    t0, x0, y0 = hist[0]
                    t1, x1p, y1p = hist[-1]
                    dt = t1 - t0
                    if dt > 0:
                        dx = x1p - x0
                        dy = y1p - y0
                        speed_px_s = math.hypot(dx, dy) / dt
                        direction = direction_label(dx, dy)

                if zone != UNASSIGNED:
                    current_by_zone[zone].add(track_id)
                    current_speed_by_zone[zone].append(speed_px_s)
                    current_direction_by_zone[zone].append(direction)

                side = line_side(point, flow_a, flow_b)
                old_side = previous_line_side.get(track_id)
                if (
                    old_side is not None
                    and side * old_side < 0
                    and timestamp - last_counted_crossing.get(track_id, -999.0) > 1.0
                ):
                    total_line_crossings += 1
                    last_counted_crossing[track_id] = timestamp
                previous_line_side[track_id] = side

                tracks_this_frame.append(FrameTrack(x1, y1, x2, y2, track_id, det.confidence, zone, speed_px_s))

            for zone in polygons:
                count = len(current_by_zone[zone])
                acc = zone_accum[zone]
                acc["count_sum"] += count
                acc["frames"] += 1
                speeds = current_speed_by_zone[zone]
                if speeds:
                    acc["speed_sum"] += float(np.mean(speeds))
                    acc["speed_n"] += 1
                for direction in current_direction_by_zone[zone]:
                    acc["directions"][direction] += 1
                zone_last_count[zone] = count

            total_people = sum(len(ids) for ids in current_by_zone.values())
            frame_tracks.append(tracks_this_frame)

            stop = False
            if on_frame is not None:
                stop = bool(
                    on_frame(
                        FrameObservation(
                            frame=frame,
                            frame_index=frame_index,
                            timestamp=timestamp,
                            tracks=tracks_this_frame,
                            total_people_in_zones=total_people,
                            total_line_crossings=total_line_crossings,
                        )
                    )
                )
            frame_index += 1
            if stop:
                break

            if progress is not None and frame_index % max(1, request.progress_every_frames) == 0:
                progress(frame_index, reported)

            if frame_index % interval_frames == 0:
                window_number = frame_index // interval_frames - 1
                end_time = frame_index / fps
                window = Window(
                    index=window_number,
                    start_s=window_number * interval_frames / fps,
                    end_s=end_time,
                )
                for zone, polygon in polygons.items():
                    acc = zone_accum[zone]
                    row = _legacy_row(acc, zone, polygon, capacities.get(zone), zone_last_count[zone],
                                      total_line_crossings, end_time, request.interval_s)
                    rows.append(row)
                    windows.append(
                        _window_features(row, request.camera_id, zone, window, acc, interval_frames,
                                         capacities.get(zone), fps_fallback_used)
                    )
                    zone_accum[zone] = _fresh_acc()

        if progress is not None:
            progress(frame_index, reported)
    finally:
        cap.release()

    meta = SourceMeta(
        fps=fps,
        fps_fallback_used=fps_fallback_used,
        width=width,
        height=height,
        frames_processed=frame_index,
        reported_frame_count=reported,
    )
    return ProcessingResult(
        meta=meta,
        interval_frames=interval_frames,
        windows=windows,
        legacy_rows=rows,
        frame_tracks=frame_tracks,
        polygons=polygons,
        zone_order=zone_order,
    )


def _legacy_row(acc, zone, polygon, capacity, end_count, total_line_crossings, end_time, interval_s) -> dict:
    """Exactly the CSV row the original CLI produced (rounding, blanks, and start-time rule included)."""

    count_avg = acc["count_sum"] / max(1, acc["frames"])
    avg_speed = acc["speed_sum"] / max(1, acc["speed_n"])
    polygon_area_px = max(1.0, abs(cv2.contourArea(polygon)))
    # Image-area proxy, not physical density.
    density_proxy = count_avg / (polygon_area_px / 10000.0)
    capacity_value = float(capacity or 0)
    occupancy = end_count / capacity_value if capacity_value > 0 else None
    dominant_direction = (
        max(acc["directions"], key=acc["directions"].get) if acc["directions"] else "UNKNOWN"
    )
    return {
        "timestamp_start_s": round(max(0.0, end_time - interval_s), 3),
        "timestamp_end_s": round(end_time, 3),
        "zone": zone,
        "people_count_avg": round(count_avg, 3),
        "people_count_end": int(end_count),
        "image_density_proxy_per_10000_px": round(density_proxy, 4),
        "mean_speed_px_per_s": round(avg_speed, 3),
        "dominant_direction": dominant_direction,
        "virtual_line_crossings_total": int(total_line_crossings),
        "flow_crossings_per_s_interval": round(total_line_crossings / max(end_time, 0.001), 4),
        "configured_capacity": int(capacity_value) if capacity_value else "",
        "occupancy_ratio": round(occupancy, 4) if occupancy is not None else "",
        "elapsed_video_s": round(end_time, 3),
    }


def _window_features(row, camera_id, zone, window, acc, interval_frames, capacity, fps_fallback_used) -> ZoneWindowFeatures:
    warnings: List[str] = []
    if acc["count_sum"] == 0:
        warnings.append(WarningCode.NO_DETECTIONS.value)
    if not capacity:
        warnings.append(WarningCode.CAPACITY_UNCONFIGURED.value)
    if fps_fallback_used:
        warnings.append(WarningCode.FPS_FALLBACK_USED.value)
    state = DataQualityState.OK
    if acc["frames"] < interval_frames:
        state = DataQualityState.DEGRADED
        warnings.append(WarningCode.FRAMES_DROPPED.value)
    return ZoneWindowFeatures(
        camera_id=camera_id,
        zone_id=zone,
        window=window,
        feature_schema=schema(),
        values=legacy_row_to_values(row),
        data_quality=DataQuality(
            state=state, warnings=warnings, frames_expected=interval_frames, frames_decoded=acc["frames"]
        ),
    )
