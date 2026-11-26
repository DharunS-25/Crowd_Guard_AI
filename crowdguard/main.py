
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ZONES = ROOT / "config" / "zones.json"
DEFAULT_OUTPUT = ROOT / "outputs"

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


# ============================================================
# 2. ZONE AND GEOMETRY FUNCTIONS
# ============================================================

def load_config(path: Path) -> dict:
    """Load zone polygons and capacity settings from JSON."""

    if not path.exists():
        raise FileNotFoundError(
            f"Zone configuration file was not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if "zones" not in config or not config["zones"]:
        raise ValueError(
            "zones.json must contain a non-empty 'zones' object."
        )

    return config


def pixel_polygon(points, width: int, height: int) -> np.ndarray:
    """Convert normalized polygon points into image coordinates."""

    polygon = np.array(
        [
            [
                round(float(x) * width),
                round(float(y) * height),
            ]
            for x, y in points
        ],
        dtype=np.int32,
    )

    if len(polygon) < 3:
        raise ValueError(
            "Each zone polygon must contain at least three points."
        )

    return polygon


def point_in_polygon(point, polygon) -> bool:
    """Check whether a point is inside or on a zone boundary."""

    return (
        cv2.pointPolygonTest(
            polygon,
            (float(point[0]), float(point[1])),
            False,
        )
        >= 0
    )


def assign_zone(point, polygons) -> str:
    """Assign a person to the first polygon containing the point."""

    for name, polygon in polygons.items():
        if point_in_polygon(point, polygon):
            return name

    return "UNASSIGNED"


def line_side(point, a, b) -> float:
    """Return which side of a directed line a point is on."""

    return (
        (b[0] - a[0]) * (point[1] - a[1])
        - (b[1] - a[1]) * (point[0] - a[0])
    )


# ============================================================
# 3. MOVEMENT DIRECTION
# ============================================================

def direction_label(
    dx: float,
    dy: float,
    threshold: float = 2.0,
) -> str:
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


# ============================================================
# 4. ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "CrowdGuard AI: video -> YOLO -> ByteTrack -> "
            "zones -> crowd feature CSV"
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to input video, or 0 for webcam.",
    )

    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Ultralytics model weights.",
    )

    parser.add_argument(
        "--zones",
        default=str(DEFAULT_ZONES),
        help="Path to config/zones.json.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Directory for generated outputs.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO detection confidence threshold.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size.",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Feature aggregation interval in seconds.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show a live OpenCV preview.",
    )

    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Skip annotated video generation.",
    )

    args = parser.parse_args()

    if not 0.0 < args.conf <= 1.0:
        parser.error("--conf must be greater than 0 and at most 1.")

    if args.imgsz <= 0:
        parser.error("--imgsz must be positive.")

    if args.interval <= 0:
        parser.error("--interval must be greater than 0.")

    return args


# ============================================================
# 5. MAIN VIDEO PROCESSING PIPELINE
# ============================================================

def main():
    args = parse_args()

    # Accept both local video paths and webcam index 0.
    source = (
        int(args.source)
        if args.source.isdigit()
        else args.source
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(args.zones))
    model = YOLO(args.model)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise SystemExit(
            f"ERROR: Cannot open video source: {args.source}"
        )

    writer = None
    frame_index = 0
    rows = []

    started = time.time()

    try:
        # ----------------------------------------------------
        # 5.1 Read video metadata
        # ----------------------------------------------------

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            raise RuntimeError(
                "Could not read the input video's dimensions."
            )

        polygons = {
            name: pixel_polygon(points, width, height)
            for name, points in config["zones"].items()
        }

        capacities = config.get("capacities", {})
        line_cfg = config.get("flow_line", {})

        line_from = line_cfg.get("from", [0.5, 0.72])
        line_to = line_cfg.get("to", [0.5, 0.98])

        flow_a = (
            int(float(line_from[0]) * width),
            int(float(line_from[1]) * height),
        )

        flow_b = (
            int(float(line_to[0]) * width),
            int(float(line_to[1]) * height),
        )

        if flow_a == flow_b:
            raise ValueError(
                "The virtual flow line must have two different endpoints."
            )

        if isinstance(source, str):
            stem = Path(source).stem
        else:
            stem = "webcam"

        video_out = output_dir / f"{stem}_annotated.mp4"
        csv_out = output_dir / f"{stem}_features.csv"

        # ----------------------------------------------------
        # 5.2 Initialize annotated video writer
        # ----------------------------------------------------

        if not args.no_save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(
                str(video_out),
                fourcc,
                fps,
                (width, height),
            )

            if not writer.isOpened():
                writer.release()
                writer = None

                raise RuntimeError(
                    "OpenCV could not create the annotated video. "
                    f"Attempted output path: {video_out}"
                )

        # ----------------------------------------------------
        # 5.3 Initialize tracking and feature aggregation
        # ----------------------------------------------------

        track_history = defaultdict(
            lambda: deque(maxlen=8)
        )

        previous_line_side = {}
        last_counted_crossing = {}

        total_line_crossings = 0

        interval_frames = max(
            1,
            int(round(fps * args.interval)),
        )

        zone_accum = defaultdict(
            lambda: {
                "count_sum": 0,
                "speed_sum": 0.0,
                "speed_n": 0,
                "directions": defaultdict(int),
                "frames": 0,
            }
        )

        zone_last_count = defaultdict(int)

        print(f"[CrowdGuard] Input: {args.source}")
        print(f"[CrowdGuard] Resolution: {width}x{height}")
        print(f"[CrowdGuard] FPS: {fps:.2f}")
        print(f"[CrowdGuard] Model: {args.model}")
        print(f"[CrowdGuard] Zones: {len(polygons)}")
        print(f"[CrowdGuard] Output directory: {output_dir}")

        # ====================================================
        # 6. FRAME-BY-FRAME PROCESSING
        # ====================================================

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            timestamp = frame_index / fps

            if (
                frame.shape[1] != width
                or frame.shape[0] != height
            ):
                frame = cv2.resize(
                    frame,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )

            # ------------------------------------------------
            # 6.1 YOLO detection + ByteTrack
            # ------------------------------------------------

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )[0]

            annotated = frame.copy()

            # ------------------------------------------------
            # 6.2 Draw configured zones and flow line
            # ------------------------------------------------

            for name, polygon in polygons.items():
                cv2.polylines(
                    annotated,
                    [polygon],
                    isClosed=True,
                    color=(255, 180, 0),
                    thickness=2,
                )

                label_point = tuple(
                    int(v) for v in polygon[0]
                )

                cv2.putText(
                    annotated,
                    name,
                    label_point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 180, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.line(
                annotated,
                flow_a,
                flow_b,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                annotated,
                "FLOW LINE",
                (flow_a[0] + 5, max(20, flow_a[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # ------------------------------------------------
            # 6.3 Per-frame feature containers
            # ------------------------------------------------

            current_by_zone = defaultdict(set)
            current_speed_by_zone = defaultdict(list)
            current_direction_by_zone = defaultdict(list)

            boxes = result.boxes

            if (
                boxes is not None
                and len(boxes) > 0
                and boxes.id is not None
            ):
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.int().cpu().tolist()
                confs = boxes.conf.cpu().numpy()

                for box, track_id, confidence in zip(
                    xyxy,
                    ids,
                    confs,
                ):
                    x1, y1, x2, y2 = map(int, box)

                    # Bottom-center approximates the ground contact point.
                    point = (
                        int((x1 + x2) / 2),
                        int(y2),
                    )

                    zone = assign_zone(point, polygons)

                    hist = track_history[track_id]
                    hist.append(
                        (timestamp, point[0], point[1])
                    )

                    speed_px_s = 0.0
                    direction = "UNKNOWN"

                    if len(hist) >= 2:
                        t0, x0, y0 = hist[0]
                        t1, x1p, y1p = hist[-1]

                        dt = t1 - t0

                        if dt > 0:
                            dx = x1p - x0
                            dy = y1p - y0

                            speed_px_s = (
                                math.hypot(dx, dy) / dt
                            )

                            direction = direction_label(dx, dy)

                    if zone != "UNASSIGNED":
                        current_by_zone[zone].add(track_id)
                        current_speed_by_zone[zone].append(
                            speed_px_s
                        )
                        current_direction_by_zone[zone].append(
                            direction
                        )

                    # ----------------------------------------
                    # Virtual-line crossing count
                    # ----------------------------------------

                    side = line_side(
                        point,
                        flow_a,
                        flow_b,
                    )

                    old_side = previous_line_side.get(track_id)

                    if (
                        old_side is not None
                        and side * old_side < 0
                        and timestamp
                        - last_counted_crossing.get(
                            track_id,
                            -999.0,
                        ) > 1.0
                    ):
                        total_line_crossings += 1
                        last_counted_crossing[track_id] = timestamp

                    previous_line_side[track_id] = side

                    # ----------------------------------------
                    # Draw detection and tracking label
                    # ----------------------------------------

                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 220, 0),
                        2,
                    )

                    cv2.circle(
                        annotated,
                        point,
                        4,
                        (0, 0, 255),
                        -1,
                    )

                    label = (
                        f"ID {track_id} {confidence:.2f} | "
                        f"{zone} | {speed_px_s:.0f}px/s"
                    )

                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 220, 0),
                        1,
                        cv2.LINE_AA,
                    )

            # ------------------------------------------------
            # 6.4 Accumulate zone features
            # ------------------------------------------------

            for zone in polygons:
                count = len(current_by_zone[zone])
                acc = zone_accum[zone]

                acc["count_sum"] += count
                acc["frames"] += 1

                speeds = current_speed_by_zone[zone]

                if speeds:
                    acc["speed_sum"] += float(
                        np.mean(speeds)
                    )
                    acc["speed_n"] += 1

                for direction in current_direction_by_zone[zone]:
                    acc["directions"][direction] += 1

                zone_last_count[zone] = count

            total_people = sum(
                len(person_ids)
                for person_ids in current_by_zone.values()
            )

            # ------------------------------------------------
            # 6.5 Draw dashboard information on the frame
            # ------------------------------------------------

            cv2.rectangle(
                annotated,
                (8, 8),
                (440, 76),
                (20, 20, 20),
                -1,
            )

            cv2.putText(
                annotated,
                f"CrowdGuard AI | People in zones: {total_people}",
                (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated,
                (
                    f"Elapsed: {timestamp:.1f}s | "
                    f"Line crossings: {total_line_crossings}"
                ),
                (18, 59),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 255),
                1,
            )

            # ------------------------------------------------
            # 6.6 Write annotated frame
            # ------------------------------------------------

            if writer is not None:
                writer.write(annotated)

            if args.show:
                cv2.imshow(
                    "CrowdGuard AI - Press q to quit",
                    annotated,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    frame_index += 1
                    break

            frame_index += 1

            # ------------------------------------------------
            # 6.7 Aggregate features at configured intervals
            # ------------------------------------------------

            if frame_index % interval_frames == 0:
                end_time = frame_index / fps

                for zone, polygon in polygons.items():
                    acc = zone_accum[zone]

                    count_avg = (
                        acc["count_sum"]
                        / max(1, acc["frames"])
                    )

                    avg_speed = (
                        acc["speed_sum"]
                        / max(1, acc["speed_n"])
                    )

                    polygon_area_px = max(
                        1.0,
                        abs(cv2.contourArea(polygon)),
                    )

                    # Image-area proxy, not physical density.
                    density_proxy = (
                        count_avg
                        / (polygon_area_px / 10000.0)
                    )

                    capacity = float(
                        capacities.get(zone, 0) or 0
                    )

                    end_count = zone_last_count[zone]

                    occupancy = (
                        end_count / capacity
                        if capacity > 0
                        else None
                    )

                    dominant_direction = (
                        max(
                            acc["directions"],
                            key=acc["directions"].get,
                        )
                        if acc["directions"]
                        else "UNKNOWN"
                    )

                    rows.append(
                        {
                            "timestamp_start_s": round(
                                interval_start_from_rows(rows, zone, end_time, args.interval),
                                3,
                            ),
                            "timestamp_end_s": round(end_time, 3),
                            "zone": zone,
                            "people_count_avg": round(
                                count_avg, 3
                            ),
                            "people_count_end": int(end_count),
                            "image_density_proxy_per_10000_px": round(
                                density_proxy, 4
                            ),
                            "mean_speed_px_per_s": round(
                                avg_speed, 3
                            ),
                            "dominant_direction": dominant_direction,
                            "virtual_line_crossings_total": int(
                                total_line_crossings
                            ),
                            "flow_crossings_per_s_interval": round(
                                total_line_crossings
                                / max(end_time, 0.001),
                                4,
                            ),
                            "configured_capacity": (
                                int(capacity)
                                if capacity
                                else ""
                            ),
                            "occupancy_ratio": (
                                round(occupancy, 4)
                                if occupancy is not None
                                else ""
                            ),
                            "elapsed_video_s": round(
                                end_time, 3
                            ),
                        }
                    )

                    zone_accum[zone] = {
                        "count_sum": 0,
                        "speed_sum": 0.0,
                        "speed_n": 0,
                        "directions": defaultdict(int),
                        "frames": 0,
                    }

                interval_start = end_time

    finally:
        # Always release video resources, even on errors.
        cap.release()

        if writer is not None:
            writer.release()

        if args.show:
            cv2.destroyAllWindows()

    # ========================================================
    # 7. SAVE CSV
    # ========================================================

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(csv_out, index=False)

    # ========================================================
    # 8. VALIDATE VIDEO OUTPUT
    # ========================================================

    print("\n[CrowdGuard] Processing complete.")
    print(f"Frames processed: {frame_index}")
    print(f"Approx video duration: {frame_index / fps:.2f}s")
    print(f"Processing time: {time.time() - started:.2f}s")
    print(f"Feature CSV: {csv_out.resolve()}")

    if not args.no_save_video:
        if not video_out.exists():
            raise RuntimeError(
                f"Annotated video was not created: {video_out}"
            )

        if video_out.stat().st_size < 1024:
            raise RuntimeError(
                f"Annotated video is empty or too small: {video_out}"
            )

        print(f"Annotated video: {video_out.resolve()}")
        print(
            f"Annotated video size: "
            f"{video_out.stat().st_size / (1024 * 1024):.2f} MB"
        )

    print(f"Feature rows generated: {len(df)}")
    print(
        "NOTE: Speed is in pixels/second; density is an "
        "image-area proxy, not people/m²."
    )
    print(
        "NOTE: This prototype extracts monitoring features; "
        "it does not produce validated crowd-risk predictions."
    )


def interval_start_from_rows(
    rows: list[dict],
    zone: str,
    end_time: float,
    interval: float,
) -> float:
    """
    Return the start time for the current feature interval.

    The interval boundary is based on the preceding interval end,
    so all zones in a sample share the same time range.
    """
    del rows, zone
    return max(0.0, end_time - interval)


if __name__ == "__main__":
    main()
