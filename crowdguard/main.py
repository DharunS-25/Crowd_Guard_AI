"""CrowdGuard CLI: video -> YOLO -> ByteTrack -> zones -> crowd feature CSV (+ annotated video).

Thin adapter over `crowdguard.processing.process_video`. Flags, console output, CSV rows, and annotated frames are
unchanged from the original prototype (frozen by tests/core/test_cli_compat.py).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import pandas as pd

from crowdguard.processing import (  # re-exported for backwards compatibility
    CSV_COLUMNS,
    AnalysisInput,
    FrameObservation,
    StartInfo,
    UltralyticsDetector,
    VideoOpenError,
    ZoneSpec,
    assign_zone,
    direction_label,
    line_side,
    load_config,
    pixel_polygon,
    point_in_polygon,
    process_video,
)

__all__ = [
    "CSV_COLUMNS",
    "assign_zone",
    "direction_label",
    "line_side",
    "load_config",
    "main",
    "parse_args",
    "pixel_polygon",
    "point_in_polygon",
]

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ZONES = ROOT / "config" / "zones.json"
DEFAULT_OUTPUT = ROOT / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="CrowdGuard AI: video -> YOLO -> ByteTrack -> zones -> crowd feature CSV"
    )
    parser.add_argument("--source", required=True, help="Path to input video, or 0 for webcam.")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model weights.")
    parser.add_argument("--zones", default=str(DEFAULT_ZONES), help="Path to config/zones.json.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Directory for generated outputs.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--interval", type=float, default=1.0, help="Feature aggregation interval in seconds.")
    parser.add_argument("--show", action="store_true", help="Show a live OpenCV preview.")
    parser.add_argument("--no-save-video", action="store_true", help="Skip annotated video generation.")

    args = parser.parse_args()

    if not 0.0 < args.conf <= 1.0:
        parser.error("--conf must be greater than 0 and at most 1.")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive.")
    if args.interval <= 0:
        parser.error("--interval must be greater than 0.")

    return args


class _LegacyAnnotator:
    """Draws exactly what the original CLI drew, writes the video, and handles the preview window."""

    def __init__(self, args, video_out: Path, source_label: str, output_dir: Path, zone_count: int):
        self.args = args
        self.video_out = video_out
        self.source_label = source_label
        self.output_dir = output_dir
        self.zone_count = zone_count
        self.writer = None
        self.start: StartInfo | None = None

    def on_start(self, info: StartInfo) -> None:
        self.start = info
        if not self.args.no_save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.video_out), fourcc, info.fps, (info.width, info.height))
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise RuntimeError(
                    "OpenCV could not create the annotated video. "
                    f"Attempted output path: {self.video_out}"
                )

        print(f"[CrowdGuard] Input: {self.source_label}")
        print(f"[CrowdGuard] Resolution: {info.width}x{info.height}")
        print(f"[CrowdGuard] FPS: {info.fps:.2f}")
        print(f"[CrowdGuard] Model: {self.args.model}")
        print(f"[CrowdGuard] Zones: {self.zone_count}")
        print(f"[CrowdGuard] Output directory: {self.output_dir}")

    def on_frame(self, obs: FrameObservation) -> bool:
        info = self.start
        annotated = obs.frame.copy()

        for name, polygon in info.polygons.items():
            cv2.polylines(annotated, [polygon], isClosed=True, color=(255, 180, 0), thickness=2)
            label_point = tuple(int(v) for v in polygon[0])
            cv2.putText(annotated, name, label_point, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2, cv2.LINE_AA)

        cv2.line(annotated, info.flow_a, info.flow_b, (0, 255, 255), 2)
        cv2.putText(
            annotated,
            "FLOW LINE",
            (info.flow_a[0] + 5, max(20, info.flow_a[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        for t in obs.tracks:
            point = (int((t.x1 + t.x2) / 2), int(t.y2))
            cv2.rectangle(annotated, (t.x1, t.y1), (t.x2, t.y2), (0, 220, 0), 2)
            cv2.circle(annotated, point, 4, (0, 0, 255), -1)
            label = f"ID {t.track_id} {t.confidence:.2f} | {t.zone} | {t.speed_px_s:.0f}px/s"
            cv2.putText(annotated, label, (t.x1, max(20, t.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)

        cv2.rectangle(annotated, (8, 8), (440, 76), (20, 20, 20), -1)
        cv2.putText(
            annotated,
            f"CrowdGuard AI | People in zones: {obs.total_people_in_zones}",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"Elapsed: {obs.timestamp:.1f}s | Line crossings: {obs.total_line_crossings}",
            (18, 59),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            1,
        )

        if self.writer is not None:
            self.writer.write(annotated)

        if self.args.show:
            cv2.imshow("CrowdGuard AI - Press q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return True
        return False

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        if self.args.show:
            cv2.destroyAllWindows()


def main():
    args = parse_args()

    # Accept both local video paths and webcam index 0.
    source = int(args.source) if args.source.isdigit() else args.source

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(args.zones))
    detector = UltralyticsDetector(args.model, conf=args.conf, imgsz=args.imgsz)

    capacities = config.get("capacities", {})
    line_cfg = config.get("flow_line", {})
    flow_line = (line_cfg.get("from", [0.5, 0.72]), line_cfg.get("to", [0.5, 0.98]))
    # Legacy CLI semantics: demonstration capacities from zones.json are used as-is (product path requires a source).
    zones = [ZoneSpec(zone_id=name, points=points, capacity=capacities.get(name)) for name, points in config["zones"].items()]

    stem = Path(source).stem if isinstance(source, str) else "webcam"
    video_out = output_dir / f"{stem}_annotated.mp4"
    csv_out = output_dir / f"{stem}_features.csv"

    annotator = _LegacyAnnotator(args, video_out, args.source, output_dir, len(zones))
    started = time.time()

    try:
        result = process_video(
            AnalysisInput(source=source, zones=zones, interval_s=args.interval, flow_line=flow_line),
            detector,
            on_start=annotator.on_start,
            on_frame=annotator.on_frame,
        )
    except VideoOpenError as exc:
        if "Cannot open video source" in str(exc):
            raise SystemExit(f"ERROR: Cannot open video source: {args.source}") from exc
        raise RuntimeError(str(exc)) from exc
    finally:
        annotator.close()

    df = pd.DataFrame(result.legacy_rows, columns=CSV_COLUMNS)
    df.to_csv(csv_out, index=False)

    fps = result.meta.fps
    frame_index = result.meta.frames_processed
    print("\n[CrowdGuard] Processing complete.")
    print(f"Frames processed: {frame_index}")
    print(f"Approx video duration: {frame_index / fps:.2f}s")
    print(f"Processing time: {time.time() - started:.2f}s")
    print(f"Feature CSV: {csv_out.resolve()}")

    if not args.no_save_video:
        if not video_out.exists():
            raise RuntimeError(f"Annotated video was not created: {video_out}")
        if video_out.stat().st_size < 1024:
            raise RuntimeError(f"Annotated video is empty or too small: {video_out}")
        print(f"Annotated video: {video_out.resolve()}")
        print(f"Annotated video size: {video_out.stat().st_size / (1024 * 1024):.2f} MB")

    print(f"Feature rows generated: {len(df)}")
    print("NOTE: Speed is in pixels/second; density is an image-area proxy, not people/m².")
    print("NOTE: This prototype extracts monitoring features; it does not produce validated crowd-risk predictions.")


if __name__ == "__main__":
    main()
