
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import imageio_ffmpeg


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "data" / "videos"
OUTPUT_DIR = ROOT / "outputs"

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. VIDEO CONVERSION
# --------------------------------------------------

def convert_to_browser_mp4(video_path: Path) -> tuple[bool, str]:
    """
    Convert an OpenCV-generated video to H.264 MP4
    for reliable browser playback.
    """

    if not video_path.exists():
        return False, f"Video not found: {video_path}"

    if video_path.stat().st_size < 1024:
        return False, (
            f"Video is empty or too small: {video_path}\n"
            "Check the VideoWriter and frame-writing loop in main.py."
        )

    converted_path = video_path.with_name(
        f"{video_path.stem}_browser.mp4"
    )

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i", str(video_path),
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(converted_path),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=None,
        )

        if (
            result.returncode != 0
            or not converted_path.exists()
            or converted_path.stat().st_size < 1024
        ):
            error = result.stderr or result.stdout or "Unknown FFmpeg error"

            if converted_path.exists():
                converted_path.unlink()

            return False, error[-5000:]

        # Replace the original only after conversion succeeds.
        converted_path.replace(video_path)

        return True, "Video converted successfully to H.264 MP4."

    except Exception as exc:
        if converted_path.exists():
            converted_path.unlink()

        return False, str(exc)


# --------------------------------------------------
# 3. STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CrowdGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }

    .cg-subtitle {
        color: #9CA3AF;
        margin-top: -12px;
    }

    .cg-note {
        padding: 12px 14px;
        border-radius: 10px;
        background: rgba(59,130,246,.10);
        border: 1px solid rgba(59,130,246,.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ CrowdGuard AI")

st.markdown(
    '<p class="cg-subtitle">'
    "Crowd monitoring · Person tracking · Zone-level feature extraction"
    "</p>",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 4. SIDEBAR CONTROLS
# --------------------------------------------------

with st.sidebar:
    st.header("Video Analysis")

    uploaded = st.file_uploader(
        "Upload a crowd video",
        type=["mp4", "avi", "mov", "mkv"],
    )

    model_name = st.selectbox(
        "YOLO model",
        ["yolo11n.pt", "yolov8n.pt"],
        index=0,
    )

    conf = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )

    imgsz = st.select_slider(
        "Inference image size",
        options=[320, 480, 640, 800],
        value=640,
    )

    interval = st.select_slider(
        "Feature interval (seconds)",
        options=[1, 2, 3, 5],
        value=1,
    )

    run = st.button(
        "▶ Process video",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.caption(
        "Prototype: outputs include estimated crowd features. "
        "Validated risk prediction is not implemented."
    )


# --------------------------------------------------
# 5. SESSION STATE
# --------------------------------------------------

if "last_video" not in st.session_state:
    st.session_state.last_video = None

if "last_csv" not in st.session_state:
    st.session_state.last_csv = None

if "last_log" not in st.session_state:
    st.session_state.last_log = ""

if "video_message" not in st.session_state:
    st.session_state.video_message = ""


# --------------------------------------------------
# 6. PROCESS THE UPLOADED VIDEO
# --------------------------------------------------

if run:
    # Clear previous results so old output files cannot
    # accidentally appear as results for the new upload.
    st.session_state.last_video = None
    st.session_state.last_csv = None
    st.session_state.last_log = ""
    st.session_state.video_message = ""

    if uploaded is None:
        st.error("Please upload a video first.")

    else:
        safe_name = Path(uploaded.name).name
        stem = Path(safe_name).stem

        input_path = VIDEO_DIR / safe_name
        output_video = OUTPUT_DIR / f"{stem}_annotated.mp4"
        output_csv = OUTPUT_DIR / f"{stem}_features.csv"

        # Save uploaded video.
        input_path.write_bytes(uploaded.getbuffer())

        # Remove stale results from previous runs.
        for old_file in [output_video, output_csv]:
            if old_file.exists():
                old_file.unlink()

        command = [
            sys.executable,
            "-m",
            "crowdguard.main",
            "--source",
            str(input_path),
            "--model",
            model_name,
            "--conf",
            str(conf),
            "--imgsz",
            str(imgsz),
            "--interval",
            str(interval),
            "--output-dir",
            str(OUTPUT_DIR),
        ]

        with st.spinner(
            "Processing video with YOLO and ByteTrack. "
            "The first run may download model weights..."
        ):
            try:
                proc = subprocess.run(
                    command,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=None,
                )

                st.session_state.last_log = (
                    (proc.stdout or "")
                    + (
                        "\n" + proc.stderr
                        if proc.stderr
                        else ""
                    )
                )

                if proc.returncode != 0:
                    st.error("Video processing failed.")
                    st.code(
                        st.session_state.last_log[-12000:]
                        or "No terminal output was returned."
                    )

                elif not output_csv.exists():
                    st.error(
                        "Processing finished, but the expected feature "
                        "CSV was not found."
                    )
                    st.code(
                        st.session_state.last_log[-12000:]
                        or "Check the output paths in crowdguard/main.py."
                    )

                else:
                    st.session_state.last_csv = str(output_csv)

                    if output_video.exists():
                        with st.spinner(
                            "Converting annotated video for browser playback..."
                        ):
                            converted, message = convert_to_browser_mp4(
                                output_video
                            )

                        if converted:
                            st.session_state.last_video = str(output_video)
                            st.session_state.video_message = message
                            st.success(
                                "Video processing completed. "
                                "Annotated video is ready."
                            )
                        else:
                            # Keep the original file available for download
                            # and diagnostics, even if conversion failed.
                            st.session_state.last_video = (
                                str(output_video)
                                if output_video.exists()
                                else None
                            )
                            st.session_state.video_message = message

                            st.warning(
                                "Feature extraction completed, but the "
                                "video could not be converted for reliable "
                                "browser playback."
                            )
                            st.code(message[-5000:])

                    else:
                        st.session_state.video_message = (
                            "The feature CSV exists, but the annotated "
                            "video was not created."
                        )
                        st.warning(st.session_state.video_message)
                        st.code(st.session_state.last_log[-8000:])

            except Exception as exc:
                st.error(f"Could not run the processing script: {exc}")


# --------------------------------------------------
# 7. LOAD THE LATEST RESULTS
# --------------------------------------------------

csv_path = (
    Path(st.session_state.last_csv)
    if st.session_state.last_csv
    else None
)

video_path = (
    Path(st.session_state.last_video)
    if st.session_state.last_video
    else None
)


# --------------------------------------------------
# 8. DASHBOARD RESULTS
# --------------------------------------------------

if csv_path and csv_path.exists():

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        st.error(f"Could not read the feature CSV: {exc}")
        df = pd.DataFrame()

    st.caption(
        f"Results for: {csv_path.stem.replace('_features', '')}"
    )

    if df.empty:
        st.warning(
            "No complete feature interval was generated. "
            "Try a longer video or check the processing log."
        )

    else:
        # ------------------------------------------
        # SUMMARY METRICS
        # ------------------------------------------

        if "zone" in df.columns:
            total_zones = int(df["zone"].nunique())

            if "timestamp_end_s" in df.columns:
                latest = (
                    df.sort_values("timestamp_end_s")
                    .groupby("zone", as_index=False)
                    .tail(1)
                )
                latest_time = float(
                    df["timestamp_end_s"].max()
                )
            else:
                latest = df.groupby(
                    "zone", as_index=False
                ).tail(1)
                latest_time = 0.0
        else:
            total_zones = 0
            latest = df
            latest_time = 0.0

        total_people = (
            int(latest["people_count_end"].fillna(0).sum())
            if "people_count_end" in latest.columns
            else 0
        )

        occupied = (
            int(
                (
                    pd.to_numeric(
                        latest["occupancy_ratio"],
                        errors="coerce",
                    ).fillna(0) >= 0.8
                ).sum()
            )
            if "occupancy_ratio" in latest.columns
            else 0
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "People in configured zones",
            total_people,
        )
        c2.metric(
            "Zones analysed",
            total_zones,
        )
        c3.metric(
            "Video time processed",
            f"{latest_time:.1f} s",
        )
        c4.metric(
            "Zones ≥80% configured capacity",
            occupied,
        )

        # ------------------------------------------
        # VIDEO AND ZONE ANALYSIS
        # ------------------------------------------

        st.subheader("Video and zone analysis")

        left, right = st.columns([1.35, 1])

        with left:
            st.markdown("**Annotated video**")

            if video_path and video_path.exists():
                video_size = video_path.stat().st_size

                if video_size < 1024:
                    st.error(
                        "The annotated video file is too small "
                        "to play. Check main.py's video writer."
                    )

                else:
                    st.caption(
                        f"File: {video_path.name} · "
                        f"Size: {video_size / (1024 * 1024):.2f} MB"
                    )

                    # Pass the actual file bytes to Streamlit.
                    st.video(
                        video_path.read_bytes(),
                        format="video/mp4",
                    )

                    st.download_button(
                        "⬇ Download annotated video",
                        data=video_path.read_bytes(),
                        file_name=video_path.name,
                        mime="video/mp4",
                        key="download_annotated_video",
                    )

                    if st.session_state.video_message:
                        st.caption(
                            st.session_state.video_message
                        )

            else:
                st.warning(
                    "Annotated video was not found. "
                    "Check the processing log below."
                )

        with right:
            st.markdown("**Latest people count by zone**")

            if (
                "people_count_end" in latest.columns
                and "zone" in latest.columns
            ):
                chart_df = latest.set_index("zone")[
                    ["people_count_end"]
                ]
                st.bar_chart(chart_df)

            st.markdown("**Latest zone statistics**")

            cols = [
                col
                for col in [
                    "zone",
                    "people_count_end",
                    "image_density_proxy_per_10000_px",
                    "mean_speed_px_per_s",
                    "dominant_direction",
                    "occupancy_ratio",
                    "virtual_line_crossings_total",
                ]
                if col in latest.columns
            ]

            if cols:
                st.dataframe(
                    latest[cols].reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                )

        # ------------------------------------------
        # CROWD FEATURES OVER TIME
        # ------------------------------------------

        st.subheader("Crowd features over time")

        if (
            "timestamp_end_s" in df.columns
            and "zone" in df.columns
        ):
            zones = sorted(
                df["zone"].dropna().unique().tolist()
            )

            if zones:
                selected_zone = st.selectbox(
                    "Choose a zone",
                    zones,
                )

                zdf = df[
                    df["zone"] == selected_zone
                ].sort_values("timestamp_end_s")

                feature_options = [
                    col
                    for col in [
                        "people_count_end",
                        "people_count_avg",
                        "mean_speed_px_per_s",
                        "image_density_proxy_per_10000_px",
                        "occupancy_ratio",
                    ]
                    if col in zdf.columns
                ]

                if feature_options:
                    selected_feature = st.selectbox(
                        "Feature to plot",
                        feature_options,
                    )

                    plot_df = zdf[
                        ["timestamp_end_s", selected_feature]
                    ].copy()

                    plot_df[selected_feature] = pd.to_numeric(
                        plot_df[selected_feature],
                        errors="coerce",
                    )

                    plot_df = plot_df.dropna().set_index(
                        "timestamp_end_s"
                    )

                    if not plot_df.empty:
                        st.line_chart(plot_df)
                    else:
                        st.info(
                            "No numeric data is available for this feature."
                        )

        # ------------------------------------------
        # FEATURE DATASET AND CSV DOWNLOAD
        # ------------------------------------------

        st.subheader("Feature dataset")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇ Download feature CSV",
            data=csv_path.read_bytes(),
            file_name=csv_path.name,
            mime="text/csv",
            key="download_feature_csv",
        )

        # ------------------------------------------
        # INTERPRETATION NOTE
        # ------------------------------------------

        st.markdown(
            """
            <div class="cg-note">
            <b>Interpretation note:</b>
            Speed is measured in pixels/second. Density is an
            image-area proxy, not people per square metre.
            This prototype extracts monitoring features but does
            not yet provide validated crowd-risk predictions.
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# 9. EMPTY DASHBOARD
# --------------------------------------------------

else:
    st.info(
        "Upload a short crowd video in the sidebar and select "
        "**Process video** to populate the dashboard."
    )

    st.subheader("What this dashboard demonstrates")

    a, b, c = st.columns(3)

    a.markdown(
        """
        ### 🎥 Video pipeline
        OpenCV reads video frames and YOLO detects people.
        """
    )

    b.markdown(
        """
        ### 🧭 Tracking and zones
        ByteTrack assigns temporary tracking IDs, and zone
        polygons are configured in `config/zones.json`.
        """
    )

    c.markdown(
        """
        ### 📊 Crowd features
        The pipeline exports people counts, movement estimates,
        occupancy, and virtual-line crossing metrics.
        """
    )

    st.markdown(
        """
        <div class="cg-note">
        The included zone polygons are examples. Adjust them to
        fit your camera view before interpreting the counts.
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("Processing log / troubleshooting"):
    st.code(
        st.session_state.last_log[-12000:]
        or "No processing log yet."
    )
