# CrowdGuard AI — Phase 4 to Phase 8 prototype

This project implements the first computer-vision milestone:

**MP4/webcam → OpenCV → YOLO person detection → ByteTrack IDs → venue zones → crowd-feature CSV + annotated video**

It does not yet include the GRU/LSTM risk predictor, FastAPI service, database, or React dashboard. Those should be added after validating the extracted features.

## Repository layout

The repository is moving to a monorepo layout. Today only the prototype exists. Everything under `apps/`, `packages/`, `infra/`, and `tests/` is **planned, not yet created**.

| Path | Status | Purpose |
|---|---|---|
| `app.py` | Current | Streamlit demo dashboard (section "Single dashboard mode") |
| `crowdguard/` | Current | Video pipeline CLI (`python -m crowdguard.main`); becomes the core vision and model library |
| `config/` | Current | Example zone polygons with demonstration capacities |
| `model_training/` | Current | GRU/LSTM experiments on UCSD Normal/Anomalous labels (see `model_training/README.md`) |
| `apps/api`, `apps/worker`, `apps/web` | Planned | FastAPI service, analysis worker, React + Vite UI |
| `packages/contracts`, `packages/platform`, `packages/api-client` | Planned | Shared data contracts, SQLite and S3 access, generated TypeScript client |
| `infra/s3`, `tests/` | Planned | S3 bucket templates; core, E2E, and fixture tests |
| `docs/` | Current | Architecture, risk model limits, E2E cases, implementation plan |

The commands in this README cover only the current prototype. They keep working while the monorepo is built: existing folders stay in place, and new code goes into new folders.

More detail:

- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md): commented directory map, current and planned.
- [docs/architecture/repository-structure.md](docs/architecture/repository-structure.md): monorepo layout and dependency rules.
- [docs/architecture/system-architecture.md](docs/architecture/system-architecture.md): product architecture.
- [CLAUDE.md](CLAUDE.md): Claude Code instructions; path-scoped rules are in `.claude/rules/`.

## 1. Requirements

- Windows 10/11, macOS, or Linux
- Python 3.10 or 3.11 recommended
- Internet for the first installation/model-weight download
- 8 GB RAM recommended; a compatible NVIDIA GPU can accelerate inference, but CPU can be used for a small demo

## 2. Setup on Windows 11

Open PowerShell in this project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this in the current PowerShell window and retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

If `py -3.11` is unavailable, install Python 3.11 first and ensure the Python launcher is enabled.

## 3. Get a sample crowd video

Use a video you are permitted to download and process. Good starting points:

- Pexels crowd videos: https://www.pexels.com/search/videos/crowd/
- Pixabay crowd videos: https://pixabay.com/videos/search/crowd/
- UCSD Anomaly Detection Dataset (research benchmark; follow its access and usage terms): http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm
- PETS surveillance benchmark information: https://www.cvg.reading.ac.uk/PETS2009/a.html

Download a short MP4 (10–60 seconds is enough for the first run) and place it at:

`data/videos/crowd.mp4`

Do not assume a video downloaded from a website is automatically licensed for redistribution. Keep attribution/licence details for your project report. UCSD anomaly clips are useful for anomaly research but are not a substitute for a varied venue-specific dataset.

## 4. Run the pipeline

From the project root:

```powershell
python -m crowdguard.main --source data/videos/crowd.mp4 --show
```

The first run may download the YOLO weights automatically. To run without a preview window:

```powershell
python -m crowdguard.main --source data/videos/crowd.mp4
```

Outputs are written to `outputs/`:
- `crowd_annotated.mp4` — boxes, track IDs, zones, and a virtual line
- `crowd_features.csv` — zone-level metrics sampled approximately once per second

Use a webcam instead of a file:

```powershell
python -m crowdguard.main --source 0 --show --no-save-video
```

Stop preview with `q`. A webcam can be slow on CPU.

## 5. Zone configuration

Edit `config/zones.json`. Polygon coordinates are normalized from 0 to 1:
- `[0, 0]` is the top-left of the image
- `[1, 1]` is the bottom-right

The included zones are only an illustration. They will not fit every camera. Open a representative frame from your video and adjust each polygon so it matches the actual stage, main crowd, entry, and exit areas. Keep polygons non-overlapping where practical. If the camera doesn't show a stage or entrance, remove or redefine that zone.

`capacities` are demonstration settings. Replace them with documented venue capacities. `flow_line` is a virtual line used to count track-side changes; orient and position it to match a real doorway or passage.

## 6. CSV columns

- `people_count_avg`: average tracked people assigned to the zone during the interval
- `people_count_end`: tracked people in the zone at the end of the interval
- `image_density_proxy_per_10000_px`: people count normalized by polygon image area; not physical density
- `mean_speed_px_per_s`: mean approximate track speed in pixels per second
- `dominant_direction`: dominant coarse direction of tracked motion in the interval
- `virtual_line_crossings_total`: cumulative line crossings detected in the video
- `flow_crossings_per_s_interval`: cumulative crossings divided by elapsed video seconds; a simple baseline, not exact interval flow
- `occupancy_ratio`: count divided by configured capacity, when a capacity is set

The initial implementation uses a bottom-centre point of each bounding box to assign a person to a zone. IDs may change or disappear during occlusion. Treat measurements as estimates.

## 7. Important limitations for your examination

1. **Physical density:** `people/m²` requires a calibrated camera view or a perspective/ground-plane mapping. The image-area proxy is not physically comparable across different cameras.
2. **Speed:** pixels/second is not metres/second. Calibrate the scene before reporting physical speed.
3. **Flow:** the sample code's virtual line is a baseline. For real entrance/exit flow, use a properly placed line and count directional crossings in a defined interval.
4. **Safety:** do not call the prototype a validated crowd-crush predictor. It extracts features only. A GRU/LSTM risk model needs labelled temporal sequences, evaluation, and validation in representative conditions.
5. **Privacy:** use authorized footage, restrict access, and avoid identifying people. This prototype uses temporary tracker IDs, not facial recognition.

## 8. Troubleshooting

- `Cannot open video source`: verify the path and that the file plays in a media player.
- Missing module: activate `.venv` and run `pip install -r requirements.txt`.
- `ModuleNotFoundError: imageio_ffmpeg` when starting the dashboard: `app.py` imports it, but `requirements.txt` does not list it yet. Run `pip install imageio-ffmpeg`.
- Slow processing: use a shorter/lower-resolution clip, reduce `--imgsz` to 480, and close other applications.
- No IDs: make sure tracking is enabled, use a video rather than isolated images, and inspect whether people are visible enough for detection.
- Wrong zones: edit `config/zones.json` for your camera's actual view.
- MP4 writer issue: try a different output directory or use `--no-save-video`.

## 9. Next phases

The product roadmap is in the [implementation plan](docs/superpowers/plans/2026-11-26-crowdguard-product-architecture.md). In order:

1. Monorepo scaffold.
2. Shared contracts.
3. The upload-to-annotated-result slice (direct browser-to-S3 upload, FastAPI, worker, React UI).
4. Experimental model evidence.
5. A validated venue-risk model.
6. Hardening before any internet-facing deployment.

Multi-camera support and evacuation planning are later, separately reviewed phases. The current GRU/LSTM models predict UCSD Normal/Anomalous labels, not Low/Medium/High venue risk. See [risk-assessment.md](docs/architecture/risk-assessment.md).


## Single dashboard mode (Streamlit)

Install the updated requirements:

```powershell
pip install -r requirements.txt
```

Run the dashboard from the project root:

```powershell
streamlit run app.py
```

Streamlit opens a browser tab, usually at `http://localhost:8501`. If it does not open automatically, paste that address into your browser. Upload an MP4 in the sidebar, choose model settings, and click **Process video**. The dashboard then displays the annotated video, latest zone counts, zone metrics, feature trends, and CSV/video download controls.

The dashboard calls the existing processing pipeline as a local subprocess. The first run may download YOLO model weights. It is a local demo dashboard, not the planned React + FastAPI product (`apps/web`, `apps/api`). It stays available until a reviewed migration replaces it.
