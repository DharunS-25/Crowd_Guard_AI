# CrowdGuard AI: Claude Code instructions

This is the project entry point. Keep it short. Detailed rules live in `.claude/rules/` (path-scoped) and in the docs below. Tests and runtime validation enforce behavior; these files are context.

## Read before changing behavior

- [System architecture](docs/architecture/system-architecture.md): current vs proposed, components, job lifecycle, result contracts, security, deferred phases.
- [Monorepo structure](docs/architecture/repository-structure.md): layout, import names, dependency rules.
- [S3 direct upload](docs/architecture/s3-upload.md): upload protocol, verification, bucket controls.
- [Risk and model boundary](docs/architecture/risk-assessment.md): labels, adapter and evidence contract, fusion policy, unknown states, open decisions D-01 to D-19.
- [Implementation plan](docs/superpowers/plans/2026-11-26-crowdguard-product-architecture.md): phased tasks with allowed-file lists.
- [E2E cases](docs/testing/e2e-test-cases.md): acceptance scenarios and harness.

## Product goal

An operator uploads one authorized venue video (browser directly to private S3), selects zones, and starts the analysis. They get per-zone feature windows and risk assessments, a playable video with matching overlays, and downloadable JSON/CSV with timestamps, model evidence, and data quality. The MVP is single video, single camera, batch. Multi-camera, RTSP, alerts, auth for internet exposure, and evacuation planning are later phases.

## Repository facts (verified 2026-11-26)

- `app.py`: Streamlit prototype; runs `crowdguard/main.py` as a subprocess.
- `crowdguard/main.py`: CLI with OpenCV, YOLO, ByteTrack, polygon zones, per-zone feature CSV, and annotated MP4.
- `model_training/`: GRU/LSTM experiments on UCSD **binary Normal/Anomalous** labels with frame-level optical-flow features. They are not venue-risk models, and their schema does not match the pipeline CSV.
- Not implemented yet: `apps/`, `packages/`, FastAPI, React/Vite, S3 upload, jobs, adapters, fusion, risk overlays. Do not describe them as implemented.

## Non-negotiable behavior

- Never map `Normal`/`Anomalous` to `LOW`/`HIGH`. Never average or vote across models without a shared, validated label space and calibration. Fusion is a separate, versioned policy.
- Never invent density, capacity, speed, or risk thresholds. Image density and pixel speed are proxies.
- Missing, invalid, stale, ineligible, or unconfigured evidence gives `INSUFFICIENT_DATA`, never `LOW`.
- Every result names its zone, window, data quality, model and policy versions, and evidence. The overlay, API, JSON, CSV, and UI agree.
- Uploaded media, client metadata, and model files are untrusted. Use generated keys, verify with S3, probe the media, and keep the bucket private. Never commit footage, weights, databases, or credentials.
- Keep `app.py`, the CLI, `model_training/`, and `config/` working and in place until a reviewed migration replaces them.

## Workflow

- Read the matching `.claude/rules/` file before editing a code area. `agent-boundaries.md` always applies.
- Do one plan task at a time, within its allowed files. Stop and report before breaking or out-of-scope changes.
- Add tests with behavior changes. Use deterministic fakes. No weight downloads, AWS, or live camera in CI.
- Run only the relevant checks and report exact commands and outcomes. Never claim an unrun check passed.
- Update the docs and plan checkboxes when a contract or boundary changes. Verify a command exists before documenting it as runnable (see `README.md`, `model_training/README.md`).

Project memory and path-scoped rules follow the [Claude Code memory docs](https://code.claude.com/docs/en/memory).
