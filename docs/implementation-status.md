# CrowdGuard AI: implementation status

**As of:** 2026-11-26. Work stopped at your request partway through plan Task 3.
**Branch:** `feature-branch`. **Nothing has been committed or pushed.** Every change below is in the working tree.
**Plan:** [2026-11-26-crowdguard-product-architecture.md](superpowers/plans/2026-11-26-crowdguard-product-architecture.md)

## 1. Summary

| Area | State |
|---|---|
| Architecture and documentation | Done: architecture docs, S3 upload design, risk/model contract, monorepo layout, E2E cases, plan, Claude Code rules, README, PROJECT_STRUCTURE |
| Task 0.1: provisional decisions | Done |
| Task 0.2: S3 behavior spike | Not started |
| Task 0.3: monorepo scaffold | Done (the `apps/web` template moved to Task 8) |
| Task 1: shared contracts | Done, 11 tests pass |
| Task 2: processing service and CLI refactor | Done, 13 tests pass, CLI output identical to the original |
| Task 3: model adapters, policy, overlay | **In progress:** tests, fakes, and policy files written; implementation modules **not written yet** |
| Tasks 4–10 (platform, upload API, worker, results API, web UI, E2E, infra) | Not started |
| Phases 2–6 | Not started (by design) |

## 2. Documentation and rules

| File | What it contains |
|---|---|
| `CLAUDE.md` | Short project entry point: doc index, repository facts, non-negotiable rules, workflow |
| `.claude/rules/architecture.md` | Always loaded: monorepo boundaries, storage, deferred features |
| `.claude/rules/agent-boundaries.md` (new) | Always loaded: one task at a time, allowed files, stop conditions |
| `.claude/rules/backend.md` | API, worker, platform, core rules (paths updated to the monorepo) |
| `.claude/rules/upload-and-storage.md` (new) | S3 direct-upload rules |
| `.claude/rules/ml-and-risk.md` | Adapter, evidence, fusion, and label rules |
| `.claude/rules/frontend-and-e2e.md` | Web UI and E2E rules |
| `docs/architecture/system-architecture.md` | Current vs proposed (tagged), diagrams, job lifecycle, result contracts, API, storage, security, deferred multi-camera and evacuation phases |
| `docs/architecture/repository-structure.md` (new) | Monorepo layout, package import names, one-way dependency rules, tooling, deferred migration |
| `docs/architecture/s3-upload.md` (new) | Direct browser-to-private-S3 upload, multipart, verification, idempotent completion, abort, CORS, IAM, lifecycle |
| `docs/architecture/risk-assessment.md` | Labels and states, model adapter contract, UCSD model limits, fusion policy, reason codes, validation gates, decisions D-01 to D-19, plus §11.1 provisional defaults |
| `docs/testing/e2e-test-cases.md` | 54 acceptance cases (UP, PR, MD, OV, DL, IS, CL), each mapped to a plan task |
| `docs/superpowers/plans/2026-11-26-crowdguard-product-architecture.md` | Phased plan with allowed-file lists per task and a coverage table |
| `docs/prompts/crowdguard-implementation-prompt.md` | Template for giving one bounded task to an agent |
| `README.md` | New "Repository layout" section (current vs planned), updated "Next phases", `imageio-ffmpeg` troubleshooting note |
| `PROJECT_STRUCTURE.md` | Current layout plus the planned monorepo tree and dependency direction |

## 3. Code written

### Task 0.3: monorepo scaffold

- Root `pyproject.toml`: the `crowdguard` core project, pytest config (`--import-mode=importlib`, `pythonpath`, `testpaths`), and five import-linter boundary contracts.
- `packages/contracts`, `packages/platform`, `apps/api`, `apps/worker`: a `pyproject.toml` and an empty package each (`crowdguard_contracts`, `crowdguard_platform`, `crowdguard_api`, `crowdguard_worker`).
- `package.json` and `pnpm-workspace.yaml` for the future `apps/web` and `packages/api-client`.
- `tests/core/test_import_boundaries.py`: the API must import without the core, torch, or OpenCV; contracts need only pydantic; no product code imports `model_training`.
- `.gitignore`: appended `node_modules/`, web and api-client `dist/`, Playwright output, `*.egg-info/`.

### Task 1: shared contracts (`packages/contracts/crowdguard_contracts/`)

| Module | Contents |
|---|---|
| `features.py` | `FeatureSchema`, `ZONE_WINDOW_FEATURES_V1` (pipeline columns with proxy units), `Window` (start < end), `DataQuality`, `ZoneWindowFeatures` (values must follow schema order) |
| `assessment.py` | `RiskLevel` (`LOW`, `MEDIUM`, `HIGH`, `INSUFFICIENT_DATA`), `AssessmentStatus`, `ReasonCode`, `WarningCode`, `RuleSignal`, `ZoneRiskAssessment` (`INSUFFICIENT_DATA` requires a reason) |
| `evidence.py` | `InferenceStatus`, `ModelDescriptor`, `ModelEvidence` (probabilities only when `OK`, keys equal `class_labels`, values sum to 1) |
| `errors.py` | Stable `ErrorCode` catalog and `ApiError` |
| `analysis.py` | Analysis states and stages, status response, start request, `ZoneDefinition` (normalized range, non-zero area, no self-intersection, capacity requires a source), `ZoneConfigCreate` (unique zone IDs) |
| `upload.py` | Upload-session request and response models; attestation required; SHA-256 part checksums; no S3 key, bucket, or UploadId fields |
| `results.py` | Artifact names, `ZoneWindowResult`, `ResultsResponse`, `OverlayManifest` |

### Task 2: processing service (`crowdguard/`)

- `crowdguard/processing.py` (new): `process_video(request, detector, progress, is_cancelled, on_start, on_frame)`.
  - Injectable `Detector`; `UltralyticsDetector` imports Ultralytics lazily.
  - Frame-aligned windows. Window `k` covers frames `[k·N, (k+1)·N)`, and tail frames get no window.
  - Per-window data quality: `NO_DETECTIONS`, `CAPACITY_UNCONFIGURED`, `FPS_FALLBACK_USED`, `FRAMES_DROPPED`.
  - `occupancy_ratio` is `null` without a sourced capacity.
  - Per-frame track records for the later render pass; cancellation and progress callbacks.
- `crowdguard/feature_schema.py` (new): maps legacy CSV rows to schema-ordered values (blank becomes `None`, never 0).
- `crowdguard/main.py` (refactored): now a thin CLI over `process_video`. Flags, console output, CSV rows, and drawing are unchanged, and the old helper names are re-exported.

### Task 3: partial (tests only)

- **Written:**
  - `tests/core/test_adapters.py`, `test_policy.py`, `test_overlay.py`;
  - fakes in `tests/fixtures/models/` (venue-risk, binary-anomaly, raising, missing-artifact, schema-mismatch, unknown-schema) and a window builder;
  - test policies in `tests/fixtures/policies/` (`demo`, `demo_no_disagreement_rule`, `demo_rules` with synthetic sourced thresholds, and `maps_ucsd`, which must be refused);
  - the shipped default policy `crowdguard/risk/policies/crowdguard-default.yaml` (no accepted label space, no thresholds, so it always yields `INSUFFICIENT_DATA`).
- **Not written:**
  - `crowdguard/models/{__init__,base,registry,orchestrator}.py`
  - `crowdguard/risk/{__init__,policy,rules}.py`
  - `crowdguard/overlay.py`

  Until they exist, these three test files fail at import, as the RED step of TDD expects.

### Test fixtures

- `tests/fixtures/synthetic.py`: generates small deterministic videos at test time (never committed), a fake `ultralytics` module (YOLO + ByteTrack stand-in), and a `FakeDetector`.
- `tests/fixtures/legacy_cli_main.py`: an exact copy of the original `crowdguard/main.py`, used only by the golden test.

## 4. Verification (commands actually run)

All commands ran in the project `.venv` from the repository root.

| Command | Result |
|---|---|
| `.venv/Scripts/python -m pytest packages/contracts/tests tests/core/test_import_boundaries.py -q` | **14 passed** |
| `PYTHONPATH=".;packages/contracts;packages/platform;apps/api;apps/worker" .venv/Scripts/lint-imports` | **5 contracts kept, 0 broken** |
| `.venv/Scripts/python -m pytest tests/core/test_processing_service.py tests/core/test_cli_compat.py -q` | **13 passed**. The golden test shows the refactored CLI gives identical CSV rows and pixel-identical annotated frames to the original, for default settings, `--interval 0.5`, and `--no-save-video`. |
| `.venv/Scripts/python -m pytest tests/core/test_adapters.py tests/core/test_policy.py tests/core/test_overlay.py -q` | **3 collection errors (expected).** The Task 3 implementation is not written yet. |

**Not run or not verified:**

- A real `python -m crowdguard.main` run with YOLO weights: `ultralytics` and `torch` are not installed in either Python environment.
- `streamlit run app.py`: Streamlit is not installed in the venv.
- `model_training` tests.
- Anything against real AWS.

A plain `pytest` over the whole repo currently fails at collection because of the three Task 3 test files.

## 5. Decisions I made on your behalf

These are provisional and recorded in `docs/architecture/risk-assessment.md` §11.1. Each can be changed through configuration or by swapping a component.

| Decision | Choice | Cost if wrong |
|---|---|---|
| D-07 overlay timing | Show the window that contains the frame time | Change in `overlay.py` |
| D-12 upload limits | 2 GiB, 600 s, max 3840×2160, MP4/MOV/AVI; per-owner quotas 3 sessions / 5 analyses, 20 creates per minute | Configuration values |
| D-15 media probe | OpenCV plus a full decode; `imageio-ffmpeg` for H.264 output (no ffprobe) | Swap the probe module |
| D-16 S3 test endpoint | moto (no Docker needed) | Swap the test endpoint |
| D-18 authorization | Upload requires `authorization_attested: true` | Contract field |
| D-19 tooling | pip plus a `pyproject.toml` per unit (uv was not installed) | Mechanical move to uv later |

Other rulings:

- **No git commits:** you haven't asked me to commit.
- **Geometry validation in one place:** zone polygon checks live in the contracts package, so the API, worker, and core share one validator.
- **Field names:** the result feature block uses `schema_id` and `schema_version`, because pydantic reserves `schema`. Architecture §7.2 is updated to match.
- **Window boundaries:** product windows use exact frame-aligned bounds, while the legacy CSV keeps its own start-time rule unchanged.
- **Legacy flow line kept:** the legacy "infinite" virtual flow line behavior is preserved. It counts any crossing of x = 0.5, not only within the drawn segment. It is documented as a baseline limitation.

## 6. Problems found in existing code (documented, not fixed)

- `model_training/src/inference.py`:
  - fills NaN with 0 and still predicts when the quality check fails;
  - never checks sequence length or column order;
  - continues without a scaler;
  - loads checkpoints with `torch.load(weights_only=False)`, which can run code from the file.
- The UCSD GRU/LSTM models use 10 frame-level optical-flow features. The video pipeline produces 13 zone-level columns, so they cannot be connected directly.
- `app.py` imports `imageio_ffmpeg`, but `requirements.txt` does not list it.
- `model_training/outputs/metrics/recommendation.txt` says "Recommend LSTM for deployment". That is an experiment comparison, not a deployment approval.

## 7. Environment changes

- Created `.venv/` (git-ignored) with numpy, opencv-python, pandas, PyYAML, pydantic, fastapi, uvicorn, httpx, boto3, moto, pytest, import-linter, and imageio-ffmpeg. No torch or ultralytics.
- `.superpowers/` holds the execution ledger (git-ignored).
- `README.md` and `.gitignore` already had uncommitted edits before this work. My edits are on top of them.

## 8. How to resume

1. Write the Task 3 modules listed in §3 until `pytest tests/core/test_adapters.py tests/core/test_policy.py tests/core/test_overlay.py` passes. The tests define the expected behavior.
2. Continue with Task 4 (SQLite repositories in `packages/platform`), then Task 5 (upload API with moto), Task 6 (worker), Task 7 (results API), Task 8 (React UI), Task 9 (E2E), and Task 10 (S3 templates).
3. Owed checks: a real-environment smoke run of the CLI and Streamlit with `ultralytics` installed, and the Task 0.2 S3 spike against a sandbox AWS bucket.
