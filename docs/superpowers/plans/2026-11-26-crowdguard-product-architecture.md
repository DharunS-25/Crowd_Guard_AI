# CrowdGuard Product Implementation Plan (monorepo)

> **For agentic workers:** Execute **one task at a time** with `superpowers:executing-plans`. Each task gives an allowed-file list, its dependencies, the interfaces it may introduce, its checks, and stop conditions. Do not edit outside the allowed list. Stop and report before any breaking change.

**Goal:** One upload-to-annotated-result vertical slice. An operator uploads an authorized venue video directly to private S3, selects zones, starts the analysis, and gets per-zone feature windows, risk assessments (or `INSUFFICIENT_DATA`), a matching overlay video, and downloadable results.

**Architecture:**

- [system-architecture.md](../../architecture/system-architecture.md)
- [repository-structure.md](../../architecture/repository-structure.md) (monorepo layout and dependency rules)
- [s3-upload.md](../../architecture/s3-upload.md)
- [risk-assessment.md](../../architecture/risk-assessment.md)
- [e2e-test-cases.md](../../testing/e2e-test-cases.md)

**Tech stack (proposed):** Python 3.11, FastAPI, boto3, SQLite (WAL), OpenCV, Ultralytics YOLO, ByteTrack, PyTorch (adapters), React + Vite + TypeScript, pytest, vitest, Playwright, uv and pnpm workspaces (D-19), and an S3-compatible local endpoint for tests (D-16).

**Command note:** test paths and commands below are **created by the task that names them**. Until then they do not exist and must not be reported as run.

## Monorepo map (summary of repository-structure.md)

| Path | Unit | Import name |
|---|---|---|
| `packages/contracts/` | Shared pydantic contracts, enums, error catalog | `crowdguard_contracts` |
| `packages/platform/` | Settings, SQLite and migrations, repositories, `ObjectStore`/S3, audit | `crowdguard_platform` |
| `packages/api-client/` | TS types and client generated from OpenAPI | n/a |
| `crowdguard/` (existing) | Core: processing, feature schema, model adapters, risk policy, overlay, CLI | `crowdguard` |
| `apps/api/` | FastAPI routers and services | `crowdguard_api` |
| `apps/worker/` | Job runner, probe, analyze, artifacts | `crowdguard_worker` |
| `apps/web/` | React + Vite UI | n/a |
| `infra/s3/` | S3, CORS, lifecycle, IAM templates | n/a |
| `tests/core/`, `tests/e2e/`, `tests/fixtures/` | Core tests, cross-app E2E, fixture generators | n/a |

Frozen during Phase 1: `app.py`, `model_training/**`, `config/**`, the CLI flags and outputs of `crowdguard/main.py`, and `requirements.txt`.

## Global constraints

- Single video, single camera per analysis. No RTSP, multi-camera, alerts, or evacuation features.
- Severity levels are `LOW`/`MEDIUM`/`HIGH`. `INSUFFICIENT_DATA` is separate and is never coerced to `LOW`.
- UCSD `Normal`/`Anomalous` is never mapped to severity. GRU/LSTM output is experimental anomaly evidence only.
- No invented thresholds. No averaging or voting across models without a shared, validated label space and calibration.
- Every result carries the zone, window, data quality, model and policy versions, and supporting evidence. Overlay, JSON, CSV, and UI agree.
- Browser-to-S3 direct upload. The API never proxies video bytes and never reveals keys, the S3 UploadId, or credentials.
- `crowdguard_api` never imports `crowdguard` core, torch, or OpenCV. The dependency graph in repository-structure §3 is enforced by import-linter.
- Tests use deterministic fakes. No weight downloads, no AWS, no live camera in CI. Never commit footage, weights, databases, or credentials.

## Agent boundaries

- **Shared contracts** (`packages/contracts/**`, the generated `packages/api-client/**`, and `docs/architecture/**`) change only in Task 1 or in an explicitly assigned contract-change task, never in parallel with another task.
- **Workspace and tooling files** (root `pyproject.toml`, lock files, `package.json`, `pnpm-workspace.yaml`, `.gitignore`, CI) change only in Task 0.3 or an assigned tooling task.
- **Parallel-safe after Task 1:**
  - Task 2 (core processing), Task 4 (platform repositories), and Task 10 (infra) can run together.
  - Task 3 (core risk and overlay) can run alongside Task 4 or Task 5.
  - Task 8 can start against a mocked client once `packages/api-client` exists.
  - Everything else is serial in dependency order.
- Additive, backwards-compatible changes are the default. **Stop and report** if a task would rename, move, or delete existing files; change CLI flags or CSV meaning; change a shared contract; add an unlisted dependency; or touch files outside its list.
- Each task ends with a report: changed files, the exact commands run and their output summary, the E2E IDs covered, open decisions encountered, and the next task.

---

## Phase A: Architecture

### Task A0: Architecture and documentation revision 2

- [x] Verify the current repository state; mark current versus proposed (`system-architecture.md` §2).
- [x] S3 direct-upload design (`s3-upload.md`).
- [x] Model adapter contract, fusion policy, reason codes, validation gates, decisions register (`risk-assessment.md`).
- [x] Monorepo layout and dependency rules (`repository-structure.md`).
- [x] E2E acceptance cases for upload, processing, models, overlays, downloads, isolation, and cleanup (`e2e-test-cases.md`).
- [x] `CLAUDE.md`, `.claude/rules/**`, and the implementation prompt updated.

## Phase 0: Decisions, spikes, scaffold

### Task 0.1: Resolve decisions needed by Phase 1

**Allowed files:** `docs/architecture/**`, this plan.

- [ ] Get answers for D-07 (overlay semantics), D-12 (upload limits and quotas), D-15 (probe tool), D-16 (test S3 endpoint), D-18 (authorization attestation), and D-19 (uv and pnpm workspaces). Record each with its owner and date in `risk-assessment.md` §11.
- [ ] Confirm that D-01 to D-05 stay unresolved for Phase 1 (the default policy yields `INSUFFICIENT_DATA`).

### Task 0.2: S3 behavior spike (throwaway code, not committed)

**Allowed files:** `docs/architecture/s3-upload.md`. Spike code lives in a scratch directory outside the repo.

- [ ] Against the chosen local endpoint, and against a sandbox AWS bucket if available: presign `UploadPart` with a required `x-amz-checksum-sha256` header, read the `ETag` cross-origin, test the 403 on an expired URL, a checksum mismatch, `ListParts` pagination, a repeat `CompleteMultipartUpload`, and abort during an in-flight part.
- [ ] Record the answers to the §2 spike questions and any parity gaps between the test endpoint and AWS.
- **Covers:** prerequisites for UP-10 and UP-15.

### Task 0.3: Monorepo scaffold (no product logic)

**Depends on:** Task 0.1 (D-19).
**Allowed files:**

- root `pyproject.toml`, `uv.lock`, `package.json`, `pnpm-workspace.yaml`, `pnpm-lock.yaml`;
- `packages/{contracts,platform}/pyproject.toml` with an empty package `__init__.py`;
- `apps/{api,worker}/pyproject.toml` with an empty package `__init__.py`;
- `apps/web/` (Vite TS template only), `packages/api-client/package.json`;
- `tests/core/__init__.py`, `tests/core/test_import_boundaries.py`;
- `.gitignore` (append only: `node_modules/`, `apps/web/dist/`, `.venv`-style entries);
- `docs/architecture/repository-structure.md` (corrections only).

Steps:

- [ ] Declare the workspace members: `packages/contracts`, `packages/platform`, `apps/api`, `apps/worker`. The root project is the `crowdguard` core. `model_training` is **not** a member.
- [ ] Add import-linter contracts from repository-structure §3, plus a test that `crowdguard_api` imports without torch or cv2.
- [ ] **Regression check (must stay green):** `pip install -r requirements.txt` still installs, `python -m crowdguard.main --help` works, and `streamlit run app.py` starts. **Stop and report** if the root `pyproject.toml` changes how either command behaves.
- [ ] Run `uv run lint-imports` and `uv run pytest tests/core/test_import_boundaries.py -q`, or the pip fallback commands if D-19 is declined.

---

## Phase 1: Upload-to-annotated-result vertical slice

### Task 1: Shared contracts and error catalog (serial gate)

**Depends on:** Task 0.3.
**Allowed files:** `packages/contracts/crowdguard_contracts/{__init__,features,evidence,assessment,analysis,upload,results,errors}.py`, `packages/contracts/tests/**`, `docs/architecture/**` (contract wording only).

**Interfaces:**

- `FeatureSchema(id, version, columns, units)`, `ZoneWindowFeatures`.
- `ModelDescriptor`, `ModelEvidence`, `InferenceStatus`, `DataQuality(state, warnings)`, as in risk §4.
- `ZoneRiskAssessment(level, assessment_status, reason_codes, policy, evidence_refs)`.
- `AnalysisStatus`, `UploadSessionState`, the `ErrorCode` enum, and the API request and response models from s3-upload §4 and architecture §7–§8.

Steps:

- [ ] Write failing tests: `INSUFFICIENT_DATA` requires a reason code; probabilities must be null unless `OK`; probability keys must equal `class_labels`; window `start_s < end_s`; unknown enum values are rejected; responses contain no S3 key or UploadId fields.
- [ ] Implement the models. Only pydantic and the stdlib.
- [ ] Run `uv run pytest packages/contracts/tests -q`.

### Task 2: Extract the reusable processing service, keeping the CLI

**Depends on:** Task 1.
**Allowed files:** `crowdguard/processing.py`, `crowdguard/feature_schema.py`, `crowdguard/main.py` (only to delegate to `process_video`; flags and outputs unchanged), `tests/core/test_processing_service.py`, `tests/core/test_cli_compat.py`, `tests/fixtures/**` (generators only, no media).

**Interfaces:** `process_video(request: AnalysisInput, detector: Detector, progress: ProgressCallback, cancel: CancelToken) -> ProcessingResult`. The result holds ordered `ZoneWindowFeatures` per zone, source metadata, and per-window frame counts. `Detector` is injectable so tests use a fake.

- [ ] Golden test: with a fake detector, the current CLI and the new path give identical CSV rows on a generated fixture. **Stop and report** if they cannot.
- [ ] Tests: source-time stamps from fps and frame index; windows ordered and never crossing zones; `NO_DETECTIONS` vs missing frames; FPS fallback warning; `occupancy_ratio = null` without a sourced capacity (MD-09); cancellation; temp-path isolation.
- [ ] Run `uv run pytest tests/core/test_processing_service.py tests/core/test_cli_compat.py -q`.
- **Covers:** PR-09, MD-09.

### Task 3: Model adapters, fusion policy, overlay renderer (core)

**Depends on:** Task 1.
**Allowed files:** `crowdguard/models/{__init__,base,registry,orchestrator}.py`, `crowdguard/risk/{__init__,policy,rules}.py`, `crowdguard/risk/policies/crowdguard-default.yaml`, `crowdguard/overlay.py`, `tests/core/test_{adapters,orchestrator,policy,overlay}.py`, `tests/fixtures/models/**`, `tests/fixtures/policies/**`.

**Interfaces:**

- `ModelAdapter.descriptor()`, `ModelAdapter.predict(seq: FeatureSequence) -> ModelEvidence`.
- `Orchestrator.run(windows, adapters) -> list[ModelEvidence]`.
- `DecisionPolicy.load(path)`, which raises `POLICY_INCOMPLETE`, and `DecisionPolicy.assess(window, evidence, rule_signals) -> ZoneRiskAssessment`.
- `render(frame, t, zones, assessments) -> frame`, plus `OverlayManifestBuilder`.

Steps:

- [ ] Fakes: `fake-venue-risk`, `fake-binary-anomaly`, raising, missing-artifact, and schema-mismatch adapters.
- [ ] Unit tests for MD-01 to MD-08: no zero-fill; no reorder; the binary label space is rejected for severity; the disagreement rule is required; the default policy yields `INSUFFICIENT_DATA` with reasons.
- [ ] Overlay tests for OV-01 to OV-04: the containing-window rule (D-07), boundary frames, the unknown style, the banner, and manifest equality with the assessments.
- [ ] Run `uv run pytest tests/core/test_adapters.py tests/core/test_orchestrator.py tests/core/test_policy.py tests/core/test_overlay.py -q`.
- **Covers:** MD-01–08, OV-01–04, PR-08 (unit).

### Task 4: Platform persistence and zone configs

**Depends on:** Task 1.
**Allowed files:** `packages/platform/crowdguard_platform/{settings,db}.py`, `packages/platform/crowdguard_platform/migrations/*.sql`, `packages/platform/crowdguard_platform/repositories/**`, `packages/platform/crowdguard_platform/zone_configs.py`, `packages/platform/tests/**`.

**Interfaces:** `AnalysisRepository`, `UploadSessionRepository`, `JobRepository` (claim, heartbeat, release with leases), `ArtifactRepository`, `AuditRepository`, `ZoneConfigRepository`. Every query is scoped by `owner_id`, and state transitions are compare-and-set.

- [ ] Tests: concurrent claims never double-claim; an expired lease is reclaimable; CAS transitions reject the wrong prior state; owner scoping returns nothing across owners; zone validation (self-intersection, range, duplicate IDs, capacity with a source or null).
- [ ] Run `uv run pytest packages/platform/tests -q`.
- **Covers:** PR-03 (validation), IS-02 (data layer).

### Task 5: Object store and upload session API

**Depends on:** Tasks 0.2, 1, 4.
**Allowed files:** `packages/platform/crowdguard_platform/storage/{__init__,object_store,s3}.py`, `packages/platform/tests/test_object_store.py`, `packages/platform/pyproject.toml` (add `boto3`; report it), `apps/api/crowdguard_api/{main,auth}.py`, `apps/api/crowdguard_api/routers/uploads.py`, `apps/api/crowdguard_api/services/{uploads,sweeper}.py`, `apps/api/tests/{conftest,test_uploads_api}.py`, `apps/api/pyproject.toml` (add `fastapi`, `uvicorn`, test deps; report them).

**Interfaces:** the endpoints in s3-upload §4; `ObjectStore` (presign put or part, create, list parts, complete, abort, head, get to file, put file, delete prefix).

- [ ] Tests against the local S3 endpoint: UP-02, UP-07, UP-08, UP-09 (including the crash reconciliation), UP-11, UP-14 (response bodies), UP-16, IS-01, IS-02, IS-03, IS-05, CL-01.
- [ ] Generated keys only; the S3 UploadId is never in a response; presigned URLs are never logged (log-capture assertion).
- [ ] Run `uv run pytest packages/platform/tests/test_object_store.py apps/api/tests/test_uploads_api.py -q`.

### Task 6: Worker, validation probe, analysis job, artifacts

**Depends on:** Tasks 2, 3, 4, 5.
**Allowed files:** `apps/worker/crowdguard_worker/{__init__,runner,validate_upload,analyze,artifacts,probe}.py`, `apps/worker/tests/**`, `apps/worker/pyproject.toml` (the core dependency and the D-15 probe dependency only; report them).

- [ ] Validate job: UP-01, UP-12, UP-13; preview frame stored; the source deleted on rejection.
- [ ] Analyze job: runs `crowdguard.processing.process_video`, the orchestrator, policy, and renderer; encodes H.264 with faststart; writes `results.json`, the CSVs, the overlay manifest, and `manifest.json` from the same objects; uploads to `runs/{run_id}/`; registers atomically.
- [ ] Failure paths: PR-04, PR-05, PR-06, CL-02; DL-01 and DL-02 at artifact level; OV-05 probe check.
- [ ] Run `uv run pytest apps/worker/tests -q`.

### Task 7: Analyses, results, artifacts, deletion API, OpenAPI client

**Depends on:** Tasks 4, 6.
**Allowed files:** `apps/api/crowdguard_api/routers/{analyses,zone_configs}.py`, `apps/api/crowdguard_api/services/{analyses,retention}.py`, `apps/api/tests/test_analyses_api.py`, `packages/api-client/**` (generation script and generated output).

- [ ] `start` (PR-02, PR-03), `cancel`, status, results (paginated), artifact presign from the enum (DL-03, DL-04), the preview frame, and deletion and retention (CL-03). Ownership on every route (IS-02).
- [ ] Generate `packages/api-client` from OpenAPI and commit the output. CI fails if the generated client is out of date.
- [ ] Run `uv run pytest apps/api/tests/test_analyses_api.py -q`.

### Task 8: Web UI, upload client, results player

**Depends on:** Tasks 5, 7. It may start against a mocked `packages/api-client` after Task 1.
**Allowed files:** `apps/web/**`.

- [ ] Upload client (`src/features/upload/`): part slicing and hashing, at most `max_concurrency` PUTs, per-part retry with backoff, URL renewal on 403, resume through `GET /uploads/{id}`, abort, and no presigned URL persistence (UP-03 to UP-07, UP-14).
- [ ] Zone selection on the preview frame (`src/features/zones/`), start, polling of all states (`src/features/analysis/`, PR-07).
- [ ] Results (`src/features/results/`): the player with the overlay, a zone/window table synchronized to playback (OV-06), experimental and unknown banners, model evidence shown separately (MD-05), and downloads.
- [ ] Run `pnpm --filter web test` (vitest, created by this task).

### Task 9: E2E harness and suite

**Depends on:** Task 8.
**Allowed files:** `tests/e2e/**`, `tests/fixtures/generate_videos.py`, `tests/e2e/playwright.config.ts`, a CI workflow file only if one is requested.

- [ ] Implement the harness from e2e §1 and every case marked for Tasks 5–9. Add CL-05 as a suite teardown assertion.
- [ ] Run `pnpm exec playwright test -c tests/e2e/playwright.config.ts` (created by this task) and report the per-case results.

### Task 10: S3 infrastructure templates and sandbox verification

**Depends on:** Task 0.2. Can run in parallel after Task 1.
**Allowed files:** `infra/s3/{bucket-policy.json,cors.json,lifecycle.json,iam-api-signer.json,iam-worker.json,README.md}`, `tests/core/test_infra_templates.py`.

- [ ] Templates per s3-upload §7 with placeholders for the bucket, origin, and account. No real IDs.
- [ ] CL-04 static test; UP-15 against the local endpoint.
- [ ] Manual checklist (human-run): the multipart path against a sandbox AWS bucket with the real IAM and CORS. Record the outcome in `infra/s3/README.md`.

**Phase 1 done when:** E2E cases UP-01 to UP-16, PR-01 to PR-09, MD-01 to MD-09, OV-01 to OV-06, DL-01 to DL-04, IS-01 to IS-05, and CL-01 to CL-05 pass, and the Streamlit and CLI regression check from Task 0.3 is still green. Real uploads show `INSUFFICIENT_DATA` with reasons, and fixture runs show the demo levels with a `DEMONSTRATION` banner.

---

## Phase 2: Real model evidence (experimental)

### Task 11: Safe adapters for the UCSD GRU/LSTM

**Depends on:** Phase 1 and D-06.
**Allowed files:** `crowdguard/models/ucsd/{__init__,arch,adapter}.py`, `crowdguard/models/feature_producers/**` (only if D-06 approves a producer), `tests/core/test_ucsd_adapter.py`, `model_training/scripts/export_scaler_json.py` (new file; the only `model_training` change).

- [ ] Port the GRU/LSTM classes into `crowdguard/models/ucsd/arch.py`, with a parity test against the same `state_dict`. Do not import `model_training/src`.
- [ ] Load with `weights_only=True` from allow-listed hashes; the scaler from JSON (no pickle); derive the descriptor from the artifact and config (sequence length 20, step 5, 10 columns in order).
- [ ] Return `INSUFFICIENT_DATA`, `SCHEMA_MISMATCH`, or `MODEL_UNAVAILABLE` where `ExperimentalPredictor` would currently zero-fill or continue.
- [ ] `eligible_for_severity = false`. Results show it as experimental anomaly evidence only.
- [ ] If there is a producer: a parity test on UCSD frames against `model_training` preprocessing, and the domain-shift warning `ZONE_CROP_UNVALIDATED`.

### Task 12: Configured rule signals

**Depends on:** D-02. **Allowed files:** `crowdguard/risk/rules.py`, `tests/core/test_rules.py`.

- [ ] Rule signals read venue thresholds with source, owner, and revision. Without them, keep `NOT_CONFIGURED`. No default values are shipped.

## Phase 3: Venue-risk model and validated fusion [Deferred]

- [ ] Task 13: Annotation protocol and label definitions (D-01) in `model_training/DATA_ANNOTATION_PROTOCOL.md`.
- [ ] Task 14: Train and evaluate the `venue_risk_v1` model with group splits and calibration. Model card.
- [ ] Task 15: Fusion policy or fusion model (D-03, D-04, D-05), validation on held-out groups, shadow pilot, approval (D-10).

## Phase 4: Internet-facing hardening [Required before any public deployment]

- [ ] Task 16: Identity provider and roles (D-17), per-user ownership through `crowdguard_api.auth`, quotas and rate limits per user, audit review, and a security review plus a threat model.

## Phase 5: Multi-camera, RTSP, alerts [Deferred]

Calibration to venue coordinates, time sync, an overlap policy, camera health, and an alert design with human acknowledgement. Each needs a separate architecture review. New deployables go under `apps/` (for example `apps/ingest/`).

## Phase 6: Evacuation planning [Deferred, research]

Venue topology graph, capacities, calibrated risks, flow estimates, blocked routes, and accessibility constraints, leading to `DRAFT_FOR_HUMAN_REVIEW` recommendations. Validate with simulation and expert review. No autonomous instructions and no public alerts. Planned home: a new `packages/evacuation/` behind an `EvacuationPlanner` interface.

## Phase M: Optional directory migration [Not scheduled]

Move `crowdguard/` to `packages/core/` and `model_training/` to `ml/training/`, and retire `app.py`, only through the reviewed migration in repository-structure §6.

---

## Coverage check

| Requirement | Where specified | Implemented by |
|---|---|---|
| Monorepo layout, unique import names, one-way dependencies | repository-structure §2–§3 | Task 0.3, enforced in every task |
| Direct browser-to-private-S3, single or multipart (100 MiB threshold, 16 MiB parts, ≤3 concurrent) | s3-upload §1, §3 | Tasks 5, 8 |
| Server-side key and UploadId, idempotent verified completion, abort, lifecycle | s3-upload §4, §5, §7 | Tasks 5, 10 |
| Probe before processing; only verified objects are queued | s3-upload §8; architecture §5 | Task 6 |
| Quotas, rate limits, CORS, encryption, IAM, retention | s3-upload §7; architecture §9–§10 | Tasks 5, 7, 10, 16 |
| Timestamped frames, YOLO, ByteTrack, zones, ordered windows | architecture §6 | Task 2 |
| Versioned model adapters with explicit contract and visible raw evidence | risk §4 | Tasks 3, 11 |
| Separate fusion and policy stage; no averaging or voting | risk §6 | Tasks 3, 15 |
| UCSD binary not mapped to severity | risk §5 | Tasks 3, 11 |
| `INSUFFICIENT_DATA` never `LOW` | risk §2, §7 | Tasks 1, 3 |
| Time-aligned overlay consistent with JSON, CSV, UI | architecture §7; risk §10 | Tasks 3, 6, 8 |
| Downloads with timestamps, evidence, quality | architecture §7.3 | Tasks 6, 7 |
| Evacuation seam only | architecture §11.2 | Phase 6 (no MVP work) |
| Preserve Streamlit and CLI | Global constraints | Task 0.3 regression check, Task 2 golden test |
