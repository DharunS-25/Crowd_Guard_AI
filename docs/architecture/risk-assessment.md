# Zone risk assessment, model evidence, and fusion boundary

**Status:** [Proposed] semantics and contracts, revision 2 (2026-11-26). This document does **not** validate any model, approve any threshold, or make any output safe to rely on. Parent: [system-architecture.md](system-architecture.md).

## 1. Scope and principles

- CrowdGuard is an **orchestrator and fusion layer** over several Python components. Each model's raw evidence stays visible next to the fused result.
- `LOW`, `MEDIUM`, and `HIGH` are **proposed** severity levels. Their meaning needs domain-owner approval (D-01).
- `INSUFFICIENT_DATA` is a separate assessment state, not a fourth severity. Missing, stale, invalid, ineligible, or unconfigured inputs produce it. **They never produce `LOW`.**
- No numeric density, capacity, speed, persistence, or risk threshold is defined in this repository. Venue-specific values come from a named owner with a revision (D-02).
- Image-area density and pixel speed are proxies unless camera calibration supports physical units.

## 2. Output states

| Field | Values | Meaning |
|---|---|---|
| `assessment.level` | `LOW`, `MEDIUM`, `HIGH` | Severity selected by an approved policy from adequate, eligible evidence. |
| | `INSUFFICIENT_DATA` | No supported estimate for this zone and window. Always has at least one reason code. |
| `assessment.assessment_status` | `DEMONSTRATION` | Produced by a fixture or fake model or policy (tests, demos). Never shown without a banner. |
| | `EXPERIMENTAL` | Real pipeline, but the model or policy has not passed the validation gates (§9). This is the default for all real output. |
| | `VALIDATED_FOR_PILOT` | Passed §9 gates for a named venue and camera scope. Promotion needs written approval (D-10). |

Display: LOW is green or neutral, MEDIUM amber, HIGH red, and INSUFFICIENT_DATA grey and hatched. Text and icons always accompany color. The status banner is always visible on the video and in the UI.

## 3. Evidence groups

These are signals, not interchangeable measurements. Record each raw value with its units, source, and quality.

1. **Occupancy pressure:** count, change in count, configured capacity, occupancy ratio, and the image-area density proxy. Capacity is venue configuration with a source. Without it, `occupancy_ratio` is `null` and gets a `CAPACITY_UNCONFIGURED` warning. The demo capacities in `config/zones.json` must not produce real-looking ratios in product output.
2. **Movement and flow:** speed distribution, direction consistency and conflict, crossings, inflow and outflow. Current values are pixels per second and one illustrative virtual line.
3. **Temporal persistence and change:** trends and repeated conditions across ordered windows, plus temporal-model evidence. One noisy frame must not become a persistent condition.

## 4. Model adapter contract

Every model, including fakes, is wrapped in a versioned adapter that implements this contract (`crowdguard/models/`, proposed).

### 4.1 `ModelDescriptor` (static, published in `results.json`)

| Field | Example | Rule |
|---|---|---|
| `model_id` | `ucsd-anomaly-lstm` | Stable name. |
| `model_version` | `sha256:ab12…` | Derived from the artifact hash, not hard-coded. |
| `artifact_sha256`, `scaler_sha256` | | Load fails if either does not match the allow-list. |
| `task` | `binary_anomaly`, `venue_risk_3class` | |
| `label_space` | `ucsd_binary_anomaly`, `venue_risk_v1` | The fusion policy accepts or rejects evidence by label space. |
| `class_labels` | `["Normal", "Anomalous"]` | Ordered. Defines how probabilities are indexed. |
| `feature_schema` | `{id: "ucsd_flow_features", version: "1.0.0", columns: [...ordered...]}` | Exact names and order. |
| `sequence_length`, `sampling` | `20`, `{"unit": "frame", "step": 5}` | The input window must match exactly. |
| `scope` | `frame` or `zone` | Whether the model was trained on whole frames or zone windows. |
| `calibration_status` | `UNCALIBRATED`, `CALIBRATED:{method, dataset}` | |
| `validation_status` | `EXPERIMENTAL`, `VALIDATED_FOR_PILOT:{scope}` | |
| `eligible_for_severity` | `false` | True only for a validated model in a label space the policy accepts. |

### 4.2 Input and checks

The orchestrator builds a `FeatureSequence` per `(analysis_id, camera_id, zone_id)`: ordered rows, the column names, source-time stamps, and per-row quality flags. The adapter runs these checks in order and stops at the first failure:

1. The artifact and scaler are loaded and their hashes match. Otherwise `MODEL_UNAVAILABLE`.
2. The column names and order equal `feature_schema.columns`. Otherwise `SCHEMA_MISMATCH`. Never reorder, rename, or fill columns.
3. The row count is at least `sequence_length`, and timestamps are strictly increasing with the declared sampling step (within a configured tolerance). Otherwise `INSUFFICIENT_DATA` with reason `WARMUP_INCOMPLETE` or `SEQUENCE_GAP`.
4. Values are finite and within schema ranges, and the row quality flags are acceptable. Otherwise `INSUFFICIENT_DATA` with reason `FEATURE_INVALID` or `FEATURE_STALE`. **No zero-fill or mean-fill.**

### 4.3 `ModelEvidence` (one per adapter, zone, and window, including failures)

```json
{
  "evidence_id": "ucsd-anomaly-lstm@sha256:ab12…/ENTRY/41",
  "model_id": "ucsd-anomaly-lstm", "model_version": "sha256:ab12…",
  "label_space": "ucsd_binary_anomaly", "class_labels": ["Normal", "Anomalous"],
  "zone_id": "ENTRY", "camera_id": "cam-1",
  "window": {"index": 41, "start_s": 41.0, "end_s": 42.0},
  "input": {"feature_schema": "ucsd_flow_features@1.0.0", "rows_used": 20, "first_row_s": 22.0, "last_row_s": 42.0},
  "inference_status": "OK",
  "probabilities": {"Normal": 0.81, "Anomalous": 0.19},
  "data_quality": {"state": "OK", "warnings": []},
  "calibration_status": "UNCALIBRATED", "validation_status": "EXPERIMENTAL",
  "latency_ms": 3.2
}
```

`inference_status` takes one of these values: `OK`, `INSUFFICIENT_DATA`, `SCHEMA_MISMATCH`, `MODEL_UNAVAILABLE`, `ERROR`, or `SKIPPED_INELIGIBLE`. `probabilities` is `null` unless the status is `OK`. `data_quality.state` takes one of `OK`, `DEGRADED`, or `INVALID`. An adapter exception becomes `ERROR` evidence for that window only, and other adapters and zones continue. If an adapter fails for every window, the job still completes, with the failure visible in the results. It does not fail silently, and the system does not substitute another model.

## 5. Existing UCSD GRU/LSTM models [Current]

- **Target:** binary `Normal`/`Anomalous` on UCSD Ped1/Ped2 (pedestrian walkway anomalies such as bikes and carts). This is not venue crowd risk.
- **Input:** 10 frame-level optical-flow features (`CORE_FEATURE_COLUMNS` in `model_training/src/preprocessing.py`), sequence length 20, every 5th frame, whole-frame scope. The video pipeline does **not** produce these features. It produces 13 zone-level columns per `--interval` seconds.
- **Gaps in `model_training/src/inference.py`** that an adapter must close, without editing training code unless a task allows it:
  - fills NaN with 0 and runs inference even when the quality check fails, so it must return `INSUFFICIENT_DATA` instead;
  - never checks sequence length or column order;
  - continues without a scaler;
  - loads the checkpoint with `weights_only=False` and a pickled scaler, both of which can run code. The adapter must load with `weights_only=True` and get the scaler from a converted, non-pickle file with a pinned hash;
  - hard-codes `model_id`.
- **Rules:**
  - never map `Normal → LOW` or `Anomalous → HIGH`, and never derive `MEDIUM` from confidence;
  - `eligible_for_severity = false`; the fusion policy rejects the `ucsd_binary_anomaly` label space;
  - in the product the outputs appear only as labelled **experimental anomaly evidence**, separate from the zone risk level and overlay color;
  - running them on zone crops instead of whole frames is an unvalidated domain shift (D-06). Until a parity-tested feature producer exists, their evidence is `SKIPPED_INELIGIBLE` with reason `NO_FEATURE_PRODUCER_FOR_SCHEMA`;
  - `outputs/metrics/recommendation.txt` ("Recommend LSTM for deployment") compares two experiments. It is not deployment approval.

## 6. Fusion and decision policy

The fusion stage (`crowdguard/risk/`, proposed) is separate from every model adapter.

**Inputs:** the `ModelEvidence[]` for the zone and window, the `RuleSignal[]` computed from features and venue config, the window `data_quality`, and a `PolicyConfig`.

**`PolicyConfig`** is versioned, has a hash, and is published in the results. It names:

- `policy_id`, `version`, `owner`, `approved_by` (or `null`), and `assessment_status`;
- `accepted_label_spaces` and, per space, the required `calibration_status` and `validation_status`;
- the rule signals used, each with its threshold values **and** source, owner, and revision. Missing values give `POLICY_THRESHOLDS_UNCONFIGURED`;
- the precedence between rules and models (D-03), the disagreement rule (D-04), and persistence or hysteresis (D-05);
- which data-quality states may contribute (`OK` only, or `OK` and `DEGRADED`).

**Algorithm shape (normative order):**

1. **Data-quality gate.** If the window is `INVALID`, the result is `INSUFFICIENT_DATA` (`FEATURE_INVALID`).
2. **Eligibility filter.** Keep evidence whose label space is accepted and whose status meets the policy. Record every rejected item with its reason (it stays in `model_evidence`).
3. If there is no eligible evidence and no configured rule signal, the result is `INSUFFICIENT_DATA` (`NO_ELIGIBLE_SEVERITY_EVIDENCE` and/or `POLICY_THRESHOLDS_UNCONFIGURED`).
4. Evaluate rule signals and eligible evidence **independently**, then combine them by the declared precedence and disagreement rule. Never average ordinal classes. Never take a majority vote unless all inputs share one validated, calibrated label space and the policy says so explicitly.
5. Apply persistence or hysteresis as declared.
6. Emit the level, `assessment_status` (the least validated of the policy and its contributing evidence), `reason_codes`, `evidence_refs`, and the policy ID and version.

A policy that lacks a required decision (for example, two eligible sources exist but no disagreement rule is set) **fails to load** with `POLICY_INCOMPLETE`. It does not fall back to a default. A separately trained fusion model is allowed instead of a rule policy. It is wrapped as an adapter over the evidence vector, with its own descriptor, label space `venue_risk_v1`, calibration, and the §9 validation.

**MVP reality:** no eligible severity model and no approved thresholds exist. The default policy `crowdguard-default@0.1.0` therefore yields `INSUFFICIENT_DATA` for real uploads. Tests use `fixture-demo` policies with a deterministic fake `venue_risk_v1` adapter, marked `DEMONSTRATION`.

## 7. Reason and warning codes

| Code | Level effect | Trigger |
|---|---|---|
| `WARMUP_INCOMPLETE` | `INSUFFICIENT_DATA` | Fewer than `sequence_length` ordered windows so far |
| `SEQUENCE_GAP` | `INSUFFICIENT_DATA` | Missing or out-of-order rows in the sequence |
| `FEATURE_INVALID` | `INSUFFICIENT_DATA` | NaN, inf, or out-of-range values; quality `INVALID` |
| `FEATURE_STALE` | `INSUFFICIENT_DATA` | Decoded frames in the window are below the configured quality fraction (a data-quality setting, not a risk threshold) |
| `SCHEMA_MISMATCH` | evidence only | Column names, order, or version differ from the descriptor |
| `MODEL_UNAVAILABLE` / `MODEL_ERROR` | evidence only | Artifact missing, hash mismatch, or exception |
| `NO_FEATURE_PRODUCER_FOR_SCHEMA` | evidence only | No producer implements the adapter's schema |
| `NO_ELIGIBLE_SEVERITY_EVIDENCE` | `INSUFFICIENT_DATA` | All evidence rejected by label space or status |
| `POLICY_THRESHOLDS_UNCONFIGURED` | `INSUFFICIENT_DATA` | A rule signal has no approved value |
| `POLICY_UNAVAILABLE` / `POLICY_INCOMPLETE` | job fails at load | Missing or incomplete policy |
| `MODEL_DISAGREEMENT` | per D-04 rule | Eligible sources disagree; recorded with both values |
| `CAPACITY_UNCONFIGURED`, `FPS_FALLBACK_USED`, `NO_DETECTIONS` | warning | Shown in UI, CSV, and overlay quality marker |

A zero count with good frames is a valid observation (`NO_DETECTIONS` warning). It is not the same as missing data.

## 8. Training data for a venue-risk model [Deferred, Phase 3]

1. Define observable Low, Medium, and High labels and operator actions with venue safety experts (D-01).
2. Collect rights-cleared, representative venue video with time-aligned per-zone windows, plus retention, privacy, and annotation protocols.
3. Use independent annotators, measure agreement, and adjudicate disagreements.
4. Split by event, venue, and camera group **before** building overlapping windows. Fit scalers on training groups only.
5. Lock the ordered feature schema, sampling, sequence length, missing-data rules, and calibration method in a versioned descriptor.

## 9. Validation gates (before any status above `EXPERIMENTAL`)

- Per-class precision and recall, macro-F1, the confusion matrix, and **false-negative analysis for HIGH** on held-out events and venues.
- Calibration (reliability curves, ECE, or equivalent) if probabilities feed fusion or are displayed.
- Data-quality coverage: the fraction of windows that are `INSUFFICIENT_DATA` and why.
- For the fusion policy or model: evaluate on the same held-out groups, review the disagreement cases, and run a shadow period with operators.
- A model card and policy card documenting limitations and scope, plus written approval (D-10).

Until these gates pass, every Low, Medium, or High output carries `EXPERIMENTAL` or `DEMONSTRATION` and must not drive safety-critical decisions.

## 10. Overlay behavior

- Draw the same polygon used for assignment, the zone name, the level or unknown state, the window `[start–end]`, and the assessment-status banner.
- The overlay at frame time `t` uses the assessment of the window that contains `t` (D-07). The window's `(zone_id, window.index)` must equal the UI row and the CSV row.
- Frames with no window (for example, the tail after the last full window) show `INSUFFICIENT_DATA`.
- The overlay manifest records every frame range, zone, window index, level, and palette token, so tests assert on it rather than on pixels alone.
- UCSD anomaly evidence, if ever shown, is a separate labelled marker. It is never the zone fill color.

## 11. Open decisions register (all areas)

| ID | Decision | Owner | Blocks |
|---|---|---|---|
| D-01 | Operational meaning of LOW, MEDIUM, HIGH and the evidence groups | Venue safety domain owner | Real severity output; Phase 3 labels |
| D-02 | Source, owner, and revision process for capacities and rule thresholds | Domain owner | Rule signals beyond `NOT_CONFIGURED` |
| D-03 | Fusion approach (rule policy or trained fusion model) and rule-versus-model precedence | Product + ML lead | Policy beyond default |
| D-04 | Disagreement rule between eligible sources (no default) | Domain owner + ML lead | Any policy with two or more eligible sources |
| D-05 | Persistence, hysteresis, and cooldown | Domain owner | Temporal smoothing |
| D-06 | Whether to build a UCSD feature producer (zone-crop domain shift), or keep GRU/LSTM offline only | ML lead | Phase 2 Task 11 |
| D-07 | Overlay time semantics. Recommended: the containing window for batch review; causal "last completed window" for future live mode | Product | Task 3 renderer |
| D-08 | Window length and sequence length for the severity model (the pipeline currently uses a 1.0 s interval) | ML lead | Phase 3 |
| D-09 | Validity or freshness window if causal overlay semantics are adopted | Product | Live mode only |
| D-10 | Criteria and approver for `VALIDATED_FOR_PILOT` | Domain owner + product | Status promotion |
| D-11 | Retention periods for source video, outputs, DB rows, and audit logs | Product + privacy | Lifecycle rules (Task 10) |
| D-12 | Upload limits (bytes, duration, resolution, containers and codecs), quotas, rate limits | Product + ops | Task 5 config values |
| D-13 | SSE-S3 or SSE-KMS | Ops/security | Task 10 |
| D-14 | Audit source (CloudTrail data events or S3 access logs) and audit retention | Ops/security | Task 10 |
| D-15 | Probe tool (ffprobe or PyAV) and adding that dependency | Engineering | Task 6 |
| D-16 | S3-compatible test endpoint (MinIO container or moto server) | Engineering | Task 0.2 / Task 9 |
| D-17 | Identity provider and roles for an internet-facing deployment | Product + security | Phase 4 |
| D-18 | Lawful basis, venue authorization attestation, and privacy notice for uploaded footage | Product + legal | Any use with real venue footage |
| D-19 | Monorepo tooling: uv and pnpm workspaces, or a pip/npm fallback ([repository-structure.md §4](repository-structure.md#4-tooling-decision-d-19)) | Engineering | Task 0.3 |

### 11.1 Provisional engineering defaults (2026-11-26, pending owner confirmation)

Phase 1 needs these to proceed. Every value is configuration or a swappable component, so the owner can override any of them without a contract change. None of them is a risk threshold.

| ID | Provisional default | Where it lives |
|---|---|---|
| D-07 | Overlay shows the assessment of the window that **contains** frame time `t` | `crowdguard/overlay.py` |
| D-12 | `UPLOAD_MAX_BYTES` 2 GiB, `UPLOAD_MAX_DURATION_S` 600, max resolution 3840×2160, containers MP4/MOV/AVI. Quotas: 3 active upload sessions and 5 active analyses per owner, 20 session creates per minute | `crowdguard_platform.settings` (environment overridable) |
| D-15 | Probe with OpenCV (already a dependency) plus a full decode pass. Encode browser H.264 with `imageio-ffmpeg` (already used by `app.py`). No ffprobe dependency | `crowdguard_worker/probe.py` |
| D-16 | moto (`moto[server]`) as the S3-compatible test endpoint (pip-installable, no Docker) | tests |
| D-18 | Upload session creation requires `authorization_attested: true`, stored with the owner and time in the audit log | upload API |
| D-19 | pip fallback: one `pyproject.toml` per unit (setuptools) and root pytest `pythonpath`. uv was not installed | root `pyproject.toml` |
