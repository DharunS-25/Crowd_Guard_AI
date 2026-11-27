# CrowdGuard end-to-end acceptance cases

**Status:** [Proposed] acceptance specification, revision 2 (2026-11-26). None of these can run against the current Streamlit prototype. They target the API, worker, UI, and S3 design in [system-architecture.md](../architecture/system-architecture.md), [s3-upload.md](../architecture/s3-upload.md), and [risk-assessment.md](../architecture/risk-assessment.md). Each case lists the plan task that must implement it ([plan](../superpowers/plans/2026-11-26-crowdguard-product-architecture.md)).

## 1. Harness

- **Services:** FastAPI, one worker process, the Vite app, a temporary SQLite file, a temporary worker temp root, and an **S3-compatible local endpoint** (D-16) with a per-run bucket. That bucket gets the same CORS template as production (app origin only, `ExposeHeaders: ETag`). No AWS account or network access in CI.
- **Tools:** Playwright for browser journeys (network interception for fault injection), and pytest with httpx and boto3 against the local endpoint for API and S3 assertions.
- **Test configuration:** multipart threshold 5 MiB, part size 5 MiB (the S3 minimum), concurrency 3, short presign TTLs for the expiry cases, small duration and size limits, and an injectable clock for session expiry.
- **Fixtures are generated at setup and never committed:**
  - `tiny.mp4`: about 3 s, known fps, dimensions, and frame count; single-PUT path.
  - `multi.mp4`: a high-entropy synthetic video of about 12 MiB, giving 3 parts.
  - `short.mp4`: shorter than the fake model's `sequence_length`.
  - `empty_scene.mp4`: no people.
  - `long.mp4`: over the test duration limit.
  - `corrupt.mp4`: random bytes. `text_as.mp4`: a text file renamed.
- **Injected components:**
  - a fake detector with deterministic boxes per frame;
  - `fake-venue-risk` adapter (label space `venue_risk_v1`, deterministic per zone and window: Zone A → HIGH, Zone B → LOW, Zone C → MEDIUM);
  - `fake-binary-anomaly` adapter (label space `ucsd_binary_anomaly`);
  - failing, raising, and schema-mismatch adapters;
  - fixture policies marked `DEMONSTRATION`.
  - No real YOLO or model weights are downloaded.
- **Principals:** two test owners (A and B), injected through the test auth dependency.
- **Isolation:** each case uses fresh IDs. After each case, assert that the temp root is empty and the bucket contains only the expected prefixes.
- **Visual checks** assert on `overlay_manifest.json` plus a tolerant interior-pixel sample from decoded frames. Never compare an MP4 byte-for-byte.

## 2. Upload (S3 direct, multipart)

| ID | Scenario | Expected result | Task |
|---|---|---|---|
| UP-01 | Small file, single presigned PUT | Session `uploaded`; analysis `validating` then `ready`; object at the generated key; preview frame exists. | 5, 6, 8 |
| UP-02 | Multipart happy path (`multi.mp4`) | 3 parts, numbered consecutively from 1; at most 3 PUTs in flight (network log); complete succeeds; `HeadObject` size equals the file size. | 5, 8 |
| UP-03 | One part fails (Playwright aborts the first PUT of part 2) | Only part 2 is retried; parts 1 and 3 are sent once; the upload completes. | 8 |
| UP-04 | Presigned part URL expires before use | The PUT gets 403; the client requests a new URL for that part only; the upload completes. | 5, 8 |
| UP-05 | Resume after page reload mid-upload | `GET /uploads/{id}` returns the S3-verified parts 1–2; after reselecting the same file, only part 3 is uploaded; complete succeeds. | 5, 8 |
| UP-06 | Resume with a different file selected | The client rejects it (size or part-hash mismatch) before any PUT; the session is unchanged. | 8 |
| UP-07 | User cancels mid-upload | In-flight PUTs are aborted; `DELETE` aborts the multipart upload; `ListParts` shows nothing remains; session `aborted`; analysis `cancelled`; a repeat `DELETE` is idempotent. | 5, 8 |
| UP-08 | Complete with a tampered part list (wrong ETag, missing part, extra or out-of-range part number, gap) | `UPLOAD_PART_MISMATCH` naming the parts; the state returns to `uploading`; no object is created; S3's list, not the client's, is used. | 5 |
| UP-09 | Complete is called twice concurrently, again later, and after an injected crash between `CompleteMultipartUpload` and the DB commit | Exactly one analysis and one `validate_upload` job; identical responses; the crash retry reconciles through `HeadObject`. | 5 |
| UP-10 | Part bytes do not match the declared checksum | S3 rejects the PUT; the client re-hashes and retries; a persistent mismatch never completes. | 0.2, 5, 8 |
| UP-11 | Declared size over the limit; part-URL request for a part number above `part_count` or with the wrong part size | Create returns a 4xx with a stable code and starts no multipart upload; the part-URL request is refused. | 5 |
| UP-12 | `corrupt.mp4`, `text_as.mp4` | Probe rejects with `UPLOAD_REJECTED_*`; source object deleted; analysis `failed`; `start` returns 409; no job queued. | 6 |
| UP-13 | `long.mp4` over the duration limit | Rejected at probe with `UPLOAD_REJECTED_DURATION`; no processing resources used. | 6 |
| UP-14 | Secret exposure check | No API response, DOM, console log, `localStorage`, or `sessionStorage` contains the bucket name, object key, S3 `UploadId`, or AWS credentials; presigned URLs are not persisted or logged. | 5, 8 |
| UP-15 | CORS from a non-app origin | The preflight for PUT to the bucket fails; the app origin succeeds. | 0.2, 10 |
| UP-16 | Quota and rate limit exceeded (active sessions, bytes per day, session creates per minute) | 429 with a stable code; nothing is created in S3. | 5 |

## 3. Processing and job lifecycle

| ID | Scenario | Expected result | Task |
|---|---|---|---|
| PR-01 | Start a `ready` analysis with a valid zone config | `queued → processing → completed`; the stage and progress advance; results and artifacts are available; the UI never shows an earlier job's output. | 4, 6, 7, 8 |
| PR-02 | Start before `ready`, or start twice | 409 before ready; a second start returns the current state with no duplicate job. | 7 |
| PR-03 | Invalid zone config (self-intersecting polygon, coordinates out of [0,1], duplicate zone IDs) | Rejected at `start` with `ZONE_CONFIG_INVALID` and details; no job; no risk output. | 4, 7 |
| PR-04 | Worker killed mid-processing | The lease expires; the job is reclaimed with a new `run_id`; the final result references only the completed run; the crashed run's prefix is deleted; after `max_attempts`, `failed` with `WORKER_LEASE_EXPIRED`. | 6 |
| PR-05 | Encoder or output-upload failure injected | `failed` with `ENCODE_FAILED` or `OUTPUT_UPLOAD_FAILED`; run prefix and temp dir deleted; other analyses untouched. | 6 |
| PR-06 | Cancel during processing | Reaches `cancelled` at the next stage or frame check; temp dir and run prefix cleaned; no artifacts listed. | 6, 7 |
| PR-07 | UI polling | Renders `awaiting_upload`, `validating`, `ready`, `queued`, `processing` with stage, `completed`, `failed` with a safe message and retry path, and `cancelled`; polling stops at a terminal state. | 8 |
| PR-08 | `short.mp4` shorter than `sequence_length` | Completes; affected windows are `INSUFFICIENT_DATA` with `WARMUP_INCOMPLETE`; never `LOW`. | 3, 6 |
| PR-09 | `empty_scene.mp4` | Counts are 0 with the `NO_DETECTIONS` warning; zero is reported as an observation, not as missing data; the level follows the fixture policy. | 2, 3 |

## 4. Model evidence and fusion

| ID | Scenario | Expected result | Task |
|---|---|---|---|
| MD-01 | Deterministic `fake-venue-risk` adapter with the demo policy | Expected HIGH, LOW, and MEDIUM per zone and window, with probabilities, reason codes, and policy version; `DEMONSTRATION` banner in the UI and on the video. | 3, 6, 8 |
| MD-02 | Adapter artifact missing or hash mismatch | Evidence `MODEL_UNAVAILABLE`; assessment `INSUFFICIENT_DATA` (`NO_ELIGIBLE_SEVERITY_EVIDENCE`); the job completes; no other model is substituted. | 3 |
| MD-03 | Adapter raises on one window | That window's evidence is `ERROR`; other windows, zones, and adapters are unaffected. | 3 |
| MD-04 | Feature columns reordered or renamed relative to the descriptor | `SCHEMA_MISMATCH`; no probabilities; columns are never reordered or filled. | 3 |
| MD-05 | `fake-binary-anomaly` adapter enabled next to `fake-venue-risk` | Binary evidence is shown as experimental anomaly evidence; levels are identical to a run without it; it never colors a zone. | 3, 8 |
| MD-06 | Two eligible same-label-space adapters disagree | With an explicit disagreement rule: the expected level plus `MODEL_DISAGREEMENT`, and both evidence items visible. Without a rule: the policy fails to load with `POLICY_INCOMPLETE` before processing. | 3 |
| MD-07 | NaN or inf injected into a feature window | `FEATURE_INVALID`; `INSUFFICIENT_DATA`; no zero-filled prediction. | 3 |
| MD-08 | Default policy and no eligible severity model (production-like config) | All windows `INSUFFICIENT_DATA` with `NO_ELIGIBLE_SEVERITY_EVIDENCE` and/or `POLICY_THRESHOLDS_UNCONFIGURED`; the UI shows grey/hatched unknown; no `LOW` anywhere. | 3, 8 |
| MD-09 | Zone capacity unconfigured | `occupancy_ratio` is `null` with `CAPACITY_UNCONFIGURED`; not computed from demo capacities. | 2 |

## 5. Risk overlays

| ID | Scenario | Expected result | Task |
|---|---|---|---|
| OV-01 | Manifest compared with results | For every frame range: `zone_id`, `window.index`, and level equal the `results.json` window; frames cover the whole video. | 3, 6 |
| OV-02 | Interior pixel sample | A decoded frame inside a Zone A HIGH window has a sampled interior color within tolerance of the HIGH palette token, and never the LOW token. | 3 |
| OV-03 | Window boundaries | Frames at `start_s` and just before `end_s` map to the same window; no one-window shift. | 3 |
| OV-04 | Unknown and tail frames | Frames without a window, or with `INSUFFICIENT_DATA`, show the grey/hatched style and the text label; the status banner is always present. | 3 |
| OV-05 | Playability | `annotated.mp4` is H.264/yuv420p with faststart (probe); it plays in the browser `<video>`; its duration matches the source within one frame interval. | 6, 8 |
| OV-06 | UI sync | During playback the zone table highlights the row for the window that contains the current time; the row matches the overlay. | 8 |

## 6. Output downloads

| ID | Scenario | Expected result | Task |
|---|---|---|---|
| DL-01 | Download every artifact | `results.json`, `features.csv`, `zone_risk.csv`, `overlay_manifest.json`, `annotated.mp4`, and `manifest.json` download; SHA-256 and size match `manifest.json`; all carry the same `analysis_id`, `run_id`, and schema, model, and policy versions as the UI. | 6, 7, 8 |
| DL-02 | CSV consistency | Columns follow the schema order; rows = zones × windows; levels, reasons, and quality states equal `results.json`. | 6 |
| DL-03 | Expired download URL | The expired URL fails; requesting the artifact again from the API returns a fresh URL that works. | 7 |
| DL-04 | Artifact name outside the enum; traversal strings (`../`, encoded variants) | 404 with no file or object read; nothing derived from the input. | 7 |

## 7. Isolation and security

| ID | Scenario | Expected result | Task |
|---|---|---|---|
| IS-01 | Two uploads with the same filename | Distinct analysis IDs, keys, runs, and results; neither UI view nor download shows the other job. | 5, 7, 8 |
| IS-02 | Owner B requests owner A's upload, analysis, results, artifacts, or cancel | 404 for every route; no presigned URL issued; an audit row records the denial. | 5, 7 |
| IS-03 | Unknown IDs and malformed IDs | Safe 404 or 422; no stack trace or internal path in the body. | 5, 7 |
| IS-04 | `display_name` containing HTML, script, or path characters | Rendered escaped; absent from all object keys and temp paths. | 5, 8 |
| IS-05 | Audit trail | Rows exist for session create, complete, abort, start, cancel, download-URL issue, and delete, with owner and analysis IDs and no presigned URLs. | 5, 7 |

## 8. Cleanup and retention

| ID | Scenario | Expected result | Task |
|---|---|---|---|
| CL-01 | Session passes its TTL (injected clock) | The sweeper aborts the multipart upload; `ListParts` shows nothing remains; session `expired`; analysis `failed` with `UPLOAD_EXPIRED`. | 5 |
| CL-02 | Failed or cancelled job | The temp dir and `runs/{run_id}/` are removed; other analyses' objects are byte-identical before and after. | 6 |
| CL-03 | Delete an analysis or run the retention job | The `uploads/{id}/` and `analyses/{id}/` prefixes and DB rows are removed; audit rows are kept per D-14; another analysis is untouched. | 7 |
| CL-04 | Infrastructure template check (static) | The lifecycle template contains `AbortIncompleteMultipartUpload`; Block Public Access is on; the TLS-deny policy is present; CORS origins contain no `*`; `ExposeHeaders` includes `ETag`. | 10 |
| CL-05 | End of suite | The worker temp root is empty; no orphan multipart uploads in the test bucket (`ListMultipartUploads`). | 9 |

## 9. Future acceptance (not MVP)

- **Multi-camera:** fusion refuses uncalibrated or time-skewed cameras; the overlap and deduplication policy is applied; track IDs stay camera-local.
- **Evacuation planning:** outputs carry `DRAFT_FOR_HUMAN_REVIEW` with input versions and assumptions; no automated dispatch, public alert, or PA path exists; scenario simulations and expert review are recorded before any pilot.
