---
paths:
  - "apps/api/**"
  - "apps/worker/**"
  - "packages/platform/**"
  - "packages/contracts/**"
  - "crowdguard/**/*.py"
---

# Backend, worker, and processing rules

- Routers (`crowdguard_api/routers/`) parse requests and format responses. Services orchestrate. Repositories and the object store live in `crowdguard_platform`. Frame and model work runs only in `crowdguard_worker` through `crowdguard` core.
- Request and response models come from `crowdguard_contracts`. Change them only in a contract task, together with the docs and the regenerated `packages/api-client`.
- Routes live under `/api/v1`. Check ownership before any other lookup, and return `404` for both missing and not-owned resources. Error bodies carry a stable `ErrorCode` and a safe message; tracebacks go only to server logs.
- Analysis states: `awaiting_upload → validating → ready → queued → processing → completed | failed | cancelled`. Every transition is a compare-and-set. Repeated `start`, `cancel`, and `complete` calls are idempotent.
- The worker claims jobs with leases and heartbeats, uses a new `run_id` per attempt and a per-job temp dir, and enforces per-stage and whole-job timeouts. On any failure it deletes the temp dir and the `runs/{run_id}/` prefix. Only fully registered runs are served.
- Probe the real media (container, codec, duration, fps, resolution) before a job reaches `ready`. MIME type and extension are hints, not proof.
- Validate zone polygons, capacity provenance, fps, and feature timestamps before inference.
- Serve artifacts only by enum name through short-lived presigned GETs for registered artifacts. Never read a path or key supplied by the client.
- `crowdguard/main.py` CLI flags and CSV meaning are frozen. Task 2's golden test must stay green.
