# Architecture rules

- `docs/architecture/system-architecture.md` is the target design. Its **[Current]**, **[Proposed]**, and **[Deferred]** tags mark the as-is/target boundary. Never describe a proposed component as implemented.
- Follow the monorepo layout and one-way dependency rules in `docs/architecture/repository-structure.md`:
  - new deployables go in `apps/`, shared libraries in `packages/`;
  - `crowdguard_api` never imports `crowdguard` core, torch, or OpenCV;
  - product code never imports `model_training/`.
- Keep `crowdguard/`, `model_training/`, `app.py`, and `config/` in place. Moving them is the reviewed migration in repository-structure §6, never a side effect.
- Keep video and model work in reusable Python modules. HTTP handlers validate, create jobs, and return results. Long work runs in the worker.
- Every upload gets a generated analysis ID and generated storage keys. Never derive a path or object key from a client filename or request field.
- Keep the UI, API, worker, model adapters, fusion policy, repositories, and object store behind explicit interfaces that deterministic fakes can replace in tests.
- Storage for the MVP: SQLite for metadata (repository interfaces allow a later PostgreSQL move), private S3 for media and artifacts, a per-job local temp dir in the worker.
- Do not implement multi-camera fusion, RTSP, alerts, or evacuation planning without an approved phase. Spatial fusion requires calibration, synchronized timestamps, and a venue-coordinate transform.
- Update the architecture docs and the plan in the same change whenever an accepted contract or component boundary changes.
