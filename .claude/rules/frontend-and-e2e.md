---
paths:
  - "apps/web/**"
  - "packages/api-client/**"
  - "tests/e2e/**"
  - "tests/fixtures/**"
---

# Frontend and E2E rules

- Use the generated `packages/api-client`; do not hand-write API types. The UI never calls the pipeline directly and never holds AWS credentials.
- Upload client: hash each part, keep at most `max_concurrency` PUTs in flight, retry only failed parts with backoff, renew expired part URLs, resume from `GET /uploads/{id}` (the server's part list), and abort cleanly. Keep only `upload_id` in `sessionStorage`; never store or log presigned URLs.
- Show every analysis state, including `validating`, `ready`, and `cancelled`, with recovery actions. Never show an earlier job's result for a new analysis.
- Zone labels, risk colors, window timestamps, unknown states, and the `EXPERIMENTAL`/`DEMONSTRATION` banner must match the overlay, `results.json`, and the CSVs. Pair color with text or icons.
- Show model evidence (for example, experimental anomaly evidence) separately from the zone risk level. It must never set the zone color.
- E2E tests use generated fixture videos, injected fake detector and adapters, and an S3-compatible local endpoint. No network downloads, AWS account, live camera, or real weights.
- Assert both the API payload and the generated artifacts. For visuals, compare the overlay manifest and tolerant interior pixel samples, never a byte-for-byte MP4.
- After each case, the worker temp root is empty and no orphan multipart uploads remain.
