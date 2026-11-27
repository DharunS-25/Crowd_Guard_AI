---
paths:
  - "apps/api/crowdguard_api/routers/uploads.py"
  - "apps/api/crowdguard_api/services/uploads.py"
  - "apps/api/crowdguard_api/services/sweeper.py"
  - "apps/api/crowdguard_api/services/retention.py"
  - "packages/platform/crowdguard_platform/storage/**"
  - "apps/web/src/features/upload/**"
  - "infra/**"
---

# S3 upload and storage rules

Read `docs/architecture/s3-upload.md` before editing these files.

- Video bytes go from the browser directly to a private S3 bucket through presigned URLs. FastAPI never proxies video bytes.
- The server creates the analysis and upload session first and generates the key `uploads/{analysis_id}/source`. The object key, bucket, and real S3 `UploadId` stay server-side. Clients see only opaque `upload_id` and `analysis_id`.
- Use a single presigned PUT below the multipart threshold, and multipart above it. Defaults: 100 MiB threshold, 16 MiB parts, 3 concurrent parts. All are configurable. Part numbers run consecutively from 1, and each part except the last is at least 5 MiB.
- Presigned URLs are short-lived bearer credentials. Issue them per part after an ownership check, renew them on expiry, and never log, persist, or send them to analytics.
- Completion verifies with S3, not with client claims: run `ListParts` (paginated), compare part numbers, ETags, checksums, and sizes, call `CompleteMultipartUpload` with the server's part list, then check size and checksum with `HeadObject`. Completion is idempotent (`Idempotency-Key`, CAS state, and `HeadObject` reconciliation after a crash).
- Only an S3-verified object that also passes the media probe may reach `ready` or the processing queue.
- Abort: call `AbortMultipartUpload`, then `ListParts` to confirm. A sweeper expires stale sessions. The bucket lifecycle rule `AbortIncompleteMultipartUpload` is mandatory.
- Treat filenames, MIME types, declared sizes, and client part lists as untrusted. Escape `display_name` and never use it in keys or paths.
- Bucket: Block Public Access, a TLS-only policy, default encryption, CORS restricted to the app origin with `ExposeHeaders: ETag`, and separate least-privilege IAM roles for the API signer and the worker. Templates under `infra/` contain placeholders only, never real account IDs or credentials.
- Enforce quotas, rate limits, size and duration limits, and audit rows from config. Do not hard-code product limits; open decision D-12 sets them.
