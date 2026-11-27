---
paths:
  - "model_training/**"
  - "crowdguard/models/**"
  - "crowdguard/risk/**"
  - "crowdguard/overlay.py"
  - "crowdguard/feature_schema.py"
  - "packages/contracts/crowdguard_contracts/evidence.py"
  - "packages/contracts/crowdguard_contracts/assessment.py"
---

# Model, fusion, and risk rules

Read `docs/architecture/risk-assessment.md` before changing labels, features, adapters, policies, thresholds, or overlays.

- Every model sits behind a versioned adapter with a `ModelDescriptor` that declares the model ID and version (derived from the artifact hash), label space, ordered class labels, feature schema and column order, sequence length, sampling, calibration, and validation status.
- Adapters return `ModelEvidence` for every zone and window, including failures (`INSUFFICIENT_DATA`, `SCHEMA_MISMATCH`, `MODEL_UNAVAILABLE`, `ERROR`, `SKIPPED_INELIGIBLE`). Never zero-fill, mean-fill, reorder, or rename features. Never substitute another model silently.
- Keep each model's raw evidence visible in the results. Fusion is a separate, versioned policy or a separately trained, calibrated fusion model. Do not average or majority-vote across models unless they share a validated label space and calibration, and the policy says so explicitly.
- The UCSD GRU/LSTM label space is `ucsd_binary_anomaly`. Never map `Normal`/`Anomalous` to `LOW`/`HIGH`, and never derive `MEDIUM` from confidence. It is experimental anomaly evidence only, with `eligible_for_severity = false`.
- Load model artifacts only from allow-listed SHA-256 hashes with `weights_only=True`. Do not unpickle scalers or checkpoints from untrusted sources.
- Missing, stale, invalid, ineligible, or unconfigured input gives `INSUFFICIENT_DATA` with reason codes, never `LOW`. A policy missing a required decision fails to load (`POLICY_INCOMPLETE`).
- Do not invent density, capacity, speed, persistence, or risk thresholds. Venue values need a source, owner, and revision. Image density and pixel speed are proxies.
- Keep time order and zone identity in every sequence. Never build a sequence across clips, cameras, zones, or jobs. Split training data by event, venue, or clip group before windowing, and fit scalers on training groups only.
- The overlay renders only the assessments it is given, using the same zone polygon and `(zone_id, window.index)` as the results, plus the assessment-status banner.
- `model_training/` is research. Product code does not import it. Port the architectures into `crowdguard/models/` and test parity.
