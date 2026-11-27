# Bounded implementation prompt (template)

Paste this into Claude Code from the repository root. Fill in **one** plan task. Never assign two tasks that touch the same contract or workspace files at the same time.

```text
You are implementing exactly one task of CrowdGuard AI.

TASK: <Task ID and title from docs/superpowers/plans/2026-11-26-crowdguard-product-architecture.md>
ALLOWED FILES: <copy the task's allowed-file list verbatim>
DEPENDS ON (must already be complete): <task IDs>

Before editing, read:
- CLAUDE.md and .claude/rules/agent-boundaries.md
- the path-scoped .claude/rules/ files that match your allowed files
- docs/architecture/system-architecture.md, repository-structure.md, and whichever of s3-upload.md / risk-assessment.md the task references
- the E2E cases (docs/testing/e2e-test-cases.md) the task lists under "Covers"

Rules:
1. Inspect the current code first. Report what exists, confirm the dependencies are done, and list the exact files you will touch (a subset of ALLOWED FILES).
2. Edit only ALLOWED FILES. If you need anything else (a shared contract, a workspace or tooling file, a new dependency, a rename, move, or delete, app.py, model_training/, config/, or CLI flags or CSV meaning), STOP and report why. Do not work around it.
3. TDD: write a focused failing test, run it, implement the smallest change, rerun it, then run the task's check command. Use deterministic fakes. No model-weight downloads, AWS account, or live camera.
4. Respect the monorepo dependency rules (crowdguard_api never imports crowdguard core, torch, or OpenCV; product code never imports model_training).
5. Risk rules: never map UCSD Normal/Anomalous to LOW/MEDIUM/HIGH; never average or vote across models without a shared validated label space; never invent thresholds; INSUFFICIENT_DATA is never LOW; keep each model's raw evidence visible.
6. Upload rules: browser-to-private-S3 only; never expose the object key, the S3 UploadId, or credentials; verify with S3 before completing; only verified and probed objects are queued.
7. Do not tick a plan checkbox or claim a check passed unless you ran it.

Finish with a report: changed files; exact commands and results; E2E IDs covered; open decisions (D-xx) hit; blockers; the next task.
```
