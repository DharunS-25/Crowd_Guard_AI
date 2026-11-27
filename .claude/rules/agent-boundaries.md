# Agent boundaries

- Work on **one** task from `docs/superpowers/plans/2026-11-26-crowdguard-product-architecture.md` at a time, and edit only that task's allowed files. If a needed change falls outside them, stop and report.
- Shared contracts (`packages/contracts/**`, the generated `packages/api-client/**`, `docs/architecture/**`) and workspace or tooling files (root `pyproject.toml`, lock files, `package.json`, `pnpm-workspace.yaml`, `.gitignore`, CI) change only in their assigned tasks, never in parallel with other work.
- Default to additive, backwards-compatible changes. Stop and report before renaming, moving, or deleting files; changing CLI flags or CSV meaning; breaking a contract; adding an unlisted dependency; or touching `app.py`, `model_training/**`, or `config/**`.
- Architecture-only tasks may edit only `CLAUDE.md`, `.claude/rules/**`, `docs/architecture/**`, `docs/testing/**`, `docs/superpowers/plans/**`, and `docs/prompts/**`. They run no product tests and change no product code.
- Report exact commands and outcomes. Never claim a check passed unless it ran. Tick plan checkboxes only for completed, verified steps.
- Never commit footage, model weights, databases, `.env` files, credentials, or real cloud account identifiers.
