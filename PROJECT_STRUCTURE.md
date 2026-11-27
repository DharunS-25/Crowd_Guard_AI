# CrowdGuard AI project structure

The repository is moving to a monorepo. This file shows the **current** layout (verified 2026-11-26) and the **planned** layout. Planned folders are created by the implementation plan tasks shown in brackets. Until then they do not exist. The full monorepo design, with dependency rules, is in [docs/architecture/repository-structure.md](docs/architecture/repository-structure.md).

## Current layout

```text
Crowd_Guard_AI/
├── .claude/
│   └── rules/                     # Path-scoped Claude Code rules
│       ├── agent-boundaries.md    #   always loaded: one task, allowed files only
│       ├── architecture.md        #   always loaded: layout and boundaries
│       ├── backend.md             #   apps/api, apps/worker, packages/*, crowdguard
│       ├── upload-and-storage.md  #   S3 upload, storage, infra
│       ├── ml-and-risk.md         #   model_training, adapters, risk, overlay
│       └── frontend-and-e2e.md    #   apps/web, api-client, tests/e2e
├── CLAUDE.md                      # Claude Code entry point and doc index
├── README.md                      # Setup and usage for the current prototype
├── app.py                         # Streamlit demo dashboard (kept until a reviewed migration)
├── config/
│   └── zones.json                 # Example zones, demo capacities, flow line
├── crowdguard/
│   ├── __init__.py
│   └── main.py                    # CLI: OpenCV + YOLO + ByteTrack + zones → CSV + annotated MP4
├── data/videos/.gitkeep           # Local authorized demo clips (git-ignored)
├── outputs/.gitkeep               # Prototype outputs (git-ignored)
├── model_training/                # Research only: UCSD Normal/Anomalous GRU/LSTM
│   ├── configs/                   #   dataset, GRU, LSTM settings
│   ├── data/{raw,processed}/      #   dataset and derived arrays (git-ignored going forward)
│   ├── outputs/{checkpoints,metrics,plots}/
│   ├── scripts/                   #   download, validate, train, evaluate
│   ├── src/                       #   preprocessing, models, training, inference
│   ├── tests/
│   ├── DATASET_CARD.md, MODEL_REPORT.md, MODEL_TRAINING_AUDIT.md, README.md
│   └── requirements.txt           #   training dependencies
├── docs/
│   ├── architecture/
│   │   ├── system-architecture.md #   current vs proposed, contracts, lifecycle, security
│   │   ├── repository-structure.md#   monorepo layout and dependency rules
│   │   ├── s3-upload.md           #   direct browser-to-S3 upload design
│   │   └── risk-assessment.md     #   labels, model evidence, fusion, open decisions
│   ├── testing/e2e-test-cases.md  #   acceptance cases
│   ├── superpowers/plans/         #   phased implementation plan
│   └── prompts/                   #   bounded implementation prompt template
├── requirements.txt               # Prototype dependencies (dashboard + pipeline)
└── .gitignore
```

## Planned monorepo layout

`[new]` marks additions. Existing folders keep their paths.

```text
Crowd_Guard_AI/
├── pyproject.toml, uv.lock        [new, Task 0.3]  Python workspace root (D-19)
├── package.json, pnpm-workspace.yaml, pnpm-lock.yaml  [new, Task 0.3]  TS workspace root
├── apps/                          [new] deployables
│   ├── api/       crowdguard_api      FastAPI: uploads, analyses, results   [Tasks 5, 7]
│   ├── worker/    crowdguard_worker   probe, analyze, artifacts             [Task 6]
│   └── web/       React + Vite        upload, zones, status, results player [Task 8]
├── packages/                      [new] shared libraries
│   ├── contracts/ crowdguard_contracts  shared models, enums, error codes   [Task 1]
│   ├── platform/  crowdguard_platform   SQLite, repositories, S3 store      [Tasks 4, 5]
│   └── api-client/                      generated TS client                 [Task 7]
├── crowdguard/                    existing core library + CLI
│   ├── main.py                    CLI (flags and outputs unchanged)
│   ├── processing.py, feature_schema.py, overlay.py   [new, Tasks 2, 3]
│   ├── models/                    [new] adapters, registry, orchestrator    [Tasks 3, 11]
│   └── risk/                      [new] decision policy, rule signals       [Tasks 3, 12]
├── model_training/                existing research (not a workspace member)
├── app.py, config/                existing prototype (frozen in Phase 1)
├── infra/s3/                      [new] bucket, CORS, lifecycle, IAM templates [Task 10]
├── tests/                         [new] core/, e2e/, fixtures/ (generators only)
├── docs/
└── data/, outputs/, var/          git-ignored runtime data (var/ = SQLite DB, temp)
```

Dependency direction:

- `apps/web` uses only `packages/api-client`.
- `apps/api` uses `contracts` and `platform`. It never uses `crowdguard` core, torch, or OpenCV.
- `apps/worker` uses `contracts`, `platform`, and `crowdguard`.
- `crowdguard` uses `contracts`.
- Nothing in the product imports `model_training/`. That folder publishes model artifacts only.

## Repository notes

- The Streamlit dashboard and the CLI keep working throughout Phase 1. Moving `crowdguard/` to `packages/core/` or `model_training/` to `ml/training/` is an optional, separately reviewed migration (Phase M in the plan), not part of feature work.
- `data/videos/` holds local, authorized footage. `outputs/` and `model_training/outputs/` are generated and may become large. Never commit footage, model weights, databases, or credentials.
- The ignore rules cover new untracked files in local data and output folders. Files already committed stay tracked until they are explicitly removed from Git's index. The checked-in training data and reports make clones heavier, and removing them is a separate decision.
- `model_training/README.md` mentions an `outputs/logs/` directory that does not exist in the current checkout, so it is omitted above.
- `app.py` imports `imageio_ffmpeg`, which `requirements.txt` does not list yet (see the README troubleshooting section).
