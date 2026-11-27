"""Monorepo import boundaries (docs/architecture/repository-structure.md §3)."""

import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHONPATH = os.pathsep.join(
    str(ROOT / p) for p in [".", "packages/contracts", "packages/platform", "apps/api", "apps/worker"]
)

BLOCKER = """
import importlib.abc, sys
BLOCKED = {blocked!r}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in BLOCKED:
            raise ImportError('blocked by boundary test: ' + name)
sys.meta_path.insert(0, Block())
import {module}
print('ok')
"""


def _import_with_blocked(module: str, blocked: set[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": PYTHONPATH}
    code = BLOCKER.format(blocked=sorted(blocked), module=module)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT)


def _api_modules() -> list[str]:
    pkg = ROOT / "apps" / "api" / "crowdguard_api"
    return [
        "crowdguard_api." + ".".join(p.relative_to(pkg).with_suffix("").parts)
        for p in pkg.rglob("*.py")
        if p.name != "__init__.py"
    ] or ["crowdguard_api"]


def test_api_imports_without_core_torch_or_opencv():
    for module in _api_modules():
        result = _import_with_blocked(module, {"crowdguard", "torch", "cv2", "ultralytics", "crowdguard_worker"})
        assert result.returncode == 0, f"{module}: {result.stderr[-2000:]}"


def test_contracts_import_with_only_pydantic():
    result = _import_with_blocked(
        "crowdguard_contracts", {"crowdguard", "crowdguard_platform", "numpy", "cv2", "boto3", "fastapi"}
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_product_code_never_imports_model_training():
    offenders = []
    for base in ["crowdguard", "packages", "apps"]:
        for path in (ROOT / base).rglob("*.py"):
            if "node_modules" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            if "import model_training" in text or "from model_training" in text:
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
