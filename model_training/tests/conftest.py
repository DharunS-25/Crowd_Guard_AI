"""conftest.py — pytest configuration for CrowdGuard AI tests."""

import sys
from pathlib import Path

# Add src/ to path so all imports work during testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
