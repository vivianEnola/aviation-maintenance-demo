from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_runtime() -> Path:
    """Keep libraries from writing configuration outside the project."""
    if os.name == "nt" and "WINDIR" not in os.environ:
        os.environ["WINDIR"] = os.environ.get("SystemRoot", r"C:\Windows")
    runtime_root = PROJECT_ROOT / ".runtime"
    yolo_root = runtime_root / "ultralytics"
    (yolo_root / "Ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_root))
    os.environ.setdefault("MPLCONFIGDIR", str(runtime_root / "matplotlib"))
    # Polars 1.42 can mis-detect feature flags on some Windows Core Ultra CPUs.
    # The processor supports the required SSE baseline; bypass only that startup check.
    os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")
    (runtime_root / "matplotlib").mkdir(parents=True, exist_ok=True)
    return runtime_root
