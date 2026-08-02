"""Save reproducible, inspectable experiment artifacts."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any


def _json_default(value: Any):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_json(path: str | Path, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return output


def save_run_artifacts(
    output_dir: str | Path,
    experiment_name: str,
    result: dict[str, Any],
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / f"{timestamp}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "metrics.json", result["metrics"])
    save_json(run_dir / "per_query.json", result["per_query"])
    save_json(run_dir / "experiment.json", result["experiment"])
    save_json(run_dir / "retrieval_runs.json", result["runs"])
    save_json(run_dir / "environment.json", collect_environment())
    return run_dir


def collect_environment() -> dict[str, Any]:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    try:
        direct_url_text = distribution("dapr-hhr").read_text("direct_url.json")
        package_source = json.loads(direct_url_text) if direct_url_text else None
    except (PackageNotFoundError, json.JSONDecodeError):
        package_source = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit,
        "package_source": package_source,
    }
