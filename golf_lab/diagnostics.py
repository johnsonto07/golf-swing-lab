"""Environment diagnostics, shared by the Settings page and the CLI.

Every probe here is defensive: hardware information is genuinely unavailable
in some environments, and a diagnostics screen that crashes is worse than one
that says "unknown".

Secrets are never revealed. The OpenAI row reports only enabled/disabled.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from golf_lab.config import (
    ANALYSIS_VERSION,
    APP_VERSION,
    DATA_DIR,
    MODELS_DIR,
    SWINGS_DIR,
    openai_integration_enabled,
)
from golf_lab.video.ffmpeg import find_ffmpeg

TRACKED_PACKAGES = (
    "streamlit",
    "opencv-python",
    "numpy",
    "scipy",
    "pydantic",
    "psutil",
    "pytest",
    "mediapipe",
)


@dataclass
class DiagnosticsReport:
    app_version: str
    analysis_version: str
    python_version: str
    python_executable: str
    platform_summary: str
    packages: Dict[str, str] = field(default_factory=dict)
    ffmpeg_version: Optional[str] = None
    ffmpeg_path: Optional[str] = None
    cpu: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    gpu: Dict[str, Any] = field(default_factory=dict)
    inference_device: str = "cpu"
    directories: List[Dict[str, Any]] = field(default_factory=list)
    openai_enabled: bool = False

    def to_lines(self) -> List[str]:
        lines = [
            "Golf Swing Lab diagnostics",
            "=" * 40,
            f"App version        : {self.app_version}",
            f"Analysis version   : {self.analysis_version}",
            f"Python             : {self.python_version}",
            f"Interpreter        : {self.python_executable}",
            f"Platform           : {self.platform_summary}",
            "",
            "Packages",
            "-" * 40,
        ]
        for name, version in self.packages.items():
            lines.append(f"  {name:<16}: {version}")
        lines += [
            "",
            "Media tooling",
            "-" * 40,
            f"  FFmpeg           : {self.ffmpeg_version or 'NOT FOUND'}",
            f"  FFmpeg path      : {self.ffmpeg_path or '-'}",
            "",
            "Hardware",
            "-" * 40,
            f"  CPU              : {self.cpu.get('summary', 'unknown')}",
            f"  Memory           : {self.memory.get('summary', 'unknown')}",
            f"  GPU              : {self.gpu.get('summary', 'unknown')}",
            f"  Inference device : {self.inference_device}",
            "",
            "Storage",
            "-" * 40,
        ]
        for entry in self.directories:
            lines.append(
                f"  {entry['label']:<16}: {entry['path']} "
                f"(writable={entry['writable']}, free={entry['free']})"
            )
        lines += [
            "",
            "Optional cloud features",
            "-" * 40,
            f"  OpenAI coaching  : {'enabled' if self.openai_enabled else 'disabled'} "
            "(key value is never displayed)",
        ]
        return lines

    def to_text(self) -> str:
        return "\n".join(self.to_lines())


def _package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not installed"
        except Exception:  # noqa: BLE001 - diagnostics must never crash
            versions[name] = "unknown"
    return versions


def _cpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"processor": platform.processor() or "unknown"}
    try:
        info["logical_cores"] = os.cpu_count()
    except Exception:  # noqa: BLE001
        info["logical_cores"] = None
    try:
        import psutil

        info["physical_cores"] = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001
        info["physical_cores"] = None

    cores = info.get("logical_cores") or "?"
    info["summary"] = f"{info['processor']} ({cores} logical cores)"
    return info


def _memory_info() -> Dict[str, Any]:
    try:
        import psutil

        virtual = psutil.virtual_memory()
        total_gb = virtual.total / 1024**3
        available_gb = virtual.available / 1024**3
        return {
            "total_gb": round(total_gb, 1),
            "available_gb": round(available_gb, 1),
            "summary": f"{total_gb:.1f} GB total, {available_gb:.1f} GB available",
        }
    except Exception:  # noqa: BLE001
        return {"summary": "unknown (psutil unavailable)"}


def _gpu_info() -> Dict[str, Any]:
    """Detect a GPU without requiring CUDA or any GPU library to be present."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            import subprocess

            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                names = [line.strip() for line in result.stdout.strip().splitlines()]
                return {"available": True, "devices": names, "summary": "; ".join(names)}
        except Exception:  # noqa: BLE001
            pass

    return {
        "available": False,
        "devices": [],
        "summary": "no NVIDIA GPU detected (CPU processing is fully supported)",
    }


def _directory_info() -> List[Dict[str, Any]]:
    entries = []
    for label, path in (
        ("data", DATA_DIR),
        ("swings", SWINGS_DIR),
        ("models", MODELS_DIR),
    ):
        entry: Dict[str, Any] = {"label": label, "path": str(path)}
        probe = path / ".golf_lab_write_test"
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe.write_text("ok", encoding="utf-8")
            # Writability is what matters here. Some mounts permit writes but
            # not unlinks, so a failed cleanup must not be reported as
            # "not writable".
            entry["writable"] = True
        except Exception:  # noqa: BLE001
            entry["writable"] = False
        finally:
            try:
                probe.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            usage = shutil.disk_usage(path if Path(path).exists() else Path.cwd())
            entry["free"] = f"{usage.free / 1024**3:.1f} GB"
        except Exception:  # noqa: BLE001
            entry["free"] = "unknown"
        entries.append(entry)
    return entries


def collect_diagnostics() -> DiagnosticsReport:
    ffmpeg_tools = find_ffmpeg(required=False)
    gpu = _gpu_info()

    return DiagnosticsReport(
        app_version=APP_VERSION,
        analysis_version=ANALYSIS_VERSION,
        python_version=platform.python_version(),
        python_executable=sys.executable,
        platform_summary=f"{platform.system()} {platform.release()} ({platform.machine()})",
        packages=_package_versions(),
        ffmpeg_version=ffmpeg_tools.version if ffmpeg_tools else None,
        ffmpeg_path=ffmpeg_tools.ffmpeg if ffmpeg_tools else None,
        cpu=_cpu_info(),
        memory=_memory_info(),
        gpu=gpu,
        # Milestone 1 does no inference at all; recorded here so the value is
        # already meaningful when Milestone 2 adds pose estimation.
        inference_device="cpu",
        directories=_directory_info(),
        openai_enabled=openai_integration_enabled(),
    )


def main() -> None:
    """CLI entry point: ``python -m golf_lab.diagnostics``"""
    print(collect_diagnostics().to_text())


if __name__ == "__main__":
    main()
