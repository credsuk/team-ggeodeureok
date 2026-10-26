#!/usr/bin/env python3
"""임베딩 및 벡터 DB 벤치마크 환경을 기록한다."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "environment" / "environment.json"


def command_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True, timeout=15)
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def main() -> None:
    disk = shutil.disk_usage(PROJECT_ROOT)
    proc_version = Path("/proc/version").read_text(errors="ignore") if Path("/proc/version").exists() else ""
    report: dict[str, object] = {
        "project_root": str(PROJECT_ROOT),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_logical": os.cpu_count(),
        "is_wsl": "microsoft" in proc_version.lower(),
        "project_on_windows_mount": str(PROJECT_ROOT).startswith("/mnt/"),
        "disk_total_gib": round(disk.total / 1024**3, 2),
        "disk_free_gib": round(disk.free / 1024**3, 2),
        "nvidia_smi": command_output([
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]),
    }

    try:
        import psutil  # type: ignore

        report["ram_total_gib"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except ImportError:
        report["ram_total_gib"] = None

    try:
        import torch  # type: ignore

        report["torch_version"] = torch.__version__
        report["torch_cuda_available"] = torch.cuda.is_available()
        report["torch_cuda_version"] = torch.version.cuda
        report["gpu_name_from_torch"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        report["torch_version"] = None
        report["torch_cuda_available"] = False

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["project_on_windows_mount"]:
        print("\n경고: /mnt/c 같은 Windows 마운트가 아니라 WSL의 ~/projects 아래에 프로젝트를 두세요.")
    if not report["torch_cuda_available"]:
        print("\n주의: PyTorch에서 CUDA GPU를 확인하지 못했습니다. 임베딩 전에 설치 상태를 점검하세요.")
    print(f"\n환경 기록: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

