"""Write actual interpreter and installed package versions to a JSON file."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

PACKAGES = ("chromadb", "sentence-transformers", "torch", "langchain-text-splitters", "openai", "pydantic", "numpy")


def environment_report():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        hardware = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        hardware = []
    return {"python": platform.python_version(), "packages": packages, "gpu": hardware}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("model/environment.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(environment_report(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
