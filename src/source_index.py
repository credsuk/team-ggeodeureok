"""Backward-compatible entry point for the source metadata indexer.

Implementation lives in :mod:`src.data.source_index`.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.source_index import *  # noqa: F403
from src.data.source_index import main


if __name__ == "__main__":
    raise SystemExit(main())
