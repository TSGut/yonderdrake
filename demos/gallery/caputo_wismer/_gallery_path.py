"""Make shared gallery helpers importable from direct demo scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def add_gallery_path() -> None:
    gallery_directory = str(Path(__file__).resolve().parent.parent)
    if gallery_directory not in sys.path:
        sys.path.insert(0, gallery_directory)
