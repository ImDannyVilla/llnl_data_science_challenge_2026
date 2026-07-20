#!/usr/bin/env python3
"""Print metadata and basic statistics for NumPy .npy arrays as JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _finite_number(value: np.generic | float | int) -> int | float | None:
    """Convert a NumPy scalar to a JSON-safe number."""
    converted = value.item() if isinstance(value, np.generic) else value
    if isinstance(converted, bool):
        return int(converted)
    if isinstance(converted, int):
        return converted
    number = float(converted)
    return number if math.isfinite(number) else None


def inspect_array(filepath: str) -> dict[str, Any]:
    """Return read-only metadata for one numeric .npy file."""
    path = Path(filepath).expanduser().resolve()
    if path.suffix.lower() != ".npy":
        raise ValueError(f"Expected a .npy file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")

    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype.kind not in "buif":
        raise TypeError(f"Expected a real numeric array, found dtype {array.dtype}")

    minimum = maximum = mean = None
    if array.size:
        minimum = _finite_number(np.min(array))
        maximum = _finite_number(np.max(array))
        mean = _finite_number(np.mean(array, dtype=np.float64))

    return {
        "path": str(path),
        "shape": list(array.shape),
        "ndim": int(array.ndim),
        "dtype": str(array.dtype),
        "voxel_count": int(array.size),
        "array_bytes": int(array.nbytes),
        "file_bytes": int(path.stat().st_size),
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect one or more numeric NumPy .npy arrays."
    )
    parser.add_argument("files", nargs="+", help="Paths to .npy arrays")
    args = parser.parse_args()

    try:
        results = [inspect_array(filepath) for filepath in args.files]
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    print(json.dumps(results, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
