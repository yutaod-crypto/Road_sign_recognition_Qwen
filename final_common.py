"""Shared GTSRB coarse-label definitions for Final (KD) scripts at repo root."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Milestone-aligned6-class grouping (43 fine classes -> 6 coarse)
GROUP_MAP: dict[str, set[int]] = {
    "speed_limit": {0, 1, 2, 3, 4, 5, 6, 7, 8, 32, 41, 42},
    "stop": {14},
    "yield": {13},
    "no_entry": {15, 16, 17},
    "warning": {18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
    "direction": {9, 10, 11, 12, 33, 34, 35, 36, 37, 38, 39, 40},
}

LABELS_6: list[str] = [
    "speed_limit",
    "stop",
    "yield",
    "no_entry",
    "warning",
    "direction",
]


def map_class_id(class_id: int) -> str:
    for name, ids in GROUP_MAP.items():
        if class_id in ids:
            return name
    raise ValueError(f"Unmapped GTSRB class id: {class_id}")


def label_to_index(label: str) -> int:
    if label not in LABELS_6:
        raise ValueError(f"Unknown label {label!r}, expected one of {LABELS_6}")
    return LABELS_6.index(label)
