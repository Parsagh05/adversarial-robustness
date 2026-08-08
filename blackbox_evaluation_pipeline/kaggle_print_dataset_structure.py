"""Print the mounted Kaggle dataset tree and a compact file summary.

This file can be pasted into a Kaggle notebook cell or uploaded and run as a
Python script. Add the dataset below to the notebook before running it:
https://www.kaggle.com/datasets/alirezasalehy/adversarial-attacks-vlm-survey
"""

from collections import Counter
from pathlib import Path


DATASET_SLUG = "adversarial-attacks-vlm-survey"
DATASET_OWNER = "alirezasalehy"
MAX_DEPTH = None  # For example, set to 4 to show only the first four levels.
MAX_ENTRIES = 20_000  # Prevent accidental notebook-output overflow.


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def resolve_dataset_root() -> Path:
    candidates = (
        Path("/kaggle/input") / DATASET_SLUG,
        Path("/kaggle/input/datasets") / DATASET_OWNER / DATASET_SLUG,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    matches = [
        path
        for path in Path("/kaggle/input").rglob(DATASET_SLUG)
        if path.is_dir()
    ]
    if len(matches) == 1:
        return matches[0]

    mounted = sorted(path.name for path in Path("/kaggle/input").iterdir())
    raise FileNotFoundError(
        "Dataset mount was not found. Add the Kaggle dataset to this notebook "
        f"and rerun. Top-level mounts: {mounted}"
    )


def print_tree(root: Path) -> None:
    shown = 0

    print(f"Dataset root: {root}\n")
    print(f"{root.name}/")

    paths = sorted(
        root.rglob("*"),
        key=lambda path: str(path.relative_to(root)).lower(),
    )
    for path in paths:
        relative = path.relative_to(root)
        depth = len(relative.parts)
        if MAX_DEPTH is not None and depth > MAX_DEPTH:
            continue
        if shown >= MAX_ENTRIES:
            print(f"... output stopped after {MAX_ENTRIES:,} entries")
            break

        indent = "    " * (depth - 1)
        if path.is_dir():
            print(f"{indent}|-- {path.name}/")
        else:
            size = path.stat().st_size
            print(f"{indent}|-- {path.name}  ({human_size(size)})")
        shown += 1

    # Count the complete dataset even when display depth/entry limits are used.
    all_files = [path for path in root.rglob("*") if path.is_file()]
    all_directories = sum(path.is_dir() for path in root.rglob("*"))
    all_size = sum(path.stat().st_size for path in all_files)
    all_extensions = Counter(path.suffix.lower() or "[no extension]" for path in all_files)

    print("\n===== DATASET SUMMARY =====")
    print(f"Directories: {all_directories:,}")
    print(f"Files:       {len(all_files):,}")
    print(f"Total size:  {human_size(all_size)}")
    print("File types:")
    for suffix, count in all_extensions.most_common():
        print(f"  {suffix:<16} {count:,}")


if __name__ == "__main__":
    print_tree(resolve_dataset_root())
