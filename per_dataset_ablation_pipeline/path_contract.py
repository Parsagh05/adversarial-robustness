"""Filesystem contracts shared by generation/evaluation orchestration."""

from __future__ import annotations

from pathlib import Path
import shutil


PROTOCOL_FILENAMES = (
    "attack_train_indices.csv",
    "evaluation_test_indices.csv",
)
BUNDLE_DIRECTORY = "canonical_clip_per_dataset_segmentation_loss_v2"


def bundle_path(results_root: Path, setup_id: str) -> Path:
    return (
        results_root
        / "setups"
        / setup_id
        / "attack_generation"
        / BUNDLE_DIRECTORY
    )


def ensure_bundle_protocol_files(bundle: Path) -> tuple[Path, ...]:
    """Ensure a directory bundle contains both protocol CSVs.

    Older generator runs wrote the CSVs only to ``attack_generation/protocol``.
    Copying those small files is safe and avoids regenerating perturbations.
    """

    bundle = Path(bundle).expanduser().resolve()
    repaired: list[Path] = []
    for filename in PROTOCOL_FILENAMES:
        destination = bundle / filename
        if destination.is_file():
            continue
        source = bundle.parent / "protocol" / filename
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing {filename} in both {bundle} and {source.parent}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        repaired.append(destination)
    return tuple(repaired)

