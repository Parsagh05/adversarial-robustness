"""Filesystem contracts shared by generation/evaluation orchestration."""

from __future__ import annotations

from pathlib import Path
import shutil


PROTOCOL_FILENAMES = (
    "attack_train_indices.csv",
    "evaluation_test_indices.csv",
)
BUNDLE_DIRECTORY = "canonical_clip_per_dataset_segmentation_loss_v2"


def _plain_name(label: str, value: str) -> str:
    if not value or value in {".", ".."} or set(value) & set("/\\"):
        raise ValueError(f"{label} must be a plain directory name, got {value!r}")
    return value


def setup_root(results_root: Path, setup_id: str) -> Path:
    return results_root / "setups" / _plain_name("setup_id", setup_id)


def condition_root(results_root: Path, setup_id: str, loss_formulation: str) -> Path:
    """Return the folder that holds one setup's one loss formulation.

    Every axis a run is allowed to subset has to appear in the path. Generation
    rewrites ``attack_manifest.csv`` for the whole bundle it writes into, and
    evaluation decides completeness from the row count of one ``summary.csv``,
    so two partial runs sharing a folder would silently discard each other's
    work.
    """

    return setup_root(results_root, setup_id) / _plain_name(
        "loss_formulation", loss_formulation
    )


def bundle_path(results_root: Path, setup_id: str, loss_formulation: str) -> Path:
    return (
        condition_root(results_root, setup_id, loss_formulation)
        / "attack_generation"
        / BUNDLE_DIRECTORY
    )


def evaluation_mode_root(
    results_root: Path, setup_id: str, loss_formulation: str, threshold_mode: str
) -> Path:
    return (
        condition_root(results_root, setup_id, loss_formulation)
        / "evaluation"
        / _plain_name("threshold_mode", threshold_mode)
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
