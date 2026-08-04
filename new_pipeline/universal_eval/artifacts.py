"""Loading and validation for canonical fixed-perturbation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import torch


MANIFEST_NAME = "all_canonical_attack_artifacts.json"
REQUIRED_FIELDS = {
    "source_dataset",
    "target_dataset",
    "direction",
    "loss_mode",
    "scope",
    "source_label",
    "target_label",
    "target_evaluation_all_sample_ids",
    "target_attacked_sample_ids",
    "artifact_path",
    "artifact_file_sha256",
    "epsilon",
    "image_size",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_embedded_path(root: Path, recorded_path: str) -> Path:
    """Resolve a path recorded on another machine relative to this archive."""

    normalized = str(recorded_path).replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if root.name in parts:
        suffix = parts[parts.index(root.name) + 1 :]
        return root.joinpath(*suffix)
    return root / PurePosixPath(normalized).name


@dataclass(frozen=True)
class AttackArtifact:
    root: Path
    record: dict[str, Any]
    delta_path: Path
    metadata_path: Path | None

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            str(self.record["source_dataset"]),
            str(self.record["target_dataset"]),
            str(self.record["direction"]),
            str(self.record["loss_mode"]),
            str(self.record["scope"]),
        )

    @property
    def name(self) -> str:
        return "__".join(self.key)

    @property
    def evaluation_ids(self) -> tuple[str, ...]:
        return tuple(self.record["target_evaluation_all_sample_ids"])

    @property
    def attacked_ids(self) -> tuple[str, ...]:
        return tuple(self.record["target_attacked_sample_ids"])

    def load_delta(self, verify_checksum: bool = True) -> torch.Tensor:
        if verify_checksum:
            actual = sha256_file(self.delta_path)
            expected = str(self.record["artifact_file_sha256"]).lower()
            if actual != expected:
                raise ValueError(
                    f"Artifact checksum mismatch for {self.delta_path}: "
                    f"expected {expected}, got {actual}"
                )
        try:
            payload = torch.load(
                self.delta_path, map_location="cpu", weights_only=True
            )
        except TypeError:  # PyTorch < 2.0
            payload = torch.load(self.delta_path, map_location="cpu")
        delta = payload.get("delta") if isinstance(payload, dict) else payload
        if not isinstance(delta, torch.Tensor):
            raise TypeError(f"No tensor named 'delta' in {self.delta_path}")
        if delta.ndim == 3:
            delta = delta.unsqueeze(0)
        image_size = int(self.record["image_size"])
        if tuple(delta.shape) != (1, 3, image_size, image_size):
            raise ValueError(
                f"Unexpected delta shape in {self.delta_path}: "
                f"{tuple(delta.shape)}, expected (1, 3, {image_size}, {image_size})"
            )
        delta = delta.detach().cpu().float().contiguous()
        if not torch.isfinite(delta).all():
            raise ValueError(f"Delta contains non-finite values: {self.delta_path}")
        epsilon = float(self.record["epsilon"])
        if float(delta.abs().max()) > epsilon + 5e-5:
            raise ValueError(
                f"Delta exceeds epsilon in {self.delta_path}: "
                f"{float(delta.abs().max())} > {epsilon}"
            )
        return delta


def load_manifest(
    root: str | Path,
    *,
    verify_files: bool = True,
    sources: Iterable[str] | None = None,
    targets: Iterable[str] | None = None,
) -> list[AttackArtifact]:
    root_path = Path(root).expanduser().resolve()
    manifest_path = root_path / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Canonical manifest not found: {manifest_path}")
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"Canonical manifest must be a non-empty list: {manifest_path}")

    source_filter = set(sources) if sources is not None else None
    target_filter = set(targets) if targets is not None else None
    artifacts: list[AttackArtifact] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise TypeError(f"Manifest record {index} is not an object")
        missing = REQUIRED_FIELDS - raw.keys()
        if missing:
            raise ValueError(f"Manifest record {index} is missing {sorted(missing)}")
        if source_filter is not None and raw["source_dataset"] not in source_filter:
            continue
        if target_filter is not None and raw["target_dataset"] not in target_filter:
            continue
        delta_path = _resolve_embedded_path(root_path, raw["artifact_path"])
        metadata_path = None
        if raw.get("metadata_path"):
            metadata_path = _resolve_embedded_path(root_path, raw["metadata_path"])
        artifact = AttackArtifact(root_path, raw, delta_path, metadata_path)
        if artifact.key in seen:
            raise ValueError(f"Duplicate attack condition in manifest: {artifact.key}")
        seen.add(artifact.key)

        evaluation_ids = artifact.evaluation_ids
        attacked_ids = artifact.attacked_ids
        if len(evaluation_ids) != len(set(evaluation_ids)):
            raise ValueError(f"Duplicate evaluation IDs in {artifact.name}")
        if len(attacked_ids) != len(set(attacked_ids)):
            raise ValueError(f"Duplicate attacked IDs in {artifact.name}")
        if not set(attacked_ids).issubset(evaluation_ids):
            raise ValueError(f"Attacked IDs are not a subset of evaluation IDs: {artifact.name}")
        if int(raw["source_label"]) == int(raw["target_label"]):
            raise ValueError(f"Source and target labels must differ: {artifact.name}")
        if verify_files and not delta_path.is_file():
            raise FileNotFoundError(f"Perturbation tensor not found: {delta_path}")
        if verify_files and metadata_path is not None and not metadata_path.is_file():
            raise FileNotFoundError(f"Perturbation metadata not found: {metadata_path}")
        artifacts.append(artifact)

    if not artifacts:
        raise ValueError("No manifest records matched the requested dataset filters")
    return artifacts


def json_safe_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable defensive copy used in result snapshots."""

    return json.loads(json.dumps(record, allow_nan=False))

