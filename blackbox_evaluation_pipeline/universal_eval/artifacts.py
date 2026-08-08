"""Loading and validation for the CSV-described canonical attack bundles."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import torch


MANIFEST_NAME = "attack_manifest.csv"
EVALUATION_INDEX_NAME = "evaluation_test_indices.csv"
BUNDLE_DIRECTORIES = {
    "per_dataset": "canonical_clip_per_dataset",
    "per_category": "canonical_clip_per_category",
    "per_image": "canonical_clip_per_image",
}
SCOPE_ALIASES = {
    "dataset": "per_dataset",
    "per_dataset": "per_dataset",
    "category": "per_category",
    "per_category": "per_category",
    "image": "per_image",
    "per_image": "per_image",
}
DIRECTION_LABELS = {
    "normal_to_abnormal": (0, 1),
    "abnormal_to_normal": (1, 0),
}
REQUIRED_MANIFEST_FIELDS = {
    "scope",
    "source_dataset",
    "target_dataset",
    "category",
    "direction",
    "source_label",
    "target_label",
    "loss_mode",
    "evaluation_attacked_image_count",
    "noise_file",
    "noise_tensor_key",
    "artifact_sha256",
    "image_size",
    "epsilon",
}
REQUIRED_EVALUATION_FIELDS = {
    "protocol_id",
    "dataset",
    "category",
    "label",
    "partition",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_scope(scope: str) -> str:
    try:
        return SCOPE_ALIASES[str(scope).strip().lower()]
    except KeyError as error:
        raise ValueError(
            f"Unknown attack scope {scope!r}; choose per_dataset, per_category, or per_image"
        ) from error


def _read_csv(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required attack CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = required_fields - fields
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Attack CSV must not be empty: {path}")
    return rows


def _resolve_noise_path(bundle_root: Path, recorded_path: str) -> Path:
    """Resolve the manifest's portable POSIX path inside its bundle."""

    normalized = str(recorded_path).replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if bundle_root.name in parts:
        parts = parts[parts.index(bundle_root.name) + 1 :]
    elif "noises" in parts:
        parts = parts[parts.index("noises") :]
    return bundle_root.joinpath(*parts)


def _bundle_roots(root: Path, scopes: tuple[str, ...]) -> list[tuple[str, Path]]:
    if (root / MANIFEST_NAME).is_file():
        return [("", root)]
    bundles: list[tuple[str, Path]] = []
    for scope in scopes:
        bundle = root / BUNDLE_DIRECTORIES[scope]
        if not bundle.is_dir():
            raise FileNotFoundError(
                f"The {scope} attack bundle is not mounted: {bundle}. "
                f"Expected it below the adversarial-attacks-vlm-survey dataset root."
            )
        bundles.append((scope, bundle))
    return bundles


@dataclass(frozen=True)
class AttackArtifact:
    bundle_root: Path
    record: dict[str, Any]
    delta_path: Path
    _evaluation_ids: tuple[str, ...]
    _attacked_ids: tuple[str, ...]

    @property
    def key(self) -> tuple[str, str, str, str, str, str]:
        return (
            str(self.record["source_dataset"]),
            str(self.record["target_dataset"]),
            str(self.record["direction"]),
            str(self.record["loss_mode"]),
            str(self.record["scope"]),
            str(self.record.get("category") or ""),
        )

    @property
    def name(self) -> str:
        parts = list(self.key[:5])
        if self.key[5]:
            parts.append(self.key[5])
        return "__".join(parts)

    @property
    def evaluation_ids(self) -> tuple[str, ...]:
        return self._evaluation_ids

    @property
    def attacked_ids(self) -> tuple[str, ...]:
        return self._attacked_ids

    def load_delta(self, verify_checksum: bool = True) -> torch.Tensor:
        if verify_checksum:
            actual = sha256_file(self.delta_path)
            expected = str(self.record["artifact_sha256"]).lower()
            if actual != expected:
                raise ValueError(
                    f"Artifact checksum mismatch for {self.delta_path}: "
                    f"expected {expected}, got {actual}"
                )
        try:
            payload = torch.load(self.delta_path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.0
            payload = torch.load(self.delta_path, map_location="cpu")
        tensor_key = str(self.record.get("noise_tensor_key") or "delta")
        delta = payload.get(tensor_key) if isinstance(payload, dict) else payload
        if not isinstance(delta, torch.Tensor):
            raise TypeError(f"No tensor named {tensor_key!r} in {self.delta_path}")
        if delta.ndim == 3:
            delta = delta.unsqueeze(0)
        image_size = int(self.record["image_size"])
        if delta.ndim != 4 or tuple(delta.shape[1:]) != (3, image_size, image_size):
            raise ValueError(
                f"Unexpected delta shape in {self.delta_path}: {tuple(delta.shape)}, "
                f"expected (N, 3, {image_size}, {image_size})"
            )
        scope = str(self.record["scope"])
        expected_count = len(self.attacked_ids) if scope == "per_image" else 1
        if int(delta.shape[0]) != expected_count:
            raise ValueError(
                f"{scope} tensor count mismatch in {self.delta_path}: "
                f"expected {expected_count}, got {int(delta.shape[0])}"
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

    def delta_indices(self) -> dict[str, int]:
        if self.record["scope"] == "per_image":
            return {sample_id: index for index, sample_id in enumerate(self.attacked_ids)}
        return {sample_id: 0 for sample_id in self.attacked_ids}


def load_manifest(
    root: str | Path,
    *,
    verify_files: bool = True,
    scopes: Iterable[str] | None = None,
    sources: Iterable[str] | None = None,
    targets: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
    directions: Iterable[str] | None = None,
    loss_modes: Iterable[str] | None = None,
) -> list[AttackArtifact]:
    """Load selected records from one bundle or the full Kaggle dataset root."""

    root_path = Path(root).expanduser().resolve()
    selected_scopes = tuple(
        dict.fromkeys(normalize_scope(scope) for scope in (scopes or ("per_dataset",)))
    )
    source_filter = set(sources) if sources is not None else None
    target_filter = set(targets) if targets is not None else None
    category_filter = set(categories) if categories is not None else None
    direction_filter = set(directions) if directions is not None else None
    loss_filter = set(loss_modes) if loss_modes is not None else None

    artifacts: list[AttackArtifact] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for forced_scope, bundle_root in _bundle_roots(root_path, selected_scopes):
        manifest_rows = _read_csv(
            bundle_root / MANIFEST_NAME, REQUIRED_MANIFEST_FIELDS
        )
        evaluation_rows = _read_csv(
            bundle_root / EVALUATION_INDEX_NAME, REQUIRED_EVALUATION_FIELDS
        )
        for index, raw in enumerate(manifest_rows):
            scope = normalize_scope(raw["scope"])
            if forced_scope and scope != forced_scope:
                raise ValueError(
                    f"{bundle_root / MANIFEST_NAME} contains scope {raw['scope']!r}, "
                    f"but directory {bundle_root.name!r} is the {forced_scope} bundle"
                )
            if scope not in selected_scopes:
                continue
            source = raw["source_dataset"]
            target = raw["target_dataset"]
            category = raw.get("category", "").strip()
            if source_filter is not None and source not in source_filter:
                continue
            if target_filter is not None and target not in target_filter:
                continue
            if category_filter is not None and category not in category_filter:
                continue
            if direction_filter is not None and raw["direction"] not in direction_filter:
                continue
            if loss_filter is not None and raw["loss_mode"] not in loss_filter:
                continue
            if scope != "per_dataset" and not category:
                raise ValueError(f"Manifest row {index} requires a category for {scope}")

            cohort = [row for row in evaluation_rows if row["dataset"] == target]
            if scope != "per_dataset":
                cohort = [row for row in cohort if row["category"] == category]
            if not cohort:
                raise ValueError(
                    f"No evaluation rows matched {target}/{category or 'all'} in {bundle_root}"
                )
            if any(row["partition"] != "evaluation" for row in cohort):
                raise ValueError(f"Non-evaluation row selected for manifest row {index}")
            evaluation_ids = tuple(row["protocol_id"] for row in cohort)
            source_label = int(raw["source_label"])
            target_label = int(raw["target_label"])
            if source_label == target_label:
                raise ValueError(f"Source and target labels must differ in manifest row {index}")
            expected_labels = DIRECTION_LABELS.get(raw["direction"])
            if expected_labels is None:
                raise ValueError(
                    f"Unknown attack direction {raw['direction']!r} in manifest row {index}"
                )
            if (source_label, target_label) != expected_labels:
                raise ValueError(
                    f"Direction/label mismatch in manifest row {index}: "
                    f"{raw['direction']} requires {expected_labels}, got "
                    f"{(source_label, target_label)}"
                )
            apply_label = raw.get("apply_only_to_clean_label", "").strip()
            if apply_label and int(apply_label) != source_label:
                raise ValueError(
                    f"Clean-label application mismatch in manifest row {index}"
                )
            keep_opposite = raw.get("keep_opposite_label_clean", "").strip().lower()
            if keep_opposite and keep_opposite not in {"1", "true", "yes"}:
                raise ValueError(
                    f"This evaluator requires the opposite label to remain clean; "
                    f"manifest row {index} declares {raw['keep_opposite_label_clean']!r}"
                )
            attacked_ids = tuple(
                row["protocol_id"] for row in cohort if int(row["label"]) == source_label
            )
            expected_attacked = int(raw["evaluation_attacked_image_count"])
            if len(attacked_ids) != expected_attacked:
                raise ValueError(
                    f"Attacked count mismatch in manifest row {index}: "
                    f"CSV says {expected_attacked}, indices select {len(attacked_ids)}"
                )
            if len(evaluation_ids) != len(set(evaluation_ids)):
                raise ValueError(f"Duplicate evaluation protocol IDs in {bundle_root}")

            record: dict[str, Any] = dict(raw)
            record.update(
                {
                    "scope": scope,
                    "category": category,
                    "source_label": source_label,
                    "target_label": target_label,
                    "evaluation_attacked_image_count": expected_attacked,
                    "image_size": int(raw["image_size"]),
                    "epsilon": float(raw["epsilon"]),
                    "target_evaluation_all_count": len(evaluation_ids),
                    "target_attacked_label_count": len(attacked_ids),
                    "target_evaluation_all_sample_ids": list(evaluation_ids),
                    "target_attacked_sample_ids": list(attacked_ids),
                }
            )
            delta_path = _resolve_noise_path(bundle_root, raw["noise_file"])
            artifact = AttackArtifact(
                bundle_root, record, delta_path, evaluation_ids, attacked_ids
            )
            if artifact.key in seen:
                raise ValueError(f"Duplicate attack condition in manifests: {artifact.key}")
            seen.add(artifact.key)
            if verify_files and not delta_path.is_file():
                raise FileNotFoundError(f"Perturbation tensor not found: {delta_path}")
            artifacts.append(artifact)

    if not artifacts:
        raise ValueError("No attack manifest records matched the requested filters")
    return artifacts


def json_safe_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable defensive copy used in result snapshots."""

    return json.loads(json.dumps(record, allow_nan=False))
