"""Create and load reproducible MVTec/VisA held-out split manifests."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .dataset import MVTecSample, discover_anomaly_datasets, group_by_category


MATCHED_SPLIT_PROTOCOL = "matched_test_per_category_v1"
MANIFEST_FIELDS = (
    "protocol_version",
    "protocol_id",
    "category",
    "defect_type",
    "label",
    "partition",
    "relative_image_path",
)
SELECTED_PARTITIONS = ("fit", "evaluation")
ALL_PARTITIONS = (*SELECTED_PARTITIONS, "excluded_balance")


@dataclass(frozen=True)
class LoadedSplitManifest:
    """Validated samples and assignments selected for one experiment run."""

    samples: List[MVTecSample]
    assignments: Dict[str, str]
    metadata: Dict[str, object]
    csv_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_order(
    samples: Sequence[MVTecSample], seed: int, namespace: str
) -> List[MVTecSample]:
    """Return a deterministic order independent of Python hash randomization."""

    ordered = sorted(samples, key=lambda sample: sample.protocol_id)
    material = f"{seed}:{namespace}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    rng = np.random.default_rng(local_seed)
    permutation = rng.permutation(len(ordered))
    return [ordered[int(index)] for index in permutation]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def create_matched_split_manifest(
    mvtec_root: Optional[str],
    csv_path: str,
    json_path: str,
    categories: Optional[Sequence[str]] = None,
    split_seed: int = 111,
    fit_fraction: float = 0.5,
    dataset: str = "mvtec",
    visa_root: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Create one category-balanced, label-balanced 50/50 test manifest.

    Every selected category contributes the same number of normal and anomalous
    test samples. An odd matched label count is reduced by one so both labels
    can be split exactly in half. Unselected majority-label rows remain in the
    CSV as ``excluded_balance`` for auditability.
    """

    if not np.isclose(fit_fraction, 0.5):
        raise ValueError(
            f"{MATCHED_SPLIT_PROTOCOL} requires fit_fraction=0.5, "
            f"got {fit_fraction}"
        )
    mode = str(dataset).lower()
    if mode not in {"mvtec", "visa", "both"}:
        raise ValueError("dataset must be one of: mvtec, visa, both")
    roots: Dict[str, Path] = {}
    if mode in {"mvtec", "both"}:
        if not mvtec_root:
            raise ValueError(f"mvtec_root is required when dataset={mode!r}")
        roots["mvtec"] = Path(mvtec_root).expanduser().resolve()
    if mode in {"visa", "both"}:
        if not visa_root:
            raise ValueError(f"visa_root is required when dataset={mode!r}")
        roots["visa"] = Path(visa_root).expanduser().resolve()
    csv_output = Path(csv_path).expanduser().resolve()
    json_output = Path(json_path).expanduser().resolve()
    if csv_output == json_output:
        raise ValueError("csv_path and json_path must be different files")

    samples = discover_anomaly_datasets(
        mode,
        mvtec_root=mvtec_root,
        visa_root=visa_root,
        categories=categories,
    )
    rows: List[Dict[str, object]] = []
    category_counts: Dict[str, Dict[str, int]] = {}
    for category, category_samples in sorted(group_by_category(samples).items()):
        by_label = {
            label: [sample for sample in category_samples if sample.label == label]
            for label in (0, 1)
        }
        matched_per_label = min(len(by_label[0]), len(by_label[1]))
        matched_per_label -= matched_per_label % 2
        if matched_per_label < 2:
            raise ValueError(
                f"Category {category!r} needs at least two test normals and two "
                "test anomalies for an exact matched 50/50 split"
            )
        fit_per_label = matched_per_label // 2
        assignments: Dict[str, str] = {}
        for label in (0, 1):
            shuffled = _stable_order(
                by_label[label], split_seed, f"manifest:{category}:{label}"
            )
            for position, sample in enumerate(shuffled):
                if position < fit_per_label:
                    partition = "fit"
                elif position < matched_per_label:
                    partition = "evaluation"
                else:
                    partition = "excluded_balance"
                assignments[sample.protocol_id] = partition

        category_counts[category] = {
            "available_normal": len(by_label[0]),
            "available_anomaly": len(by_label[1]),
            "fit_normal": fit_per_label,
            "fit_anomaly": fit_per_label,
            "evaluation_normal": fit_per_label,
            "evaluation_anomaly": fit_per_label,
            "excluded_balance": len(category_samples) - (4 * fit_per_label),
        }
        for sample in sorted(category_samples, key=lambda item: item.protocol_id):
            rows.append(
                {
                    "protocol_version": MATCHED_SPLIT_PROTOCOL,
                    "protocol_id": sample.protocol_id,
                    "category": sample.category,
                    "defect_type": sample.defect_type,
                    "label": sample.label,
                    "partition": assignments[sample.protocol_id],
                    "relative_image_path": sample.image_path.resolve()
                    .relative_to(roots[sample.dataset])
                    .as_posix(),
                }
            )

    _write_csv(csv_output, rows)
    csv_digest = _sha256(csv_output)
    metadata: Dict[str, object] = {
        "protocol_version": MATCHED_SPLIT_PROTOCOL,
        "dataset": mode,
        "dataset_description": {
            "mvtec": "MVTec AD official test split",
            "visa": "VisA official test split",
            "both": "MVTec AD and VisA official test splits",
        }[mode],
        "split_seed": int(split_seed),
        "fit_fraction": 0.5,
        "balancing_strategy": "match_smaller_label_count_per_category_then_even_50_50",
        "csv_file": csv_output.name,
        "csv_sha256": csv_digest,
        "categories": sorted(category_counts),
        "dataset_counts": {
            name: sum(1 for sample in samples if sample.dataset == name)
            for name in sorted(roots)
        },
        "category_counts": category_counts,
        "total_manifest_rows": len(rows),
        "total_selected_rows": sum(
            1 for row in rows if row["partition"] in SELECTED_PARTITIONS
        ),
    }
    _write_json(json_output, metadata)
    return csv_output, json_output


def _read_manifest_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != set(MANIFEST_FIELDS):
            raise ValueError(
                f"Split manifest columns must be exactly {MANIFEST_FIELDS}; "
                f"got {reader.fieldnames}"
            )
        return list(reader)


def _reindex(samples: Sequence[MVTecSample]) -> List[MVTecSample]:
    return [
        MVTecSample(
            index=index,
            category=sample.category,
            defect_type=sample.defect_type,
            image_path=sample.image_path,
            mask_path=sample.mask_path,
            label=sample.label,
            split=sample.split,
            dataset=sample.dataset,
        )
        for index, sample in enumerate(samples)
    ]


def load_matched_split_manifest(
    test_samples: Sequence[MVTecSample],
    csv_path: str,
    json_path: str,
    split_seed: int,
    fit_fraction: float,
    max_samples_per_category: Optional[int] = None,
    expected_dataset: Optional[str] = None,
) -> LoadedSplitManifest:
    """Validate a saved manifest and select a balanced run subset from it."""

    csv_input = Path(csv_path).expanduser().resolve()
    json_input = Path(json_path).expanduser().resolve()
    if not csv_input.is_file():
        raise FileNotFoundError(f"Split manifest CSV not found: {csv_input}")
    if not json_input.is_file():
        raise FileNotFoundError(f"Split manifest metadata JSON not found: {json_input}")
    metadata = json.loads(json_input.read_text(encoding="utf-8"))
    if metadata.get("protocol_version") != MATCHED_SPLIT_PROTOCOL:
        raise ValueError(
            "Unsupported split manifest protocol: "
            f"{metadata.get('protocol_version')!r}"
        )
    stored_dataset = metadata.get("dataset")
    # Historical v1 artifacts used a descriptive MVTec string. Accept those,
    # while requiring newly generated manifests to match the configured mode.
    if (
        expected_dataset is not None
        and stored_dataset in {"mvtec", "visa", "both"}
        and stored_dataset != str(expected_dataset).lower()
    ):
        raise ValueError(
            f"Split manifest was generated for dataset={stored_dataset!r}, but "
            f"the experiment requested dataset={expected_dataset!r}"
        )
    csv_digest = _sha256(csv_input)
    if metadata.get("csv_sha256") != csv_digest:
        raise ValueError("Split manifest CSV hash does not match its metadata JSON")
    if int(metadata.get("split_seed", -1)) != int(split_seed):
        raise ValueError(
            "Experiment split_seed does not match the saved split manifest: "
            f"{split_seed} != {metadata.get('split_seed')}"
        )
    if not np.isclose(float(metadata.get("fit_fraction", -1.0)), fit_fraction):
        raise ValueError(
            "Experiment fit_fraction does not match the saved split manifest: "
            f"{fit_fraction} != {metadata.get('fit_fraction')}"
        )

    live_by_id = {sample.protocol_id: sample for sample in test_samples}
    if len(live_by_id) != len(test_samples):
        raise ValueError("Discovered test samples contain duplicate protocol IDs")
    selected_categories = {sample.category for sample in test_samples}
    rows = [
        row for row in _read_manifest_rows(csv_input)
        if row["category"] in selected_categories
    ]
    row_by_id: Dict[str, Dict[str, str]] = {}
    for row in rows:
        protocol_id = row["protocol_id"]
        if protocol_id in row_by_id:
            raise ValueError(f"Duplicate protocol_id in split manifest: {protocol_id}")
        if row["protocol_version"] != MATCHED_SPLIT_PROTOCOL:
            raise ValueError(f"Wrong protocol_version for manifest row {protocol_id}")
        if row["partition"] not in ALL_PARTITIONS:
            raise ValueError(
                f"Unknown partition {row['partition']!r} for manifest row {protocol_id}"
            )
        sample = live_by_id.get(protocol_id)
        if sample is None:
            raise ValueError(
                f"Manifest sample is missing from the selected dataset mount: {protocol_id}"
            )
        if (
            row["category"] != sample.category
            or row["defect_type"] != sample.defect_type
            or int(row["label"]) != sample.label
        ):
            raise ValueError(f"Manifest metadata disagrees with dataset for {protocol_id}")
        relative_parts = PurePosixPath(row["relative_image_path"]).parts
        if not relative_parts or tuple(sample.image_path.parts[-len(relative_parts):]) != tuple(
            relative_parts
        ):
            raise ValueError(f"Manifest image path disagrees with dataset for {protocol_id}")
        row_by_id[protocol_id] = row

    missing_rows = sorted(set(live_by_id) - set(row_by_id))
    if missing_rows:
        preview = ", ".join(missing_rows[:5])
        raise ValueError(
            f"Split manifest is missing {len(missing_rows)} discovered test samples; "
            f"first entries: {preview}"
        )

    selected_ids = {
        protocol_id
        for protocol_id, row in row_by_id.items()
        if row["partition"] in SELECTED_PARTITIONS
    }
    if max_samples_per_category is not None:
        if max_samples_per_category < 4:
            raise ValueError(
                "Manifest-based max_samples_per_category must be at least 4 "
                "to retain normal/anomaly x fit/evaluation strata"
            )
        reduced_ids: set[str] = set()
        for category in sorted(selected_categories):
            strata: Dict[Tuple[int, str], List[MVTecSample]] = {}
            for label in (0, 1):
                for partition in SELECTED_PARTITIONS:
                    stratum = [
                        live_by_id[protocol_id]
                        for protocol_id in selected_ids
                        if live_by_id[protocol_id].category == category
                        and live_by_id[protocol_id].label == label
                        and row_by_id[protocol_id]["partition"] == partition
                    ]
                    strata[(label, partition)] = _stable_order(
                        stratum,
                        split_seed,
                        f"cap:{max_samples_per_category}:{category}:{label}:{partition}",
                    )
            per_stratum = min(
                max_samples_per_category // 4,
                *(len(values) for values in strata.values()),
            )
            if per_stratum < 1:
                raise ValueError(
                    f"Category {category!r} cannot provide all four balanced strata"
                )
            for values in strata.values():
                reduced_ids.update(sample.protocol_id for sample in values[:per_stratum])
        selected_ids = reduced_ids

    selected_samples = _reindex(
        [sample for sample in test_samples if sample.protocol_id in selected_ids]
    )
    assignments = {
        sample.protocol_id: row_by_id[sample.protocol_id]["partition"]
        for sample in selected_samples
    }
    if not selected_samples:
        raise ValueError("Split manifest selected no samples for this experiment")
    runtime_counts: Dict[str, Dict[str, int]] = {}
    for sample in selected_samples:
        key = f"{assignments[sample.protocol_id]}_{'normal' if sample.label == 0 else 'anomaly'}"
        runtime_counts.setdefault(sample.category, {}).setdefault(key, 0)
        runtime_counts[sample.category][key] += 1
    for category, counts in runtime_counts.items():
        values = [
            counts.get("fit_normal", 0),
            counts.get("fit_anomaly", 0),
            counts.get("evaluation_normal", 0),
            counts.get("evaluation_anomaly", 0),
        ]
        if len(set(values)) != 1 or values[0] < 1:
            raise ValueError(
                f"Runtime split is not exactly balanced for {category!r}: {counts}"
            )
    runtime_metadata = dict(metadata)
    runtime_metadata["runtime_counts"] = runtime_counts
    runtime_metadata["runtime_selected_rows"] = len(selected_samples)
    runtime_metadata["max_samples_per_category"] = max_samples_per_category
    return LoadedSplitManifest(
        samples=selected_samples,
        assignments=assignments,
        metadata=runtime_metadata,
        csv_sha256=csv_digest,
    )
