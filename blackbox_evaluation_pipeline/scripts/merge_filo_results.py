import csv
import hashlib
import json
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows[0], rows[1:]


def write_atomic(path: Path, writer):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            writer(handle)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main():
    root = Path(sys.argv[1]).resolve()
    shards = [root / "1", root / "2"]
    csv_names = ["category_metrics.csv", "summary.csv", "per_image.csv"]
    json_names = ["manifest_snapshot.json", "run_config.json"]
    source_paths = [shard / name for shard in shards for name in csv_names + json_names]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    hashes_before = {str(path): sha256(path) for path in source_paths}
    report = {"csv": {}, "json": {}}

    for name in csv_names:
        header1, rows1 = read_csv(shards[0] / name)
        header2, rows2 = read_csv(shards[1] / name)
        if header1 != header2:
            raise ValueError(f"CSV headers differ for {name}")
        expected = rows1 + rows2

        def emit_csv(stream, header=header1, rows=expected):
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)

        output = root / name
        write_atomic(output, emit_csv)
        merged_header, merged_rows = read_csv(output)
        if merged_header != header1 or merged_rows != expected:
            raise ValueError(f"Merged CSV verification failed for {name}")
        report["csv"][name] = {
            "shard_1_rows": len(rows1),
            "shard_2_rows": len(rows2),
            "merged_rows": len(merged_rows),
        }

    manifests = []
    for shard in shards:
        with (shard / "manifest_snapshot.json").open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, list):
            raise TypeError(f"Manifest is not a list: {shard}")
        manifests.append(value)
    merged_manifest = manifests[0] + manifests[1]
    manifest_output = root / "manifest_snapshot.json"
    write_atomic(
        manifest_output,
        lambda stream: json.dump(merged_manifest, stream, indent=2, ensure_ascii=False),
    )
    with manifest_output.open("r", encoding="utf-8") as stream:
        if json.load(stream) != merged_manifest:
            raise ValueError("Merged manifest verification failed")
    report["json"]["manifest_snapshot.json"] = {
        "shard_1_items": len(manifests[0]),
        "shard_2_items": len(manifests[1]),
        "merged_items": len(merged_manifest),
    }

    configs = []
    for shard in shards:
        with (shard / "run_config.json").open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise TypeError(f"Run config is not an object: {shard}")
        configs.append(value)
    differing = {
        key
        for key in configs[0]
        if configs[0].get(key) != configs[1].get(key)
    } | (set(configs[0]) ^ set(configs[1]))
    expected_differences = {"thresholds_by_target", "target_datasets"}
    if differing != expected_differences:
        raise ValueError(f"Unexpected run-config differences: {sorted(differing)}")

    merged_config = deepcopy(configs[0])
    merged_config["thresholds_by_target"] = {
        **configs[0]["thresholds_by_target"],
        **configs[1]["thresholds_by_target"],
    }
    merged_targets = []
    for config in configs:
        for target in config["target_datasets"]:
            if target not in merged_targets:
                merged_targets.append(target)
    merged_config["target_datasets"] = merged_targets
    config_output = root / "run_config.json"
    write_atomic(
        config_output,
        lambda stream: json.dump(merged_config, stream, indent=2, ensure_ascii=False),
    )
    with config_output.open("r", encoding="utf-8") as stream:
        if json.load(stream) != merged_config:
            raise ValueError("Merged run-config verification failed")
    report["json"]["run_config.json"] = {
        "target_datasets": merged_targets,
        "threshold_targets": list(merged_config["thresholds_by_target"]),
    }

    hashes_after = {str(path): sha256(path) for path in source_paths}
    if hashes_after != hashes_before:
        raise ValueError("A source shard changed during the merge")
    report["source_files_unchanged"] = True
    report["outputs"] = {
        name: sha256(root / name) for name in csv_names + json_names
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()