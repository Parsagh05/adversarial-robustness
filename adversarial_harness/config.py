"""Typed configuration for the adversarial robustness benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


VALID_SCOPES = ("per_image", "per_category", "dataset")
VALID_DIRECTIONS = ("normal_to_abnormal", "abnormal_to_normal")
VALID_LOSS_MODES = ("global", "local", "combined")
VALID_UNIVERSAL_PROTOCOLS = ("transductive", "held_out")
VALID_THRESHOLD_MODES = ("normal_train_quantile",)


def _tuple(value: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(item) for item in value)


@dataclass
class AttackConfig:
    """PGD and attack-protocol settings.

    Pixel values and epsilon/step size are expressed in the unnormalized
    ``[0, 1]`` image domain. Universal attacks use ``universal_steps`` update
    batches; per-image attacks use ``steps`` updates per image.
    """

    image_size: int = 518
    epsilon: float = 8.0 / 255.0
    step_size: float = 2.0 / 255.0
    steps: int = 20
    universal_steps: int = 200
    random_start: bool = True
    temperature: float = 0.07
    global_weight: float = 0.5
    local_weight: float = 0.5
    feature_layers: Tuple[int, ...] = (6, 12, 18, 24)
    scopes: Tuple[str, ...] = VALID_SCOPES
    directions: Tuple[str, ...] = VALID_DIRECTIONS
    loss_modes: Tuple[str, ...] = VALID_LOSS_MODES
    per_image_batch_size: int = 1
    universal_batch_size: int = 2
    seed: int = 111

    def __post_init__(self) -> None:
        self.feature_layers = tuple(int(layer) for layer in self.feature_layers)
        self.scopes = _tuple(self.scopes)
        self.directions = _tuple(self.directions)
        self.loss_modes = _tuple(self.loss_modes)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 < self.epsilon <= 1.0:
            raise ValueError("epsilon must be in (0, 1]")
        if not 0.0 < self.step_size <= 1.0:
            raise ValueError("step_size must be in (0, 1]")
        if self.steps <= 0 or self.universal_steps <= 0:
            raise ValueError("steps and universal_steps must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.per_image_batch_size <= 0 or self.universal_batch_size <= 0:
            raise ValueError("attack batch sizes must be positive")
        for value, valid, name in (
            (self.scopes, VALID_SCOPES, "scopes"),
            (self.directions, VALID_DIRECTIONS, "directions"),
            (self.loss_modes, VALID_LOSS_MODES, "loss_modes"),
        ):
            if not value:
                raise ValueError(f"{name} cannot be empty")
            unknown = sorted(set(value) - set(valid))
            if unknown:
                raise ValueError(f"Unknown {name}: {unknown}. Valid values: {valid}")
        if not self.feature_layers or any(layer <= 0 for layer in self.feature_layers):
            raise ValueError("feature_layers must contain positive layer indices")
        if self.global_weight < 0 or self.local_weight < 0:
            raise ValueError("loss weights cannot be negative")
        if self.global_weight + self.local_weight <= 0:
            raise ValueError("at least one loss weight must be positive")


@dataclass
class ExperimentConfig:
    """End-to-end benchmark settings."""

    mvtec_root: str
    output_root: str
    anomalyclip_root: str
    anomalyclip_checkpoint: str
    device: str = "cuda"
    target_model: str = "AnomalyCLIP"
    categories: Optional[Tuple[str, ...]] = None
    target_batch_size: int = 2
    metric_size: int = 518
    aupro_fpr_limit: float = 0.30
    aupro_max_thresholds: int = 200
    anomaly_map_sigma: float = 4.0
    compute_lpips: bool = True
    lpips_backbone: str = "alex"
    save_universal_perturbations: bool = True
    save_adversarial_examples: int = 0
    universal_protocol: str = "transductive"
    fit_fraction: float = 0.5
    split_seed: int = 111
    diagnostic_max_samples: int = 64
    threshold_mode: str = "normal_train_quantile"
    threshold_quantile: float = 0.95
    thresholds_path: Optional[str] = None
    max_samples_per_category: Optional[int] = None
    resume: bool = True
    attack: AttackConfig = field(default_factory=AttackConfig)
    target_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.categories is not None:
            self.categories = tuple(self.categories)
        if self.target_batch_size <= 0:
            raise ValueError("target_batch_size must be positive")
        if self.metric_size <= 0:
            raise ValueError("metric_size must be positive")
        if not 0.0 < self.aupro_fpr_limit <= 1.0:
            raise ValueError("aupro_fpr_limit must be in (0, 1]")
        if self.aupro_max_thresholds < 2:
            raise ValueError("aupro_max_thresholds must be at least 2")
        if self.anomaly_map_sigma < 0:
            raise ValueError("anomaly_map_sigma cannot be negative")
        if self.save_adversarial_examples < 0:
            raise ValueError("save_adversarial_examples cannot be negative")
        self.universal_protocol = str(self.universal_protocol)
        if self.universal_protocol not in VALID_UNIVERSAL_PROTOCOLS:
            raise ValueError(
                "universal_protocol must be one of "
                f"{VALID_UNIVERSAL_PROTOCOLS}, got {self.universal_protocol!r}"
            )
        if not 0.0 < self.fit_fraction < 1.0:
            raise ValueError("fit_fraction must be in (0, 1)")
        if self.diagnostic_max_samples <= 0:
            raise ValueError("diagnostic_max_samples must be positive")
        self.threshold_mode = str(self.threshold_mode)
        if self.threshold_mode not in VALID_THRESHOLD_MODES:
            raise ValueError(
                "threshold_mode must be one of "
                f"{VALID_THRESHOLD_MODES}, got {self.threshold_mode!r}"
            )
        if not 0.0 < self.threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0, 1)")
        if self.thresholds_path is not None:
            self.thresholds_path = str(self.thresholds_path)
        if self.max_samples_per_category is not None and self.max_samples_per_category < 2:
            raise ValueError(
                "max_samples_per_category must be at least 2 so both labels remain present"
            )

    @property
    def output_path(self) -> Path:
        return Path(self.output_root).expanduser()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
