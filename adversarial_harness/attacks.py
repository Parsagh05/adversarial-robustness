"""Targeted PGD attacks for instance, category, and dataset scopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import AttackConfig, VALID_LOSS_MODES
from .prompts import ensemble_class_logits


@dataclass
class UniversalAttackResult:
    """Universal perturbation plus crash-persistable optimization diagnostics."""

    delta: torch.Tensor
    history: List[Dict[str, float]]
    initial_losses: Dict[str, float]
    final_losses: Dict[str, float]
    diagnostic_sample_ids: List[str]


def direction_labels(direction: str) -> Tuple[int, int]:
    """Return ``(source_label, target_label)`` for a threat direction."""

    if direction == "normal_to_abnormal":
        return 0, 1
    if direction == "abnormal_to_normal":
        return 1, 0
    raise ValueError(f"Unknown direction: {direction}")


class TargetedPGD:
    """PGD optimizer that never queries or differentiates through the target."""

    def __init__(self, surrogate, config: AttackConfig):
        self.surrogate = surrogate
        self.config = config
        self.device = surrogate.device
        self.rng = np.random.default_rng(config.seed)

    def _group_losses(
        self,
        global_features: torch.Tensor,
        patch_features: Sequence[torch.Tensor],
        categories: Sequence[str],
        target_label: int,
        mode: str,
    ) -> Dict[str, torch.Tensor]:
        if mode not in VALID_LOSS_MODES:
            raise ValueError(f"Unknown loss mode: {mode}")
        total_global = global_features.new_zeros(())
        total_local = global_features.new_zeros(())
        group_count = 0

        for category in sorted(set(categories)):
            indices = [index for index, value in enumerate(categories) if value == category]
            index_tensor = torch.as_tensor(indices, device=self.device, dtype=torch.long)
            bank = self.surrogate.prompts[category]
            target = torch.full(
                (len(indices),), target_label, device=self.device, dtype=torch.long
            )

            if mode in {"global", "combined"}:
                global_logits = ensemble_class_logits(
                    global_features.index_select(0, index_tensor),
                    bank,
                    self.config.temperature,
                )
                total_global = total_global + F.cross_entropy(global_logits, target)

            if mode in {"local", "combined"}:
                layer_losses = []
                for patch in patch_features:
                    selected = patch.index_select(0, index_tensor)
                    # The first token is CLS; dense loss is defined only on patches.
                    if selected.shape[1] > 1:
                        selected = selected[:, 1:, :]
                    local_logits = ensemble_class_logits(
                        selected, bank, self.config.temperature
                    )
                    local_target = target[:, None].expand(-1, selected.shape[1]).reshape(-1)
                    layer_losses.append(
                        F.cross_entropy(local_logits.reshape(-1, 2), local_target)
                    )
                if not layer_losses:
                    raise RuntimeError("The surrogate returned no patch features")
                total_local = total_local + torch.stack(layer_losses).mean()
            group_count += 1

        result: Dict[str, torch.Tensor] = {}
        if mode in {"global", "combined"}:
            result["global"] = total_global / group_count
        if mode in {"local", "combined"}:
            result["local"] = total_local / group_count
        if mode == "global":
            result["total"] = result["global"]
        elif mode == "local":
            result["total"] = result["local"]
        else:
            result["total"] = (
                self.config.global_weight * result["global"]
                + self.config.local_weight * result["local"]
            )
        return result

    def objective_components(
        self,
        images_01: torch.Tensor,
        categories: Sequence[str],
        target_label: int,
        mode: str,
    ) -> Dict[str, torch.Tensor]:
        global_features, patch_features = self.surrogate.encode_visual(
            images_01, include_patches=mode in {"local", "combined"}
        )
        return self._group_losses(
            global_features, patch_features, categories, target_label, mode
        )

    def objective(
        self,
        images_01: torch.Tensor,
        categories: Sequence[str],
        target_label: int,
        mode: str,
    ) -> torch.Tensor:
        return self.objective_components(
            images_01, categories, target_label, mode
        )["total"]

    def surrogate_scores(
        self,
        images_01: torch.Tensor,
        categories: Sequence[str],
        mode: str,
    ) -> Dict[str, np.ndarray]:
        """Return global/local/mode anomaly scores from the public surrogate."""

        include_local = mode in {"local", "combined"}
        with torch.no_grad():
            global_features, patch_features = self.surrogate.encode_visual(
                images_01.to(self.device), include_patches=include_local
            )
            batch_size = global_features.shape[0]
            global_logits = global_features.new_zeros((batch_size, 2))
            local_logits: Optional[torch.Tensor] = (
                global_features.new_zeros((batch_size, 2)) if include_local else None
            )
            for category in sorted(set(categories)):
                indices = [
                    index for index, value in enumerate(categories) if value == category
                ]
                index_tensor = torch.as_tensor(
                    indices, device=self.device, dtype=torch.long
                )
                bank = self.surrogate.prompts[category]
                category_global = ensemble_class_logits(
                    global_features.index_select(0, index_tensor),
                    bank,
                    self.config.temperature,
                )
                global_logits.index_copy_(0, index_tensor, category_global)
                if include_local and local_logits is not None:
                    layer_logits = []
                    for patch in patch_features:
                        selected = patch.index_select(0, index_tensor)
                        if selected.shape[1] > 1:
                            selected = selected[:, 1:, :]
                        patch_logits = ensemble_class_logits(
                            selected, bank, self.config.temperature
                        )
                        layer_logits.append(patch_logits.mean(dim=1))
                    category_local = torch.stack(layer_logits).mean(dim=0)
                    local_logits.index_copy_(0, index_tensor, category_local)

            global_scores = global_logits.softmax(dim=-1)[:, 1]
            if mode == "global":
                mode_logits = global_logits
            elif mode == "local":
                if local_logits is None:
                    raise RuntimeError("Local surrogate logits were not computed")
                mode_logits = local_logits
            else:
                if local_logits is None:
                    raise RuntimeError("Combined surrogate logits require local logits")
                mode_logits = (
                    self.config.global_weight * global_logits
                    + self.config.local_weight * local_logits
                )
            mode_scores = mode_logits.softmax(dim=-1)[:, 1]
            if local_logits is None:
                local_scores = torch.full_like(global_scores, float("nan"))
            else:
                local_scores = local_logits.softmax(dim=-1)[:, 1]
        return {
            "global_score": global_scores.detach().cpu().numpy().astype(np.float32),
            "local_score": local_scores.detach().cpu().numpy().astype(np.float32),
            "mode_score": mode_scores.detach().cpu().numpy().astype(np.float32),
        }

    def _initial_delta(self, shape: Sequence[int], clean: torch.Tensor) -> torch.Tensor:
        if self.config.random_start:
            delta = torch.empty(shape, device=self.device, dtype=clean.dtype).uniform_(
                -self.config.epsilon, self.config.epsilon
            )
            delta = (clean + delta).clamp(0.0, 1.0) - clean
        else:
            delta = torch.zeros(shape, device=self.device, dtype=clean.dtype)
        return delta.detach()

    def perturb_batch(
        self,
        clean_images: torch.Tensor,
        categories: Sequence[str],
        target_label: int,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimize independent per-image perturbations for a batch."""

        clean = clean_images.to(self.device)
        delta = self._initial_delta(clean.shape, clean)
        for _ in range(self.config.steps):
            delta.requires_grad_(True)
            adversarial = (clean + delta).clamp(0.0, 1.0)
            loss = self.objective(adversarial, categories, target_label, mode)
            gradient = torch.autograd.grad(loss, delta, only_inputs=True)[0]
            # Targeted PGD minimizes CE for the requested incorrect class.
            delta = delta.detach() - self.config.step_size * gradient.sign()
            delta = delta.clamp(-self.config.epsilon, self.config.epsilon)
            delta = ((clean + delta).clamp(0.0, 1.0) - clean).detach()
        return (clean + delta).clamp(0.0, 1.0).detach(), delta.detach()

    def optimize_universal(
        self,
        samples: Sequence[object],
        image_loader: Callable[[object], torch.Tensor],
        target_label: int,
        mode: str,
        diagnostic_samples: Optional[Sequence[object]] = None,
        progress: Callable[[int, int, float], None] | None = None,
    ) -> UniversalAttackResult:
        """Optimize one shared perturbation across the supplied samples."""

        if not samples:
            raise ValueError("Universal optimization requires at least one sample")
        size = self.config.image_size
        reference = image_loader(samples[0]).unsqueeze(0).to(self.device)
        delta = self._initial_delta((1, 3, size, size), reference)
        order = np.arange(len(samples))
        cursor = len(order)
        diagnostic_samples = list(diagnostic_samples or samples[:1])
        diagnostic_ids = [
            str(
                getattr(
                    sample,
                    "protocol_id",
                    getattr(sample, "sample_id", index),
                )
            )
            for index, sample in enumerate(diagnostic_samples)
        ]
        initial_losses = self._diagnostic_losses(
            diagnostic_samples, image_loader, delta, target_label, mode
        )
        history: List[Dict[str, float]] = []

        for step in range(self.config.universal_steps):
            batch_size = min(self.config.universal_batch_size, len(samples))
            if cursor >= len(order):
                self.rng.shuffle(order)
                cursor = 0
            # Consume the short final batch instead of discarding it when the
            # dataset size is not divisible by universal_batch_size. The next
            # update starts a newly shuffled pass over all samples.
            stop = min(cursor + batch_size, len(order))
            indices = order[cursor:stop]
            cursor = stop
            batch_samples = [samples[int(index)] for index in indices]
            clean = torch.stack([image_loader(sample) for sample in batch_samples]).to(
                self.device
            )
            categories = [str(getattr(sample, "category")) for sample in batch_samples]

            delta.requires_grad_(True)
            adversarial = (clean + delta).clamp(0.0, 1.0)
            components = self.objective_components(
                adversarial, categories, target_label, mode
            )
            global_gradient_norm = float("nan")
            local_gradient_norm = float("nan")
            if mode == "combined":
                global_gradient = torch.autograd.grad(
                    components["global"], delta, retain_graph=True, only_inputs=True
                )[0]
                local_gradient = torch.autograd.grad(
                    components["local"], delta, only_inputs=True
                )[0]
                global_gradient_norm = float(global_gradient.norm().detach())
                local_gradient_norm = float(local_gradient.norm().detach())
                gradient = (
                    self.config.global_weight * global_gradient
                    + self.config.local_weight * local_gradient
                )
            else:
                gradient = torch.autograd.grad(
                    components["total"], delta, only_inputs=True
                )[0]
                if mode == "global":
                    global_gradient_norm = float(gradient.norm().detach())
                else:
                    local_gradient_norm = float(gradient.norm().detach())
            pre_update_loss = float(components["total"].detach())
            delta = delta.detach() - self.config.step_size * gradient.sign()
            delta = delta.clamp(-self.config.epsilon, self.config.epsilon)
            delta = delta.detach()
            with torch.no_grad():
                updated_components = self.objective_components(
                    (clean + delta).clamp(0.0, 1.0),
                    categories,
                    target_label,
                    mode,
                )
            history.append(
                {
                    "step": float(step + 1),
                    "pre_update_total_loss": pre_update_loss,
                    "total_loss": float(updated_components["total"].detach()),
                    "global_loss": float(updated_components["global"].detach())
                    if "global" in updated_components
                    else float("nan"),
                    "local_loss": float(updated_components["local"].detach())
                    if "local" in updated_components
                    else float("nan"),
                    "global_gradient_l2": global_gradient_norm,
                    "local_gradient_l2": local_gradient_norm,
                    "combined_gradient_l2": float(gradient.norm().detach()),
                }
            )
            if progress is not None:
                progress(
                    step + 1,
                    self.config.universal_steps,
                    float(updated_components["total"].detach()),
                )
        final_losses = self._diagnostic_losses(
            diagnostic_samples, image_loader, delta, target_label, mode
        )
        return UniversalAttackResult(
            delta=delta.detach(),
            history=history,
            initial_losses=initial_losses,
            final_losses=final_losses,
            diagnostic_sample_ids=diagnostic_ids,
        )

    def _diagnostic_losses(
        self,
        samples: Sequence[object],
        image_loader: Callable[[object], torch.Tensor],
        delta: torch.Tensor,
        target_label: int,
        mode: str,
    ) -> Dict[str, float]:
        """Average losses on one fixed bounded subset before and after PGD."""

        totals: Dict[str, float] = {"total": 0.0, "global": 0.0, "local": 0.0}
        counts: Dict[str, int] = {"total": 0, "global": 0, "local": 0}
        batch_size = min(self.config.universal_batch_size, len(samples))
        with torch.no_grad():
            for start in range(0, len(samples), batch_size):
                batch_samples = samples[start : start + batch_size]
                clean = torch.stack(
                    [image_loader(sample) for sample in batch_samples]
                ).to(self.device)
                attacked = self.apply_universal(clean, delta)
                categories = [
                    str(getattr(sample, "category")) for sample in batch_samples
                ]
                components = self.objective_components(
                    attacked, categories, target_label, mode
                )
                for key, value in components.items():
                    totals[key] += float(value.detach()) * len(batch_samples)
                    counts[key] += len(batch_samples)
        return {
            key: totals[key] / counts[key]
            for key in totals
            if counts[key] > 0
        }

    @staticmethod
    def apply_universal(clean_images: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return (clean_images.to(delta.device) + delta).clamp(0.0, 1.0)
