# Adversarial anomaly-model pipelines

The repository contains three independent top-level projects:

- [`blackbox_evaluation_pipeline`](blackbox_evaluation_pipeline/README.md):
  evaluates models against fixed, manifest-defined perturbations.
- [`full_attack_generation_pipeline`](full_attack_generation_pipeline/README.md):
  the complete end-to-end attack-generation and evaluation benchmark.
- [`per_dataset_ablation_pipeline`](per_dataset_ablation_pipeline/README.md):
  generates and evaluates the per-dataset step/epsilon ablation matrix on
  MVTec AD and VisA.

No project imports Python code or requirements from either of the other two.
