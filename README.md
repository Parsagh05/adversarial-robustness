# Adversarial anomaly-model pipelines

The repository contains three independent top-level projects:

- [`blackbox_evaluation_pipeline`](blackbox_evaluation_pipeline/README.md):
  evaluates models against fixed, manifest-defined perturbations.
- [`full_attack_generation_pipeline`](full_attack_generation_pipeline/README.md):
  the complete end-to-end attack-generation and evaluation benchmark.
- [`perturbation_generation`](perturbation_generation/README.md): generates the
  versioned focal-plus-Dice perturbation dataset and includes its own attack
  harness and Kaggle notebook.

No project imports Python code or requirements from either of the other two.
