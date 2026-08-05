# Adversarial anomaly-model evaluation

The active implementation is in
[`blackbox_evaluation_pipeline`](blackbox_evaluation_pipeline/README.md). It
evaluates anomaly-detection models against fixed, manifest-defined universal
perturbations so every model sees exactly the same images and tensors.

The original end-to-end attack-generation implementation is preserved under
[`attack_generation_pipeline`](attack_generation_pipeline/README.md).
