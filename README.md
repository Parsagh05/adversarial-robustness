# Adversarial anomaly-model evaluation

The active implementation is in [`new_pipeline`](new_pipeline/README.md). It
evaluates anomaly-detection models against fixed, manifest-defined universal
perturbations so every model sees exactly the same images and tensors.

The previous attack-generation and AnomalyCLIP-specific implementation has
been preserved under [`old_pipeline`](old_pipeline/README.md).
