# Supervisor-review delivery — 10 July 2026

## Completed evidence

- Updated thesis PDF with scored CaT successes, false-safe, false-block, and
  boundary-error cases, plus explicitly unscored ALICE examples.
- Frozen PIDNet-S results verified by two identical CPU evaluations of the
  official 544-image test manifest: mIoU 0.8939407, FSR 0.0431759, FBR
  0.0703687; TN 73,151,244, FP 3,300,886, FN 4,027,996, TP 53,213,314.
- Portable configuration, checkpoint, evaluation code, artifact hashes, and
  reproduction command packaged in `supervisor_reproducibility_bundle.zip`.
- Claims reviewed for protocol comparability, deployment scope, and causal
  attribution. The EfficientViT discussion separates observed duplication,
  published arithmetic inconsistencies, expected overlap, and personal
  correspondence from conclusions that require an unpublished split manifest.

## Frozen baseline identity

- Config: `configs/frozen_pidnet_cat.yaml`
- Checkpoint: `outputs/controlled_ablation_augment_off/checkpoints/best.pt`
- Checkpoint SHA-256:
  `b9862fe49713c0622d760b94aa3fc055fdaf6549a67a4ee3492004d9c591f471`
- Test manifest SHA-256:
  `198fef93fc45568e1dae49ae8780906fa6ef479548214d20244cd12e51201fa2`
- Threshold: 0.50; positive class: traversable; mapping: CaT blue + green.

## Work remaining after this delivery

No further experiment is required for the frozen thesis result. Remaining work
is limited to incorporating supervisor comments, applying the official UC3M
cover and PDF/A requirements, final language proofreading, and submission
validation. New architectures, datasets, and training campaigns remain future
work.
