# Frozen PIDNet-S baseline

This is the single result to reproduce for the thesis. Optional ROD,
EfficientSAM, VLM, and new-dataset work is outside the frozen thesis baseline.

## Exact artifacts

- Portable frozen configuration: `configs/frozen_pidnet_cat.yaml`
- Original training configuration: `configs/ablations/controlled_ablation_augment_off.yaml`
- Checkpoint: `outputs/controlled_ablation_augment_off/checkpoints/best.pt`
- Processed split manifest: `data_processed_blue_green/manifest.json`
- Evaluation entry point: `src/ttfm/eval.py`, invoked through `ttfm.cli eval`
- Saved official test result: `outputs/controlled_ablation_augment_off/test_metrics.json`
- Test split: `mixed`, 544 images, evaluated once at 640 x 384
- Positive class: traversable; CaT blue (sedan) and green (pickup) are merged
- Decision threshold: 0.50

SHA-256 identities recorded on 10 July 2026:

```text
b9862fe49713c0622d760b94aa3fc055fdaf6549a67a4ee3492004d9c591f471  outputs/controlled_ablation_augment_off/checkpoints/best.pt
07b07bbb1bb9e6f0586df6598d2cf982f8af845fb3e16785810615c80ebdaa74  configs/frozen_pidnet_cat.yaml
62869964e3fd30b3cba59d28f1799c6add73002c1a39af4dfb32641626d27928  configs/ablations/controlled_ablation_augment_off.yaml
198fef93fc45568e1dae49ae8780906fa6ef479548214d20244cd12e51201fa2  data_processed_blue_green/manifest.json
```

## Re-evaluation

From the repository root, with the dependencies installed:

```bash
PYTHONPATH=src python -m ttfm.cli \
  --config configs/frozen_pidnet_cat.yaml \
  eval --split test
```

The command loads `best.pt`, does not train or update weights, iterates over the
test manifest without shuffling, aggregates a 2 x 2 pixel confusion matrix, and
writes `test_metrics.json`. Two consecutive CPU evaluations on 10 July 2026
produced identical results: mIoU 0.8939407, FSR 0.0431759, and FBR 0.0703687.
Expected `[ground truth, prediction]` counts are TN 73,151,244; FP 3,300,886;
FN 4,027,996; TP 53,213,314. These fresh CPU counts supersede the earlier
GPU-produced saved counts; the small numerical difference affects 1,765 pixel
decisions out of 133,693,440 and does not change any headline value at the
reported precision.

CPU speed is a separate forward-pass-only benchmark on an AMD Ryzen 5 5500
(batch size 1, 160 measured passes): 45.17 ms mean latency and 22.14 FPS. It is
not end-to-end camera latency and is not a hardware-matched literature result.

## Qualitative review reproduction

```bash
PYTHONPATH=src python -m ttfm.cli \
  --config configs/frozen_pidnet_cat.yaml \
  review --split test --top-k 10
```

This ranks all 544 images by per-image FSR, FBR, and boundary-band error and
writes the selected triptychs and `summary.json` under
`outputs/controlled_ablation_augment_off/test_review/`.

## Remaining work after the frozen package

- Incorporate supervisor feedback without changing the frozen experiment.
- Apply the official university cover and PDF/A requirements.
- Perform final language proofreading and submission checks.
- Do not begin a new training campaign after 17 July; keep optional architectures
  and datasets as future work.
