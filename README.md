# TTFM Traversability Segmentation

Binary off-road traversability experiments on the [CaT dataset](https://github.com/Suvashsharma/CaT-CAVS-Traversability-Dataset-for-Off-Road-Autonomous-Driving), including the PIDNet-S baseline and the frozen EfficientSAM ViT-S architecture from [ROD](https://arxiv.org/abs/2508.08697).

## H100/A100 Setup

```bash
git clone https://github.com/hax2/selfdrive.git
cd selfdrive

python3 -m venv .venv
source .venv/bin/activate
python -m pip --isolated install --no-user --upgrade pip
python -m pip --isolated install --no-user -r requirements.txt

python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
make prepare-rod
```

`make prepare-rod` downloads the official 3.92 GB CaT archive and the 106 MB EfficientSAM ViT-S checkpoint, then audits and preprocesses the blue+green binary labels. It produces 1,002 training, 266 validation, and 544 test samples.
The download scripts use `curl` and `unzip` when available, with Python standard-library fallbacks; no system package installation is required.

Start the controlled 15-epoch blue+green comparison as a detached job:

```bash
nohup make run-rod > rod_full.log 2>&1 < /dev/null &
echo $! > rod_full.pid
```

Monitor with `tail -f rod_full.log`. Results are written under `outputs/mixed_binary_traversability_rod_vits/`.

Run the separate blue-only policy without overwriting blue+green data or results:

```bash
make prepare-rod-blue
nohup make run-rod-blue > rod_blue_full.log 2>&1 < /dev/null &
echo $! > rod_blue_full.pid
```

Blue-only outputs are written under `outputs/mixed_binary_traversability_rod_vits_blue/`. The strongest existing blue-only PIDNet-S reference has mIoU `0.8735`.

## Individual Steps

```bash
make download-cat
make download-rod-weights
make preprocess-rod
make train-rod
make eval-rod
make review-rod
make prepare-rod-blue
make run-rod-blue
```

The CaT archive comes from the dataset authors at `cavs.msstate.edu`. ROD uses the official EfficientSAM ViT-S weights from `yformer/EfficientSAM`; neither large download is stored in this repository.

## Full Follow-up Suite

After the controlled blue-only and blue+green runs exist, launch all remaining experiments with:

```bash
nohup make run-rod-suite > rod_suite.log 2>&1 < /dev/null &
echo $! > rod_suite.pid
tail -f rod_suite.log
```

## Additional Real-Time Baselines

The optional blue-only real-time suite trains SegFormer-B0, DDRNet-23-Slim,
and BiSeNetV2 under the same split, resolution, loss, seed, and 15-epoch budget.
It skips completed runs and writes a batch-one GPU benchmark for each model.

```bash
nohup bash scripts/run_modern_realtime_suite.sh > modern_realtime_suite.log 2>&1 < /dev/null &
echo $! > modern_realtime_suite.pid
tail -f modern_realtime_suite.log
```

SegFormer-B0 and BiSeNetV2 use pretrained backbones. DDRNet-23-Slim is trained
from scratch because its official ImageNet checkpoint is not distributed through
a stable programmatic URL. Keep that initialization difference explicit when
interpreting the results.

For the complete comparison, including the full FPN/U-Net by
MobileNetV2/EfficientNet-B0 2x2 design and seeds 1337, 2027, and 4242 for
every lightweight model, run:

```bash
nohup bash scripts/run_realtime_comparison_suite.sh > realtime_comparison_suite.log 2>&1 < /dev/null &
echo $! > realtime_comparison_suite.pid
tail -f realtime_comparison_suite.log
```

Completed seed-1337 runs are reused. Generated repeat-seed configurations are
saved under `configs/generated/realtime/`.

Run the same seven models and three seeds with the blue+green traversability
policy in separate output directories:

```bash
nohup bash scripts/run_realtime_blue_green_suite.sh > realtime_blue_green_suite.log 2>&1 < /dev/null &
echo $! > realtime_blue_green_suite.pid
tail -f realtime_blue_green_suite.log
```

The blue+green suite omits duplicate latency benchmarks because changing the
label mapping does not change any model architecture or inference operation.
Generated configurations are saved under
`configs/generated/realtime_blue_green/`.

The suite is resumable: experiments with an existing `test_metrics.json` are skipped. It runs:

- Controlled ROD seeds `1337`, `2027`, and `4242` for blue-only and blue+green.
- The closest documented paper recipe for both policies: CE only, neutral weights, AdamW `1e-3`, weight decay `0.01`, batch 8, and iteration-level polynomial decay with power `0.9`.
- Matched PIDNet-S seeds `1337`, `2027`, and `4242` for both policies, enabling seed-wise architecture comparisons and same-H100 latency baselines.
- Forward-pass benchmarks over 128 test images with 5 warmups and 10 repeats.

When complete, download `rod_suite_results.zip`. It contains aggregate JSON/Markdown reports, individual metrics and histories, exact generated configs, benchmarks, the suite log, and both paper-recipe best checkpoints. Other repeat-seed checkpoints remain on the server to avoid an unnecessarily large archive.

## ORFD ROD vs FPN Comparison

After pulling the repository on a CUDA server, the complete controlled ORFD
comparison is one command:

```bash
nohup bash scripts/run_orfd_comparison.sh > orfd_comparison.log 2>&1 < /dev/null &
echo $! > orfd_comparison.pid
tail -f orfd_comparison.log
```

The script creates `.venv-orfd`, installs the Python dependencies, downloads
only ORFD RGB images and ground-truth masks from the Academic Torrents release
(about 16.8 GB), downloads EfficientSAM ViT-S, validates and preprocesses the
available train/validation/test partition, trains ROD ViT-S and
FPN/EfficientNet-B0 for seeds
1337, 2027, and 4242, evaluates their best checkpoints, benchmarks batch-one
CUDA inference, and writes `reports/orfd/comparison.md` and `.json`.

Training uses a maximum safety cap of 60 epochs, not a fixed 60-epoch budget.
It stops after validation mIoU fails to improve by at least 0.0005 for eight
epochs, after a minimum of 15 epochs. Non-finite training/validation loss or
metrics abort the run immediately. Completed runs and downloaded files are
reused when the script is restarted.

For a single-seed validation run before committing to the full comparison:

```bash
ORFD_SEEDS=1337 bash scripts/run_orfd_comparison.sh
```

This is a controlled comparison at 640 by 384 with a matched optimization
recipe. The Academic Torrents mirror contains 8,392/1,245/2,193 usable pairs,
whereas the paper reports 8,398/1,245/2,555. The preprocessing step labels this
as the known incomplete mirror and refuses any unrecognised partial download.
If the complete dataset is already present under `datasets/ORFD`, it detects
and uses the full profile; set `ORFD_SKIP_DOWNLOAD=1` to avoid fetching the
mirror in that case, and set `ORFD_RAW_ROOT` if it lives elsewhere. A
mirror-based result must not be described as an
exact reproduction of ROD's published score; the paper also does not disclose
its epoch budget or complete timing procedure.
