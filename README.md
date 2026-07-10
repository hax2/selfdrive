# TTFM Traversability Segmentation

Binary off-road traversability experiments on the [CaT dataset](https://github.com/Suvashsharma/CaT-CAVS-Traversability-Dataset-for-Off-Road-Autonomous-Driving), including the PIDNet-S baseline and the frozen EfficientSAM ViT-S architecture from [ROD](https://arxiv.org/abs/2508.08697).

## H100/A100 Setup

```bash
git clone https://github.com/hax2/selfdrive.git
cd selfdrive

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

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
