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

Start the controlled 15-epoch comparison in a persistent session:

```bash
tmux new -s rod
make run-rod
```

Detach with `Ctrl-b d` and reconnect with `tmux attach -t rod`. Results are written under `outputs/mixed_binary_traversability_rod_vits/`. The reference PIDNet-S result is mIoU `0.8780`, traversable F1 `0.9251`, false-safe rate `0.0487`, and false-block rate `0.0834`.

## Individual Steps

```bash
make download-cat
make download-rod-weights
make preprocess-rod
make train-rod
make eval-rod
make review-rod
```

The CaT archive comes from the dataset authors at `cavs.msstate.edu`. ROD uses the official EfficientSAM ViT-S weights from `yformer/EfficientSAM`; neither large download is stored in this repository.

