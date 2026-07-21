# Convergence-aware architecture suite

This suite retrains all nine CaT architectures for seeds 1337, 2027, and 4242
under the thesis's principal blue+green policy. The default command creates 27
independent experiments and runs at most two GPU processes concurrently.
Passing `--policy both` remains available if a later run also needs the
blue-only policy.

The maximum duration and learning-rate schedule are deliberately separate:

- maximum training ceiling: 300 epochs;
- cosine decay to the configured minimum learning rate: 60 epochs;
- minimum training duration: 60 epochs;
- validation-mIoU early stopping: patience 25, minimum improvement 0.0001.

After epoch 60 the learning rate remains at its minimum instead of beginning a
new cosine cycle. Existing 15-epoch configurations and outputs are not changed.
The generated suite configs retain a deployment/evaluation-ready `best.pt`,
but omit its unused optimizer state and skip the otherwise once-per-epoch
`last.pt` rewrite to reduce disk traffic. This does not change optimization
or checkpoint selection.

## Server launch

```bash
git pull
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"

python scripts/run_convergence_suite.py --prepare-only

nohup python scripts/run_convergence_suite.py \
  --policy blue-green \
  --jobs 2 \
  > convergence_suite.log 2>&1 < /dev/null &
echo $! > convergence_suite.pid
```

The real launch performs a strict preflight for the selected processed dataset,
its mapping file, and the EfficientSAM weight used by ROD. If the server is
missing the blue+green dataset, prepare it first with `make prepare-rod`.
`--prepare-only` does not require those large local artifacts; it checks
config generation.

Follow the live suite log:

```bash
tail -f convergence_suite.log
```

Print the latest saved status without attaching to the running process:

```bash
python scripts/run_convergence_suite.py --policy blue-green --status-only
```

The runner writes:

- `outputs/convergence_blue_green_e300_c60_m60_p25/status.json`: per-job progress and
  ETA estimates;
- `outputs/convergence_blue_green_e300_c60_m60_p25/results.csv`: completed test
  metrics in spreadsheet form;
- `outputs/convergence_blue_green_e300_c60_m60_p25/results.json`: the same completed
  results as JSON;
- `outputs/convergence_blue_green_e300_c60_m60_p25/summary.csv`: model means
  and standard deviations across the completed seeds;
- `outputs/convergence_blue_green_e300_c60_m60_p25/summary.json`: the same aggregate
  summary as JSON;
- `outputs/convergence_blue_green_e300_c60_m60_p25/logs/`: one append-only log per
  experiment;
- `outputs/<experiment>/training_progress.json`: current epoch, best epoch,
  rolling seconds per epoch, patience counter, and per-run ETA bounds.

The suite-level ETA is intentionally shown as a range. `eta_plan` assumes
unfinished runs stop around the minimum duration plus patience; `eta_cap`
assumes every run reaches the 300-epoch ceiling. Both are recalculated from the
median observed epoch time and become useful after the first few epochs.

## Restart and failure handling

Rerun the identical launch command after an interruption. Experiments with a
saved `train_summary.json`, `best.pt`, and `test_metrics.json` are skipped.
If training completed but evaluation did not, only evaluation is relaunched.
An interrupted training process restarts that one experiment from epoch 1;
completed experiments are never repeated.

Full qualitative review generation is disabled by default because it writes
many images and is unnecessary for the ranking. Add `--with-review` if those
artifacts are required.

The launcher prevents two ROD training jobs from running simultaneously. If
the H100 partition still runs out of memory, relaunch with `--jobs 1`; completed
experiments will be retained.
