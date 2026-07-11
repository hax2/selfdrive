# ROD Experiment Suite

## Controlled Seeds

| Model | Policy | mIoU mean +/- std | F1 mean +/- std | False-safe mean | False-block mean |
|---|---|---:|---:|---:|---:|
| ROD | blue_green | 0.9138 +/- 0.0003 | 0.9482 +/- 0.0003 | 0.0351 | 0.0562 |
| ROD | blue_only | 0.9211 +/- 0.0010 | 0.9401 +/- 0.0008 | 0.0230 | 0.0597 |
| PIDNet-S | blue_green | 0.8822 +/- 0.0002 | 0.9281 +/- 0.0002 | 0.0498 | 0.0767 |
| PIDNet-S | blue_only | 0.8832 +/- 0.0016 | 0.9093 +/- 0.0014 | 0.0353 | 0.0897 |

## Paper Recipe

| Policy | mIoU | F1 | False-safe | False-block |
|---|---:|---:|---:|---:|
| blue_green | 0.9236 | 0.9547 | 0.0385 | 0.0397 |
| blue_only | 0.9280 | 0.9457 | 0.0220 | 0.0514 |

## Same-GPU Batch-1 Benchmarks

| Experiment | FPS | Mean latency (ms) | P95 latency (ms) |
|---|---:|---:|---:|
| rod_suite_benchmark_pidnet_blue_green | 272.00 | 3.67 | 3.70 |
| rod_suite_benchmark_rod_blue_green | 21.80 | 45.86 | 45.90 |
| rod_suite_benchmark_pidnet_blue_only | 270.38 | 3.69 | 4.06 |
| rod_suite_benchmark_rod_blue_only | 21.79 | 45.88 | 45.92 |

## Notes

- Paper-recipe epochs are approximated at 15 because the paper does not disclose an epoch budget.
- Hardware benchmarks are forward-pass-only, batch 1, on the same server GPU.
- Controlled seed standard deviation is population standard deviation across seeds 1337, 2027, and 4242.
