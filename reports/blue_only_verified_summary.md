# Verified Blue-Only Results Ledger

All accuracy and safety values are population mean +/- population standard deviation over seeds 1337, 2027, and 4242.
H100 and Ryzen throughput use batch 1 and 640 x 384 input. Throughput is forward-pass-only.

| Model | mIoU | F1 | FSR | FBR | H100 FPS | Ryzen eager | Ryzen compiled | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FPN/EfficientNet-B0 | 0.9376 +/- 0.0006 | 0.9531 +/- 0.0005 | 0.0180 +/- 0.0007 | 0.0469 +/- 0.0016 | 131.40 | 5.92 | 11.70 | 5,759,614 |
| U-Net/EfficientNet-B0 | 0.9372 +/- 0.0005 | 0.9529 +/- 0.0004 | 0.0192 +/- 0.0013 | 0.0445 +/- 0.0028 | 101.24 | 6.96 | 9.66 | 6,251,614 |
| FPN/MobileNetV2 | 0.9332 +/- 0.0009 | 0.9497 +/- 0.0007 | 0.0193 +/- 0.0015 | 0.0502 +/- 0.0040 | 147.36 | 7.23 | 14.26 | 4,215,554 |
| SegFormer-B0 | 0.9286 +/- 0.0013 | 0.9461 +/- 0.0010 | 0.0214 +/- 0.0008 | 0.0523 +/- 0.0011 | 141.23 | 4.38 | 8.23 | 3,714,658 |
| U-Net/MobileNetV2 | 0.9283 +/- 0.0001 | 0.9460 +/- 0.0000 | 0.0233 +/- 0.0017 | 0.0479 +/- 0.0040 | 131.86 | 7.10 | 10.15 | 6,629,090 |
| ROD ViT-S | 0.9211 +/- 0.0010 | 0.9401 +/- 0.0008 | 0.0230 +/- 0.0013 | 0.0597 +/- 0.0019 | 21.79 | -- | -- | 29,106,050 |
| BiSeNetV2 | 0.8885 +/- 0.0026 | 0.9140 +/- 0.0020 | 0.0363 +/- 0.0028 | 0.0788 +/- 0.0052 | 167.80 | 11.48 | 16.34 | 3,341,202 |
| PIDNet-S | 0.8832 +/- 0.0016 | 0.9093 +/- 0.0014 | 0.0353 +/- 0.0016 | 0.0897 +/- 0.0039 | 270.38 | 21.16 | 27.04 | 7,623,522 |
| DDRNet-23-Slim | 0.8743 +/- 0.0042 | 0.9019 +/- 0.0039 | 0.0384 +/- 0.0022 | 0.0963 +/- 0.0116 | 245.17 | 27.74 | 36.50 | 5,694,882 |

## Audit status

- Runs present: 27/27
- Metric integrity checks: PASS
- Seed-only configuration checks: PASS
- Checkpoint hashes present: 27/27
- The six controlled PIDNet-S and ROD checkpoints were omitted from the earlier download bundle; their SHA-256 values were recovered directly from the retained server files and recorded in `blue_only_legacy_checkpoint_sha256.txt`.

Detailed per-run paths and SHA-256 values: `blue_only_verified_ledger.json`.
