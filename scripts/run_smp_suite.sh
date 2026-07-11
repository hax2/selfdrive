#!/usr/bin/env bash
set -euo pipefail

echo "Running SMP FPN MobileNetV2 suite..."
bash scripts/run_train_eval_review.sh configs/smp_fpn_mobilenetv2.yaml

echo "Running SMP UNet EfficientNet-b0 suite..."
bash scripts/run_train_eval_review.sh configs/smp_unet_efficientnetb0.yaml

echo "SMP suite completed successfully!"
