# ROD CaT Architecture Comparison

## Method

ROD is based on the architecture in [ROD: RGB-Only Fast and Efficient Off-road Freespace Detection](https://arxiv.org/abs/2508.08697). It uses the pretrained EfficientSAM ViT-S image encoder, freezes all encoder parameters, and trains a residual multiscale convolutional decoder for binary segmentation.

The implementation follows the authors' released code where it is more specific than the paper: decoder fusion uses transformer blocks 2 through 12 and sequential residual fusion. Images are normalized with the EfficientSAM ImageNet statistics and resized to the encoder's fixed 1024 by 1024 input.

## Controlled Comparison

`configs/rod_vits_cat.yaml` uses the same processed blue-and-green CaT labels, split, 640 by 384 training targets, augmentation, class weighting, CE plus Dice objective, learning rate, and 15-epoch schedule as `configs/blue_green_second_run.yaml`. Fused scaled-dot-product attention allows physical batch size 4 on the RTX 5060, exactly matching the PIDNet-S baseline.

The existing PIDNet-S reference test result is mIoU 0.8780, traversable IoU 0.8606, traversable F1 0.9251, false-safe rate 0.0487, and false-block rate 0.0834.

## Commands

```bash
make download-rod-weights
make run-rod
```

The smoke pipeline uses `configs/rod_vits_cat_smoke.yaml`. Its metrics are not suitable for architecture comparison because it intentionally trains on only 8 images.
