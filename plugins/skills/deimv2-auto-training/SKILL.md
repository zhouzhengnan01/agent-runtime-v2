---
name: deimv2-auto-training
description: Train a selected DEIMv2 DINOv3 S/M/L/X object detector from a prepared COCO dataset, selecting CUDA GPU first, Ascend NPU second, and CPU last. Use when the full-cycle algorithm workflow should reuse existing auto-annotation and synthetic-image preparation but replace YOLO training with DEIMv2.
---

# deimv2-auto-training

## Contract

Accept a prepared COCO dataset from the workflow:

- `dataset.root_dir` or `dataset_root`: image root for `training_coco`.
- `dataset.coco_json` or `coco_json`: merged COCO annotation file.
- `dataset.class_names`: detection classes in final order.
- `dataset.split`: train/val/test ratios.
- `training`: model-generated DEIMv2 training choices.
- `output.project_dir`: run output directory.

## Behavior

1. Convert the merged COCO dataset into DEIMv2 layout under `<project_dir>/prepared_dataset`.
2. Keep synthetic images in train only by reading `is_synthetic` or `source` fields from COCO images.
3. Generate `configs/dataset.yml` and `configs/train.yml` from the model-generated training spec.
4. Use the selected official DEIMv2 DINOv3 config, defaulting to `configs/deimv2/deimv2_dinov3_s_coco.yml`.
5. Resolve DEIMv2 source from `DEIMV2_ROOT`, then `spec.deimv2_root`, then bundled `vendor/deimv2`.
6. Resolve `vitt_distill.pt` from `DEIMV2_BACKBONE_CHECKPOINT`, explicit absolute spec, project `models/deimv2`, project `models`, `<DEIMV2_ROOT>/ckpts`, bundled vendor, then `/models/deimv2`.
7. Select hardware in order: CUDA GPU, Ascend NPU, CPU. Enable AMP only on CUDA.
8. Run only the formal training stage. Do not generate smoke-test configs, logs, or run directories.

## Outputs

The plugin writes:

```text
prepared_dataset/
  images/train|val|test/
  annotations/instances_train|val|test.json
configs/
  dataset.yml
  train.yml
runs/
training_summary.json
run_summary.json
logs/
```

Use `best_stg2.pth`, then `best_stg1.pth`, then `last.pth` as checkpoint preference.
