---
name: data-auto-annotation
description: Prepare detection datasets for YOLO training: inspect uploaded datasets, auto-annotate images with SAM3, convert YOLO labels to COCO, optionally call image generation, annotate synthetic images, merge COCO files, and create real-only val/test YOLO prepared_dataset.
compatibility: Designed for Python 3 with network access to a SAM3 HTTP endpoint.
metadata:
  author: jetlinks
  version: "1.1"
  language: zh-CN
---

# data-auto-annotation

Use this skill for all dataset preparation before YOLO training. This skill owns data processing; `gpu-training-orchestrator` only trains from the prepared `dataset.yaml`.

## Inputs

- `image_path`: local absolute path to one image or an image folder.
- `labels`: detection labels entered by the user in Codex chat, mapped to `--text-prompts`.
- `format`: always COCO.
- `source`: `real` for user-uploaded images, `synthetic` for generated images.
- `is_synthetic`: true only for generated images.

## CLI

Real uploaded images:

```powershell
python scripts/sam3-predict.py --input-dir D:/dataset/images --text-prompts smoking_person cigarette --source real --output D:/work/real_coco.json
```

Generated images:

```powershell
python scripts/sam3-predict.py --input-dir D:/work/synthetic_images --text-prompts smoking_person cigarette --source synthetic --is-synthetic --output D:/work/synthetic_coco.json
```

## Output Contract

The output must be standard COCO and include:

- `images`
- `annotations`
- `categories`
- `licenses`
- `info`

Each `images[]` item must include:

```json
{
  "id": 1,
  "file_name": "example.jpg",
  "width": 1280,
  "height": 720,
  "source": "real",
  "is_synthetic": false
}
```

This source metadata is required by this skill's YOLO split step so that validation and test splits use real images only.

## Dataset Cases

If the uploaded `dataset/` contains only `images/`, annotate the real images first and output `real_coco.json`.

If the generated dataset is produced by `image-dataset-generation`, annotate it with `--source synthetic --is-synthetic` and output `synthetic_coco.json`.

Do not hard-code labels. They must come from the user's chat request or the orchestrator's parsed pipeline spec.

## Full Data Preparation Pipeline

The application workflow calls:

```powershell
python scripts/run_data_preparation_pipeline.py --input-json input.json
```

This pipeline outputs:

- `pipeline_work/synthetic_plan.json`
- `pipeline_work/real_coco.json`
- `pipeline_work/synthetic_coco.json` when generation is enabled
- `pipeline_work/merged_coco.json` when synthetic images exist
- `prepared_data/prepared_dataset/images/train`
- `prepared_data/prepared_dataset/images/val`
- `prepared_data/prepared_dataset/images/test`
- `prepared_data/prepared_dataset/labels/train`
- `prepared_data/prepared_dataset/labels/val`
- `prepared_data/prepared_dataset/labels/test`
- `prepared_data/dataset.yaml`
- `prepared_data/data_preparation_summary.json`

The hard rule is enforced here:

- synthetic images only go to train
- val uses real images only
- test uses real images only
