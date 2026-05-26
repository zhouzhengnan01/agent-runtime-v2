---
name: data-auto-annotation
description: Use SAM3 to auto-annotate one image or an image folder, export standard COCO JSON, and tag images as real or synthetic for downstream training splits.
compatibility: Designed for Python 3 with network access to a SAM3 HTTP endpoint.
metadata:
  author: jetlinks
  version: "1.1"
  language: zh-CN
---

# data-auto-annotation

Use this skill when uploaded detection data has images but no labels, or when generated images need automatic COCO annotation before training.

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

This source metadata is required by `gpu-training-orchestrator` so that validation and test splits use real images only.

## Dataset Cases

If the uploaded `dataset/` contains only `images/`, annotate the real images first and output `real_coco.json`.

If the generated dataset is produced by `image-dataset-generation`, annotate it with `--source synthetic --is-synthetic` and output `synthetic_coco.json`.

Do not hard-code labels. They must come from the user's chat request or the orchestrator's parsed pipeline spec.
