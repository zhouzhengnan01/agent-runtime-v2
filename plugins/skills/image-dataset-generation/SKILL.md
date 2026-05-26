---
name: image-dataset-generation
description: Composite two input images or image folders into realistic synthetic training images through the TokenCloud/Bailian wan2.7-image-pro image generation API.
metadata:
  author: jetlinks
  version: "2.0"
  language: zh-CN
---

# image-dataset-generation

This project keeps the skill id `image-dataset-generation` for workflow compatibility, but its implementation is now the Image Dataset Composite capability.

Use this skill to synthesize composite images for detection training from two input image sources and a text prompt. It only creates composite images; dataset inspection, count planning, SAM3 annotation, train/val/test splitting, and YOLO training are handled by downstream workflow steps.

## Required Inputs

- `image1`: local absolute file path, local absolute folder path, URL, or data URL for the first/reference/background image source.
- `image2`: local absolute file path, local absolute folder path, URL, or data URL for the second/reference/foreground image source.
- `prompt`: composition prompt entered by the user or passed by the workflow.
- `output_dir`: local absolute directory for generated composite images.

When `image1` or `image2` is a folder, each generated image randomly samples one supported image from that folder. Single image paths, URLs, and data URLs are also supported.

Optional:

- `count`, default `1`
- `size`, default `768*768`
- `negative_prompt`
- `timeout`, default `180`
- `sleep`, default `1.0`

The model service is configured in `scripts/image-composite.py`.

Current default:

```python
BASE_URL = "https://www.tokencloud.yun/v1/images/generations"
API_KEY = "sk-6qbqlTAZV7qVszLplZhxZH2yvQwSMRotrOT8wr9aVPII8BYo"
MODEL_NAME = "wan2.7-image-pro"
```

## CLI

```powershell
python scripts/run_composite.py --input D:/work/composite_inputs/scene_01_00001.json
```

Input JSON:

```json
{
  "model": {
    "timeout": 180,
    "size": "768*768",
    "count": 1
  },
  "task": {
    "image1": "D:/dataset/background_images",
    "image2": "D:/dataset/object_images",
    "prompt": "Composite the object naturally into the scene, realistic surveillance camera frame",
    "output_dir": "D:/work/composite_images/scene_01"
  }
}
```

## Full Workflow Role

In `yolo_training_flow`, the user uploads the training dataset first, then uploads `image1` and `image2` for composition, then provides the composition prompt, labels, and optional training parameters.

Composite images must be marked as synthetic before downstream splitting. They must only be used in `train`; `val` and `test` must remain real-image-only.
