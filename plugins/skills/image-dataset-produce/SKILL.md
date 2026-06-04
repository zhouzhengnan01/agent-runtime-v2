---
name: image-dataset-produce
description: Generate synthetic training images through the local flux2 image-gen HTTP API from image1 reference images and prompt plans.
metadata:
  author: jetlinks
  version: "1.1"
  language: zh-CN
---

# image-dataset-produce

Use this skill to create synthetic images for detection training. In the full YOLO training workflow, this is the fallback image producer when `image-dataset-generation` composite generation is unavailable or out of quota. Do not let the user guess the count manually; use the synthetic plan produced by `data-auto-annotation/scripts/plan_synthetic_augmentation.py`.

## Required Inputs

- `input_image`: local absolute path to one reference image from the uploaded `image1` folder.
- `prompt`: generation prompt entered by the user in Codex chat or passed by `gpu-training-orchestrator`.
- `output_dir`: local absolute directory for generated images.

Optional:

- `api_url`, default `http://218.67.242.10:58801/flux2/generate`
- `token`, default `abc@123`
- `timeout`, default `120`

## CLI

```powershell
python scripts/run_generation.py --input D:/work/generation_inputs/scene_01_00001.json
```

Input JSON:

```json
{
  "model": {
    "api_url": "http://192.168.33.25:8801/flux2/generate",
    "token": "abc@123",
    "timeout": 120
  },
  "task": {
    "input_image": "D:/dataset/images/example.jpg",
    "prompt": "smoking detection, contains clearly annotatable smoking_person, surveillance camera view",
    "output_dir": "D:/work/synthetic_images/scene_01"
  }
}
```

## Full Workflow Role

This skill only generates images. It does not decide train/val/test splits.

When used from Codex chat, the prompt and reference image should come from the user's message/attachments. Do not hard-code a prompt or reference image path in the skill.

After generation, call `data-auto-annotation` with:

```powershell
--source synthetic --is-synthetic
```

Then `gpu-training-orchestrator` merges real and synthetic COCO files and ensures:

- synthetic images go only to `train`
- `val` contains real images only
- `test` contains real images only

## Important Constraint

Generated images must never be used directly as validation data. They must be labeled as synthetic in COCO metadata before training.
