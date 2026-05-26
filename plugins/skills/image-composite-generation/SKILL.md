---
name: image-composite-generation
description: Composite two input images or image folders into realistic synthetic training images through the TokenCloud/Bailian image generation API.
metadata:
  author: jetlinks
  version: "0.1"
  language: zh-CN
---

# image-composite-generation

Use this skill to synthesize composite images for detection training from two image sources and a text prompt.

## Required Inputs

- `image1`: local absolute file path, local absolute folder path, URL, or data URL for the first/reference/background image source.
- `image2`: local absolute file path, local absolute folder path, URL, or data URL for the second/reference/foreground image source.
- `prompt`: composition prompt entered by the user or passed by a workflow.
- `output_dir`: local absolute directory for generated composite images. Defaults to the current thread outputs directory when executed through the runtime.

When `image1` or `image2` is a folder, each generated image randomly samples one supported image from that folder. Single image paths, URLs, and data URLs are also supported.

## Optional Inputs

- `count`, default `1`
- `size`, default `768*768`
- `negative_prompt`
- `timeout`, default `180`
- `sleep`, default `1.0`
- `prompt_extend`, default `false`
- `watermark`, default `false`

## Runtime Notes

The current implementation keeps the API URL, model, and token defaults inside the script for compatibility with the imported prototype. They can also be overridden in the skill spec.

Composite images are synthetic data. Downstream dataset preparation must keep synthetic images out of validation and test splits.
