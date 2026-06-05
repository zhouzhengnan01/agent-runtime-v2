---
name: behavior-detection
description: Evidence-first behavior detection; never emit visual scores without image, video, or structured visual evidence
tags:
  - behavior
  - detection
  - evidence
  - vision
---

# When To Use

- Use when the user asks for behavior detection, intrusion, falling, lingering, or restricted-area judgment.
- Text-only requests are treated as rule candidates, not visual recognition results.
- Image, video, or structured visual evidence is required before returning a visual confidence score.

# Inputs

- `text_rule_candidates`: candidate behavior categories from text.
- `has_visual_evidence`: whether the request includes visual or structured evidence.
- `attachments`: optional image/video metadata.

# Output Rules

- With text only, return `needs_visual_confirmation` and keep `confidence` at `0.0`.
- Keep text-rule confidence separate from visual confidence.
- With visual or structured evidence, return the detection decision, category, confidence, evidence mode, and reason.

# Verification

- `evidence_mode` must be declared.
- Text-only mode must not include a positive visual score.
- Text-rule confidence must be represented separately when no visual evidence exists.
