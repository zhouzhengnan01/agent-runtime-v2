---
name: algorithm-research-scout
description: >
  Research and shortlist candidate object detection algorithms (YOLO, RT-DETR, D-FINE, DEIM, RF-DETR, DAMO-YOLO etc.)
  by searching Semantic Scholar, arXiv, and GitHub for papers, implementations, pretrained weights, license risks,
  and public metrics. Use when the user asks to compare detection algorithms, find algorithms better suited than YOLO
  for specific tasks (dense/small/occluded targets, agricultural detection), list candidate algorithms for benchmark,
  or check license/weights/metrics of a detection algorithm. Do NOT use for training commands, model export,
  ONNX/TensorRT/BModel compilation, or image classification/segmentation/OCR/tracking tasks.
---

# Algorithm Research Scout

Online research verification skill for object detection algorithm shortlisting.
Outputs a Markdown report with candidate algorithm table, paper evidence, GitHub verification,
weight checks, license risks, public mAP/FPS metrics, and benchmark recommendations.

**This skill only does research verification — no local ONNX/TensorRT/BModel compilation.**

## When to Use

Trigger when the user asks:
- "调研比 YOLO 更适合当前目标检测任务的算法"
- "列一张候选检测算法表" / "列出候选算法"
- "对比 YOLO、RT-DETR、D-FINE、RF-DETR、DAMO-YOLO"
- "找适合小目标/密集目标/遮挡目标检测的模型"
- "查某个检测算法有没有官方代码、权重、license"
- "给出应该 benchmark 的目标检测算法短名单"
- Any request about researching, comparing, or shortlisting object detection algorithms

## When NOT to Use

- User wants training commands only
- User wants model export commands only
- User wants algorithm packaging (model.json / py_plugin_imp.py)
- User wants local ONNX/TensorRT/BModel compilation verification
- User wants image classification, segmentation, OCR, tracking, or VLM tasks

## Prerequisites

Install dependencies before running:

```bash
pip install -r <skill-dir>/scripts/requirements.txt
```

Optional environment variables:
- `GITHUB_TOKEN` — increases GitHub API rate limit
- `SEMANTIC_SCHOLAR_API_KEY` — improves Semantic Scholar API stability

## How to Run

### CLI mode (recommended)

```bash
python <skill-dir>/scripts/algorithm-research-scout.py \
  --task-type "目标检测" \
  --baseline "YOLO11s" \
  --constraints "小目标" "密集目标" "遮挡严重" "边缘部署" "商用落地" \
  --mode online-research \
  --max-results 6 \
  --output object_detection_algorithm_research.md \
  --json-output-dir outputs
```

### JSON input mode

```bash
python <skill-dir>/scripts/algorithm-research-scout.py \
  --input-json '{"skill_name":"algorithm-research-scout","task_type":"目标检测","baseline":"YOLO11s","constraints":["小目标","密集目标","遮挡严重"],"mode":"online-research"}' \
  --output report.md \
  --json-output-dir outputs
```

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `--task-type` | Detection task description | "目标检测" |
| `--baseline` | Current baseline algorithm | "YOLO" |
| `--constraints` | Space-separated constraint list | None |
| `--mode` | `online-research` or `offline-template` | `online-research` |
| `--max-results` | Max results per search source | 6 |
| `--output` | Markdown output file path | stdout |
| `--json-output-dir` | Directory for JSON evidence artifacts | None |
| `--input-json` | JSON string or path to JSON file | None |

## Output Structure

The script generates a Markdown report with these sections:

1. **任务与约束** — task type, baseline, constraints
2. **调研结果摘要** — summary and usage notes
3. **候选算法总表** — algorithm table with priority, score, repo, weights, license, recommendation
4. **论文证据表** — papers with relevance scores and sources
5. **GitHub 实现核验表** — repos with match scores and verification details
6. **预训练权重核验表** — weight assets and links
7. **License 风险表** — license and commercial risk analysis
8. **公开 mAP/FPS 指标表** — extracted public metrics with comparability notes
9. **业务场景适配评分** — per-algorithm suitability scores (0–100)
10. **Benchmark 推荐短名单** — prioritized benchmark recommendations
11. **风险与不确定项** — known risks and uncertainties
12. **还需要人工补充的信息** — information requiring human follow-up

## Default Candidate Pool

The skill evaluates these algorithm families by default:

- **YOLO family** (baseline) — mature engineering pipeline
- **RT-DETR family** (P0) — real-time DETR, good for dense/occluded scenes
- **D-FINE / DEIM family** (P1) — real-time DETR improvements
- **RF-DETR family** (P1) — Roboflow DETR variant
- **DAMO-YOLO family** (P2) — industrial detection alternative

Search results may add new candidates with documented justification.

## Quality Gates

The report MUST include:
- Candidate algorithm summary table
- Paper evidence table
- GitHub implementation table
- Weight verification table
- License risk table
- Public metrics table
- Suitability scoring (0–100)
- Benchmark recommendation priorities
- Clear "已核验/未核验" (verified/unverified) status
- Disclaimer that public metrics ≠ your dataset results

## Important Notes

- Scores indicate "worth benchmarking", not guaranteed superiority over YOLO
- Public mAP/FPS from COCO or other datasets are NOT directly comparable to your dataset
- Code license ≠ weight license — check both
- Missing license check = risk, NOT "no risk"
- If no `GITHUB_TOKEN` is set, GitHub API rate limiting may cause incomplete results
- The script degrades gracefully: API failures produce "待核验" (pending verification) fields
