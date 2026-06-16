import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List
from urllib import request


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _base_ratio(real_train_count: int) -> float:
    if real_train_count < 70:
        return 4.0
    if real_train_count < 210:
        return 3.0
    if real_train_count < 560:
        return 2.0
    if real_train_count < 1050:
        return 1.0
    return 0.5


def _difficulty_signals(task_description: str, class_names: List[str]) -> List[Dict]:
    text = (task_description + " " + " ".join(class_names)).lower()
    groups = [
        (
            ["small object", "tiny", "far", "distant", "bottle", "cup", "cigarette", "phone", "helmet", "小目标", "远距离", "水瓶", "瓶子", "香烟"],
            0.25,
            "small or distant target",
        ),
        (["occlusion", "crowd", "clutter", "overlap", "遮挡", "人群", "拥挤", "杂乱"], 0.20, "occlusion, crowding, or clutter"),
        (
            ["behavior", "action", "pose", "smoking", "vape", "fall", "fight", "行为", "动作", "姿态", "抽烟", "吸烟", "跌倒", "打架"],
            0.20,
            "behavior/action detection",
        ),
        (
            ["surveillance", "cctv", "night", "low light", "low resolution", "blur", "监控", "低清", "弱光", "夜间", "模糊"],
            0.20,
            "surveillance or low-quality imagery",
        ),
        (["fine-grained", "similar", "defect", "scratch", "damage", "细粒度", "相似", "缺陷", "划痕", "破损"], 0.15, "fine-grained or visually similar classes"),
    ]
    signals: List[Dict] = []
    for keywords, boost, reason in groups:
        matched = [keyword for keyword in keywords if keyword in text]
        if matched:
            signals.append({"reason": reason, "boost": boost, "matched_keywords": matched})
    return signals


def _difficulty_multiplier(task_description: str, class_names: List[str]) -> float:
    multiplier = 1.0 + sum(item["boost"] for item in _difficulty_signals(task_description, class_names))
    return min(multiplier, 1.6)


def _category_counts(coco: Dict) -> Dict[str, int]:
    cat_id_to_name = {int(cat["id"]): str(cat["name"]) for cat in coco.get("categories", [])}
    counts = Counter()
    for ann in coco.get("annotations", []):
        name = cat_id_to_name.get(int(ann.get("category_id", -1)), "unknown")
        counts[name] += 1
    return dict(counts)


def _image_size_summary(coco: Dict) -> Dict[str, Any]:
    sizes = []
    for image in coco.get("images", []):
        width = image.get("width")
        height = image.get("height")
        if width and height:
            try:
                sizes.append((int(width), int(height)))
            except (TypeError, ValueError):
                continue
    if not sizes:
        return {"known_size_count": 0}
    counts = Counter(sizes)
    common = [
        {"width": width, "height": height, "count": count}
        for (width, height), count in counts.most_common(5)
    ]
    widths = [item[0] for item in sizes]
    heights = [item[1] for item in sizes]
    return {
        "known_size_count": len(sizes),
        "min_width": min(widths),
        "max_width": max(widths),
        "min_height": min(heights),
        "max_height": max(heights),
        "common_sizes": common,
    }


def _scarcity_multiplier(category_counts: Dict[str, int]) -> float:
    if not category_counts:
        return 1.15
    min_count = min(category_counts.values())
    if min_count < 20:
        return 1.30
    if min_count < 50:
        return 1.15
    return 1.0


def _round_count(value: float) -> int:
    return int(math.ceil(max(0.0, value) / 10.0) * 10)


def _build_prompt_plan(
    task_description: str,
    class_names: List[str],
    synthetic_count: int,
    negative_ratio: float,
    generation_prompt: str = "",
) -> List[Dict]:
    synthetic_count = max(0, int(synthetic_count))
    negative_count = min(synthetic_count, max(0, int(round(synthetic_count * negative_ratio))))
    positive_count = max(0, synthetic_count - negative_count)
    labels = ", ".join(class_names) if class_names else "target classes"
    scenes = [
        "realistic photo, target objects clearly visible, natural background, complete target visible",
        "fixed camera or surveillance view, moderate distance, complex background, target remains labelable",
        "indoor scene, ordinary lighting, realistic application context, mild occlusion",
        "outdoor scene, natural light, diverse viewpoint, target scale and pose vary",
        "multi-object scene, only some objects belong to the requested target classes",
        "weak light, slight motion blur, or image noise, target still recognizable and annotatable",
    ]

    per_scene = positive_count // len(scenes) if scenes else positive_count
    remainder = positive_count - per_scene * len(scenes)
    custom_prompt = generation_prompt.strip()
    plan: List[Dict] = []

    for idx, scene in enumerate(scenes):
        count = per_scene + (1 if idx < remainder else 0)
        if count <= 0:
            continue
        if custom_prompt:
            prompt = (
                f"{custom_prompt}, target classes: {labels}, scene variation: {scene}, "
                "suitable for precise object-detection bounding boxes"
            )
        else:
            prompt = (
                f"{task_description}, contains clearly annotatable target classes: {labels}, {scene}, "
                "realistic image, suitable for object-detection bounding boxes"
            )
        plan.append({"type": "positive", "prompt": prompt, "count": count})

    if negative_count > 0:
        plan.append(
            {
                "type": "negative",
                "prompt": (
                    f"realistic scene similar to {task_description}, but does not contain target classes: {labels}, "
                    "useful only as optional hard negative samples"
                ),
                "count": negative_count,
            }
        )
    return plan


def _heuristic_recommendation(real_train_count: int, task_description: str, class_names: List[str], category_counts: Dict[str, int]) -> Dict[str, Any]:
    base_ratio = _base_ratio(real_train_count)
    difficulty_multiplier = _difficulty_multiplier(task_description, class_names)
    scarcity_multiplier = _scarcity_multiplier(category_counts)
    raw_recommended = real_train_count * base_ratio * difficulty_multiplier * scarcity_multiplier
    return {
        "recommended_synthetic_count": _round_count(raw_recommended),
        "base_ratio": base_ratio,
        "difficulty_multiplier": round(difficulty_multiplier, 3),
        "difficulty_signals": _difficulty_signals(task_description, class_names),
        "scarcity_multiplier": round(scarcity_multiplier, 3),
    }


def _llm_recommendation(
    *,
    llm_config: Dict[str, Any],
    task_description: str,
    generation_prompt: str,
    class_names: List[str],
    real_image_count: int,
    real_train_count: int,
    split_train: float,
    max_synthetic: int,
    min_real_images_for_skip: int,
    category_counts: Dict[str, int],
    image_size_summary: Dict[str, Any],
) -> Dict[str, Any]:
    base_url = str(llm_config.get("base_url") or "").rstrip("/")
    model = str(llm_config.get("model") or "").strip()
    if not base_url or not model:
        raise ValueError("LLM planner requires base_url and model")

    system_prompt = (
        "You are a senior computer vision dataset planner for YOLO object detection. "
        "You must independently decide how many synthetic images to generate from the provided dataset facts. "
        "Do not use a fixed default such as 100. Choose any integer from 0 to max_synthetic based on the actual task, "
        "class scarcity, annotation counts, image count, visual difficulty, and risk of overfitting. "
        "For critically small datasets with action detection, small objects, or rare classes, hundreds of synthetic images can be appropriate. "
        "Return JSON only. Do not include markdown."
    )
    planner_facts = {
        "task": task_description,
        "generation_prompt": generation_prompt,
        "labels": class_names,
        "real_image_count": real_image_count,
        "estimated_real_train_count": real_train_count,
        "annotation_counts_per_class": category_counts,
        "image_size_summary": image_size_summary,
        "split_ratio": {"train": split_train, "val": None, "test": None},
        "max_synthetic": max_synthetic,
        "skip_generation_threshold_real_images": min_real_images_for_skip,
        "constraints": [
            "recommend 0 synthetic images if the real dataset is already sufficient",
            "recommended_synthetic_count must be an integer from 0 to max_synthetic",
            "sum(generation_plan[].count) must equal recommended_synthetic_count if generation_plan is provided",
            "synthetic images are train-only; do not plan validation or test synthetic images",
            "prefer realistic, annotatable detection samples over artistic images",
            "do not exceed max_synthetic when max_synthetic is greater than 0",
        ],
        "expected_json_schema": {
            "should_generate": "boolean",
            "recommended_synthetic_count": "integer",
            "confidence": "number from 0 to 1",
            "reason": "short explanation",
            "difficulty_signals": "array of short strings",
            "risk_notes": "array of short strings",
            "generation_plan": "optional array of {type,prompt,count,purpose}",
        },
    }
    user_payload = {
        "instruction": (
            "Based only on planner_facts, choose the synthetic image count. "
            "Do not infer the count from any hidden rule or ratio. Explain the dataset-specific reason."
        ),
        "planner_facts": planner_facts,
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": float(llm_config.get("temperature", 0.2)),
        "max_tokens": int(llm_config.get("max_tokens", 1024)),
    }
    headers = {"Content-Type": "application/json"}
    api_key = str(llm_config.get("api_key") or "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(f"{base_url}/chat/completions", data=data, headers=headers, method="POST")
    timeout = int(llm_config.get("timeout", 120) or 120)
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    parsed = json.loads(raw)
    choices = parsed.get("choices") or []
    if not choices:
        raise ValueError("LLM planner returned no choices")
    content = str((choices[0].get("message") or {}).get("content") or "").strip()
    result = _parse_llm_json(content)
    result["_llm_request_facts"] = planner_facts
    result["_llm_raw_content"] = content
    result["_llm_model"] = model
    result["_llm_base_url"] = base_url
    return result


def _parse_llm_json(content: str) -> Dict[str, Any]:
    if not content:
        raise ValueError("LLM planner returned empty content")
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    if not cleaned.startswith("{"):
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"LLM planner did not return JSON: {content[:300]}")
        cleaned = match.group(0)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("LLM planner JSON must be an object")
    return parsed


def _validated_llm_count(llm_plan: Dict[str, Any], real_image_count: int, max_synthetic: int, min_real_images_for_skip: int) -> tuple[bool, int]:
    should_generate = bool(llm_plan.get("should_generate"))
    try:
        recommended = int(float(llm_plan.get("recommended_synthetic_count", 0)))
    except (TypeError, ValueError):
        recommended = 0
    if real_image_count >= min_real_images_for_skip:
        should_generate = False
        recommended = 0
    if not should_generate:
        recommended = 0
    recommended = max(0, recommended)
    if max_synthetic >= 0:
        recommended = min(recommended, max_synthetic)
    return should_generate and recommended > 0, recommended


def build_plan(
    coco_path: Path,
    task_description: str,
    split_train: float,
    max_synthetic: int,
    negative_ratio: float,
    min_real_images_for_skip: int,
    generation_prompt: str = "",
    planner: str = "llm",
    fixed_synthetic_count: int = 20,
    llm_config: Dict[str, Any] | None = None,
) -> Dict:
    coco = _load_json(coco_path)
    images = coco.get("images", [])
    categories = coco.get("categories", [])
    class_names = [str(cat.get("name", "")).strip() for cat in categories if str(cat.get("name", "")).strip()]
    real_image_count = len(images)
    if real_image_count <= 0:
        raise ValueError("COCO images is empty; cannot plan synthetic augmentation.")

    real_train_count = max(1, int(real_image_count * split_train))
    category_counts = _category_counts(coco)
    image_size_summary = _image_size_summary(coco)
    heuristic = _heuristic_recommendation(real_train_count, task_description, class_names, category_counts)
    planner_type = "heuristic_rules"
    planner_error = ""
    llm_raw: Dict[str, Any] = {}
    should_generate = real_image_count < min_real_images_for_skip
    recommended = 0

    if planner == "fixed":
        recommended = max(0, int(fixed_synthetic_count))
        if max_synthetic >= 0:
            recommended = min(recommended, max_synthetic)
        should_generate = recommended > 0
        planner_type = "fixed_count"
    elif planner == "llm":
        try:
            llm_raw = _llm_recommendation(
                llm_config=llm_config or {},
                task_description=task_description,
                generation_prompt=generation_prompt,
                class_names=class_names,
                real_image_count=real_image_count,
                real_train_count=real_train_count,
                split_train=split_train,
                max_synthetic=max_synthetic,
                min_real_images_for_skip=min_real_images_for_skip,
                category_counts=category_counts,
                image_size_summary=image_size_summary,
            )
            should_generate, recommended = _validated_llm_count(llm_raw, real_image_count, max_synthetic, min_real_images_for_skip)
            planner_type = "llm_dynamic"
        except Exception as exc:
            planner_error = str(exc)
            planner_type = "heuristic_rules_fallback"

    if planner_type not in {"llm_dynamic", "fixed_count"}:
        if should_generate:
            recommended = int(heuristic["recommended_synthetic_count"])
            if max_synthetic >= 0:
                recommended = min(recommended, max_synthetic)
        else:
            recommended = 0

    prompt_plan = _build_prompt_plan(task_description, class_names, recommended, negative_ratio, generation_prompt)

    return {
        "task_description": task_description,
        "real_image_count": real_image_count,
        "estimated_real_train_count": real_train_count,
        "split_train": split_train,
        "class_names": class_names,
        "category_annotation_counts": category_counts,
        "image_size_summary": image_size_summary,
        "planner_type": planner_type,
        "planner_error": planner_error,
        "llm_request_facts": llm_raw.get("_llm_request_facts", {}),
        "llm_raw_content": llm_raw.get("_llm_raw_content", ""),
        "llm_model": llm_raw.get("_llm_model", ""),
        "llm_base_url": llm_raw.get("_llm_base_url", ""),
        "llm_planner_raw": llm_raw,
        "should_generate": should_generate,
        "skip_generation_threshold_real_images": min_real_images_for_skip,
        "recommended_synthetic_count": recommended,
        "recommended_synthetic_to_real_train_ratio": round(recommended / real_train_count, 3),
        "heuristic_reference": heuristic,
        "base_ratio": heuristic["base_ratio"],
        "difficulty_multiplier": heuristic["difficulty_multiplier"],
        "difficulty_signals": llm_raw.get("difficulty_signals") if planner_type == "llm_dynamic" else heuristic["difficulty_signals"],
        "scarcity_multiplier": heuristic["scarcity_multiplier"],
        "negative_sample_count": sum(item["count"] for item in prompt_plan if item["type"] == "negative"),
        "val_real_only": True,
        "test_real_only": True,
        "synthetic_to_train_only": True,
        "generation_plan": prompt_plan,
        "user_generation_prompt": generation_prompt.strip(),
        "reason": (
            str(llm_raw.get("reason") or "").strip()
            if planner_type == "llm_dynamic"
            else "LLM planner was unavailable or returned invalid output, so heuristic fallback was used."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend synthetic image count and prompt plan for detection training.")
    parser.add_argument("--coco", required=True, help="Real dataset COCO JSON path")
    parser.add_argument("--task", required=True, help="Task description, e.g. bottle detection, helmet detection, smoking detection")
    parser.add_argument("--split-train", type=float, default=0.7, help="Train ratio for real dataset")
    parser.add_argument("--max-synthetic", type=int, default=2000, help="Hard cap for synthetic image count; 0 disables synthetic generation")
    parser.add_argument("--planner", choices=["llm", "heuristic", "fixed"], default="llm", help="Planner used for synthetic image count")
    parser.add_argument("--fixed-synthetic-count", type=int, default=20, help="Synthetic image count used when --planner=fixed")
    parser.add_argument("--llm-config", default="", help="JSON file with base_url, api_key, model, temperature, max_tokens, timeout")
    parser.add_argument("--negative-ratio", type=float, default=0.08, help="Recommended hard-negative synthetic sample ratio")
    parser.add_argument("--generation-prompt", default="", help="User-provided prompt used as the base positive generation prompt")
    parser.add_argument(
        "--min-real-images-for-skip",
        type=int,
        default=1000,
        help="If real image count reaches this threshold, recommend 0 synthetic images by default.",
    )
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    llm_config = _load_json(Path(args.llm_config).resolve()) if args.llm_config else {}
    plan = build_plan(
        coco_path=Path(args.coco).resolve(),
        task_description=args.task,
        split_train=args.split_train,
        max_synthetic=args.max_synthetic,
        negative_ratio=max(0.0, min(args.negative_ratio, 0.3)),
        min_real_images_for_skip=max(1, args.min_real_images_for_skip),
        generation_prompt=args.generation_prompt,
        planner=args.planner,
        fixed_synthetic_count=args.fixed_synthetic_count,
        llm_config=llm_config,
    )
    payload = json.dumps(plan, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
