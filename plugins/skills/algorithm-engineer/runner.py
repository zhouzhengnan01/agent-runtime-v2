from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


SKILL_TITLES = {
    "algorithm-engineer": "完整算法工程师工作台",
    "algorithm-research-scout": "检测算法调研清单",
    "dataset-curator": "数据治理与训练集构建方案",
    "model-candidate-selector": "候选模型选择方案",
    "remote-gpu-ops": "远端 GPU 作业检查方案",
    "gpu-training-orchestrator": "GPU 训练编排方案",
    "cpu-training-runner": "CPU 训练沙盒",
    "detector-evaluator": "检测与业务指标评估方案",
    "deployment-candidate-reviewer": "模型上线候选评审",
    "experiment-ledger": "实验台账记录",
}


def run(skill_name: str, spec: dict[str, Any], paths: Any, artifact_store: Any) -> dict[str, Any]:
    if skill_name == "cpu-training-runner":
        return _cpu_training_local_notice(skill_name, spec, paths, artifact_store)
    title = str(spec.get("title") or SKILL_TITLES.get(skill_name, skill_name))
    payload = _payload(skill_name, spec)
    required_inputs = _required_inputs(skill_name, spec)
    if required_inputs:
        payload["requires_input"] = True
        payload["required_inputs"] = required_inputs
        payload["blocked_stages"] = [item["stage"] for item in required_inputs]
    markdown = _markdown(skill_name, title, spec, payload)
    base = _safe_name(skill_name)
    md_artifact = artifact_store.write_text_artifact(paths, f"{base}.md", markdown)
    json_artifact = artifact_store.write_text_artifact(
        paths,
        f"{base}.json",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
    return {
        "skill_name": skill_name,
        "outputs": [md_artifact, json_artifact],
        "data": {
            "summary": payload["summary"],
            "phase_count": len(payload["phases"]),
            "artifact_names": [md_artifact.name, json_artifact.name],
            "requires_input": bool(required_inputs),
            "required_inputs": required_inputs,
            "blocked_stages": [item["stage"] for item in required_inputs],
        },
    }


def format_reply(skill_name: str, verification: object, run_result: Any) -> str:
    data = getattr(run_result, "data", {}) or {}
    artifacts = getattr(run_result, "outputs", []) or []
    names = [getattr(item, "name", "") for item in artifacts if getattr(item, "name", "")]
    if skill_name == "cpu-training-runner":
        status = str(data.get("status") or "completed")
        execution = str(data.get("execution_mode") or data.get("execution_type") or "")
        if status == "completed" or "best.pt" in names:
            return "\n".join(
                [
                    "CPU 训练沙盒已完成。",
                    f"执行模式：{execution or 'cpu_training'}",
                    f"产物：{', '.join(names)}" if names else "",
                    "如果这是 mock smoke run，best.pt 只用于验证链路，不是可上线模型。",
                ]
            ).strip()
        return "\n".join(
            [
                "CPU 训练沙盒未完成。",
                str(data.get("error") or data.get("message") or "请检查 data.yaml、依赖和 train.log。"),
                f"产物：{', '.join(names)}" if names else "",
            ]
        ).strip()
    sequence = data.get("sequence") if isinstance(data, dict) else None
    if isinstance(sequence, list) and sequence:
        return _algorithm_engineer_sequence_reply(sequence, names)
    return "\n".join(
        [
            f"{SKILL_TITLES.get(skill_name, skill_name)}已生成。",
            data.get("summary", ""),
            _required_inputs_reply(data),
            f"产物：{', '.join(names)}" if names else "",
        ]
    ).strip()


def _required_inputs_reply(data: dict[str, Any]) -> str:
    raw_items = data.get("required_inputs")
    if not isinstance(raw_items, list) or not raw_items:
        return ""
    lines = ["继续真实执行前需要补齐："]
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        lines.append(f"- {item.get('stage')}: {item.get('reason')}")
    return "\n".join(lines)


def _payload(skill_name: str, spec: dict[str, Any]) -> dict[str, Any]:
    base = {
        "skill_name": skill_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objective": spec.get("objective") or "目标检测算法迭代",
        "domain": spec.get("domain") or "computer vision object detection",
        "baseline": spec.get("baseline") or "production YOLO baseline",
        "dataset_path": spec.get("dataset_path") or "",
        "machines": spec.get("machines") or ["AGX Orin", "5090 GPU server"],
        "candidate_algorithms": spec.get("candidate_algorithms") or [],
        "safety_rules": spec.get("safety_rules") or [],
    }
    if skill_name == "algorithm-engineer":
        return _algorithm_engineer_workspace_payload(base, spec)
    phases = _phases(skill_name, base)
    return {
        **base,
        "summary": _summary(skill_name, base),
        "phases": phases,
        "commands": _commands(skill_name, base),
        "metrics": _metrics(skill_name),
        "deliverables": _deliverables(skill_name),
        "risks": _risks(skill_name),
        "next_actions": _next_actions(skill_name),
    }


def _required_inputs(skill_name: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    requirements: list[dict[str, Any]] = []
    dataset = str(spec.get("data_yaml") or spec.get("dataset_path") or "").strip()
    if skill_name in {"algorithm-engineer", "dataset-curator", "gpu-training-orchestrator", "cpu-training-runner"}:
        if not dataset:
            requirements.append(
                _requirement(
                    "dataset-curator",
                    "dataset",
                    ".zip,.tar,.tar.gz,.csv,.json,.jsonl,.parquet,.yaml,.yml",
                    "没有数据集，不能完成数据盘点或 baseline。请提供 YOLO data.yaml、已标注数据集压缩包，或可访问的数据集路径。",
                )
            )
    if skill_name in {"algorithm-engineer", "remote-gpu-ops", "gpu-training-orchestrator"}:
        if not _has_gpu_connection(spec):
            requirements.append(
                _requirement(
                    "remote-gpu-ops",
                    "json",
                    "application/json",
                    "没有 GPU 连接信息，不能完成 AGX/5090 benchmark 或正式训练。请提供 AGX/5090 host、SSH/密钥、训练镜像/环境和数据挂载路径。",
                )
            )
    if skill_name in {"algorithm-engineer", "detector-evaluator", "deployment-candidate-reviewer"}:
        if not _has_training_artifact(spec):
            requirements.append(
                _requirement(
                    "detector-evaluator",
                    "model",
                    ".onnx,.pt,.pth,.bin,.safetensors,.gguf,.pkl,.joblib,.csv",
                    "没有训练产物，不能完成评估。请提供 baseline.pt/best.pt、results.csv 或训练输出目录。",
                )
            )
    if skill_name in {"algorithm-engineer", "deployment-candidate-reviewer"}:
        if not _has_evaluation_result(spec):
            requirements.append(
                _requirement(
                    "deployment-candidate-reviewer",
                    "file",
                    "*/*",
                    "没有评估结果，不能完成上线评审。请提供评估报告、候选模型指标、上线阈值和回滚要求。",
                )
            )
    return requirements


def _requirement(stage: str, input_type: str, accept: str, reason: str) -> dict[str, Any]:
    return {"stage": stage, "type": input_type, "accept": accept, "required": True, "reason": reason}


def _has_gpu_connection(spec: dict[str, Any]) -> bool:
    text = _spec_text(spec).lower()
    markers = ("ssh://", "agx_host", "5090_host", "gpu_host", "nvidia-smi", "cuda", "docker image", "训练机")
    return any(marker in text for marker in markers)


def _has_training_artifact(spec: dict[str, Any]) -> bool:
    text = _spec_text(spec).lower()
    markers = ("best.pt", "baseline.pt", "last.pt", "results.csv", "runs/detect", "训练输出")
    return any(marker in text for marker in markers)


def _has_evaluation_result(spec: dict[str, Any]) -> bool:
    text = _spec_text(spec).lower()
    markers = ("evaluation_report", "评估报告", "map50", "map50-95", "precision", "recall", "上线阈值")
    return any(marker in text for marker in markers)


def _spec_text(spec: dict[str, Any]) -> str:
    return json.dumps(spec, ensure_ascii=False, default=str)


def _summary(skill_name: str, base: dict[str, Any]) -> str:
    if skill_name == "algorithm-engineer":
        return "把算法工程师日常工作收敛成一个工作台入口：任务澄清、数据治理、算法选型、GPU 训练、评估上线和实验台账。"
    if skill_name == "dataset-curator":
        return "把散乱图像和标签整理成可复现 YOLO 训练集，并输出冲突、空标签和类别统计。"
    if skill_name == "gpu-training-orchestrator":
        return "先跑小 benchmark，再按显存和耗时决定正式训练参数，并持续记录日志。"
    if skill_name == "cpu-training-runner":
        return "在本机 CPU 沙盒中启动受限 YOLO 训练任务，并收集 best.pt、last.pt、results.csv 和训练摘要。"
    if skill_name == "detector-evaluator":
        return "同时解释检测指标和业务验收指标，避免只按 mAP 上线。"
    if skill_name == "deployment-candidate-reviewer":
        return "检查候选权重、评估报告、推理服务加载、回滚路径和全量回刷计划。"
    return f"围绕 {base['objective']} 生成结构化算法工程产物。"


def _phases(skill_name: str, base: dict[str, Any]) -> list[dict[str, Any]]:
    common = [
        {"name": "现状盘点", "outputs": ["生产 baseline", "数据集路径", "GPU 可用性", "当前训练/推理服务"]},
        {"name": "数据治理", "outputs": ["去重报告", "标签冲突报告", "YOLO data.yaml", "train/val/test 统计"]},
        {"name": "候选算法", "outputs": ["YOLO baseline", "RT-DETR", "DEIM/D-FINE", "RF-DETR", "DAMO-YOLO"]},
        {"name": "小样本 benchmark", "outputs": ["epochs=1", "fraction=0.1", "显存峰值", "单 epoch 耗时"]},
        {"name": "正式训练", "outputs": ["best.pt", "last.pt", "results.csv", "训练命令快照"]},
        {"name": "业务评估", "outputs": ["mAP 表", "计数误差表", "误检/漏检样例"]},
        {"name": "上线评审", "outputs": ["候选模型", "推荐 conf", "回滚路径", "回刷计划"]},
        {"name": "实验台账", "outputs": ["experiment_id", "参数", "机器", "状态", "结论"]},
    ]
    focused: dict[str, list[dict[str, Any]]] = {
        "algorithm-research-scout": [common[2], {"name": "许可与实现检查", "outputs": ["权重来源", "许可证", "训练脚本成熟度", "ONNX/TensorRT 支持"]}],
        "dataset-curator": [common[1], {"name": "质量审计", "outputs": ["exact hash 去重", "同 sha 多标签冲突", "空标签", "缺标签"]}],
        "model-candidate-selector": [common[0], common[2], {"name": "成本评估", "outputs": ["训练显存", "推理速度", "部署难度", "业务风险"]}],
        "remote-gpu-ops": [common[0], {"name": "远端检查", "outputs": ["nvidia-smi", "df -h", "docker ps", "CUDA/PyTorch/Ultralytics 版本"]}],
        "gpu-training-orchestrator": [common[3], common[4], {"name": "监控", "outputs": ["tail results.csv", "nvidia-smi", "ETA", "best/last 权重"]}],
        "cpu-training-runner": [common[3], common[4], {"name": "产物收集", "outputs": ["best.pt", "last.pt", "results.csv", "args.yaml", "training-summary.md"]}],
        "detector-evaluator": [common[5], {"name": "候选排序", "outputs": ["mAP50-95 best", "mAP50 best", "recall best", "业务误差最低"]}],
        "deployment-candidate-reviewer": [common[5], common[6]],
        "experiment-ledger": [common[7], {"name": "实验对比", "outputs": ["compare_experiments", "summarize_results_csv", "复盘 notes"]}],
    }
    return focused.get(skill_name, common)


def _commands(skill_name: str, base: dict[str, Any]) -> list[str]:
    dataset = base.get("dataset_path") or "<dataset_or_data.yaml>"
    commands = {
        "dataset-curator": [
            f"python scripts/build_yolo_dataset.py --source {dataset} --output /data/training/<dataset-version>",
            "python scripts/audit_yolo_dataset.py --data /data/training/<dataset-version>/data.yaml",
            f"python scripts/find_label_conflicts.py --source {dataset}",
        ],
        "remote-gpu-ops": [
            "nvidia-smi",
            "df -h",
            "docker ps --format 'table {{.Names}}\\t{{.Status}}\\t{{.Ports}}'",
            "rsync -avP --partial --inplace <source> <target>",
        ],
        "gpu-training-orchestrator": [
            "yolo detect train model=rtdetr-l.pt data=data.yaml epochs=1 fraction=0.1 batch=2 imgsz=640 name=rtdetr_l_bench_e1_frac01_b2",
            "yolo detect train model=rtdetr-l.pt data=data.yaml epochs=30 batch=2 imgsz=640 patience=8 amp=False name=rtdetr_l_integrated_dedup_v1_e30_ampoff",
            "tail -f runs/detect/<name>/results.csv",
        ],
        "cpu-training-runner": [
            "yolo detect train model=yolo11n.pt data=<data.yaml> epochs=1 imgsz=320 batch=1 device=cpu workers=0 name=cpu_train_smoke",
        ],
        "detector-evaluator": [
            "yolo detect val model=runs/detect/<name>/weights/best.pt data=data.yaml imgsz=640 conf=0.25",
            "python scripts/compare_business_ab.py --baseline <baseline.pt> --candidate <best.pt> --review-set <review.csv>",
        ],
        "deployment-candidate-reviewer": [
            "python scripts/check_model_package.py --model best.pt --data data.yaml --results results.csv",
            "python scripts/smoke_load_detector_service.py --weights best.pt --conf 0.25",
        ],
        "experiment-ledger": [
            "python scripts/append_experiment_ledger.py --experiment-id <id> --status running",
            "python scripts/summarize_results_csv.py runs/detect/<name>/results.csv",
            "python scripts/compare_experiments.py experiments/*.yaml",
        ],
    }
    return commands.get(skill_name, ["按阶段执行：现状盘点 -> 数据治理 -> benchmark -> 正式训练 -> 业务评估 -> 上线评审 -> 实验台账。"])


def _metrics(skill_name: str) -> list[str]:
    if skill_name in {"detector-evaluator", "deployment-candidate-reviewer", "algorithm-engineer"}:
        return [
            "precision",
            "recall",
            "mAP50",
            "mAP50-95",
            "businessPrecision",
            "businessRecall",
            "absDiffAvg",
            "per-object count error",
        ]
    if skill_name == "dataset-curator":
        return ["unique images", "labeled images", "empty labels", "box count", "class distribution", "conflict groups"]
    if skill_name == "gpu-training-orchestrator":
        return ["epoch time", "GPU memory peak", "GPU utilization", "best mAP50-95", "ETA"]
    if skill_name == "cpu-training-runner":
        return ["best.pt exists", "results.csv exists", "epoch time", "CPU smoke status"]
    return ["feasibility", "risk", "cost", "deployment fit"]


def _deliverables(skill_name: str) -> list[str]:
    mapping = {
        "algorithm-engineer": ["full_cycle_plan.md", "experiment_ledger.yaml", "deployment_review.md"],
        "algorithm-research-scout": ["candidate_algorithm_table.md", "license_checklist.md"],
        "dataset-curator": ["dataset_audit.md", "data.yaml", "conflict_report.json"],
        "model-candidate-selector": ["candidate_model_plan.md"],
        "remote-gpu-ops": ["remote_gpu_snapshot.md", "sync_plan.sh"],
        "gpu-training-orchestrator": ["training_plan.md", "monitoring_checklist.md"],
        "cpu-training-runner": ["best.pt", "last.pt", "results.csv", "args.yaml", "training-summary.md", "train.log"],
        "detector-evaluator": ["evaluation_report.md", "business_ab_table.csv"],
        "deployment-candidate-reviewer": ["deployment_review.md", "rollback_plan.md"],
        "experiment-ledger": ["experiment_ledger.yaml", "experiment_summary.md"],
    }
    return mapping.get(skill_name, ["algorithm_engineering_output.md"])


def _risks(skill_name: str) -> list[str]:
    risks = [
        "外部 GPU / SAM3 / 训练服务不可达时，只能生成计划，不能完成真实训练或标注。",
        "mAP 不等于业务成功，必须保留人工审核集 A/B 和业务验收指标。",
        "长训练任务必须先做小 benchmark，避免占满 GPU 后才发现配置错误。",
    ]
    if skill_name == "deployment-candidate-reviewer":
        risks.append("没有回滚模型和推理服务加载验证时，不建议替换线上模型。")
    return risks


def _next_actions(skill_name: str) -> list[str]:
    if skill_name == "algorithm-engineer":
        return ["确认生产 baseline", "审计当前数据集", "选择 2-3 个候选算法跑 benchmark", "建立实验台账"]
    if skill_name == "dataset-curator":
        return ["确认源数据路径", "运行去重和标签冲突脚本", "生成 YOLO data.yaml", "抽样人工复核"]
    if skill_name == "gpu-training-orchestrator":
        return ["确认 GPU 空闲和磁盘空间", "跑 epochs=1 benchmark", "评估显存和耗时", "启动正式训练"]
    if skill_name == "cpu-training-runner":
        return ["提供 YOLO data.yaml", "确认 CPU smoke 参数", "安装 ultralytics/torch 依赖", "启动 epochs=1 训练并收集 best.pt"]
    if skill_name == "detector-evaluator":
        return ["收集 baseline 和候选权重", "在同一审核集上跑检测和计数评估", "按业务误差排序"]
    return ["补齐输入路径和约束", "生成执行命令", "记录实验结论"]


def _markdown(skill_name: str, title: str, spec: dict[str, Any], payload: dict[str, Any]) -> str:
    if skill_name == "algorithm-engineer":
        return _algorithm_engineer_workspace_markdown(title, spec, payload)
    lines = [
        f"# {title}",
        "",
        f"- Skill: `{skill_name}`",
        f"- Objective: {payload['objective']}",
        f"- Domain: {payload['domain']}",
        f"- Baseline: {payload['baseline']}",
        f"- Dataset: `{payload['dataset_path'] or '待确认'}`",
        f"- Machines: {', '.join(payload['machines'])}",
        "",
        "## Summary",
        "",
        payload["summary"],
        "",
        "## Phases",
        "",
    ]
    for index, phase in enumerate(payload["phases"], start=1):
        lines.append(f"{index}. {phase['name']}")
        for output in phase.get("outputs", []):
            lines.append(f"   - {output}")
    lines.extend(["", "## Candidate Algorithms", ""])
    for item in payload["candidate_algorithms"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Commands", ""])
    for command in payload["commands"]:
        lines.extend(["```bash", command, "```"])
    lines.extend(["", "## Metrics", ""])
    for metric in payload["metrics"]:
        lines.append(f"- {metric}")
    lines.extend(["", "## Safety Rules", ""])
    for rule in payload["safety_rules"]:
        lines.append(f"- {rule}")
    lines.extend(["", "## Risks", ""])
    for risk in payload["risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", "## Next Actions", ""])
    for action in payload["next_actions"]:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-") or "algorithm-output"


def _algorithm_engineer_sequence_reply(sequence: list[object], artifact_names: list[str]) -> str:
    completed = [str(item.get("skill_name") or "") for item in sequence if isinstance(item, dict) and item.get("skill_name")]
    completed_set = set(completed)
    stage_lines = [
        _stage_status("任务澄清", "algorithm-engineer" in completed_set),
        _stage_status("数据治理", "dataset-curator" in completed_set),
        _stage_status("算法调研", "algorithm-research-scout" in completed_set and "model-candidate-selector" in completed_set),
        _stage_status("训练编排", "remote-gpu-ops" in completed_set and "gpu-training-orchestrator" in completed_set),
        _stage_status("评估上线", "detector-evaluator" in completed_set and "deployment-candidate-reviewer" in completed_set),
        _stage_status("实验台账", "experiment-ledger" in completed_set),
    ]
    key_artifacts = [
        name
        for name in artifact_names
        if name
        in {
            "algorithm-engineer.md",
            "dataset-curator.md",
            "algorithm-research-scout.md",
            "model-candidate-selector.md",
            "gpu-training-orchestrator.md",
            "detector-evaluator.md",
            "deployment-candidate-reviewer.md",
            "experiment-ledger.md",
        }
    ]
    lines = [
        "算法工程师全流程工作台已搭好。",
        "",
        "这不是单纯生成文档，而是把一次视觉模型迭代拆成可执行的工程闭环：需求澄清、数据治理、候选算法、AGX/5090 训练、检测/分割/计数评估、上线评审、实验台账。",
        "",
        "当前看板：",
        *stage_lines,
        "",
        "当前还没有启动真实训练，因为缺少真实训练输入：",
        "- YOLO 数据集或 `data.yaml`",
        "- 当前生产 baseline 权重和评估集",
        "- 训练目标机器：AGX、5090，或先用 CPU 沙盒 smoke run",
        "- 验收指标：mAP、recall、业务误差、推理耗时、上线阈值",
        "",
        f"已生成 {len(artifact_names)} 个工程产物，关键产物：{', '.join(key_artifacts) if key_artifacts else '已写入右侧文件面板'}。",
        "",
        "下一步建议直接选一个动作：",
        "1. `数据治理`：给我图片/标签目录或 data.yaml，我先审计数据集。",
        "2. `CPU 训练沙盒`：先用 `mock=true` 或一个小 data.yaml 验证 best.pt 闭环。",
        "3. `5090 训练编排`：确认远端机器和数据路径后，生成 benchmark 与正式训练命令。",
        "4. `评估上线`：给 baseline.pt、best.pt 和 review set，输出上线/继续训练建议。",
    ]
    return "\n".join(lines).strip()


def _stage_status(label: str, done: bool) -> str:
    return f"- {label}：{'已建立' if done else '待补齐'}"


def _cpu_training_local_notice(skill_name: str, spec: dict[str, Any], paths: Any, artifact_store: Any) -> dict[str, Any]:
    content = "\n".join(
        [
            "# CPU 训练沙盒需要沙盒执行",
            "",
            "当前请求没有进入 sandbox/local_subprocess 执行路径，因此没有启动真实训练。",
            "",
            "请确认：",
            "",
            "- `SANDBOX_PROVIDER=local_subprocess`",
            "- `SANDBOX_EXECUTOR_ENABLED=true`",
            "- `cpu-training-runner` 的 sandbox profile 为 `python-skill`",
            "- 已提供 `data_yaml`，或者仅 smoke test 时设置 `mock=true`",
            "",
        ]
    )
    artifact = artifact_store.write_text_artifact(paths, "cpu-training-runner-not-started.md", content)
    return {
        "skill_name": skill_name,
        "outputs": [artifact],
        "data": {
            "status": "not_started",
            "execution_mode": "local_notice",
            "message": "cpu-training-runner must run through sandbox/local_subprocess to start training.",
            "spec": spec,
        },
    }


def _algorithm_engineer_workspace_payload(base: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    stages = [
        {
            "id": "intake",
            "name": "任务澄清",
            "owner": "algorithm-engineer",
            "goal": "把业务目标翻译成可验收的算法任务。",
            "inputs": ["目标类型", "当前 baseline", "数据路径", "部署目标", "成功指标"],
            "outputs": ["objective.md", "acceptance_metrics.yaml"],
            "ready_gate": "明确业务目标和至少一个可比较 baseline。",
            "done_gate": "指标、数据、机器和交付物都已确认。",
        },
        {
            "id": "dataset",
            "name": "数据治理",
            "owner": "dataset-curator",
            "goal": "把散乱图片和标签整理成可复现训练集。",
            "inputs": ["原始图片库", "manual/training/review 标签", "类别表"],
            "outputs": ["dataset_audit.md", "conflict_report.json", "data.yaml"],
            "ready_gate": "源数据可访问，标签优先级已确认。",
            "done_gate": "train/val/test、空标签、冲突和类别分布统计完成。",
        },
        {
            "id": "research",
            "name": "算法调研与候选筛选",
            "owner": "algorithm-research-scout + model-candidate-selector",
            "goal": "只保留 2-3 个值得 benchmark 的模型。",
            "inputs": ["baseline 指标", "小目标/遮挡/密集程度", "AGX/5090 约束"],
            "outputs": ["candidate_algorithm_table.md", "benchmark_shortlist.yaml"],
            "ready_gate": "业务约束和部署目标已明确。",
            "done_gate": "每个候选都有权重来源、许可证、训练成本和部署风险。",
        },
        {
            "id": "gpu",
            "name": "GPU 环境与训练编排",
            "owner": "remote-gpu-ops + gpu-training-orchestrator",
            "goal": "先小样本 benchmark，再正式训练，保留完整日志。",
            "inputs": ["目标机器", "数据集路径", "候选模型", "训练参数"],
            "outputs": ["remote_gpu_snapshot.md", "training_plan.md", "results.csv"],
            "ready_gate": "GPU、磁盘、CUDA/PyTorch、容器和数据同步检查通过。",
            "done_gate": "benchmark 通过，正式训练有 best.pt/last.pt/results.csv。",
        },
        {
            "id": "evaluate",
            "name": "检测指标与业务评估",
            "owner": "detector-evaluator",
            "goal": "同时看 mAP 和业务计数误差，避免只按论文指标上线。",
            "inputs": ["baseline 权重", "候选权重", "review set", "推荐 conf"],
            "outputs": ["evaluation_report.md", "business_ab_table.csv", "error_examples/"],
            "ready_gate": "baseline 和候选模型在同一审核集上可评估。",
            "done_gate": "给出上线/继续训练/补数据的明确建议。",
        },
        {
            "id": "release",
            "name": "上线评审与回刷计划",
            "owner": "deployment-candidate-reviewer",
            "goal": "让模型替换、回滚和全量回刷都有可执行路径。",
            "inputs": ["候选权重", "评估报告", "推理服务", "回滚模型"],
            "outputs": ["deployment_review.md", "rollback_plan.md", "reprocess_plan.md"],
            "ready_gate": "模型包完整，服务加载 smoke test 通过。",
            "done_gate": "上线风险、回滚步骤和全量回刷计划评审完成。",
        },
        {
            "id": "ledger",
            "name": "实验台账与复盘",
            "owner": "experiment-ledger",
            "goal": "让每次算法迭代可追溯、可对比、可复现。",
            "inputs": ["实验参数", "机器信息", "数据版本", "指标结果", "结论"],
            "outputs": ["experiment_ledger.yaml", "iteration_summary.md"],
            "ready_gate": "每次训练都有唯一 experiment_id。",
            "done_gate": "结果、失败原因和下一轮动作已沉淀。",
        },
    ]
    permissions = [
        {"action": "只读检查", "examples": ["nvidia-smi", "df -h", "docker ps"], "approval": "可自动执行"},
        {"action": "数据同步", "examples": ["rsync 数据集到 GPU 机器"], "approval": "需要用户确认目标路径"},
        {"action": "benchmark", "examples": ["epochs=1 fraction=0.1"], "approval": "需要确认占用 GPU"},
        {"action": "正式训练", "examples": ["epochs>=30", "长时间训练"], "approval": "必须二次确认"},
        {"action": "上线替换", "examples": ["替换 best.pt", "重启推理服务"], "approval": "必须人工审批并准备回滚"},
    ]
    app_routes = [
        {"intent": "我有新数据", "route_to": "数据治理", "skill": "dataset-curator"},
        {"intent": "我想找比 YOLO 更好的算法", "route_to": "算法调研与候选筛选", "skill": "algorithm-research-scout"},
        {"intent": "我要开始训练", "route_to": "GPU 环境与训练编排", "skill": "gpu-training-orchestrator"},
        {"intent": "我有 best.pt 想看能不能上线", "route_to": "检测指标与业务评估", "skill": "detector-evaluator"},
        {"intent": "我要复盘多次实验", "route_to": "实验台账与复盘", "skill": "experiment-ledger"},
    ]
    artifact_tree = {
        "00_intake": ["objective.md", "acceptance_metrics.yaml"],
        "10_dataset": ["dataset_audit.md", "data.yaml", "conflict_report.json"],
        "20_research": ["candidate_algorithm_table.md", "benchmark_shortlist.yaml"],
        "30_training": ["remote_gpu_snapshot.md", "training_plan.md", "results.csv"],
        "40_evaluation": ["evaluation_report.md", "business_ab_table.csv"],
        "50_release": ["deployment_review.md", "rollback_plan.md", "reprocess_plan.md"],
        "90_ledger": ["experiment_ledger.yaml", "iteration_summary.md"],
    }
    dashboard = {
        "objective": base["objective"],
        "domain": base["domain"],
        "baseline": base["baseline"],
        "dataset_path": base["dataset_path"],
        "machines": base["machines"],
        "candidate_algorithms": base["candidate_algorithms"],
        "current_status": "ready_for_planning",
        "recommended_first_action": "先完成任务澄清和数据治理，只读检查 GPU 资源。",
    }
    return {
        **base,
        "summary": _summary("algorithm-engineer", base),
        "dashboard": dashboard,
        "stages": stages,
        "phases": [{"name": item["name"], "outputs": item["outputs"]} for item in stages],
        "app_routes": app_routes,
        "permission_gates": permissions,
        "artifact_tree": artifact_tree,
        "commands": _app_commands(base),
        "metrics": [
            "mAP50",
            "mAP50-95",
            "precision",
            "recall",
            "businessPrecision",
            "businessRecall",
            "absDiffAvg",
            "per-object count error",
            "GPU memory peak",
            "epoch time",
        ],
        "deliverables": ["algorithm-engineer.md", "algorithm-engineer.json", "experiment_ledger.yaml", "deployment_review.md"],
        "risks": _risks("algorithm-engineer"),
        "next_actions": ["确认 baseline 和数据路径", "运行数据治理审计", "只读检查 GPU 机器", "选择 2-3 个候选模型跑 benchmark", "把结果写入实验台账"],
    }


def _app_commands(base: dict[str, Any]) -> list[str]:
    dataset = base.get("dataset_path") or "<dataset_or_data.yaml>"
    return [
        f"python scripts/audit_dataset_entrypoint.py --source {dataset}",
        "nvidia-smi && df -h && docker ps --format 'table {{.Names}}\\t{{.Status}}\\t{{.Ports}}'",
        "rsync -avP --partial --inplace <dataset> <gpu-host>:/data/training/<dataset-version>/",
        "yolo detect train model=rtdetr-l.pt data=data.yaml epochs=1 fraction=0.1 batch=2 imgsz=640 name=bench_rtdetr_l_e1_frac01",
        "yolo detect train model=rtdetr-l.pt data=data.yaml epochs=30 batch=2 imgsz=640 patience=8 amp=False name=formal_rtdetr_l_e30",
        "python scripts/compare_business_ab.py --baseline <baseline.pt> --candidate <best.pt> --review-set <review.csv>",
        "python scripts/append_experiment_ledger.py --experiment-id <id> --status completed --results runs/detect/<name>/results.csv",
    ]


def _algorithm_engineer_workspace_markdown(title: str, spec: dict[str, Any], payload: dict[str, Any]) -> str:
    dashboard = payload["dashboard"]
    lines = [
        f"# {title}",
        "",
        "## 工作台概览",
        "",
        f"- 目标: {dashboard['objective']}",
        f"- 领域: {dashboard['domain']}",
        f"- Baseline: {dashboard['baseline']}",
        f"- 数据路径: `{dashboard['dataset_path'] or '待确认'}`",
        f"- 机器: {', '.join(dashboard['machines'])}",
        f"- 候选算法: {', '.join(dashboard['candidate_algorithms'])}",
        f"- 当前状态: {dashboard['current_status']}",
        f"- 建议第一步: {dashboard['recommended_first_action']}",
        "",
        "## 阶段看板",
        "",
        "| 阶段 | Owner Skill | Ready Gate | Done Gate | 产物 |",
        "|---|---|---|---|---|",
    ]
    for stage in payload["stages"]:
        lines.append(
            f"| {stage['name']} | `{stage['owner']}` | {stage['ready_gate']} | {stage['done_gate']} | {', '.join(stage['outputs'])} |"
        )
    lines.extend(["", "## 子流程路由", ""])
    for route in payload["app_routes"]:
        lines.append(f"- {route['intent']} -> {route['route_to']} (`{route['skill']}`)")
    lines.extend(["", "## 权限闸门", ""])
    lines.extend(["| 动作 | 示例 | 审批规则 |", "|---|---|---|"])
    for gate in payload["permission_gates"]:
        lines.append(f"| {gate['action']} | {', '.join(gate['examples'])} | {gate['approval']} |")
    lines.extend(["", "## 产物目录", ""])
    for folder, files in payload["artifact_tree"].items():
        lines.append(f"- `{folder}/`: {', '.join(files)}")
    lines.extend(["", "## 执行命令草案", ""])
    for command in payload["commands"]:
        lines.extend(["```bash", command, "```"])
    lines.extend(["", "## 验收指标", ""])
    for metric in payload["metrics"]:
        lines.append(f"- {metric}")
    lines.extend(["", "## 风险", ""])
    for risk in payload["risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", "## 下一步动作", ""])
    for action in payload["next_actions"]:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"
