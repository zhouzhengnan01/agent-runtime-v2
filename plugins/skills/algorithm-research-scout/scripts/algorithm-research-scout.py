#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""algorithm-research-scout: online research verification for object detection algorithms.

This script performs research verification only:
- paper search through Semantic Scholar and arXiv
- GitHub repository discovery
- license check through GitHub API
- pretrained weight existence check through releases and README
- public mAP/FPS metric extraction from README/abstract text
- suitability scoring for dense, small-object, or domain-specific detection tasks

It does NOT run local ONNX/TensorRT/BModel compilation.

The script is designed to degrade gracefully: if network/API calls fail, it still
generates a Markdown report with "待核验" fields instead of fabricating facts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import requests
except Exception as exc:  # pragma: no cover
    print("ERROR: requests is required. Install with: pip install -r requirements.txt", file=sys.stderr)
    raise


WEIGHT_EXTENSIONS = (
    ".pt", ".pth", ".onnx", ".safetensors", ".ckpt", ".pdparams", ".weights", ".engine", ".bin"
)

RISK_KEYWORDS = [
    "non-commercial", "noncommercial", "research only", "academic use", "not for commercial",
    "commercial use is prohibited", "cc-by-nc", "agpl", "gpl"
]

TRAINING_KEYWORDS = ["train.py", "training", "train", "custom dataset", "coco", "dataset", "fine-tune", "finetune"]
EXPORT_KEYWORDS = ["onnx", "tensorrt", "openvino", "export", "deploy", "inference"]
SMALL_DENSE_KEYWORDS = [
    "small object", "dense", "occlusion", "occluded", "fruit", "palm", "oil palm",
    "agriculture", "agricultural", "tiny object", "crowded"
]

DETECTION_KEYWORDS = [
    "object detection", "detector", "detection", "real-time detection", "real time detection",
    "target detection", "instance detection", "dense detection"
]

TASK_SYNONYMS = {
    "棕榈果": ["palm fruit", "oil palm fruit", "fresh fruit bunch", "ffb", "palm oil"],
    "密集": ["dense", "crowded", "overlap", "occlusion", "occluded"],
    "小目标": ["small object", "tiny object", "small target"],
    "遮挡": ["occlusion", "occluded", "overlap"],
    "边缘": ["edge", "embedded", "real-time", "real time", "latency", "fps"],
    "农业": ["agriculture", "agricultural", "fruit", "orchard", "crop"],
}

PAPER_NEGATIVE_KEYWORDS = [
    "autonomous driving", "trajectory", "reinforcement learning", "language model", "llm",
    "communication", "wireless", "medical", "medical image", "retinopathy", "segmentation only", "classification",
    "semantic segmentation", "3d object detection", "lidar", "point cloud"
]

REPO_NOISE_KEYWORDS = [
    "awesome", "paper-list", "survey", "comparison", "benchmark-only", "tutorial",
    "dotnet", "android", "ros", "collection", "list", "model_zoo", "model-zoo",
    "transformers.js", "script"
]

OFFICIAL_HINTS = [
    "official", "official implementation", "model zoo", "weights", "pretrained",
    "training", "train.py", "custom dataset", "coco", "export", "onnx"
]

METRIC_PATTERNS = {
    "mAP": re.compile(r"(?i)\b(?:mAP|AP)\s*(?:@?\s*0\.5:0\.95|50:95|box)?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)"),
    "AP50": re.compile(r"(?i)\b(?:AP50|mAP50|AP@0\.5|mAP@0\.5)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)"),
    "FPS": re.compile(r"(?i)\b([0-9]+(?:\.[0-9]+)?)\s*(?:FPS|frames/s|frames per second)\b"),
    "latency": re.compile(r"(?i)\b([0-9]+(?:\.[0-9]+)?)\s*(?:ms|milliseconds)\b"),
    "params": re.compile(r"(?i)\b([0-9]+(?:\.[0-9]+)?)\s*(?:M|million)?\s*(?:params|parameters)\b"),
    "flops": re.compile(r"(?i)\b([0-9]+(?:\.[0-9]+)?)\s*(?:G|GFLOPs|FLOPs)\b"),
}


DEFAULT_CANDIDATES = [
    {
        "name": "YOLO family",
        "aliases": ["YOLOv8", "YOLOv9", "YOLOv10", "YOLO11", "YOLO12", "Ultralytics YOLO"],
        "official_repos": ["ultralytics/ultralytics"],
        "priority_hint": "baseline",
        "why": "工程链路成熟，适合作为强 baseline 和部署可行性参考。",
        "risk": "不同分支 license、后处理、导出和部署支持差异较大，需要逐项核验。",
    },
    {
        "name": "RT-DETR family",
        "aliases": ["RT-DETR", "RT-DETRv2", "Real-Time DETR"],
        "official_repos": ["lyuwenyu/RT-DETR", "PaddlePaddle/PaddleDetection"],
        "priority_hint": "p0",
        "why": "实时端到端检测候选，适合与 YOLO 对比召回、定位和密集场景表现。",
        "risk": "DETR/Transformer 结构的部署适配和延迟需要重点核验。",
    },
    {
        "name": "D-FINE / DEIM family",
        "aliases": ["D-FINE", "DEIM", "D-FINE object detection", "DEIM object detection"],
        "official_repos": ["Peterande/D-FINE", "ShihuaHuang95/DEIM"],
        "priority_hint": "p1",
        "why": "实时 DETR 改进方向，适合进入新算法调研候选池。",
        "risk": "工程成熟度、权重、训练文档和部署案例需要核验。",
    },
    {
        "name": "RF-DETR family",
        "aliases": ["RF-DETR", "Roboflow RF-DETR"],
        "official_repos": ["roboflow/rf-detr"],
        "priority_hint": "p1",
        "why": "DETR 类新候选，可用于判断是否存在比 YOLO 更适合特定数据集的替代方向。",
        "risk": "需核验训练支持、权重 license、导出支持和实际维护状态。",
    },
    {
        "name": "DAMO-YOLO family",
        "aliases": ["DAMO-YOLO", "DAMO YOLO"],
        "official_repos": ["tinyvision/DAMO-YOLO"],
        "priority_hint": "p2",
        "why": "YOLO 外的工业检测候选，可作为补充对比。",
        "risk": "维护状态、版本兼容、权重可获取性和部署链路可能不如主流 YOLO。",
    },
]


@dataclass
class RepoEvidence:
    query: str
    full_name: str = "未找到"
    html_url: str = "未找到"
    stars: int | None = None
    forks: int | None = None
    pushed_at: str = "未核验"
    license_spdx: str = "未核验"
    license_name: str = "未核验"
    readme_found: bool = False
    release_count: int = 0
    weight_assets: list[str] = field(default_factory=list)
    weight_links_in_readme: list[str] = field(default_factory=list)
    training_support: str = "未核验"
    export_support: str = "未核验"
    commercial_risk: str = "未核验"
    risk_reason: str = "未核验"
    readme_excerpt: str = ""
    relevance_score: int = 0
    relevance_reasons: list[str] = field(default_factory=list)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def http_get(url: str, headers: dict[str, str] | None = None, timeout: int = 20) -> requests.Response | None:
    try:
        response = requests.get(url, headers=headers or {}, timeout=timeout)
        if response.status_code == 403:
            time.sleep(1)
        if response.status_code >= 400:
            return None
        return response
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research candidate object detection algorithms.")
    parser.add_argument("--input-json", default=None, help="JSON string or path to JSON file.")
    parser.add_argument("--task-type", default="目标检测", help="Task type.")
    parser.add_argument("--baseline", default="YOLO", help="Current baseline.")
    parser.add_argument("--constraints", nargs="*", default=None, help="Constraints.")
    parser.add_argument("--mode", choices=["offline-template", "online-research"], default="online-research")
    parser.add_argument("--max-results", type=int, default=6, help="Max results per source.")
    parser.add_argument("--output", default=None, help="Markdown output path.")
    parser.add_argument("--json-output-dir", default=None, help="Directory to save JSON evidence artifacts.")
    return parser.parse_args()


def load_input_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    p = Path(value)
    if p.exists() and p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(value)


def normalize_constraints(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        parts = re.split(r"[,，;；]", value)
        return [p.strip() for p in parts if p.strip()]
    return [str(value)]


def merge_inputs(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_input_json(args.input_json)
    data = dict(payload)
    if args.task_type:
        data["task_type"] = args.task_type
    if args.baseline:
        data["baseline"] = args.baseline
    if args.constraints is not None:
        data["constraints"] = args.constraints
    data.setdefault("skill_name", "algorithm-research-scout")
    data.setdefault("mode", args.mode)
    data.setdefault("max_results", args.max_results)
    data["constraints"] = normalize_constraints(data.get("constraints"))
    return data


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").lower()).strip()


def expand_task_terms(task_type: str, constraints: list[str]) -> list[str]:
    terms = [task_type, *constraints]
    combined = " ".join(terms)
    for key, values in TASK_SYNONYMS.items():
        if key in combined:
            terms.extend(values)
    terms.extend(SMALL_DENSE_KEYWORDS)
    seen = set()
    out = []
    for term in terms:
        norm = normalize_text(str(term))
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def candidate_alias_terms() -> list[str]:
    terms = []
    for cand in DEFAULT_CANDIDATES:
        terms.append(cand["name"])
        terms.extend(cand.get("aliases", []))
    return [normalize_text(t) for t in terms if normalize_text(t)]


def paper_identity(paper: dict[str, Any]) -> str:
    title = normalize_text(paper.get("title") or "")
    external = paper.get("externalIds") or {}
    for key in ["DOI", "ArXiv", "CorpusId"]:
        if external.get(key):
            return f"{key.lower()}:{external[key]}"
    return title


def paper_relevance_score(paper: dict[str, Any], task_type: str, constraints: list[str], query: str) -> tuple[int, list[str]]:
    title = normalize_text(paper.get("title") or "")
    abstract = normalize_text(paper.get("abstract") or "")
    text = f"{title} {abstract}"
    query_text = normalize_text(query)
    score = 0
    reasons: list[str] = []

    if any(kw in text for kw in DETECTION_KEYWORDS):
        score += 30
        reasons.append("检测任务相关")
    if any(alias in text for alias in candidate_alias_terms()):
        score += 22
        reasons.append("命中候选算法名")

    task_hits = [term for term in expand_task_terms(task_type, constraints) if term in text]
    if task_hits:
        score += min(24, len(task_hits) * 4)
        reasons.append("命中任务/约束：" + ", ".join(task_hits[:4]))

    query_hits = [part for part in re.split(r"\W+", query_text) if len(part) >= 4 and part in text]
    if query_hits:
        score += min(12, len(set(query_hits)) * 2)

    year = str(paper.get("year") or "")
    if year.isdigit() and int(year) >= 2020:
        score += 6
        reasons.append("近年论文")
    citations = paper.get("citationCount")
    if isinstance(citations, int) and citations >= 20:
        score += 4
        reasons.append("有引用基础")

    negatives = [kw for kw in PAPER_NEGATIVE_KEYWORDS if kw in text]
    if negatives:
        score -= 18
        if not any(kw in text for kw in ["fruit", "palm", "agricultur", "yolo", "detr"]):
            score -= 10
        reasons.append("疑似无关领域：" + ", ".join(negatives[:3]))

    return max(0, min(100, score)), reasons[:6]


def semantic_scholar_search(query: str, limit: int = 6) -> list[dict[str, Any]]:
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    headers = {"User-Agent": "algorithm-research-scout/0.2"}
    if api_key:
        headers["x-api-key"] = api_key
    fields = "title,year,abstract,url,venue,citationCount,authors,externalIds,openAccessPdf"
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode({
        "query": query,
        "limit": limit,
        "fields": fields,
    })
    response = http_get(url, headers=headers)
    if not response:
        return []
    try:
        data = response.json()
        return data.get("data", []) or []
    except Exception:
        return []


def arxiv_search(query: str, limit: int = 6) -> list[dict[str, Any]]:
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    response = http_get(url, headers={"User-Agent": "algorithm-research-scout/0.2"})
    if not response:
        return []
    try:
        root = ET.fromstring(response.text)
    except Exception:
        return []
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").replace("\n", " ").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").replace("\n", " ").strip()
        published = entry.findtext("atom:published", default="", namespaces=ns) or ""
        link = entry.findtext("atom:id", default="", namespaces=ns) or ""
        authors = [a.findtext("atom:name", default="", namespaces=ns) for a in entry.findall("atom:author", ns)]
        papers.append({
            "title": title,
            "year": published[:4] if published else "",
            "abstract": summary,
            "url": link,
            "venue": "arXiv",
            "citationCount": None,
            "authors": [{"name": a} for a in authors if a],
            "source": "arXiv",
        })
    return papers


def build_queries(task_type: str, constraints: list[str]) -> list[str]:
    extra = " ".join(constraints)
    expanded = " ".join(expand_task_terms(task_type, constraints)[:8])
    base_queries = [
        f"{task_type} object detection {extra}",
        f"{expanded} object detection",
        f"{task_type} dense object detection",
        f"{task_type} small object detection",
        "oil palm fruit detection object detection",
        "palm fruit dense detection",
        "occluded small object detection fruit agriculture",
    ]
    for cand in DEFAULT_CANDIDATES:
        for alias in cand["aliases"][:2]:
            base_queries.append(f"{alias} {task_type} object detection")
    # Deduplicate while preserving order.
    seen = set()
    out = []
    for q in base_queries:
        qn = q.strip()
        if qn and qn.lower() not in seen:
            seen.add(qn.lower())
            out.append(qn)
    return out


def collect_papers(task_type: str, constraints: list[str], max_results: int, online: bool) -> list[dict[str, Any]]:
    if not online:
        return []
    papers: list[dict[str, Any]] = []
    seen = set()
    seen_titles = set()
    for query in build_queries(task_type, constraints)[:10]:
        for p in semantic_scholar_search(query, limit=max_results * 2):
            key = paper_identity(p)
            title_key = normalize_text(p.get("title") or "")
            if key and key not in seen and title_key not in seen_titles:
                relevance, reasons = paper_relevance_score(p, task_type, constraints, query)
                if relevance < 35:
                    continue
                seen.add(key)
                seen_titles.add(title_key)
                p["source"] = "Semantic Scholar"
                p["query"] = query
                p["relevance_score"] = relevance
                p["relevance_reasons"] = reasons
                papers.append(p)
        for p in arxiv_search(query, limit=min(max_results * 2, 8)):
            key = paper_identity(p)
            title_key = normalize_text(p.get("title") or "")
            if key and key not in seen and title_key not in seen_titles:
                relevance, reasons = paper_relevance_score(p, task_type, constraints, query)
                if relevance < 40:
                    continue
                seen.add(key)
                seen_titles.add(title_key)
                p["query"] = query
                p["relevance_score"] = relevance
                p["relevance_reasons"] = reasons
                papers.append(p)
        time.sleep(0.2)
    papers.sort(
        key=lambda p: (
            int(p.get("relevance_score") or 0),
            int(p.get("citationCount") or 0),
            int(p.get("year") or 0) if str(p.get("year") or "").isdigit() else 0,
        ),
        reverse=True,
    )
    return papers[: max_results * 8]


def github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "algorithm-research-scout/0.2",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def repo_relevance_score(repo: dict[str, Any], readme: str, alias: str, cand: dict[str, Any]) -> tuple[int, list[str]]:
    full_name = normalize_text(repo.get("full_name") or "")
    description = normalize_text(repo.get("description") or "")
    topics = " ".join(repo.get("topics") or [])
    readme_low = normalize_text(readme[:50000])
    text = f"{full_name} {description} {topics} {readme_low}"
    alias_terms = [normalize_text(alias), *[normalize_text(a) for a in cand.get("aliases", [])]]
    alias_terms = [a for a in alias_terms if a]

    score = 0
    reasons: list[str] = []
    official_repos = [normalize_text(r) for r in cand.get("official_repos", [])]
    if full_name in official_repos:
        score += 60
        reasons.append("命中预置官方/主仓库")
    if any(a in full_name for a in alias_terms):
        score += 30
        reasons.append("仓库名命中算法别名")
    elif any(a in text for a in alias_terms):
        score += 20
        reasons.append("README/描述命中算法别名")

    if any(kw in text for kw in DETECTION_KEYWORDS):
        score += 15
        reasons.append("检测任务相关")
    if any(h in text for h in OFFICIAL_HINTS):
        score += 16
        reasons.append("存在官方性/模型库/训练部署线索")
    if repo.get("stargazers_count", 0) >= 100:
        score += 8
        reasons.append("仓库关注度较高")
    if repo.get("pushed_at"):
        score += 4
        reasons.append("可获取维护时间")
    if any(ext in readme_low for ext in WEIGHT_EXTENSIONS) or "huggingface.co" in readme_low:
        score += 10
        reasons.append("README 有权重线索")

    noise = [kw for kw in REPO_NOISE_KEYWORDS if kw in full_name or kw in description]
    if noise and full_name not in official_repos:
        score -= 22
        reasons.append("疑似列表/封装/非主实现：" + ", ".join(noise[:3]))

    return max(0, min(100, score)), reasons[:6]


def cheap_repo_score(repo: dict[str, Any], alias: str, cand: dict[str, Any]) -> int:
    full_name = normalize_text(repo.get("full_name") or "")
    description = normalize_text(repo.get("description") or "")
    text = f"{full_name} {description}"
    alias_terms = [normalize_text(alias), *[normalize_text(a) for a in cand.get("aliases", [])]]
    official_repos = [normalize_text(r) for r in cand.get("official_repos", [])]
    score = 0
    if full_name in official_repos:
        score += 80
    if any(a and a in full_name for a in alias_terms):
        score += 35
    elif any(a and a in text for a in alias_terms):
        score += 20
    if any(kw in text for kw in DETECTION_KEYWORDS):
        score += 10
    if repo.get("stargazers_count", 0) >= 100:
        score += 8
    if any(kw in text for kw in REPO_NOISE_KEYWORDS) and full_name not in official_repos:
        score -= 25
    return score


def github_search_repos(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    q = f"{query} object detection in:name,description,readme"
    url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode({
        "q": q,
        "sort": "stars",
        "order": "desc",
        "per_page": max_results,
    })
    response = http_get(url, headers=github_headers())
    if not response:
        return []
    try:
        return response.json().get("items", []) or []
    except Exception:
        return []


def github_api(path: str) -> dict[str, Any] | list[Any] | None:
    url = "https://api.github.com" + path
    response = http_get(url, headers=github_headers())
    if not response:
        return None
    try:
        return response.json()
    except Exception:
        return None


def github_get_repo(full_name: str) -> dict[str, Any] | None:
    data = github_api(f"/repos/{full_name}")
    return data if isinstance(data, dict) and data.get("full_name") else None


def official_repo_fallback(full_name: str) -> dict[str, Any]:
    return {
        "full_name": full_name,
        "html_url": f"https://github.com/{full_name}",
        "stargazers_count": 0,
        "forks_count": 0,
        "pushed_at": "API限流/待核验",
        "default_branch": "main",
        "license": None,
        "description": "预置官方/主仓库，GitHub API 不可用时使用 URL 回退核验",
    }


def github_raw_readme(full_name: str, default_branch: str = "main") -> str:
    # Try API first to support repos with README extensions.
    data = github_api(f"/repos/{full_name}/readme")
    if isinstance(data, dict) and data.get("download_url"):
        response = http_get(data["download_url"], headers={"User-Agent": "algorithm-research-scout/0.2"})
        if response:
            return response.text[:200000]
    for branch in [default_branch, "main", "master"]:
        for name in ["README.md", "README.rst", "readme.md"]:
            raw = f"https://raw.githubusercontent.com/{full_name}/{branch}/{name}"
            response = http_get(raw, headers={"User-Agent": "algorithm-research-scout/0.2"})
            if response:
                return response.text[:200000]
    return ""


def find_weight_links(text: str) -> list[str]:
    if not text:
        return []
    found = set()
    # Markdown links or direct URLs containing known model extensions/services.
    url_pattern = re.compile(r"https?://[^\s\]\)\"']+")
    for url in url_pattern.findall(text):
        low = url.lower()
        if any(ext in low for ext in WEIGHT_EXTENSIONS) or any(s in low for s in ["huggingface.co", "drive.google.com", "modelscope.cn", "pan.baidu.com", "onedrive.live.com"]):
            found.add(url.rstrip(".,)"))
    return sorted(found)[:20]


def infer_support(text: str, keywords: list[str]) -> str:
    if not text:
        return "未核验"
    low = text.lower()
    hits = [kw for kw in keywords if kw.lower() in low]
    if len(hits) >= 2:
        return "有迹象支持：" + ", ".join(hits[:5])
    if hits:
        return "可能支持：" + ", ".join(hits[:3])
    return "未发现明确证据"


def license_risk(license_spdx: str, license_name: str, readme: str) -> tuple[str, str]:
    text = " ".join([license_spdx or "", license_name or "", readme[:50000] or ""]).lower()
    if not license_spdx or license_spdx in ["NOASSERTION", "未核验"]:
        base = "中"
        reason = "未识别明确代码 license"
    else:
        base = "低"
        reason = f"识别到代码 license: {license_spdx}"
    for kw in RISK_KEYWORDS:
        if kw in text:
            if kw in ["agpl", "gpl"] and "lgpl" not in text:
                return "中/高", f"发现潜在强 copyleft 或限制关键词：{kw}"
            return "高", f"发现潜在非商用/限制关键词：{kw}"
    return base, reason


def inspect_repo(repo: dict[str, Any], query: str, cand: dict[str, Any]) -> RepoEvidence:
    full_name = repo.get("full_name", "未找到")
    default_branch = repo.get("default_branch", "main")
    ev = RepoEvidence(
        query=query,
        full_name=full_name,
        html_url=repo.get("html_url", "未找到"),
        stars=repo.get("stargazers_count"),
        forks=repo.get("forks_count"),
        pushed_at=repo.get("pushed_at", "未核验"),
        license_spdx=(repo.get("license") or {}).get("spdx_id") or "未核验",
        license_name=(repo.get("license") or {}).get("name") or "未核验",
    )

    readme = github_raw_readme(full_name, default_branch=default_branch)
    ev.readme_found = bool(readme)
    ev.readme_excerpt = readme[:4000].replace("\n", " ") if readme else ""
    ev.relevance_score, ev.relevance_reasons = repo_relevance_score(repo, readme, query, cand)

    # License endpoint can be more precise.
    license_data = github_api(f"/repos/{full_name}/license")
    if isinstance(license_data, dict):
        lic = license_data.get("license") or {}
        ev.license_spdx = lic.get("spdx_id") or ev.license_spdx
        ev.license_name = lic.get("name") or ev.license_name

    releases = github_api(f"/repos/{full_name}/releases?per_page=20")
    if isinstance(releases, list):
        ev.release_count = len(releases)
        for rel in releases:
            for asset in rel.get("assets", []) or []:
                name = asset.get("name", "")
                if name.lower().endswith(WEIGHT_EXTENSIONS):
                    ev.weight_assets.append(name)

    ev.weight_links_in_readme = find_weight_links(readme)
    ev.training_support = infer_support(readme, TRAINING_KEYWORDS)
    ev.export_support = infer_support(readme, EXPORT_KEYWORDS)
    ev.commercial_risk, ev.risk_reason = license_risk(ev.license_spdx, ev.license_name, readme)
    return ev


def collect_repos(max_results: int, online: bool) -> dict[str, list[RepoEvidence]]:
    results: dict[str, list[RepoEvidence]] = {}
    if not online:
        return results
    for cand in DEFAULT_CANDIDATES:
        aliases = cand["aliases"][:2]
        repo_evs: list[RepoEvidence] = []
        seen = set()
        cheap_candidates: list[tuple[int, dict[str, Any], str]] = []
        for official in cand.get("official_repos", []):
            repo = github_get_repo(official)
            if not repo:
                repo = official_repo_fallback(official)
            full = repo.get("full_name")
            if not full or full in seen:
                continue
            seen.add(full)
            cheap_candidates.append((100, repo, cand["aliases"][0]))
        for alias in aliases:
            repos = github_search_repos(alias, max_results=max(2, min(4, max_results + 1)))
            for repo in repos:
                full = repo.get("full_name")
                if not full or full in seen:
                    continue
                seen.add(full)
                cheap = cheap_repo_score(repo, alias, cand)
                if cheap < 15:
                    continue
        cheap_candidates.append((cheap, repo, alias))
        cheap_candidates.sort(key=lambda item: (item[0], item[1].get("stargazers_count", 0)), reverse=True)
        for _, repo, alias in cheap_candidates[: max_results + 1]:
            ev = inspect_repo(repo, query=alias, cand=cand)
            if ev.relevance_score < 25:
                continue
            repo_evs.append(ev)
            time.sleep(0.2)
        repo_evs.sort(key=lambda ev: (ev.relevance_score, ev.stars or 0), reverse=True)
        results[cand["name"]] = repo_evs[:max_results]
    return results


def extract_metrics_from_text(text: str) -> list[dict[str, str]]:
    if not text:
        return []
    findings = []
    for metric, pattern in METRIC_PATTERNS.items():
        for match in pattern.finditer(text[:80000]):
            value = match.group(1)
            start = max(0, match.start() - 80)
            end = min(len(text), match.end() + 80)
            context = text[start:end].replace("\n", " ")
            findings.append({"metric": metric, "value": value, "context": context})
            if len(findings) >= 12:
                return findings
    return findings


def collect_metrics(papers: list[dict[str, Any]], repos: dict[str, list[RepoEvidence]]) -> dict[str, list[dict[str, str]]]:
    metrics: dict[str, list[dict[str, str]]] = {}
    for cand_name, evs in repos.items():
        combined = []
        for ev in evs:
            combined.extend(extract_metrics_from_text(ev.readme_excerpt))
        metrics[cand_name] = combined[:10]
    # Also create a "task_related_papers" metrics bucket.
    paper_text = "\n".join([(p.get("title") or "") + "\n" + (p.get("abstract") or "") for p in papers])
    metrics["papers"] = extract_metrics_from_text(paper_text)
    return metrics


def count_task_hits(text: str, task_type: str, constraints: list[str]) -> int:
    low = (text or "").lower()
    terms = [task_type.lower(), *[c.lower() for c in constraints], *SMALL_DENSE_KEYWORDS]
    return sum(1 for t in terms if t and t in low)


def score_candidate(cand: dict[str, Any], papers: list[dict[str, Any]], repo_evs: list[RepoEvidence], constraints: list[str]) -> tuple[int, list[str]]:
    score = 20
    reasons = [cand.get("why", "候选算法")]
    text_blob = " ".join(
        [cand["name"], " ".join(cand.get("aliases", []))]
        + [(p.get("title") or "") + " " + (p.get("abstract") or "") for p in papers[:30]]
        + [ev.readme_excerpt for ev in repo_evs[:3]]
    )
    hits = count_task_hits(text_blob, " ".join(cand.get("aliases", [])), constraints)
    score += min(20, hits * 3)
    if repo_evs:
        score += 10
        reasons.append("找到 GitHub 候选仓库")
        best = repo_evs[0]
        if best.stars and best.stars > 100:
            score += 5
            reasons.append("仓库有一定关注度")
        if best.readme_found:
            score += 5
            reasons.append("README 可读取")
        if best.weight_assets or best.weight_links_in_readme:
            score += 10
            reasons.append("发现权重资产或权重链接")
        if "有迹象" in best.training_support or "可能支持" in best.training_support:
            score += 8
            reasons.append("发现训练支持迹象")
        if "有迹象" in best.export_support or "可能支持" in best.export_support:
            score += 5
            reasons.append("发现导出/部署支持迹象")
        if best.commercial_risk == "低":
            score += 8
            reasons.append("代码 license 风险较低")
        elif best.commercial_risk == "高":
            score -= 15
            reasons.append("license 存在高风险")
        elif best.commercial_risk == "中/高":
            score -= 10
            reasons.append("license 存在中高风险")
        else:
            score -= 3
            reasons.append("license 仍需核验")
    else:
        reasons.append("未找到可核验 GitHub 仓库，需人工补充")
    if papers:
        score += 5
        reasons.append("找到相关论文资料")
    score = max(0, min(100, score))
    return score, reasons[:8]


def md(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def table_candidate_summary(scores: dict[str, tuple[int, list[str]]], repos: dict[str, list[RepoEvidence]]) -> str:
    lines = [
        "| 优先级 | 算法/系列 | 调研评分 | 代表仓库 | 权重 | License风险 | 训练支持 | 部署文档迹象 | 是否建议 benchmark |",
        "|---|---|---:|---|---|---|---|---|---|",
    ]
    for cand in DEFAULT_CANDIDATES:
        name = cand["name"]
        score, _ = scores.get(name, (0, []))
        ev = repos.get(name, [None])[0] if repos.get(name) else None
        if score >= 70:
            pri = "P0"
            rec = "强烈建议"
        elif score >= 55:
            pri = "P1"
            rec = "建议"
        elif score >= 40:
            pri = "P2"
            rec = "有条件"
        else:
            pri = "P3"
            rec = "暂不建议/待补证据"
        repo = f"[{ev.full_name}]({ev.html_url})" if ev and ev.html_url != "未找到" else "待核验"
        weights = "有迹象" if ev and (ev.weight_assets or ev.weight_links_in_readme) else "未核验"
        lic = ev.commercial_risk if ev else "未核验"
        train = ev.training_support if ev else "未核验"
        deploy = ev.export_support if ev else "未核验"
        lines.append(f"| {pri} | {md(name)} | {score} | {repo} | {md(weights)} | {md(lic)} | {md(train)} | {md(deploy)} | {rec} |")
    return "\n".join(lines)


def table_papers(papers: list[dict[str, Any]], max_rows: int = 20) -> str:
    lines = [
        "| 标题 | 年份 | 来源 | venue | citation | 相关性分 | URL | 相关性提示 |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for p in papers[:max_rows]:
        title = p.get("title") or "未命名"
        year = p.get("year") or "未知"
        source = p.get("source") or "unknown"
        venue = p.get("venue") or ""
        cite = p.get("citationCount")
        cite_s = "" if cite is None else str(cite)
        relevance = p.get("relevance_score", "")
        url = p.get("url") or (p.get("openAccessPdf") or {}).get("url") or ""
        rel = list(p.get("relevance_reasons") or [])
        text = ((p.get("title") or "") + " " + (p.get("abstract") or "")).lower()
        for kw in SMALL_DENSE_KEYWORDS:
            if kw in text and kw not in rel:
                rel.append(kw)
        rel_s = ", ".join(rel[:5]) if rel else "待人工判断"
        url_md = f"[link]({url})" if url else "无"
        lines.append(f"| {md(title[:120])} | {md(year)} | {md(source)} | {md(venue)} | {md(cite_s)} | {md(relevance)} | {url_md} | {md(rel_s)} |")
    if not papers:
        lines.append("| 未获取到论文 | - | - | - | - | - | - | 检查网络/API 或补充关键词 |")
    return "\n".join(lines)


def table_repos(repos: dict[str, list[RepoEvidence]]) -> str:
    lines = [
        "| 算法 | 仓库 | 匹配分 | 匹配依据 | Stars | 最近更新 | License | Release数 | README | 训练支持 | 导出/部署文档 |",
        "|---|---|---:|---|---:|---|---|---:|---|---|---|",
    ]
    any_row = False
    for name, evs in repos.items():
        for ev in evs[:3]:
            any_row = True
            repo = f"[{ev.full_name}]({ev.html_url})"
            relevance = ", ".join(ev.relevance_reasons[:4]) if ev.relevance_reasons else "待人工判断"
            lines.append(
                f"| {md(name)} | {repo} | {ev.relevance_score} | {md(relevance)} | {ev.stars or 0} | {md(ev.pushed_at)} | {md(ev.license_spdx)} | {ev.release_count} | {'是' if ev.readme_found else '否'} | {md(ev.training_support)} | {md(ev.export_support)} |"
            )
    if not any_row:
        lines.append("| 未获取到仓库 | - | - | - | - | - | - | - | - | - | - |")
    return "\n".join(lines)


def table_weights(repos: dict[str, list[RepoEvidence]]) -> str:
    lines = [
        "| 算法 | 仓库 | Release权重资产 | README权重链接 | 权重状态 |",
        "|---|---|---|---|---|",
    ]
    any_row = False
    for name, evs in repos.items():
        for ev in evs[:3]:
            any_row = True
            assets = "<br>".join(ev.weight_assets[:8]) if ev.weight_assets else "未发现"
            links = "<br>".join(ev.weight_links_in_readme[:5]) if ev.weight_links_in_readme else "未发现"
            status = "有迹象" if ev.weight_assets or ev.weight_links_in_readme else "未核验"
            repo = f"[{ev.full_name}]({ev.html_url})"
            lines.append(f"| {md(name)} | {repo} | {md(assets)} | {md(links)} | {status} |")
    if not any_row:
        lines.append("| 未获取到仓库 | - | - | - | 待核验 |")
    return "\n".join(lines)


def table_license(repos: dict[str, list[RepoEvidence]]) -> str:
    lines = [
        "| 算法 | 仓库 | 代码License | 商用风险 | 风险原因 | 处理建议 |",
        "|---|---|---|---|---|---|",
    ]
    any_row = False
    for name, evs in repos.items():
        for ev in evs[:3]:
            any_row = True
            repo = f"[{ev.full_name}]({ev.html_url})"
            suggestion = "低风险也建议法务确认权重 license" if ev.commercial_risk == "低" else "进入 benchmark 前必须人工复核"
            lines.append(f"| {md(name)} | {repo} | {md(ev.license_spdx)} | {md(ev.commercial_risk)} | {md(ev.risk_reason)} | {md(suggestion)} |")
    if not any_row:
        lines.append("| 未获取到仓库 | - | 未核验 | 未核验 | 无网络/API结果 | 补充官方链接或配置 GitHub Token |")
    return "\n".join(lines)


def table_metrics(metrics: dict[str, list[dict[str, str]]]) -> str:
    lines = [
        "| 来源/算法 | 指标 | 数值 | 上下文 | 可比性说明 |",
        "|---|---|---|---|---|",
    ]
    any_row = False
    for source, findings in metrics.items():
        for f in findings[:8]:
            any_row = True
            lines.append(
                f"| {md(source)} | {md(f.get('metric'))} | {md(f.get('value'))} | {md(f.get('context'))} | 公开资料抽取，需核验数据集/硬件/输入尺寸，不能直接代表当前业务数据集效果 |"
            )
    if not any_row:
        lines.append("| 未抽取到公开指标 | - | - | - | 需要人工查看论文表格或官方 model zoo |")
    return "\n".join(lines)


def render_report(data: dict[str, Any], papers: list[dict[str, Any]], repos: dict[str, list[RepoEvidence]], metrics: dict[str, list[dict[str, str]]]) -> str:
    constraints = data.get("constraints", [])
    constraints_md = "\n".join(f"- {md(c)}" for c in constraints) if constraints else "- 未提供"
    all_paper_text = " ".join([(p.get("title") or "") + " " + (p.get("abstract") or "") for p in papers])
    scores = {}
    for cand in DEFAULT_CANDIDATES:
        scores[cand["name"]] = score_candidate(cand, papers, repos.get(cand["name"], []), constraints)

    recommendations = []
    for cand in DEFAULT_CANDIDATES:
        name = cand["name"]
        score, reasons = scores[name]
        if score >= 70:
            level = "P0"
        elif score >= 55:
            level = "P1"
        elif score >= 40:
            level = "P2"
        else:
            level = "P3"
        recommendations.append((level, score, name, reasons))

    rec_lines = []
    for level, score, name, reasons in sorted(recommendations, key=lambda x: (x[0], -x[1])):
        rec_lines.append(f"### {level}｜{name}｜调研评分 {score}/100\n")
        rec_lines.append("\n".join(f"- {md(r)}" for r in reasons))
        rec_lines.append("")

    task_type = data.get("task_type") or data.get("objective") or "目标检测"

    return f"""# 目标检测候选算法调研报告

生成时间：{now()}

## 1. 任务与约束

- Skill：`algorithm-research-scout`
- 模式：`{md(data.get("mode", "online-research"))}`
- 任务类型：{md(task_type)}
- 当前 baseline：{md(data.get("baseline", "YOLO"))}

约束条件：

{constraints_md}

## 2. 调研结论摘要

本报告只做调研核验，不做本地编译验证，也不下载大权重做训练实测。

结论使用方式：

- 调研评分代表“是否值得进入 benchmark”，不代表最终一定优于 YOLO。
- 公开 mAP/FPS 只能作为参考，不能直接代表当前业务数据集效果。
- License 未核验或权重 license 未核验时，不能直接进入商用交付。
- 最终是否比 YOLO 更适合当前业务目标，需要在统一数据集上 benchmark 后确认。

## 3. 候选算法总表

{table_candidate_summary(scores, repos)}

## 4. 论文证据表

{table_papers(papers)}

## 5. GitHub 实现核验表

{table_repos(repos)}

## 6. 预训练权重核验表

{table_weights(repos)}

## 7. License 风险表

{table_license(repos)}

## 8. 公开 mAP/FPS 指标表

{table_metrics(metrics)}

## 9. 业务场景适配评分

{chr(10).join(rec_lines)}

## 10. Benchmark 推荐短名单

建议按以下顺序推进：

1. **当前 YOLO baseline**：必须保留，作为统一对照组。
2. **P0 候选**：优先 benchmark。
3. **P1 候选**：在 license、权重、训练支持核验后 benchmark。
4. **P2 候选**：仅在项目时间允许且部署资料充足时 benchmark。
5. **P3 候选**：暂不建议，除非人工补充强证据。

## 11. 风险与不确定项

- 公开论文和 README 指标可能来自 COCO 或其他数据集，与当前业务数据集不可直接比较。
- GitHub 搜索可能找到非官方仓库，需要人工确认官方性。
- 代码 license 不等于权重 license。
- README 中的权重链接可能失效，需要进一步验证下载可用性。
- “支持 ONNX/TensorRT/BModel”在本版本仅做文档迹象判断，不做真实编译验证。
- 如果没有配置 `GITHUB_TOKEN`，GitHub API 可能触发限流，导致结果不完整。

## 12. 还需要人工补充的信息

- 当前 YOLO baseline 的实际指标：Precision、Recall、mAP50、mAP50-95。
- 当前业务数据集规模、图片分辨率、目标大小分布。
- 是否要求商用 license 完全明确。
- 训练资源和期望模型大小。
- 是否更重视 Recall 还是 Precision。
- 是否有必须兼容的部署平台。
"""


def serialize_repos(repos: dict[str, list[RepoEvidence]]) -> dict[str, list[dict[str, Any]]]:
    return {k: [ev.__dict__ for ev in v] for k, v in repos.items()}


def save_json_artifacts(out_dir: Path, papers: list[dict[str, Any]], repos: dict[str, list[RepoEvidence]], metrics: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    files = {
        "papers.json": papers,
        "repos.json": serialize_repos(repos),
        "metrics.json": metrics,
    }
    for name, data in files.items():
        p = out_dir / name
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts.append({"type": "json", "path": str(p)})
    return artifacts


def main() -> None:
    args = parse_args()
    data = merge_inputs(args)
    online = data.get("mode", args.mode) == "online-research"

    papers = collect_papers(data.get("task_type", ""), data.get("constraints", []), int(data.get("max_results", args.max_results)), online=online)
    repos = collect_repos(max_results=int(data.get("max_results", args.max_results)), online=online)
    metrics = collect_metrics(papers, repos)

    report = render_report(data, papers, repos, metrics)

    artifacts: list[dict[str, str]] = []
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        artifacts.append({"type": "markdown", "path": str(out)})
    else:
        print(report)

    json_output_dir = args.json_output_dir
    if json_output_dir:
        artifacts.extend(save_json_artifacts(Path(json_output_dir), papers, repos, metrics))

    if args.output:
        print(json.dumps({"artifacts": artifacts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
