#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any


class SkillAdapterError(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description="JetLinks OpenSandbox skill adapter.")
    parser.add_argument("--request", required=True, help="Path to skill-run.v1 request JSON.")
    parser.add_argument("--outputs", required=True, help="Directory where artifacts must be written.")
    args = parser.parse_args()

    try:
        request_path = Path(args.request)
        outputs_dir = Path(args.outputs)
        request = _load_request(request_path)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        result = run_skill(request, outputs_dir)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}), file=sys.stderr)
        return 1

    print(json.dumps({"ok": True, **result}, ensure_ascii=False))
    return 0


def run_skill(request: dict[str, Any], outputs_dir: Path) -> dict[str, Any]:
    skill_name = _required_string(request, "skill_name")
    spec = request.get("spec", {})
    if not isinstance(spec, dict):
        raise SkillAdapterError("request.spec must be an object")

    if skill_name == "drawio-generation":
        artifacts = _run_drawio(spec, outputs_dir)
        return {"skill_name": skill_name, "artifacts": artifacts}

    raise SkillAdapterError(f"Unsupported sandbox skill adapter: {skill_name}")


def _run_drawio(spec: dict[str, Any], outputs_dir: Path) -> list[dict[str, str]]:
    title = str(spec.get("title") or "JetLinks Architecture")
    nodes = _string_list(spec.get("nodes")) or [
        "User",
        "Agent Runtime",
        "Skill Adapter",
        "Artifact Store",
    ]
    edges = _edge_list(spec.get("edges")) or [
        ("User", "Agent Runtime"),
        ("Agent Runtime", "Skill Adapter"),
        ("Skill Adapter", "Artifact Store"),
    ]
    lanes = _string_list(spec.get("swimlanes")) or ["Access", "Runtime", "Execution", "Outputs"]
    lane_nodes = _lane_nodes(spec.get("lane_nodes"), lanes, nodes)
    layout = _layout(nodes, lanes, lane_nodes)
    stem = _next_stem(outputs_dir, "prototype" if spec.get("diagram_type") == "prototype_wireframe" else "architecture")

    drawio_path = outputs_dir / f"{stem}.drawio"
    png_path = outputs_dir / f"{stem}.png"
    drawio_path.write_text(_drawio_xml(title, layout, edges), encoding="utf-8")
    _write_png(title, layout, edges, png_path)
    return [
        {"name": drawio_path.name, "mime_type": "application/vnd.jgraph.mxfile", "kind": "text"},
        {"name": png_path.name, "mime_type": "image/png", "kind": "image"},
    ]


def _load_request(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SkillAdapterError("request JSON must be an object")
    version = data.get("request_schema_version")
    if version not in (None, "skill-run.v1"):
        raise SkillAdapterError(f"Unsupported request_schema_version: {version}")
    return data


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SkillAdapterError(f"request.{key} must be a non-empty string")
    return value.strip()


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _edge_list(value: object) -> list[tuple[str, str]]:
    if not isinstance(value, list):
        return []
    edges: list[tuple[str, str]] = []
    for item in value:
        if isinstance(item, list | tuple) and len(item) >= 2:
            edges.append((str(item[0]), str(item[1])))
    return edges


def _lane_nodes(value: object, lanes: list[str], nodes: list[str]) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return _auto_lane_nodes(lanes, nodes)
    known = set(nodes)
    result: dict[str, list[str]] = {}
    assigned: set[str] = set()
    for lane in lanes:
        raw_items = value.get(lane, [])
        items = [str(item) for item in raw_items] if isinstance(raw_items, list) else []
        result[lane] = [item for item in items if item in known and item not in assigned]
        assigned.update(result[lane])
    remaining = [node for node in nodes if node not in assigned]
    for idx, node in enumerate(remaining):
        result.setdefault(lanes[idx % len(lanes)], []).append(node)
    return result


def _auto_lane_nodes(lanes: list[str], nodes: list[str]) -> dict[str, list[str]]:
    result = {lane: [] for lane in lanes}
    for idx, node in enumerate(nodes):
        result[lanes[idx % len(lanes)]].append(node)
    return result


def _layout(nodes: list[str], lanes: list[str], lane_nodes: dict[str, list[str]]) -> dict[str, Any]:
    lane_width = 270
    lane_gap = 24
    margin = 44
    title_h = 64
    header_h = 44
    node_w = 214
    node_h = 62
    row_gap = 18
    max_rows = max(1, max(len(lane_nodes.get(lane, [])) for lane in lanes))
    lane_h = header_h + 26 + max_rows * node_h + (max_rows - 1) * row_gap + 24
    width = margin * 2 + len(lanes) * lane_width + (len(lanes) - 1) * lane_gap
    height = margin + title_h + lane_h + margin
    palette = ["#dbeafe", "#dcfce7", "#fef3c7", "#fce7f3", "#fee2e2", "#ede9fe"]
    accents = ["#2563eb", "#16a34a", "#d97706", "#db2777", "#dc2626", "#7c3aed"]
    lane_layouts: list[dict[str, Any]] = []
    node_layouts: dict[str, dict[str, Any]] = {}
    for lane_idx, lane in enumerate(lanes):
        x = margin + lane_idx * (lane_width + lane_gap)
        y = margin + title_h
        lane_layouts.append(
            {
                "name": lane,
                "x": x,
                "y": y,
                "width": lane_width,
                "height": lane_h,
                "fill": "#f8fafc",
                "accent": accents[lane_idx % len(accents)],
            }
        )
        for row, node in enumerate(lane_nodes.get(lane, [])):
            node_layouts[node] = {
                "lane_index": lane_idx,
                "x": x + (lane_width - node_w) // 2,
                "y": y + header_h + 26 + row * (node_h + row_gap),
                "width": node_w,
                "height": node_h,
                "fill": palette[lane_idx % len(palette)],
                "accent": accents[lane_idx % len(accents)],
            }
    return {"width": width, "height": height, "lanes": lane_layouts, "nodes": node_layouts, "node_order": nodes}


def _drawio_xml(title: str, layout: dict[str, Any], edges: list[tuple[str, str]]) -> str:
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    node_ids: dict[str, str] = {}
    for idx, lane in enumerate(layout["lanes"], start=1):
        cells.append(
            f'<mxCell id="lane{idx}" value="{html.escape(str(lane["name"]))}" '
            'style="swimlane;whiteSpace=wrap;html=1;startSize=44;rounded=1;arcSize=8;'
            f'fontStyle=1;fontSize=14;fillColor={lane["fill"]};strokeColor={lane["accent"]};" '
            'vertex="1" parent="1">'
            f'<mxGeometry x="{lane["x"]}" y="{lane["y"]}" width="{lane["width"]}" height="{lane["height"]}" as="geometry"/>'
            "</mxCell>"
        )
    for idx, node in enumerate(layout["node_order"], start=1):
        item = layout["nodes"][node]
        node_id = f"n{idx}"
        node_ids[node] = node_id
        lane_parent = f'lane{int(item["lane_index"]) + 1}'
        lane = layout["lanes"][int(item["lane_index"])]
        cells.append(
            f'<mxCell id="{node_id}" value="{html.escape(node)}" '
            'style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;spacing=10;fontStyle=1;fontSize=13;'
            f'fillColor={item["fill"]};strokeColor={item["accent"]};fontColor=#0f172a;shadow=1;" '
            f'vertex="1" parent="{lane_parent}">'
            f'<mxGeometry x="{item["x"] - lane["x"]}" y="{item["y"] - lane["y"]}" '
            f'width="{item["width"]}" height="{item["height"]}" as="geometry"/></mxCell>'
        )
    for idx, (source, target) in enumerate(edges, start=1):
        if source not in node_ids or target not in node_ids:
            continue
        cells.append(
            f'<mxCell id="e{idx}" value="" '
            'style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;endFill=1;strokeWidth=2;strokeColor=#2563eb;" '
            f'edge="1" parent="1" source="{node_ids[source]}" target="{node_ids[target]}">'
            '<mxGeometry relative="1" as="geometry"/></mxCell>'
        )
    return (
        '<mxfile host="JetLinks OpenSandbox">'
        f'<diagram name="{html.escape(title)}"><mxGraphModel grid="1" gridSize="10" page="1" '
        f'pageWidth="{layout["width"]}" pageHeight="{layout["height"]}">'
        f'<root>{"".join(cells)}</root></mxGraphModel></diagram></mxfile>'
    )


def _write_png(title: str, layout: dict[str, Any], edges: list[tuple[str, str]], target: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SkillAdapterError("Pillow is required to render PNG previews") from exc

    image = Image.new("RGB", (int(layout["width"]), int(layout["height"])), "#ffffff")
    draw = ImageDraw.Draw(image)
    title_font = _font(ImageFont, 24)
    head_font = _font(ImageFont, 15)
    body_font = _font(ImageFont, 13)
    draw.text((44, 28), title, fill="#0f172a", font=title_font)
    for lane in layout["lanes"]:
        box = (lane["x"], lane["y"], lane["x"] + lane["width"], lane["y"] + lane["height"])
        draw.rounded_rectangle(box, radius=12, fill=lane["fill"], outline=lane["accent"], width=2)
        draw.text((lane["x"] + 16, lane["y"] + 13), lane["name"], fill="#0f172a", font=head_font)
    for node, item in layout["nodes"].items():
        box = (item["x"], item["y"], item["x"] + item["width"], item["y"] + item["height"])
        draw.rounded_rectangle(box, radius=10, fill=item["fill"], outline=item["accent"], width=2)
        _draw_wrapped(draw, node, (item["x"] + 12, item["y"] + 14), int(item["width"]) - 24, body_font)
    for source, target_name in edges:
        source_box = layout["nodes"].get(source)
        target_box = layout["nodes"].get(target_name)
        if not source_box or not target_box:
            continue
        start = (source_box["x"] + source_box["width"], source_box["y"] + source_box["height"] // 2)
        end = (target_box["x"], target_box["y"] + target_box["height"] // 2)
        draw.line((start, end), fill="#2563eb", width=2)
        draw.polygon([(end[0], end[1]), (end[0] - 8, end[1] - 5), (end[0] - 8, end[1] + 5)], fill="#2563eb")
    image.save(target, format="PNG")


def _font(image_font: Any, size: int) -> Any:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return image_font.truetype(path, size=size)
            except OSError:
                continue
    return image_font.load_default()


def _draw_wrapped(draw: Any, text: str, xy: tuple[int, int], max_width: int, font: Any) -> None:
    words = re.split(r"(\s+)", text)
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = current + word
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current.strip())
            current = word.strip()
        else:
            current = candidate
    if current.strip():
        lines.append(current.strip())
    y = xy[1]
    for line in lines[:2]:
        draw.text((xy[0], y), line, fill="#0f172a", font=font)
        y += 18


def _next_stem(outputs_dir: Path, stem: str) -> str:
    if not (outputs_dir / f"{stem}.drawio").exists() and not (outputs_dir / f"{stem}.png").exists():
        return stem
    version = 2
    while (outputs_dir / f"{stem}_v{version}.drawio").exists() or (outputs_dir / f"{stem}_v{version}.png").exists():
        version += 1
    return f"{stem}_v{version}"


if __name__ == "__main__":
    raise SystemExit(main())
