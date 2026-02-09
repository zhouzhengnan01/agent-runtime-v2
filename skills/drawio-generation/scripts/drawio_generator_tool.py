"""
Draw.io 架构图生成工具 - 生成产品方向规划的架构图
"""
from __future__ import annotations

import logging
import argparse
import sys
import re
from pathlib import Path
from typing import Any, Dict, Optional
import xml.etree.ElementTree as ET
import base64
import zlib
import json
from urllib.parse import quote

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from app.core.tools.base import BaseTool, register_tool
except Exception:  # pragma: no cover - skill script fallback
    class BaseTool:  # type: ignore
        def __init__(self, config: Optional[Dict[str, Any]] = None):
            self.config = config or {}

    def register_tool(name: str = None, allow_overwrite: bool = False):  # type: ignore
        def decorator(cls):
            return cls
        return decorator

logger = logging.getLogger(__name__)

_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_.\u4e00-\u9fff-]+")


def _default_output_path(title: str) -> Path:
    out_dir = _PROJECT_ROOT / "storage" / "skill_outputs" / "drawio-generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = _FILENAME_SAFE_RE.sub("_", str(title).strip()) or "diagram"
    if not safe.endswith(".drawio"):
        safe = safe + ".drawio"
    return out_dir / safe


@register_tool("DrawIOGenerator")
class DrawIOGeneratorTool(BaseTool):
    """Draw.io 架构图生成工具 - 自动生成可编辑的架构图"""

    name: str = "DrawIOGenerator"
    description: str = "生成 Draw.io 格式的架构图，支持产品架构、系统架构、流程图等。生成的文件可以在 https://app.diagrams.net/ 中打开和编辑。"

    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "diagram_type": {
                "type": "string",
                "enum": ["product_architecture", "system_architecture", "flowchart", "mindmap"],
                "description": "架构图类型：product_architecture(产品架构)、system_architecture(系统架构)、flowchart(流程图)、mindmap(思维导图)"
            },
            "title": {
                "type": "string",
                "description": "架构图标题"
            },
            "data": {
                "type": "object",
                "description": "架构图数据，格式根据 diagram_type 不同而不同。例如：{'modules': [...], 'connections': [...]}"
            },
            "output_path": {
                "type": "string",
                "description": "输出文件路径（可选），不指定则使用默认路径"
            },
            "style": {
                "type": "string",
                "enum": ["default", "modern", "colorful", "minimal"],
                "description": "样式风格：default(默认)、modern(现代)、colorful(多彩)、minimal(极简)",
                "default": "default"
            }
        },
        "required": ["diagram_type", "title", "data"]
    }

    output: Dict[str, Any] = {
        "type": "object",
        "properties": [
            {
                "id": "status",
                "name": "执行状态",
                "valueType": {"type": "string"},
                "description": "执行状态（success/error）"
            },
            {
                "id": "file_path",
                "name": "文件路径",
                "valueType": {"type": "string"},
                "description": "生成的 Draw.io 文件路径"
            },
            {
                "id": "diagram_type",
                "name": "架构图类型",
                "valueType": {"type": "string"},
                "description": "生成的架构图类型"
            },
            {
                "id": "title",
                "name": "标题",
                "valueType": {"type": "string"},
                "description": "架构图标题"
            },
            {
                "id": "web_url",
                "name": "在线打开链接",
                "valueType": {"type": "string"},
                "description": "可以在 diagrams.net 打开的链接"
            },
            {
                "id": "error",
                "name": "错误信息",
                "valueType": {"type": "string"},
                "description": "错误信息（如果失败）"
            }
        ]
    }

    require_confirmation: bool = False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化工具"""
        super().__init__(config)

        # 样式配置
        self.styles = {
            "default": {
                "fillColor": "#dae8fc",
                "strokeColor": "#6c8ebf",
                "fontColor": "#000000"
            },
            "modern": {
                "fillColor": "#667eea",
                "strokeColor": "#764ba2",
                "fontColor": "#ffffff"
            },
            "colorful": {
                "fillColor": "#f8cecc",
                "strokeColor": "#b85450",
                "fontColor": "#000000"
            },
            "minimal": {
                "fillColor": "#ffffff",
                "strokeColor": "#000000",
                "fontColor": "#000000"
            }
        }

    def run(
        self,
        diagram_type: str,
        title: str,
        data: Dict[str, Any],
        output_path: str = None,
        style: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行 Draw.io 架构图生成

        Args:
            diagram_type: 架构图类型
            title: 标题
            data: 数据
            output_path: 输出路径（可选）
            style: 样式风格
            **kwargs: 其他参数

        Returns:
            {
                "success": True/False,
                "file_path": "生成的文件路径",
                "diagram_type": "架构图类型",
                "title": "标题",
                "web_url": "在线打开链接",
                "error": "错误信息"（如果失败）
            }
        """
        try:
            logger.info(f"🎨 [Draw.io生成器] 开始生成架构图: {title}")
            logger.info(f"   类型: {diagram_type}, 样式: {style}")

            # 设置默认输出路径
            if not output_path:
                output_path = str(_default_output_path(title))
            else:
                output_path = str(Path(output_path).expanduser().resolve())

            # 根据类型生成不同的架构图
            if diagram_type == "product_architecture":
                xml_content = self._generate_product_architecture(title, data, style)
            elif diagram_type == "system_architecture":
                xml_content = self._generate_system_architecture(title, data, style)
            elif diagram_type == "flowchart":
                xml_content = self._generate_flowchart(title, data, style)
            elif diagram_type == "mindmap":
                xml_content = self._generate_mindmap(title, data, style)
            else:
                raise ValueError(f"不支持的架构图类型: {diagram_type}")

            # 保存文件
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)

            logger.info(f"✅ [Draw.io生成器] 生成成功: {output_path}")

            # 生成在线查看链接
            web_url = self._generate_web_url(xml_content)

            return {
                "success": True,
                "file_path": output_path,
                "diagram_type": diagram_type,
                "title": title,
                "web_url": web_url
            }

        except Exception as e:
            logger.error(f"❌ [Draw.io生成器] 生成失败: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "title": title
            }

    def _generate_product_architecture(
        self,
        title: str,
        data: Dict[str, Any],
        style: str
    ) -> str:
        """生成产品架构图"""

        style_config = self.styles.get(style, self.styles["default"])

        # 创建 mxGraphModel
        root = ET.Element("mxfile", host="app.diagrams.net", modified="2024-12-02T00:00:00.000Z", agent="DrawIOGeneratorTool", version="21.0.0")
        diagram = ET.SubElement(root, "diagram", name=title, id="diagram1")
        mxGraphModel = ET.SubElement(diagram, "mxGraphModel", dx="1422", dy="794", grid="1", gridSize="10", guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth="1169", pageHeight="827", math="0", shadow="0")

        root_element = ET.SubElement(mxGraphModel, "root")
        ET.SubElement(root_element, "mxCell", id="0")
        ET.SubElement(root_element, "mxCell", id="1", parent="0")

        # 绘制标题
        cell_id = 2
        title_cell = ET.SubElement(root_element, "mxCell", id=str(cell_id), value=title, style=f"text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=24;fontStyle=1;fontColor={style_config['fontColor']};", vertex="1", parent="1")
        ET.SubElement(title_cell, "mxGeometry", x="400", y="40", width="400", height="40", **{"as": "geometry"})
        cell_id += 1

        # 获取模块数据
        modules = data.get("modules", [])
        connections = data.get("connections", [])

        # 计算布局
        y_start = 120
        x_start = 100
        module_width = 200
        module_height = 100
        spacing_x = 250
        spacing_y = 150

        # 绘制模块
        module_positions = {}
        row = 0
        col = 0
        max_cols = 4

        for idx, module in enumerate(modules):
            module_name = module.get("name", f"模块{idx+1}")
            module_desc = module.get("description", "")

            x = x_start + col * spacing_x
            y = y_start + row * spacing_y

            # 创建矩形
            module_cell = ET.SubElement(
                root_element,
                "mxCell",
                id=str(cell_id),
                value=f"<b>{module_name}</b><br/>{module_desc}",
                style=f"rounded=1;whiteSpace=wrap;html=1;fillColor={style_config['fillColor']};strokeColor={style_config['strokeColor']};fontColor={style_config['fontColor']};",
                vertex="1",
                parent="1"
            )
            ET.SubElement(module_cell, "mxGeometry", x=str(x), y=str(y), width=str(module_width), height=str(module_height), **{"as": "geometry"})

            module_positions[module_name] = {
                "id": cell_id,
                "x": x + module_width / 2,
                "y": y + module_height / 2
            }
            cell_id += 1

            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        # 绘制连接线
        for connection in connections:
            source = connection.get("source")
            target = connection.get("target")
            label = connection.get("label", "")

            if source in module_positions and target in module_positions:
                source_id = module_positions[source]["id"]
                target_id = module_positions[target]["id"]

                edge_cell = ET.SubElement(
                    root_element,
                    "mxCell",
                    id=str(cell_id),
                    value=label,
                    style=f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor={style_config['strokeColor']};fontColor={style_config['fontColor']};",
                    edge="1",
                    parent="1",
                    source=str(source_id),
                    target=str(target_id)
                )
                ET.SubElement(edge_cell, "mxGeometry", relative="1", **{"as": "geometry"})
                cell_id += 1

        # 转换为字符串
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")

        import io
        output = io.BytesIO()
        tree.write(output, encoding='utf-8', xml_declaration=True)
        return output.getvalue().decode('utf-8')

    def _generate_system_architecture(
        self,
        title: str,
        data: Dict[str, Any],
        style: str
    ) -> str:
        """生成系统架构图 - 分层架构"""

        style_config = self.styles.get(style, self.styles["default"])

        root = ET.Element("mxfile", host="app.diagrams.net")
        diagram = ET.SubElement(root, "diagram", name=title)
        mxGraphModel = ET.SubElement(diagram, "mxGraphModel")

        root_element = ET.SubElement(mxGraphModel, "root")
        ET.SubElement(root_element, "mxCell", id="0")
        ET.SubElement(root_element, "mxCell", id="1", parent="0")

        cell_id = 2

        # 标题
        title_cell = ET.SubElement(root_element, "mxCell", id=str(cell_id), value=title, style=f"text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=24;fontStyle=1;fontColor={style_config['fontColor']};", vertex="1", parent="1")
        ET.SubElement(title_cell, "mxGeometry", x="400", y="40", width="400", height="40", **{"as": "geometry"})
        cell_id += 1

        # 分层架构
        layers = data.get("layers", [])
        y_start = 120
        layer_height = 150
        layer_spacing = 30

        for idx, layer in enumerate(layers):
            layer_name = layer.get("name", f"Layer {idx+1}")
            components = layer.get("components", [])

            y = y_start + idx * (layer_height + layer_spacing)

            # 层容器
            layer_cell = ET.SubElement(
                root_element,
                "mxCell",
                id=str(cell_id),
                value=f"<b>{layer_name}</b>",
                style=f"swimlane;fillColor={style_config['fillColor']};strokeColor={style_config['strokeColor']};fontColor={style_config['fontColor']};",
                vertex="1",
                parent="1"
            )
            ET.SubElement(layer_cell, "mxGeometry", x="100", y=str(y), width="1000", height=str(layer_height), **{"as": "geometry"})
            cell_id += 1

            # 组件
            comp_x = 20
            comp_width = 180
            comp_spacing = 20

            for comp_idx, component in enumerate(components):
                comp_name = component.get("name", f"Component {comp_idx+1}")
                comp_cell = ET.SubElement(
                    root_element,
                    "mxCell",
                    id=str(cell_id),
                    value=comp_name,
                    style=f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={style_config['strokeColor']};fontColor={style_config['fontColor']};",
                    vertex="1",
                    parent=str(cell_id - 1)
                )
                comp_x_pos = comp_x + comp_idx * (comp_width + comp_spacing)
                ET.SubElement(comp_cell, "mxGeometry", x=str(comp_x_pos), y="40", width=str(comp_width), height="80", **{"as": "geometry"})
                cell_id += 1

        import io
        output = io.BytesIO()
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output, encoding='utf-8', xml_declaration=True)
        return output.getvalue().decode('utf-8')

    def _generate_flowchart(
        self,
        title: str,
        data: Dict[str, Any],
        style: str
    ) -> str:
        """生成流程图"""
        style_config = self.styles.get(style, self.styles["default"])

        root = ET.Element("mxfile", host="app.diagrams.net")
        diagram = ET.SubElement(root, "diagram", name=title)
        mxGraphModel = ET.SubElement(diagram, "mxGraphModel")

        root_element = ET.SubElement(mxGraphModel, "root")
        ET.SubElement(root_element, "mxCell", id="0")
        ET.SubElement(root_element, "mxCell", id="1", parent="0")

        cell_id = 2

        # 标题
        title_cell = ET.SubElement(root_element, "mxCell", id=str(cell_id), value=title, style=f"text;html=1;align=center;fontSize=24;fontStyle=1;fontColor={style_config['fontColor']};", vertex="1", parent="1")
        ET.SubElement(title_cell, "mxGeometry", x="400", y="40", width="400", height="40", **{"as": "geometry"})
        cell_id += 1

        # 流程节点
        steps = data.get("steps", [])
        y = 120
        x = 400
        step_height = 60
        step_width = 200
        spacing = 80

        step_ids = {}

        for idx, step in enumerate(steps):
            step_name = step.get("name", f"Step {idx+1}")
            step_type = step.get("type", "process")  # start, process, decision, end

            # 根据类型选择形状
            if step_type == "start":
                shape_style = "ellipse"
            elif step_type == "decision":
                shape_style = "rhombus"
            elif step_type == "end":
                shape_style = "ellipse"
            else:
                shape_style = "rounded=1;whiteSpace=wrap"

            step_cell = ET.SubElement(
                root_element,
                "mxCell",
                id=str(cell_id),
                value=step_name,
                style=f"{shape_style};html=1;fillColor={style_config['fillColor']};strokeColor={style_config['strokeColor']};fontColor={style_config['fontColor']};",
                vertex="1",
                parent="1"
            )
            ET.SubElement(step_cell, "mxGeometry", x=str(x), y=str(y), width=str(step_width), height=str(step_height), **{"as": "geometry"})

            step_ids[step_name] = cell_id
            cell_id += 1
            y += step_height + spacing

        import io
        output = io.BytesIO()
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output, encoding='utf-8', xml_declaration=True)
        return output.getvalue().decode('utf-8')

    def _generate_mindmap(
        self,
        title: str,
        data: Dict[str, Any],
        style: str
    ) -> str:
        """生成思维导图"""
        # 简化版思维导图实现
        return self._generate_product_architecture(title, data, style)

    def _generate_web_url(self, xml_content: str) -> str:
        """生成可在 diagrams.net 打开的链接"""
        try:
            # 压缩和编码
            compressed = zlib.compress(xml_content.encode('utf-8'))
            encoded = base64.b64encode(compressed).decode('utf-8')
            url_encoded = quote(encoded)

            return f"https://app.diagrams.net/?lightbox=1#R{url_encoded}"
        except Exception as e:
            logger.warning(f"生成 Web URL 失败: {e}")
            return "https://app.diagrams.net/"

    async def run_async(
        self,
        diagram_type: str,
        title: str,
        data: Dict[str, Any],
        output_path: str = None,
        style: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """异步执行"""
        import asyncio
        return await asyncio.to_thread(self.run, diagram_type, title, data, output_path, style, **kwargs)


# 工具实例
drawio_generator_tool = DrawIOGeneratorTool()


def _load_data_arg(value: Optional[str]) -> Dict[str, Any]:
    if not value:
        return {}
    raw = value.strip()
    if raw.startswith("@"):
        path = Path(raw[1:]).expanduser().resolve()
        return json.loads(path.read_text(encoding="utf-8"))
    # JSON string
    return json.loads(raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 Draw.io (.drawio) 文件")
    parser.add_argument(
        "--diagram-type",
        default="system_architecture",
        choices=["product_architecture", "system_architecture", "flowchart", "mindmap"],
        help="架构图类型",
    )
    parser.add_argument("--title", default="Demo Diagram", help="标题")
    parser.add_argument(
        "--data",
        help="架构图数据（JSON 字符串），或使用 @path.json 读取文件",
    )
    parser.add_argument("--output", help="输出文件路径（.drawio）")
    parser.add_argument(
        "--style",
        default="modern",
        choices=["default", "modern", "colorful", "minimal"],
        help="样式风格",
    )
    args = parser.parse_args()

    if args.data:
        data = _load_data_arg(args.data)
    else:
        # 默认示例数据
        data = {
            "layers": [
                {"name": "Access", "components": [{"name": "Gateway"}]},
                {"name": "Service", "components": [{"name": "API"}, {"name": "Worker"}]},
                {"name": "Storage", "components": [{"name": "DB"}, {"name": "Object Storage"}]},
            ]
        }

    result = drawio_generator_tool.run(
        diagram_type=args.diagram_type,
        title=args.title,
        data=data,
        output_path=args.output,
        style=args.style,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
