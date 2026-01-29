"""
提示词优化相关API接口
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
from dashscope import Generation
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prompt", tags=["Prompt"])


class PromptOptimizeRequest(BaseModel):
    """提示词优化请求"""
    prompt: str = Field(..., description="原始提示词")
    context: Optional[str] = Field(None, description="上下文信息")
    optimization_goal: Optional[str] = Field(
        "clarity", 
        description="优化目标: clarity(清晰度), specificity(具体性), creativity(创造性), technical(技术性)"
    )
    target_model: Optional[str] = Field(None, description="目标模型")
    language: Optional[str] = Field("zh", description="语言: zh(中文), en(英文)")


class PromptOptimizeResponse(BaseModel):
    """提示词优化响应"""
    original_prompt: str = Field(..., description="原始提示词")
    optimized_prompt: str = Field(..., description="优化后的提示词")
    improvements: List[str] = Field(default_factory=list, description="改进建议")
    score: Optional[Dict[str, float]] = Field(None, description="评分")


class PromptTemplateRequest(BaseModel):
    """提示词模板请求"""
    task_type: str = Field(..., description="任务类型")
    variables: Dict[str, Any] = Field(default_factory=dict, description="模板变量")
    
    
class PromptAnalyzeRequest(BaseModel):
    """提示词分析请求"""
    prompt: str = Field(..., description="要分析的提示词")
    

@router.post("/optimize", response_model=PromptOptimizeResponse)
async def optimize_prompt(request: PromptOptimizeRequest):
    """
    优化提示词
    
    该接口会分析输入的提示词，并根据优化目标进行改进。
    """
    try:
        logger.info(f"优化提示词，目标: {request.optimization_goal}")
        
        # 构建优化提示词的系统提示
        system_prompt = f"""你是一个专业的提示词优化专家。请根据以下要求优化用户提供的提示词：

优化目标: {request.optimization_goal}
目标语言: {'中文' if request.language == 'zh' else '英文'}

优化原则：
1. 清晰度(clarity): 使提示词更加清晰明确，减少歧义
2. 具体性(specificity): 添加具体的细节和约束条件
3. 创造性(creativity): 增加创意元素，激发更有创意的回答
4. 技术性(technical): 增加技术细节和专业术语

请按以下格式输出：
【优化后的提示词】
<这里是优化后的提示词>

【改进说明】
1. <改进点1>
2. <改进点2>
...

【评分】
- 清晰度: <0-10分>
- 具体性: <0-10分>
- 完整性: <0-10分>
- 可执行性: <0-10分>
"""
        
        # 构建用户消息
        user_message = f"原始提示词：\n{request.prompt}"
        if request.context:
            user_message += f"\n\n上下文信息：\n{request.context}"
        
        # 调用DashScope进行优化
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = Generation.call(
            model="qwen-max",
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            result_format='message'
        )
        
        # 解析响应
        if response.status_code == 200:
            response_text = response.output.choices[0].message.content
        else:
            raise Exception(f"API调用失败: {response.message}")
        
        # 提取优化后的提示词
        optimized_prompt = ""
        improvements = []
        score = {}
        
        if "【优化后的提示词】" in response_text:
            parts = response_text.split("【优化后的提示词】")[1]
            if "【改进说明】" in parts:
                optimized_prompt = parts.split("【改进说明】")[0].strip()
                
                # 提取改进说明
                improvement_part = parts.split("【改进说明】")[1]
                if "【评分】" in improvement_part:
                    improvement_text = improvement_part.split("【评分】")[0]
                    lines = improvement_text.strip().split("\n")
                    for line in lines:
                        line = line.strip()
                        if line and (line[0].isdigit() or line.startswith("-")):
                            improvements.append(line.lstrip("0123456789. -"))
                    
                    # 提取评分
                    score_text = improvement_part.split("【评分】")[1]
                    score_lines = score_text.strip().split("\n")
                    for line in score_lines:
                        if ":" in line or "：" in line:
                            parts = line.replace("：", ":").split(":")
                            if len(parts) == 2:
                                key = parts[0].strip().lstrip("- ")
                                try:
                                    value = float(parts[1].strip().rstrip("分"))
                                    score[key] = value
                                except:
                                    pass
        else:
            # 如果格式不对，直接使用响应作为优化后的提示词
            optimized_prompt = response_text
        
        # 如果没有提取到优化后的提示词，使用原始提示词
        if not optimized_prompt:
            optimized_prompt = request.prompt
            improvements.append("优化失败，返回原始提示词")
        
        return PromptOptimizeResponse(
            original_prompt=request.prompt,
            optimized_prompt=optimized_prompt,
            improvements=improvements,
            score=score if score else None
        )
        
    except Exception as e:
        logger.error(f"优化提示词失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze_prompt(request: PromptAnalyzeRequest):
    """
    分析提示词质量
    
    分析提示词的结构、清晰度、完整性等方面。
    """
    try:
        logger.info("分析提示词质量")
        
        system_prompt = """你是一个提示词分析专家。请分析用户提供的提示词，从以下维度进行评估：

1. 清晰度：提示词是否清晰明确
2. 具体性：是否包含足够的细节
3. 完整性：是否包含必要的信息
4. 结构性：是否有良好的结构
5. 可执行性：AI是否能准确理解并执行

请给出详细的分析报告和改进建议。"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请分析这个提示词：\n{request.prompt}"}
        ]
        
        response = Generation.call(
            model="qwen-max",
            messages=messages,
            temperature=0.5,
            max_tokens=1500,
            result_format='message'
        )
        
        if response.status_code == 200:
            analysis_text = response.output.choices[0].message.content
        else:
            raise Exception(f"API调用失败: {response.message}")
        
        return {
            "prompt": request.prompt,
            "analysis": analysis_text,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"分析提示词失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def get_prompt_templates():
    """
    获取提示词模板列表
    """
    templates = {
        "code_generation": {
            "name": "代码生成",
            "template": "请帮我编写一个{language}程序，实现以下功能：\n{requirements}\n\n要求：\n- {constraints}\n- 包含详细注释\n- 考虑错误处理",
            "variables": ["language", "requirements", "constraints"]
        },
        "content_writing": {
            "name": "内容创作",
            "template": "请以{tone}的语气，为{audience}撰写一篇关于{topic}的{content_type}。\n\n要点：\n{key_points}\n\n字数要求：{word_count}",
            "variables": ["tone", "audience", "topic", "content_type", "key_points", "word_count"]
        },
        "data_analysis": {
            "name": "数据分析",
            "template": "请分析以下{data_type}数据：\n{data}\n\n分析维度：\n- {dimensions}\n\n请提供：\n1. 关键发现\n2. 数据趋势\n3. 建议措施",
            "variables": ["data_type", "data", "dimensions"]
        },
        "problem_solving": {
            "name": "问题解决",
            "template": "问题描述：{problem}\n\n背景信息：{background}\n\n限制条件：{constraints}\n\n请提供：\n1. 问题分析\n2. 解决方案\n3. 实施步骤\n4. 预期效果",
            "variables": ["problem", "background", "constraints"]
        },
        "translation": {
            "name": "翻译",
            "template": "请将以下{source_lang}文本翻译成{target_lang}：\n\n{text}\n\n翻译要求：\n- 保持原意\n- 符合{target_lang}表达习惯\n- {style}风格",
            "variables": ["source_lang", "target_lang", "text", "style"]
        }
    }
    
    return {
        "templates": templates,
        "total": len(templates)
    }


@router.post("/template/apply")
async def apply_prompt_template(request: PromptTemplateRequest):
    """
    应用提示词模板
    """
    try:
        templates = {
            "code_generation": "请帮我编写一个{language}程序，实现以下功能：\n{requirements}\n\n要求：\n- {constraints}\n- 包含详细注释\n- 考虑错误处理",
            "content_writing": "请以{tone}的语气，为{audience}撰写一篇关于{topic}的{content_type}。\n\n要点：\n{key_points}\n\n字数要求：{word_count}",
            "data_analysis": "请分析以下{data_type}数据：\n{data}\n\n分析维度：\n- {dimensions}\n\n请提供：\n1. 关键发现\n2. 数据趋势\n3. 建议措施",
            "problem_solving": "问题描述：{problem}\n\n背景信息：{background}\n\n限制条件：{constraints}\n\n请提供：\n1. 问题分析\n2. 解决方案\n3. 实施步骤\n4. 预期效果",
            "translation": "请将以下{source_lang}文本翻译成{target_lang}：\n\n{text}\n\n翻译要求：\n- 保持原意\n- 符合{target_lang}表达习惯\n- {style}风格"
        }
        
        if request.task_type not in templates:
            raise HTTPException(status_code=400, detail=f"未知的任务类型: {request.task_type}")
        
        template = templates[request.task_type]
        
        # 替换模板变量
        prompt = template
        for key, value in request.variables.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        
        # 检查是否还有未替换的变量
        import re
        remaining_vars = re.findall(r'\{(\w+)\}', prompt)
        if remaining_vars:
            return {
                "status": "warning",
                "prompt": prompt,
                "missing_variables": remaining_vars,
                "message": f"模板中还有未替换的变量: {', '.join(remaining_vars)}"
            }
        
        return {
            "status": "success",
            "prompt": prompt,
            "task_type": request.task_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"应用提示词模板失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-optimize")
async def batch_optimize_prompts(prompts: List[str]):
    """
    批量优化提示词
    """
    try:
        results = []
        for prompt in prompts[:10]:  # 限制最多10个
            request = PromptOptimizeRequest(prompt=prompt)
            result = await optimize_prompt(request)
            results.append(result)
        
        return {
            "status": "success",
            "results": results,
            "total": len(results)
        }
        
    except Exception as e:
        logger.error(f"批量优化提示词失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))