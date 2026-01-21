import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any

from src.video_workflow.config import settings

logger = logging.getLogger(__name__)

async def analyze_reference_image(image_path: str) -> Optional[Dict[str, Any]]:
    """
    使用多模态 LLM 分析参考图，自动生成角色描述。支持豆包和 GLM。
    
    Returns:
        Optional[Dict[str, Any]]: {"character": "...", "style": "..."} or None if failed.
    """
    image_file = Path(image_path)
    if not image_file.exists():
        logger.error(f"参考图不存在: {image_path}")
        return None
    
    try:
        with open(image_file, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"读取参考图失败: {e}")
        return None
    
    # 要求返回 JSON 格式，同时包含角色和风格
    analysis_prompt = """请仔细分析这张图片，提取以下两部分信息，以 JSON 格式返回：

1. character（角色外貌描述，50-100字）：
   - 包含：物种/类型、体型、毛色/肤色、五官特征、服装配饰、表情气质

2. style（视觉风格描述，20-50字）：
   - 分析图片整体视觉风格（如：3D卡通渲染、写实摄影、水彩手绘、赛璐璐动画等）
   - 包含：画面质感、光影风格、色彩特点

请严格按照以下 JSON 格式输出，不要添加其他文字：
{"character": "角色描述内容", "style": "风格描述内容"}"""
    
    loop = asyncio.get_running_loop()
    
    # 方案1: 优先使用智谱 GLM 多模态 (glm-4.7)
    if settings.GLM_API_KEY:
        try:
            from zhipuai import ZhipuAI
            client = ZhipuAI(api_key=settings.GLM_API_KEY)
            
            def _call_glm():
                # 根据文件扩展名确定 MIME 类型
                ext = image_file.suffix.lower()
                mime_type = "image/png" if ext == ".png" else "image/jpeg"
                
                response = client.chat.completions.create(
                    model=settings.GLM_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_b64}"}},
                                {"type": "text", "text": analysis_prompt}
                            ]
                        }
                    ],
                    stream=False
                )
                return response.choices[0].message.content
            
            logger.info(f"使用智谱 GLM 多模态模型 ({settings.GLM_MODEL}) 分析 reference image...")
            result = await loop.run_in_executor(None, _call_glm)
            return _parse_json_result(result)
            
        except Exception as e:
            logger.warning(f"GLM 分析失败: {e}，尝试豆包...")
    
    # 方案2: 回退到豆包多模态 (doubao-seed-1-6-251015)
    if settings.ARK_API_KEY:
        try:
            from volcenginesdkarkruntime import Ark
            client = Ark(api_key=settings.ARK_API_KEY, base_url=settings.ARK_BASE_URL)
            
            def _call_doubao():
                response = client.chat.completions.create(
                    model=settings.ARK_VISION_MODEL,  # 豆包多模态模型
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                                {"type": "text", "text": analysis_prompt}
                            ]
                        }
                    ]
                )
                return response.choices[0].message.content
            
            logger.info(f"使用豆包多模态模型 ({settings.ARK_VISION_MODEL}) 分析 reference image...")
            result = await loop.run_in_executor(None, _call_doubao)
            return _parse_json_result(result)
            
        except Exception as e:
            logger.warning(f"豆包分析失败: {e}")
    
    logger.warning("未配置多模态 API (GLM 或 ARK)，或分析全部失败")
    return None

def _parse_json_result(result: str) -> Dict[str, Any]:
    """Helper to safely parse JSON from LLM response"""
    if not result:
        return None
        
    result_str = result.strip()
    if result_str.startswith("```json"):
        result_str = result_str[7:]
    if result_str.startswith("```"):
        result_str = result_str[3:]
    if result_str.endswith("```"):
        result_str = result_str[:-3]
    result_str = result_str.strip()
    
    try:
        return json.loads(result_str)
    except:
        return {"character": result_str, "style": None}
