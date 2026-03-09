"""
Reference image analysis module.
Uses multimodal LLMs (GLM or Doubao) to extract character description and visual style.
"""

import asyncio
import base64
import json
import logging
from pathlib import Path

from src.video_workflow.config import settings

logger = logging.getLogger(__name__)


async def analyze_reference_image(image_path: str) -> dict | None:
    """
    Analyze a reference image using multimodal LLM.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict with 'character' and 'style' keys, or None if analysis fails
    """
    image_file = Path(image_path)
    if not image_file.exists():
        logger.error(f"Reference image not found: {image_path}")
        return None
    
    with open(image_file, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    # Prompt for extracting character and style info
    analysis_prompt = """请仔细分析这张图片，提取以下两部分信息，以 JSON 格式返回：

1. character（角色外貌描述，50-100字）：
   - 包含：物种/类型、体型、毛色/肤色、五官特征、服装配饰、表情气质

2. style（视觉风格描述，20-50字）：
   - 分析图片整体视觉风格（如：3D卡通渲染、写实摄影、水彩手绘、赛璐璐动画等）
   - 包含：画面质感、光影风格、色彩特点

请严格按照以下 JSON 格式输出，不要添加其他文字：
{"character": "角色描述内容", "style": "风格描述内容"}"""
    
    loop = asyncio.get_running_loop()
    
    # Method 1: Try GLM first (better multimodal support)
    if settings.GLM_API_KEY:
        try:
            from zhipuai import ZhipuAI
            client = ZhipuAI(api_key=settings.GLM_API_KEY)
            
            def _call_glm():
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
            
            logger.info(f"Analyzing image with GLM ({settings.GLM_MODEL})...")
            result = await loop.run_in_executor(None, _call_glm)
            
            if result:
                return _parse_json_result(result)
                
        except Exception as e:
            logger.warning(f"GLM analysis failed: {e}, trying Doubao...")
    
    # Method 2: Fallback to Doubao Vision
    if settings.ARK_API_KEY:
        try:
            from volcenginesdkarkruntime import Ark
            client = Ark(api_key=settings.ARK_API_KEY, base_url=settings.ARK_BASE_URL)
            
            def _call_doubao():
                response = client.chat.completions.create(
                    model=settings.ARK_VISION_MODEL,
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
            
            logger.info(f"Analyzing image with Doubao Vision ({settings.ARK_VISION_MODEL})...")
            result = await loop.run_in_executor(None, _call_doubao)
            
            if result:
                return _parse_json_result(result)
                
        except Exception as e:
            logger.warning(f"Doubao analysis failed: {e}")
    
    logger.error("No multimodal API configured (GLM or ARK)")
    return None


def _parse_json_result(result: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    result_str = result.strip()
    
    # Remove markdown code block wrappers
    if result_str.startswith("```json"):
        result_str = result_str[7:]
    if result_str.startswith("```"):
        result_str = result_str[3:]
    if result_str.endswith("```"):
        result_str = result_str[:-3]
    result_str = result_str.strip()
    
    try:
        return json.loads(result_str)
    except json.JSONDecodeError:
        # If JSON parsing fails, treat entire result as character description
        return {"character": result_str, "style": None}
