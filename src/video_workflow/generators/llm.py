import json
import base64
from pathlib import Path
from openai import AsyncOpenAI
from zhipuai import ZhipuAI
from src.video_workflow.config import settings
from src.video_workflow.types import Storyboard, Scene
from src.video_workflow.generators.base import LLMGenerator

class DeepSeekGenerator(LLMGenerator):
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL
        )
        self.system_prompt = """
你是一位专业的短视频分镜师和导演。
你的任务是根据给定的主题生成详细的分镜脚本。
输出必须是符合以下结构的有效 JSON 对象：
{
    "topic": "string",
    "scenes": [
        {
            "id": 1,
            "narrative": "分镜的旁白或画外音文本...",
            "visual_prompt": "详细的静态画面描述，用于生成首帧图像，包含光影、构图、风格和主体细节...",
            "motion_prompt": "详细的动态描述，用于生成视频，清晰描述运镜（如推拉摇移）和主体动作..."
        }
    ]
}

- visual_prompt 应当非常详细，专注于视觉表现。
- motion_prompt 应当清晰描述动作和运镜。
- narrative (旁白) 应当简洁生动，富有感染力。
- 严禁使用 Markdown 格式，仅返回纯 JSON 字符串。
"""

    async def generate_storyboard(self, topic: str, count: int = 5, reference_image: str | None = None, template: str | None = None) -> Storyboard:
        prompt = f"请为一个关于 '{topic}' 的短视频创作分镜脚本。请精确生成 {count} 个分镜。"
        
        # 添加爆款模板指导
        if template:
            from src.video_workflow.templates import get_template_prompt_enhancement
            template_prompt = get_template_prompt_enhancement(template)
            if template_prompt:
                prompt = template_prompt + "\n\n【用户主题】" + prompt
        
        # 添加角色描述（非常重要！确保脚本和图像角色一致）
        if settings.CHARACTER_DESCRIPTION:
            prompt += f"\n\n【重要！角色设定】\n主角外貌描述：{settings.CHARACTER_DESCRIPTION}\n请在所有分镜的 visual_prompt 和 narrative 中严格使用这个角色设定，不要更改或创造新角色！"
        
        # 添加风格一致性要求
        if settings.IMAGE_STYLE:
            prompt += f"\n\n【重要！视觉风格】\n所有分镜必须保持统一的视觉风格：{settings.IMAGE_STYLE}\n请在每个 visual_prompt 中体现这种风格，不要混用卡通、写实等不同风格！"
        elif not settings.IMAGE_STYLE:
            # 即使没有指定风格，也要强调一致性
            prompt += "\n\n【重要！风格一致性】\n所有分镜的视觉风格必须保持一致！不要在某些场景使用卡通风格，某些场景使用写实风格。请选择一种统一的画风并贯穿始终。"
        
        if reference_image:
            prompt += "\n注意：DeepSeek 不支持图像输入，将忽略参考图。建议使用 GLM 或 Claude。"
        
        response = await self.client.chat.completions.create(
            model=settings.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if not content:
            raise ValueError("DeepSeek returned empty content")
            
        try:
            data = json.loads(content)
            return Storyboard(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse DeepSeek response as JSON: {e}\nContent: {content}")
        except Exception as e:
            raise ValueError(f"Failed to validate storyboard data: {e}")


class GLMGenerator(LLMGenerator):
    def __init__(self):
        if not settings.GLM_API_KEY:
            raise ValueError("GLM_API_KEY not configured")
        self.client = ZhipuAI(api_key=settings.GLM_API_KEY)
        self.system_prompt = """
你是一位专业的短视频分镜师和导演。
你的任务是根据给定的主题和参考图生成详细的分镜脚本。

**重要**：如果提供了参考图，请仔细分析图中的角色特征（外貌、服装、风格、色彩），并在所有分镜的描述中保持这些特征的一致性。

输出必须是符合以下结构的有效 JSON 对象：
{
    "topic": "string",
    "scenes": [
        {
            "id": 1,
            "narrative": "分镜的旁白或画外音文本...",
            "visual_prompt": "详细的静态画面描述，用于生成首帧图像。必须包含：角色外貌（发型、五官、服装）、场景环境、光影效果、构图细节...",
            "motion_prompt": "详细的动态描述，用于生成视频，清晰描述运镜（如推拉摇移）和主体动作..."
        }
    ]
}

- visual_prompt 必须包含详细的角色描述，确保所有分镜中角色外观一致
- motion_prompt 应当清晰描述动作和运镜
- narrative (旁白) 应当简洁生动，富有感染力
- 严禁使用 Markdown 格式，仅返回纯 JSON 字符串
"""

    async def generate_storyboard(self, topic: str, count: int = 5, reference_image: str | None = None, template: str | None = None) -> Storyboard:
        import asyncio
        
        # Template enhancement
        template_enhancement = ""
        if template:
            from src.video_workflow.templates import get_template_prompt_enhancement
            template_enhancement = get_template_prompt_enhancement(template)
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Build user message with optional image
        user_content = []
        
        if reference_image:
            # Read and encode image
            image_path = Path(reference_image)
            if not image_path.exists():
                raise FileNotFoundError(f"参考图不存在: {reference_image}")
            
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
            user_content.append({
                "type": "text",
                "text": f"请仔细观察这张参考图中的角色特征。然后为主题 '{topic}' 创作 {count} 个分镜脚本。\n\n**关键要求**：所有分镜中的角色外貌、服装、风格必须与参考图保持一致。"
            })
        else:
            text_prompt = f"请为主题 '{topic}' 创作 {count} 个分镜脚本。"
            
            # 添加角色描述和风格要求
            if settings.CHARACTER_DESCRIPTION:
                text_prompt += f"\n\n【重要！角色设定】\n主角外貌描述：{settings.CHARACTER_DESCRIPTION}\n请在所有分镜中严格使用这个角色设定。"
            
            if settings.IMAGE_STYLE:
                text_prompt += f"\n\n【重要！视觉风格】\n所有分镜必须保持统一的视觉风格：{settings.IMAGE_STYLE}\n请在每个 visual_prompt 中体现这种风格。"
            elif not settings.IMAGE_STYLE:
                text_prompt += "\n\n【重要！风格一致性】\n所有分镜的视觉风格必须保持一致！请选择一种统一的画风并贯穿始终。"
            
            user_content.append({
                "type": "text",
                "text": text_prompt
            })
        
        messages.append({"role": "user", "content": user_content})
        
        # Call GLM API (synchronous, run in executor)
        loop = asyncio.get_running_loop()
        
        def _call_glm():
            response = self.client.chat.completions.create(
                model=settings.GLM_MODEL,
                messages=messages,
            )
            return response.choices[0].message.content
        
        try:
            content = await loop.run_in_executor(None, _call_glm)
        except Exception as e:
            raise RuntimeError(f"GLM API 调用失败: {e}")
        
        if not content:
            raise ValueError("GLM returned empty content")
        
        # Parse JSON (GLM might wrap in markdown code blocks)
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            data = json.loads(content)
            return Storyboard(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse GLM response as JSON: {e}\nContent: {content}")
        except Exception as e:
            raise ValueError(f"Failed to validate storyboard data: {e}")
    
    async def analyze_reference_image(self, image_path: str) -> str | None:
        """使用 GLM 多模态能力分析参考图，生成角色描述"""
        import asyncio
        
        image_file = Path(image_path)
        if not image_file.exists():
            return None
        
        with open(image_file, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        analysis_prompt = """请仔细分析这张图片中的主体角色（人物/动物/卡通形象），生成一段详细的外貌描述。

要求：
1. 描述要具体、准确，可用于后续 AI 图像生成
2. 包含：物种/角色类型、体型、毛色/肤色、五官特征、服装配饰、表情气质
3. 描述长度约50-100字
4. 只输出描述文本，不要其他解释

示例输出格式：
一只圆润可爱的橘色猫咪，毛发蓬松柔软，戴着白色厨师帽，穿着蓝色围裙，大眼睛水汪汪的，表情憨态可掬，尾巴毛茸茸的"""
        
        loop = asyncio.get_running_loop()
        
        def _call_glm():
            response = self.client.chat.completions.create(
                model=settings.GLM_MODEL,
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
        
        try:
            result = await loop.run_in_executor(None, _call_glm)
            return result.strip() if result else None
        except Exception as e:
            print(f"GLM 图像分析失败: {e}")
            return None


class ArkLLMGenerator(LLMGenerator):
    """火山方舟托管的 LLM（豆包1.8、DeepSeek 3.2 等）"""
    
    def __init__(self):
        from volcenginesdkarkruntime import Ark
        self.client = Ark(
            api_key=settings.ARK_API_KEY,
            base_url=settings.ARK_BASE_URL
        )
        self.model = settings.ARK_LLM_MODEL
        self.system_prompt = """
你是一位专业的短视频分镜师和导演。
你的任务是根据给定的主题生成详细的分镜脚本。
输出必须是符合以下结构的有效 JSON 对象：
{
    "topic": "string",
    "scenes": [
        {
            "id": 1,
            "narrative": "分镜的旁白或画外音文本...",
            "visual_prompt": "详细的静态画面描述，用于生成首帧图像，包含光影、构图、风格和主体细节...",
            "motion_prompt": "详细的动态描述，用于生成视频，清晰描述运镜（如推拉摇移）和主体动作..."
        }
    ]
}

- visual_prompt 应当非常详细，专注于视觉表现。
- motion_prompt 应当清晰描述动作和运镜。
- narrative (旁白) 应当简洁生动，富有感染力。
- 严禁使用 Markdown 格式，仅返回纯 JSON 字符串。
"""
        print(f"🤖 使用火山方舟 LLM: {self.model}")

    async def generate_storyboard(self, topic: str, count: int = 5, reference_image: str | None = None, template: str | None = None) -> Storyboard:
        import asyncio
        
        prompt = f"请为一个关于 '{topic}' 的短视频创作分镜脚本。请精确生成 {count} 个分镜。"
        
        # 添加爆款模板指导
        if template:
            from src.video_workflow.templates import get_template_prompt_enhancement
            template_prompt = get_template_prompt_enhancement(template)
            if template_prompt:
                prompt = template_prompt + "\n\n【用户主题】" + prompt
        
        # 添加角色描述（确保脚本和图像角色一致）
        if settings.CHARACTER_DESCRIPTION:
            prompt += f"\n\n【重要！角色设定】\n主角外貌描述：{settings.CHARACTER_DESCRIPTION}\n请在所有分镜的 visual_prompt 和 narrative 中严格使用这个角色设定，不要更改或创造新角色！"
        
        # 添加风格一致性要求
        if settings.IMAGE_STYLE:
            prompt += f"\n\n【重要！视觉风格】\n所有分镜必须保持统一的视觉风格：{settings.IMAGE_STYLE}\n请在每个 visual_prompt 中体现这种风格，不要混用卡通、写实等不同风格！"
        elif not settings.IMAGE_STYLE:
            prompt += "\n\n【重要！风格一致性】\n所有分镜的视觉风格必须保持一致！不要在某些场景使用卡通风格，某些场景使用写实风格。请选择一种统一的画风并贯穿始终。"
        
        if reference_image:
            prompt += "\n注意：该模型不支持图像输入，将忽略参考图。"
        
        loop = asyncio.get_running_loop()
        
        def _call_ark():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        
        try:
            content = await loop.run_in_executor(None, _call_ark)
        except Exception as e:
            raise RuntimeError(f"火山方舟 LLM 调用失败: {e}")
        
        if not content:
            raise ValueError("火山方舟 LLM 返回空内容")
        
        # Parse JSON
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            data = json.loads(content)
            return Storyboard(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Ark LLM response as JSON: {e}\nContent: {content}")
        except Exception as e:
            raise ValueError(f"Failed to validate storyboard data: {e}")
