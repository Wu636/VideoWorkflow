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
你是一位专业的AI短视频分镜师和导演，擅长创作爆款短视频脚本。
你的任务是根据给定的主题生成详细的分镜脚本。

【输出格式】必须是符合以下结构的有效 JSON 对象：
{
    "topic": "string",
    "scenes": [
        {
            "id": 1,
            "duration": 5,
            "narrative": "【角色名】: '台词内容'（语气/音色/情绪）",
            "visual_prompt": "详细的静态画面描述，包含：角色情绪状态（如'眼眶微红'而非'伤心'）、光影氛围、构图、风格...",
            "motion_prompt": "详细的动态描述，每个镜头只描述1-2个连贯动作，不要堆砌动作..."
        }
    ]
}

【创作铁律】
1. narrative 是角色说的台词（不是旁白！），必须包含角色名和声音描述：
   - 示例：【大橘猫】: '今天又要加班...'（无奈叹气，声音低沉疲惫）
   - 示例：【小柯基】: '这是什么？好香啊！'（兴奋上扬，声音轻快好奇）
2. 台词长度严格控制：5秒视频 = 10-15个字，绝对不能超过20个字！
3. 情绪状态要具体："眼眶微红/嘴角微扬/眉头紧锁" 比 "伤心/开心/生气" 效果好10倍
4. 动作描述要细化："向前迈步 + 挥出右爪" 比 "打架" 效果好
5. 一个镜头只做1-2个动作：太多动作会让画面很乱
6. 营造氛围感：描写光影（如"暖黄色的夕阳余晖"）
7. 视觉风格一致：所有分镜保持统一画风，不要卡通和写实混用
8. 每个分镜必须包含 duration 字段，取值 4-12 秒，根据动作复杂度和台词长度智能调整

- visual_prompt 应当非常详细，专注于视觉表现和情绪细节。
- motion_prompt 应当清晰描述动作和运镜，每镜头1-2个动作。
- narrative 必须是【角色名】: '台词'（语气），不要写旁白！
- 严禁使用 Markdown 格式，仅返回纯 JSON 字符串。
"""

    async def generate_storyboard(self, topic: str, count: int = 5, reference_image: str | None = None, template: str | None = None, include_dialogue: bool = True) -> Storyboard:
        prompt = f"请为一个关于 '{topic}' 的短视频创作分镜脚本。请精确生成 {count} 个分镜。"
        
        # 添加爆款模板指导
        if template:
            from src.video_workflow.templates import get_template_prompt_enhancement
            template_prompt = get_template_prompt_enhancement(template)
            if template_prompt:
                prompt = template_prompt + "\n\n【用户主题】" + prompt
        
        # 核心：根据是否包含台词调整 Prompt
        if include_dialogue:
            prompt += """
\n【重要！必须严格遵循的台词规则】
1. narrative 必须是【角色名】: '台词'（语气）格式
2. visual/motion 必须描述角色说话时的动作（如张嘴、肢体配合）
3. 确保画面与台词在时间线上融合，不要出现"画外音"感觉
"""
        else:
            prompt += """
\n【重要！必须严格遵循的规则：无台词模式】
1. narrative 字段**严禁**包含任何角色台词！
2. narrative 只可以写简短的动作描述、画面补充说明，或者直接留空。
3. visual_prompt 和 motion_prompt 必须侧重纯视觉叙事，通过画面和动作传达剧情，而不是靠台词。
4. 绝对不要出现角色开口说话的描述。
"""

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
你是一位专业的AI短视频分镜师和导演，擅长创作爆款短视频脚本。
你的任务是根据给定的主题和参考图生成详细的分镜脚本。

**重要**：如果提供了参考图，请仔细分析图中的角色特征（外貌、服装、风格、色彩），并在所有分镜的描述中保持这些特征的一致性。

【输出格式】必须是符合以下结构的有效 JSON 对象：
{
    "topic": "string",
    "scenes": [
        {
            "id": 1,
            "duration": 5,
            "narrative": "【角色名】: '台词内容'（语气/音色/情绪）",
            "visual_prompt": "详细的静态画面描述，包含：角色情绪状态（如'眼眶微红'而非'伤心'）、光影氛围、构图、风格...",
            "motion_prompt": "详细的动态描述，每个镜头只描述1-2个连贯动作，不要堆砌动作..."
        }
    ]
}

【创作铁律】
1. narrative 是角色说的台词（不是旁白！），必须包含角色名和声音描述：
   - 示例：【大橘猫】: '今天又要加班...'（无奈叹气，声音低沉疲惫）
   - 示例：【小柯基】: '这是什么？好香啊！'（兴奋上扬，声音轻快好奇）
2. 台词长度严格控制：5秒视频 = 10-15个字，绝对不能超过20个字！
3. 情绪状态要具体："眼眶微红/嘴角微扬/眉头紧锁" 比 "伤心/开心/生气" 效果好10倍
4. 动作描述要细化，一个镜头只做1-2个动作
5. 营造氛围感：描写光影（如"暖黄色的夕阳余晖"）
6. 视觉风格一致：所有分镜保持统一画风
7. 每个分镜必须包含 duration 字段，取值 4-12 秒，根据动作复杂度和台词长度智能调整

- visual_prompt 必须包含详细的角色描述和情绪细节，确保所有分镜中角色外观一致
- motion_prompt 应当清晰描述动作和运镜，每镜头1-2个动作
- narrative 必须是【角色名】: '台词'（语气），不要写旁白！
- 严禁使用 Markdown 格式，仅返回纯 JSON 字符串
"""

    async def generate_storyboard(self, topic: str, count: int = 5, reference_image: str | None = None, template: str | None = None, include_dialogue: bool = True) -> Storyboard:
        import asyncio
        
        # Template enhancement
        template_enhancement = ""
        if template:
            from src.video_workflow.templates import get_template_prompt_enhancement
            template_enhancement = get_template_prompt_enhancement(template)
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # 核心：根据是否包含台词调整 Prompt
        dialogue_prompt_addendum = ""
        if include_dialogue:
            dialogue_prompt_addendum = """
\n【重要！必须严格遵循的台词规则】
1. narrative 必须是【角色名】: '台词'（语气）格式
2. visual/motion 必须描述角色说话时的动作（如张嘴、肢体配合）
3. 确保画面与台词在时间线上融合，不要出现"画外音"感觉
"""
        else:
            dialogue_prompt_addendum = """
\n【重要！必须严格遵循的规则：无台词模式】
1. narrative 字段**严禁**包含任何角色台词！
2. narrative 只可以写简短的动作描述、画面补充说明，或者直接留空。
3. visual_prompt 和 motion_prompt 必须侧重纯视觉叙事，通过画面和动作传达剧情，而不是靠台词。
4. 绝对不要出现角色开口说话的描述。
"""
        
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
                "text": f"请仔细观察这张参考图中的角色特征。然后为主题 '{topic}' 创作 {count} 个分镜脚本。\n\n**关键要求**：所有分镜中的角色外貌、服装、风格必须与参考图保持一致。" + dialogue_prompt_addendum
            })
        else:
            text_prompt = f"请为主题 '{topic}' 创作 {count} 个分镜脚本。" + dialogue_prompt_addendum
            
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
你是一位专业的AI短视频分镜师和导演，擅长创作爆款短视频脚本。
你的任务是根据给定的主题生成详细的分镜脚本。

【输出格式】必须是符合以下结构的有效 JSON 对象：
{
    "topic": "string",
    "scenes": [
        {
            "id": 1,
            "duration": 5,
            "narrative": "【角色名】: '台词内容'（语气/音色/情绪）",
            "visual_prompt": "详细的静态画面描述，包含：角色情绪状态（如'眼眶微红'而非'伤心'）、光影氛围、构图、风格...",
            "motion_prompt": "详细的动态描述，每个镜头只描述1-2个连贯动作，不要堆砌动作..."
        }
    ]
}

【创作铁律】
1. narrative 是角色说的台词（不是旁白！），必须包含角色名和声音描述：
   - 示例：【大橘猫】: '今天又要加班...'（无奈叹气，声音低沉疲惫）
   - 示例：【小柯基】: '这是什么？好香啊！'（兴奋上扬，声音轻快好奇）
2. 台词长度严格控制：5秒视频 = 10-15个字，绝对不能超过20个字！
3. 情绪状态要具体："眼眶微红/嘴角微扬/眉头紧锁" 比 "伤心/开心/生气" 效果好10倍
4. 动作描述要细化，一个镜头只做1-2个动作
5. 营造氛围感：描写光影（如"暖黄色的夕阳余晖"）
6. 视觉风格一致：所有分镜保持统一画风
7. 每个分镜必须包含 duration 字段，取值 4-12 秒，根据动作复杂度和台词长度智能调整

- visual_prompt 应当非常详细，专注于视觉表现和情绪细节。
- motion_prompt 应当清晰描述动作和运镜，每镜头1-2个动作。
- narrative 必须是【角色名】: '台词'（语气），不要写旁白！
- 严禁使用 Markdown 格式，仅返回纯 JSON 字符串。
"""
        print(f"🤖 使用火山方舟 LLM: {self.model}")

    async def generate_storyboard(self, topic: str, count: int = 5, reference_image: str | None = None, template: str | None = None, include_dialogue: bool = True) -> Storyboard:
        import asyncio
        
        prompt = f"请为一个关于 '{topic}' 的短视频创作分镜脚本。请精确生成 {count} 个分镜。"
        
        # 添加爆款模板指导
        if template:
            from src.video_workflow.templates import get_template_prompt_enhancement
            template_prompt = get_template_prompt_enhancement(template)
            if template_prompt:
                prompt = template_prompt + "\n\n【用户主题】" + prompt
        
        # 核心：根据是否包含台词调整 Prompt
        if include_dialogue:
            prompt += """
\n【重要！必须严格遵循的台词规则】
1. narrative 必须是【角色名】: '台词'（语气）格式
2. visual/motion 必须描述角色说话时的动作（如张嘴、肢体配合）
3. 确保画面与台词在时间线上融合，不要出现"画外音"感觉
"""
        else:
            prompt += """
\n【重要！必须严格遵循的规则：无台词模式】
1. narrative 字段**严禁**包含任何角色台词！
2. narrative 只可以写简短的动作描述、画面补充说明，或者直接留空。
3. visual_prompt 和 motion_prompt 必须侧重纯视觉叙事，通过画面和动作传达剧情，而不是靠台词。
4. 绝对不要出现角色开口说话的描述。
"""
        
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
