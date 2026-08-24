import json
import base64
import logging
from pathlib import Path
from openai import AsyncOpenAI
from zhipuai import ZhipuAI
from src.video_workflow.config import settings
from src.video_workflow.types import Storyboard
from src.video_workflow.generators.base import LLMGenerator


logger = logging.getLogger(__name__)


def _parse_json_object(content: str, provider: str) -> dict:
    normalized = content.strip()
    if normalized.startswith("```json"):
        normalized = normalized[7:]
    elif normalized.startswith("```"):
        normalized = normalized[3:]
    if normalized.endswith("```"):
        normalized = normalized[:-3]
    normalized = normalized.strip()
    try:
        value = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{provider} 返回的结构化结果不是有效 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{provider} 返回的结构化结果必须是 JSON 对象")
    return value


def _fallback_motion_prompt() -> str:
    return (
        "主体完成一个明确、克制的核心动作，动作连续且身体结构稳定；"
        "镜头保持固定或仅做极慢的单向移动，最后停留在清晰结果态。"
    )


def _normalize_scene_duration(duration: int | str | None) -> int:
    try:
        value = int(duration)
    except (TypeError, ValueError):
        value = 6
    return max(4, min(8, value))


def _build_rich_motion_prompt(base_prompt: str, duration: int) -> str:
    normalized_base = (base_prompt or "").strip().rstrip("。")
    if not normalized_base:
        normalized_base = "主体先观察环境后迅速执行核心动作"

    if "时长" in normalized_base or "秒" in normalized_base:
        return normalized_base
    return (
        f"总时长约{duration}秒。{normalized_base}。"
        "本镜头只完成这一项核心动作；人物位移和肢体幅度克制，镜头固定或仅做一次缓慢单向移动，"
        "禁止临时增加转身、走位、开关物品或第二次运镜。"
    )


def _normalize_storyboard_payload(payload: dict, include_dialogue: bool) -> dict:
    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        return payload

    for scene in scenes:
        if not isinstance(scene, dict):
            continue

        scene["duration"] = _normalize_scene_duration(scene.get("duration"))
        motion_prompt = scene.get("motion_prompt")
        if not isinstance(motion_prompt, str) or len(motion_prompt.strip()) < 12:
            motion_prompt = _fallback_motion_prompt()
        scene["motion_prompt"] = _build_rich_motion_prompt(motion_prompt, scene["duration"])

        if not include_dialogue:
            scene["narrative"] = ""
            scene["dialogue"] = ""
        elif not isinstance(scene.get("narrative"), str):
            scene["narrative"] = str(scene.get("narrative") or "")
        if include_dialogue and not isinstance(scene.get("dialogue"), str):
            scene["dialogue"] = ""
        if not isinstance(scene.get("dialogue_speaker"), str):
            scene["dialogue_speaker"] = ""
        if not scene.get("story_beat"):
            scene["story_beat"] = scene.get("visual_prompt") or scene.get("motion_prompt") or ""
        if not isinstance(scene.get("keyframe_prompt"), str) or not scene.get("keyframe_prompt", "").strip():
            scene["keyframe_prompt"] = scene.get("visual_prompt") or ""

    return payload


def _resolve_prompt_context(
    character_description: str | None = None,
    image_style: str | None = None,
) -> tuple[str | None, str | None]:
    resolved_character = character_description or settings.CHARACTER_DESCRIPTION
    resolved_style = image_style if image_style is not None else settings.IMAGE_STYLE
    return resolved_character, resolved_style


def _build_prompt_context_text(
    character_description: str | None = None,
    image_style: str | None = None,
) -> str:
    resolved_character, resolved_style = _resolve_prompt_context(character_description, image_style)
    prompt_suffix = ""

    if resolved_character:
        prompt_suffix += (
            f"\n\n【重要！角色设定】\n主角外貌描述：{resolved_character}"
            "\n请在所有分镜的 visual_prompt、keyframe_prompt 和 narrative 中严格使用这个角色设定，不要更改或创造新角色！"
        )

    if resolved_style:
        prompt_suffix += (
            f"\n\n【重要！视觉风格】\n所有分镜必须保持统一的视觉风格：{resolved_style}"
            "\n请在每个 visual_prompt 和 keyframe_prompt 中体现这种风格，不要混用卡通、写实等不同风格！"
        )
    else:
        prompt_suffix += (
            "\n\n【重要！风格一致性】\n所有分镜的视觉风格必须保持一致！"
            "不要在某些场景使用卡通风格，某些场景使用写实风格。请选择一种统一的画风并贯穿始终。"
        )

    prompt_suffix += (
        "\n\n【MiniMax H3 高质量拆镜规则】\n"
        "1. 单镜建议 4–8 秒，只允许一个核心动作；涉及说话、走动、取放物品、开门等多个动作时必须拆成不同镜头。\n"
        "2. 每镜采用固定镜头或一次极慢单向运镜，禁止为了丰富画面强行加入第二次运镜。\n"
        "3. narrative 只写画面剧情，dialogue 只写真正说出口的台词；无台词时 dialogue 必须是空字符串。\n"
        "4. dialogue_speaker 必须填写准确角色名；旁白填“旁白”；无对白时为空字符串。\n"
        "5. motion_prompt 只写当前核心动作和稳定结果态，不要机械套用起势/发展/收束三段动作。"
    )
    return prompt_suffix


def _build_user_suggestions_text(user_suggestions: str | None = None) -> str:
    suggestions = (user_suggestions or "").strip()
    if not suggestions:
        return ""
    return (
        "\n\n【用户本次分镜建议｜高优先级】\n"
        f"{suggestions}\n"
        "请把这些建议落实到镜头拆分、叙事顺序、景别、运镜、角色动作、对白和节奏中。"
        "若建议与基础剧情存在细节冲突，以用户本次建议为准，但不要遗漏故事的关键因果。"
    )


def _resolve_reference_image_url(reference_image: str | None) -> str | None:
    if not reference_image:
        return None

    first_source = reference_image.split(",")[0].strip()
    if not first_source:
        return None

    if first_source.startswith(("http://", "https://", "data:")):
        return first_source

    image_path = Path(first_source)
    if not image_path.exists() or not image_path.is_file():
        return None

    suffix = image_path.suffix.lower()
    mime_type = "image/png" if suffix == ".png" else "image/jpeg"
    with open(image_path, "rb") as file_handle:
        img_b64 = base64.b64encode(file_handle.read()).decode("utf-8")

    return f"data:{mime_type};base64,{img_b64}"

class DeepSeekGenerator(LLMGenerator):
    def __init__(self):
        if not settings.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY is not configured")
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
            "story_beat": "本镜头发生的剧情事件（不要写生成参数）",
            "narrative": "本镜头可见的剧情事件，不写台词",
            "dialogue": "真正说出口的纯台词；无台词时为空字符串",
            "dialogue_speaker": "准确发言角色名；旁白填旁白；无台词时为空字符串",
            "shot_size": "全景/中景/近景/特写",
            "camera_angle": "平视/俯拍/仰拍/过肩等",
            "lens": "镜头焦段与景深",
            "camera_motion": "固定/推/拉/摇/移/跟/环绕",
            "transition": "硬切/叠化/匹配剪辑等",
            "audio_design": "环境声、动作音效、音乐和声音情绪",
            "visual_prompt": "通用分镜视觉描述，包含整镜场景、角色、视觉叙事、光影、构图和风格...",
            "keyframe_prompt": "只描述本镜开场第一帧的静态画面：人物起始姿态、站位、表情、环境、构图、机位、光线和材质，不写运镜、动作过程、台词或声音...",
            "motion_prompt": "详细的动态描述，每个镜头只描述1-2个连贯动作，不要堆砌动作..."
        }
    ]
}

【创作铁律】
1. narrative 只写画面事件；dialogue 只写纯台词；dialogue_speaker 精确绑定唯一发言者。
2. 台词长度按时长控制：5-10秒镜头建议 10-30 字，绝对不能超过35个字！
3. 情绪状态要具体："眼眶微红/嘴角微扬/眉头紧锁" 比 "伤心/开心/生气" 效果好10倍
4. 动作描述要细化："向前迈步 + 挥出右爪" 比 "打架" 效果好
5. 一个镜头只完成一个核心动作；多动作、多地点或多人轮流发言必须拆镜，镜头固定或只做一次极慢单向运镜
6. 营造氛围感：描写光影（如"暖黄色的夕阳余晖"）
7. 视觉风格一致：所有分镜保持统一画风，不要卡通和写实混用
8. 每个分镜必须包含 duration 字段，取值 4-8 秒，根据动作复杂度和台词长度智能调整

- visual_prompt 应当详细描述整镜的通用视觉方向；keyframe_prompt 必须单独描述视频开始前的静态首帧，二者不得复制成同一段文字。
- motion_prompt 只描述当前单一主动作和稳定结果态，不要强塞三段动作。
- 发言镜头必须填写 dialogue_speaker，其他人物明确保持闭嘴和静止反应。
- 严禁使用 Markdown 格式，仅返回纯 JSON 字符串。
"""

    async def generate_storyboard(
        self,
        topic: str,
        count: int = 5,
        reference_image: str | None = None,
        template: str | None = None,
        include_dialogue: bool = True,
        character_description: str | None = None,
        image_style: str | None = None,
        user_suggestions: str | None = None,
    ) -> Storyboard:
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
3. visual_prompt、keyframe_prompt 和 motion_prompt 必须侧重纯视觉叙事，通过画面和动作传达剧情，而不是靠台词。
4. 绝对不要出现角色开口说话的描述。
"""

        prompt += _build_prompt_context_text(character_description, image_style)
        prompt += _build_user_suggestions_text(user_suggestions)
        
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
            data = _normalize_storyboard_payload(json.loads(content), include_dialogue=include_dialogue)
            return Storyboard(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse DeepSeek response as JSON: {e}\nContent: {content}")
        except Exception as e:
            raise ValueError(f"Failed to validate storyboard data: {e}")

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        reference_images: list[str] | None = None,
    ) -> dict:
        if reference_images:
            logger.info("DeepSeek 文本模型将使用参考图文件名与已有角色描述，跳过图像像素输入")
        response = await self.client.chat.completions.create(
            model=settings.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("DeepSeek 返回空内容")
        return _parse_json_object(content, "DeepSeek")

    async def revise_storyboard(self, storyboard: Storyboard, feedback: str, reference_image: str | None = None) -> Storyboard:
        current_script = storyboard.model_dump_json(indent=2)
        prompt = f"""请根据以下用户反馈，修改现有的分镜脚本。
        
【用户反馈】
{feedback}

【当前脚本】
{current_script}

请输出修改后的完整脚本（JSON格式），保持原有的 id 结构（除非需要增删分镜）。
"""
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
            data = _normalize_storyboard_payload(json.loads(content), include_dialogue=True)
            return Storyboard(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse DeepSeek response as JSON: {e}")


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
            "story_beat": "本镜头发生的剧情事件（不要写生成参数）",
            "narrative": "本镜头可见的剧情事件，不写台词",
            "dialogue": "真正说出口的纯台词；无台词时为空字符串",
            "dialogue_speaker": "准确发言角色名；旁白填旁白；无台词时为空字符串",
            "shot_size": "全景/中景/近景/特写",
            "camera_angle": "平视/俯拍/仰拍/过肩等",
            "lens": "镜头焦段与景深",
            "camera_motion": "固定/推/拉/摇/移/跟/环绕",
            "transition": "硬切/叠化/匹配剪辑等",
            "audio_design": "环境声、动作音效、音乐和声音情绪",
            "visual_prompt": "通用分镜视觉描述，包含整镜场景、角色、视觉叙事、光影、构图和风格...",
            "keyframe_prompt": "只描述本镜开场第一帧的静态画面：人物起始姿态、站位、表情、环境、构图、机位、光线和材质，不写运镜、动作过程、台词或声音...",
            "motion_prompt": "详细的动态描述，每个镜头只描述1-2个连贯动作，不要堆砌动作..."
        }
    ]
}

【创作铁律】
1. narrative 只写画面事件；dialogue 只写纯台词；dialogue_speaker 精确绑定唯一发言者。
2. 台词长度按时长控制：5-10秒镜头建议 10-30 字，绝对不能超过35个字！
3. 情绪状态要具体："眼眶微红/嘴角微扬/眉头紧锁" 比 "伤心/开心/生气" 效果好10倍
4. 一个镜头只完成一个核心动作；多动作、多地点或多人轮流发言必须拆镜，镜头固定或只做一次极慢单向运镜
5. 营造氛围感：描写光影（如"暖黄色的夕阳余晖"）
6. 视觉风格一致：所有分镜保持统一画风
7. 每个分镜必须包含 duration 字段，取值 4-8 秒，根据动作复杂度和台词长度智能调整

- visual_prompt 必须包含整镜的通用视觉方向；keyframe_prompt 必须单独描述静态开场首帧，并保持角色外观一致
- motion_prompt 只描述当前单一主动作和稳定结果态，不要强塞三段动作
- 发言镜头必须填写 dialogue_speaker，其他人物明确保持闭嘴和静止反应
- 严禁使用 Markdown 格式，仅返回纯 JSON 字符串
"""

    async def generate_storyboard(
        self,
        topic: str,
        count: int = 5,
        reference_image: str | None = None,
        template: str | None = None,
        include_dialogue: bool = True,
        character_description: str | None = None,
        image_style: str | None = None,
        user_suggestions: str | None = None,
    ) -> Storyboard:
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
3. visual_prompt、keyframe_prompt 和 motion_prompt 必须侧重纯视觉叙事，通过画面和动作传达剧情，而不是靠台词。
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
            text_prompt = (
                f"请仔细观察这张参考图中的角色特征。然后为主题 '{topic}' 创作 {count} 个分镜脚本。"
                "\n\n**关键要求**：所有分镜中的角色外貌、服装、风格必须与参考图保持一致。"
            )
            if template_enhancement:
                text_prompt = template_enhancement + "\n\n" + text_prompt

            text_prompt += dialogue_prompt_addendum
            text_prompt += _build_prompt_context_text(character_description, image_style)
            text_prompt += _build_user_suggestions_text(user_suggestions)

            user_content.append({
                "type": "text",
                "text": text_prompt
            })
        else:
            text_prompt = f"请为主题 '{topic}' 创作 {count} 个分镜脚本。" + dialogue_prompt_addendum
            if template_enhancement:
                text_prompt = template_enhancement + "\n\n【用户主题】" + text_prompt

            text_prompt += _build_prompt_context_text(character_description, image_style)
            text_prompt += _build_user_suggestions_text(user_suggestions)
            
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
            data = _normalize_storyboard_payload(json.loads(content), include_dialogue=include_dialogue)
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

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        reference_images: list[str] | None = None,
    ) -> dict:
        import asyncio

        content_items = []
        for image_path in reference_images or []:
            image_url = _resolve_reference_image_url(image_path)
            if image_url:
                content_items.append({"type": "image_url", "image_url": {"url": image_url}})
        content_items.append({"type": "text", "text": user_prompt})
        loop = asyncio.get_running_loop()

        def _call_glm():
            response = self.client.chat.completions.create(
                model=settings.GLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_items},
                ],
            )
            return response.choices[0].message.content

        content = await loop.run_in_executor(None, _call_glm)
        if not content:
            raise ValueError("GLM 返回空内容")
        return _parse_json_object(content, "GLM")

    async def revise_storyboard(self, storyboard: Storyboard, feedback: str, reference_image: str | None = None) -> Storyboard:
        import asyncio
        current_script = storyboard.model_dump_json(indent=2)
        
        # Build prompt
        prompt_text = f"""请根据以下用户反馈，修改现有的分镜脚本。
        
【用户反馈】
{feedback}

【当前脚本】
{current_script}

请输出修改后的完整脚本（JSON格式），保持原有的 id 结构（除非需要增删分镜）。
"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Handle reference image if provided (new one overrides or supplements)
        user_content = []
        if reference_image:
             image_path = Path(reference_image)
             if image_path.exists():
                with open(image_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
        
        user_content.append({"type": "text", "text": prompt_text})
        messages.append({"role": "user", "content": user_content})
        
        loop = asyncio.get_running_loop()
        def _call_glm():
            response = self.client.chat.completions.create(
                model=settings.GLM_MODEL,
                messages=messages,
            )
            return response.choices[0].message.content

        content = await loop.run_in_executor(None, _call_glm)
        
        # Parse JSON
        content = content.strip()
        if content.startswith("```json"): content = content[7:]
        if content.startswith("```"): content = content[3:]
        if content.endswith("```"): content = content[:-3]
        content = content.strip()
        
        try:
            data = _normalize_storyboard_payload(json.loads(content), include_dialogue=True)
            return Storyboard(**data)
        except Exception as e:
             raise ValueError(f"Failed to parse GLM response: {e}")


class ArkLLMGenerator(LLMGenerator):
    """火山方舟托管的 LLM（豆包1.8、DeepSeek 3.2 等）"""
    
    def __init__(self):
        if not settings.ARK_API_KEY:
            raise ValueError("ARK_API_KEY is not configured")
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
            "story_beat": "本镜头发生的剧情事件（不要写生成参数）",
            "narrative": "本镜头可见的剧情事件，不写台词",
            "dialogue": "真正说出口的纯台词；无台词时为空字符串",
            "dialogue_speaker": "准确发言角色名；旁白填旁白；无台词时为空字符串",
            "shot_size": "全景/中景/近景/特写",
            "camera_angle": "平视/俯拍/仰拍/过肩等",
            "lens": "镜头焦段与景深",
            "camera_motion": "固定/推/拉/摇/移/跟/环绕",
            "transition": "硬切/叠化/匹配剪辑等",
            "audio_design": "环境声、动作音效、音乐和声音情绪",
            "visual_prompt": "通用分镜视觉描述，包含整镜场景、角色、视觉叙事、光影、构图和风格...",
            "keyframe_prompt": "只描述本镜开场第一帧的静态画面：人物起始姿态、站位、表情、环境、构图、机位、光线和材质，不写运镜、动作过程、台词或声音...",
            "motion_prompt": "详细的动态描述，每个镜头只描述1-2个连贯动作，不要堆砌动作..."
        }
    ]
}

【创作铁律】
1. narrative 只写画面事件；dialogue 只写纯台词；dialogue_speaker 精确绑定唯一发言者。
2. 台词长度按时长控制：5-10秒镜头建议 10-30 字，绝对不能超过35个字！
3. 情绪状态要具体："眼眶微红/嘴角微扬/眉头紧锁" 比 "伤心/开心/生气" 效果好10倍
4. 一个镜头只完成一个核心动作；多动作、多地点或多人轮流发言必须拆镜，镜头固定或只做一次极慢单向运镜
5. 营造氛围感：描写光影（如"暖黄色的夕阳余晖"）
6. 视觉风格一致：所有分镜保持统一画风
7. 每个分镜必须包含 duration 字段，取值 4-8 秒，根据动作复杂度和台词长度智能调整

- visual_prompt 应当详细描述整镜的通用视觉方向；keyframe_prompt 必须单独描述视频开始前的静态首帧，二者不得复制成同一段文字。
- motion_prompt 只描述当前单一主动作和稳定结果态，不要强塞三段动作。
- 发言镜头必须填写 dialogue_speaker，其他人物明确保持闭嘴和静止反应。
- 严禁使用 Markdown 格式，仅返回纯 JSON 字符串。
"""
        print(f"🤖 使用火山方舟 LLM: {self.model}")

    async def generate_storyboard(
        self,
        topic: str,
        count: int = 5,
        reference_image: str | None = None,
        template: str | None = None,
        include_dialogue: bool = True,
        character_description: str | None = None,
        image_style: str | None = None,
        user_suggestions: str | None = None,
    ) -> Storyboard:
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
2. narrative 必须为 "" 空字符串，不要写旁白，不要写对白。
3. visual_prompt、keyframe_prompt 和 motion_prompt 必须侧重纯视觉叙事，通过画面和动作传达剧情，而不是靠台词。
4. 绝对不要出现角色开口说话的描述。
5. 每个分镜的 motion_prompt 必须非空且至少一句完整动作+运镜描述。
"""
        
        prompt += _build_prompt_context_text(character_description, image_style)
        prompt += _build_user_suggestions_text(user_suggestions)

        reference_image_url = _resolve_reference_image_url(reference_image)
        if reference_image and not reference_image_url:
            logger.warning("Reference image provided but could not be loaded for Ark multimodal input: %s", reference_image)
            prompt += "\n注意：参考图读取失败，已按纯文本方式生成。"

        if reference_image_url:
            prompt = (
                "请先仔细观察参考图中的角色外观、服饰、场景气质与色彩风格，"
                "并把这些特征稳定地体现在所有分镜中，避免角色漂移。\n\n" + prompt
            )
        
        loop = asyncio.get_running_loop()
        
        def _call_ark():
            user_content = []
            if reference_image_url:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": reference_image_url}
                })
            user_content.append({"type": "text", "text": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content}
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
            data = _normalize_storyboard_payload(json.loads(content), include_dialogue=include_dialogue)
            return Storyboard(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Ark LLM response as JSON: {e}\nContent: {content}")
        except Exception as e:
            raise ValueError(f"Failed to validate storyboard data: {e}")

    async def revise_storyboard(self, storyboard: Storyboard, feedback: str, reference_image: str | None = None) -> Storyboard:
        import asyncio
        current_script = storyboard.model_dump_json(indent=2)
        reference_image_url = _resolve_reference_image_url(reference_image)
        prompt = f"""请根据以下用户反馈，修改现有的分镜脚本。
        
【用户反馈】
{feedback}

【当前脚本】
{current_script}

请输出修改后的完整脚本（JSON格式），保持原有的 id 结构。
"""
        loop = asyncio.get_running_loop()
        def _call_ark():
            user_content = []
            if reference_image_url:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": reference_image_url}
                })
            user_content.append({"type": "text", "text": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content}
                ]
            )
            return response.choices[0].message.content
            
        content = await loop.run_in_executor(None, _call_ark)
        
        # Parse JSON
        content = content.strip()
        if content.startswith("```json"): content = content[7:]
        if content.startswith("```"): content = content[3:]
        if content.endswith("```"): content = content[:-3]
        content = content.strip()
        
        try:
            data = _normalize_storyboard_payload(json.loads(content), include_dialogue=True)
            return Storyboard(**data)
        except Exception as e:
            raise ValueError(f"Failed to parse Ark response: {e}")

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        reference_images: list[str] | None = None,
    ) -> dict:
        import asyncio

        content_items = []
        for image_path in reference_images or []:
            image_url = _resolve_reference_image_url(image_path)
            if image_url:
                content_items.append({"type": "image_url", "image_url": {"url": image_url}})
        content_items.append({"type": "text", "text": user_prompt})
        loop = asyncio.get_running_loop()

        def _call_ark():
            response = self.client.chat.completions.create(
                model=settings.ARK_VISION_MODEL if reference_images else self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_items},
                ],
            )
            return response.choices[0].message.content

        content = await loop.run_in_executor(None, _call_ark)
        if not content:
            raise ValueError("火山方舟 LLM 返回空内容")
        return _parse_json_object(content, "火山方舟")
