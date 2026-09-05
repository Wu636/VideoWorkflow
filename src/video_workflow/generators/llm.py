import json
import base64
import hashlib
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from openai import AsyncOpenAI
from zhipuai import ZhipuAI
from src.video_workflow.config import settings
from src.video_workflow.speech_budget import (
    fit_voice_event_payloads,
)
from src.video_workflow.types import Storyboard
from src.video_workflow.generators.base import LLMGenerator


logger = logging.getLogger(__name__)


COMPACT_STORYBOARD_SYSTEM_PROMPT = """
你是短视频分镜导演。根据剧本、角色档案、风格和用户建议，输出可直接生产的紧凑 JSON；只返回 JSON，不要 Markdown 或解释。
结构：
{"topic":"", "scenes":[{"duration":6,"event":"本镜完整可见事件","opening_state":"0 秒静态起始状态","dialogue":"纯台词或空字符串","dialogue_speaker":"发言角色名/旁白/空字符串","character_names":["本镜可辨认角色完整名"],"shot_size":"中景","camera_angle":"平视","lens":"50mm","camera_motion":"固定或一种主要运镜","visual_beats":[{"start_seconds":0,"end_seconds":3,"purpose":"开场钩子/推进/结果","subject_action":"一个主动作","environment_action":"环境反馈","shot_size":"","camera_angle":"","camera_motion":"","sound_cue":""}],"voice_events":[{"kind":"character/system_vo/narration/offscreen","speaker_name":"","text":"","start_seconds":0.5,"end_seconds":2.5,"lip_sync":false}]}]}
每个镜头只保留一个 event 和一个 opening_state；visual_beats 从 0 秒连续覆盖到 duration，每 2–4 秒出现新的可见变化，形成触发、执行、反馈、结果的动作弧线。镜头时长 4–15 秒，严格按用户要求的镜头数输出。
只在确有需要时输出 transition、audio_design、text_policy；不要输出 visual_prompt、keyframe_prompt、motion_prompt、story_beat、narrative、id 等冗余字段，这些字段由系统本地编译。不要在 beats 重复整镜的景别、角度和运镜，留空表示继承镜头级设置。
所有声音统一以 voice_events 为唯一时序来源；dialogue 仅为兼容字段，已有 voice_events 时保持空字符串，避免重复输出同一句话。角色声音写 character 并绑定唯一 speaker_name，旁白/系统播报/画外音使用对应 kind 且 lip_sync=false。整镜所有可朗读文字合计按每秒 3.4 个中文字符控制：5 秒最多 17 字、10 秒最多 34 字、15 秒最多 51 字；声音最多占镜头约 85%，开头和结尾必须留出画面反应与环境声，不安排重叠人声。长规则、旁白和系统播报只保留推动本镜的关键信息，不逐条朗读设定。character_names 只列出当前镜头真正可辨认的人物，并逐字沿用角色档案中的名称。首帧 opening_state 只写静态状态，不写未来动作、声音或字幕；画面文字留给后期叠加。保持角色身份、服装、场景和统一风格连续。
""".strip()


def _coerce_openlux_text(value: Any) -> str:
    """Flatten OpenAI/Claude-compatible text blocks into one string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_coerce_openlux_text(item) for item in value)
    if isinstance(value, dict):
        if "text" in value:
            return _coerce_openlux_text(value.get("text"))
        if "content" in value:
            return _coerce_openlux_text(value.get("content"))
        return ""
    text = getattr(value, "text", None)
    if text is not None:
        return _coerce_openlux_text(text)
    content = getattr(value, "content", None)
    if content is not None and content is not value:
        return _coerce_openlux_text(content)
    return str(value) if isinstance(value, (int, float, bool)) else ""


def _parse_json_object(content: str, provider: str) -> dict:
    normalized = content.strip()
    if normalized.startswith("```json"):
        normalized = normalized[7:]
    elif normalized.startswith("```"):
        normalized = normalized[3:]
    if normalized.endswith("```"):
        normalized = normalized[:-3]
    normalized = normalized.strip()
    # Some Claude/GPT-compatible endpoints may prepend a short explanation even
    # when instructed to return JSON. Recover the outermost JSON object without
    # weakening the object-only validation below.
    if not normalized.startswith("{"):
        start = normalized.find("{")
        end = normalized.rfind("}")
        if start >= 0 and end > start:
            normalized = normalized[start:end + 1]
    try:
        value = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{provider} 返回的结构化结果不是有效 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{provider} 返回的结构化结果必须是 JSON 对象")
    return value


def _fallback_motion_prompt() -> str:
    return (
        "主体从首帧状态立即响应事件，连续完成触发、执行与结果三个动作阶段；"
        "环境同步产生可见反馈，结尾落在清晰且与下一镜可衔接的结果态。"
    )


def _normalize_scene_duration(duration: int | str | None) -> int:
    try:
        value = int(duration)
    except (TypeError, ValueError):
        value = 6
    return max(4, min(15, value))


_EXTERNAL_VOICE_LABELS = {
    "旁白": "narration",
    "旁白配音": "narration",
    "画外音": "offscreen",
    "系统": "system_vo",
    "系统vo": "system_vo",
    "系统播报": "system_vo",
    "系统语音": "system_vo",
    "广播": "offscreen",
}


def _voice_kind_for_label(label: str) -> str:
    normalized = re.sub(r"[\s（）()【】\[\]：:]", "", (label or "").strip()).casefold()
    for marker, kind in _EXTERNAL_VOICE_LABELS.items():
        if marker.casefold() in normalized:
            return kind
    return "character"


def _float_in_range(value: Any, default: float, maximum: float) -> float:
    try:
        return max(0.0, min(maximum, float(value)))
    except (TypeError, ValueError):
        return default


def _normalize_visual_beats(scene: dict, duration: int) -> list[dict]:
    raw_beats = scene.get("visual_beats")
    beats: list[dict] = []
    if isinstance(raw_beats, list):
        for index, item in enumerate(raw_beats):
            if not isinstance(item, dict):
                continue
            start = _float_in_range(
                item.get("start_seconds", item.get("start")),
                index * duration / max(1, len(raw_beats)),
                float(duration),
            )
            end = _float_in_range(
                item.get("end_seconds", item.get("end")),
                (index + 1) * duration / max(1, len(raw_beats)),
                float(duration),
            )
            if end <= start:
                continue
            beats.append(
                {
                    "start_seconds": round(start, 2),
                    "end_seconds": round(end, 2),
                    "purpose": str(item.get("purpose") or "").strip(),
                    "subject_action": str(
                        item.get("subject_action") or item.get("action") or ""
                    ).strip(),
                    "environment_action": str(
                        item.get("environment_action") or item.get("environment") or ""
                    ).strip(),
                    "shot_size": str(item.get("shot_size") or scene.get("shot_size") or "").strip(),
                    "camera_angle": str(item.get("camera_angle") or scene.get("camera_angle") or "").strip(),
                    "camera_motion": str(item.get("camera_motion") or scene.get("camera_motion") or "固定镜头").strip(),
                    "sound_cue": str(item.get("sound_cue") or "").strip(),
                }
            )
    if beats:
        beats.sort(key=lambda item: (item["start_seconds"], item["end_seconds"]))
        return beats[:6]

    motion = str(scene.get("motion_prompt") or _fallback_motion_prompt()).strip()
    motion = re.sub(
        r"^总时长\s*(?:约|大约)?\s*\d+(?:\.\d+)?\s*秒[。.]?",
        "",
        motion,
    ).strip()
    clauses = [
        item.strip(" ，。；")
        for item in re.split(r"(?:随后|然后|接着|继而|最终|最后|；|。)", motion)
        if item.strip(" ，。；")
    ]
    target_count = max(2, min(5, math.ceil(duration / 3.0)))
    generic_stages = [
        "主体从首帧状态立即响应触发事件",
        "主体动作发生清晰的方向、姿态或空间变化",
        "环境或道具对主体动作产生同步反馈",
        "动作后果进一步显现并推动剧情",
        "主体落在清晰结果态并给下一镜留下衔接点",
    ]
    actions = clauses[:target_count]
    while len(actions) < target_count:
        actions.append(generic_stages[len(actions)])
    return [
        {
            "start_seconds": round(index * duration / target_count, 2),
            "end_seconds": round((index + 1) * duration / target_count, 2),
            "purpose": ("开场钩子" if index == 0 else "结果落点" if index == target_count - 1 else "推进事件"),
            "subject_action": action,
            "environment_action": "",
            "shot_size": str(scene.get("shot_size") or "").strip(),
            "camera_angle": str(scene.get("camera_angle") or "").strip(),
            "camera_motion": str(scene.get("camera_motion") or "固定镜头").strip(),
            "sound_cue": "",
        }
        for index, action in enumerate(actions)
    ]


def _normalize_voice_events(scene: dict, duration: int, include_dialogue: bool) -> list[dict]:
    if not include_dialogue:
        return []
    raw_events = scene.get("voice_events")
    events: list[dict] = []
    if isinstance(raw_events, list):
        for item in raw_events:
            if not isinstance(item, dict) or not str(item.get("text") or "").strip():
                continue
            speaker_name = str(item.get("speaker_name") or item.get("speaker") or "").strip()
            requested_kind = str(item.get("kind") or "").strip()
            kind = requested_kind if requested_kind in {"character", "system_vo", "narration", "offscreen"} else _voice_kind_for_label(speaker_name)
            start = _float_in_range(item.get("start_seconds", item.get("start")), 0.35, float(duration))
            end = _float_in_range(item.get("end_seconds", item.get("end")), min(float(duration), start + 3.0), float(duration))
            if end <= start:
                end = min(float(duration), start + 0.5)
            events.append(
                {
                    "kind": kind,
                    "speaker_name": speaker_name,
                    "text": str(item.get("text") or "").strip(),
                    "start_seconds": round(start, 2),
                    "end_seconds": round(end, 2),
                    "lip_sync": bool(item.get("lip_sync", kind == "character")) if kind == "character" else False,
                }
            )
    if events:
        return fit_voice_event_payloads(events, duration)

    dialogue = str(scene.get("dialogue") or "").strip()
    if not dialogue:
        return []
    speaker_name = str(scene.get("dialogue_speaker") or "旁白").strip()
    kind = _voice_kind_for_label(speaker_name)
    return fit_voice_event_payloads([
        {
            "kind": kind,
            "speaker_name": speaker_name,
            "text": dialogue,
            "start_seconds": 0.35,
            "end_seconds": min(float(duration), max(1.5, len(dialogue) / 3.7 + 0.2)),
            "lip_sync": kind == "character",
        }
    ], duration)


def _build_rich_motion_prompt(base_prompt: str, duration: int) -> str:
    normalized_base = (base_prompt or "").strip().rstrip("。")
    if not normalized_base:
        normalized_base = _fallback_motion_prompt().rstrip("。")

    legacy_limits = (
        r"本镜头只完成这一项核心动作[；，,。]*",
        r"人物位移和肢体幅度克制[；，,。]*",
        r"镜头固定或仅做一次缓慢单向移动[；，,。]*",
        r"禁止临时增加转身、走位、开关物品或第二次运镜[；，,。]*",
        r"镜头保持固定或仅做极慢的单向移动[；，,。]*",
    )
    for pattern in legacy_limits:
        normalized_base = re.sub(pattern, "", normalized_base).strip("；，,。 ")

    if "时长" in normalized_base or "秒" in normalized_base:
        return normalized_base
    return f"总时长约{duration}秒。{normalized_base}。"


def _normalize_storyboard_payload(payload: dict, include_dialogue: bool) -> dict:
    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        return payload

    payload["topic"] = str(payload.get("topic") or "")
    for index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue

        try:
            scene["id"] = int(scene.get("id") or index)
        except (TypeError, ValueError):
            scene["id"] = index
        scene["duration"] = _normalize_scene_duration(scene.get("duration"))
        event = str(
            scene.get("event")
            or scene.get("story_beat")
            or scene.get("narrative")
            or scene.get("visual_prompt")
            or scene.get("motion_prompt")
            or ""
        ).strip()
        opening_state = str(
            scene.get("opening_state")
            or scene.get("keyframe_prompt")
            or scene.get("visual_prompt")
            or event
        ).strip()
        scene["event"] = event
        scene["opening_state"] = opening_state
        motion_prompt = scene.get("motion_prompt")
        if not isinstance(motion_prompt, str) or len(motion_prompt.strip()) < 12:
            beat_actions = [
                str(item.get("subject_action") or item.get("action") or "").strip()
                for item in (scene.get("visual_beats") or [])
                if isinstance(item, dict) and str(item.get("subject_action") or item.get("action") or "").strip()
            ]
            motion_prompt = "；".join(beat_actions) if beat_actions else _fallback_motion_prompt()
        scene["motion_prompt"] = _build_rich_motion_prompt(motion_prompt, scene["duration"])
        scene["visual_beats"] = _normalize_visual_beats(scene, scene["duration"])
        scene["voice_events"] = _normalize_voice_events(scene, scene["duration"], include_dialogue)
        text_policy = str(scene.get("text_policy") or "post_overlay").strip()
        scene["text_policy"] = text_policy if text_policy in {"none", "post_overlay", "reference_locked"} else "post_overlay"

        if not include_dialogue:
            scene["narrative"] = ""
            scene["dialogue"] = ""
            scene["dialogue_speaker"] = ""
        else:
            scene["narrative"] = str(scene.get("narrative") or event)
        if include_dialogue and not isinstance(scene.get("dialogue"), str):
            scene["dialogue"] = ""
        if not isinstance(scene.get("dialogue_speaker"), str):
            scene["dialogue_speaker"] = ""
        character_names = scene.get("character_names")
        if isinstance(character_names, str):
            character_names = [
                item.strip()
                for item in character_names.replace("，", ",").replace("、", ",").split(",")
                if item.strip()
            ]
        elif not isinstance(character_names, list):
            character_names = []
        scene["character_names"] = [str(item).strip() for item in character_names if str(item).strip()]
        if not scene.get("story_beat"):
            scene["story_beat"] = event
        if not isinstance(scene.get("keyframe_prompt"), str) or not scene.get("keyframe_prompt", "").strip():
            scene["keyframe_prompt"] = opening_state
        # The local compiler owns the generic visual prompt. Keep an empty
        # legacy field here so persisted models remain backward compatible.
        if not isinstance(scene.get("visual_prompt"), str):
            scene["visual_prompt"] = ""

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
            "\n请在所有 scenes 的 event、opening_state、visual_beats 和 voice_events 中严格使用这个角色设定，不要更改或创造新角色。"
        )

    if resolved_style:
        prompt_suffix += (
            f"\n\n【重要！视觉风格】\n所有分镜必须保持统一的视觉风格：{resolved_style}"
            "\n请在 opening_state、visual_beats 的环境与光线中体现这种风格，不要混用不同画风。"
        )
    else:
        prompt_suffix += (
            "\n\n【重要！风格一致性】\n所有分镜的视觉风格必须保持一致！"
            "不要在某些场景使用卡通风格，某些场景使用写实风格。请选择一种统一的画风并贯穿始终。"
        )

    prompt_suffix += (
        "\n\n【AI 视频导演节拍规则】\n"
        "1. duration 必须按本项目真实目标填写 4–15 秒；visual_beats 必须从 0 秒连续覆盖到 duration，每 2–4 秒发生一次新的可见变化。\n"
        "2. 长镜头要有完整动作弧线：触发、执行、环境反馈、后果和结果；一个 visual beat 内只安排一个主动作与一种主要运镜，但整镜允许多个连续节拍。\n"
        "3. 前 0–2 秒必须有明确视觉钩子；最后一个节拍必须形成可读结果态或下一镜衔接点，禁止用静止等待填时长。\n"
        "4. event 只写本镜完整可见事件，opening_state 只写 0 秒静态状态；dialogue 只写真正说出口的台词；voice_events 必须明确 kind、speaker_name、起止时间和 lip_sync。\n"
        "5. 系统播报、旁白和画外音不是画面角色：kind 分别使用 system_vo、narration 或 offscreen，lip_sync=false；只有 character 声音允许口型同步。\n"
        "6. character_names 必须列出本镜所有可辨认角色，并逐字使用角色设定中的完整名称；不得以泛称替代或凭空新增可辨认人物。\n"
        "7. opening_state 只描述 0 秒静态画面，不写未来动作、运镜、声音或时长；默认 text_policy=post_overlay，画面中的字幕、标题、UI 文案均留给后期叠加。\n"
        "8. 具体时序、景别变化、环境反馈、单段运镜和声音卡点写入 visual_beats；镜头级景别、角度和运镜只写一次，beat 中相同值留空。\n"
        "9. 声音密度使用硬预算：整镜所有 voice_events 的可朗读文字合计不超过 duration×3.4 个中文字符（5秒17字、10秒34字、15秒51字）；按正常语速编排，声音占用不超过约85%，前后留出反应和环境声。\n"
        "10. voice_events 是唯一声音时序来源；已有 voice_events 时 dialogue 留空，同一句话只出现一次。系统规则或长旁白压缩成推动本镜的1–3条关键信息，15秒镜头最多4个声音事件，事件不得重叠。"
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
            "character_names": ["本镜所有可辨认角色的完整角色名"],
            "shot_size": "全景/中景/近景/特写",
            "camera_angle": "平视/俯拍/仰拍/过肩等",
            "lens": "镜头焦段与景深",
            "camera_motion": "固定/推/拉/摇/移/跟/环绕",
            "transition": "硬切/叠化/匹配剪辑等",
            "audio_design": "环境声、动作音效、音乐和声音情绪",
            "visual_prompt": "通用分镜视觉描述，包含整镜场景、角色、视觉叙事、光影、构图和风格...",
            "keyframe_prompt": "只描述本镜开场第一帧的静态画面：人物起始姿态、站位、表情、环境、构图、机位、光线和材质，不写运镜、动作过程、台词或声音...",
            "motion_prompt": "整镜动作弧线的简要总述...",
            "visual_beats": [{"start_seconds": 0, "end_seconds": 3, "purpose": "开场钩子/推进/结果", "subject_action": "本段唯一主动作", "environment_action": "环境或道具反馈", "shot_size": "本段景别", "camera_angle": "本段角度", "camera_motion": "本段唯一主要运镜", "sound_cue": "本段声音卡点"}],
            "voice_events": [{"kind": "character/system_vo/narration/offscreen", "speaker_name": "角色名或声音来源", "text": "纯台词", "start_seconds": 0.5, "end_seconds": 3.5, "lip_sync": false}],
            "text_policy": "post_overlay"
        }
    ]
}

【创作铁律】
1. narrative 只写画面事件；dialogue 只写纯台词；dialogue_speaker 精确绑定唯一发言者。
2. 台词长度按时长控制：5-10秒镜头建议 10-30 字，绝对不能超过35个字！
3. 情绪状态要具体："眼眶微红/嘴角微扬/眉头紧锁" 比 "伤心/开心/生气" 效果好10倍
4. 动作描述要细化："向前迈步 + 挥出右爪" 比 "打架" 效果好
5. 每镜具备完整动作弧线；每个 visual beat 只完成一个主动作并使用一种主要运镜，整镜每 2–4 秒产生新的可见变化
6. 营造氛围感：描写光影（如"暖黄色的夕阳余晖"）
7. 视觉风格一致：所有分镜保持统一画风，不要卡通和写实混用
8. 每个分镜必须包含 duration 字段，取值 4–15 秒，并让 visual_beats 从 0 秒连续覆盖到 duration

- visual_prompt 应当详细描述整镜的通用视觉方向；keyframe_prompt 必须单独描述视频开始前的静态首帧，二者不得复制成同一段文字。
- motion_prompt 总结动作弧线，visual_beats 负责逐段动作、环境反馈、景别、运镜和声音卡点。
- keyframe_prompt 默认不含可读文字；标题、字幕、对白字卡和 UI 文案使用 post_overlay 后期叠加。
- 发言镜头必须填写 dialogue_speaker，其他人物明确保持闭嘴和静止反应。
- 严禁使用 Markdown 格式，仅返回纯 JSON 字符串。
"""

        self.system_prompt = COMPACT_STORYBOARD_SYSTEM_PROMPT

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
1. event 只写可见事件，dialogue 只写说出口的纯台词，二者不得互相复制
2. character 台词需要匹配口型；系统播报、旁白、画外音必须写入 voice_events 且 lip_sync=false
3. 每个声音事件填写准确起止时间，并与对应 visual beat 的画面反应对齐
"""
        else:
            prompt += """
\n【重要！必须严格遵循的规则：无台词模式】
1. event 字段只写简短的可见动作和画面事件，不包含任何角色台词。
2. opening_state 只描述静态首帧状态；通过 event 和 visual_beats 传达剧情，而不是靠台词。
3. dialogue、voice_events 只在确有声音时填写。
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


class OpenLuxGenerator(LLMGenerator):
    """OpenLux OpenAI-compatible GPT / Claude / Gemini multimodal adapter."""

    def __init__(self):
        if not settings.OPENLUX_API_KEY:
            raise ValueError("OPENLUX_API_KEY 未配置")
        self.client = AsyncOpenAI(
            api_key=settings.OPENLUX_API_KEY,
            base_url=settings.OPENLUX_BASE_URL.rstrip("/"),
            timeout=httpx.Timeout(
                float(settings.OPENLUX_REQUEST_TIMEOUT_SECONDS),
                connect=min(30.0, float(settings.OPENLUX_REQUEST_TIMEOUT_SECONDS)),
            ),
            # Never retry a paid generation invisibly. A user retry must be an
            # explicit new action after the previous result/error is visible.
            max_retries=0,
        )
        self.model = settings.OPENLUX_MODEL
        self.vision_model = settings.OPENLUX_VISION_MODEL or settings.OPENLUX_MODEL
        self.last_response_path: Path | None = None
        self.system_prompt = """
你是一位专业影视编剧、分镜导演和 AI 视频生成提示词工程师。请把用户的剧情与制作约束转成可直接生产的分镜脚本。

只返回一个有效 JSON 对象，禁止 Markdown、注释和前后解释：
{
  "topic": "主题",
  "scenes": [{
    "id": 1,
    "duration": 6,
    "story_beat": "本镜头承担的完整剧情事件",
    "narrative": "镜头中可见的剧情，不包含说出口的台词",
    "dialogue": "纯台词；无台词为空字符串",
    "dialogue_speaker": "唯一发言角色名；旁白填旁白；无台词为空字符串",
    "character_names": ["本镜所有可辨认角色的完整角色名"],
    "shot_size": "景别",
    "camera_angle": "机位与角度",
    "lens": "焦段、景深",
    "camera_motion": "固定或一种明确运镜",
    "transition": "转场",
    "audio_design": "对白、环境声、动作音效与音乐",
    "visual_prompt": "整镜视觉方向、场景、角色、光影、构图与统一风格",
    "keyframe_prompt": "只描述开场第一帧的静态画面，不写动作过程、声音或运镜",
    "motion_prompt": "整镜动作弧线总述",
    "visual_beats": [{"start_seconds": 0, "end_seconds": 3, "purpose": "开场钩子/推进/结果", "subject_action": "本段唯一主动作", "environment_action": "环境反馈", "shot_size": "本段景别", "camera_angle": "本段角度", "camera_motion": "本段唯一主要运镜", "sound_cue": "声音卡点"}],
    "voice_events": [{"kind": "character/system_vo/narration/offscreen", "speaker_name": "角色名或声音来源", "text": "纯台词", "start_seconds": 0.5, "end_seconds": 3.5, "lip_sync": false}],
    "text_policy": "post_overlay"
  }]
}

硬性规则：
1. 用户指定 N 个镜头时必须从剧情层面重新规划为恰好 N 镜，完整覆盖起因、发展、转折和结果；不得先生成更多镜头再裁剪，也不得用空镜头补数。
2. 单镜 4–15 秒并具备完整动作弧线；visual_beats 从 0 秒覆盖到 duration，每 2–4 秒出现新的可见变化，每个 beat 只含一个主动作和一种主要运镜。
3. narrative 与 dialogue 分离；dialogue_speaker 必须唯一、准确，非发言者保持闭嘴，只做明确的倾听反应。
4. character_names 必须列出所有可辨认人物并始终使用相同角色名，不得临时改成泛称或凭空新增人物。
5. 每个 visual beat 只指定一种机位和一种主要运镜；keyframe_prompt、visual_prompt、motion_prompt、visual_beats 各司其职，不得互相复制。
6. 所有镜头严格遵守用户提供的角色参考图、场景资产和风格参考，保持人物、服装、空间结构、色彩、材质和光线连续。
7. 系统播报、旁白与画外音必须使用非 character 的 voice kind 且 lip_sync=false；默认 text_policy=post_overlay，首帧与生成视频内不生成字幕、标题或 UI 文案。
""".strip()

        self.system_prompt = COMPACT_STORYBOARD_SYSTEM_PROMPT

    @staticmethod
    def _content(text: str, reference_images: list[str] | None = None) -> list[dict]:
        items: list[dict] = []
        for source in reference_images or []:
            image_url = _resolve_reference_image_url(source)
            if image_url:
                items.append({"type": "image_url", "image_url": {"url": image_url}})
        items.append({"type": "text", "text": text})
        return items

    def _save_raw_response(
        self,
        *,
        model: str,
        content: str,
        system_prompt: str,
        user_prompt: str,
        image_count: int,
    ) -> Path | None:
        self.last_response_path = None
        if not settings.OPENLUX_SAVE_RAW_RESPONSES:
            return None
        try:
            directory = Path(settings.OUTPUT_DIR) / "ai_responses"
            directory.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            destination = directory / f"openlux-{timestamp}-{uuid4().hex[:8]}.json"
            temporary = destination.with_suffix(".tmp")
            payload = {
                "provider": "openlux",
                "model": model,
                "received_at": datetime.now(timezone.utc).isoformat(),
                "image_count": image_count,
                "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
                "user_prompt_sha256": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
                "content": content,
            }
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(destination)
            self.last_response_path = destination.resolve()
            logger.info("OpenLux 原始响应已保存: %s", self.last_response_path)
            return self.last_response_path
        except OSError:
            logger.exception("保存 OpenLux 原始响应失败")
            return None

    def _raw_result_hint(self) -> str:
        if self.last_response_path:
            return f"；已付费的原始结果保存在 {self.last_response_path}"
        return ""

    async def _chat(self, *, system_prompt: str, user_prompt: str, reference_images: list[str] | None = None) -> str:
        content_items = self._content(user_prompt, reference_images)
        has_images = any(item.get("type") == "image_url" for item in content_items)
        model = self.vision_model if has_images else self.model
        use_stream = bool(settings.OPENLUX_STREAM)
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_items},
                ],
                stream=use_stream,
            )
        except Exception as exc:
            raise RuntimeError(f"OpenLux API 调用失败（模型 {model}）: {exc}") from exc
        if use_stream:
            chunks: list[str] = []
            try:
                async for chunk in response:
                    choices = getattr(chunk, "choices", None) or []
                    for choice in choices:
                        delta = getattr(choice, "delta", None)
                        chunks.append(_coerce_openlux_text(getattr(delta, "content", None)))
            except Exception as exc:
                raise RuntimeError(f"OpenLux 流式结果接收失败（模型 {model}）: {exc}") from exc
            content = "".join(chunks)
        else:
            choices = getattr(response, "choices", None) or []
            message = getattr(choices[0], "message", None) if choices else None
            content = _coerce_openlux_text(getattr(message, "content", None))
        if not content:
            raise ValueError(f"OpenLux 返回空内容（模型 {model}）")
        self._save_raw_response(
            model=model,
            content=content,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_count=sum(item.get("type") == "image_url" for item in content_items),
        )
        return content

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
        prompt = (
            f"请为以下项目创作恰好 {count} 个分镜。必须重新组织完整剧情使其天然适配 {count} 镜，"
            "不要生成多余镜头，不要事后截断。\n\n【项目剧情/主题】\n"
            f"{topic}"
        )
        if template:
            from src.video_workflow.templates import get_template_prompt_enhancement
            enhancement = get_template_prompt_enhancement(template)
            if enhancement:
                prompt = f"{enhancement}\n\n{prompt}"
        if include_dialogue:
            prompt += (
                "\n\n【对白要求】dialogue 只写真正说出口的台词，dialogue_speaker 精确填写唯一发言者；"
                "画面动作必须匹配发言者，其他角色闭嘴并保持自然倾听。"
            )
        else:
            prompt += (
                "\n\n【无台词模式】每镜 event 只写可见动作；dialogue 与 dialogue_speaker 必须为空字符串；"
                "不得描述人物开口说话。"
            )
        prompt += _build_prompt_context_text(character_description, image_style)
        prompt += _build_user_suggestions_text(user_suggestions)
        references = [reference_image] if reference_image else None
        content = await self._chat(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            reference_images=references,
        )
        try:
            payload = _parse_json_object(content, "OpenLux")
        except ValueError as exc:
            raise ValueError(f"{exc}{self._raw_result_hint()}") from exc
        payload = _normalize_storyboard_payload(payload, include_dialogue=include_dialogue)
        try:
            return Storyboard(**payload)
        except Exception as exc:
            raise ValueError(
                f"OpenLux 分镜结构校验失败: {exc}{self._raw_result_hint()}"
            ) from exc

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        reference_images: list[str] | None = None,
    ) -> dict:
        content = await self._chat(
            system_prompt=system_prompt + "\n只返回有效 JSON 对象，不要 Markdown 或额外解释。",
            user_prompt=user_prompt,
            reference_images=reference_images,
        )
        try:
            return _parse_json_object(content, "OpenLux")
        except ValueError as exc:
            raise ValueError(f"{exc}{self._raw_result_hint()}") from exc

    async def analyze_reference_image(self, image_path: str) -> str | None:
        if not _resolve_reference_image_url(image_path):
            return None
        prompt = """分析附图中的主体人物或角色，输出一段可用于跨镜头一致性生成的中文外貌圣经。
包含年龄感、脸型、五官、肤色、发型、体型、服装版型/材质/颜色、配饰、气质和最稳定的可识别特征。
区分永久身份特征与当前服装，不把姿势、镜头角度和背景当成人物特征。只输出描述正文。"""
        return (await self._chat(
            system_prompt="你是影视角色造型师和人物一致性分析师。",
            user_prompt=prompt,
            reference_images=[image_path],
        )).strip()

    async def revise_storyboard(
        self,
        storyboard: Storyboard,
        feedback: str,
        reference_image: str | None = None,
    ) -> Storyboard:
        expected_count = len(storyboard.scenes)
        prompt = f"""根据用户建议重做以下分镜。保持完整叙事，并输出恰好 {expected_count} 个镜头；
不得生成更多镜头后裁剪，也不得以重复或空镜补数。需要改变节奏时，应重新分配每镜剧情职责。

【用户建议】
{feedback}

【当前分镜】
{storyboard.model_dump_json(indent=2)}
"""
        content = await self._chat(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            reference_images=[reference_image] if reference_image else None,
        )
        payload = _normalize_storyboard_payload(
            _parse_json_object(content, "OpenLux"),
            include_dialogue=True,
        )
        try:
            return Storyboard(**payload)
        except Exception as exc:
            raise ValueError(f"OpenLux 分镜修改结果校验失败: {exc}") from exc


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
            "character_names": ["本镜所有可辨认角色的完整角色名"],
            "shot_size": "全景/中景/近景/特写",
            "camera_angle": "平视/俯拍/仰拍/过肩等",
            "lens": "镜头焦段与景深",
            "camera_motion": "固定/推/拉/摇/移/跟/环绕",
            "transition": "硬切/叠化/匹配剪辑等",
            "audio_design": "环境声、动作音效、音乐和声音情绪",
            "visual_prompt": "通用分镜视觉描述，包含整镜场景、角色、视觉叙事、光影、构图和风格...",
            "keyframe_prompt": "只描述本镜开场第一帧的静态画面：人物起始姿态、站位、表情、环境、构图、机位、光线和材质，不写运镜、动作过程、台词或声音...",
            "motion_prompt": "整镜动作弧线的简要总述...",
            "visual_beats": [{"start_seconds": 0, "end_seconds": 3, "purpose": "开场钩子/推进/结果", "subject_action": "本段唯一主动作", "environment_action": "环境反馈", "shot_size": "本段景别", "camera_angle": "本段角度", "camera_motion": "本段唯一主要运镜", "sound_cue": "声音卡点"}],
            "voice_events": [{"kind": "character/system_vo/narration/offscreen", "speaker_name": "角色名或声音来源", "text": "纯台词", "start_seconds": 0.5, "end_seconds": 3.5, "lip_sync": false}],
            "text_policy": "post_overlay"
        }
    ]
}

【创作铁律】
1. narrative 只写画面事件；dialogue 只写纯台词；dialogue_speaker 精确绑定唯一发言者。
2. 台词长度按时长控制：5-10秒镜头建议 10-30 字，绝对不能超过35个字！
3. 情绪状态要具体："眼眶微红/嘴角微扬/眉头紧锁" 比 "伤心/开心/生气" 效果好10倍
4. 每镜具备完整动作弧线；每个 visual beat 只完成一个主动作并使用一种主要运镜，整镜每 2–4 秒产生新的可见变化
5. 营造氛围感：描写光影（如"暖黄色的夕阳余晖"）
6. 视觉风格一致：所有分镜保持统一画风
7. 每个分镜必须包含 duration 字段，取值 4–15 秒，并让 visual_beats 从 0 秒连续覆盖到 duration

- visual_prompt 必须包含整镜的通用视觉方向；keyframe_prompt 必须单独描述静态开场首帧，并保持角色外观一致
- motion_prompt 总结动作弧线，visual_beats 负责逐段动作、环境反馈、景别、运镜和声音卡点
- keyframe_prompt 默认不含可读文字；标题、字幕、对白字卡和 UI 文案使用 post_overlay 后期叠加
- 发言镜头必须填写 dialogue_speaker，其他人物明确保持闭嘴和静止反应
- 严禁使用 Markdown 格式，仅返回纯 JSON 字符串
"""
        self.system_prompt = COMPACT_STORYBOARD_SYSTEM_PROMPT

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
1. event 只写可见事件，dialogue 只写说出口的纯台词，二者不得互相复制
2. character 台词需要匹配口型；系统播报、旁白、画外音必须写入 voice_events 且 lip_sync=false
3. 每个声音事件填写准确起止时间，并与对应 visual beat 的画面反应对齐
"""
        else:
            dialogue_prompt_addendum = """
\n【重要！必须严格遵循的规则：无台词模式】
1. event 字段只写简短的可见动作和画面事件，不包含任何角色台词。
2. opening_state 只描述静态首帧状态；通过 event 和 visual_beats 传达剧情，而不是靠台词。
3. dialogue、voice_events 只在确有声音时填写。
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
            "character_names": ["本镜所有可辨认角色的完整角色名"],
            "shot_size": "全景/中景/近景/特写",
            "camera_angle": "平视/俯拍/仰拍/过肩等",
            "lens": "镜头焦段与景深",
            "camera_motion": "固定/推/拉/摇/移/跟/环绕",
            "transition": "硬切/叠化/匹配剪辑等",
            "audio_design": "环境声、动作音效、音乐和声音情绪",
            "visual_prompt": "通用分镜视觉描述，包含整镜场景、角色、视觉叙事、光影、构图和风格...",
            "keyframe_prompt": "只描述本镜开场第一帧的静态画面：人物起始姿态、站位、表情、环境、构图、机位、光线和材质，不写运镜、动作过程、台词或声音...",
            "motion_prompt": "整镜动作弧线的简要总述...",
            "visual_beats": [{"start_seconds": 0, "end_seconds": 3, "purpose": "开场钩子/推进/结果", "subject_action": "本段唯一主动作", "environment_action": "环境反馈", "shot_size": "本段景别", "camera_angle": "本段角度", "camera_motion": "本段唯一主要运镜", "sound_cue": "声音卡点"}],
            "voice_events": [{"kind": "character/system_vo/narration/offscreen", "speaker_name": "角色名或声音来源", "text": "纯台词", "start_seconds": 0.5, "end_seconds": 3.5, "lip_sync": false}],
            "text_policy": "post_overlay"
        }
    ]
}

【创作铁律】
1. narrative 只写画面事件；dialogue 只写纯台词；dialogue_speaker 精确绑定唯一发言者。
2. 台词长度按时长控制：5-10秒镜头建议 10-30 字，绝对不能超过35个字！
3. 情绪状态要具体："眼眶微红/嘴角微扬/眉头紧锁" 比 "伤心/开心/生气" 效果好10倍
4. 每镜具备完整动作弧线；每个 visual beat 只完成一个主动作并使用一种主要运镜，整镜每 2–4 秒产生新的可见变化
5. 营造氛围感：描写光影（如"暖黄色的夕阳余晖"）
6. 视觉风格一致：所有分镜保持统一画风
7. 每个分镜必须包含 duration 字段，取值 4–15 秒，并让 visual_beats 从 0 秒连续覆盖到 duration

- visual_prompt 应当详细描述整镜的通用视觉方向；keyframe_prompt 必须单独描述视频开始前的静态首帧，二者不得复制成同一段文字。
- motion_prompt 总结动作弧线，visual_beats 负责逐段动作、环境反馈、景别、运镜和声音卡点。
- keyframe_prompt 默认不含可读文字；标题、字幕、对白字卡和 UI 文案使用 post_overlay 后期叠加。
- 发言镜头必须填写 dialogue_speaker，其他人物明确保持闭嘴和静止反应。
- 严禁使用 Markdown 格式，仅返回纯 JSON 字符串。
"""
        self.system_prompt = COMPACT_STORYBOARD_SYSTEM_PROMPT
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
1. event 只写可见事件，dialogue 只写说出口的纯台词，二者不得互相复制
2. character 台词需要匹配口型；系统播报、旁白、画外音必须写入 voice_events 且 lip_sync=false
3. 每个声音事件填写准确起止时间，并与对应 visual beat 的画面反应对齐
"""
        else:
            prompt += """
\n【重要！必须严格遵循的规则：无台词模式】
1. event 字段只写可见动作和画面事件，不包含任何角色台词。
2. opening_state 只描述静态首帧状态；不要把旁白或对白塞进画面字段。
3. dialogue、voice_events 只在确有声音时填写。
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
