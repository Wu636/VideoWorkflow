"""Versioned, user-editable prompt profiles for storyboard production.

The machine contract is intentionally kept separate from creative direction.
Users can tune the latter without accidentally removing fields that the local
storyboard normalizer and downstream prompt compilers require.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


SYSTEM_PROMPT_TEMPLATE_ID = "system-default"
SYSTEM_PROMPT_TEMPLATE_VERSION = 1
PROMPT_PROFILE_SCHEMA_VERSION = 1

PROMPT_TEMPLATE_VARIABLES = (
    "story",
    "title",
    "shot_count",
    "target_duration",
    "average_shot_duration",
    "characters",
    "style",
    "aspect_ratio",
    "pacing",
    "speech_pacing",
    "spoken_text_policy",
    "delivery_notes",
    "scene_profiles",
    "series_constraints",
    "run_notes",
    "prompt_targets",
    "count_contract",
    "speech_contract",
)

# Keep this metadata next to the whitelist so the editor and API can explain
# exactly what is injected at generation time.  The values are intentionally
# short: they are help text for template authors, not another prompt layer.
PROMPT_TEMPLATE_VARIABLE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "title": {
        "label": "项目标题",
        "description": "当前项目的名称。适合放在模板开头，帮助模型识别本剧。",
        "source": "项目资料",
        "example": "南网十禁--第九集",
    },
    "story": {
        "label": "项目剧情",
        "description": "本次项目的完整剧本或故事文本，是分镜拆解的主要输入。",
        "source": "项目资料",
        "example": "第九关：严禁未落实防坠落、防倒杆措施开展高处作业……",
    },
    "characters": {
        "label": "角色档案",
        "description": "角色姓名、外貌、服装、声音和稳定性要求等角色资料。",
        "source": "角色资料",
        "example": "陆峥：28岁男性，深藏青工装……",
    },
    "style": {
        "label": "统一视觉风格",
        "description": "项目视觉风格与风格圣经，用来约束所有镜头的画面气质。",
        "source": "项目分析",
        "example": "写实电影感、冷灰色调、硬朗侧光",
    },
    "scene_profiles": {
        "label": "场景档案",
        "description": "项目中已建立的场景名称和空间资料，用来维持场景连续性。",
        "source": "场景资料",
        "example": "空旷电杆试炼场；老旧台区",
    },
    "shot_count": {
        "label": "镜头数量",
        "description": "本次要求生成的镜头数；模板可以强调严格遵守数量。",
        "source": "本次生成设置",
        "example": "12",
    },
    "target_duration": {
        "label": "目标总时长",
        "description": "本次分镜希望覆盖的总时长，单位为秒。",
        "source": "项目设置",
        "example": "120",
    },
    "average_shot_duration": {
        "label": "平均镜头时长",
        "description": "根据总时长和镜头数量计算出的平均时长，帮助模型分配节奏。",
        "source": "本次生成设置",
        "example": "10.00",
    },
    "aspect_ratio": {
        "label": "画幅比例",
        "description": "视频画幅，例如横屏、竖屏或方形。",
        "source": "项目设置",
        "example": "16:9",
    },
    "pacing": {
        "label": "节奏偏好",
        "description": "项目设定的剪辑或叙事节奏偏好。",
        "source": "项目设置",
        "example": "前快后稳，前三秒出现异常",
    },
    "speech_pacing": {
        "label": "口播节奏",
        "description": "对白或旁白的语速档位，用于控制台词密度。",
        "source": "项目设置",
        "example": "natural",
    },
    "spoken_text_policy": {
        "label": "口播文字策略",
        "description": "项目对台词、旁白和需配音文字的处理策略。",
        "source": "项目设置",
        "example": "adaptive",
    },
    "delivery_notes": {
        "label": "项目交付备注",
        "description": "客户或制作方对输出格式、内容和交付的额外要求。",
        "source": "项目设置",
        "example": "保留安全培训要点，避免出现无法执行的动作",
    },
    "series_constraints": {
        "label": "系列固定约束",
        "description": "系列剧集共用的人物、世界观、画面和制作限制。",
        "source": "系列设置",
        "example": "陆峥始终穿深藏青工装，南网标识位置固定",
    },
    "run_notes": {
        "label": "本次生成建议",
        "description": "用户针对本次生成临时补充的注意事项，每次生成可以不同。",
        "source": "本次生成输入",
        "example": "加强外来试炼者的反面示范，结尾落到安全检查结果",
    },
    "prompt_targets": {
        "label": "输出目标",
        "description": "本次需要服务的下游 Prompt 类型，例如 seedance 或 h3。",
        "source": "本次生成设置",
        "example": "seedance, h3",
    },
    "count_contract": {
        "label": "镜头数量硬约束",
        "description": "系统根据本次镜头设置生成的数量约束说明。",
        "source": "系统生成",
        "example": "按指定镜头数量输出，不新增、删除或拆分 scenes",
    },
    "speech_contract": {
        "label": "口播硬约束",
        "description": "系统根据项目设置生成的声音事件和台词约束说明。",
        "source": "系统生成",
        "example": "声音事件按镜头时长安排，事件不得重叠",
    },
}

PROMPT_TEMPLATE_EXAMPLES: list[dict[str, str]] = [
    {
        "id": "realistic-safety-training",
        "label": "现实主义安全培训 / 流程还原",
        "description": "适合电力、施工、制造和操作规范类项目。",
        "content": """【项目约束补充｜现实主义安全培训】
- 严格按项目剧情中的真实因果拆分镜头，不虚构未登记的设备、岗位、流程或事故原因。
- 每个关键操作按“操作前状态 → 检查或动作 → 设备/环境反馈 → 结果”呈现，让观众看得懂步骤如何发生。
- 违规示范要清晰、具体、可辨认，但不要用夸张动作替代真实风险；后果必须通过画面中的状态变化表现。
- 正确示范要交代完整检查、监护、确认、操作和收尾，不要只用一句台词概括流程。
- 专业术语可以保留，但第一次出现时尽量让画面、道具或角色动作帮助观众理解。
- 同一角色的身份、工装、工具、站位和操作权限保持连续，不让角色在镜头之间无理由换装或换岗位。""",
    },
    {
        "id": "suspense-short-drama",
        "label": "悬疑短剧 / 信息递进",
        "description": "适合悬疑、规则怪谈、调查和反转类项目。",
        "content": """【项目约束补充｜悬疑信息递进】
- 前 2 秒出现一个可见异常、危险征兆或不合常理的细节，先制造问题再解释背景。
- 每个镜头至少推进一条新信息：人物发现、道具变化、关系变化、风险升级或判断改变，不能只重复氛围。
- 不提前说出最终真相；让线索通过视线、手部动作、环境细节和角色反应逐步显露。
- 反转必须能回溯到前面已经出现的视觉线索，避免依靠突然新增的人物、规则或设定完成反转。
- 关键线索出现时安排清晰的特写或视线承接，确保观众知道应该关注什么。
- 每镜结尾形成结果态、疑问或新的行动动机，推动下一镜自然接续。""",
    },
    {
        "id": "spoken-advertising",
        "label": "口播广告 / 卖点节奏",
        "description": "适合产品介绍、品牌宣传、主持人口播和快节奏广告。",
        "content": """【项目约束补充｜口播广告节奏】
- 开场尽快给出痛点、反常识事实、结果画面或明确利益点，不用空泛寒暄铺垫。
- 每一条台词都对应一个可见动作、产品细节、屏幕信息或观众反应，避免画面与口播各说各话。
- 产品名称、数字、规格、责任范围和行动号召按原文保留，重要信息用景别、视线或道具变化加强。
- 主持人的动作服务于语义：介绍时展示，强调时指向，比较时切换对象，结论时落到产品或行动入口。
- 镜头内部用短而清晰的节拍变化保持注意力，但不使用无意义挥手、乱跑或与台词无关的夸张运镜。
- 结尾明确呈现记忆点和下一步行动，不用新的复杂剧情抢走产品信息。""",
    },
    {
        "id": "explanatory-science",
        "label": "科普解释 / 先现象后原理",
        "description": "适合知识科普、流程讲解、产品原理和安全常识。",
        "content": """【项目约束补充｜科普解释结构】
- 先展示观众能感知的现象或问题，再用动作、道具、示意画面和简短台词解释原因。
- 一镜只解决一个小问题；复杂概念拆成连续的“现象 → 原因 → 正确做法 → 结果”。
- 抽象概念必须落到可见的物体、位置、方向、前后状态或对比实验，不用空泛形容词代替解释。
- 需要强调的数字、步骤和结论在画面中安排明确承载物，但不要把整段旁白机械铺成满屏文字。
- 解释者的指向、视线和画面重点保持一致，观众能判断当前正在讲哪一个对象。
- 结尾回到开场问题，展示正确做法带来的结果，并留下简洁、准确的行动建议。""",
    },
    {
        "id": "series-fixed-rules",
        "label": "本剧固定规则 / 系列连续性",
        "description": "适合需要每一集持续遵守的剧情表达、角色和悬念规则。",
        "content": """【本剧固定规则】
- 所有违规行为必须先展示动作，再展示系统判定。
- 安全操作步骤必须通过近景或特写交代清楚。
- 角色不得脱离已有服装和身份设定。
- 少用解释性旁白，优先通过动作、道具和人物反应推进。
- 每集结尾保留下一关的悬念。""",
    },
]

_VARIABLE_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


DEFAULT_MACHINE_CONTRACT = r'''
只返回一个有效 JSON 对象，不要 Markdown、注释或额外解释。
结构：
{"topic":"", "scenes":[{"duration":6,"event":"本镜完整可见事件","opening_state":"0 秒静态起始状态","dialogue":"纯台词或空字符串","dialogue_speaker":"发言角色名/旁白/空字符串","character_names":["本镜可辨认角色完整名"],"shot_size":"中景","camera_angle":"平视","lens":"50mm","camera_motion":"固定或一种主要运镜","visual_beats":[{"start_seconds":0,"end_seconds":3,"purpose":"开场钩子/推进/结果","subject_action":"一个主动作","environment_action":"环境反馈","shot_size":"","camera_angle":"","camera_motion":"","sound_cue":""}],"voice_events":[{"kind":"character/system_vo/narration/offscreen/inner_monologue","speaker_name":"","text":"","start_seconds":0.5,"end_seconds":2.5,"lip_sync":false}]}]}
字段约束：duration 为 4–15 秒；visual_beats 从 0 秒连续覆盖到 duration；voice_events 时间不得重叠；character_names 只列出本镜真正可辨认且来自角色档案的角色；dialogue 只写说出口的纯台词；opening_state 只写静态首帧状态。
'''.strip()


DEFAULT_DIRECTOR_TEMPLATE = r'''
你是短视频分镜导演，负责把项目故事转化为可执行、可拍摄、可生成的分镜方案。

创作时先理解故事的因果关系，再安排镜头。每个镜头必须承担一个清晰的叙事任务：引入信息、制造问题、推进冲突、给出反馈、完成转折或落到结果。不要用只有氛围、没有事件的空镜头凑时长。

动作必须具体到主体、方向、力度、视线、手部状态和环境反馈。情绪必须通过表情、姿态、呼吸、停顿和可见反应表现，不只写“开心、难过、生气”等抽象词。

镜头语言要服务叙事。景别、角度和运镜的改变必须有明确原因；一个 visual beat 只安排一个主要主体动作和一种主要运镜。镜头结尾应形成明确结果态、视觉悬念或下一镜的承接点。

保持人物身份、服装、道具、空间结构、光线方向和整体视觉风格连续。只有真正出现在画面中的角色才列入 character_names；其他角色保持不可辨认或不出现。

对白要口语化、符合角色身份，并与画面中的说话人、动作和反应同步。没有必要时减少解释性旁白，用可见动作和道具变化推动剧情。所有规则最终都要落实到镜头内容，而不是写成泛泛的创作口号。
'''.strip()


DEFAULT_CONTEXT_TEMPLATE = r'''
【项目标题】
{{title}}

【项目剧情】
{{story}}

【角色档案】
{{characters}}

【统一视觉风格】
{{style}}

【场景档案】
{{scene_profiles}}

【制作目标】
总时长：{{target_duration}} 秒；镜头数量：{{shot_count}}；平均每镜：{{average_shot_duration}} 秒；画幅：{{aspect_ratio}}
节奏：{{pacing}}；口播档：{{speech_pacing}}；需配音文字策略：{{spoken_text_policy}}

【项目交付备注】
{{delivery_notes}}

【系列固定约束】
{{series_constraints}}

【镜头数量与时长硬约束】
{{count_contract}}

【口播硬约束】
{{speech_contract}}

【本次生成建议】
{{run_notes}}

【输出目标】
{{prompt_targets}}
'''.strip()


DEFAULT_DOWNSTREAM_RULES = {
    "visual": "通用视觉 Prompt 具体描述场景、可见角色、道具、构图、光影和统一风格；只写本镜真正需要的内容，不复制整部项目资料。",
    "keyframe": "首帧只描述视频 0.00 秒的静态状态：角色站位、朝向、表情、视线、手部状态、空间关系、构图、机位、光线和材质；不写未来动作、动作过程、声音或运镜。",
    "seedance": "动作必须在给定时长内可执行且可观察；按时间轴逐段推进，每段只保留一个主动作和一种主要运镜；结尾落到稳定、清晰、可衔接的结果态。",
    "h3": "最终 H3 Prompt 以已确认的分镜时间轴、角色、声音事件和参考素材为准；优先保持人物、服装、空间、走位和声音归属连续，再增加复杂运镜。",
}


class PromptProfileContent(BaseModel):
    schema_version: int = PROMPT_PROFILE_SCHEMA_VERSION
    machine_contract: str = DEFAULT_MACHINE_CONTRACT
    director_template: str = DEFAULT_DIRECTOR_TEMPLATE
    context_template: str = DEFAULT_CONTEXT_TEMPLATE
    downstream_rules: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_DOWNSTREAM_RULES))

    @field_validator("machine_contract", "director_template", "context_template")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Prompt 模板内容不能为空")
        if len(value) > 30000:
            raise ValueError("Prompt 模板单段不能超过 30000 字符")
        return value

    @field_validator("downstream_rules")
    @classmethod
    def validate_rules(cls, value: dict[str, str]) -> dict[str, str]:
        allowed = {"visual", "keyframe", "seedance", "h3"}
        normalized: dict[str, str] = {}
        for key, text in value.items():
            if key not in allowed:
                continue
            normalized[key] = str(text or "").strip()[:12000]
        return {**DEFAULT_DOWNSTREAM_RULES, **normalized}


class PromptTemplateVersion(BaseModel):
    id: str
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=1000)
    source: Literal["system", "user", "ai"] = "user"
    version: int = Field(default=1, ge=1)
    based_on_default_version: int = SYSTEM_PROMPT_TEMPLATE_VERSION
    content: PromptProfileContent = Field(default_factory=PromptProfileContent)
    change_summary: list[str] = Field(default_factory=list, max_length=50)
    created_by: Literal["system", "user", "ai"] = "user"
    created_at: str = ""
    updated_at: str = ""


def system_default_profile() -> PromptTemplateVersion:
    return PromptTemplateVersion(
        id=SYSTEM_PROMPT_TEMPLATE_ID,
        name="系统默认分镜模板",
        description="VideoWorkflow 内置的通用分镜与下游 Prompt 规则",
        source="system",
        version=SYSTEM_PROMPT_TEMPLATE_VERSION,
        based_on_default_version=SYSTEM_PROMPT_TEMPLATE_VERSION,
        content=PromptProfileContent(),
        created_by="system",
    )


def validate_template_variables(content: PromptProfileContent) -> list[str]:
    unknown: set[str] = set()
    texts = [content.director_template, content.context_template, *content.downstream_rules.values()]
    for text in texts:
        unknown.update(
            name for name in _VARIABLE_PATTERN.findall(text) if name not in PROMPT_TEMPLATE_VARIABLES
        )
    return sorted(unknown)


def render_prompt_template(template: str, values: dict[str, Any]) -> str:
    """Replace only whitelisted ``{{variable}}`` placeholders."""
    unknown = [name for name in _VARIABLE_PATTERN.findall(template) if name not in PROMPT_TEMPLATE_VARIABLES]
    if unknown:
        raise ValueError(f"Prompt 模板包含未知变量: {', '.join(sorted(set(unknown)))}")

    def replace(match: re.Match[str]) -> str:
        value = values.get(match.group(1), "")
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value if value is not None else "")

    return _VARIABLE_PATTERN.sub(replace, template).strip()


def profile_hash(profile: PromptTemplateVersion | dict[str, Any]) -> str:
    payload = profile.model_dump(mode="json") if isinstance(profile, PromptTemplateVersion) else profile
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def content_from_snapshot(snapshot: dict[str, Any] | None) -> PromptProfileContent:
    if not snapshot:
        return system_default_profile().content
    try:
        return PromptProfileContent.model_validate(snapshot.get("content", snapshot))
    except Exception:
        return system_default_profile().content
