from __future__ import annotations

from dataclasses import asdict, dataclass

from src.video_workflow.domain import GenerationMode


OFFICIAL_H3_SKILLS_URL = "https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills"
OFFICIAL_H3_SKILLS_COMMIT = "d21241f0a4b3acbb34c97dae47fa417b7065e438"


@dataclass(frozen=True)
class H3PromptSkill:
    id: str
    name: str
    version: str
    category: str
    summary: str
    directive: str

    def public_dict(self) -> dict[str, str]:
        payload = asdict(self)
        payload.pop("directive")
        if self.id == "short-video-talking-ad-generator":
            payload["source_url"] = ""
            payload["source_commit"] = ""
        else:
            payload["source_url"] = f"{OFFICIAL_H3_SKILLS_URL}/{self.id}"
            payload["source_commit"] = OFFICIAL_H3_SKILLS_COMMIT
        return payload


# These compact execution directives adapt the official MiniMax H3 Skills to
# VideoWorkflow's existing per-shot pipeline.  The upstream skills describe a
# complete interactive production workflow; here the project brief, shot table,
# references and approval gates already exist, so only their prompt-writing
# rules are applied at this stage.
H3_PROMPT_SKILLS: tuple[H3PromptSkill, ...] = (
    H3PromptSkill(
        id="h3-prompt-writing",
        name="通用 H3 提示词",
        version="d21241f",
        category="通用",
        summary="遵循 MiniMax 官方 I2V/R2V 结构，完整表达画面、动作、镜头、对白和声音。",
        directive=(
            "沿用项目已经确认的视觉风格，不用预设风格替换。优先描述具体可见的动作、空间连续性、精确的摄影机行为、"
            "同步现场声，以及克制的非剧情音乐方案。"
        ),
    ),
    H3PromptSkill(
        id="3d-animation-short-generator",
        name="3D 动画短片",
        version="0.5.4",
        category="动画",
        summary="风格化 3D 电影质感、清晰动作线、角色表演与逐秒节拍。",
        directive=(
            "按精致的风格化3D动画电影导演镜头：轮廓清楚、材质柔和且有电影感、全局光照受控、面部表演有表现力但身份稳定，"
            "动作包含预备与跟随，动作弧线清晰，并写出明确的逐秒进程。保持角色、服装与场景空间关系稳定。"
        ),
    ),
    H3PromptSkill(
        id="brand-promo-video-generator",
        name="品牌宣传短片",
        version="0.1.9",
        category="商业广告",
        summary="围绕真实产品卖点、品牌资产、功能演示和行动号召组织镜头。",
        directive=(
            "编写针对具体产品的品牌宣传镜头。保留已提供的品牌资产与确认过的宣称，展示因果清楚的产品交互，使用精确节拍；"
            "只有分镜要求时才显示清晰、目的明确的界面或文案，并以有效证据点或行动号召收尾，避免空泛抽象的奇观。"
        ),
    ),
    H3PromptSkill(
        id="short-video-talking-ad-generator",
        name="舞台互动型抓眼广告",
        version="1.2.0",
        category="商业广告",
        summary="中文成片级时间轴、多机位硬切、主持人动态表演与独立观众反应，并规避相邻生成段同景别硬接。",
        directive=(
            "按高转化竖屏舞台口播广告执行，提示词正文使用中文，并逐字保留已确认的产品名、数字、保障责任和行动号召。"
            "锁定同一主持人的身份、脸、服装、持麦手，以及舞台结构、LED版式、灯光方向和观众席位置。主持人不能长时间站桩，"
            "动作与台词语义对应，可使用一至三步自然走位、转向观众、开放手势、指向大屏、轻微前倾、点头和情绪起伏。"
            "单次4至15秒生成允许采用真实商业发布会多机位导播：按台词信息点在斜侧中景、中近景、近景、稍宽中景、不同角度机位和独立观众反应镜头之间干净硬切；"
            "画面切到观众时可以完全不出现主持人，但主持人的同一条现场口播必须连续播放，观众只作自然错落的点头、微笑、鼓掌、交换眼神或举手机记录。"
            "切镜只改变摄影机画面，不改变真实时间线、主持人走位、舞台布局、LED内容或灯光逻辑；不使用黑场、闪白、溶解、旋转、故障、慢动作或夸张甩镜。"
            "相邻独立生成段默认主动错开景别和角度：前一段尾镜头不得与下一段首帧同时使用相同的中景正面构图，优先用近景、稍宽斜侧或观众反应作为前段收尾；若下一段启用连续续接，则下一段0.00秒严格复用本段真实尾帧，取消错位要求。"
            "所有对白标签只说一次，不漏字、不改写、不重复。首尾帧I2V严格从首帧起步并准确落到指定尾帧；不得给锁定主持人镜头绑定无关场景参考。"
        ),
    ),
    H3PromptSkill(
        id="co-op-game-intro-generator",
        name="双人游戏开场",
        version="0.1.5",
        category="创意实验",
        summary="双角色身份锚定、游戏主菜单、玩家卡片与界面交互动效。",
        directive=(
            "把镜头设计为高品质双人合作游戏开场。明确区分两位玩家的身份、左右位置和服装锚点；背景、玩家卡片、按钮、图标和字体"
            "共用一套有限配色；每次只表现一个清楚可读的菜单交互，保持界面干净，并让角色自然融入界面。"
        ),
    ),
    H3PromptSkill(
        id="handdrawn-live-video-generator",
        name="手绘实拍融合",
        version="1.0.2",
        category="创意实验",
        summary="粗粝发光手绘与实拍空间接触、连续变形和慢半拍手持追拍。",
        directive=(
            "把粗粝发光的手绘动画融入可信的实拍空间。尽早让手绘实体与真实手掌或物体发生物理接触，在一次有创意的变形和逃离路径中"
            "保持其连续性，并让手机手持镜头略慢半拍跟随。保留粉笔或蜡笔质感、接触阴影和顽皮但非恐怖的氛围。"
        ),
    ),
    H3PromptSkill(
        id="minimalist-product-ad-generator",
        name="极简产品广告",
        version="0.5.6",
        category="商业广告",
        summary="高级极简产品展示、产品锚点、文字卡点与干净科技配乐。",
        directive=(
            "制作高级极简产品广告镜头：只设一个主导产品锚点，保留充足留白，灯光精确受控，材质细节可感，运镜目的明确，转场与节拍同步。"
            "任何可见文案都必须简短、意图清楚且容易辨认；画面不得变成网格、故事板、拼贴或多窗口布局。"
        ),
    ),
    H3PromptSkill(
        id="music-video-subtitle-generator",
        name="音乐 MV 动态字幕",
        version="0.6.6",
        category="音频音乐",
        summary="按节拍和人声组织镜头、空间歌词、角色表演与跨镜衔接。",
        directive=(
            "围绕已提供的主音频时间窗设计音乐视频镜头：让表演、切点、镜头能量与空间歌词排版对齐明确节拍和人声时序。"
            "相邻镜头持续锁定角色与美术风格；只显示逐字确认的歌词片段，位置清楚可读，运动随音乐响应。"
        ),
    ),
    H3PromptSkill(
        id="paper-collage-explainer-generator",
        name="纸拼贴讲解动画",
        version="0.3.9",
        category="教育",
        summary="半调纸拼贴、视觉隐喻、触感停格组装和纸张音效。",
        directive=(
            "把概念转译为清楚、可触摸的编辑式纸拼贴隐喻：半调网点剪纸、醒目色块、暖白切边、真实纸张阴影和干净构图。"
            "使用分步的滑入、弹出、轻点和按压定格节拍，不做平滑数字漂移；优先配合纸张动作音，除非明确要求，否则不加旁白、字幕和背景音乐。"
        ),
    ),
    H3PromptSkill(
        id="papercraft-stop-motion-explainer",
        name="纸艺定格科普",
        version="0.6.5",
        category="教育",
        summary="分层纸雕布景、纸偶定格、立体书展开与科普信息可视化。",
        directive=(
            "用手工纸艺定格动画解释概念：分层卡纸平面、清楚可见的切边、微缩立体布景深度、真实层间阴影、纸偶与纸道具、"
            "逐帧阶梯式动作和克制视差。科普隐喻要让人立即理解，并用有触感的纸张运动声辅助。"
        ),
    ),
)

_H3_PROMPT_SKILL_MAP = {skill.id: skill for skill in H3_PROMPT_SKILLS}


def list_h3_prompt_skills() -> list[dict[str, str]]:
    return [skill.public_dict() for skill in H3_PROMPT_SKILLS]


def get_h3_prompt_skill(skill_id: str) -> H3PromptSkill:
    try:
        return _H3_PROMPT_SKILL_MAP[skill_id]
    except KeyError as exc:
        raise ValueError(f"未知的 H3 Prompt Skill: {skill_id}") from exc


def h3_prompt_system_instruction(
    skill: H3PromptSkill,
    mode: GenerationMode,
    frame_contract: str = "first_only",
) -> str:
    if mode == GenerationMode.R2V:
        format_instruction = """严格使用以下字段顺序，字段名保留英文，字段内容全部使用中文：
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:

稳定使用 <Subject N>、<Picture N>、<Video N> 和 <Audio N> 标签，并明确说明播放顺序和参考素材用途。"""
    else:
        if frame_contract == "first_last":
            frame_opening = """目标视频0.00秒严格完整参考 <Picture 1>（来自[Shot 1]）。
API另行提供已确认的last_frame；结尾必须准确落到该构图，不给last_frame分配Picture编号。"""
        elif frame_contract == "last_only":
            frame_opening = """API另行提供已确认的last_frame；设计连贯的到达过程并准确落到该构图，不给last_frame分配Picture编号。"""
        else:
            frame_opening = "目标视频0.00秒严格完整参考 <Picture 1>（来自[Shot 1]）。"
        format_instruction = f"""严格使用以下结构，字段名保留英文，字段内容全部使用中文：
{frame_opening}

integrated_multimodal_description: [Shot 1] ...
overall_soundscape: ...
non_diegetic_music: ...

这是互斥的帧输入模板，不输出 subject_definitions、retention_analysis、detailed_description、<Subject N>、<Video N> 或 <Audio N>。"""
    return f"""你负责根据 MiniMax H3 官方提示词规则与选定风格 Skill，编写可直接生产的 H3 提示词。

已选风格 Skill：{skill.name}（{skill.id}，版本 {skill.version}）
风格执行指令：{skill.directive}

{format_instruction}

H3提示词正文全部使用中文，只有官方结构字段名、参考标签和 <d>[Chinese] 台词</d> 标签格式保留英文。H3可以渲染分镜明确要求的标牌、标签、界面、文件或其他可读文字：text_policy=post_overlay不禁止H3或Seedance画面文字；Seedance 按分镜明确写出的屏幕、标牌和界面内容自然呈现，不追加通用禁字规则。出现已编写的画面文案时，必须在指定节拍逐字显示，不翻译、不改写、不补充。text_policy=none禁止任何可读文字，text_policy=reference_locked只允许锁定参考图中已经清楚存在的文字。除非分镜明确要求字幕，不得把对白或旁白变成字幕。提示词长度控制在500至5000字符。用明确时间轴组织完整动作与导播弧线：短视频广告约每1.1至1.8秒产生一次有意义的画面变化，普通内容每2至4秒产生一次。每个时间段只包含一个主要主体动作和一种主要运镜，但同一次4至15秒生成可以在时间段边界进行多机位硬切、改变景别和切入独立观众画面。切镜必须服从台词语义，并保持人物身份、服装、走位、场景空间与真实时间线连续；同步现场声音，并明确背景音乐策略。system_vo、narration和offscreen都是画外声音，不驱动画面人物嘴唇；只有标记lip_sync的character声音可以驱动口型。

用户载荷包含口播硬预算。所有声音事件必须按可懂语速落在预算内，声音事件不得重叠。不得扩写、改写、重复或把原台词拆成额外发言。每个声音事件只在其指定时间出现一次 <d>[Chinese] ...</d> 标签。把已确认的声音锚点压缩成中文的“非口播制作说明”，放在对应对白标签之前；保留年龄感、性别表现、音色、音高、口音、节奏和情绪层级，不把制作说明当成台词。只有 <d> 标签内部文字可以发声；提示词正文、括号、标签、时间、元数据、表演说明和声音描述都不得发声。不得在其他时间段、summary、overall_soundscape或音乐段落中重复台词。保留给定的人名、数字、事实、声音事件类型、归属与口型标记；没有声音事件时不输出 <d> 标签。

不得虚构具名角色、产品、对白、宣称、参考素材或画面文案。只有选定舞台广告 Skill 和已确认场景明确需要时，才可加入无名观众；其人数范围、席位布局和行为保持稳定。不要输出Markdown或解释，只返回JSON：{{"video_prompt": "..."}}。"""
