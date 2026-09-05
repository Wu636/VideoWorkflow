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
            "Follow the project's approved visual style without replacing it with a preset style. "
            "Prioritize concrete visible action, spatial continuity, precise camera behavior, synchronized diegetic audio, "
            "and a restrained non-diegetic music plan."
        ),
    ),
    H3PromptSkill(
        id="3d-animation-short-generator",
        name="3D 动画短片",
        version="0.5.4",
        category="动画",
        summary="风格化 3D 电影质感、清晰动作线、角色表演与逐秒节拍。",
        directive=(
            "Direct the shot as a polished stylized 3D animated film: readable silhouettes, soft cinematic materials, "
            "controlled global illumination, expressive but identity-safe facial acting, anticipation and follow-through, "
            "clear action arcs, and an explicit second-by-second progression. Keep characters, wardrobe and scene geography stable."
        ),
    ),
    H3PromptSkill(
        id="brand-promo-video-generator",
        name="品牌宣传短片",
        version="0.1.9",
        category="商业广告",
        summary="围绕真实产品卖点、品牌资产、功能演示和行动号召组织镜头。",
        directive=(
            "Write a product-specific brand promo shot. Preserve supplied brand assets and verified claims, show a clear "
            "cause-and-effect product interaction, use exact beat timing, legible intentional UI or copy only when the shot requests it, "
            "and finish on a useful proof point or call-to-action beat. Avoid generic abstract spectacle."
        ),
    ),
    H3PromptSkill(
        id="co-op-game-intro-generator",
        name="双人游戏开场",
        version="0.1.5",
        category="创意实验",
        summary="双角色身份锚定、游戏主菜单、玩家卡片与界面交互动效。",
        directive=(
            "Frame the shot as a premium two-player co-op game intro. Keep the two player identities, left/right placement and "
            "costume anchors distinct; coordinate a limited color system across background, player cards, buttons, icons and typography; "
            "animate one readable menu interaction at a time while keeping the interface clean and the characters integrated with it."
        ),
    ),
    H3PromptSkill(
        id="handdrawn-live-video-generator",
        name="手绘实拍融合",
        version="1.0.2",
        category="创意实验",
        summary="粗粝发光手绘与实拍空间接触、连续变形和慢半拍手持追拍。",
        directive=(
            "Blend rough glowing hand-drawn animation with a believable live-action space. Establish physical contact with a real hand "
            "or object early, keep the drawn entity continuous through one inventive morph and escape path, and let a handheld phone camera "
            "follow slightly late. Preserve chalk/crayon texture, contact shadows and a playful non-horror tone."
        ),
    ),
    H3PromptSkill(
        id="minimalist-product-ad-generator",
        name="极简产品广告",
        version="0.5.6",
        category="商业广告",
        summary="高级极简产品展示、产品锚点、文字卡点与干净科技配乐。",
        directive=(
            "Create a premium minimalist product-ad shot with one dominant product anchor, generous negative space, precise controlled "
            "lighting, tactile material detail, purposeful camera motion and beat-synchronized transitions. Any visible copy must be short, "
            "intentional and readable. Never turn the frame into a grid, storyboard, collage or multi-window layout."
        ),
    ),
    H3PromptSkill(
        id="music-video-subtitle-generator",
        name="音乐 MV 动态字幕",
        version="0.6.6",
        category="音频音乐",
        summary="按节拍和人声组织镜头、空间歌词、角色表演与跨镜衔接。",
        directive=(
            "Design a music-video shot around the supplied master-audio window: align performance, cuts, camera energy and spatial lyric "
            "typography to explicit beats and vocal timing. Keep character and aesthetic locks continuous across adjacent shots. Show only "
            "the exact approved lyric fragment, with readable placement and motion that reacts to the music."
        ),
    ),
    H3PromptSkill(
        id="paper-collage-explainer-generator",
        name="纸拼贴讲解动画",
        version="0.3.9",
        category="教育",
        summary="半调纸拼贴、视觉隐喻、触感停格组装和纸张音效。",
        directive=(
            "Translate the idea into a clear tactile editorial paper-collage metaphor: halftone cutouts, bold color blocks, warm white edges, "
            "physical paper shadows and clean composition. Animate with discrete slide, pop, tap and press stop-motion beats rather than "
            "smooth digital drifting. Favor coordinated paper sounds; omit narration, subtitles and background music unless requested."
        ),
    ),
    H3PromptSkill(
        id="papercraft-stop-motion-explainer",
        name="纸艺定格科普",
        version="0.6.5",
        category="教育",
        summary="分层纸雕布景、纸偶定格、立体书展开与科普信息可视化。",
        directive=(
            "Explain the concept through handcrafted papercraft stop motion: layered cardstock planes, visible cut edges, miniature diorama "
            "depth, real inter-layer shadows, paper puppets and props, stepped frame-by-frame motion and restrained parallax. Keep the learning "
            "metaphor immediately understandable and support it with tactile paper movement sounds."
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


def h3_prompt_system_instruction(skill: H3PromptSkill, mode: GenerationMode) -> str:
    if mode == GenerationMode.R2V:
        format_instruction = """Use this exact section order:
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:

Use stable <Subject N>, <Picture N>, <Video N> and <Audio N> labels. Describe playback order and reference use explicitly."""
    else:
        format_instruction = """Use this exact structure:
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] ...
overall_soundscape: ...
non_diegetic_music: ..."""
    return f"""You write production-ready MiniMax H3 prompts by applying the official h3-prompt-writing rules and one selected style Skill.

Selected style Skill: {skill.name} ({skill.id}, version {skill.version})
Style execution directive: {skill.directive}

{format_instruction}

Write the H3 prompt itself in English. Preserve Chinese only inside spoken dialogue tags such as <d>[Chinese] 台词</d>, in character or place names when identity requires it, and inside exact on-screen copy explicitly authored by the shot. H3 is allowed to render authored signs, labels, UI, documents or other readable text: text_policy=post_overlay does not prohibit H3 on-screen text; it only keeps the no-text rule for still frames and Seedance. When authored on-screen copy is present, show it at the specified beat, preserve every character exactly, and do not translate, paraphrase, add or invent copy. text_policy=none forbids readable text, while text_policy=reference_locked allows only text already clear in the locked reference. Never turn dialogue or narration into subtitles unless the shot explicitly requests visible subtitles. Keep the prompt between 500 and 3200 characters. Direct a coherent action arc made of explicit timed visual beats: the first 0–2 seconds establish a visual hook, every 2–4 seconds introduces a new visible change, and the final beat lands on a readable result. Each beat may contain only one main subject action and one main camera move, while the complete shot may contain several consecutive beats. Keep identity, wardrobe and geography stable; synchronize diegetic sound and make a clear music decision. Treat system_vo, narration and offscreen voice as disembodied audio with no character lip movement; only character voice events marked lip_sync may drive a mouth.

The user payload includes a hard speech budget. Keep every spoken event inside it at natural delivery speed; speech may occupy at most about 85% of the shot and voice events must not overlap. Never expand, paraphrase, repeat, or split an authored utterance into extra speech. Put each voice-event text in exactly one <d>[Chinese] ...</d> tag at its authored time. Translate each approved Chinese voice anchor into a compact English performance direction marked as silent, non-spoken production metadata; place that direction before its corresponding dialogue tag, never copy the Chinese voice-description prose into the final H3 prompt, and never place performance prose immediately after a dialogue tag. Preserve perceived age, gender presentation, timbre, pitch range, accent, pacing, and emotional delivery rather than reducing the direction to a generic phrase. Only the exact contents of <d> tags may be vocalized: never vocalize prompt prose, parentheses, labels, timing, metadata, acting directions, or voice descriptions. Do not repeat spoken text in another visual beat, summary, overall_soundscape, or music section; overall_soundscape may describe voice source, tone, and mix level only. Preserve the supplied names, numbers, safety facts, voice-event kind, ownership, and lip-sync flag. If there is no voice event, emit no <d> tag.

Do not invent characters, products, dialogue, claims, reference assets, or on-screen text. Do not output Markdown or commentary. Return JSON only: {{"video_prompt": "..."}}."""
