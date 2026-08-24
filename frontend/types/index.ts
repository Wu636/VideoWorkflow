export type GenerationStatus = 'pending' | 'processing' | 'completed' | 'failed';

export type ProjectStatus =
    | 'brief_draft'
    | 'storyboard_draft'
    | 'storyboard_review'
    | 'storyboard_approved'
    | 'keyframes_review'
    | 'render_plan_approved'
    | 'rendering'
    | 'clips_review'
    | 'editing'
    | 'final_review'
    | 'delivered';

export type ApprovalStatus = 'draft' | 'pending' | 'approved' | 'changes_requested';
export type GenerationMode = 'auto' | 'i2v' | 'r2v';
export type AssetType = 'image' | 'video' | 'audio' | 'subtitle' | 'document';
export type AssetRole =
    | 'character'
    | 'style'
    | 'scene'
    | 'keyframe'
    | 'last_frame'
    | 'motion'
    | 'voice'
    | 'music'
    | 'sound_effect'
    | 'output'
    | 'other';
export type JobStatus = 'queued' | 'submitting' | 'running' | 'completed' | 'failed' | 'cancel_requested' | 'cancelled';

export interface ProjectBrief {
    title: string;
    client_name: string;
    story: string;
    target_duration_seconds: number;
    aspect_ratio: string;
    width: number;
    height: number;
    fps: number;
    language: string;
    visual_style: string;
    pacing: string;
    audience: string;
    negative_prompt: string;
    delivery_notes: string;
}

export interface CharacterProfile {
    id: string;
    name: string;
    description: string;
    wardrobe: string;
    voice_description: string;
    tts_voice: string;
    reference_asset_ids: string[];
}

export interface Project {
    id: string;
    status: ProjectStatus;
    brief: ProjectBrief;
    style_bible: string;
    characters: CharacterProfile[];
    ai_recommended_shot_count: number | null;
    review_token: string;
    storyboard_version: number;
    created_at: string;
    updated_at: string;
}

export interface CharacterAnalysisDraft {
    character_id: string | null;
    name: string;
    description: string;
    wardrobe: string;
    voice_description: string;
    reference_observations: string;
}

export interface ProjectAnalysisDraft {
    visual_style: string;
    pacing: string;
    audience: string;
    style_bible: string;
    negative_prompt: string;
    delivery_notes: string;
    recommended_shot_count: number;
    shot_count_reason: string;
    characters: CharacterAnalysisDraft[];
    analysis_notes: string[];
}

export interface ScriptRewriteDraft {
    rewritten_story: string;
    rewrite_mode: "expand" | "shorten" | "balanced";
    estimated_duration_seconds: number;
    change_summary: string;
    feasibility_notes: string[];
}

export interface RuntimeSettingField {
    key: string;
    label: string;
    description: string;
    secret: boolean;
    options: string[];
    value: unknown;
    configured?: boolean;
    masked?: string;
}

export interface RuntimeSettingsPayload {
    routes: RuntimeFunctionRoute[];
    groups: { id: string; label: string; fields: RuntimeSettingField[] }[];
    path: string;
    restart_required: boolean;
}

export interface RuntimeFunctionRoute {
    id: string;
    label: string;
    setting_key: string | null;
    selected: string;
    options: { value: string; label: string }[];
    effective_label: string;
    model: string;
    configured: boolean;
    priority: string[];
    description: string;
}

export interface RuntimeLogRecord {
    timestamp: string;
    level: string;
    logger: string;
    message: string;
    exception: string;
}

export interface Shot {
    id: string;
    project_id: string;
    ordinal: number;
    title: string;
    narrative: string;
    dialogue: string;
    dialogue_speaker_id: string | null;
    dialogue_start_seconds: number;
    dialogue_rate_percent: number;
    dialogue_turns: { speaker_id: string | null; text: string }[];
    duration_seconds: number;
    scene_description: string;
    character_ids: string[];
    shot_size: string;
    camera_angle: string;
    lens: string;
    camera_motion: string;
    subject_motion: string;
    transition: string;
    audio_design: string;
    visual_prompt: string;
    keyframe_prompt: string;
    video_prompt_source: string;
    video_prompt: string;
    negative_prompt: string;
    generation_mode: GenerationMode;
    resolved_generation_mode: GenerationMode | null;
    ref_image_size: string;
    reference_asset_ids: string[];
    keyframe_asset_id: string | null;
    last_frame_asset_id: string | null;
    image_path: string | null;
    video_path: string | null;
    image_status: GenerationStatus;
    video_status: GenerationStatus;
    approval_status: ApprovalStatus;
    render_frames: number;
    h3_width: number | null;
    h3_height: number | null;
    h3_turbo: boolean;
    h3_steps: number;
    h3_scheduler: "simple" | "sgm_uniform" | "karras" | "exponential" | "ddim_uniform" | "beta" | "normal" | "linear_quadratic" | "kl_optimal";
    h3_denoise: number;
    h3_lora_strength: number;
    h3_low_vram: boolean;
    h3_shift_video: number;
    h3_shift_audio: number;
    h3_seed: number | null;
    version: number;
    created_at: string;
    updated_at: string;
}

export interface Asset {
    id: string;
    project_id: string;
    type: AssetType;
    role: AssetRole;
    name: string;
    path: string;
    mime_type: string;
    character_id: string | null;
    description: string;
    tags: string[];
    approved: boolean;
    sha256: string;
    size_bytes: number;
    created_at: string;
}

export interface RenderJob {
    id: string;
    project_id: string;
    shot_id: string | null;
    type: 'image' | 'video' | 'finalize';
    status: JobStatus;
    provider: string;
    mode: GenerationMode | null;
    prompt_id: string | null;
    progress: number;
    queue_position: number | null;
    seed: number;
    attempt: number;
    max_attempts: number;
    input_snapshot: Record<string, unknown>;
    workflow_snapshot: Record<string, unknown>;
    output_path: string | null;
    error: string | null;
    elapsed_seconds: number | null;
    estimated_cost: number | null;
    created_at: string;
    started_at: string | null;
    completed_at: string | null;
    updated_at: string;
}

export interface Review {
    id: string;
    project_id: string;
    target_type: string;
    target_id: string;
    decision: ApprovalStatus;
    comment: string;
    reviewer: string;
    created_at: string;
}

export interface Delivery {
    id: string;
    project_id: string;
    output_path: string;
    preview_path: string | null;
    subtitle_path: string | null;
    duration_seconds: number;
    qc_report: Record<string, unknown>;
    created_at: string;
}

export interface ProjectBundle {
    project: Project;
    shots: Shot[];
    assets: Asset[];
    jobs: RenderJob[];
    reviews: Review[];
    deliveries: Delivery[];
}

export interface Scene {
    id: number;
    duration: number;
    narrative: string;
    visual_prompt: string;
    motion_prompt: string;
    image_path?: string;
    video_path?: string;
    image_status?: GenerationStatus;
    video_status?: GenerationStatus;
    error_message?: string;
}

export interface Storyboard {
    topic: string;
    scenes: Scene[];
}

export interface SessionResponse {
    session_id: string;
    status: string;
    storyboard?: Storyboard;
}

export interface TemplateOption {
    name: string;
    description: string;
    example_prompt: string;
}

export interface ImageModelOption {
    id: string;
    label: string;
    description: string;
}

export interface ImageProviderOption {
    provider: string;
    label: string;
    description: string;
    default_model: string;
    api_key_env: string;
    models: ImageModelOption[];
}

export interface VideoModelOption {
    id: string;
    label: string;
    description: string;
}

export interface VideoProviderOption {
    provider: string;
    label: string;
    description: string;
    default_model: string;
    default_aspect_ratio: string;
    api_key_env: string;
    supported_aspect_ratios: string[];
    models: VideoModelOption[];
}

export const DEFAULT_VIRAL_TEMPLATES: TemplateOption[] = [
    { name: "反转剧", description: "开头设悬念，结尾神反转，让观众恍然大悟", example_prompt: "一只看起来凶巴巴的流浪猫，其实一直在默默保护小区里的孩子们" },
    { name: "萌宠日常", description: "拟人化的萌宠角色，展示可爱有趣的日常生活", example_prompt: "一只每天准时叫主人起床的猫咪闹钟" },
    { name: "治愈系", description: "温暖人心的故事，让观众感到被治愈和感动", example_prompt: "一只被遗弃的小狗，遇到了一个同样孤独的老人" },
    { name: "猎奇科普", description: "有趣的冷知识或猎奇内容，让观众惊叹'原来是这样'", example_prompt: "你知道吗？猫咪每天要睡16个小时" },
    { name: "搞笑剧场", description: "轻松搞笑的短剧，让观众开怀大笑", example_prompt: "一只自以为很优雅的猫，结果每次都出糗" },
    { name: "情感共鸣", description: "触动人心的情感故事，引发观众强烈共鸣", example_prompt: "一只陪伴主人从小到大的狗狗，见证了主人的成长" },
    { name: "委屈萌宠", description: "萌宠遭遇不公待遇，引发观众心疼和共情", example_prompt: "小狗送外卖迟到了2分钟，被顾客劈头盖脸骂了一顿" },
    { name: "萌宠开口说话", description: "萌宠模仿人类说话，用第一人称台词讲述故事，当前最火爆款类型", example_prompt: "橘猫打工人的一天：早上被闹钟吵醒，挤地铁，被老板骂，晚上回家只想躺平" },
];

export const DEFAULT_IMAGE_PROVIDERS: ImageProviderOption[] = [
    {
        provider: "ark",
        label: "Volcengine Ark",
        description: "使用方舟图像能力生成分镜首帧。",
        default_model: "doubao-seedream-4-5-251128",
        api_key_env: "ARK_API_KEY",
        models: [
            {
                id: "doubao-seedream-4-5-251128",
                label: "doubao-seedream-4-5-251128",
                description: "默认方舟图像模型",
            },
        ],
    },
    {
        provider: "grsai",
        label: "GRSAI Nano Banana",
        description: "接入 Nano Banana 系列模型，适合批量分镜和高质量关键帧。",
        default_model: "nano-banana-fast",
        api_key_env: "GRSAI_API_KEY",
        models: [
            { id: "nano-banana-fast", label: "nano-banana-fast", description: "速度优先，适合批量分镜" },
            { id: "nano-banana", label: "nano-banana", description: "平衡质量与速度的标准模型" },
            { id: "nano-banana-2", label: "nano-banana-2", description: "新版本 Nano Banana" },
            { id: "nano-banana-pro", label: "nano-banana-pro", description: "质量优先，适合关键画面" },
            { id: "nano-banana-pro-vt", label: "nano-banana-pro-vt", description: "高质量 Pro 变体" },
            { id: "nano-banana-pro-cl", label: "nano-banana-pro-cl", description: "高质量 Pro 变体" },
            { id: "nano-banana-pro-vip", label: "nano-banana-pro-vip", description: "支持更高规格输出" },
            { id: "nano-banana-pro-4k-vip", label: "nano-banana-pro-4k-vip", description: "4K 输出专用版本" },
        ],
    },
];

export const DEFAULT_VIDEO_PROVIDERS: VideoProviderOption[] = [
    {
        provider: "grsai",
        label: "GRSAI Veo3",
        description: "通过 /v1/video/veo 生成视频片段。",
        default_model: "veo3.1-fast",
        default_aspect_ratio: "16:9",
        api_key_env: "GRSAI_API_KEY",
        supported_aspect_ratios: ["16:9", "9:16", "1:1"],
        models: [
            { id: "veo3.1-fast", label: "veo3.1-fast", description: "速度优先" },
            { id: "veo3.1-pro", label: "veo3.1-pro", description: "质量优先" },
        ],
    },
    {
        provider: "ark",
        label: "Volcengine Ark",
        description: "通过方舟视频接口生成片段。",
        default_model: "doubao-seedance-1-5-pro",
        default_aspect_ratio: "16:9",
        api_key_env: "ARK_API_KEY",
        supported_aspect_ratios: ["16:9"],
        models: [
            { id: "doubao-seedance-1-5-pro", label: "doubao-seedance-1-5-pro", description: "默认方舟视频模型" },
        ],
    },
];
