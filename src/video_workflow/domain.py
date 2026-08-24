from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class ProjectStatus(str, Enum):
    BRIEF_DRAFT = "brief_draft"
    STORYBOARD_DRAFT = "storyboard_draft"
    STORYBOARD_REVIEW = "storyboard_review"
    STORYBOARD_APPROVED = "storyboard_approved"
    KEYFRAMES_REVIEW = "keyframes_review"
    RENDER_PLAN_APPROVED = "render_plan_approved"
    RENDERING = "rendering"
    CLIPS_REVIEW = "clips_review"
    EDITING = "editing"
    FINAL_REVIEW = "final_review"
    DELIVERED = "delivered"


class ApprovalStatus(str, Enum):
    DRAFT = "draft"
    PENDING = "pending"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"


class GenerationMode(str, Enum):
    AUTO = "auto"
    I2V = "i2v"
    R2V = "r2v"


class AssetType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    SUBTITLE = "subtitle"
    DOCUMENT = "document"


class AssetRole(str, Enum):
    CHARACTER = "character"
    STYLE = "style"
    SCENE = "scene"
    KEYFRAME = "keyframe"
    LAST_FRAME = "last_frame"
    MOTION = "motion"
    VOICE = "voice"
    MUSIC = "music"
    SOUND_EFFECT = "sound_effect"
    OUTPUT = "output"
    OTHER = "other"


class JobType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    FINALIZE = "finalize"


class JobStatus(str, Enum):
    QUEUED = "queued"
    SUBMITTING = "submitting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"


class ProjectBrief(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    client_name: str = ""
    story: str = Field(min_length=1)
    target_duration_seconds: float = Field(default=30.0, ge=1.0, le=3600.0)
    aspect_ratio: str = "16:9"
    width: int = Field(default=1344, ge=32, le=4096)
    height: int = Field(default=768, ge=32, le=4096)
    fps: float = Field(default=24.0, ge=1.0, le=120.0)
    language: str = "zh-CN"
    visual_style: str = ""
    pacing: str = ""
    audience: str = ""
    negative_prompt: str = ""
    delivery_notes: str = ""

    @field_validator("width", "height")
    @classmethod
    def round_canvas(cls, value: int) -> int:
        return max(32, round(value / 32) * 32)


class CharacterProfile(BaseModel):
    id: str = Field(default_factory=lambda: new_id("character"))
    name: str
    description: str = ""
    wardrobe: str = ""
    voice_description: str = ""
    tts_voice: str = ""
    reference_asset_ids: list[str] = Field(default_factory=list)


class Project(BaseModel):
    id: str = Field(default_factory=lambda: new_id("project"))
    status: ProjectStatus = ProjectStatus.BRIEF_DRAFT
    brief: ProjectBrief
    style_bible: str = ""
    characters: list[CharacterProfile] = Field(default_factory=list)
    ai_recommended_shot_count: int | None = Field(default=None, ge=1, le=500)
    review_token: str = Field(default_factory=lambda: new_id("review"))
    storyboard_version: int = 1
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class CharacterAnalysisDraft(BaseModel):
    character_id: str | None = None
    name: str
    description: str = ""
    wardrobe: str = ""
    voice_description: str = ""
    reference_observations: str = ""


class ProjectAnalysisDraft(BaseModel):
    visual_style: str = ""
    pacing: str = ""
    audience: str = ""
    style_bible: str = ""
    negative_prompt: str = ""
    delivery_notes: str = ""
    recommended_shot_count: int = Field(default=1, ge=1, le=500)
    shot_count_reason: str = ""
    characters: list[CharacterAnalysisDraft] = Field(default_factory=list)
    analysis_notes: list[str] = Field(default_factory=list)


class DialogueTurn(BaseModel):
    speaker_id: str | None = None
    text: str = ""


class ScriptRewriteDraft(BaseModel):
    rewritten_story: str = Field(min_length=1)
    rewrite_mode: Literal["expand", "shorten", "balanced"]
    estimated_duration_seconds: float = Field(ge=1)
    change_summary: str = ""
    feasibility_notes: list[str] = Field(default_factory=list)


class Shot(BaseModel):
    id: str = Field(default_factory=lambda: new_id("shot"))
    project_id: str
    ordinal: int = Field(ge=1)
    title: str = ""
    narrative: str = ""
    dialogue: str = ""
    dialogue_speaker_id: str | None = None
    dialogue_start_seconds: float = Field(default=0.35, ge=0.0, le=15.0)
    dialogue_rate_percent: int = Field(default=0, ge=-50, le=100)
    dialogue_turns: list[DialogueTurn] = Field(default_factory=list)
    duration_seconds: float = Field(default=5.0, ge=0.25, le=15.0)
    scene_description: str = ""
    character_ids: list[str] = Field(default_factory=list)
    shot_size: str = "中景"
    camera_angle: str = "平视"
    lens: str = "标准镜头"
    camera_motion: str = "固定镜头"
    subject_motion: str = ""
    transition: str = "硬切"
    audio_design: str = ""
    # General visual direction for the storyboard.  This intentionally stays
    # separate from the still-image prompt used to generate the first frame.
    visual_prompt: str = ""
    keyframe_prompt: str = ""
    # User-authored/base motion prompt. ``video_prompt`` is the compiled H3
    # prompt shown to the user and submitted to ComfyUI.
    video_prompt_source: str = ""
    video_prompt: str = ""
    h3_prompt_skill_id: str = "h3-prompt-writing"
    h3_prompt_skill_version: str = ""
    h3_prompt_skill_output: str = ""
    negative_prompt: str = ""
    generation_mode: GenerationMode = GenerationMode.AUTO
    resolved_generation_mode: GenerationMode | None = None
    ref_image_size: str = "match"
    reference_asset_ids: list[str] = Field(default_factory=list)
    keyframe_asset_id: str | None = None
    last_frame_asset_id: str | None = None
    image_path: str | None = None
    video_path: str | None = None
    image_status: str = "pending"
    video_status: str = "pending"
    approval_status: ApprovalStatus = ApprovalStatus.DRAFT
    render_frames: int = 124
    h3_width: int | None = Field(default=None, ge=32, le=4096)
    h3_height: int | None = Field(default=None, ge=32, le=4096)
    # True preserves existing 4-step jobs.  The quality preset explicitly
    # switches this off and uses the official native 20-step path.
    h3_turbo: bool = True
    h3_steps: int = Field(default=4, ge=1, le=100)
    h3_scheduler: Literal[
        "simple",
        "sgm_uniform",
        "karras",
        "exponential",
        "ddim_uniform",
        "beta",
        "normal",
        "linear_quadratic",
        "kl_optimal",
    ] = "simple"
    h3_denoise: float = Field(default=1.0, ge=0.0, le=1.0)
    h3_lora_strength: float = Field(default=1.0, ge=-10.0, le=10.0)
    h3_low_vram: bool = False
    h3_shift_video: float = Field(default=12.0, ge=0.01, le=100.0)
    h3_shift_audio: float = Field(default=3.0, ge=0.01, le=100.0)
    h3_seed: int | None = Field(default=None, ge=1, le=2**31 - 1)
    version: int = 1
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_keyframe_prompt(cls, value: Any) -> Any:
        """Give legacy shots a clean first-frame prompt without losing data.

        Older records used ``visual_prompt`` for two different purposes.  Their
        uncompiled scene description is the closest representation of the
        intended static opening frame, so expose it as ``keyframe_prompt`` when
        loading those records.  Once a shot is saved the new field is persisted.
        """
        if isinstance(value, dict) and "keyframe_prompt" not in value:
            migrated = dict(value)
            migrated["keyframe_prompt"] = str(
                migrated.get("scene_description") or migrated.get("visual_prompt") or ""
            )
            return migrated
        return value


class Asset(BaseModel):
    id: str = Field(default_factory=lambda: new_id("asset"))
    project_id: str
    type: AssetType
    role: AssetRole = AssetRole.OTHER
    name: str
    path: str
    mime_type: str = "application/octet-stream"
    character_id: str | None = None
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    approved: bool = False
    sha256: str = ""
    size_bytes: int = 0
    created_at: str = Field(default_factory=utc_now)


class RenderJob(BaseModel):
    id: str = Field(default_factory=lambda: new_id("job"))
    project_id: str
    shot_id: str | None = None
    type: JobType = JobType.VIDEO
    status: JobStatus = JobStatus.QUEUED
    provider: str = "comfyui_h3"
    mode: GenerationMode | None = None
    prompt_id: str | None = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    queue_position: int | None = None
    seed: int = 0
    attempt: int = 0
    max_attempts: int = 2
    input_snapshot: dict[str, Any] = Field(default_factory=dict)
    workflow_snapshot: dict[str, Any] = Field(default_factory=dict)
    output_path: str | None = None
    error: str | None = None
    elapsed_seconds: float | None = None
    estimated_cost: float | None = None
    created_at: str = Field(default_factory=utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str = Field(default_factory=utc_now)


class Review(BaseModel):
    id: str = Field(default_factory=lambda: new_id("review"))
    project_id: str
    target_type: str
    target_id: str
    decision: ApprovalStatus
    comment: str = ""
    reviewer: str = ""
    created_at: str = Field(default_factory=utc_now)


class Delivery(BaseModel):
    id: str = Field(default_factory=lambda: new_id("delivery"))
    project_id: str
    output_path: str
    preview_path: str | None = None
    subtitle_path: str | None = None
    duration_seconds: float = 0.0
    qc_report: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)


class FinalizeRequest(BaseModel):
    shot_ids: list[str] | None = None
    output_name: str = "final_video.mp4"
    crossfade_seconds: float = Field(default=0.0, ge=0.0, le=2.0)
    burn_subtitles: bool = False
    subtitle_asset_id: str | None = None
    background_music_asset_id: str | None = None
    background_music_volume: float = Field(default=0.18, ge=0.0, le=1.0)
    normalize_audio: bool = True
    preview: bool = False
