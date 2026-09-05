from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from src.video_workflow.domain import VisualBeat, VoiceEvent

class GenerationStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Scene(BaseModel):
    id: int = Field(default=0, description="Scene number")
    # Compact storyboard responses use one event and one opening-state field.
    # Keep the legacy names optional so old provider responses and imported
    # projects continue to deserialize without a migration.
    event: str = Field(default="", description="The complete visible story event in this shot")
    opening_state: str = Field(default="", description="The static state at time 0, before the action starts")
    narrative: str = Field(default="", description="Voiceover or narrative text for the scene")
    visual_prompt: str = Field(default="", description="Legacy general visual direction; compiled locally when omitted")
    keyframe_prompt: str = Field(default="", description="Static opening-frame image generation prompt")
    motion_prompt: str = Field(default="", description="Legacy motion summary; derived locally from visual beats when omitted")
    duration: int = Field(default=5, ge=4, le=15, description="Video duration in seconds (4-15)")
    story_beat: str = Field(default="", description="What happens in this shot, separate from dialogue")
    dialogue: str = Field(default="", description="Spoken dialogue or voiceover")
    dialogue_speaker: str = Field(default="", description="Exact character name who speaks; empty for narration or silence")
    character_names: List[str] = Field(
        default_factory=list,
        description="Exact project character names visibly present in this shot",
    )
    shot_size: str = Field(default="", description="Shot size, such as wide, medium, close-up")
    camera_angle: str = Field(default="", description="Camera angle")
    lens: str = Field(default="", description="Lens choice")
    camera_motion: str = Field(default="", description="Camera movement")
    transition: str = Field(default="硬切", description="Transition to next shot")
    audio_design: str = Field(default="", description="Sound effects, ambience, music, and voice notes")
    visual_beats: List[VisualBeat] = Field(
        default_factory=list,
        description="Timed visual changes covering the complete shot",
    )
    voice_events: List[VoiceEvent] = Field(
        default_factory=list,
        description="Timed character, narration, system-VO, or off-screen voice events",
    )
    text_policy: Literal["none", "post_overlay", "reference_locked"] = Field(
        default="post_overlay",
        description="none, post_overlay, or reference_locked",
    )
    
    # Paths to generated assets
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    
    # Status tracking
    image_status: GenerationStatus = GenerationStatus.PENDING
    video_status: GenerationStatus = GenerationStatus.PENDING
    error_message: Optional[str] = None

class Storyboard(BaseModel):
    topic: str = ""
    scenes: List[Scene] = Field(default_factory=list)
