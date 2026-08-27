from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class GenerationStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Scene(BaseModel):
    id: int = Field(..., description="Scene number")
    narrative: str = Field(..., description="Voiceover or narrative text for the scene")
    visual_prompt: str = Field(..., description="General visual direction for the storyboard shot")
    keyframe_prompt: str = Field(default="", description="Static opening-frame image generation prompt")
    motion_prompt: str = Field(..., description="Prompt describing the movement/action for video generation")
    duration: int = Field(default=5, ge=4, le=15, description="Video duration in seconds (4-15)")
    story_beat: str = Field(default="", description="What happens in this shot, separate from dialogue")
    dialogue: str = Field(default="", description="Spoken dialogue or voiceover")
    dialogue_speaker: str = Field(default="", description="Exact character name who speaks; empty for narration or silence")
    shot_size: str = Field(default="", description="Shot size, such as wide, medium, close-up")
    camera_angle: str = Field(default="", description="Camera angle")
    lens: str = Field(default="", description="Lens choice")
    camera_motion: str = Field(default="", description="Camera movement")
    transition: str = Field(default="硬切", description="Transition to next shot")
    audio_design: str = Field(default="", description="Sound effects, ambience, music, and voice notes")
    
    # Paths to generated assets
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    
    # Status tracking
    image_status: GenerationStatus = GenerationStatus.PENDING
    video_status: GenerationStatus = GenerationStatus.PENDING
    error_message: Optional[str] = None

class Storyboard(BaseModel):
    topic: str
    scenes: List[Scene]
