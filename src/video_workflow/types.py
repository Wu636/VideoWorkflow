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
    visual_prompt: str = Field(..., description="Detailed prompt for generating the static keyframe image")
    motion_prompt: str = Field(..., description="Prompt describing the movement/action for video generation")
    
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
