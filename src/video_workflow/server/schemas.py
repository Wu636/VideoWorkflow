from pydantic import BaseModel
from typing import Optional, List
from src.video_workflow.types import Storyboard, Scene

class CreateSessionRequest(BaseModel):
    topic: str
    reference_image: Optional[str] = None  # File path (from upload endpoint)
    template: Optional[str] = None
    count: int = 5
    include_dialogue: bool = True
    character_description: Optional[str] = None  # From AI analysis or user input
    image_style: Optional[str] = None  # From AI analysis or user input

class SessionResponse(BaseModel):
    session_id: str
    status: str
    storyboard: Optional[Storyboard] = None

class ReviseScriptRequest(BaseModel):
    feedback: str
    reference_image: Optional[str] = None

class GenerateImagesRequest(BaseModel):
    scene_ids: Optional[List[int]] = None
    reference_image: Optional[str] = None

class FeedbackRequest(BaseModel):
    feedback: str

class GenerateVideosRequest(BaseModel):
    scene_ids: Optional[List[int]] = None
