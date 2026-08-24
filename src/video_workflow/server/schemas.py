from pydantic import BaseModel
from typing import Optional, List
from src.video_workflow.types import Storyboard

class CreateSessionRequest(BaseModel):
    topic: str
    reference_image: Optional[str] = None  # File path (from upload endpoint)
    template: Optional[str] = None
    count: int = 5
    include_dialogue: bool = False
    character_description: Optional[str] = None  # From AI analysis or user input
    image_style: Optional[str] = None  # From AI analysis or user input
    image_provider: Optional[str] = None
    image_model: Optional[str] = None
    video_provider: Optional[str] = None
    video_model: Optional[str] = None
    video_aspect_ratio: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    status: str
    storyboard: Optional[Storyboard] = None


class TemplateOption(BaseModel):
    name: str
    description: str
    example_prompt: str


class ImageModelOption(BaseModel):
    id: str
    label: str
    description: str


class ImageProviderOption(BaseModel):
    provider: str
    label: str
    description: str
    default_model: str
    api_key_env: str
    models: List[ImageModelOption]


class VideoModelOption(BaseModel):
    id: str
    label: str
    description: str


class VideoProviderOption(BaseModel):
    provider: str
    label: str
    description: str
    default_model: str
    default_aspect_ratio: str
    api_key_env: str
    supported_aspect_ratios: List[str]
    models: List[VideoModelOption]

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
    retry_failed_only: bool = True
    video_provider: Optional[str] = None
    video_model: Optional[str] = None
    video_aspect_ratio: Optional[str] = None
