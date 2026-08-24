from abc import ABC, abstractmethod
from typing import Any, List
from src.video_workflow.types import Storyboard, Scene

class LLMGenerator(ABC):
    @abstractmethod
    async def generate_storyboard(
        self,
        topic: str,
        count: int = 5,
        reference_image: str | None = None,
        template: str | None = None,
        include_dialogue: bool = True,
        character_description: str | None = None,
        image_style: str | None = None,
        user_suggestions: str | None = None,
    ) -> Storyboard:
        """Generate a storyboard from the brief plus optional user suggestions."""
        pass
    
    async def analyze_reference_image(self, image_path: str) -> str | None:
        """Analyze reference image and return a character description. Default returns None."""
        return None

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        reference_images: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate a provider-independent structured JSON response."""
        raise NotImplementedError("当前 LLM provider 未实现结构化分析")

    @abstractmethod
    async def revise_storyboard(self, storyboard: Storyboard, feedback: str, reference_image: str | None = None) -> Storyboard:
        """Revise an existing storyboard based on user feedback."""
        pass

class ImageGenerator(ABC):
    @abstractmethod
    async def generate_image(
        self, 
        scene: Scene, 
        output_path: str,
        reference_image_path: str | None = None,
        seed: int | None = None,
        character_description: str | None = None,
        image_style: str | None = None,
        aspect_ratio: str | None = None,
    ) -> str:
        """Generate an image for the scene and return the file path."""
        pass

class VideoGenerator(ABC):
    @abstractmethod
    async def generate_video(self, scene: Scene, image_path: str, output_path: str) -> str:
        """Generate a video from the scene and status image, returning the file path."""
        pass
