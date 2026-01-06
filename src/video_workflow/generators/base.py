from abc import ABC, abstractmethod
from typing import Any, List
from src.video_workflow.types import Storyboard, Scene

class LLMGenerator(ABC):
    @abstractmethod
    async def generate_storyboard(self, topic: str, count: int = 5, reference_image: str | None = None, template: str | None = None) -> Storyboard:
        """Generate a storyboard based on the topic, optional reference image, and template."""
        pass
    
    async def analyze_reference_image(self, image_path: str) -> str | None:
        """Analyze reference image and return a character description. Default returns None."""
        return None

class ImageGenerator(ABC):
    @abstractmethod
    async def generate_image(
        self, 
        scene: Scene, 
        output_path: str,
        reference_image_path: str | None = None
    ) -> str:
        """Generate an image for the scene and return the file path."""
        pass

class VideoGenerator(ABC):
    @abstractmethod
    async def generate_video(self, scene: Scene, image_path: str, output_path: str) -> str:
        """Generate a video from the scene and status image, returning the file path."""
        pass
