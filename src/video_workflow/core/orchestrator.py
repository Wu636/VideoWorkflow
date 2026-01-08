import asyncio
import logging
from pathlib import Path
from typing import List

from src.video_workflow.config import settings
from src.video_workflow.types import Storyboard, Scene, GenerationStatus
from src.video_workflow.generators.llm import DeepSeekGenerator, GLMGenerator, ArkLLMGenerator
from src.video_workflow.generators.image import ArkImageGenerator
from src.video_workflow.generators.video import ArkVideoGenerator

logger = logging.getLogger(__name__)

class WorkflowOrchestrator:
    def __init__(self):
        # Select LLM based on configuration
        if settings.LLM_PROVIDER == "glm":
            self.llm = GLMGenerator()
        elif settings.LLM_PROVIDER in ("ark_doubao", "ark_deepseek", "ark"):
            self.llm = ArkLLMGenerator()
        else:
            self.llm = DeepSeekGenerator()
        
        self.image_gen = None
        self.video_gen = None
        
        self.output_dir = settings.OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)

    async def initialize(self):
        """Lazy initialization of generators that might require async setup or valid keys."""
        try:
            self.image_gen = ArkImageGenerator()
            self.video_gen = ArkVideoGenerator()
        except Exception as e:
            logger.warning(f"Failed to init generators (likely missing keys): {e}")

    async def process_scene(self, scene: Scene, session_dir: Path, reference_image: str | None = None):
        """Pipeline for a single scene: Image -> Video"""
        if not self.image_gen or not self.video_gen:
            logger.error("生成器未初始化")
            scene.error_message = "生成器未初始化"
            scene.image_status = GenerationStatus.FAILED
            return

        # 1. Generate Image
        try:
            logger.info(f"正在生成场景 {scene.id} 的首帧图像...")
            scene.image_status = GenerationStatus.PROCESSING
            image_dir = session_dir / "images"
            image_dir.mkdir(exist_ok=True)
            
            img_path = await self.image_gen.generate_image(scene, str(image_dir), reference_image)
            scene.image_path = img_path
            scene.image_status = GenerationStatus.COMPLETED
        except Exception as e:
            logger.error(f"场景 {scene.id} 图像生成失败: {e}")
            scene.image_status = GenerationStatus.FAILED
            scene.error_message = str(e)
            return

        # 2. Generate Video
        try:
            logger.info(f"正在生成场景 {scene.id} 的视频片段...")
            scene.video_status = GenerationStatus.PROCESSING
            video_dir = session_dir / "videos"
            video_dir.mkdir(exist_ok=True)
            
            vid_path = await self.video_gen.generate_video(scene, scene.image_path, str(video_dir))
            scene.video_path = vid_path
            scene.video_status = GenerationStatus.COMPLETED
        except Exception as e:
            logger.error(f"场景 {scene.id} 视频生成失败: {e}")
            scene.video_status = GenerationStatus.FAILED
            scene.error_message = str(e)

    async def run(self, topic: str, count: int = 5, reference_image: str | None = None):
        await self.initialize()
        
        session_id = str(int(asyncio.get_event_loop().time()))
        session_dir = self.output_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"开始执行工作流，主题: {topic} (Session: {session_id})")
        if reference_image:
            logger.info(f"使用参考图: {reference_image}")

        # 1. Generate Storyboard
        try:
            logger.info("正在生成分镜脚本...")
            storyboard = await self.llm.generate_storyboard(topic, count, reference_image)
            
            # Save script
            with open(session_dir / "script.json", "w") as f:
                f.write(storyboard.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"分镜脚本生成失败: {e}")
            return

        # 2. Concurrent Processing
        semaphore = asyncio.Semaphore(settings.WORKFLOW_CONCURRENCY)
        
        async def _bounded_process(scene: Scene):
            async with semaphore:
                await self.process_scene(scene, session_dir, reference_image)
        
        tasks = [_bounded_process(scene) for scene in storyboard.scenes]
        await asyncio.gather(*tasks)
        
        logger.info("工作流执行完成。")
        return session_dir

    async def run_generation(self, storyboard: Storyboard, reference_image: str | None = None):
        """
        仅执行图像和视频生成部分（用于审阅后继续执行）- 保留向后兼容
        """
        session_dir, _ = await self.run_image_generation(storyboard, reference_image)
        await self.run_video_generation(storyboard, session_dir)
        return session_dir

    async def run_image_generation(self, storyboard: Storyboard, reference_image: str | None = None, existing_session_dir: str | None = None, scene_ids: list[int] | None = None):
        """
        仅生成图像（支持图像审阅工作流）
        
        Args:
            storyboard: 分镜脚本
            reference_image: 参考图路径
            existing_session_dir: 已有的会话目录（用于重新生成）
            scene_ids: 要生成的场景ID列表，None表示全部生成
            
        Returns: (session_dir, success_flag)
        """
        await self.initialize()
        
        # Create or use existing session directory
        if existing_session_dir:
            session_dir = Path(existing_session_dir)
        else:
            session_id = str(int(asyncio.get_event_loop().time()))
            session_dir = self.output_dir / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the reviewed script
        with open(session_dir / "script.json", "w") as f:
            f.write(storyboard.model_dump_json(indent=2))
        
        logger.info(f"开始生成图像 (Session: {session_dir.name})")
        if reference_image:
            logger.info(f"使用参考图: {reference_image}")
        
        # 筛选要生成的场景
        if scene_ids:
            scenes_to_generate = [s for s in storyboard.scenes if s.id in scene_ids]
            logger.info(f"选择性重新生成: 场景 {scene_ids}")
        else:
            scenes_to_generate = storyboard.scenes
        
        # Generate images concurrently
        semaphore = asyncio.Semaphore(settings.WORKFLOW_CONCURRENCY)
        
        async def _bounded_image_gen(scene: Scene):
            async with semaphore:
                if not self.image_gen:
                    logger.error("生成器未初始化")
                    scene.image_status = GenerationStatus.FAILED
                    return
                
                try:
                    logger.info(f"正在生成场景 {scene.id} 的首帧图像...")
                    scene.image_status = GenerationStatus.PROCESSING
                    image_dir = session_dir / "images"
                    image_dir.mkdir(exist_ok=True)
                    
                    img_path = await self.image_gen.generate_image(scene, str(image_dir), reference_image)
                    scene.image_path = img_path
                    scene.image_status = GenerationStatus.COMPLETED
                except Exception as e:
                    logger.error(f"场景 {scene.id} 图像生成失败: {e}")
                    scene.image_status = GenerationStatus.FAILED
                    scene.error_message = str(e)
        
        tasks = [_bounded_image_gen(scene) for scene in scenes_to_generate]
        await asyncio.gather(*tasks)
        
        # Save updated script
        with open(session_dir / "script.json", "w") as f:
            f.write(storyboard.model_dump_json(indent=2))
        
        # Check if all succeeded
        all_success = all(scene.image_status == GenerationStatus.COMPLETED for scene in storyboard.scenes)
        
        logger.info("图像生成完成。")
        return str(session_dir), all_success

    async def run_video_generation(self, storyboard: Storyboard, session_dir: str):
        """
        仅生成视频（在图像审阅后调用）
        """
        await self.initialize()
        session_path = Path(session_dir)
        
        logger.info(f"开始生成视频 (Session: {session_path.name})")
        
        # Generate videos concurrently
        semaphore = asyncio.Semaphore(settings.WORKFLOW_CONCURRENCY)
        
        async def _bounded_video_gen(scene: Scene):
            async with semaphore:
                if not self.video_gen or not scene.image_path:
                    logger.error(f"场景 {scene.id} 无法生成视频")
                    scene.video_status = GenerationStatus.FAILED
                    return
                
                try:
                    logger.info(f"正在生成场景 {scene.id} 的视频片段...")
                    scene.video_status = GenerationStatus.PROCESSING
                    video_dir = session_path / "videos"
                    video_dir.mkdir(exist_ok=True)
                    
                    vid_path = await self.video_gen.generate_video(scene, scene.image_path, str(video_dir))
                    scene.video_path = vid_path
                    scene.video_status = GenerationStatus.COMPLETED
                except Exception as e:
                    logger.error(f"场景 {scene.id} 视频生成失败: {e}")
                    scene.video_status = GenerationStatus.FAILED
                    scene.error_message = str(e)
        
        tasks = [_bounded_video_gen(scene) for scene in storyboard.scenes]
        await asyncio.gather(*tasks)
        
        # Save final script
        with open(session_path / "script.json", "w") as f:
            f.write(storyboard.model_dump_json(indent=2))
        
        logger.info("视频生成完成。")
