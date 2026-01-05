import asyncio
import base64
import random
from pathlib import Path
from typing import List
import aiofiles
from volcenginesdkarkruntime import Ark
from src.video_workflow.config import settings
from src.video_workflow.types import Scene
from src.video_workflow.generators.base import ImageGenerator

class ArkImageGenerator(ImageGenerator):
    def __init__(self):
        self.client = Ark(
            api_key=settings.ARK_API_KEY,
            base_url=settings.ARK_BASE_URL
        )
        self.model = settings.ARK_IMAGE_MODEL
        
        # Generate a session seed for consistency if not configured
        seed_str = settings.IMAGE_SEED
        if seed_str and seed_str.strip():
            self._session_seed = int(seed_str.strip())
        else:
            self._session_seed = random.randint(1, 999999999)
        print(f"🎲 图像生成种子 (Seed): {self._session_seed}")
        
        # Aspect ratio mapping
        self.aspect_ratio_sizes = {
            "1:1": "2048x2048",    # 正方形 (4,194,304 pixels)
            "16:9": "2560x1440",   # 横向 (3,686,400 pixels)
            "9:16": "1440x2560",   # 竖向 (3,686,400 pixels)
            "4:3": "2304x1728",    # 横向 4:3 (3,981,312 pixels)
            "3:4": "1728x2304",    # 竖向 3:4 (3,981,312 pixels)
        }

    async def generate_image(
        self, 
        scene: Scene, 
        output_dir: str,
        reference_image_path: str | None = None
    ) -> str:
        loop = asyncio.get_running_loop()
        
        # Get size from aspect ratio
        size = self.aspect_ratio_sizes.get(settings.IMAGE_ASPECT_RATIO, "2048x1800")
        
        # Build enhanced prompt with character description prefix
        prompt = scene.visual_prompt
        
        # 1. Add character description prefix for consistency
        if settings.CHARACTER_DESCRIPTION:
            prompt = f"【角色特征】{settings.CHARACTER_DESCRIPTION}。\n【场景描述】{prompt}"
        
        # 2. Add style suffix
        if settings.IMAGE_STYLE:
            prompt = f"{prompt}。\n【画面风格】{settings.IMAGE_STYLE}"

        def _generate():
            # Build request parameters
            params = {
                "model": self.model,
                "prompt": prompt,
                "size": size,
                "response_format": "b64_json"
            }
            
            # 3. Add fixed seed for consistency
            # Note: Using extra_body to pass seed parameter
            extra_body = {
                "seed": self._session_seed,
                "watermark": False  # 去除AI生成水印
            }
            
            # 4. Add reference images if provided (支持多图融合)
            if reference_image_path:
                ref_images = self._load_reference_images(reference_image_path)
                if ref_images:
                    extra_body["reference_images"] = ref_images
                    extra_body["reference_weight"] = settings.IMAGE_STYLE_WEIGHT
                    # Use character mode for better identity preservation
                    extra_body["reference_mode"] = "character"
                    print(f"✅ 使用 {len(ref_images)} 张参考图 (权重: {settings.IMAGE_STYLE_WEIGHT}, 模式: character)")
            
            if extra_body:
                params["extra_body"] = extra_body
            
            return self.client.images.generate(**params)
        
        try:
            response = await loop.run_in_executor(None, _generate)
        except Exception as e:
            raise RuntimeError(f"Ark Image Gen Failed: {e}")

        # Extract Image Data
        try:
            b64_data = response.data[0].b64_json
            image_content = base64.b64decode(b64_data)
        except Exception as e:
            raise RuntimeError(f"Failed to parse image response: {e}. Response: {response}")

        # Save to file
        filename = f"{scene.id}_keyframe.png"
        filepath = Path(output_dir) / filename
        
        async with aiofiles.open(filepath, "wb") as f:
            await f.write(image_content)
            
        return str(filepath)
    
    def _load_reference_images(self, ref_path: str) -> List[dict]:
        """
        Load reference images. Supports:
        - Single image path: "references/dog.png"
        - Multiple images (comma-separated): "references/dog1.png,references/dog2.png"
        - Directory with images: "references/" (loads all .jpg/.png files)
        """
        ref_images = []
        path = Path(ref_path)
        
        if path.is_dir():
            # Load all images from directory
            image_files = list(path.glob("*.png")) + list(path.glob("*.jpg")) + list(path.glob("*.jpeg"))
            for img_file in image_files[:10]:  # Max 10 images
                ref_images.append(self._encode_image(img_file))
        elif "," in ref_path:
            # Multiple comma-separated paths
            for single_path in ref_path.split(","):
                single_path = single_path.strip()
                if Path(single_path).exists():
                    ref_images.append(self._encode_image(Path(single_path)))
        elif path.exists():
            # Single image
            ref_images.append(self._encode_image(path))
        else:
            print(f"⚠️  参考图不存在: {ref_path}")
        
        return [img for img in ref_images if img is not None]
    
    def _encode_image(self, path: Path) -> dict | None:
        """Encode image to base64 format for API"""
        try:
            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            
            # Determine MIME type
            suffix = path.suffix.lower()
            mime_type = "image/png" if suffix == ".png" else "image/jpeg"
            
            return {
                "url": f"data:{mime_type};base64,{img_b64}",
                "role": "character"  # 指定为角色参考图
            }
        except Exception as e:
            print(f"⚠️  无法加载参考图 {path}: {e}")
            return None
