import asyncio
import base64
import os
import time
from pathlib import Path
import aiofiles
from volcenginesdkarkruntime import Ark
from src.video_workflow.config import settings
from src.video_workflow.types import Scene
from src.video_workflow.generators.base import VideoGenerator

class ArkVideoGenerator(VideoGenerator):
    def __init__(self):
        self.client = Ark(
            api_key=settings.ARK_API_KEY,
            base_url=settings.ARK_BASE_URL
        )
        self.model = settings.ARK_VIDEO_MODEL

    async def generate_video(self, scene: Scene, image_path: str, output_dir: str) -> str:
        # 1. Read and Encode Image
        async with aiofiles.open(image_path, "rb") as f:
            img_data = await f.read()
            img_b64 = base64.b64encode(img_data).decode("utf-8")
        
        # Format as Data URI
        # Assuming png for now, or detect from extension
        ext = Path(image_path).suffix.lstrip(".") or "png"
        img_data_uri = f"data:image/{ext};base64,{img_b64}"

        # 2. Submit Task
        # Note: The SDK usage in async context. 
        # The official SDK might be synchronous. We should run it in an executor.
        loop = asyncio.get_running_loop()

        def _submit_task():
            return self.client.content_generation.tasks.create(
                model=self.model,
                content=[
                    {
                        "type": "text", 
                        "text": scene.motion_prompt  # Remove unsupported parameters
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": img_data_uri}
                    }
                ]
            )

        try:
            create_result = await loop.run_in_executor(None, _submit_task)
        except Exception as e:
            raise RuntimeError(f"Ark API Submit Failed: {e}")

        task_id = create_result.id
        print(f"Task Submitted: {task_id}")

        # 3. Poll Status with timeout
        video_url = None
        max_wait_time = 600  # 10 minutes timeout
        poll_interval = 10  # Poll every 10 seconds instead of 3
        start_time = time.time()
        
        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                raise TimeoutError(f"视频生成超时（{max_wait_time}秒）")
            
            # Polling delay
            await asyncio.sleep(poll_interval)
            
            def _get_status():
                return self.client.content_generation.tasks.get(task_id=task_id)
            
            try:
                result = await loop.run_in_executor(None, _get_status)
            except Exception as e:
                print(f"轮询失败: {e}")
                continue
                
            status = result.status
            print(f"视频任务 {task_id} 状态: {status} (已等待 {int(elapsed)}秒)")
            
            if status == "succeeded":
                # Extract Video URL from Content object
                try:
                    video_url = result.content.video_url
                    print(f"✅ 成功获取视频URL")
                except Exception as e:
                    print(f"提取视频URL失败: {e}")
                    print(f"响应结构: {result}")
                    raise RuntimeError(f"无法从响应中提取视频URL")
                break
            elif status == "failed":
                error_msg = getattr(result, 'error', '未知错误')
                print(f"❌ 视频生成失败: {error_msg}")
                raise RuntimeError(f"视频生成失败: {error_msg}")
            else:
                # running, pending, etc.
                continue

        if not video_url:
            raise RuntimeError("Video URL not found in success response")

        # 4. Download Video with retry
        max_retries = 3
        video_content = None
        
        for attempt in range(max_retries):
            try:
                print(f"📥 正在下载视频... (尝试 {attempt + 1}/{max_retries})")
                import httpx
                async with httpx.AsyncClient(timeout=120.0) as client:  # 增加超时到120秒
                    resp = await client.get(video_url)
                    resp.raise_for_status()
                    video_content = resp.content
                
                print(f"✅ 视频下载成功 ({len(video_content)} bytes)")
                break  # 成功则跳出重试循环
            except Exception as e:
                print(f"⚠️  下载失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避: 1s, 2s, 4s
                    print(f"⏳ 等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"❌ 视频下载失败，已达最大重试次数")
                    raise RuntimeError(f"视频下载失败: {e}")
        
        if not video_content:
            raise RuntimeError("视频下载失败：未获取到内容")

        # 5. Save Video
        try:
            output_path = Path(output_dir) / f"{scene.id}_video.mp4"
            async with aiofiles.open(output_path, "wb") as f:
                await f.write(video_content)
            
            print(f"💾 视频已保存: {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"❌ 视频保存失败: {e}")
            raise RuntimeError(f"视频保存失败: {e}")
