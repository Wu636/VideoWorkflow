import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

def concatenate_videos(session_dir: str, video_files: List[Path]) -> str:
    """
    使用 ffmpeg 将所有分镜视频按顺序拼接成一个完整视频
    
    Args:
        session_dir: 会话目录路径
        video_files: 视频文件路径列表
        
    Returns:
        str: 最终视频的绝对路径
    """
    session_path = Path(session_dir)
    output_path = session_path / "final_video.mp4"
    
    # 创建临时文件列表供 ffmpeg 使用
    # 注意：在 Windows 上 NamedTemporaryFile 默认无法被其他进程打开，需 delete=False 并手动清理
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        filelist_path = f.name
        for video_file in video_files:
            # ffmpeg concat 格式需要绝对路径并转义单引号
            abs_path = str(video_file.absolute()).replace("'", "'\\''")
            f.write(f"file '{abs_path}'\n")
    
    try:
        logger.info("正在拼接视频...")
        
        # 使用 ffmpeg concat demuxer 拼接视频
        cmd = [
            "ffmpeg", "-y",  # 覆盖输出文件
            "-f", "concat",
            "-safe", "0",
            "-i", filelist_path,
            "-c", "copy",  # 直接复制流，不重新编码
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            # 如果直接复制失败，尝试重新编码
            logger.warning("直接合并失败，尝试重新编码...")
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", filelist_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg 拼接失败: {result.stderr}")
        
        logger.info(f"完整视频已生成: {output_path}")
        return str(output_path)
        
    finally:
        # 清理临时文件
        Path(filelist_path).unlink(missing_ok=True)
