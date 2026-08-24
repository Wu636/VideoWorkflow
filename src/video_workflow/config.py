from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # DeepSeek
    DEEPSEEK_API_KEY: str | None = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # GLM (智谱AI)
    GLM_API_KEY: str | None = None
    GLM_MODEL: str = "glm-4v-plus"

    # LLM Provider Selection
    # Options: "deepseek" | "glm" | "ark_doubao" | "ark_deepseek"
    LLM_PROVIDER: str = "deepseek"
    # Per-function routing. "auto" resolves from configured/ready providers.
    BRIEF_ANALYSIS_PROVIDER: str = "auto"
    SHOT_COUNT_PROVIDER: str = "auto"
    REFERENCE_ANALYSIS_PROVIDER: str = "auto"
    
    # Ark LLM Model (火山方舟托管的模型)
    ARK_LLM_MODEL: str = "doubao-1-5-pro-32k"  # 豆包1.5/1.8 或 deepseek-v3

    # Volcengine Ark
    ARK_API_KEY: str | None = None
    ARK_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/v3"
    
    # Models
    ARK_VIDEO_MODEL: str = "doubao-seedance-1-5-pro"  # 豆宝-Seedance-1.5-pro
    ARK_IMAGE_MODEL: str = "doubao-seedream-4-5-251128"  # 豆宝-Seedream-4.5
    ARK_VISION_MODEL: str = "doubao-seed-1-6-251015"  # 豆包多模态（用于参考图分析）

    # Video Generation Provider
    # 可选: "grsai" | "ark"
    VIDEO_PROVIDER: str = "grsai"

    # GRSAI Nano Banana
    GRSAI_API_KEY: str | None = None
    GRSAI_BASE_URL: str = "https://grsai.dakka.com.cn"
    GRSAI_IMAGE_MODEL: str = "nano-banana-fast"
    GRSAI_VIDEO_MODEL: str = "veo3.1-fast"
    GRSAI_VIDEO_ASPECT_RATIO: str = "16:9"
    GRSAI_VIDEO_WEBHOOK: str = "-1"
    GRSAI_VIDEO_SHUT_PROGRESS: bool = False
    GRSAI_WEBHOOK_TOKEN: str | None = None
    GRSAI_IMAGE_SIZE: str = "1K"
    GRSAI_RESULT_POLL_INTERVAL_SECONDS: float = 2.0
    
    # Image Generation Parameters
    IMAGE_PROVIDER: str = "grsai"  # 可选: "ark" | "grsai"
    IMAGE_ASPECT_RATIO: str = "16:9"  # 支持: "1:1", "16:9", "9:16", "4:3", "3:4"
    IMAGE_STYLE: str | None = None  # 可选风格描述，如 "赛璐璐渲染", "工作室灯光"
    IMAGE_STYLE_WEIGHT: float = 0.7  # 参考图风格权重 (0.0-1.0)
    
    # Character Consistency Parameters
    IMAGE_SEED: str | None = None  # 固定随机种子，留空则自动生成
    CHARACTER_DESCRIPTION: str | None = None  # 角色外貌描述前缀

    # Workflow
    WORKFLOW_CONCURRENCY: int = 3
    IMAGE_GENERATION_TIMEOUT_SECONDS: int = 180
    IMAGE_TIMEOUT_RETRY_COUNT: int = 1
    IMAGE_TIMEOUT_RETRY_DELAY_SECONDS: float = 1.5
    VIDEO_GENERATION_TIMEOUT_SECONDS: int = 900
    OUTPUT_DIR: Path = Path("outputs")

    # Project workspace / durable state
    DATABASE_PATH: Path = Path("outputs/video_workflow.sqlite3")
    PROJECTS_DIR: Path = Path("outputs/projects")

    # Self-hosted ComfyUI + MiniMax H3
    COMFYUI_BASE_URL: str = "http://127.0.0.1:6006"
    COMFYUI_API_TOKEN: str | None = None
    COMFYUI_REQUEST_TIMEOUT_SECONDS: float = 120.0
    COMFYUI_POLL_INTERVAL_SECONDS: float = 2.0
    COMFYUI_JOB_TIMEOUT_SECONDS: int = 3600
    COMFYUI_VERIFY_TLS: bool = True
    COMFYUI_TRUST_ENV: bool = False
    COMFYUI_UPLOAD_FIELD: str = "image"
    COMFYUI_H3_I2V_WORKFLOW: Path | None = None
    COMFYUI_H3_R2V_WORKFLOW: Path | None = None
    COMFYUI_HOURLY_RATE: float = 3.03
    H3_RENDER_CONCURRENCY: int = 1
    H3_DEFAULT_WIDTH: int = 1344
    H3_DEFAULT_HEIGHT: int = 768
    H3_DEFAULT_FPS: float = 24.0
    H3_DEFAULT_REF_IMAGE_SIZE: str = "match"
    H3_AUTO_SEGMENT_COMPLEX_SHOTS: bool = True
    H3_MAX_SEGMENT_SECONDS: float = 7.0
    H3_POSTPROCESS_AUDIO: bool = True
    H3_POSTPROCESS_AUDIO_BITRATE: str = "192k"
    # clean_tts discards H3's unstable synthetic audio and overlays independent
    # dialogue; mute writes a clean silent track; native preserves H3 audio.
    H3_AUDIO_MODE: str = "clean_tts"
    TTS_PROVIDER: str = "edge"
    TTS_DEFAULT_FEMALE_VOICE: str = "zh-CN-XiaoxiaoNeural"
    TTS_DEFAULT_MALE_VOICE: str = "zh-CN-YunxiNeural"
    TTS_DEFAULT_MATURE_FEMALE_VOICE: str = "zh-CN-XiaoyiNeural"
    TTS_DEFAULT_MATURE_MALE_VOICE: str = "zh-CN-YunyangNeural"

    # Final render
    FFMPEG_BIN: str = "ffmpeg"
    FFPROBE_BIN: str = "ffprobe"
    FINAL_VIDEO_CODEC: str = "libx264"
    FINAL_VIDEO_PRESET: str = "slow"
    FINAL_VIDEO_CRF: int = 14
    FINAL_AUDIO_CODEC: str = "aac"
    FINAL_AUDIO_SAMPLE_RATE: int = 48000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
