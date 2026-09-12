from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # DeepSeek
    DEEPSEEK_API_KEY: str | None = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # GLM (智谱AI)
    GLM_API_KEY: str | None = None
    GLM_MODEL: str = "glm-4v-plus"

    # OpenLux (OpenAI-compatible gateway for GPT / Claude / Gemini / DeepSeek)
    OPENLUX_API_KEY: str | None = None
    OPENLUX_BASE_URL: str = "https://api.openlux.ai/v1"
    OPENLUX_MODEL: str = "claude-sonnet-5"
    OPENLUX_VISION_MODEL: str = "gpt-5.6-sol"
    # OpenLux 的旗舰模型可能需要数分钟。使用流式连接避免网关长时间无响应
    # 触发 SDK 自动重试；自动重试会让同一次用户操作产生多次计费调用。
    OPENLUX_REQUEST_TIMEOUT_SECONDS: float = 900.0
    OPENLUX_STREAM: bool = True
    OPENLUX_SAVE_RAW_RESPONSES: bool = True

    # LLM Provider Selection
    # Options: "deepseek" | "glm" | "openlux" | "ark_doubao" | "ark_deepseek"
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
    ARK_VIDEO_MODEL: str = "doubao-seedance-2-0-260128"
    ARK_IMAGE_MODEL: str = "doubao-seedream-4-5-251128"  # 豆宝-Seedream-4.5
    ARK_VISION_MODEL: str = "doubao-seed-1-6-251015"  # 豆包多模态（用于参考图分析）

    # Video Generation Provider
    # 可选: "grsai" | "ark"
    VIDEO_PROVIDER: str = "grsai"

    # GRSAI 图像模型（GPT Image / Nano Banana）
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
    SEEDANCE_DEFAULT_MODEL: str = "doubao-seedance-2-0-mini-260615"
    SEEDANCE_DEFAULT_RESOLUTION: str = "480p"
    # Number of Seedance jobs that may be submitted/polled concurrently. The
    # account/model quota may be lower; the queue backs off on provider limits.
    SEEDANCE_RENDER_CONCURRENCY: int = Field(default=3, ge=1, le=8)
    SEEDANCE_POLL_INTERVAL_SECONDS: float = 10.0
    SEEDANCE_JOB_TIMEOUT_SECONDS: int = 3600
    SEEDANCE_INLINE_ASSET_MAX_MB: int = 25
    # Ark video generation only accepts remotely downloadable HTTP(S) media.
    # Point this at the public origin/tunnel that exposes the signed
    # /api/projects/seedance-assets/* endpoint below.
    SEEDANCE_PUBLIC_ASSET_BASE_URL: str | None = None
    SEEDANCE_ASSET_URL_TTL_SECONDS: int = 86400
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
    # Model defaults can be overridden per shot.  pruned_int8 + nvfp4 is the
    # low-memory baseline; higher precision profiles require CPU offload or a
    # larger/multi-GPU host and are intentionally never selected implicitly.
    H3_MODEL_PROFILE: str = "pruned_int8"
    H3_TEXT_ENCODER_PROFILE: str = "nvfp4"
    H3_AUTO_SEGMENT_COMPLEX_SHOTS: bool = True
    H3_MAX_SEGMENT_SECONDS: float = 7.0
    H3_POSTPROCESS_AUDIO: bool = True
    H3_POSTPROCESS_AUDIO_BITRATE: str = "192k"
    # Preserve H3's generated speech, ambience and effects by default. Users
    # can explicitly select clean_tts for an independent replacement version,
    # or mute for a silent delivery copy.
    H3_AUDIO_MODE: str = "native"
    # comfyui_h3 renders on the self-hosted ComfyUI instance; atlas_h3 and
    # metaso_h3 are independent hosted MiniMax H3 API routes.
    H3_PROVIDER: str = "metaso_h3"
    ATLASCLOUD_API_KEY: str | None = None
    ATLASCLOUD_BASE_URL: str = "https://api.atlascloud.ai"
    ATLASCLOUD_POLL_INTERVAL_SECONDS: float = 2.0
    ATLASCLOUD_JOB_TIMEOUT_SECONDS: int = 1800
    # Atlas H3 submit parameters are user-selected in the settings UI; there is
    # no automatic mapping from the project resolution/aspect ratio.
    H3_ATLAS_RESOLUTION: str = "768P"
    H3_ATLAS_RATIO: str = "adaptive"
    # MetaSo proxies the official MiniMax H3 v2 content API. Keep Context IR
    # explicitly opt-in because the gateway bills it separately per request.
    METASO_H3_API_KEY: str | None = None
    METASO_H3_BASE_URL: str = "https://metaso.cn/api/minimax"
    METASO_H3_POLL_INTERVAL_SECONDS: float = 5.0
    METASO_H3_JOB_TIMEOUT_SECONDS: int = 1800
    METASO_H3_RESOLUTION: str = "768P"
    METASO_H3_RATIO: str = "adaptive"
    METASO_H3_CONTEXT_IR_ENABLED: bool = False
    TTS_PROVIDER: str = "edge"
    TTS_DEFAULT_FEMALE_VOICE: str = "zh-CN-XiaoxiaoNeural"
    TTS_DEFAULT_MALE_VOICE: str = "zh-CN-YunxiNeural"
    # Xiaoxiao's warm/news profile is the steadier mature-female baseline;
    # Xiaoyi is lively/cartoon-oriented and is selected only for young,
    # energetic character descriptions.
    TTS_DEFAULT_MATURE_FEMALE_VOICE: str = "zh-CN-XiaoxiaoNeural"
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
