from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # DeepSeek
    DEEPSEEK_API_KEY: str
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # GLM (智谱AI)
    GLM_API_KEY: str | None = None
    GLM_MODEL: str = "glm-4v-plus"

    # LLM Provider Selection
    # Options: "deepseek" | "glm" | "ark_doubao" | "ark_deepseek"
    LLM_PROVIDER: str = "deepseek"
    
    # Ark LLM Model (火山方舟托管的模型)
    ARK_LLM_MODEL: str = "doubao-1-5-pro-32k"  # 豆包1.5/1.8 或 deepseek-v3

    # Volcengine Ark
    ARK_API_KEY: str
    ARK_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/v3"
    
    # Models
    ARK_VIDEO_MODEL: str = "doubao-seedance-1-5-pro"  # 豆宝-Seedance-1.5-pro
    ARK_IMAGE_MODEL: str = "doubao-seedream-4-5-251128"  # 豆宝-Seedream-4.5
    
    # Image Generation Parameters
    IMAGE_ASPECT_RATIO: str = "16:9"  # 支持: "1:1", "16:9", "9:16", "4:3", "3:4"
    IMAGE_STYLE: str | None = None  # 可选风格描述，如 "赛璐璐渲染", "工作室灯光"
    IMAGE_STYLE_WEIGHT: float = 0.7  # 参考图风格权重 (0.0-1.0)
    
    # Character Consistency Parameters
    IMAGE_SEED: str | None = None  # 固定随机种子，留空则自动生成
    CHARACTER_DESCRIPTION: str | None = None  # 角色外貌描述前缀

    # Workflow
    WORKFLOW_CONCURRENCY: int = 3
    OUTPUT_DIR: Path = Path("outputs")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
