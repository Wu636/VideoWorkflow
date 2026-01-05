# 自动短剧分镜视频生成 Agent (Video Workflow Agent)

## 📖 项目简介
本项目是一个高度自动化的 AI 视频生产工作流 Agent，旨在通过简单的文本主题输入，自动生成完整的短剧分镜视频。核心流程包括：由大模型生成包含视觉和动态描述的分镜脚本，利用 AI 绘画生成一致性首帧，最后合成高质量的视频片段。

## ✨ 核心特性

- **多模态脚本生成**：
  - 支持 **GLM-4V** 多模态模型，可分析参考图中的角色特征（外貌、服装、风格），生成精准的分镜描述。
  - 兼容 **DeepSeek**（纯文本模式）进行快速脚本创作。
  
- **一致性图像生成**：
  - 集成 **火山方舟 (Doubao Seedream)** 图像生成模型。
  - 支持 **图生图 (Image-to-Image)** 技术，利用参考图确保所有分镜中的角色形象和画风高度统一。

- **高质量视频生成**：
  - 集成 **火山方舟 (Doubao Seedance)** 视频生成模型。
  - 自动根据分镜的动态描述（Motion Prompt）生成 5-8 秒的高清视频片段。
  - 支持异步并发任务提交与状态轮询，大幅提升生成效率。

- **自动化工作流**：
  - 一键启动：`python -m src.video_workflow.main "主题"`
  - 自动编排：脚本 -> 图像 -> 视频 的全链路自动化。
  - 容错机制：自动重试、超时控制和详细的日志记录。

## 🏗 技术架构

项目基于 Python 3.10+ 开发，采用模块化设计：

### 核心模块
1.  **`generators` (生成器层)**：
    -   `LLMGenerator`: 抽象基类，适配 DeepSeek 和 GLM-4V。
    -   `ImageGenerator`: 适配火山方舟 Doubao Seedream。
    -   `VideoGenerator`: 适配火山方舟 Doubao Seedance。
2.  **`core` (核心层)**：
    -   `WorkflowOrchestrator`: 负责任务编排、并发控制、状态管理和文件保存。
3.  **`config` (配置层)**：
    -   基于 `pydantic-settings` 管理环境变量和应用配置。
4.  **`main.py` (接口层)**：
    -   基于 `Typer` 构建的命令行工具 (CLI)。

### 技术栈
- **语言**: Python
- **框架**: Pydantic, Typer, Asyncio
- **SDK**: `volcengine-python-sdk[ark]`, `openai`, `zhipuai`
- **工具**: Rich (美化终端输出)

## 🚀 快速开始

### 1. 环境准备
```bash
# 1. 安装依赖
pip install -r requirements.txt
# 或直接安装
pip install .

# 2. 创建目录
mkdir -p references outputs
```

### 2. 配置密钥
复制 `.env.example` 为 `.env` 并填入密钥：

```bash
# LLM 配置 (二选一)
LLM_PROVIDER=glm
GLM_API_KEY=your_glm_key_here
# 或
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_key_here

# 火山方舟配置 (图像/视频生成)
ARK_API_KEY=your_ark_api_key_here
ARK_IMAGE_MODEL=doubao-seedream-4-5-251128
ARK_VIDEO_MODEL=doubao-seedance-1-5-pro
```

### 3. 运行指南

#### 基础模式（仅文本主题）
适用于不需要特定角色形象的通用场景。
```bash
python -m src.video_workflow.main "赛博朋克风格的雨夜侦探" --count 5
```

#### 参考图模式（推荐 ⭐）
适用于需要固定角色形象（如 IP 角色、特定人物）的短剧。
1. 将角色参考图放入 `references` 目录（例如 `hero.jpg`）。
2. 运行命令并指定参考图：
```bash
python -m src.video_workflow.main "赛博朋克风格的雨夜侦探" --count 5 --ref references/hero.jpg
```
Agent 将自动分析 `hero.jpg` 的特征，并在生成的所有分镜图片和视频中保持该角色的外观一致。

## 📂 目录结构

```
VideoWorkflow/
├── references/          # 存放参考图片
├── outputs/             # 输出产物目录
│   └── <timestamp>/     # 按时间戳隔离的每次运行结果
│       ├── script.json  # 生成的分镜脚本
│       ├── images/      # 分镜首帧图
│       └── videos/      # 最终视频片段
├── src/
│   └── video_workflow/
│       ├── core/        # 编排逻辑
│       ├── generators/  # 各类 AI 模型适配器
│       └── ...
├── .env                 # 配置文件
└── README.md            # 项目说明
```

## 🆕 最新功能

### 1. 多 LLM 提供商支持
支持多种 LLM 提供商用于脚本生成：

| 提供商 | 配置值 | 特点 |
|--------|--------|------|
| 火山方舟 LLM | `LLM_PROVIDER=ark` | 豆包 1.5/1.8、DeepSeek V3 |
| GLM 多模态 | `LLM_PROVIDER=glm` | 支持参考图输入 |
| DeepSeek 官方 | `LLM_PROVIDER=deepseek` | 纯文本模式 |

### 2. 交互式脚本审阅
生成脚本后暂停，提供四种操作：
- ✅ **确认** - 继续生成图像和视频
- 🤖 **AI 修改** - 输入修改建议，AI 自动调整
- ✏️ **手动编辑** - 编辑 JSON 文件
- ❌ **取消** - 退出工作流

### 3. 交互式图像审阅 🆕
图像生成后暂停审阅：
- ✅ **确认图像** - 继续生成视频
- 🔄 **重新生成** - 输入修改建议重新生成所有图像
- ❌ **取消** - 退出工作流

### 4. 角色外貌描述交互 🆕
工作流开始时提示：
- 使用 `.env` 中的默认描述
- 或现场输入自定义描述
- 描述会自动添加到每个分镜开头

### 5. 角色一致性优化
**固定随机种子**：
```bash
IMAGE_SEED=42  # 所有分镜使用相同种子
```

**多参考图支持**：
```bash
# 单图
--ref references/dog.png
# 多图
--ref "references/dog1.png,references/dog2.png"
# 目录
--ref references/
```

**角色描述前缀**：
```bash
CHARACTER_DESCRIPTION=一只金色柯基犬，戴着红色围巾，大眼睛，圆润的脸
```

**图像生成参数**：
```bash
IMAGE_ASPECT_RATIO=16:9         # 宽高比
IMAGE_STYLE=赛璐璐渲染           # 风格描述
IMAGE_STYLE_WEIGHT=0.8          # 参考图权重
```

### 6. 视频下载重试机制
- 自动重试 3 次
- 指数退避策略（1s → 2s → 4s）
- 超时时间 120 秒

## 💡 使用示例

```bash
# 完整交互模式（推荐新用户）
python -m src.video_workflow.main "小狗送外卖" --count 2 --ref references/dog.png

# 全自动模式（跳过所有审阅）
python -m src.video_workflow.main "小狗送外卖" --count 2 --skip-review

# 多参考图模式
python -m src.video_workflow.main "橘猫做菜" --count 3 --ref references/

# 查看帮助
python -m src.video_workflow.main --help
```

## 🎯 完整工作流程

1. **角色描述设置** - 选择使用默认或自定义
2. **生成脚本** → 审阅/修改脚本 → 确认
3. **生成图像** → 查看图像 → 确认或重新生成
4. **生成视频** → 自动下载到 `outputs/` 目录
