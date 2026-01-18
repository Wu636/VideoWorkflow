# 短剧分镜视频自动化工作流

> 🎬 一键生成高质量短剧分镜视频的 AI 工作流系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## ✨ 核心特性

### 🤖 多模态 AI 生成
- **脚本生成**: 支持 DeepSeek、GLM-4V、火山方舟豆包/DeepSeek
- **图像生成**: 火山方舟 Seedream 4.5 高质量图像
- **视频生成**: 火山方舟 Seedance 1.5 Pro 视频合成

### 🎨 角色一致性保证
- **参考图支持**: 单图/多图/目录批量输入
- **图生图技术**: Seedream image-to-image 保持角色外观
- **固定随机种子**: 确保风格统一
- **角色描述前缀**: 自动添加详细外貌描述

### 🔄 交互式审阅工作流
- **脚本审阅**: AI 修改 / 手动编辑 / 确认
- **图像审阅**: 查看生成结果并可重新生成
- **角色描述**: 运行时自定义或使用配置

### ⚡ 高效并发处理
- 异步并发生成多个分镜
- 智能重试机制（视频下载）
- 详细的进度日志

## 🚀 快速开始

### 1. 安装

```bash
# 克隆项目
git clone https://github.com/your-username/VideoWorkflow.git
cd VideoWorkflow

# 安装依赖
pip install .
```

### 2. 配置

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 填入你的 API Keys
```

**必需配置**：
```bash
# 选择 LLM 提供商（三选一）
LLM_PROVIDER=ark  # 或 deepseek 或 glm

# 火山方舟配置（图像/视频生成必需）
ARK_API_KEY=your-ark-api-key
ARK_LLM_MODEL=deepseek-v3-2-251201  # 如果使用 ark 作为 LLM
```

### 3. 运行

```bash
# 完整交互模式（推荐）
python -m src.video_workflow.main "小狗送外卖" --count 2 --ref references/dog.png

# 自动化模式（跳过所有审阅）
python -m src.video_workflow.main "小狗送外卖" --count 2 --skip-review
```

## 📖 使用指南

### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `topic` | 视频主题（使用已有图像时可省略） | `"小狗送外卖"` |
| `--count, -c` | 分镜数量 | `--count 5` |
| `--ref, -r` | 参考图路径 | `--ref references/dog.png` |
| `--skip-review` | 跳过所有审阅步骤 | `--skip-review` |
| `--from-images, -i` | 从已有图像生成视频 | `-i outputs/12345` |

### 参考图使用方式

```bash
# 单张图片
--ref references/dog.png

# 多张图片（逗号分隔）
--ref "references/dog1.png,references/dog2.png"

# 目录（加载所有图片）
--ref references/
```

### 从已有图像生成视频

```bash
# 使用之前生成的图像直接生成视频
python -m src.video_workflow.main --from-images outputs/12345

# 或使用简写
python -m src.video_workflow.main -i outputs/12345
```

# 命令行直接指定
python -m src.video_workflow.main "橘猫开会" --template 搞笑剧场

# 或简写
python -m src.video_workflow.main "流浪狗的故事" -t 治愈系

# 交互式选择（不指定模板时自动提示）
python -m src.video_workflow.main "小狗送外卖"

**使用场景**：
- 已有满意的图像，想重新生成视频
- 手动修改过图像后生成视频
- 测试不同视频生成参数

### 工作流程

1. **角色描述设置** - 自定义或使用默认配置
2. **脚本生成与审阅** - AI 生成分镜脚本，支持修改
3. **图像生成与审阅** - 生成首帧图像，可重新生成
4. **视频生成** - 基于图像生成视频片段

## 🎯 高级功能

### 去除AI生成水印

**自动配置**：代码已默认去除水印（图像和视频）

如需开启水印，可在代码中修改：
```python
# image.py 和 video.py
"watermark": True  # 显示"AI生成"水印
```

### 角色一致性优化

在 `.env` 中配置：

```bash
# 角色外貌描述（会添加到每个分镜）
CHARACTER_DESCRIPTION=一只金色柯基犬，戴着红色围巾，大眼睛，圆润的脸

# 固定种子（保证风格一致）
IMAGE_SEED=42

# 参考图权重
IMAGE_STYLE_WEIGHT=0.8

# 图像宽高比
IMAGE_ASPECT_RATIO=16:9
```

### LLM 提供商选择

```bash
# 方案1: 火山方舟托管（推荐）
LLM_PROVIDER=ark
ARK_LLM_MODEL=deepseek-v3-2-251201  # 或 doubao-1-5-pro-32k

# 方案2: GLM 多模态
LLM_PROVIDER=glm
GLM_API_KEY=your-glm-key

# 方案3: DeepSeek 官方
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-deepseek-key
```

## 📂 项目结构

```
VideoWorkflow/
├── src/video_workflow/
│   ├── main.py              # CLI 入口
│   ├── config.py            # 配置管理
│   ├── types.py            # 数据模型
│   ├── core/
│   │   └── orchestrator.py  # 工作流编排
│   └── generators/
│       ├── llm.py          # LLM 生成器
│       ├── image.py        # 图像生成器
│       └── video.py        # 视频生成器
├── references/              # 参考图目录
├── outputs/                # 输出目录
│   └── <timestamp>/
│       ├── script.json     # 分镜脚本
│       ├── images/         # 生成的图像
│       └── videos/         # 生成的视频
├── .env.example            # 配置模板
└── README.md
```

## 🔧 故障排查

### 常见问题

1. **视频下载失败**
   - 已内置自动重试（3次）
   - 检查网络连接

2. **角色不一致**
   - 设置 `CHARACTER_DESCRIPTION`
   - 使用参考图（`--ref`）
   - 增加 `IMAGE_STYLE_WEIGHT`

3. **API 调用失败**
   - 检查 `.env` 中的 API Key
   - 确认模型 ID 正确
   - 查看火山方舟控制台配额

## 📝 更新日志

查看 [PROJECT_INTRO.md](PROJECT_INTRO.md) 了解详细功能说明。

## 📄 许可证

MIT License

## 🙏 致谢

- [火山方舟](https://www.volcengine.com/product/ark) - 图像/视频生成
- [DeepSeek](https://www.deepseek.com/) - LLM 服务
- [智谱 AI](https://www.zhipuai.cn/) - GLM 多模态

---

**🌟 如果这个项目对你有帮助，请给个 Star！**
