# VideoWorkflow Studio

面向 AI 视频制作订单的一站式生产系统：从客户需求、角色与风格设定、详细分镜、分镜图、MiniMax H3 视频生成，到客户审片、自动剪辑和成片交付。

系统保留原有 DeepSeek / GLM / 火山方舟 / GRSAI 能力，新增 OpenLux（GPT / Claude / Gemini / DeepSeek）多模态大模型路由，以及自部署 MiniMax H3 的 I2V（首帧图生视频）与 R2V（全能参考）生产链路。

## 已实现的完整流程

1. 项目建档：录入客户、故事脚本、目标时长、画幅、风格、受众和交付要求。
2. 角色圣经：记录固定外貌、服饰、声音描述，并绑定人物参考图。
3. AI 分镜：按目标时长拆分镜头，每镜 `0.25–15s`，输出剧情、对白、场景、景别、机位、镜头、运镜、动作、声音、首帧 Prompt、H3 Prompt 和转场。
4. 分镜表审阅：逐镜修改、版本保存、CSV 导出、客户确认。
5. 素材库：上传人物图、画风图、场景图、动作视频、声音、音乐和字幕。
6. 分镜图：调用现有图像模型生成首帧，并将结果登记为项目资产。
7. H3 计划：AUTO 模式根据素材自动选择 I2V/R2V，计算合法帧数，并生成 `<Picture n>`、`<Video n>`、`<Audio n>` 标签。
8. 云端队列：上传输入、提交 ComfyUI、轮询状态、失败重试、取消任务、下载视频、记录种子/耗时/成本。
9. 客户审片：独立免登录链接，支持整体和逐镜确认或退回修改。
10. 自动成片：统一分辨率、帧率和音轨，按设计时长裁剪，顺序拼接、淡化、BGM、字幕烧录、预览版与 QC。
11. 交付：浏览器播放、下载成片、保留每次交付记录。
12. 旧项目迁移：自动扫描 `outputs/*/script.json`，复制旧分镜图和视频到新项目。
13. 剧本智能建档：上传 TXT / Markdown / DOCX / PDF，AI 分析后按勾选项回填视觉风格、节奏、受众、风格圣经、负面提示词、交付备注和角色一致性。
14. 智能拆镜数量：可手动指定，也可由 AI 根据目标时长、剧情转折、场景变化、动作和对白节奏推荐数量。
15. 统一模型设置：在 `/settings` 管理各环节 API、模型和 ComfyUI 地址；密钥仅显示配置状态，保存后即时生效。
16. 运行日志中心：在 `/logs` 实时筛选、搜索、暂停、下载和清空 API、生成队列与成片日志。

模型设置页顶部的“各功能当前优先配置”会分别显示剧本分析、AI 镜头数、详细分镜、角色参考图、分镜首帧、H3 视频、旧版视频与最终剪辑正在使用的服务和模型。可切换项直接在路由卡片中修改；自动模式会列出备用顺序和配置状态。

首次生成分镜和 AI 重做都会先打开“本次分镜建议”面板。用户可补充要保留的剧情、节奏、景别、运镜、对白或角色一致性要求；建议会和剧本、角色及风格设定一并进入模型 Prompt，AI 判断镜头数时也会参考这些要求。

客户剧本与目标时长不匹配时，可在“需求与角色”中使用“AI 扩写/缩写”。支持自动判断、只扩写、只缩写及用户改写建议；AI 先返回完整新剧本、预计可实现时长、改动摘要和制作提醒，用户对比原稿并确认后才会覆盖保存。

## 目录结构

```text
VideoWorkflow/
├── frontend/                         # Next.js 制作工作台与客户审片页
├── src/video_workflow/
│   ├── domain.py                     # 项目、分镜、素材、任务、审核、交付模型
│   ├── storage.py                    # SQLite 持久化仓库
│   ├── runtime_settings.py            # 浏览器统一配置、密钥脱敏与热更新
│   ├── logging_runtime.py             # 页面日志缓冲与轮转日志文件
│   ├── integrations/comfyui.py       # MiniMax H3 / ComfyUI 客户端
│   ├── services/
│   │   ├── projects.py               # 分镜、Prompt、模式路由、旧项目迁移
│   │   ├── render_queue.py           # 可恢复的云端渲染队列
│   │   └── finalize.py               # FFmpeg 自动剪辑与 QC
│   ├── server/routers/projects.py    # 新工作台 API
│   └── workflows/                    # 对齐官方 res_multistep 的 H3 API JSON（原生20步 / Turbo4步）
├── tests/                            # API、H3、迁移和成片测试
├── outputs/
│   ├── video_workflow.sqlite3        # 项目数据库
│   └── projects/<project_id>/        # 素材、分镜图、视频和交付文件
└── scripts/start-studio.sh           # 前后端一键启动
```

## 首次安装

推荐 Python 3.11 或 3.12。当前代码支持 Python 3.10+；Python 3.14 下第三方 `zhipuai` 会打印兼容性提醒，因此生产环境优先使用 3.12。

```bash
cd /Users/w/PycharmProjects/VideoWorkflow

python3.12 -m venv .venv
.venv/bin/pip install -e .
npm --prefix frontend install
cp .env.example .env
cp frontend/.env.example frontend/.env.local
```

系统还需要本机安装 `ffmpeg` / `ffprobe`。macOS 可执行：

```bash
brew install ffmpeg
```

## 必要配置

可以在工作台右上角“模型设置”统一配置并即时生效；也可以首次启动前在 `.env` 中提供默认值：

```dotenv
# OpenLux：填一个 Key 即可使用默认 Claude 文本模型和 GPT 多模态模型
OPENLUX_API_KEY=你的OpenLuxKey
OPENLUX_BASE_URL=https://api.openlux.ai/v1
OPENLUX_MODEL=claude-sonnet-5
OPENLUX_VISION_MODEL=gpt-5.6-sol
LLM_PROVIDER=openlux
BRIEF_ANALYSIS_PROVIDER=auto
SHOT_COUNT_PROVIDER=auto
REFERENCE_ANALYSIS_PROVIDER=auto

# 火山方舟继续用于 Seedance / Seedream，也可作为剧本模型备用
ARK_API_KEY=你的方舟Key
ARK_LLM_MODEL=你的模型或推理接入点
SEEDANCE_DEFAULT_MODEL=doubao-seedance-2-0-mini-260615
SEEDANCE_DEFAULT_RESOLUTION=720p

IMAGE_PROVIDER=ark
ARK_IMAGE_MODEL=doubao-seedream-4-5-251128

# 当前已实测实例；更换实例后只替换域名，不要带 # 后面的前端工作流 ID
COMFYUI_BASE_URL=https://u71482-7873955e7ee9.westd.seetacloud.com:8443
COMFYUI_VERIFY_TLS=true
COMFYUI_HOURLY_RATE=3.03

DATABASE_PATH=outputs/video_workflow.sqlite3
PROJECTS_DIR=outputs/projects
```

如果客户需要从其他电脑打开审片链接，在 `frontend/.env.local` 设置实际可访问域名：

```dotenv
NEXT_PUBLIC_API_URL=https://你的后端域名/api
NEXT_PUBLIC_APP_URL=https://你的工作台域名
```

仅在本机制作时保留默认 `localhost` 即可。

### Seedance 2.0 逐镜生成

进入项目的“视频生成”页后：选择 `Seedance 2.0 API` → 勾选镜头 → 选择
2.0 / Fast / Mini 与清晰度 → “编译所选 Prompt” → 核对实时费用 → 提交。
系统按镜头把任务写入本地持久队列，后台向方舟异步提交、轮询并下载 MP4；重启工作台后仍可继续查询。
Seedance Prompt 与 H3 Prompt 分开保存，素材编号严格按实际发送的 `图片1`、`视频1`、`音频1` 顺序生成。

MiniMax H3 支持三条互不覆盖的生成通道，可在“模型设置 → MiniMax H3 / ComfyUI → H3 生成通道”即时切换：

- `comfyui_h3`：自部署 ComfyUI。
- `metaso_h3`：MetaSo 的 `MiniMax-H3` v2 多模态接口，支持 `768P`（默认）/`2K`、首尾帧、参考图片/视频/音频、任务轮询和断点续查；Context IR 默认关闭。
- `atlas_h3`：Atlas Cloud 的 `minimax/h3-developer/reference-to-video`，原有实现与模型保持不变。

三条通道共用逐镜 Prompt、素材绑定、长镜头续帧和音频策略；默认保留 H3 原声，只有主动选择 `clean_tts` 时才输出独立配音替换版。

MiniMax H3 必须使用已经实测通过的文件：

- I2V Turbo LoRA：`minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors`
- R2V Turbo LoRA：`minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`
- R2V API 节点：`MiniMaxH3ReferenceToVideoAPI`

## 启动

```bash
cd /Users/w/PycharmProjects/VideoWorkflow
./scripts/start-studio.sh
```

打开 [http://localhost:3002](http://localhost:3002)。也可以分别启动：

```bash
.venv/bin/uvicorn src.video_workflow.server.app:app --reload --port 8001
npm --prefix frontend run dev
```

- 制作工作台：`http://localhost:3002`
- 后端 API：`http://localhost:8001`
- API 文档：`http://localhost:8001/docs`
- 模型设置：`http://localhost:3002/settings`
- 运行日志：`http://localhost:3002/logs`
- 保留的旧版工作台：`http://localhost:3002/legacy`

## 实际使用顺序

1. 新建项目，粘贴故事，或进入项目后上传 TXT / Markdown / DOCX / PDF 剧本。
2. 在“参考素材”上传人物和风格参考图。
3. 在“需求与角色”添加角色、绑定人物参考图，点击“AI 分析并回填”，审阅并勾选需要应用的需求和角色草稿。
4. 在“分镜设计”选择“AI 判断镜头数”或“手动指定镜头数”，生成详细分镜；逐镜修改后导出 CSV 给客户。
5. 在“客户确认”复制审片链接；客户确认后继续。
6. 生成分镜首帧，或上传客户确认过的分镜图。
7. 在“H3 生成”逐镜选择：
   - `I2V`：绑定一张首帧；
   - `R2V`：多选人物图、场景图、动作视频、声音；
   - `AUTO`：有全能参考则走 R2V，否则走 I2V。
8. 点击“编译计划”，确认模式、帧数和提示词；服务器开机后执行“检查服务器”。
9. 提交镜头，队列依次上传、生成并下载到本地。
10. 所有镜头完成后，在“剪辑交付”生成字幕、选择 BGM，再生成最终成片。
11. 客户在同一审片链接确认成片后，项目自动进入“已交付”。

## 云服务器什么时候需要开机

日常建档、写分镜、上传本地素材、客户审阅、编辑提示词和本地成片处理都不依赖云端 H3，AutoDL 实例可以关闭。

以下环节再开机：

- 点击“检查服务器”；
- 正式提交 H3 视频任务；
- 最后一次真实 I2V/R2V 验收。

如果任务已经加入本地队列而实例处于关闭状态，任务会保持 `queued`；连接失败不消耗模型重试次数。实例恢复后队列会继续尝试。

## H3 时长与帧数

H3 的有效长度采用 `17k + 5` 帧网格，工作台自动计算：

- 不足 5 秒：生成最小 H3 片段，再由成片器裁剪到设计时长；
- 5 秒约 124 帧；
- 15 秒为 362 帧；
- 单镜头编辑器强制上限 15 秒。

## H3 速度与画质参数

“H3 生成”页面支持逐镜保存参数，也可以把预设批量应用到当前勾选镜头：

- **快速预览**：608 长边、Turbo 4 步，适合先检查动作、构图和人物一致性；
- **均衡**：使用项目分辨率与 Turbo 4 步，适合低成本确认镜头；
- **清晰优先**：1344 长边、官方原生 20 步，绕过加速 LoRA，作为最终成片默认；
- **自定义**：宽高、5–15 秒时长/帧数、原生/Turbo 链路、步数、scheduler、denoise、固定/随机 seed、LoRA 强度和 R2V 参考图尺寸。

卡片会显示相对于 768×448、124 帧、4 步基线的预计算力倍率。提交任务时参数会写入任务快照，因此排队后再修改镜头不会悄悄改变已经提交的任务。相同 seed 和相同参数便于复现、对比 Prompt 或参考素材。

官方 Turbo LoRA 针对 4 步训练，必须与 4 步配套；最终质量模式使用无 LoRA 的原生 20 步。两条链路都采用官方 `res_multistep` Sampler、`simple` Scheduler 与 Denoise 1。旧版自定义 TurboSampler、Sigma Shift 和 low-VRAM 合并参数不再进入实际提交工作流。画质不足时仍应优先使用完整高清首帧、缩短并简化动作、减少互相冲突的参考素材。

## 对白、杂音与成片质量门禁

- 默认 `native` 直接保留 H3 生成的人声、环境音和动作音效。需要独立配音版时再主动选择 `clean_tts`；处理前会在视频同目录保留 `.native-audio.mp4` 原声备份，再按明确的发言者生成 48 kHz 双声道 AAC 配音版。
- 同一镜头有多名发言者时，工作台会按发言者拆成连续短段，当前段只允许对应角色开口，其他人物保持闭嘴。
- 任一应有对白未成功生成时，任务不会以静音或缺台词状态完成，而会换 Seed 重试；分辨率、时长、帧、音轨、采样率或完整解码不合格也会被媒体门禁拦截。
- 最终剪辑使用 `FINAL_VIDEO_PRESET=slow`、`FINAL_VIDEO_CRF=14` 的高质量编码；最终 QC 未通过时不会进入交付状态。

## 数据、费用和备份

- 本地项目数据在 `outputs/`，不写入代码目录。
- 页面配置保存在 `outputs/runtime_settings.json`（权限 `0600`），密钥不会由查询接口回传；运行日志在 `outputs/logs/video_workflow.log` 轮转保存。
- 云端大模型继续放 AutoDL 文件存储盘 `/root/autodl-fs`，不要下载到系统盘。
- 数据库使用 SQLite WAL，后端重启会恢复中断的渲染任务。
- 每个 H3 任务记录 `seed`、尝试次数、Comfy prompt ID、耗时和按小时单价估算的费用。
- 备份项目只需备份 `outputs/video_workflow.sqlite3` 与 `outputs/projects/`。

## Docker

```bash
docker compose up --build
```

浏览器通过宿主机端口访问后端，默认 `NEXT_PUBLIC_API_URL=http://localhost:8001/api`、`NEXT_PUBLIC_APP_URL=http://localhost:3002`。如果部署到其他主机，在构建前将两者设置为客户可访问的公开 URL。

## 自动测试

```bash
.venv/bin/python -m compileall -q src tests
.venv/bin/python -W error::ResourceWarning -m unittest discover -s tests -v
npm --prefix frontend run lint
npm --prefix frontend run build
```

测试覆盖：SQLite CRUD 与连接释放、项目 API、剧本上传、统一配置脱敏、参考图参与剧本分析、素材上传、公开审片、I2V/R2V 工作流构建、标准 UUID Prompt ID、H3 帧数、全能参考标签顺序、重复编译幂等性、项目级分镜图比例传递、旧项目迁移、FFmpeg 两镜成片和 QC。

## 真实链路验收记录（2026-08-22）

已通过本项目 API（不是在 ComfyUI 页面手工执行）完成：项目建档 → 素材上传 → I2V/R2V 自动路由 → 原生20步或 Turbo4步提交 → 状态轮询 → 本地下载 → 独立对白替换 → 自动字幕 → 两镜转场合片 → 预览版 → QC → 客户公开页确认。

- I2V：768×448，24fps，124 帧，H.264 + AAC，5.167 秒；云端推理约 506 秒。
- R2V：`MiniMaxH3ReferenceToVideoAPI` + `<Picture 1>` / `ref_image_1`，768×448，24fps，124 帧，H.264 + AAC，5.167 秒；云端推理约 225 秒。
- 最终成片：9.776 秒，含音轨与烧录字幕；QC 通过，原片及客户预览版下载接口均返回完整文件。
- 真实脚本拆镜：方舟 LLM 将 10 秒剧情拆成 4.667 秒与 5.333 秒两镜，详细画面、运镜、动作、声音、转场和生成 Prompt 均已入库。
- 真实分镜首帧：方舟图像接口生成 2560×1440 横屏首帧，项目设置为 16:9 时不再受全局 9:16 环境变量干扰。
- 实测期间修复了标准 UUID Prompt ID、`ref_image_size` 被动态字段清理误删、重复编译导致 H3 Prompt 叠加、动作文本时长与实际镜头不一致、分镜图比例未继承项目设置、图像文件后缀与真实编码不一致、宿主机与 Docker 绝对路径互不兼容等问题，并添加回归测试。

## 常见问题

### “服务器离线”

开发阶段属于正常状态。正式生成前启动 AutoDL 实例，然后在 H3 页面点击“检查服务器”。

### 预检提示缺少节点或模型

核对 `MiniMaxH3ReferenceToVideoAPI` 自定义节点和两份 Turbo LoRA；再检查 VAE、文本编码器、扩散模型是否仍是失效软链接。

### R2V 参考没有生效

确认镜头最终模式显示为 `R2V`，素材已在“全能参考”勾选，Prompt 存在同序号的 `<Picture n>` / `<Video n>` / `<Audio n>`。

### 成片按钮不可用

每个分镜都需要已完成的视频输出。可从旧项目导入已有视频，也可等待 H3 队列全部完成。

### Python 3.14 出现 zhipuai 提醒

这是第三方 SDK 的兼容性提醒；改用 Python 3.12 虚拟环境即可消除，项目自身测试不受影响。
