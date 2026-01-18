import asyncio
import json
import typer
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.syntax import Syntax
from src.video_workflow.core.orchestrator import WorkflowOrchestrator
from src.video_workflow.types import Storyboard

console = Console()


async def analyze_reference_image(image_path: str) -> str | None:
    """使用多模态 LLM 分析参考图，自动生成角色描述。支持豆包和 GLM。"""
    from src.video_workflow.config import settings
    import base64
    
    image_file = Path(image_path)
    if not image_file.exists():
        console.print(f"[red]参考图不存在: {image_path}[/red]")
        return None
    
    with open(image_file, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    # 要求返回 JSON 格式，同时包含角色和风格
    analysis_prompt = """请仔细分析这张图片，提取以下两部分信息，以 JSON 格式返回：

1. character（角色外貌描述，50-100字）：
   - 包含：物种/类型、体型、毛色/肤色、五官特征、服装配饰、表情气质

2. style（视觉风格描述，20-50字）：
   - 分析图片整体视觉风格（如：3D卡通渲染、写实摄影、水彩手绘、赛璐璐动画等）
   - 包含：画面质感、光影风格、色彩特点

请严格按照以下 JSON 格式输出，不要添加其他文字：
{"character": "角色描述内容", "style": "风格描述内容"}"""
    
    loop = asyncio.get_running_loop()
    
    # 方案1: 优先使用智谱 GLM 多模态 (glm-4.7)
    if settings.GLM_API_KEY:
        try:
            from zhipuai import ZhipuAI
            client = ZhipuAI(api_key=settings.GLM_API_KEY)
            
            def _call_glm():
                # 根据文件扩展名确定 MIME 类型
                ext = image_file.suffix.lower()
                mime_type = "image/png" if ext == ".png" else "image/jpeg"
                
                response = client.chat.completions.create(
                    model=settings.GLM_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_b64}"}},
                                {"type": "text", "text": analysis_prompt}
                            ]
                        }
                    ],
                    stream=False
                )
                return response.choices[0].message.content
            
            console.print(f"[dim]使用智谱 GLM 多模态模型 ({settings.GLM_MODEL}) 分析...[/dim]")
            result = await loop.run_in_executor(None, _call_glm)
            
            # 解析 JSON 返回
            if result:
                import json
                result_str = result.strip()
                if result_str.startswith("```json"):
                    result_str = result_str[7:]
                if result_str.startswith("```"):
                    result_str = result_str[3:]
                if result_str.endswith("```"):
                    result_str = result_str[:-3]
                result_str = result_str.strip()
                
                try:
                    return json.loads(result_str)
                except:
                    return {"character": result_str, "style": None}
            return None
        except Exception as e:
            console.print(f"[yellow]⚠️ GLM 分析失败: {e}，尝试豆包...[/yellow]")
    
    # 方案2: 回退到豆包多模态 (doubao-seed-1-6-251015)
    if settings.ARK_API_KEY:
        try:
            from volcenginesdkarkruntime import Ark
            client = Ark(api_key=settings.ARK_API_KEY, base_url=settings.ARK_BASE_URL)
            
            def _call_doubao():
                response = client.chat.completions.create(
                    model=settings.ARK_VISION_MODEL,  # 豆包多模态模型
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                                {"type": "text", "text": analysis_prompt}
                            ]
                        }
                    ]
                )
                return response.choices[0].message.content
            
            console.print(f"[dim]使用豆包多模态模型 ({settings.ARK_VISION_MODEL}) 分析...[/dim]")
            result = await loop.run_in_executor(None, _call_doubao)
            
            # 解析 JSON 返回
            if result:
                import json
                result_str = result.strip()
                if result_str.startswith("```json"):
                    result_str = result_str[7:]
                if result_str.startswith("```"):
                    result_str = result_str[3:]
                if result_str.endswith("```"):
                    result_str = result_str[:-3]
                result_str = result_str.strip()
                
                try:
                    return json.loads(result_str)
                except:
                    return {"character": result_str, "style": None}
        except Exception as e:
            console.print(f"[yellow]⚠️ 豆包分析失败: {e}[/yellow]")
    
    console.print("[yellow]⚠️ 未配置多模态 API (GLM 或 ARK)，无法自动分析参考图[/yellow]")
    return None


def concatenate_videos(session_dir: str, video_files: list) -> str:
    """使用 ffmpeg 将所有分镜视频按顺序拼接成一个完整视频"""
    import subprocess
    import tempfile
    
    session_path = Path(session_dir)
    output_path = session_path / "final_video.mp4"
    
    # 创建临时文件列表供 ffmpeg 使用
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        filelist_path = f.name
        for video_file in video_files:
            # ffmpeg concat 格式需要绝对路径并转义单引号
            abs_path = str(video_file.absolute()).replace("'", "'\\''")
            f.write(f"file '{abs_path}'\n")
    
    try:
        console.print("[dim]正在拼接视频...[/dim]")
        
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
            console.print("[yellow]直接合并失败，尝试重新编码...[/yellow]")
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
        
        return str(output_path)
        
    finally:
        # 清理临时文件
        Path(filelist_path).unlink(missing_ok=True)


def display_script(storyboard: Storyboard):
    """美化显示分镜脚本"""
    console.print("\n[bold cyan]═══════════════ 分镜脚本预览 ═══════════════[/bold cyan]\n")
    
    for scene in storyboard.scenes:
        console.print(Panel(
            f"[bold]旁白：[/bold]{scene.narrative}\n\n"
            f"[bold]画面描述：[/bold]{scene.visual_prompt[:100]}...\n\n"
            f"[bold]动态描述：[/bold]{scene.motion_prompt[:100]}...",
            title=f"[yellow]场景 {scene.id}[/yellow]",
            border_style="blue"
        ))
    
    console.print("\n[bold cyan]═══════════════════════════════════════════[/bold cyan]\n")

def main(
    topic: str = typer.Argument(None, help="视频主题（使用已有图像时可省略）"),
    count: int = typer.Option(5, "--count", "-c", help="生成的场景数量"),
    reference_image: str = typer.Option(None, "--ref", "-r", help="参考图路径（用于保持角色一致性）"),
    skip_review: bool = typer.Option(False, "--skip-review", help="跳过所有审阅步骤，直接生成"),
    from_images: str = typer.Option(None, "--from-images", "-i", help="从已有图像目录生成视频（如 outputs/12345）"),
    template: str = typer.Option(None, "--template", "-t", help="爆款脚本模板（反转剧/萌宠日常/治愈系/猎奇科普/搞笑剧场/情感共鸣）")
):
    """
    为指定主题运行视频生成工作流。
    """
    orchestrator = WorkflowOrchestrator()
    
    try:
        # 模式1: 从已有图像生成视频
        if from_images:
            console.print(f"[bold green]📁 从已有图像生成视频[/bold green]")
            console.print(f"[cyan]图像目录：{from_images}[/cyan]")
            
            session_path = Path(from_images)
            
            if not session_path.exists():
                console.print(f"[red]错误：目录不存在 {from_images}[/red]")
                return
            
            # 读取已有的 script.json
            script_file = session_path / "script.json"
            if not script_file.exists():
                console.print(f"[red]错误：未找到 script.json 文件[/red]")
                return
            
            import json
            with open(script_file, "r", encoding="utf-8") as f:
                script_data = json.load(f)
            
            from src.video_workflow.types import Storyboard
            storyboard = Storyboard(**script_data)
            
            # 检查和加载图像路径
            images_dir = session_path / "images"
            if not images_dir.exists():
                console.print(f"[red]错误：未找到 images 目录[/red]")
                return
            
            image_files = sorted(images_dir.glob("*_keyframe.png"))
            if not image_files:
                console.print(f"[red]错误：未找到首帧图像文件[/red]")
                return
            
            # 将图像路径关联到场景
            for idx, scene in enumerate(storyboard.scenes):
                if idx < len(image_files):
                    scene.image_path = str(image_files[idx])
                    console.print(f"  ✅ 场景 {scene.id}: {image_files[idx].name}")
            
            console.print(f"\n[bold yellow]开始生成视频...[/bold yellow]")
            
            # 初始化并生成视频
            asyncio.run(orchestrator.initialize())
            asyncio.run(orchestrator.run_video_generation(storyboard, str(session_path)))
            
            console.print(f"\n[bold blue]✅ 成功！[/bold blue] 视频已保存至：{session_path / 'videos'}")
            return
        
        # 模式2: 完整工作流
        if not topic:
            console.print("[red]错误：请指定视频主题或使用 --from-images 参数[/red]")
            return
        
        console.print(f"[bold green]正在启动视频生成工作流，主题：[/bold green] {topic}")
        if reference_image:
            console.print(f"[bold cyan]使用参考图：[/bold cyan] {reference_image}")
        
        # 初始化
        asyncio.run(orchestrator.initialize())
        
        # 0. 自动分析参考图生成角色描述（如果提供了参考图且没有跳过审阅）
        character_desc = None
        if not skip_review:
            from src.video_workflow.config import settings
            import src.video_workflow.config as config_module
            
            # 如果有参考图，先自动分析
            if reference_image:
                console.print("\n[bold cyan]🔍 正在分析参考图，自动生成角色和风格描述...[/bold cyan]")
                try:
                    analysis_result = asyncio.run(analyze_reference_image(reference_image))
                    if analysis_result:
                        # 处理返回结果（可能是字符串或字典）
                        if isinstance(analysis_result, dict):
                            auto_character = analysis_result.get("character", "")
                            auto_style = analysis_result.get("style", "")
                        else:
                            auto_character = str(analysis_result)
                            auto_style = ""
                        
                        # 显示分析结果（角色+风格一起显示）
                        if auto_character or auto_style:
                            console.print(f"\n[green]✅ AI 自动分析结果：[/green]")
                            console.print(f"[bold]角色描述：[/bold]{auto_character or '（未识别）'}")
                            console.print(f"[bold]视觉风格：[/bold]{auto_style or '（未识别）'}")
                            
                            # 编辑审阅循环
                            while True:
                                console.print("\n[cyan]请选择操作：[/cyan]")
                                console.print("  [dim]1[/dim] - ✅ 确认使用以上设置")
                                console.print("  [dim]2[/dim] - ✏️  编辑角色描述和风格（打开编辑器）")
                                console.print("  [dim]3[/dim] - 📝 手动输入")
                                console.print("  [dim]4[/dim] - ❌ 跳过，不使用")
                                
                                choice = Prompt.ask("请选择", choices=["1", "2", "3", "4"], default="1")
                                
                                if choice == "1":
                                    character_desc = auto_character
                                    if auto_style:
                                        config_module.settings.IMAGE_STYLE = auto_style
                                    console.print("[green]✅ 设置已确认[/green]")
                                    break
                                    
                                elif choice == "2":
                                    # 创建临时文件让用户编辑
                                    temp_file = Path("temp_character_style.txt")
                                    content = f"""# 角色描述和视觉风格设置
# 请直接修改下方内容，保存后关闭文件

【角色描述】
{auto_character}

【视觉风格】
{auto_style or '（未识别，可手动填写）'}
"""
                                    temp_file.write_text(content, encoding="utf-8")
                                    console.print(f"\n[cyan]已创建编辑文件：[bold]{temp_file}[/bold][/cyan]")
                                    console.print("[cyan]请修改后保存，然后按回车继续...[/cyan]")
                                    input()
                                    
                                    # 读取修改后的内容
                                    try:
                                        edited = temp_file.read_text(encoding="utf-8")
                                        lines = edited.split("\n")
                                        in_char, in_style = False, False
                                        char_lines, style_lines = [], []
                                        
                                        for line in lines:
                                            if line.startswith("#"):
                                                continue
                                            if "【角色描述】" in line:
                                                in_char, in_style = True, False
                                                continue
                                            if "【视觉风格】" in line:
                                                in_char, in_style = False, True
                                                continue
                                            if in_char:
                                                char_lines.append(line)
                                            elif in_style:
                                                style_lines.append(line)
                                        
                                        auto_character = "\n".join(char_lines).strip()
                                        auto_style = "\n".join(style_lines).strip()
                                        if auto_style.startswith("（"):
                                            auto_style = ""
                                        
                                        temp_file.unlink()
                                        
                                        # 显示修改后内容供审阅
                                        console.print(f"\n[green]修改后的设置：[/green]")
                                        console.print(f"[bold]角色描述：[/bold]{auto_character}")
                                        console.print(f"[bold]视觉风格：[/bold]{auto_style or '（无）'}")
                                        
                                    except Exception as e:
                                        console.print(f"[red]读取失败: {e}[/red]")
                                        if temp_file.exists():
                                            temp_file.unlink()
                                
                                elif choice == "3":
                                    auto_character = Prompt.ask("[yellow]角色描述[/yellow]", default=auto_character)
                                    auto_style = Prompt.ask("[yellow]视觉风格（可跳过）[/yellow]", default=auto_style or "")
                                    console.print(f"\n[green]设置完成：[/green]")
                                    console.print(f"[bold]角色描述：[/bold]{auto_character}")
                                    console.print(f"[bold]视觉风格：[/bold]{auto_style or '（无）'}")
                                
                                elif choice == "4":
                                    break
                            
                except Exception as e:
                    console.print(f"[yellow]⚠️  自动分析失败: {e}，使用手动模式[/yellow]")
            
            # 如果没有参考图或分析失败，使用原有逻辑
            if not character_desc:
                if settings.CHARACTER_DESCRIPTION:
                    console.print(f"\n[cyan]📝 当前配置的角色描述：[/cyan]{settings.CHARACTER_DESCRIPTION}")
                    use_default = Confirm.ask("是否使用此描述？", default=True)
                    if use_default:
                        character_desc = settings.CHARACTER_DESCRIPTION
                    else:
                        # 让用户在原描述基础上修改
                        character_desc = Prompt.ask("[yellow]请修改角色描述[/yellow]", default=settings.CHARACTER_DESCRIPTION)
                else:
                    if Confirm.ask("\n是否需要自定义角色外貌描述？", default=False):
                        character_desc = Prompt.ask("[yellow]请输入角色外貌描述[/yellow]")
            
            if character_desc:
                config_module.settings.CHARACTER_DESCRIPTION = character_desc
                console.print(f"[green]✅ 角色描述已设置[/green]")
        
        # 2. 选择分镜数量（交互模式）
        if not skip_review:
            count_input = Prompt.ask(
                "\n[cyan]📊 请输入分镜数量[/cyan]",
                default=str(count)
            )
            try:
                count = int(count_input)
                if count < 1:
                    count = 5
            except:
                count = 5
            console.print(f"[green]✅ 将生成 {count} 个分镜[/green]")
        
        # 3. 选择爆款模板
        selected_template = template
        if not skip_review and not selected_template:
            from src.video_workflow.templates import VIRAL_TEMPLATES, get_template_description
            console.print("\n[bold cyan]📋 选择爆款脚本模板：[/bold cyan]")
            console.print("  [dim]0[/dim] - 不使用模板（自由创作）")
            for idx, (name, tmpl) in enumerate(VIRAL_TEMPLATES.items(), 1):
                console.print(f"  [dim]{idx}[/dim] - [bold]{name}[/bold] - {tmpl.description}")
            
            template_choice = Prompt.ask(
                "请选择模板编号", 
                choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"],
                default="0"
            )
            
            if template_choice != "0":
                template_names = list(VIRAL_TEMPLATES.keys())
                selected_template = template_names[int(template_choice) - 1]
                console.print(f"[green]✅ 已选择模板：{selected_template}[/green]")
        
        if selected_template:
            console.print(f"[cyan]📋 使用爆款模板：{selected_template}[/cyan]")
        
        # 3.5 是否包含台词
        include_dialogue = True
        if not skip_review:
            # 如果选择了"萌宠开口说话"(8号)模板，默认开启台词
            default_dialogue = True if selected_template == "萌宠开口说话" else False
            include_dialogue = Confirm.ask(
                "\n[cyan]💬 脚本中是否包含角色台词？[/cyan]\n[dim](Yes=生成台词+声音描述, No=纯视觉动作叙事)[/dim]", 
                default=default_dialogue
            )
        
        # 2. 生成分镜脚本
        console.print("\n[bold yellow]步骤 1/4：生成分镜脚本...[/bold yellow]")
        storyboard = asyncio.run(
            orchestrator.llm.generate_storyboard(topic, count, reference_image, selected_template, include_dialogue)
        )
        
        # 2. 脚本审阅（除非跳过）
        if not skip_review:
            storyboard = review_script_loop(orchestrator, storyboard, topic, reference_image)
            if storyboard is None:
                console.print("[yellow]用户取消了工作流。[/yellow]")
                return
        
        # 3. 生成图像并审阅
        console.print("\n[bold yellow]步骤 2/4：生成分镜图像...[/bold yellow]")
        session_dir, images_generated = asyncio.run(
            orchestrator.run_image_generation(storyboard, reference_image)
        )
        
        if not images_generated:
            console.print("[red]图像生成失败。[/red]")
            return
        
        # 图像审阅（除非跳过）
        if not skip_review:
            approved = review_images_loop(session_dir, storyboard, orchestrator, reference_image)
            if not approved:
                console.print("[yellow]用户取消了工作流。[/yellow]")
                return
        
        # 4. 生成视频（可选择性生成）
        console.print("\n[bold yellow]步骤 3/4：生成视频片段...[/bold yellow]")
        
        # 显示可用图像并让用户选择要生成视频的场景
        if not skip_review:
            console.print("\n[cyan]可生成视频的分镜：[/cyan]")
            for scene in storyboard.scenes:
                if scene.image_path:
                    console.print(f"  [dim]{scene.id}[/dim] - 📷 {Path(scene.image_path).name}")
            
            console.print("\n[bold]请选择操作：[/bold]")
            console.print("  [green]1[/green] - ✅ 生成全部分镜的视频")
            console.print("  [yellow]2[/yellow] - 🎯 选择指定分镜生成视频")
            console.print("  [red]3[/red] - ❌ 跳过视频生成")
            
            video_choice = Prompt.ask("请选择", choices=["1", "2", "3"], default="1")
            
            if video_choice == "3":
                console.print("[yellow]跳过视频生成。[/yellow]")
                console.print(f"\n[bold blue]✅ 图像已保存至：{session_dir}[/bold blue]")
                return
            
            video_scene_ids = None
            if video_choice == "2":
                scene_ids_input = Prompt.ask(
                    "[yellow]请输入要生成视频的分镜编号（如：1,3,5）[/yellow]",
                    default=",".join(str(s.id) for s in storyboard.scenes if s.image_path)
                )
                try:
                    video_scene_ids = [int(x.strip()) for x in scene_ids_input.split(",")]
                    video_scene_ids = [x for x in video_scene_ids if 1 <= x <= len(storyboard.scenes)]
                    console.print(f"[green]✅ 将生成分镜 {video_scene_ids} 的视频[/green]")
                except:
                    console.print("[yellow]格式错误，将生成全部视频[/yellow]")
                    video_scene_ids = None
            
            asyncio.run(orchestrator.run_video_generation(storyboard, session_dir, scene_ids=video_scene_ids))
        else:
            asyncio.run(orchestrator.run_video_generation(storyboard, session_dir))
        
        # 视频审阅循环
        if not skip_review:
            while True:
                # 显示已生成的视频
                videos_dir = Path(session_dir) / "videos"
                video_files = sorted(list(videos_dir.glob("*_video.mp4"))) if videos_dir.exists() else []
                
                console.print("\n[bold cyan]═══════════════ 生成的视频 ═══════════════[/bold cyan]\n")
                for vf in video_files:
                    console.print(f"  🎬 {vf.name}")
                console.print(f"\n[cyan]👉 请在文件浏览器中查看: {videos_dir}[/cyan]")
                console.print("[bold cyan]═══════════════════════════════════════════[/bold cyan]\n")
                
                console.print("[bold]请选择操作：[/bold]")
                console.print("  [green]1[/green] - ✅ 确认全部视频，进入拼接步骤")
                console.print("  [yellow]2[/yellow] - 🖼️  重新生成某个分镜的首帧图，再重新生成视频")
                console.print("  [cyan]3[/cyan] - 🎬 直接重新生成某个分镜的视频（保持首帧图不变）")
                console.print("  [red]4[/red] - ⏭️  跳过视频审阅，直接拼接")
                
                video_review_choice = Prompt.ask("请选择", choices=["1", "2", "3", "4"], default="1")
                
                if video_review_choice == "1" or video_review_choice == "4":
                    break
                
                elif video_review_choice == "2":
                    # 重新生成首帧图再生成视频
                    scene_id_input = Prompt.ask("[yellow]请输入要重新生成的分镜编号[/yellow]", default="1")
                    try:
                        scene_id = int(scene_id_input.strip())
                        if 1 <= scene_id <= len(storyboard.scenes):
                            scene = storyboard.scenes[scene_id - 1]
                            
                            # 选择参考图
                            console.print("\n[bold]请选择参考图来源：[/bold]")
                            console.print("  [green]1[/green] - 🖼️  使用原始参考图")
                            console.print("  [yellow]2[/yellow] - 📷 使用已生成的某个分镜图像")
                            ref_choice = Prompt.ask("请选择", choices=["1", "2"], default="1")
                            
                            regen_ref = reference_image
                            if ref_choice == "2":
                                ref_scene_id = Prompt.ask("请输入要作为参考的分镜编号", default="1")
                                try:
                                    ref_id = int(ref_scene_id.strip())
                                    ref_scene = storyboard.scenes[ref_id - 1]
                                    if ref_scene.image_path:
                                        regen_ref = ref_scene.image_path
                                        console.print(f"[green]✅ 使用 {Path(regen_ref).name} 作为参考图[/green]")
                                except:
                                    pass
                            
                            # 修改建议
                            feedback = Prompt.ask("[yellow]请输入修改建议（可回车跳过）[/yellow]", default="")
                            if feedback.strip():
                                scene.visual_prompt = f"{scene.visual_prompt.split(chr(10) + '修改要求：')[0]}\n修改要求：{feedback}"
                            
                            # 重新生成图像
                            console.print(f"[bold]正在重新生成分镜 {scene_id} 的首帧图...[/bold]")
                            asyncio.run(orchestrator.run_image_generation(
                                storyboard, regen_ref, session_dir, scene_ids=[scene_id]
                            ))
                            
                            # 显示生成的图像并让用户确认
                            images_dir = Path(session_dir) / "images"
                            regenerated_image = images_dir / f"{scene_id}_keyframe.png"
                            
                            console.print(f"\n[cyan]📷 首帧图已重新生成：{regenerated_image}[/cyan]")
                            console.print("[cyan]请在文件浏览器中查看图像[/cyan]\n")
                            
                            console.print("[bold]请选择操作：[/bold]")
                            console.print("  [green]1[/green] - ✅ 确认图像，继续生成视频")
                            console.print("  [yellow]2[/yellow] - 🔄 重新生成该图像")
                            console.print("  [red]3[/red] - ❌ 取消，返回视频审阅")
                            
                            img_confirm_choice = Prompt.ask("请选择", choices=["1", "2", "3"], default="1")
                            
                            if img_confirm_choice == "1":
                                # 重新生成视频
                                console.print(f"[bold]正在重新生成分镜 {scene_id} 的视频...[/bold]")
                                asyncio.run(orchestrator.run_video_generation(
                                    storyboard, session_dir, scene_ids=[scene_id]
                                ))
                                console.print(f"[green]✅ 分镜 {scene_id} 已重新生成！[/green]")
                            elif img_confirm_choice == "2":
                                console.print("[yellow]请重新选择操作重新生成图像[/yellow]")
                            else:
                                console.print("[yellow]已取消视频生成[/yellow]")
                        else:
                            console.print("[red]无效的分镜编号[/red]")
                    except Exception as e:
                        console.print(f"[red]错误：{e}[/red]")
                
                elif video_review_choice == "3":
                    # 只重新生成视频
                    scene_ids_input = Prompt.ask("[yellow]请输入要重新生成视频的分镜编号（如：1,3）[/yellow]", default="1")
                    try:
                        scene_ids = [int(x.strip()) for x in scene_ids_input.split(",")]
                        scene_ids = [x for x in scene_ids if 1 <= x <= len(storyboard.scenes)]
                        if scene_ids:
                            console.print(f"[bold]正在重新生成分镜 {scene_ids} 的视频...[/bold]")
                            asyncio.run(orchestrator.run_video_generation(
                                storyboard, session_dir, scene_ids=scene_ids
                            ))
                            console.print(f"[green]✅ 分镜 {scene_ids} 的视频已重新生成！[/green]")
                        else:
                            console.print("[red]无效的分镜编号[/red]")
                    except Exception as e:
                        console.print(f"[red]错误：{e}[/red]")
        
        # 步骤 4/4：询问是否拼接所有视频
        console.print("\n[bold yellow]步骤 4/4：视频拼接...[/bold yellow]")
        
        # 检查是否有生成的视频
        videos_dir = Path(session_dir) / "videos"
        video_files = sorted(list(videos_dir.glob("*_video.mp4"))) if videos_dir.exists() else []
        
        if len(video_files) > 1:
            console.print(f"\n[cyan]已生成 {len(video_files)} 个分镜视频[/cyan]")
            if Confirm.ask("是否将所有分镜视频按顺序拼接成一个完整视频？", default=True):
                try:
                    final_video_path = concatenate_videos(session_dir, video_files)
                    console.print(f"[green]✅ 拼接完成！完整视频：{final_video_path}[/green]")
                except Exception as e:
                    console.print(f"[red]❌ 拼接失败：{e}[/red]")
        elif len(video_files) == 1:
            console.print("[dim]只有1个视频，无需拼接[/dim]")
        
        console.print(f"\n[bold blue]✅ 成功！[/bold blue] 输出已保存至：{session_dir}")
            
    except KeyboardInterrupt:
        console.print("\n[bold yellow]用户中断了工作流。[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]错误：[/bold red] {e}")
        import traceback
        traceback.print_exc()


def review_script_loop(orchestrator, storyboard: Storyboard, topic: str, reference_image: str | None) -> Storyboard | None:
    """交互式脚本审阅循环 - 使用临时文件让用户完整查看和编辑"""
    
    temp_file = Path("temp_script_review.json")
    
    while True:
        # 将脚本保存到临时文件供用户审阅
        import json
        temp_file.write_text(json.dumps(storyboard.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
        
        console.print("\n[bold cyan]═══════════════ 分镜脚本预览 ═══════════════[/bold cyan]\n")
        console.print(f"[cyan]📄 完整脚本已保存到临时文件：[bold]{temp_file.absolute()}[/bold][/cyan]")
        console.print(f"[dim]请用编辑器打开查看完整内容（共 {len(storyboard.scenes)} 个分镜）[/dim]\n")
        
        # 在控制台显示简要概览
        for scene in storyboard.scenes:
            duration_str = f"[{scene.duration}s]" if hasattr(scene, 'duration') else ""
            narrative_preview = scene.narrative[:50] + "..." if len(scene.narrative) > 50 else scene.narrative
            console.print(f"  [dim]{scene.id}[/dim] {duration_str} {narrative_preview}")
        
        console.print("\n[bold cyan]═══════════════════════════════════════════[/bold cyan]\n")
        
        console.print("[bold]请选择操作：[/bold]")
        console.print("  [green]1[/green] - ✅ 确认脚本，继续生成图像和视频")
        console.print("  [yellow]2[/yellow] - 🤖 输入修改建议，让 AI 修改脚本")
        console.print("  [cyan]3[/cyan] - ✏️  我已编辑临时文件，重新加载")
        console.print("  [red]4[/red] - ❌ 取消并退出")
        
        choice = Prompt.ask("请输入选项", choices=["1", "2", "3", "4"], default="1")
        
        if choice == "1":
            console.print("[green]✅ 脚本已确认，开始生成...[/green]")
            # 删除临时文件
            if temp_file.exists():
                temp_file.unlink()
            return storyboard
        
        elif choice == "2":
            # AI 修改模式
            feedback = Prompt.ask("[yellow]请输入修改建议[/yellow]")
            if feedback.strip():
                console.print("[bold]正在根据您的建议修改脚本...[/bold]")
                try:
                    storyboard = asyncio.run(
                        revise_storyboard(orchestrator, storyboard, feedback, reference_image)
                    )
                    console.print("[green]✅ 脚本已更新！请查看修改后的内容：[/green]")
                except Exception as e:
                    console.print(f"[red]修改失败: {e}[/red]")
        
        elif choice == "3":
            # 从临时文件重新加载
            try:
                with open(temp_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                storyboard = Storyboard(**data)
                console.print("[green]✅ 脚本已从文件加载！[/green]")
            except Exception as e:
                console.print(f"[red]加载失败: {e}[/red]")
        
        elif choice == "4":
            # 删除临时文件
            if temp_file.exists():
                temp_file.unlink()
            return None


def review_images_loop(session_dir, storyboard: Storyboard, orchestrator, reference_image: str | None) -> bool:
    """交互式图像审阅循环，支持选择性重新生成"""
    from pathlib import Path
    
    while True:
        # 显示图像路径（带编号）
        console.print("\n[bold cyan]═══════════════ 生成的图像 ═══════════════[/bold cyan]\n")
        
        images_dir = Path(session_dir) / "images"
        image_files = sorted(list(images_dir.glob("*_keyframe.png")))
        
        if not image_files:
            console.print("[red]未找到生成的图像！[/red]")
            return False
        
        for idx, img_file in enumerate(image_files, 1):
            scene_id = idx
            console.print(f"  [dim]{idx}[/dim] 📷 {img_file.name}")
        
        console.print(f"\n[cyan]👉 请在文件浏览器中查看: {images_dir}[/cyan]\n")
        console.print("[bold cyan]═══════════════════════════════════════════[/bold cyan]\n")
        
        console.print("[bold]请选择操作：[/bold]")
        console.print("  [green]1[/green] - ✅ 确认全部图像，继续生成视频")
        console.print("  [yellow]2[/yellow] - 🔄 重新生成指定分镜的图像")
        console.print("  [blue]3[/blue] - 🔄 重新生成所有图像")
        console.print("  [red]4[/red] - ❌ 取消并退出")
        
        choice = Prompt.ask("请输入选项", choices=["1", "2", "3", "4"], default="1")
        
        if choice == "1":
            console.print("[green]✅ 图像已确认，开始生成视频...[/green]")
            return True
        
        elif choice == "2":
            # 选择性重新生成
            scene_ids_input = Prompt.ask(
                "[yellow]请输入要重新生成的分镜编号[/yellow]",
                default="1"
            )
            
            # 解析编号（支持逗号分隔，如 "1,3,5"）
            try:
                scene_ids = [int(x.strip()) for x in scene_ids_input.split(",")]
                scene_ids = [x for x in scene_ids if 1 <= x <= len(storyboard.scenes)]
            except:
                console.print("[red]编号格式错误，请输入数字（如：1 或 1,3,5）[/red]")
                continue
            
            if not scene_ids:
                console.print("[red]未选择有效的分镜编号[/red]")
                continue
            
            # 选择参考图来源
            console.print("\n[cyan]请选择参考图来源：[/cyan]")
            console.print("  [dim]1[/dim] - 🖼️  使用原始参考图")
            if len(image_files) > 0:
                console.print("  [dim]2[/dim] - 📷 使用已生成的某个分镜图像")
            
            ref_choice = Prompt.ask("请选择", choices=["1", "2"] if len(image_files) > 0 else ["1"], default="1")
            
            current_ref = reference_image
            if ref_choice == "2":
                ref_scene = Prompt.ask(
                    "[yellow]请输入要作为参考的分镜编号[/yellow]",
                    default="1"
                )
                try:
                    ref_idx = int(ref_scene) - 1
                    if 0 <= ref_idx < len(image_files):
                        current_ref = str(image_files[ref_idx])
                        console.print(f"[green]✅ 使用 {image_files[ref_idx].name} 作为参考图[/green]")
                except:
                    console.print("[yellow]编号无效，使用原始参考图[/yellow]")
            
            # 输入修改建议
            feedback = Prompt.ask("[yellow]请输入修改建议（可回车跳过）[/yellow]", default="")
            
            console.print(f"[bold]正在重新生成分镜 {scene_ids}...[/bold]")
            
            # 更新选中场景的 visual_prompt（只保留最新的修改建议）
            for scene_id in scene_ids:
                scene = storyboard.scenes[scene_id - 1]
                if feedback.strip():
                    # 移除旧的修改要求，只保留原始 prompt 和最新修改
                    base_prompt = scene.visual_prompt.split("\n修改要求：")[0]
                    scene.visual_prompt = f"{base_prompt}\n修改要求：{feedback}"
            
            # 只重新生成选中的场景
            try:
                asyncio.run(orchestrator.run_image_generation(
                    storyboard, current_ref, str(session_dir), 
                    scene_ids=scene_ids
                ))
                console.print("[green]✅ 选中的图像已重新生成！[/green]")
            except Exception as e:
                console.print(f"[red]重新生成失败: {e}[/red]")
        
        elif choice == "3":
            # 重新生成所有图像
            feedback = Prompt.ask("[yellow]请输入修改建议（如：让角色更可爱，改变光线）[/yellow]")
            if feedback.strip():
                console.print("[bold]正在根据您的建议重新生成所有图像...[/bold]")
                
                # 移除旧的修改要求，只保留原始 prompt 和最新修改
                for scene in storyboard.scenes:
                    base_prompt = scene.visual_prompt.split("\n修改要求：")[0]
                    scene.visual_prompt = f"{base_prompt}\n修改要求：{feedback}"
                
                try:
                    asyncio.run(orchestrator.run_image_generation(storyboard, reference_image, str(session_dir)))
                    console.print("[green]✅ 图像已重新生成！请查看新的结果：[/green]")
                except Exception as e:
                    console.print(f"[red]重新生成失败: {e}[/red]")
        
        elif choice == "4":
            return False


async def revise_storyboard(orchestrator, storyboard: Storyboard, feedback: str, reference_image: str | None) -> Storyboard:
    """让 AI 根据用户反馈修改脚本"""
    from src.video_workflow.config import settings
    
    # 构建修改请求
    current_script = storyboard.model_dump_json(indent=2)
    
    revision_prompt = f"""
当前分镜脚本如下：
```json
{current_script}
```

用户反馈：{feedback}

请根据用户的反馈修改脚本，保持相同的 JSON 结构。只修改需要修改的部分，保持其他内容不变。
返回完整的修改后的 JSON，不要包含任何解释文字。
"""
    
    # 调用 LLM 进行修改
    try:
        if settings.LLM_PROVIDER == "deepseek":
            # DeepSeek
            response = await orchestrator.llm.client.chat.completions.create(
                model=settings.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": orchestrator.llm.system_prompt},
                    {"role": "user", "content": revision_prompt}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
        else:
            # GLM - 使用同步调用
            import asyncio
            loop = asyncio.get_running_loop()
            
            def _call_glm():
                return orchestrator.llm.client.chat.completions.create(
                    model=settings.GLM_MODEL,
                    messages=[
                        {"role": "system", "content": orchestrator.llm.system_prompt},
                        {"role": "user", "content": [{"type": "text", "text": revision_prompt}]}
                    ]
                )
            
            response = await loop.run_in_executor(None, _call_glm)
            content = response.choices[0].message.content
        
        if not content:
            raise ValueError("LLM 返回空内容")
        
        # 解析响应
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        data = json.loads(content.strip())
        return Storyboard(**data)
    except Exception as e:
        console.print(f"[red]AI 修改失败: {e}[/red]")
        raise


if __name__ == "__main__":
    typer.run(main)
