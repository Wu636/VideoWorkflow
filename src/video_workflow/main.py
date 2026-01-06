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
    
    analysis_prompt = """请仔细分析这张图片中的主体角色（人物/动物/卡通形象），生成一段详细的外貌描述。

要求：
1. 描述要具体、准确，可用于后续 AI 图像生成
2. 包含：物种/角色类型、体型、毛色/肤色、五官特征、服装配饰、表情气质
3. 描述长度约50-100字
4. 只输出描述文本，不要其他解释

示例输出格式：
一只圆润可爱的橘色猫咪，毛发蓬松柔软，戴着白色厨师帽，穿着蓝色围裙，大眼睛水汪汪的，表情憨态可掬，尾巴毛茸茸的"""
    
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
            return result.strip() if result else None
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
            return result.strip() if result else None
        except Exception as e:
            console.print(f"[yellow]⚠️ 豆包分析失败: {e}[/yellow]")
    
    console.print("[yellow]⚠️ 未配置多模态 API (GLM 或 ARK)，无法自动分析参考图[/yellow]")
    return None

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
            
            from pathlib import Path
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
                console.print("\n[bold cyan]🔍 正在分析参考图，自动生成角色描述...[/bold cyan]")
                try:
                    auto_desc = asyncio.run(analyze_reference_image(reference_image))
                    if auto_desc:
                        console.print(f"\n[green]✅ AI 自动生成的角色描述：[/green]")
                        console.print(f"[bold white]{auto_desc}[/bold white]")
                        
                        # 先显示菜单，再获取用户选择
                        console.print("\n[cyan]请选择操作：[/cyan]")
                        console.print("  [dim]1[/dim] - ✅ 使用此描述")
                        console.print("  [dim]2[/dim] - ✏️  在此基础上修改")
                        console.print("  [dim]3[/dim] - 📝 手动输入新描述")
                        
                        choice = Prompt.ask(
                            "请选择",
                            choices=["1", "2", "3"],
                            default="1"
                        )
                        
                        if choice == "1":
                            character_desc = auto_desc
                        elif choice == "2":
                            modified_desc = Prompt.ask("[yellow]请修改描述[/yellow]", default=auto_desc)
                            character_desc = modified_desc
                        else:
                            character_desc = Prompt.ask("[yellow]请输入新的角色描述[/yellow]")
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
                        character_desc = Prompt.ask("[yellow]请输入自定义角色外貌描述[/yellow]", default="")
                else:
                    if Confirm.ask("\n是否需要自定义角色外貌描述？", default=False):
                        character_desc = Prompt.ask("[yellow]请输入角色外貌描述[/yellow]")
            
            if character_desc:
                config_module.settings.CHARACTER_DESCRIPTION = character_desc
                console.print(f"[green]✅ 角色描述已设置[/green]")
        
        # 1. 选择爆款模板
        selected_template = template
        if not skip_review and not selected_template:
            from src.video_workflow.templates import VIRAL_TEMPLATES, get_template_description
            console.print("\n[bold cyan]📋 选择爆款脚本模板：[/bold cyan]")
            console.print("  [dim]0[/dim] - 不使用模板（自由创作）")
            for idx, (name, tmpl) in enumerate(VIRAL_TEMPLATES.items(), 1):
                console.print(f"  [dim]{idx}[/dim] - [bold]{name}[/bold] - {tmpl.description}")
            
            template_choice = Prompt.ask(
                "请选择模板编号", 
                choices=["0", "1", "2", "3", "4", "5", "6", "7"],
                default="0"
            )
            
            if template_choice != "0":
                template_names = list(VIRAL_TEMPLATES.keys())
                selected_template = template_names[int(template_choice) - 1]
                console.print(f"[green]✅ 已选择模板：{selected_template}[/green]")
        
        if selected_template:
            console.print(f"[cyan]📋 使用爆款模板：{selected_template}[/cyan]")
        
        # 2. 生成分镜脚本
        console.print("\n[bold yellow]步骤 1/4：生成分镜脚本...[/bold yellow]")
        storyboard = asyncio.run(
            orchestrator.llm.generate_storyboard(topic, count, reference_image, selected_template)
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
        
        # 4. 生成视频
        console.print("\n[bold yellow]步骤 3/4：生成视频片段...[/bold yellow]")
        asyncio.run(orchestrator.run_video_generation(storyboard, session_dir))
        
        console.print(f"\n[bold blue]✅ 成功！[/bold blue] 输出已保存至：{session_dir}")
            
    except KeyboardInterrupt:
        console.print("\n[bold yellow]用户中断了工作流。[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]错误：[/bold red] {e}")
        import traceback
        traceback.print_exc()


def review_script_loop(orchestrator, storyboard: Storyboard, topic: str, reference_image: str | None) -> Storyboard | None:
    """交互式脚本审阅循环"""
    
    while True:
        # 显示当前脚本
        display_script(storyboard)
        
        console.print("[bold]请选择操作：[/bold]")
        console.print("  [green]1[/green] - ✅ 确认脚本，继续生成图像和视频")
        console.print("  [yellow]2[/yellow] - 🤖 输入修改建议，让 AI 修改脚本")
        console.print("  [cyan]3[/cyan] - ✏️  手动编辑脚本文件")
        console.print("  [red]4[/red] - ❌ 取消并退出")
        
        choice = Prompt.ask("请输入选项", choices=["1", "2", "3", "4"], default="1")
        
        if choice == "1":
            console.print("[green]✅ 脚本已确认，开始生成...[/green]")
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
            # 手动编辑模式
            temp_file = Path("temp_script.json")
            temp_file.write_text(storyboard.model_dump_json(indent=2), encoding="utf-8")
            console.print(f"\n[cyan]脚本已保存到 [bold]{temp_file}[/bold][/cyan]")
            console.print("[cyan]请使用编辑器修改后保存，然后按回车继续...[/cyan]")
            input()
            
            try:
                with open(temp_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                storyboard = Storyboard(**data)
                console.print("[green]✅ 脚本已从文件加载！请查看修改后的内容：[/green]")
                temp_file.unlink()  # 删除临时文件
            except Exception as e:
                console.print(f"[red]加载失败: {e}[/red]")
        
        elif choice == "4":
            return None


def review_images_loop(session_dir, storyboard: Storyboard, orchestrator, reference_image: str | None) -> bool:
    """交互式图像审阅循环"""
    from pathlib import Path
    
    while True:
        # 显示图像路径
        console.print("\n[bold cyan]═══════════════ 生成的图像 ═══════════════[/bold cyan]\n")
        
        images_dir = Path(session_dir) / "images"
        image_files = list(images_dir.glob("*.png"))
        
        if not image_files:
            console.print("[red]未找到生成的图像！[/red]")
            return False
        
        for img_file in sorted(image_files):
            console.print(f"  📷 {img_file}")
        
        console.print(f"\n[cyan]👉 请在文件浏览器中查看: {images_dir}[/cyan]\n")
        console.print("[bold cyan]═══════════════════════════════════════════[/bold cyan]\n")
        
        console.print("[bold]请选择操作：[/bold]")
        console.print("  [green]1[/green] - ✅ 确认图像，继续生成视频")
        console.print("  [yellow]2[/yellow] - 🔄 输入修改建议，重新生成所有图像")
        console.print("  [red]3[/red] - ❌ 取消并退出")
        
        choice = Prompt.ask("请输入选项", choices=["1", "2", "3"], default="1")
        
        if choice == "1":
            console.print("[green]✅ 图像已确认，开始生成视频...[/green]")
            return True
        
        elif choice == "2":
            feedback = Prompt.ask("[yellow]请输入修改建议（如：让角色更可爱，改变光线）[/yellow]")
            if feedback.strip():
                console.print("[bold]正在根据您的建议重新生成图像...[/bold]")
                
                # 更新每个场景的 visual_prompt
                for scene in storyboard.scenes:
                    scene.visual_prompt += f"\n修改要求：{feedback}"
                
                # 重新生成图像
                try:
                    asyncio.run(orchestrator.run_image_generation(storyboard, reference_image, str(session_dir)))
                    console.print("[green]✅ 图像已重新生成！请查看新的结果：[/green]")
                except Exception as e:
                    console.print(f"[red]重新生成失败: {e}[/red]")
        
        elif choice == "3":
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
