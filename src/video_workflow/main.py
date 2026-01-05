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
    topic: str = typer.Argument(..., help="视频主题"),
    count: int = typer.Option(5, "--count", "-c", help="生成的场景数量"),
    reference_image: str = typer.Option(None, "--ref", "-r", help="参考图路径（用于保持角色一致性）"),
    skip_review: bool = typer.Option(False, "--skip-review", help="跳过所有审阅步骤，直接生成")
):
    """
    为指定主题运行视频生成工作流。
    """
    console.print(f"[bold green]正在启动视频生成工作流，主题：[/bold green] {topic}")
    if reference_image:
        console.print(f"[bold cyan]使用参考图：[/bold cyan] {reference_image}")
    
    orchestrator = WorkflowOrchestrator()
    
    try:
        # 初始化
        asyncio.run(orchestrator.initialize())
        
        # 0. 询问角色外貌描述（如果没有跳过审阅）
        character_desc = None
        if not skip_review:
            from src.video_workflow.config import settings
            if settings.CHARACTER_DESCRIPTION:
                console.print(f"\n[cyan]📝 当前角色描述：[/cyan]{settings.CHARACTER_DESCRIPTION}")
                use_default = Confirm.ask("是否使用此描述？", default=True)
                if use_default:
                    character_desc = settings.CHARACTER_DESCRIPTION
                else:
                    character_desc = Prompt.ask("[yellow]请输入自定义角色外貌描述[/yellow]", default="")
            else:
                if Confirm.ask("\n是否需要自定义角色外貌描述？", default=False):
                    character_desc = Prompt.ask("[yellow]请输入角色外貌描述[/yellow]")
            
            if character_desc:
                # 临时覆盖配置
                import src.video_workflow.config as config_module
                config_module.settings.CHARACTER_DESCRIPTION = character_desc
                console.print(f"[green]✅ 角色描述已设置：{character_desc}[/green]")
        
        # 1. 生成分镜脚本
        console.print("\n[bold yellow]步骤 1/4：生成分镜脚本...[/bold yellow]")
        storyboard = asyncio.run(
            orchestrator.llm.generate_storyboard(topic, count, reference_image)
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
