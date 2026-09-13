from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.video_workflow.config import settings
from src.video_workflow.domain import new_id
from src.video_workflow.generators.llm import OpenLuxGenerator
from src.video_workflow.prompt_profiles import (
    PROMPT_TEMPLATE_VARIABLES,
    PROMPT_TEMPLATE_VARIABLE_DESCRIPTIONS,
    PROMPT_TEMPLATE_EXAMPLES,
    PromptProfileContent,
    PromptTemplateVersion,
    profile_hash,
    render_prompt_template,
    system_default_profile,
    validate_template_variables,
)
from src.video_workflow.services.projects import ProjectService
from src.video_workflow.storage import ProjectStore


router = APIRouter(prefix="/prompt-templates", tags=["prompt-templates"])
store = ProjectStore(settings.DATABASE_PATH)
project_service = ProjectService(store)


class PromptTemplateSaveRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=1000)
    content: PromptProfileContent
    change_summary: list[str] = Field(default_factory=list, max_length=50)


class PromptTemplateApplyRequest(BaseModel):
    template_id: str = Field(min_length=1, max_length=200)
    version: int | None = Field(default=None, ge=1)


class PromptTemplatePreviewRequest(BaseModel):
    content: PromptProfileContent
    context_values: dict[str, Any] = Field(default_factory=dict)
    user_notes: str = Field(default="", max_length=4000)


class PromptTemplateOptimizeRequest(BaseModel):
    project_id: str | None = None
    template_id: str | None = None
    version: int | None = Field(default=None, ge=1)
    instruction: str = Field(min_length=1, max_length=4000)
    scopes: list[Literal["storyboard", "visual", "keyframe", "seedance", "h3"]] = Field(
        default_factory=lambda: ["storyboard", "visual", "keyframe", "seedance", "h3"],
        max_length=5,
    )


def _with_system_default(templates: list[PromptTemplateVersion]) -> list[PromptTemplateVersion]:
    return [system_default_profile(), *[item for item in templates if item.id != "system-default"]]


def _load_template(template_id: str, version: int | None = None) -> PromptTemplateVersion:
    if template_id == "system-default":
        return system_default_profile()
    template = store.get_prompt_template(template_id, version)
    if template is None:
        raise HTTPException(status_code=404, detail="Prompt 模板不存在")
    return template


@router.get("")
async def list_prompt_templates():
    return _with_system_default(store.list_prompt_templates())


@router.get("/metadata")
async def get_prompt_template_metadata():
    """Return authoring help for the editable prompt template fields."""
    return {
        "schema_version": 1,
        "variables": [
            {"name": name, **PROMPT_TEMPLATE_VARIABLE_DESCRIPTIONS[name]}
            for name in PROMPT_TEMPLATE_VARIABLES
        ],
        "examples": PROMPT_TEMPLATE_EXAMPLES,
        "machine_contract_note": "机器协议由系统固定维护，用户可查看但不直接编辑；其余创作规则和下游规则可自定义。",
    }


@router.get("/{template_id}")
async def get_prompt_template(template_id: str, version: int | None = None):
    return _load_template(template_id, version)


@router.get("/{template_id}/versions")
async def list_prompt_template_versions(template_id: str):
    if template_id == "system-default":
        return [system_default_profile()]
    if store.get_prompt_template(template_id) is None:
        raise HTTPException(status_code=404, detail="Prompt 模板不存在")
    return store.list_prompt_template_versions(template_id)


@router.post("")
async def create_prompt_template(request: PromptTemplateSaveRequest):
    unknown = validate_template_variables(request.content)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Prompt 模板包含未知变量: {', '.join(unknown)}")
    content = request.content.model_copy(deep=True)
    content.machine_contract = system_default_profile().content.machine_contract
    template = PromptTemplateVersion(
        id=new_id("prompt_template"),
        name=request.name.strip(),
        description=request.description.strip(),
        source="user",
        version=1,
        content=content,
        change_summary=request.change_summary,
        created_by="user",
    )
    return store.save_prompt_template(template)


@router.post("/{template_id}/versions")
async def create_prompt_template_version(template_id: str, request: PromptTemplateSaveRequest):
    current = _load_template(template_id)
    if template_id == "system-default":
        raise HTTPException(status_code=400, detail="系统默认模板不可覆盖，请另存为自定义模板")
    unknown = validate_template_variables(request.content)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Prompt 模板包含未知变量: {', '.join(unknown)}")
    content = request.content.model_copy(deep=True)
    content.machine_contract = system_default_profile().content.machine_contract
    template = PromptTemplateVersion(
        id=template_id,
        name=request.name.strip(),
        description=request.description.strip(),
        source="user",
        version=current.version + 1,
        based_on_default_version=current.based_on_default_version,
        content=content,
        change_summary=request.change_summary,
        created_by="user",
        created_at=current.created_at,
    )
    return store.save_prompt_template(template)


@router.post("/validate")
async def validate_prompt_template(request: PromptTemplatePreviewRequest):
    unknown = validate_template_variables(request.content)
    return {
        "valid": not unknown,
        "unknown_variables": unknown,
        "available_variables": list(PROMPT_TEMPLATE_VARIABLES),
        "variable_descriptions": [
            {"name": name, **PROMPT_TEMPLATE_VARIABLE_DESCRIPTIONS[name]}
            for name in PROMPT_TEMPLATE_VARIABLES
        ],
        "profile_hash": profile_hash({"content": request.content.model_dump(mode="json")}),
    }


@router.post("/render-preview")
async def render_prompt_template_preview(request: PromptTemplatePreviewRequest):
    unknown = validate_template_variables(request.content)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Prompt 模板包含未知变量: {', '.join(unknown)}")
    values = dict(request.context_values)
    values["run_notes"] = request.user_notes or values.get("run_notes", "无")
    return {
        "system_prompt": f"{request.content.machine_contract}\n\n{render_prompt_template(request.content.director_template, values)}".strip(),
        "context_prompt": render_prompt_template(request.content.context_template, values),
        "downstream_rules": request.content.downstream_rules,
        "profile_hash": profile_hash({"content": request.content.model_dump(mode="json")}),
    }


@router.post("/ai-draft")
async def create_ai_prompt_template_draft(request: PromptTemplateOptimizeRequest):
    project = store.get_project(request.project_id) if request.project_id else None
    if request.project_id and project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if request.template_id:
        base = _load_template(request.template_id, request.version)
    elif project is not None:
        base = project_service.prompt_profile_for_project(project)
    else:
        base = system_default_profile()
    project_context = {
        "title": project.brief.title if project else "未提供",
        "story": project.brief.story if project else "未提供",
        "characters": project_service.character_bible(project) if project else "未提供",
        "style": project_service.project_style_text(project) if project else "未提供",
        "target_duration": project.brief.target_duration_seconds if project else "未提供",
        "aspect_ratio": project.brief.aspect_ratio if project else "未提供",
    }
    optimizer_system = """你是资深短视频总导演、分镜系统设计师和 Prompt 工程师。你的任务是根据现有模板、项目故事和用户要求，产出一份可直接用于生产的 Prompt 模板草稿。
只返回 JSON，不要 Markdown 或解释。必须保留机器协议中的字段和结构，不得修改 machine_contract；只优化 director_template、context_template 和选定的 downstream_rules。模板必须使用白名单变量，不能虚构项目事实。返回结构：{"name":"","description":"","director_template":"","context_template":"","downstream_rules":{"visual":"","keyframe":"","seedance":"","h3":""},"change_summary":[],"expected_effects":[],"warnings":[],"variables_used":[]}"""
    optimizer_user = json.dumps(
        {
            "现有模板": base.content.model_dump(mode="json"),
            "项目上下文": project_context,
            "优化范围": request.scopes,
            "用户要求": request.instruction,
        },
        ensure_ascii=False,
        indent=2,
    )
    try:
        llm = OpenLuxGenerator(
            model=settings.PROMPT_OPTIMIZER_MODEL,
            vision_model=settings.PROMPT_OPTIMIZER_MODEL,
            provider_name="OpenLux Prompt 优化器",
        )
        payload = await llm.generate_json(optimizer_system, optimizer_user)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenLux Prompt 优化失败: {exc}") from exc

    raw_content = payload.get("content") if isinstance(payload.get("content"), dict) else payload
    try:
        content = PromptProfileContent.model_validate(raw_content)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Opus 5 返回的模板草稿结构无效: {exc}") from exc
    # The machine contract remains owned by the application even when the AI
    # returns a malformed or overly creative replacement.
    content.machine_contract = base.content.machine_contract
    if "storyboard" not in request.scopes:
        content.director_template = base.content.director_template
        content.context_template = base.content.context_template
    for target in ("visual", "keyframe", "seedance", "h3"):
        if target not in request.scopes:
            content.downstream_rules[target] = base.content.downstream_rules[target]
    unknown = validate_template_variables(content)
    if unknown:
        raise HTTPException(status_code=502, detail=f"Opus 5 草稿包含未知变量: {', '.join(unknown)}")
    draft = PromptTemplateVersion(
        id=new_id("prompt_ai"),
        name=str(payload.get("name") or f"{base.name} · AI 优化").strip()[:200],
        description=str(payload.get("description") or "由 OpenLux Claude Opus 5 根据本剧要求生成的模板草稿").strip()[:1000],
        source="ai",
        version=1,
        based_on_default_version=base.based_on_default_version,
        content=content,
        change_summary=[str(item) for item in payload.get("change_summary", [])][:50],
        created_by="ai",
    )
    return {
        "draft": draft,
        "expected_effects": [str(item) for item in payload.get("expected_effects", [])][:50],
        "warnings": [str(item) for item in payload.get("warnings", [])][:50],
        "variables_used": [str(item) for item in payload.get("variables_used", [])][:50],
    }


@router.put("/projects/{project_id}/prompt-profile")
async def apply_project_prompt_template(project_id: str, request: PromptTemplateApplyRequest):
    project = store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    profile = _load_template(request.template_id, request.version)
    return project_service.apply_prompt_profile(project_id, profile)


@router.post("/projects/{project_id}/prompt-profile/reset")
async def reset_project_prompt_template(project_id: str):
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project_service.reset_prompt_profile(project_id)
