"use client";

import { useEffect, useMemo, useState } from "react";
import { BookOpen, Check, ChevronDown, ChevronUp, Info, RefreshCw, RotateCcw, Save, Sparkles, WandSparkles, X } from "lucide-react";

import {
    applyProjectPromptTemplate,
    createPromptTemplate,
    createPromptTemplateVersion,
    getPromptTemplateMetadata,
    listPromptTemplates,
    optimizePromptTemplate,
    previewPromptTemplate,
    resetProjectPromptTemplate,
} from "@/lib/api";
import type { Project, PromptProfileContent, PromptTemplateAiDraft, PromptTemplateExampleInfo, PromptTemplatePreview, PromptTemplateVariableInfo, PromptTemplateVersion } from "@/types";

type Props = {
    project: Project;
    setProject: (project: Project) => void;
};

const DOWNSTREAM_LABELS: Array<[keyof PromptProfileContent["downstream_rules"], string]> = [
    ["visual", "通用视觉 Prompt"],
    ["keyframe", "首帧图 Prompt"],
    ["seedance", "Seedance Prompt"],
    ["h3", "H3 Prompt"],
];

function profileContent(template: PromptTemplateVersion | null, projectSnapshot: Record<string, unknown> | undefined): PromptProfileContent | null {
    const snapshot = projectSnapshot as { content?: PromptProfileContent } | undefined;
    return snapshot?.content || template?.content || null;
}

function copyContent(content: PromptProfileContent): PromptProfileContent {
    return {
        ...content,
        downstream_rules: { ...content.downstream_rules },
    };
}

export default function PromptTemplateManager({ project, setProject }: Props) {
    const projectId = project.id;
    const projectTemplateId = project.prompt_template_id || "system-default";
    const projectTemplateVersion = project.prompt_template_version || 1;
    const projectTemplateSnapshot = project.prompt_template_snapshot;
    const [expanded, setExpanded] = useState(false);
    const [templates, setTemplates] = useState<PromptTemplateVersion[]>([]);
    const [variableGuide, setVariableGuide] = useState<PromptTemplateVariableInfo[]>([]);
    const [promptExamples, setPromptExamples] = useState<PromptTemplateExampleInfo[]>([]);
    const [selectedId, setSelectedId] = useState(project.prompt_template_id || "system-default");
    const [selectedVersion, setSelectedVersion] = useState(project.prompt_template_version || 1);
    const [draft, setDraft] = useState<PromptTemplateVersion | null>(null);
    const [tab, setTab] = useState<"director" | "context" | "downstream" | "preview">("director");
    const [optimizerOpen, setOptimizerOpen] = useState(false);
    const [optimizerInstruction, setOptimizerInstruction] = useState("");
    const [optimizerScopes, setOptimizerScopes] = useState<string[]>(["storyboard", "visual", "keyframe", "seedance", "h3"]);
    const [aiResult, setAiResult] = useState<PromptTemplateAiDraft | null>(null);
    const [preview, setPreview] = useState<PromptTemplatePreview | null>(null);
    const [showVariableGuide, setShowVariableGuide] = useState(true);
    const [showDefaultDetails, setShowDefaultDetails] = useState(false);
    const [selectedExampleId, setSelectedExampleId] = useState("");
    const [busy, setBusy] = useState("");
    const [message, setMessage] = useState("");
    const [error, setError] = useState("");

    const currentTemplate = useMemo(
        () => templates.find((template) => template.id === selectedId && template.version === selectedVersion)
            || templates.find((template) => template.id === selectedId)
            || null,
        [selectedId, selectedVersion, templates],
    );
    const defaultTemplate = useMemo(
        () => templates.find((template) => template.id === "system-default") || null,
        [templates],
    );

    useEffect(() => {
        if (!expanded) return;
        let active = true;
        void listPromptTemplates().then((items) => {
            if (!active) return;
            setTemplates(items);
            const selected = items.find((item) => item.id === projectTemplateId) || items[0];
            if (selected) {
                setSelectedId(selected.id);
                setSelectedVersion(projectTemplateVersion || selected.version);
                setDraft({ ...selected, content: copyContent(profileContent(selected, projectTemplateSnapshot) || selected.content) });
            }
        }).catch((caught) => active && setError(caught instanceof Error ? caught.message : String(caught)));
        void getPromptTemplateMetadata().then((metadata) => {
            if (!active) return;
            setVariableGuide(metadata.variables);
            setPromptExamples(metadata.examples);
            setSelectedExampleId((current) => current || metadata.examples[0]?.id || "");
        });
        return () => { active = false; };
    }, [expanded, projectId, projectTemplateId, projectTemplateVersion, projectTemplateSnapshot]);

    const selectTemplate = (id: string) => {
        const next = templates.find((template) => template.id === id) || null;
        if (!next) return;
        setSelectedId(next.id);
        setSelectedVersion(next.version);
        setAiResult(null);
        setPreview(null);
        setDraft({ ...next, content: copyContent(next.content) });
        setMessage("");
        setError("");
    };

    const updateContent = (patch: Partial<PromptProfileContent>) => {
        setDraft((current) => current ? { ...current, content: { ...current.content, ...patch } } : current);
    };

    const saveAsCustom = async () => {
        if (!draft) return;
        setBusy("save"); setError(""); setMessage("");
        try {
            const saved = await createPromptTemplate(draft.name, draft.description, draft.content, ["用户保存模板"]);
            const projectNext = await applyProjectPromptTemplate(project.id, saved.id, saved.version);
            setProject(projectNext);
            setTemplates((current) => [saved, ...current.filter((item) => item.id !== saved.id)]);
            setSelectedId(saved.id); setSelectedVersion(saved.version); setDraft(saved);
            setMessage("已保存为自定义模板，并应用到当前项目");
        } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); }
        finally { setBusy(""); }
    };

    const saveVersion = async () => {
        if (!draft || selectedId === "system-default") return saveAsCustom();
        setBusy("save"); setError(""); setMessage("");
        try {
            const saved = await createPromptTemplateVersion(selectedId, draft.name, draft.description, draft.content, ["用户更新模板"]);
            const projectNext = await applyProjectPromptTemplate(project.id, saved.id, saved.version);
            setProject(projectNext);
            setTemplates((current) => [saved, ...current.filter((item) => item.id !== saved.id)]);
            setSelectedVersion(saved.version); setDraft(saved);
            setMessage(`已保存模板 v${saved.version}，并应用到当前项目`);
        } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); }
        finally { setBusy(""); }
    };

    const applySelected = async () => {
        if (!currentTemplate) return;
        setBusy("apply"); setError(""); setMessage("");
        try {
            setProject(await applyProjectPromptTemplate(project.id, currentTemplate.id, currentTemplate.version));
            setMessage(`已应用“${currentTemplate.name}” v${currentTemplate.version}`);
        } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); }
        finally { setBusy(""); }
    };

    const resetDefault = async () => {
        setBusy("reset"); setError(""); setMessage("");
        try {
            setProject(await resetProjectPromptTemplate(project.id));
            selectTemplate("system-default");
            setMessage("已恢复系统默认模板；现有镜头不会自动覆盖");
        } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); }
        finally { setBusy(""); }
    };

    const content = draft?.content || profileContent(currentTemplate, projectTemplateSnapshot);
    const selectedExample = promptExamples.find((item) => item.id === selectedExampleId) || null;

    const appendExample = () => {
        if (!selectedExample || !content) return;
        const current = content.context_template.trimEnd();
        updateContent({ context_template: `${current}\n\n${selectedExample.content}` });
        setTab("context");
        setMessage(`已将“${selectedExample.label}”追加到项目约束模板末尾，请结合本剧修改`);
        setError("");
    };

    const copyExample = async () => {
        if (!selectedExample) return;
        try {
            await navigator.clipboard.writeText(selectedExample.content);
            setMessage("示例已复制，可以粘贴到任意自定义模板中");
            setError("");
        } catch (caught) {
            setError(caught instanceof Error ? caught.message : "复制示例失败");
        }
    };

    const optimize = async () => {
        if (!optimizerInstruction.trim()) return;
        setBusy("optimize"); setError(""); setMessage("");
        try {
            const result = await optimizePromptTemplate({
                projectId: project.id,
                templateId: selectedId,
                version: selectedVersion,
                instruction: optimizerInstruction.trim(),
                scopes: optimizerScopes as ("storyboard" | "visual" | "keyframe" | "seedance" | "h3")[],
            });
            setAiResult(result);
            setDraft(result.draft);
            setMessage("Opus 5 已生成模板草稿，请检查后保存或应用");
        } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); }
        finally { setBusy(""); }
    };

    const loadPreview = async () => {
        if (!content) return;
        setBusy("preview"); setError("");
        try {
            setPreview(await previewPromptTemplate(content, {
                title: project.brief.title,
                story: project.brief.story,
                shot_count: project.manual_shot_count || project.ai_recommended_shot_count || "AI 判断",
                target_duration: project.brief.target_duration_seconds,
                average_shot_duration: "按本次镜头数计算",
                characters: project.characters.map((item) => `${item.name}：${item.description}；${item.wardrobe}`).join("；") || "未提供",
                style: project.brief.visual_style || project.style_bible || "未提供",
                aspect_ratio: project.brief.aspect_ratio,
                pacing: project.brief.pacing || "未填写",
                speech_pacing: project.brief.speech_pacing,
                spoken_text_policy: project.brief.spoken_text_policy || "adaptive",
                delivery_notes: project.brief.delivery_notes || "未填写",
                scene_profiles: project.scene_profiles.map((item) => item.name).join("；") || "未提供",
                series_constraints: project.seedance_global_constraints || "未提供",
                prompt_targets: project.preferred_prompt_targets.join(", ") || "storyboard",
                count_contract: "按本次实际选择的镜头数输出。",
                speech_contract: "声音事件按镜头时长安排，事件不得重叠。",
            }));
        } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); }
        finally { setBusy(""); }
    };

    return <section className="studio-panel mb-5 border-cyan-300/10 bg-cyan-300/[.025]">
        <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
                <p className="studio-kicker">PROMPT PROFILE</p>
                <h3 className="font-semibold">分镜 Prompt 模板</h3>
                <p className="mt-1 text-xs leading-5 text-white/40">
                    当前：{project.prompt_template_id === "system-default" ? "系统默认" : project.prompt_template_id} v{project.prompt_template_version} · {project.prompt_template_source === "system" ? "系统" : project.prompt_template_source === "series" ? "系列继承" : "项目自定义"}
                </p>
            </div>
            <button type="button" className="studio-secondary" onClick={() => setExpanded((current) => !current)}>{expanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}{expanded ? "收起模板" : "编辑模板"}</button>
        </div>
        {expanded && <div className="mt-5 border-t border-white/8 pt-5">
            <div className="grid gap-4 lg:grid-cols-[260px_1fr]">
                <div className="space-y-3">
                    <label><span className="studio-label">选择模板</span><select className="studio-input mt-1 w-full" value={selectedId} onChange={(event) => selectTemplate(event.target.value)}>{templates.map((template) => <option key={`${template.id}-${template.version}`} value={template.id}>{template.name} · v{template.version}{template.source === "system" ? " · 系统" : ""}</option>)}</select></label>
                    <label><span className="studio-label">模板名称</span><input className="studio-input mt-1 w-full" value={draft?.name || ""} onChange={(event) => setDraft((current) => current ? { ...current, name: event.target.value } : current)} /></label>
                    {draft?.description && <p className="text-[11px] leading-5 text-white/40">{draft.description}</p>}
                    <div className="rounded-lg border border-cyan-300/10 bg-black/15 p-3 text-xs leading-5 text-white/55">
                        <div className="flex items-center justify-between gap-2"><span className="font-medium text-white/70">怎么修改</span><button type="button" className="inline-flex items-center gap-1 text-cyan-200/80 hover:text-cyan-100" onClick={() => setShowVariableGuide((current) => !current)}><Info size={13} />{showVariableGuide ? "收起参数说明" : "查看参数说明"}</button></div>
                        <p className="mt-1">选中系统默认后直接编辑，点击“另存为自定义模板”；自定义模板修改后点击“保存模板版本”。</p>
                        <p className="mt-1 text-white/40">双大括号变量会在生成时自动替换成当前项目资料，系统机器协议保持只读。</p>
                    </div>
                    <button type="button" className="studio-secondary w-full" onClick={() => setShowDefaultDetails((current) => !current)}><BookOpen size={15} />{showDefaultDetails ? "收起系统默认全文" : "查看系统默认 v1 全文"}</button>
                    <button type="button" className="studio-secondary w-full" disabled={!!busy} onClick={() => setOptimizerOpen((current) => !current)}><Sparkles size={15} />让 Opus 5 优化本剧模板</button>
                    <button type="button" className="studio-secondary w-full" disabled={!!busy} onClick={() => void resetDefault()}><RotateCcw size={15} />恢复系统默认</button>
                </div>
                <div>
                    <div className="mb-3 flex flex-wrap gap-2">{([["director", "导演模板"], ["context", "项目约束"], ["downstream", "下游 Prompt"], ["preview", "最终预览"]] as const).map(([key, label]) => <button key={key} type="button" className={tab === key ? "studio-primary px-3 py-2 text-xs" : "studio-secondary px-3 py-2 text-xs"} onClick={() => setTab(key)}>{label}</button>)}</div>
                    {showVariableGuide && <div className="mb-4 rounded-xl border border-cyan-300/10 bg-cyan-300/[.035] p-4"><div className="flex flex-wrap items-start justify-between gap-3"><div><p className="studio-kicker">TEMPLATE VARIABLES</p><strong>参数说明：生成时会自动替换</strong><p className="mt-1 text-[11px] leading-5 text-white/40">可以在“项目约束”模板中重复使用、调整顺序或删除变量；保存时系统会检查变量名称是否正确。</p></div><button type="button" className="studio-secondary px-2 py-1 text-xs" onClick={() => setShowVariableGuide(false)}>收起</button></div>{variableGuide.length > 0 ? <div className="mt-3 grid gap-2 sm:grid-cols-2">{variableGuide.map((item) => <div key={item.name} className="rounded-lg border border-white/8 bg-black/15 p-3"><div className="flex flex-wrap items-center gap-2"><code className="text-cyan-200">{`{{${item.name}}}`}</code><span className="text-xs text-white/75">{item.label}</span><span className="text-[10px] text-white/30">{item.source}</span></div><p className="mt-1 text-[11px] leading-5 text-white/50">{item.description}</p><p className="mt-1 truncate text-[10px] text-white/30">示例：{item.example}</p></div>)}</div> : <p className="mt-3 text-xs text-white/35">正在加载参数说明…</p>}</div>}
                    {content && tab === "director" && <div><label><span className="studio-label">核心分镜导演模板</span><textarea className="studio-input mt-1 min-h-80 w-full resize-y font-mono text-xs leading-5" value={content.director_template} onChange={(event) => updateContent({ director_template: event.target.value })} /></label><p className="mt-2 text-[11px] text-white/35">机器 JSON 协议单独锁定；这里只调整叙事、动作、镜头和风格方法。</p></div>}
                    {content && tab === "context" && <div><label><span className="studio-label">用户建议与项目约束拼接模板</span><textarea className="studio-input mt-1 min-h-80 w-full resize-y font-mono text-xs leading-5" value={content.context_template} onChange={(event) => updateContent({ context_template: event.target.value })} /></label><p className="mt-2 text-[11px] leading-5 text-white/35">这里写“这类项目长期都要遵守的创作规则”；项目剧情、角色资料和本次临时要求继续通过变量注入。使用双大括号变量；未知变量保存时会被校验。</p>{selectedExample && <div className="mt-4 rounded-xl border border-amber-300/15 bg-amber-300/[.025] p-4"><div className="flex flex-wrap items-start justify-between gap-3"><div><p className="studio-kicker">PROMPT EXAMPLES</p><strong>适合写进项目约束的示例</strong><p className="mt-1 text-[11px] leading-5 text-white/40">示例是固定规则片段，不是本剧剧情。可以先追加，再把不适合本剧的句子删掉或改掉。</p></div><select className="studio-input max-w-full text-xs" value={selectedExampleId} onChange={(event) => setSelectedExampleId(event.target.value)}>{promptExamples.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></div><p className="mt-3 text-xs text-white/50">{selectedExample.description}</p><pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap rounded-lg border border-white/8 bg-black/20 p-3 text-[11px] leading-5 text-white/60">{selectedExample.content}</pre><div className="mt-3 flex flex-wrap justify-end gap-2"><button type="button" className="studio-secondary px-3 py-2 text-xs" onClick={() => void copyExample()}>复制示例</button><button type="button" className="studio-primary px-3 py-2 text-xs" onClick={appendExample}>追加到当前模板</button></div></div>}</div>}
                    {content && tab === "downstream" && <div className="space-y-3">{DOWNSTREAM_LABELS.map(([key, label]) => <label key={key} className="block"><span className="studio-label">{label}</span><textarea className="studio-input mt-1 min-h-24 w-full resize-y text-xs leading-5" value={content.downstream_rules[key]} onChange={(event) => updateContent({ downstream_rules: { ...content.downstream_rules, [key]: event.target.value } })} /></label>)}</div>}
                    {content && tab === "preview" && <div className="space-y-3"><div className="flex flex-wrap items-center justify-between gap-3"><div><span className="studio-label">按当前项目渲染的实际 Prompt</span><p className="mt-1 text-[11px] text-white/35">预览只读取当前项目资料，不会保存模板或触发模型调用。</p></div><button type="button" className="studio-secondary px-3 py-2 text-xs" disabled={busy === "preview"} onClick={() => void loadPreview()}>{busy === "preview" ? <WandSparkles className="animate-pulse" size={13} /> : <RefreshCw size={13} />}刷新预览</button></div>{preview ? <><div><span className="studio-label">系统 Prompt（机器协议 + 导演模板）</span><pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap rounded-lg border border-white/8 bg-black/25 p-3 text-[11px] leading-5 text-white/60">{preview.system_prompt}</pre></div><div><span className="studio-label">项目约束拼接结果</span><pre className="mt-1 max-h-72 overflow-auto whitespace-pre-wrap rounded-lg border border-white/8 bg-black/25 p-3 text-[11px] leading-5 text-white/60">{preview.context_prompt}</pre></div><div><span className="studio-label">下游 Prompt 规则</span><pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap rounded-lg border border-white/8 bg-black/25 p-3 text-[11px] leading-5 text-white/60">{Object.entries(preview.downstream_rules).map(([key, value]) => `${key}：${value}`).join("\n\n")}</pre></div></> : <div className="rounded-lg border border-dashed border-white/10 p-6 text-center text-xs text-white/35">点击“刷新预览”查看变量替换后的实际内容。</div>}</div>}
                </div>
            </div>
            {showDefaultDetails && <div className="mt-5 rounded-xl border border-amber-300/15 bg-amber-300/[.025] p-4"><div className="flex flex-wrap items-start justify-between gap-3"><div><p className="studio-kicker">SYSTEM DEFAULT · v1</p><strong>系统默认模板全文（只读参考）</strong><p className="mt-1 text-xs leading-5 text-white/45">这是当前系统内置的默认版本。想在它的基础上修改时，先编辑上方内容，再点击“另存为自定义模板”。</p></div><button type="button" className="studio-secondary px-2 py-1 text-xs" onClick={() => setShowDefaultDetails(false)}>收起</button></div>{defaultTemplate ? <div className="mt-4 grid gap-3 lg:grid-cols-2"><details open className="rounded-lg border border-white/8 bg-black/15 p-3"><summary className="cursor-pointer text-sm text-white/80">系统固定机器协议（输出结构）</summary><pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap text-[11px] leading-5 text-white/55">{defaultTemplate.content.machine_contract}</pre></details><details open className="rounded-lg border border-white/8 bg-black/15 p-3"><summary className="cursor-pointer text-sm text-white/80">核心分镜导演规则</summary><pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap text-[11px] leading-5 text-white/55">{defaultTemplate.content.director_template}</pre></details><details open className="rounded-lg border border-white/8 bg-black/15 p-3"><summary className="cursor-pointer text-sm text-white/80">用户建议与项目约束拼接模板</summary><pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap text-[11px] leading-5 text-white/55">{defaultTemplate.content.context_template}</pre></details><details open className="rounded-lg border border-white/8 bg-black/15 p-3"><summary className="cursor-pointer text-sm text-white/80">下游 Prompt 规则</summary><pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap text-[11px] leading-5 text-white/55">{Object.entries(defaultTemplate.content.downstream_rules).map(([key, value]) => `${key}：${value}`).join("\n\n")}</pre></details></div> : <p className="mt-4 text-xs text-white/35">正在加载系统默认模板…</p>}</div>}
            {optimizerOpen && <div className="mt-5 rounded-xl border border-violet-300/15 bg-violet-300/[.035] p-4"><div className="flex items-start justify-between gap-3"><div><p className="studio-kicker">OPENLUX · CLAUDE OPUS 5</p><strong>告诉 AI 这部剧想要什么</strong><p className="mt-1 text-xs leading-5 text-white/45">Opus 5 只生成草稿，不会自动覆盖当前模板。</p></div><button type="button" className="studio-secondary px-2" onClick={() => setOptimizerOpen(false)}><X size={14} /></button></div><textarea className="studio-input mt-3 min-h-24 w-full resize-y" maxLength={4000} value={optimizerInstruction} onChange={(event) => setOptimizerInstruction(event.target.value)} placeholder="例如：这是低成本现实主义悬疑短剧，前3秒必须出现异常线索，少用旁白，多用道具和视线推动，结尾要留下可追更的问题。" /><div className="mt-3 flex flex-wrap gap-3 text-xs text-white/60">{([["storyboard", "分镜"], ["visual", "视觉"], ["keyframe", "首帧"], ["seedance", "Seedance"], ["h3", "H3"]] as const).map(([key, label]) => <label key={key} className="flex items-center gap-1.5"><input type="checkbox" checked={optimizerScopes.includes(key)} onChange={() => setOptimizerScopes((current) => current.includes(key) ? current.filter((item) => item !== key) : [...current, key])} />{label}</label>)}</div><div className="mt-3 flex justify-end"><button type="button" className="studio-primary" disabled={busy === "optimize" || !optimizerInstruction.trim()} onClick={() => void optimize()}>{busy === "optimize" ? <span className="inline-flex items-center gap-2"><WandSparkles className="animate-pulse" size={15} />Opus 5 处理中…</span> : <><WandSparkles size={15} />生成优化草稿</>}</button></div></div>}
            {aiResult && <div className="mt-4 rounded-lg border border-emerald-300/15 bg-emerald-300/[.035] p-3 text-xs leading-5 text-white/60"><strong className="text-emerald-200">AI 优化摘要</strong>{aiResult.draft.change_summary.length > 0 && <ul className="mt-1 list-disc pl-5">{aiResult.draft.change_summary.map((item) => <li key={item}>{item}</li>)}</ul>}{aiResult.expected_effects.length > 0 && <p className="mt-2">预期效果：{aiResult.expected_effects.join("；")}</p>}{aiResult.warnings.length > 0 && <p className="mt-2 text-amber-200/75">注意：{aiResult.warnings.join("；")}</p>}</div>}
            {(error || message) && <div className={error ? "studio-error mt-4" : "studio-notice mt-4"}>{error || message}</div>}
            <div className="mt-5 flex flex-wrap justify-end gap-2"><button type="button" className="studio-secondary" disabled={!!busy || !currentTemplate} onClick={() => void applySelected()}><Check size={15} />应用选中模板</button><button type="button" className="studio-secondary" disabled={!!busy || !draft} onClick={() => void saveAsCustom()}><Save size={15} />另存为自定义模板</button><button type="button" className="studio-primary" disabled={!!busy || !draft} onClick={() => void saveVersion()}><Save size={15} />保存模板版本</button></div>
        </div>}
    </section>;
}
