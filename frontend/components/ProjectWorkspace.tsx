"use client";

/* eslint-disable @next/next/no-img-element */
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import {
    ArrowLeft,
    Check,
    ChevronDown,
    ChevronUp,
    Clipboard,
    CloudOff,
    Download,
    ExternalLink,
    FileText,
    FilePenLine,
    FileClock,
    Film,
    ImageIcon,
    LayoutList,
    Loader2,
    Maximize2,
    MessageSquareText,
    PackageCheck,
    Play,
    Plus,
    RefreshCw,
    Save,
    Server,
    Settings2,
    Sparkles,
    Trash2,
    Upload,
    Users,
    WandSparkles,
    X,
} from "lucide-react";

import {
    approveStoryboard,
    analyzeProjectBrief,
    cancelRenderJob,
    comfyPreflight,
    createBlankShot,
    deleteProjectAsset,
    deleteShot,
    enqueueRender,
    exportKeyframes,
    finalizeProject,
    generateKeyframes,
    generateStoryboard,
    generateSubtitles,
    getProject,
    getRenderJobs,
    planRender,
    projectDownloadUrl,
    reorderStoryboard,
    rewriteProjectScript,
    storyboardCsvUrl,
    updateProject,
    updateShot,
    uploadProjectAsset,
    uploadProjectScript,
} from "@/lib/api";
import type { Asset, AssetRole, CharacterProfile, Delivery, Project, ProjectAnalysisDraft, ProjectBundle, RenderJob, ScriptRewriteDraft, Shot } from "@/types";

type Tab = "brief" | "storyboard" | "assets" | "production" | "review" | "delivery";

const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: "brief", label: "需求与角色", icon: <FileText size={16} /> },
    { id: "storyboard", label: "分镜设计", icon: <LayoutList size={16} /> },
    { id: "assets", label: "参考素材", icon: <ImageIcon size={16} /> },
    { id: "production", label: "H3 生成", icon: <Film size={16} /> },
    { id: "review", label: "客户确认", icon: <Users size={16} /> },
    { id: "delivery", label: "剪辑交付", icon: <PackageCheck size={16} /> },
];

const ACTIVE_JOBS = new Set(["queued", "submitting", "running", "cancel_requested"]);

export default function ProjectWorkspace({ projectId }: { projectId: string }) {
    const [bundle, setBundle] = useState<ProjectBundle | null>(null);
    const [tab, setTab] = useState<Tab>("brief");
    const [loading, setLoading] = useState(true);
    const [busy, setBusy] = useState("");
    const [error, setError] = useState("");
    const [notice, setNotice] = useState("");

    const refresh = useCallback(async () => {
        try {
            setBundle(await getProject(projectId));
            setError("");
        } catch (caught) {
            setError(caught instanceof Error ? caught.message : String(caught));
        } finally {
            setLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        void refresh();
    }, [refresh]);

    const jobsActive = bundle?.jobs.some((job) => ACTIVE_JOBS.has(job.status)) || false;
    useEffect(() => {
        if (!jobsActive) return;
        const timer = window.setInterval(async () => {
            const jobs = await getRenderJobs(projectId).catch(() => null);
            if (jobs) setBundle((current) => current ? { ...current, jobs } : current);
        }, 2500);
        return () => window.clearInterval(timer);
    }, [jobsActive, projectId]);

    const action = async (key: string, task: () => Promise<unknown>, success: string, reload = true) => {
        setBusy(key);
        setError("");
        setNotice("");
        try {
            await task();
            if (reload) await refresh();
            setNotice(success);
        } catch (caught) {
            setError(caught instanceof Error ? caught.message : String(caught));
            if (reload) await refresh();
        } finally {
            setBusy("");
        }
    };

    if (loading) return <div className="flex min-h-screen items-center justify-center bg-[#080a0f] text-white/50"><Loader2 className="mr-2 animate-spin" />载入项目…</div>;
    if (!bundle) return <div className="min-h-screen bg-[#080a0f] p-10 text-white"><div className="studio-error">{error || "项目不存在"}</div></div>;

    const project = bundle.project;
    return (
        <main className="min-h-screen bg-[#080a0f] text-white">
            <header className="sticky top-0 z-40 border-b border-white/8 bg-[#0d1017]/95 backdrop-blur-xl">
                <div className="mx-auto flex max-w-[1700px] items-center gap-4 px-4 py-3 md:px-8">
                    <Link href="/" className="rounded-lg p-2 text-white/45 hover:bg-white/6 hover:text-white"><ArrowLeft size={19} /></Link>
                    <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2"><h1 className="truncate font-semibold">{project.brief.title}</h1><span className="studio-status">{project.status.replaceAll("_", " ")}</span></div>
                        <p className="text-xs text-white/35">{project.brief.client_name || "未填写客户"} · {project.brief.target_duration_seconds}s · {project.brief.aspect_ratio}</p>
                    </div>
                    <Link href="/settings" className="studio-secondary px-3" title="模型设置"><Settings2 size={15} /></Link><Link href="/logs" className="studio-secondary px-3" title="运行日志"><FileClock size={15} /></Link><button className="studio-secondary" onClick={() => void refresh()}><RefreshCw size={15} /> <span className="hidden md:inline">刷新</span></button>
                </div>
                <nav className="mx-auto flex max-w-[1700px] overflow-x-auto px-4 md:px-8">
                    {TABS.map((item) => <button key={item.id} className={`studio-tab whitespace-nowrap ${tab === item.id ? "studio-tab-active" : ""}`} onClick={() => setTab(item.id)}>{item.icon}{item.label}</button>)}
                </nav>
            </header>

            <div className="mx-auto max-w-[1700px] px-4 py-6 md:px-8">
                {tab !== "production" && error && <div className="studio-error mb-4">{error}</div>}
                {tab !== "production" && notice && <div className="studio-notice mb-4">{notice}</div>}
                {tab === "brief" && <BriefPanel project={project} assets={bundle.assets} busy={busy} setLocal={(next) => setBundle({ ...bundle, project: next })} save={(next) => action("save-project", () => updateProject(next), "项目需求与角色设定已保存")} />}
                {tab === "storyboard" && <StoryboardPanel bundle={bundle} busy={busy} action={action} refresh={refresh} updateLocal={(shots) => setBundle({ ...bundle, shots })} />}
                {tab === "assets" && <AssetsPanel project={project} assets={bundle.assets} busy={busy} action={action} />}
                {tab === "production" && <ProductionPanel bundle={bundle} busy={busy} error={error} notice={notice} action={action} refresh={refresh} />}
                {tab === "review" && <ReviewPanel bundle={bundle} busy={busy} action={action} />}
                {tab === "delivery" && <DeliveryPanel bundle={bundle} busy={busy} action={action} />}
            </div>
        </main>
    );
}

function BriefPanel({ project, assets, busy, setLocal, save }: { project: Project; assets: Asset[]; busy: string; setLocal: (project: Project) => void; save: (project: Project) => Promise<void> }) {
    const [analysis, setAnalysis] = useState<ProjectAnalysisDraft | null>(null);
    const [rewriteOpen, setRewriteOpen] = useState(false);
    const [rewriteMode, setRewriteMode] = useState<"auto" | "expand" | "shorten">("auto");
    const [rewriteSuggestions, setRewriteSuggestions] = useState("");
    const [rewriteDraft, setRewriteDraft] = useState<ScriptRewriteDraft | null>(null);
    const [analysisBusy, setAnalysisBusy] = useState("");
    const [analysisError, setAnalysisError] = useState("");
    const [analysisNotice, setAnalysisNotice] = useState("");
    const [selected, setSelected] = useState<Record<string, boolean>>({ visual_style: true, pacing: true, audience: true, style_bible: true, negative_prompt: true, delivery_notes: true, shot_count: true, characters: true });
    const changeBrief = (key: keyof Project["brief"], value: string | number) => setLocal({ ...project, brief: { ...project.brief, [key]: value } });
    const changeAspect = (aspectRatio: string) => {
        const [width, height] = aspectRatio === "9:16" ? [768, 1344] : aspectRatio === "1:1" ? [1024, 1024] : [1344, 768];
        setLocal({ ...project, brief: { ...project.brief, aspect_ratio: aspectRatio, width, height } });
    };
    const updateCharacter = (id: string, key: keyof CharacterProfile, value: string | string[]) => setLocal({ ...project, characters: project.characters.map((item) => item.id === id ? { ...item, [key]: value } : item) });
    const addCharacter = () => setLocal({ ...project, characters: [...project.characters, { id: `character_${crypto.randomUUID().replaceAll("-", "")}`, name: "新角色", description: "", wardrobe: "", voice_description: "", tts_voice: "", reference_asset_ids: [] }] });
    const uploadScript = async (file: File) => {
        setAnalysisBusy("upload"); setAnalysisError(""); setAnalysisNotice("");
        try { const result = await uploadProjectScript(project.id, file); setLocal(result.project); setAnalysisNotice(`已从 ${file.name} 提取 ${result.extracted_characters} 个字符并填入剧情脚本。`); }
        catch (caught) { setAnalysisError(caught instanceof Error ? caught.message : String(caught)); }
        finally { setAnalysisBusy(""); }
    };
    const runAnalysis = async () => {
        setAnalysisBusy("analyze"); setAnalysisError(""); setAnalysisNotice("");
        try { await save(project); setAnalysis(await analyzeProjectBrief(project.id)); }
        catch (caught) { setAnalysisError(caught instanceof Error ? caught.message : String(caught)); }
        finally { setAnalysisBusy(""); }
    };
    const runRewrite = async () => {
        setAnalysisBusy("rewrite"); setAnalysisError(""); setAnalysisNotice("");
        try { setLocal(await updateProject(project)); setRewriteDraft(await rewriteProjectScript(project.id, rewriteMode, rewriteSuggestions.trim())); }
        catch (caught) { setAnalysisError(caught instanceof Error ? caught.message : String(caught)); }
        finally { setAnalysisBusy(""); }
    };
    const closeRewrite = () => { if (!analysisBusy) { setRewriteOpen(false); setRewriteDraft(null); } };
    const applyRewrite = async () => {
        if (!rewriteDraft) return;
        const next: Project = { ...project, brief: { ...project.brief, story: rewriteDraft.rewritten_story }, ai_recommended_shot_count: null };
        setLocal(next); setAnalysisBusy("apply-rewrite"); setAnalysisError("");
        try { setLocal(await updateProject(next)); setRewriteOpen(false); setRewriteDraft(null); setAnalysisNotice("AI 改写剧本已应用并保存，原有 AI 镜头数建议已清空，可重新分析或生成分镜。"); }
        catch (caught) { setAnalysisError(caught instanceof Error ? caught.message : String(caught)); }
        finally { setAnalysisBusy(""); }
    };
    const applyAnalysis = async () => {
        if (!analysis) return;
        const brief = { ...project.brief };
        for (const key of ["visual_style", "pacing", "audience", "negative_prompt", "delivery_notes"] as const) if (selected[key]) brief[key] = analysis[key];
        let characters = project.characters;
        if (selected.characters) {
            const drafts = analysis.characters;
            characters = project.characters.map((current) => {
                const draft = drafts.find((item) => item.character_id === current.id || (!item.character_id && item.name === current.name));
                return draft ? { ...current, name: draft.name, description: draft.description, wardrobe: draft.wardrobe, voice_description: draft.voice_description } : current;
            });
            const existingIds = new Set(characters.map((item) => item.id));
            const existingNames = new Set(characters.map((item) => item.name));
            for (const draft of drafts) if ((!draft.character_id || !existingIds.has(draft.character_id)) && !existingNames.has(draft.name)) characters.push({ id: `character_${crypto.randomUUID().replaceAll("-", "")}`, name: draft.name, description: draft.description, wardrobe: draft.wardrobe, voice_description: draft.voice_description, tts_voice: "", reference_asset_ids: [] });
        }
        const next: Project = { ...project, brief, style_bible: selected.style_bible ? analysis.style_bible : project.style_bible, characters, ai_recommended_shot_count: selected.shot_count ? analysis.recommended_shot_count : project.ai_recommended_shot_count };
        setLocal(next); setAnalysisBusy("apply");
        try { await save(next); setAnalysis(null); setAnalysisNotice("已应用所选分析结果并保存。你仍可手动调整任何字段。"); }
        catch (caught) { setAnalysisError(caught instanceof Error ? caught.message : String(caught)); }
        finally { setAnalysisBusy(""); }
    };
    return <div className="grid gap-5 xl:grid-cols-[1.3fr_.7fr]">
        <section className="studio-panel">
            <div className="mb-5 flex flex-wrap items-center justify-between gap-3"><div><p className="studio-kicker">CLIENT BRIEF</p><h2 className="text-xl font-semibold">客户需求</h2></div><div className="flex flex-wrap gap-2"><label className="studio-secondary cursor-pointer"><input className="hidden" type="file" accept=".txt,.md,.markdown,.docx,.pdf" disabled={!!analysisBusy} onChange={(event) => event.target.files?.[0] && void uploadScript(event.target.files[0])} />{analysisBusy === "upload" ? <Loader2 className="animate-spin" size={15} /> : <Upload size={15} />}上传剧本</label><button className="studio-secondary" disabled={!!analysisBusy || !project.brief.story.trim()} onClick={() => { setRewriteOpen(true); setRewriteDraft(null); }}><FilePenLine size={15} />AI 扩写/缩写</button><button className="studio-secondary" disabled={!!analysisBusy || !project.brief.story.trim()} onClick={() => void runAnalysis()}>{analysisBusy === "analyze" ? <Loader2 className="animate-spin" size={15} /> : <WandSparkles size={15} />}AI 分析并回填</button><button className="studio-primary" disabled={busy === "save-project"} onClick={() => void save(project)}>{busy === "save-project" ? <Loader2 className="animate-spin" size={16} /> : <Save size={16} />} 保存</button></div></div>
            {analysisError && <div className="studio-error mb-4">{analysisError}</div>}{analysisNotice && <div className="studio-notice mb-4">{analysisNotice}</div>}
            <div className="grid gap-4 md:grid-cols-2">
                <Field label="项目名称"><input value={project.brief.title} onChange={(event) => changeBrief("title", event.target.value)} /></Field>
                <Field label="客户名称"><input value={project.brief.client_name} onChange={(event) => changeBrief("client_name", event.target.value)} /></Field>
                <Field label="目标时长（秒）"><input type="number" min={1} value={project.brief.target_duration_seconds} onChange={(event) => changeBrief("target_duration_seconds", Number(event.target.value))} /></Field>
                <Field label="画幅"><select value={project.brief.aspect_ratio} onChange={(event) => changeAspect(event.target.value)}><option>16:9</option><option>9:16</option><option>1:1</option></select><p className="mt-1 text-[11px] text-white/25">{project.brief.width}×{project.brief.height}</p></Field>
                <Field label="剧情/故事脚本" wide><textarea rows={10} value={project.brief.story} onChange={(event) => changeBrief("story", event.target.value)} /></Field>
                <Field label="视觉风格"><textarea rows={4} value={project.brief.visual_style} onChange={(event) => changeBrief("visual_style", event.target.value)} /></Field>
                <Field label="叙事节奏"><textarea rows={4} value={project.brief.pacing} onChange={(event) => changeBrief("pacing", event.target.value)} /></Field>
                <Field label="目标受众"><textarea rows={4} value={project.brief.audience} onChange={(event) => changeBrief("audience", event.target.value)} /></Field>
                <Field label="统一风格圣经" wide><textarea rows={5} value={project.style_bible} onChange={(event) => setLocal({ ...project, style_bible: event.target.value })} placeholder="统一色彩、材质、镜头语言、禁止出现的风格漂移…" /></Field>
                <Field label="负面提示词"><textarea rows={3} value={project.brief.negative_prompt} onChange={(event) => changeBrief("negative_prompt", event.target.value)} /></Field>
                <Field label="交付备注"><textarea rows={3} value={project.brief.delivery_notes} onChange={(event) => changeBrief("delivery_notes", event.target.value)} /></Field>
            </div>
        </section>
        <section className="studio-panel">
            <div className="mb-5 flex items-center justify-between"><div><p className="studio-kicker">CHARACTER BIBLE</p><h2 className="text-xl font-semibold">角色一致性</h2></div><button className="studio-secondary" onClick={addCharacter}><Plus size={15} />角色</button></div>
            <div className="space-y-4">
                {project.characters.length === 0 && <p className="rounded-lg border border-dashed border-white/10 p-5 text-sm text-white/35">添加主要角色，并给角色绑定已上传的参考图；这些描述会写入所有相关分镜提示词。</p>}
                {project.characters.map((character) => <div key={character.id} className="rounded-lg border border-white/8 bg-black/15 p-4">
                    <div className="mb-3 flex gap-2"><input className="studio-input font-semibold" value={character.name} onChange={(event) => updateCharacter(character.id, "name", event.target.value)} /><button className="rounded px-2 text-white/25 hover:text-red-300" onClick={() => setLocal({ ...project, characters: project.characters.filter((item) => item.id !== character.id) })}><Trash2 size={15} /></button></div>
                    <textarea className="studio-input mb-2" rows={3} value={character.description} onChange={(event) => updateCharacter(character.id, "description", event.target.value)} placeholder="外貌、发型、年龄感、身体特征…" />
                    <textarea className="studio-input mb-2" rows={2} value={character.wardrobe} onChange={(event) => updateCharacter(character.id, "wardrobe", event.target.value)} placeholder="固定服装、配饰、颜色…" />
                    <textarea className="studio-input mb-2" rows={2} value={character.voice_description} onChange={(event) => updateCharacter(character.id, "voice_description", event.target.value)} placeholder="声音、语气、口音与说话节奏…" />
                    <input className="studio-input mb-2" value={character.tts_voice || ""} onChange={(event) => updateCharacter(character.id, "tts_voice", event.target.value)} placeholder="可选：专属 TTS Voice ID；留空按男女声自动选择" />
                    <p className="mb-2 mt-3 text-xs text-white/40">绑定角色参考图</p>
                    <div className="space-y-1">{assets.filter((asset) => asset.type === "image").map((asset) => <label key={asset.id} className="flex items-center gap-2 text-xs text-white/60"><input type="checkbox" checked={character.reference_asset_ids.includes(asset.id)} onChange={() => updateCharacter(character.id, "reference_asset_ids", toggle(character.reference_asset_ids, asset.id))} />{asset.name}</label>)}</div>
                </div>)}
            </div>
        </section>
        {rewriteOpen && <div className="studio-modal" onMouseDown={closeRewrite}><section className="studio-dialog max-w-6xl" onMouseDown={(event) => event.stopPropagation()}><p className="studio-kicker">AI SCRIPT REWRITE</p><div className="mb-5 flex items-start justify-between gap-4"><div><h2 className="text-2xl font-semibold">按目标时长扩写或缩写剧本</h2><p className="mt-1 text-sm leading-6 text-white/45">目标成片 {project.brief.target_duration_seconds} 秒。AI 先生成预览，确认后才会覆盖并保存当前剧本。</p></div><button className="studio-secondary" disabled={!!analysisBusy} onClick={closeRewrite}>关闭</button></div>{!rewriteDraft ? <div className="space-y-5"><div className="grid gap-4 md:grid-cols-2"><Field label="改写方式"><select value={rewriteMode} onChange={(event) => setRewriteMode(event.target.value as typeof rewriteMode)}><option value="auto">AI 根据时长自动判断</option><option value="shorten">只缩写 · 保留核心剧情</option><option value="expand">只扩写 · 补足目标时长</option></select></Field><div className="rounded-lg border border-cyan-300/10 bg-cyan-300/[.035] px-4 py-3"><p className="text-xs text-white/35">当前约束</p><p className="mt-1 text-sm text-cyan-100/75">原稿 {project.brief.story.length} 字符 · 目标 {project.brief.target_duration_seconds} 秒 · 单镜 ≤15 秒</p></div></div><label><span className="studio-label">给 AI 的改写建议（可选）</span><textarea className="studio-input min-h-40 resize-y" maxLength={4000} value={rewriteSuggestions} onChange={(event) => setRewriteSuggestions(event.target.value)} placeholder="例如：保留所有关键问答，但合并重复流程；重点突出人物冲突和结尾反思。或者：增加开场铺垫、人物动机和两个情绪转折，不改变原结局……" autoFocus /></label><div className="flex items-center justify-between text-xs text-white/30"><span>建议会作为本次改写的高优先级要求</span><span>{rewriteSuggestions.length}/4000</span></div><div className="flex justify-end gap-2"><button className="studio-secondary" onClick={closeRewrite}>取消</button><button className="studio-primary" disabled={analysisBusy === "rewrite"} onClick={() => void runRewrite()}>{analysisBusy === "rewrite" ? <Loader2 className="animate-spin" size={16} /> : <WandSparkles size={16} />}生成改写预览</button></div></div> : <div><div className="mb-4 grid gap-3 sm:grid-cols-3"><div className="rounded-lg border border-white/8 bg-black/15 p-3"><p className="text-xs text-white/35">改写方向</p><strong className="mt-1 block">{rewriteDraft.rewrite_mode === "expand" ? "扩写" : rewriteDraft.rewrite_mode === "shorten" ? "缩写" : "平衡改写"}</strong></div><div className="rounded-lg border border-white/8 bg-black/15 p-3"><p className="text-xs text-white/35">预计可实现时长</p><strong className="mt-1 block text-cyan-200">{rewriteDraft.estimated_duration_seconds.toFixed(0)} 秒 / 目标 {project.brief.target_duration_seconds} 秒</strong></div><div className="rounded-lg border border-white/8 bg-black/15 p-3"><p className="text-xs text-white/35">文本长度变化</p><strong className="mt-1 block">{project.brief.story.length} → {rewriteDraft.rewritten_story.length} 字符</strong></div></div><div className="mb-4 rounded-lg border border-cyan-300/10 bg-cyan-300/[.03] px-4 py-3"><p className="text-xs text-white/35">改动摘要</p><p className="mt-1 text-sm leading-6 text-white/70">{rewriteDraft.change_summary || "AI 未提供摘要"}</p></div>{rewriteDraft.feasibility_notes.length > 0 && <div className="mb-4 rounded-lg border border-amber-300/15 bg-amber-300/[.04] px-4 py-3"><p className="text-xs font-semibold text-amber-100/70">制作提醒</p><ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-white/55">{rewriteDraft.feasibility_notes.map((note) => <li key={note}>{note}</li>)}</ul></div>}<div className="grid gap-4 lg:grid-cols-2"><label><span className="studio-label">原始剧本（不会直接修改）</span><textarea className="studio-input min-h-[42vh] resize-y text-white/45" value={project.brief.story} readOnly /></label><label><span className="studio-label">AI 改写预览（应用前仍可手动调整）</span><textarea className="studio-input min-h-[42vh] resize-y" value={rewriteDraft.rewritten_story} onChange={(event) => setRewriteDraft({ ...rewriteDraft, rewritten_story: event.target.value })} /></label></div><div className="mt-5 flex flex-wrap justify-end gap-2"><button className="studio-secondary" disabled={!!analysisBusy} onClick={() => setRewriteDraft(null)}>返回调整建议</button><button className="studio-primary" disabled={analysisBusy === "apply-rewrite" || !rewriteDraft.rewritten_story.trim()} onClick={() => void applyRewrite()}>{analysisBusy === "apply-rewrite" ? <Loader2 className="animate-spin" size={16} /> : <Check size={16} />}应用并保存新剧本</button></div></div>}</section></div>}
        {analysis && <div className="studio-modal" onMouseDown={() => setAnalysis(null)}><section className="studio-dialog max-w-5xl" onMouseDown={(event) => event.stopPropagation()}><p className="studio-kicker">AI SCRIPT ANALYSIS</p><div className="mb-5 flex items-start justify-between gap-4"><div><h2 className="text-2xl font-semibold">选择要回填的分析结果</h2><p className="mt-1 text-sm text-white/40">当前表单不会立刻被覆盖；勾选后再应用并保存。</p></div><button className="studio-secondary" onClick={() => setAnalysis(null)}>关闭</button></div><div className="grid max-h-[65vh] gap-3 overflow-y-auto pr-1 md:grid-cols-2">{([
            ["visual_style", "视觉风格", analysis.visual_style], ["pacing", "叙事节奏", analysis.pacing], ["audience", "目标受众", analysis.audience], ["style_bible", "统一风格圣经", analysis.style_bible], ["negative_prompt", "负面提示词", analysis.negative_prompt], ["delivery_notes", "交付备注", analysis.delivery_notes],
        ] as [string, string, string][]).map(([key, label, value]) => <label key={key} className="rounded-lg border border-white/8 bg-black/15 p-4"><div className="mb-2 flex items-center gap-2"><input type="checkbox" checked={selected[key]} onChange={(event) => setSelected({ ...selected, [key]: event.target.checked })} /><strong>{label}</strong></div><p className="whitespace-pre-wrap text-sm leading-6 text-white/55">{value || "AI 未提供"}</p></label>)}<label className="rounded-lg border border-cyan-300/15 bg-cyan-300/[.03] p-4"><div className="mb-2 flex items-center gap-2"><input type="checkbox" checked={selected.shot_count} onChange={(event) => setSelected({ ...selected, shot_count: event.target.checked })} /><strong>AI 推荐分镜数：{analysis.recommended_shot_count} 镜</strong></div><p className="text-sm leading-6 text-white/55">{analysis.shot_count_reason}</p></label><label className="rounded-lg border border-cyan-300/15 bg-cyan-300/[.03] p-4 md:col-span-2"><div className="mb-3 flex items-center gap-2"><input type="checkbox" checked={selected.characters} onChange={(event) => setSelected({ ...selected, characters: event.target.checked })} /><strong>角色一致性草稿（{analysis.characters.length} 个角色）</strong></div><div className="grid gap-3 md:grid-cols-2">{analysis.characters.map((character, index) => <div key={`${character.character_id}-${index}`} className="rounded border border-white/8 p-3"><strong>{character.name}</strong><p className="mt-2 text-xs leading-5 text-white/55">{character.description}</p><p className="mt-2 text-xs leading-5 text-white/45">服装：{character.wardrobe}</p><p className="mt-2 text-xs leading-5 text-white/45">声音：{character.voice_description}</p>{character.reference_observations && <p className="mt-2 text-[11px] leading-5 text-cyan-100/45">参考图观察：{character.reference_observations}</p>}</div>)}</div></label>{analysis.analysis_notes.length > 0 && <div className="rounded-lg border border-amber-300/15 bg-amber-300/[.03] p-4 md:col-span-2"><strong className="text-amber-100/80">需要确认</strong><ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-white/50">{analysis.analysis_notes.map((note) => <li key={note}>{note}</li>)}</ul></div>}</div><div className="mt-5 flex justify-end gap-2"><button className="studio-secondary" onClick={() => setAnalysis(null)}>暂不应用</button><button className="studio-primary" disabled={analysisBusy === "apply"} onClick={() => void applyAnalysis()}>{analysisBusy === "apply" ? <Loader2 className="animate-spin" size={15} /> : <Check size={15} />}应用所选并保存</button></div></section></div>}
    </div>;
}

function StoryboardPanel({ bundle, busy, action, refresh, updateLocal }: { bundle: ProjectBundle; busy: string; action: Action; refresh: () => Promise<void>; updateLocal: (shots: Shot[]) => void }) {
    const [count, setCount] = useState(Math.max(1, Math.ceil(bundle.project.brief.target_duration_seconds / 8)));
    const [countMode, setCountMode] = useState<"manual" | "ai">("ai");
    const [open, setOpen] = useState<string | null>(bundle.shots[0]?.id || null);
    const [showGenerate, setShowGenerate] = useState(false);
    const [userSuggestions, setUserSuggestions] = useState("");
    const duration = bundle.shots.reduce((sum, shot) => sum + shot.duration_seconds, 0);
    const isRedo = bundle.shots.length > 0;
    const patchShot = (id: string, patch: Partial<Shot>) => updateLocal(bundle.shots.map((shot) => shot.id === id ? { ...shot, ...patch } : shot));
    const moveShot = (index: number, direction: -1 | 1) => {
        const target = index + direction;
        if (target < 0 || target >= bundle.shots.length) return;
        const ids = bundle.shots.map((shot) => shot.id);
        [ids[index], ids[target]] = [ids[target], ids[index]];
        void action("reorder", () => reorderStoryboard(bundle.project.id, ids), "分镜顺序已更新");
    };
    const submitGeneration = () => {
        setShowGenerate(false);
        void action(
            "storyboard",
            () => generateStoryboard(bundle.project.id, countMode === "manual" ? count : undefined, countMode, userSuggestions.trim()),
            isRedo ? "AI 已根据本次建议重新生成分镜草稿" : "AI 已根据本次建议生成详细分镜草稿",
        );
    };
    return <div>
        <section className="studio-panel mb-5 flex flex-wrap items-center justify-between gap-4">
            <div><p className="studio-kicker">STORYBOARD V{bundle.project.storyboard_version}</p><h2 className="text-xl font-semibold">详细分镜设计表</h2><p className="mt-1 text-sm text-white/40">{bundle.shots.length} 镜 · 合计 {duration.toFixed(1)} 秒 · 每镜强制 ≤15 秒</p></div>
            <div className="flex flex-wrap gap-2"><select className="studio-input w-36" value={countMode} onChange={(event) => setCountMode(event.target.value as "manual" | "ai")}><option value="ai">AI 判断镜头数</option><option value="manual">手动指定镜头数</option></select>{countMode === "manual" ? <input className="studio-input w-20" type="number" min={1} max={500} value={count} onChange={(event) => setCount(Number(event.target.value))} /> : <span className="inline-flex items-center rounded-lg border border-cyan-300/10 bg-cyan-300/[.035] px-3 text-xs text-cyan-100/60">{bundle.project.ai_recommended_shot_count ? `已推荐 ${bundle.project.ai_recommended_shot_count} 镜` : "将按剧情节奏自动判断"}</span>}<button className="studio-primary" disabled={!!busy} onClick={() => setShowGenerate(true)}>{busy === "storyboard" ? <Loader2 className="animate-spin" size={16} /> : <WandSparkles size={16} />} {isRedo ? "AI 按建议重做" : "AI 生成分镜"}</button><button className="studio-secondary" disabled={!!busy} onClick={() => void action("blank-shot", () => createBlankShot(bundle.project.id), "已添加空白镜头")}><Plus size={15} />空白镜头</button><a className="studio-secondary" href={storyboardCsvUrl(bundle.project.id)}><Download size={15} />导出表格</a></div>
        </section>
        {bundle.shots.length === 0 ? <div className="studio-empty"><LayoutList size={30} className="text-cyan-300" /><strong>还没有分镜</strong><span>{countMode === "ai" ? "让 AI 判断镜头数，或切换为手动数量后生成" : "设置镜头数后点击“AI 生成/重做”"}</span></div> : <div className="space-y-3">
            {bundle.shots.map((shot, index) => <article key={shot.id} className="studio-panel !p-0 overflow-hidden">
                <button className="flex w-full items-center gap-4 px-4 py-4 text-left" onClick={() => setOpen(open === shot.id ? null : shot.id)}>
                    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-cyan-300/10 font-mono text-sm text-cyan-200">{String(shot.ordinal).padStart(2, "0")}</span>
                    <div className="min-w-0 flex-1"><strong className="block truncate">{shot.title || shot.narrative}</strong><span className="line-clamp-1 text-xs text-white/40">{shot.narrative}</span></div>
                    <span className={shot.duration_seconds > 15 ? "text-red-300" : "text-white/45"}>{shot.duration_seconds}s</span><span className="studio-status uppercase">{shot.generation_mode}</span>{open === shot.id ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </button>
                {open === shot.id && <div className="border-t border-white/8 p-4 md:p-5">
                    <div className="grid gap-4 md:grid-cols-3">
                        <Field label="镜头标题"><input value={shot.title} onChange={(event) => patchShot(shot.id, { title: event.target.value })} /></Field>
                        <Field label="时长（0.25–15秒）"><input type="number" min={0.25} max={15} step={0.25} value={shot.duration_seconds} onChange={(event) => patchShot(shot.id, { duration_seconds: Math.min(15, Math.max(.25, Number(event.target.value))) })} /></Field>
                        <Field label="生成策略"><select value={shot.generation_mode} onChange={(event) => patchShot(shot.id, { generation_mode: event.target.value as Shot["generation_mode"] })}><option value="auto">自动判断</option><option value="i2v">首帧 I2V</option><option value="r2v">全能参考 R2V</option></select></Field>
                        <Field label="画面叙事" wide><textarea rows={3} value={shot.narrative} onChange={(event) => patchShot(shot.id, { narrative: event.target.value })} /></Field>
                        <Field label="对白/旁白"><textarea rows={3} value={shot.dialogue} onChange={(event) => patchShot(shot.id, { dialogue: event.target.value, dialogue_turns: [] })} /><span className="mt-1 block text-[11px] leading-5 text-white/35">修改整段对白后，系统会在保存/编译计划时重新分析逐句发言者。</span></Field>
                        <Field label="本镜发言者"><select value={shot.dialogue_speaker_id || ""} onChange={(event) => patchShot(shot.id, { dialogue_speaker_id: event.target.value || null, dialogue_turns: [] })}><option value="">旁白 / 自动推断</option>{bundle.project.characters.map((character) => <option key={character.id} value={character.id}>{character.name}</option>)}</select><span className="mt-1 block text-[11px] leading-5 text-white/35">单人对白可直接指定；多人问答请在下方逐句确认。</span></Field>
                        {shot.dialogue_turns.length > 1 && <Field label="逐句发言者（多人问答）" wide><div className="space-y-2 rounded-lg border border-cyan-300/10 bg-cyan-300/[.025] p-3">{shot.dialogue_turns.map((turn, turnIndex) => <div key={`${shot.id}-turn-${turnIndex}`} className="grid gap-2 md:grid-cols-[150px_1fr]"><select value={turn.speaker_id || ""} onChange={(event) => patchShot(shot.id, { dialogue_speaker_id: null, dialogue_turns: shot.dialogue_turns.map((item, index) => index === turnIndex ? { ...item, speaker_id: event.target.value || null } : item) })}><option value="">画外旁白</option>{bundle.project.characters.map((character) => <option key={character.id} value={character.id}>{character.name}</option>)}</select><textarea rows={2} value={turn.text} onChange={(event) => patchShot(shot.id, { dialogue_speaker_id: null, dialogue_turns: shot.dialogue_turns.map((item, index) => index === turnIndex ? { ...item, text: event.target.value } : item) })} /></div>)}<p className="text-[11px] leading-5 text-cyan-100/45">每一行会独立生成该角色的稳定短画面和干净人声，其他人物强制闭嘴。</p></div></Field>}
                        <Field label="对白起始（秒）"><input type="number" min={0} max={15} step={0.05} value={shot.dialogue_start_seconds} onChange={(event) => patchShot(shot.id, { dialogue_start_seconds: Math.max(0, Number(event.target.value)) })} /></Field>
                        <Field label="对白语速调整（%）"><input type="number" min={-50} max={100} step={5} value={shot.dialogue_rate_percent} onChange={(event) => patchShot(shot.id, { dialogue_rate_percent: Math.min(100, Math.max(-50, Number(event.target.value))) })} /><span className="mt-1 block text-[11px] leading-5 text-white/35">正数加快，负数放慢；若台词超时，系统还会自动轻微压缩到镜头内。</span></Field>
                        <Field label="声音设计"><textarea rows={3} value={shot.audio_design} onChange={(event) => patchShot(shot.id, { audio_design: event.target.value })} /></Field>
                        <Field label="场景"><textarea rows={3} value={shot.scene_description} onChange={(event) => patchShot(shot.id, { scene_description: event.target.value })} /></Field>
                        <Field label="景别"><input value={shot.shot_size} onChange={(event) => patchShot(shot.id, { shot_size: event.target.value })} /></Field>
                        <Field label="机位/角度"><input value={shot.camera_angle} onChange={(event) => patchShot(shot.id, { camera_angle: event.target.value })} /></Field>
                        <Field label="镜头/焦段"><input value={shot.lens} onChange={(event) => patchShot(shot.id, { lens: event.target.value })} /></Field>
                        <Field label="运镜"><input value={shot.camera_motion} onChange={(event) => patchShot(shot.id, { camera_motion: event.target.value })} /></Field>
                        <Field label="主体动作"><textarea rows={3} value={shot.subject_motion} onChange={(event) => patchShot(shot.id, { subject_motion: event.target.value })} /></Field>
                        <Field label="本镜头角色" wide><div className="grid gap-2 rounded-lg border border-white/8 p-3 sm:grid-cols-2 lg:grid-cols-3">{bundle.project.characters.length === 0 ? <span className="text-xs text-white/30">请先在“需求与角色”中建立角色。</span> : bundle.project.characters.map((character) => <label key={character.id} className="flex items-center gap-2 text-xs text-white/60"><input type="checkbox" checked={shot.character_ids.includes(character.id)} onChange={() => patchShot(shot.id, { character_ids: toggle(shot.character_ids, character.id) })} />{character.name}</label>)}</div></Field>
                        <Field label="首帧图 Prompt" wide><textarea rows={5} value={shot.visual_prompt} onChange={(event) => patchShot(shot.id, { visual_prompt: event.target.value })} /></Field>
                        <Field label="MiniMax H3 Prompt" wide><textarea rows={6} value={shot.video_prompt} onChange={(event) => patchShot(shot.id, { video_prompt: event.target.value })} /></Field>
                    </div>
                    <div className="mt-4 flex flex-wrap justify-end gap-2"><button className="studio-secondary px-3" disabled={index === 0 || !!busy} onClick={() => moveShot(index, -1)}><ChevronUp size={14} />上移</button><button className="studio-secondary px-3" disabled={index === bundle.shots.length - 1 || !!busy} onClick={() => moveShot(index, 1)}><ChevronDown size={14} />下移</button><button className="studio-danger" onClick={() => void action(`delete-${shot.id}`, () => deleteShot(bundle.project.id, shot.id), `镜头 ${shot.ordinal} 已删除`)}><Trash2 size={14} />删除</button><button className="studio-primary" disabled={busy === `shot-${shot.id}`} onClick={() => void action(`shot-${shot.id}`, () => updateShot(shot), `镜头 ${shot.ordinal} 已保存`, false).then(refresh)}>{busy === `shot-${shot.id}` ? <Loader2 className="animate-spin" size={15} /> : <Save size={15} />}保存镜头</button></div>
                </div>}
            </article>)}
        </div>}
        {showGenerate && <div className="studio-modal" onMouseDown={() => setShowGenerate(false)}><section className="studio-dialog max-w-2xl" onMouseDown={(event) => event.stopPropagation()}><p className="studio-kicker">AI STORYBOARD DIRECTION</p><div className="mb-5 flex items-start gap-3"><span className="rounded-xl bg-cyan-300/10 p-3 text-cyan-200"><MessageSquareText size={22} /></span><div><h2 className="text-2xl font-semibold">{isRedo ? "让 AI 按建议重新修改分镜" : "生成分镜前补充你的建议"}</h2><p className="mt-1 text-sm leading-6 text-white/45">你的文字会和剧情脚本、角色设定、视觉风格一起交给当前分镜模型，并作为本次生成的高优先级要求。</p></div></div>{isRedo && <div className="mb-4 rounded-lg border border-amber-300/15 bg-amber-300/[.04] px-4 py-3 text-sm leading-6 text-amber-100/70">本次会重新生成并替换当前 {bundle.shots.length} 个镜头。需要保留的剧情、镜头或对白，请在建议中明确写出。</div>}<label><span className="studio-label">给 AI 的本次建议（可选）</span><textarea className="studio-input min-h-40 resize-y" maxLength={4000} value={userSuggestions} onChange={(event) => setUserSuggestions(event.target.value)} placeholder={isRedo ? "例如：保留前 3 镜的剧情；中段减少对白、增加动作；结尾改成角色回头的近景，并让节奏更紧凑……" : "例如：前 3 秒必须有强钩子；人物多用近景；减少旁白、用动作推进；整体保持压抑悬疑感……"} autoFocus /></label><div className="mt-2 flex flex-wrap items-center justify-between gap-3 text-xs text-white/30"><span>{countMode === "ai" ? "AI 会结合这份建议重新判断镜头数" : `本次固定生成 ${count} 个镜头`}</span><span>{userSuggestions.length}/4000</span></div><div className="mt-6 flex flex-wrap justify-end gap-2"><button className="studio-secondary" onClick={() => setShowGenerate(false)}>取消</button><button className="studio-primary" onClick={submitGeneration}><WandSparkles size={16} />{userSuggestions.trim() ? (isRedo ? "按建议重新生成" : "按建议生成分镜") : (isRedo ? "不填建议直接重做" : "不填建议直接生成")}</button></div></section></div>}
    </div>;
}

function AssetsPanel({ project, assets, busy, action }: { project: Project; assets: Asset[]; busy: string; action: Action }) {
    const [role, setRole] = useState<AssetRole>("character");
    const [preview, setPreview] = useState<MediaPreviewState | null>(null);
    const upload = (file: File) => action("upload", () => uploadProjectAsset(project.id, file, role), "参考素材已上传");
    return <div className="grid gap-5 xl:grid-cols-[360px_1fr]">
        <section className="studio-panel h-fit"><p className="studio-kicker">REFERENCE LIBRARY</p><h2 className="mb-5 text-xl font-semibold">上传参考素材</h2><Field label="素材用途"><select value={role} onChange={(event) => setRole(event.target.value as AssetRole)}><option value="character">人物形象</option><option value="style">画风参考</option><option value="scene">场景参考</option><option value="motion">动作/运镜视频</option><option value="voice">声音参考</option><option value="music">背景音乐</option><option value="sound_effect">音效</option><option value="other">其他</option></select></Field><label className="studio-empty mt-4 min-h-44 cursor-pointer"><input className="hidden" type="file" accept="image/*,video/*,audio/*,.srt,.vtt" disabled={!!busy} onChange={(event) => event.target.files?.[0] && void upload(event.target.files[0])} />{busy === "upload" ? <Loader2 className="animate-spin text-cyan-300" /> : <Upload className="text-cyan-300" />}<strong>选择图片、视频或音频</strong><span>R2V 可同时绑定多张图、参考视频和音频</span></label><p className="mt-4 text-xs leading-5 text-white/35">素材保存在项目目录；真正提交 H3 时才上传到 ComfyUI。云端服务器现在保持关机即可。</p></section>
        <section className="studio-panel"><div className="mb-5 flex items-center justify-between"><div><p className="studio-kicker">{assets.length} ASSETS</p><h2 className="text-xl font-semibold">项目素材库</h2></div></div>{assets.length === 0 ? <div className="studio-empty"><ImageIcon className="text-cyan-300" /><strong>尚未上传参考素材</strong></div> : <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{assets.map((asset) => <AssetCard key={asset.id} asset={asset} projectId={project.id} preview={() => setPreview({ name: asset.name, url: projectDownloadUrl(project.id, "asset", asset.id), description: asset.description })} remove={() => action(`asset-${asset.id}`, () => deleteProjectAsset(project.id, asset.id), "素材记录已删除")} />)}</div>}</section>
        {preview && <MediaPreview preview={preview} onClose={() => setPreview(null)} />}
    </div>;
}

type H3Preset = "fast" | "balanced" | "quality";
type KeyframeEditorState = {
    shotId: string;
    prompt: string;
    suggestions: string;
    revisionMode: "fresh" | "iterate";
};
type MediaPreviewState = { name: string; url: string; description?: string };

const H3_SCHEDULERS: Shot["h3_scheduler"][] = ["simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic", "kl_optimal"];

function roundTo32(value: number) {
    return Math.max(32, Math.round(value / 32) * 32);
}

function h3FramesForSeconds(seconds: number) {
    const requested = Math.max(5, Math.min(15, seconds)) * 24;
    return Math.max(124, Math.min(362, 5 + 17 * Math.ceil((requested - 5) / 17)));
}

function h3PresetPatch(preset: H3Preset, project: Project): Partial<Shot> {
    const [ratioWidth, ratioHeight] = project.brief.aspect_ratio.split(":").map(Number);
    const ratio = ratioWidth > 0 && ratioHeight > 0 ? ratioWidth / ratioHeight : project.brief.width / project.brief.height;
    const landscape = ratio >= 1;
    const longEdge = preset === "fast" ? 608 : preset === "quality" ? Math.max(1344, project.brief.width, project.brief.height) : null;
    const width = longEdge === null ? project.brief.width : landscape ? longEdge : roundTo32(longEdge * ratio);
    const height = longEdge === null ? project.brief.height : landscape ? roundTo32(longEdge / ratio) : longEdge;
    return {
        h3_width: roundTo32(width),
        h3_height: roundTo32(height),
        generation_mode: "auto",
        ref_image_size: "match",
        h3_turbo: preset !== "quality",
        h3_steps: preset === "quality" ? 20 : 4,
        h3_scheduler: "simple",
        h3_denoise: 1,
        h3_lora_strength: 1,
        h3_low_vram: false,
        h3_shift_video: 12,
        h3_shift_audio: 3,
    };
}

function H3ShotCard({ shot, project, assets, checked, busy, toggleChecked, save }: { shot: Shot; project: Project; assets: Asset[]; checked: boolean; busy: boolean; toggleChecked: () => void; save: (shot: Shot) => Promise<void> }) {
    const [draft, setDraft] = useState(shot);
    const width = draft.h3_width || project.brief.width;
    const height = draft.h3_height || project.brief.height;
    const baseline = Math.max(1, project.brief.width * project.brief.height * 124 * 4);
    const relativeWork = width * height * draft.render_frames * draft.h3_steps / baseline;
    const patch = (values: Partial<Shot>) => setDraft((current) => ({ ...current, ...values }));
    const applyPreset = (preset: H3Preset) => patch(h3PresetPatch(preset, project));
    const effectiveMode = draft.generation_mode === "auto"
        ? (draft.keyframe_asset_id || draft.image_path ? "i2v" : draft.reference_asset_ids.length ? "r2v" : "i2v")
        : draft.generation_mode;
    const imageReferenceCount = draft.reference_asset_ids.filter((assetId) => assets.some((asset) => asset.id === assetId && asset.type === "image")).length;
    const tuningWarnings = [
        draft.generation_mode === "r2v" && !!(draft.keyframe_asset_id || draft.image_path) ? "已绑定完整首帧但强制选择了 R2V，H3 会忽略首帧并从零重组人物，容易复制人物和重影；建议改为“自动判断”或“首帧 I2V”。" : "",
        effectiveMode === "r2v" && imageReferenceCount > 1 ? `当前 R2V 同时使用 ${imageReferenceCount} 张人物图；多人构图更容易混脸和重影。成片优先先生成一张完整分镜首帧，再用 I2V。` : "",
        effectiveMode === "i2v" && !(draft.keyframe_asset_id || draft.image_path) ? "当前会走 I2V，但还没有首帧；请先点“生成所选首帧”。" : "",
        draft.ref_image_size === "max" ? "参考图尺寸为 max；会明显增加参考编码开销。官方模板默认 match，除非强制 R2V 且身份仍不稳，否则建议 match。" : "",
        draft.h3_turbo && draft.h3_steps !== 4 ? `当前启用了 4-step Turbo，但步数是 ${draft.h3_steps}；LoRA 与步数不匹配会让画面发糊、过锐或动作不稳。` : "",
        !draft.h3_turbo && draft.h3_steps < 20 ? `当前走官方原生采样，但只有 ${draft.h3_steps} 步；最终成片建议 20 步，低步数更容易脸、手和肢体未收敛。` : "",
        draft.h3_scheduler !== "simple" ? `当前 Scheduler 是 ${draft.h3_scheduler}；4-step Turbo 建议使用 simple。` : "",
        draft.h3_denoise !== 1 ? `Denoise 当前为 ${draft.h3_denoise}；低于 1 可能出现去噪不足、画面灰糊或动作偏弱。` : "",
        draft.h3_turbo && draft.h3_lora_strength !== 1 ? `Turbo LoRA 强度当前为 ${draft.h3_lora_strength}；匹配值是 1.0。` : "",
        draft.h3_low_vram ? "低显存是旧版自定义链路参数；新版官方采样链路不会提交它。RTX 5090 32GB 也不需要开启。" : "",
        width * height < 900_000 ? `当前仅 ${(width * height / 1_000_000).toFixed(2)}MP，适合预览；最终成片建议使用 1344×768（约 0.98MP）。` : "",
    ].filter(Boolean);
    return <article className="rounded-xl border border-white/8 bg-black/15 p-4">
        <div className="flex flex-wrap items-start gap-3">
            <input type="checkbox" checked={checked} onChange={toggleChecked} />
            <div className="min-w-48 flex-1"><strong>#{shot.ordinal} {shot.title}</strong><p className="line-clamp-1 text-xs text-white/35">{shot.video_prompt}</p></div>
            <span className="studio-status uppercase">{draft.generation_mode === "auto" ? `AUTO→${effectiveMode}` : effectiveMode}</span>
            <span className="rounded bg-cyan-300/8 px-2 py-1 font-mono text-[11px] text-cyan-100/70">{width}×{height} · {draft.render_frames}帧 · {draft.h3_turbo ? "Turbo" : "原生"}{draft.h3_steps}步 · 约 {relativeWork.toFixed(1)}×算力</span>
        </div>
        {tuningWarnings.length > 0 && <div className="mt-3 rounded-lg border border-amber-300/15 bg-amber-300/[.04] px-3 py-2.5"><strong className="text-xs text-amber-100/80">当前参数有 {tuningWarnings.length} 项画质提醒</strong><ul className="mt-1 list-disc space-y-1 pl-4 text-[11px] leading-5 text-amber-50/55">{tuningWarnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></div>}
        <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <Field label="速度/质量预设" hint="快速和均衡使用官方 4 步加速；清晰优先改用官方原生 20 步并关闭 Turbo LoRA，主要用于最终成片。"><select defaultValue="custom" onChange={(event) => event.target.value !== "custom" && applyPreset(event.target.value as H3Preset)}><option value="custom">当前/自定义</option><option value="fast">快速预览 · 608长边 / Turbo 4步</option><option value="balanced">均衡 · 项目分辨率 / Turbo 4步</option><option value="quality">清晰优先 · 1344长边 / 原生20步</option></select></Field>
            <Field label="生成宽度" hint="决定横向细节；越高越清晰，也越慢、越占显存。必须是 32 的倍数。"><input type="number" min={32} max={4096} step={32} value={width} onChange={(event) => patch({ h3_width: Number(event.target.value) || null })} /></Field>
            <Field label="生成高度" hint="决定纵向细节；需与宽度保持目标画幅。16:9 成片推荐 1344×768。"><input type="number" min={32} max={4096} step={32} value={height} onChange={(event) => patch({ h3_height: Number(event.target.value) || null })} /></Field>
            <Field label="采样质量链路" hint="Turbo 4步适合预览；原生20步绕过加速 LoRA，让脸、手、服装和运动有更多收敛机会，是最终成片默认。"><select value={draft.h3_turbo ? "turbo" : "native"} onChange={(event) => patch(event.target.value === "turbo" ? { h3_turbo: true, h3_steps: 4 } : { h3_turbo: false, h3_steps: 20, h3_lora_strength: 1 })}><option value="turbo">Turbo · 4步快速预览</option><option value="native">原生 · 20步质量优先</option></select></Field>
            <Field label="采样步数" hint={draft.h3_turbo ? "Turbo LoRA 与 4 步成套匹配，请保持 4。" : "原生链路官方默认 20 步；减少会更快，但人物细节和动作稳定性会下降。"}><input type="number" min={1} max={100} value={draft.h3_steps} onChange={(event) => patch({ h3_steps: Math.max(1, Number(event.target.value) || 1) })} /></Field>
            <Field label="H3 时长（5–15秒）" hint="越长帧数越多、生成越慢，也更容易人物漂移。复杂动作建议拆成 5–8 秒小镜头。"><input type="number" min={5} max={15} step={0.25} value={draft.duration_seconds} onChange={(event) => { const duration = Math.max(5, Math.min(15, Number(event.target.value) || 5)); patch({ duration_seconds: duration, render_frames: h3FramesForSeconds(duration) }); }} /></Field>
            <Field label="固定 Seed（留空则随机）" hint="固定后可用同一构图比较不同参数；换 Seed 会改变人物姿态、构图和细节，但不代表质量一定更高。"><input type="number" min={1} max={2147483647} value={draft.h3_seed ?? ""} placeholder="每次随机" onChange={(event) => patch({ h3_seed: event.target.value ? Number(event.target.value) : null })} /></Field>
            <Field label="首帧（I2V）" hint="最强的构图与人物锚点。高质量首帧能明显改善脸、服装和场景稳定性；首帧缺陷也会被继承。"><select value={draft.keyframe_asset_id || ""} onChange={(event) => patch({ keyframe_asset_id: event.target.value || null })}><option value="">未绑定</option>{assets.filter((asset) => asset.type === "image").map((asset) => <option key={asset.id} value={asset.id}>{asset.name}</option>)}</select></Field>
            <div className="md:col-span-2"><span className="studio-label">全能参考（R2V，可多选）</span><div className="max-h-28 space-y-1 overflow-y-auto rounded-lg border border-white/8 p-2">{assets.filter((asset) => ["image", "video", "audio"].includes(asset.type)).map((asset) => <label key={asset.id} className="flex items-center gap-2 text-xs text-white/60"><input type="checkbox" checked={draft.reference_asset_ids.includes(asset.id)} onChange={() => patch({ reference_asset_ids: toggle(draft.reference_asset_ids, asset.id) })} />{asset.type.toUpperCase()} · {asset.name}</label>)}</div><p className="mt-1.5 text-[11px] leading-5 text-white/32">只绑定本镜真正出现的人物、动作或声音。无关或互相冲突的参考越多，模型越容易混脸、串衣服、构图失控。</p></div>
        </div>
        <details className="mt-4 rounded-lg border border-white/8 bg-white/[.02] p-3">
            <summary className="cursor-pointer text-xs font-semibold text-white/55">高级参数（已对齐当前官方 ComfyUI H3 模板）</summary>
            <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                <Field label="Sampler / Scheduler" hint="实际 Sampler 固定为官方 res_multistep；这里控制噪声日程。官方模板使用 simple，其他选项可能改变对比度、锐度与运动稳定性。"><select value={draft.h3_scheduler} onChange={(event) => patch({ h3_scheduler: event.target.value as Shot["h3_scheduler"] })}>{H3_SCHEDULERS.map((scheduler) => <option key={scheduler} value={scheduler}>{scheduler}{scheduler === "simple" ? " · 官方默认" : " · 实验"}</option>)}</select></Field>
                <Field label="Denoise（去噪强度）" hint="1.0 是完整生成。调低会减少变化，但在这套工作流中容易去噪不足、画面灰糊或动作弱；建议保持 1。"><input type="number" min={0} max={1} step={0.01} value={draft.h3_denoise} onChange={(event) => patch({ h3_denoise: Number(event.target.value) })} /></Field>
                <Field label="Turbo LoRA 强度" hint={draft.h3_turbo ? "只在 Turbo 链路生效。1.0 是 4 步模型匹配值；偏离会产生未收敛、锐化、纹理噪点或重影。" : "原生20步已绕过 Turbo LoRA，因此本项不参与生成。"}><input disabled={!draft.h3_turbo} type="number" min={-10} max={10} step={0.05} value={draft.h3_lora_strength} onChange={(event) => patch({ h3_lora_strength: Number(event.target.value) })} /></Field>
                <Field label="参考图尺寸" hint="只影响 R2V。match 是官方模板默认值；max 保留更大参考编码，但不修复多人重影，反而更慢、更占显存。"><select value={draft.ref_image_size} onChange={(event) => patch({ ref_image_size: event.target.value })}><option value="match">match · 官方默认</option><option value="max">max · 高显存实验</option></select></Field>
            </div>
            <p className="mt-3 text-[11px] leading-5 text-amber-100/55">最终成片推荐：原生 20 步 + res_multistep + simple + Denoise 1 + 1344×768 + 清晰完整首帧。旧版自定义 Sigma Shift、TurboSampler 和低显存合并参数已从提交链路移除，避免和官方模板不一致。</p>
        </details>
        <div className="mt-4 flex justify-end"><button className="studio-primary" disabled={busy} onClick={() => void save(draft)}>{busy ? <Loader2 className="animate-spin" size={14} /> : <Save size={14} />}保存本镜参数</button></div>
    </article>;
}

function ProductionPanel({ bundle, busy, error, notice, action, refresh }: { bundle: ProjectBundle; busy: string; error: string; notice: string; action: Action; refresh: () => Promise<void> }) {
    const [preflight, setPreflight] = useState<Record<string, unknown> | null>(null);
    const [selected, setSelected] = useState<string[]>(bundle.shots[0] ? [bundle.shots[0].id] : []);
    const [keyframeBoardOpen, setKeyframeBoardOpen] = useState(false);
    const [keyframeEditor, setKeyframeEditor] = useState<KeyframeEditorState | null>(null);
    const [keyframePreview, setKeyframePreview] = useState<MediaPreviewState | null>(null);
    const active = bundle.jobs.filter((job) => ACTIVE_JOBS.has(job.status));
    const cost = bundle.jobs.reduce((sum, job) => sum + (job.estimated_cost || 0), 0);
    const keyframeAssets = new Map(bundle.assets.map((asset) => [asset.id, asset]));
    const generatedShotIds = bundle.shots.filter((shot) => shot.keyframe_asset_id && keyframeAssets.has(shot.keyframe_asset_id)).map((shot) => shot.id);
    const generatedKeyframes = generatedShotIds.length;
    const selectedKeyframes = generatedShotIds.filter((shotId) => selected.includes(shotId));
    const allGeneratedSelected = generatedShotIds.length > 0 && selectedKeyframes.length === generatedShotIds.length;
    const editorShot = keyframeEditor ? bundle.shots.find((shot) => shot.id === keyframeEditor.shotId) : undefined;
    const editorKeyframe = editorShot?.keyframe_asset_id ? keyframeAssets.get(editorShot.keyframe_asset_id) : undefined;
    const runPreflight = () => action("preflight", async () => { setPreflight(await comfyPreflight(bundle.project.id)); }, "服务器检查完成；结果显示在当前按钮下方。", false);
    const generateSelectedKeyframes = () => action("keyframes", () => generateKeyframes(bundle.project.id, selected), `所选 ${selected.length} 镜的分镜首帧已处理；成功图片显示在下方并自动绑定到 I2V 首帧。`);
    const toggleAllGeneratedKeyframes = () => setSelected(allGeneratedSelected ? [] : generatedShotIds);
    const exportSelectedKeyframes = () => action(
        "export-keyframes",
        async () => {
            const { blob, filename } = await exportKeyframes(bundle.project.id, selectedKeyframes);
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.setTimeout(() => URL.revokeObjectURL(url), 1000);
        },
        `已将所选 ${selectedKeyframes.length} 张分镜图打包导出。`,
        false,
    );
    const openKeyframeEditor = (shot: Shot) => {
        const currentKeyframe = shot.keyframe_asset_id ? keyframeAssets.get(shot.keyframe_asset_id) : undefined;
        setKeyframeEditor({
            shotId: shot.id,
            prompt: currentKeyframe?.description || shot.visual_prompt,
            suggestions: "",
            revisionMode: currentKeyframe ? "iterate" : "fresh",
        });
    };
    const saveKeyframePrompt = async () => {
        if (!editorShot || !keyframeEditor?.prompt.trim()) return;
        let completed = false;
        await action(
            `keyframe-prompt-${editorShot.id}`,
            async () => {
                await updateShot({ ...editorShot, visual_prompt: keyframeEditor.prompt.trim() });
                completed = true;
            },
            `镜头 ${editorShot.ordinal} 的首帧 Prompt 已保存。`,
        );
        if (completed) setKeyframeEditor(null);
    };
    const regenerateKeyframe = async () => {
        if (!editorShot || !keyframeEditor?.prompt.trim()) return;
        let completed = false;
        await action(
            `keyframe-${editorShot.id}`,
            async () => {
                await updateShot({ ...editorShot, visual_prompt: keyframeEditor.prompt.trim() });
                await generateKeyframes(bundle.project.id, [editorShot.id], {
                    revisionMode: keyframeEditor.revisionMode,
                    userSuggestions: keyframeEditor.suggestions.trim(),
                });
                completed = true;
            },
            `镜头 ${editorShot.ordinal} 的分镜首帧已生成并自动绑定。`,
        );
        if (completed) setKeyframeEditor(null);
    };
    const saveShot = (shot: Shot) => action(`h3-${shot.id}`, () => updateShot(shot), `镜头 ${shot.ordinal} 的 H3 参数已保存`);
    const applyBulkPreset = (preset: H3Preset) => action(`preset-${preset}`, () => Promise.all(bundle.shots.filter((shot) => selected.includes(shot.id)).map((shot) => updateShot({ ...shot, ...h3PresetPatch(preset, bundle.project) }))), `已将${preset === "fast" ? "快速预览" : preset === "balanced" ? "均衡" : "清晰优先"}应用到 ${selected.length} 个镜头`);
    const busyText = busy === "preflight" ? "正在检查 ComfyUI 节点与模型…"
        : busy === "keyframes" ? `正在调用分镜图模型生成 ${selected.length} 张首帧，请稍候…`
            : busy === "plan" ? "正在编译所有镜头的 H3 模式、参考素材、帧数和参数…"
                : busy === "render" ? `正在把 ${selected.length} 个镜头写入本地持久队列…`
                    : busy.startsWith("preset-") ? "正在保存所选镜头的批量预设…"
                        : busy.startsWith("keyframe-prompt-") ? "正在保存首帧 Prompt…"
                            : busy.startsWith("keyframe-") ? "正在生成这个镜头的首帧…" : "";
    return <div className="space-y-5">
        <section className="studio-panel">
            <div className="flex flex-wrap items-center justify-between gap-4"><div><p className="studio-kicker">MINIMAX H3</p><h2 className="text-xl font-semibold">云端生成控制台</h2><p className="mt-1 text-sm text-white/40">最终成片默认使用官方原生 20 步质量链路；Turbo 4 步只用于低成本预览。默认只选中第 1 镜，避免误提交整批任务。</p></div><div className="flex flex-wrap gap-2"><button className="studio-secondary" disabled={!!busy} title="只检查连接、必需节点和模型，不生成内容" onClick={() => void runPreflight()}>{busy === "preflight" ? <Loader2 className="animate-spin" size={15} /> : <Server size={15} />}检查服务器</button><button className="studio-secondary" disabled={!!busy || selected.length === 0} title="调用模型设置中的分镜图服务，生成静态分镜图/首帧" onClick={() => void generateSelectedKeyframes()}>{busy === "keyframes" ? <Loader2 className="animate-spin" size={15} /> : <Sparkles size={15} />}生成所选首帧</button><button className="studio-secondary" disabled={!!busy || bundle.shots.length === 0} title="只整理参数和参考绑定，不生成图片或视频" onClick={() => void action("plan", () => planRender(bundle.project.id), "编译完成：H3 模式、帧数、参数和参考标签已保存。")}>{busy === "plan" ? <Loader2 className="animate-spin" size={15} /> : <Settings2 size={15} />}编译计划</button><button className="studio-primary" disabled={!!busy || selected.length === 0} title="将所选镜头加入 H3 视频生成队列" onClick={() => void action("render", () => enqueueRender(bundle.project.id, selected), "任务已加入本地持久队列；可在本页底部查看进度与成片。")}>{busy === "render" ? <Loader2 className="animate-spin" size={15} /> : <Play size={15} />}提交 {selected.length} 镜</button></div></div>
            <div className="mt-5 grid gap-2 text-xs sm:grid-cols-2 xl:grid-cols-4">
                <div className="rounded-lg border border-white/8 bg-white/[.02] p-3"><strong className="text-white/75">1. 检查服务器</strong><p className="mt-1 leading-5 text-white/35">验证 ComfyUI 在线、H3 节点和模型齐全；不出图、不耗 GPU 生成费。</p></div>
                <div className="rounded-lg border border-white/8 bg-white/[.02] p-3"><strong className="text-white/75">2. 生成所选首帧</strong><p className="mt-1 leading-5 text-white/35">调用“模型设置 → 分镜首帧生成”；产物既是分镜图，也是 I2V 的第一帧。</p></div>
                <div className="rounded-lg border border-white/8 bg-white/[.02] p-3"><strong className="text-white/75">3. 编译计划</strong><p className="mt-1 leading-5 text-white/35">按素材自动决定 I2V/R2V，整理 Prompt、帧数和参数；不开始生成。</p></div>
                <div className="rounded-lg border border-white/8 bg-white/[.02] p-3"><strong className="text-white/75">4. 提交镜头</strong><p className="mt-1 leading-5 text-white/35">把勾选镜头送入 H3 持久队列；这是实际的视频生成步骤，会占用云端 GPU。</p></div>
            </div>
            {(busyText || error || notice) && <div aria-live="polite" className={`mt-4 ${error ? "studio-error" : "studio-notice"}`}>{busyText && <span className="inline-flex items-center gap-2"><Loader2 className="animate-spin" size={15} />{busyText}</span>}{!busyText && (error || notice)}{error && <Link className="ml-2 underline underline-offset-2" href="/logs">查看运行日志</Link>}</div>}
        </section>
        {preflight && <div className={preflight.online ? "studio-notice" : "rounded-lg border border-amber-400/20 bg-amber-400/6 px-4 py-3 text-sm text-amber-100/75"}>{preflight.online ? `服务器在线 · 节点/模型检查：${preflight.ok ? "通过" : "有缺失"}` : <span className="inline-flex items-center gap-2"><CloudOff size={15} />服务器处于关闭状态，参数仍可在本地编辑保存。</span>} {preflight.error ? String(preflight.error) : ""}</div>}
        <section className="studio-panel">
            <div className={`flex flex-wrap items-center justify-between gap-3 ${keyframeBoardOpen ? "mb-4" : ""}`}><div><p className="studio-kicker">KEYFRAME BOARD</p><h3 className="font-semibold">分镜图 / I2V 首帧</h3><p className="mt-1 text-xs text-white/35">已生成 {generatedKeyframes}/{bundle.shots.length}。{keyframeBoardOpen ? "可全选已生成图片并打包导出；也可在卡片中单独生成或重做。" : "当前已收起，展开后可查看、选择和生成分镜图。"}</p></div><div className="flex flex-wrap items-center gap-2"><span className="studio-status">已选 {selected.length} 镜 · 可导出 {selectedKeyframes.length} 张</span><button type="button" className="studio-secondary px-3" disabled={!!busy || generatedShotIds.length === 0} onClick={toggleAllGeneratedKeyframes}><Check size={15} />{allGeneratedSelected ? "取消全选" : `全选已生成 ${generatedShotIds.length} 张`}</button><button type="button" className="studio-secondary px-3" disabled={!!busy || selectedKeyframes.length === 0} onClick={() => void exportSelectedKeyframes()}>{busy === "export-keyframes" ? <Loader2 className="animate-spin" size={15} /> : <Download size={15} />}导出选中 {selectedKeyframes.length} 张</button><Link href="/settings" className="studio-secondary px-3">分镜图模型设置</Link><button type="button" className="studio-secondary px-3" aria-expanded={keyframeBoardOpen} aria-controls="keyframe-board-content" onClick={() => setKeyframeBoardOpen((open) => !open)}>{keyframeBoardOpen ? <ChevronUp size={15} /> : <ChevronDown size={15} />}{keyframeBoardOpen ? "收起" : "展开"}</button></div></div>
            {keyframeBoardOpen && <div id="keyframe-board-content">{bundle.shots.length === 0 ? <p className="rounded-lg border border-dashed border-white/10 p-5 text-sm text-white/35">请先在“分镜设计”中生成或导入分镜。</p> : <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">{bundle.shots.map((shot) => {
                const keyframe = shot.keyframe_asset_id ? keyframeAssets.get(shot.keyframe_asset_id) : undefined;
                const imageStatus = shot.image_status === "processing"
                    ? "生成中"
                    : shot.image_status === "failed" && keyframe
                        ? "重生成失败 · 已保留上一版"
                        : keyframe
                            ? "已生成"
                            : shot.image_status === "failed"
                                ? "生成失败"
                                : "待生成";
                const imageUrl = keyframe ? projectDownloadUrl(bundle.project.id, "asset", keyframe.id) : "";
                const prompt = keyframe?.description || shot.visual_prompt;
                return <article key={`keyframe-${shot.id}`} className="overflow-hidden rounded-xl border border-white/8 bg-black/15">
                    <div className="relative flex aspect-video items-center justify-center overflow-hidden bg-black/35">
                        {keyframe ? <button type="button" className="group h-full w-full cursor-zoom-in" title="放大查看首帧原图" onClick={() => setKeyframePreview({ name: `镜头 ${shot.ordinal} 首帧`, url: imageUrl, description: prompt })}><img className="h-full w-full object-cover transition-transform duration-200 group-hover:scale-[1.02]" src={imageUrl} alt={`镜头 ${shot.ordinal} 首帧`} /><span className="absolute bottom-2 right-2 rounded-md bg-black/65 p-2 text-white/70 opacity-0 backdrop-blur transition-opacity group-hover:opacity-100"><Maximize2 size={14} /></span></button> : <div className="flex flex-col items-center gap-2 text-white/25"><ImageIcon size={26} /><span className="text-xs">{imageStatus}</span></div>}
                        <label className="absolute left-2 top-2 flex cursor-pointer items-center gap-2 rounded-lg bg-black/70 px-2.5 py-1.5 text-xs backdrop-blur"><input type="checkbox" checked={selected.includes(shot.id)} onChange={() => setSelected(toggle(selected, shot.id))} />选择</label>
                        <span className={`absolute right-2 top-2 max-w-[65%] rounded-md px-2 py-1 text-right text-[10px] ${shot.image_status === "failed" ? "bg-red-500/20 text-red-200" : "bg-black/70 text-white/65"}`}>{imageStatus}</span>
                    </div>
                    <div className="p-3">
                        <strong className="block truncate text-sm">#{shot.ordinal} {shot.title}</strong>
                        <p className="mt-1 line-clamp-2 text-[11px] leading-5 text-white/35">{prompt || "尚未填写首帧图 Prompt"}</p>
                        <div className="mt-3 grid grid-cols-2 gap-2">
                            {keyframe && <button type="button" className="studio-secondary px-3 py-2 text-xs" onClick={() => setKeyframePreview({ name: `镜头 ${shot.ordinal} 首帧`, url: imageUrl, description: prompt })}><Maximize2 size={13} />查看大图</button>}
                            <button type="button" className="studio-secondary px-3 py-2 text-xs" disabled={!!busy} onClick={() => openKeyframeEditor(shot)}><FilePenLine size={13} />Prompt 详情</button>
                            <button type="button" className={`${keyframe ? "studio-secondary" : "studio-primary"} col-span-2 px-3 py-2 text-xs`} disabled={!!busy} onClick={() => openKeyframeEditor(shot)}>{busy === `keyframe-${shot.id}` ? <Loader2 className="animate-spin" size={13} /> : <Sparkles size={13} />}{keyframe || shot.image_status === "failed" ? "输入建议并重新生成" : "编辑 Prompt 并生成本镜"}</button>
                        </div>
                    </div>
                </article>;
            })}</div>}</div>}
        </section>
        <section className="studio-panel"><div className="mb-4 flex flex-wrap items-center justify-between gap-3"><div><h3 className="font-semibold">逐镜 H3 参数与参考绑定</h3><p className="text-xs text-white/35">批量快速＝608 长边 / Turbo 4步；批量均衡＝项目分辨率 / Turbo 4步；批量清晰＝1344 长边 / 官方原生20步。按钮只保存参数，不会开始生成。</p></div><div className="flex flex-wrap gap-2"><button className="studio-secondary" title="608 长边、Turbo 4 步，适合低成本测试构图与动作" disabled={!selected.length || !!busy} onClick={() => void applyBulkPreset("fast")}>批量快速</button><button className="studio-secondary" title="使用项目设定分辨率和 Turbo 4 步参数" disabled={!selected.length || !!busy} onClick={() => void applyBulkPreset("balanced")}>批量均衡</button><button className="studio-secondary" title="至少 1344 长边、官方原生 20 步，适合最终成片" disabled={!selected.length || !!busy} onClick={() => void applyBulkPreset("quality")}>批量清晰</button><button className="studio-secondary" title="切换全部镜头的勾选状态，不修改参数" disabled={bundle.shots.length === 0} onClick={() => setSelected(selected.length === bundle.shots.length ? [] : bundle.shots.map((shot) => shot.id))}>{bundle.shots.length > 0 && selected.length === bundle.shots.length ? "取消全选" : "全选"}</button></div></div>
            <details open className="mb-4 rounded-xl border border-cyan-300/12 bg-cyan-300/[.025] p-4"><summary className="cursor-pointer text-sm font-semibold text-cyan-100/80">视频质量太差时，按画面问题这样调</summary><div className="mt-3 grid gap-2 text-xs sm:grid-cols-2 xl:grid-cols-4"><div className="rounded-lg border border-white/8 bg-black/15 p-3"><strong className="text-white/75">画面模糊、细节少</strong><p className="mt-1 leading-5 text-white/38">选择“清晰优先”：1344×768、原生20步；使用清晰完整首帧。Turbo 4步只用于预览。</p></div><div className="rounded-lg border border-white/8 bg-black/15 p-3"><strong className="text-white/75">人物脸或服装漂移</strong><p className="mt-1 leading-5 text-white/38">优先 I2V 并绑定高质量首帧；R2V 只保留本镜出现的人物参考。多人镜头先合成完整首帧。</p></div><div className="rounded-lg border border-white/8 bg-black/15 p-3"><strong className="text-white/75">动作崩、拖影、重影</strong><p className="mt-1 leading-5 text-white/38">每镜只保留一个主动作，复杂镜头拆成 5–7 秒；固定机位和轻微动作更利于多人对白稳定。</p></div><div className="rounded-lg border border-white/8 bg-black/15 p-3"><strong className="text-white/75">生成太慢</strong><p className="mt-1 leading-5 text-white/38">先用 608×352 / Turbo4步预览；确认 Seed、Prompt、构图后，再切清晰优先生成最终版。</p></div></div></details>
            {bundle.shots.length === 0 ? <p className="rounded-lg border border-dashed border-white/10 p-5 text-sm text-white/35">请先在“分镜设计”中生成或导入分镜。</p> : <div className="space-y-3">{bundle.shots.map((shot) => <H3ShotCard key={`${shot.id}-${shot.updated_at}`} shot={shot} project={bundle.project} assets={bundle.assets} checked={selected.includes(shot.id)} busy={busy === `h3-${shot.id}`} toggleChecked={() => setSelected(toggle(selected, shot.id))} save={saveShot} />)}</div>}</section>
        <section className="studio-panel"><div className="mb-4 flex items-center justify-between"><div><h3 className="font-semibold">渲染任务</h3><p className="text-xs text-white/35">活动 {active.length} · 已记录估算成本 ¥{cost.toFixed(2)}</p></div><button className="studio-secondary" onClick={() => void refresh()}><RefreshCw size={14} />刷新</button></div>{bundle.jobs.length === 0 ? <p className="text-sm text-white/35">尚未提交任务。</p> : <div className="space-y-2">{bundle.jobs.map((job) => <JobRow key={job.id} job={job} projectId={bundle.project.id} cancel={() => action(`cancel-${job.id}`, () => cancelRenderJob(bundle.project.id, job.id), "已发送取消请求")} />)}</div>}</section>
        {keyframeEditor && editorShot && <div className="studio-modal" role="dialog" aria-modal="true" aria-labelledby="keyframe-editor-title" onMouseDown={() => !busy && setKeyframeEditor(null)}>
            <section className="studio-dialog max-w-5xl" onMouseDown={(event) => event.stopPropagation()}>
                <div className="mb-5 flex items-start justify-between gap-4">
                    <div><p className="studio-kicker">KEYFRAME REVISION</p><h2 id="keyframe-editor-title" className="text-2xl font-semibold">镜头 {editorShot.ordinal} · 首帧 Prompt 与重新生成</h2><p className="mt-1 text-sm leading-6 text-white/45">可先修改完整 Prompt，再补充本次修改建议，并决定是否把当前首帧作为视觉参考。</p></div>
                    <button type="button" className="rounded-lg p-2 text-white/35 hover:bg-white/8 hover:text-white" disabled={!!busy} aria-label="关闭" onClick={() => setKeyframeEditor(null)}><X size={20} /></button>
                </div>
                <div className="grid gap-5 lg:grid-cols-[minmax(260px,.72fr)_minmax(0,1.28fr)]">
                    <div>
                        <div className="flex aspect-video items-center justify-center overflow-hidden rounded-xl border border-white/10 bg-black/35">
                            {editorKeyframe ? <img className="h-full w-full object-contain" src={projectDownloadUrl(bundle.project.id, "asset", editorKeyframe.id)} alt={`镜头 ${editorShot.ordinal} 当前首帧`} /> : <div className="flex flex-col items-center gap-2 text-white/25"><ImageIcon size={30} /><span className="text-xs">当前还没有首帧图</span></div>}
                        </div>
                        {editorKeyframe && <button type="button" className="studio-secondary mt-3 w-full" onClick={() => setKeyframePreview({ name: `镜头 ${editorShot.ordinal} 当前首帧`, url: projectDownloadUrl(bundle.project.id, "asset", editorKeyframe.id), description: editorKeyframe.description || keyframeEditor.prompt })}><Maximize2 size={14} />查看当前原图</button>}
                        <div className="mt-4 rounded-lg border border-cyan-300/12 bg-cyan-300/[.03] p-3 text-xs leading-5 text-cyan-50/50">人物设定、统一风格和已绑定的角色/场景参考图仍会自动附加到 Prompt；下面的模式只控制是否额外参考当前生成图。</div>
                    </div>
                    <div className="space-y-4">
                        <label><span className="studio-label">首帧图详细 Prompt（可直接修改）</span><textarea className="studio-input min-h-44 resize-y" maxLength={12000} value={keyframeEditor.prompt} onChange={(event) => setKeyframeEditor((current) => current ? { ...current, prompt: event.target.value } : current)} placeholder="描述人物、动作瞬间、环境、构图、机位、光线、色彩和材质…" /></label>
                        <div className="flex justify-end text-[11px] text-white/25">{keyframeEditor.prompt.length}/12000</div>
                        <label><span className="studio-label">本次修改建议（可选）</span><textarea className="studio-input min-h-28 resize-y" maxLength={4000} value={keyframeEditor.suggestions} onChange={(event) => setKeyframeEditor((current) => current ? { ...current, suggestions: event.target.value } : current)} placeholder="例如：保留人物和构图，把女主表情改得更克制；门口增加逆光；去掉右侧多余人物……" /></label>
                        <div className="flex justify-end text-[11px] text-white/25">{keyframeEditor.suggestions.length}/4000</div>
                        <div>
                            <span className="studio-label">重新生成方式</span>
                            <div className="grid gap-2 sm:grid-cols-2">
                                <button type="button" disabled={!editorKeyframe} aria-pressed={keyframeEditor.revisionMode === "iterate"} className={`rounded-xl border p-4 text-left ${keyframeEditor.revisionMode === "iterate" ? "border-cyan-300/45 bg-cyan-300/8" : "border-white/10 bg-black/20 hover:border-white/20"}`} onClick={() => setKeyframeEditor((current) => current ? { ...current, revisionMode: "iterate" } : current)}><strong className="block text-sm text-white/80">基于当前图修改</strong><span className="mt-1 block text-xs leading-5 text-white/38">将当前首帧作为第一参考图，更适合保留人物、构图和整体风格后做局部调整。</span></button>
                                <button type="button" aria-pressed={keyframeEditor.revisionMode === "fresh"} className={`rounded-xl border p-4 text-left ${keyframeEditor.revisionMode === "fresh" ? "border-cyan-300/45 bg-cyan-300/8" : "border-white/10 bg-black/20 hover:border-white/20"}`} onClick={() => setKeyframeEditor((current) => current ? { ...current, revisionMode: "fresh" } : current)}><strong className="block text-sm text-white/80">不参考当前图，全新生成</strong><span className="mt-1 block text-xs leading-5 text-white/38">只按 Prompt、修改建议和项目参考素材重做，适合彻底更换构图或当前图偏差很大时使用。</span></button>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-white/8 pt-5">
                    <p className="text-xs text-white/30">生成成功后会保留旧图文件，并把新图自动绑定为本镜 I2V 首帧。</p>
                    <div className="flex flex-wrap gap-2"><button type="button" className="studio-secondary" disabled={!!busy || !keyframeEditor.prompt.trim()} onClick={() => void saveKeyframePrompt()}>{busy === `keyframe-prompt-${editorShot.id}` ? <Loader2 className="animate-spin" size={15} /> : <Save size={15} />}仅保存 Prompt</button><button type="button" className="studio-primary" disabled={!!busy || !keyframeEditor.prompt.trim()} onClick={() => void regenerateKeyframe()}>{busy === `keyframe-${editorShot.id}` ? <Loader2 className="animate-spin" size={15} /> : <Sparkles size={15} />}{editorKeyframe ? "按以上设置重新生成" : "按以上设置生成首帧"}</button></div>
                </div>
            </section>
        </div>}
        {keyframePreview && <MediaPreview preview={keyframePreview} onClose={() => setKeyframePreview(null)} />}
    </div>;
}

function ReviewPanel({ bundle, busy, action }: { bundle: ProjectBundle; busy: string; action: Action }) {
    const [comment, setComment] = useState("");
    const publicBase = (process.env.NEXT_PUBLIC_APP_URL || (typeof window === "undefined" ? "" : window.location.origin)).replace(/\/$/, "");
    const reviewUrl = `${publicBase}/review/${bundle.project.review_token}`;
    return <div className="grid gap-5 xl:grid-cols-[.65fr_1.35fr]"><section className="studio-panel h-fit"><p className="studio-kicker">CLIENT REVIEW</p><h2 className="mb-4 text-xl font-semibold">客户确认链接</h2><p className="mb-4 text-sm leading-6 text-white/45">客户打开链接即可查看分镜、参考图和已生成片段，并提交“确认”或“需要修改”。不需要登录。</p><div className="flex gap-2"><input className="studio-input min-w-0 flex-1" readOnly value={reviewUrl} /><button className="studio-secondary px-3" onClick={() => void navigator.clipboard.writeText(reviewUrl)}><Clipboard size={15} /></button><a className="studio-secondary px-3" href={reviewUrl} target="_blank"><ExternalLink size={15} /></a></div><textarea className="studio-input mt-4" rows={4} value={comment} onChange={(event) => setComment(event.target.value)} placeholder="内部备注或确认说明…" /><div className="mt-3 flex gap-2"><button className="studio-primary" disabled={!!busy} onClick={() => void action("approve", () => approveStoryboard(bundle.project.id, true, comment, "制作方"), "分镜版本已锁定为确认状态")}><Check size={15} />确认分镜</button><button className="studio-secondary" disabled={!!busy} onClick={() => void action("changes", () => approveStoryboard(bundle.project.id, false, comment, "制作方"), "已标记为需要修改")}>退回修改</button></div></section><section className="studio-panel"><p className="studio-kicker">REVIEW LOG</p><h2 className="mb-4 text-xl font-semibold">反馈记录</h2>{bundle.reviews.length === 0 ? <p className="text-sm text-white/35">暂无客户反馈。</p> : <div className="space-y-3">{bundle.reviews.map((review) => <div key={review.id} className="rounded-lg border border-white/8 bg-black/15 p-4"><div className="mb-2 flex justify-between"><strong className={review.decision === "approved" ? "text-emerald-300" : "text-amber-300"}>{review.decision === "approved" ? "已确认" : "需要修改"}</strong><span className="text-xs text-white/30">{new Date(review.created_at).toLocaleString("zh-CN")}</span></div><p className="text-sm text-white/65">{review.comment || "未填写文字说明"}</p><p className="mt-2 text-xs text-white/30">{review.reviewer || "匿名客户"} · {review.target_type}</p></div>)}</div>}</section></div>;
}

function DeliveryPanel({ bundle, busy, action }: { bundle: ProjectBundle; busy: string; action: Action }) {
    const [subtitle, setSubtitle] = useState<string | null>(bundle.assets.find((asset) => asset.type === "subtitle")?.id || null);
    const [music, setMusic] = useState<string | null>(bundle.assets.find((asset) => asset.role === "music")?.id || null);
    const [crossfade, setCrossfade] = useState(0);
    const [makePreview, setMakePreview] = useState(true);
    const completed = bundle.shots.filter((shot) => shot.video_status === "completed" && !!shot.video_path).length;
    return <div className="grid gap-5 xl:grid-cols-[420px_1fr]"><section className="studio-panel h-fit"><p className="studio-kicker">AUTOMATIC EDIT</p><h2 className="mb-2 text-xl font-semibold">剪辑与成片</h2><p className="mb-5 text-sm leading-6 text-white/40">按分镜顺序统一分辨率、帧率与音轨，裁剪到设计时长，合并视频并执行 QC。已完成片段 {completed}/{bundle.shots.length}。</p><button className="studio-secondary mb-5 w-full" disabled={!!busy} onClick={() => void action("subtitle", () => generateSubtitles(bundle.project.id), "已根据分镜对白生成 SRT 字幕")}>{busy === "subtitle" ? <Loader2 className="animate-spin" size={15} /> : <FileText size={15} />}自动生成字幕</button><Field label="字幕"><select value={subtitle || ""} onChange={(event) => setSubtitle(event.target.value || null)}><option value="">不烧录字幕</option>{bundle.assets.filter((asset) => asset.type === "subtitle").map((asset) => <option key={asset.id} value={asset.id}>{asset.name}</option>)}</select></Field><Field label="背景音乐"><select value={music || ""} onChange={(event) => setMusic(event.target.value || null)}><option value="">不混入 BGM</option>{bundle.assets.filter((asset) => asset.type === "audio").map((asset) => <option key={asset.id} value={asset.id}>{asset.name}</option>)}</select></Field><Field label="镜头交叉淡化（秒）"><input type="number" min={0} max={2} step={.1} value={crossfade} onChange={(event) => setCrossfade(Number(event.target.value))} /></Field><label className="mt-3 flex items-center gap-2 text-xs text-white/50"><input type="checkbox" checked={makePreview} onChange={(event) => setMakePreview(event.target.checked)} />同时生成小体积客户预览版</label><button className="studio-primary mt-4 w-full" disabled={!!busy || completed < bundle.shots.length} onClick={() => void action("finalize", () => finalizeProject(bundle.project.id, { output_name: "final_video.mp4", crossfade_seconds: crossfade, burn_subtitles: !!subtitle, subtitle_asset_id: subtitle, background_music_asset_id: music, normalize_audio: true, preview: makePreview }), "成片剪辑、编码与 QC 已完成")}>{busy === "finalize" ? <Loader2 className="animate-spin" size={16} /> : <PackageCheck size={16} />}生成最终成片</button></section><section className="studio-panel"><p className="studio-kicker">DELIVERIES</p><h2 className="mb-5 text-xl font-semibold">交付版本</h2>{bundle.deliveries.length === 0 ? <div className="studio-empty"><Film className="text-cyan-300" /><strong>暂无成片</strong><span>所有分镜视频完成后即可自动剪辑</span></div> : <div className="space-y-4">{bundle.deliveries.map((delivery) => <DeliveryCard key={delivery.id} delivery={delivery} projectId={bundle.project.id} />)}</div>}</section></div>;
}

function AssetCard({ asset, projectId, preview, remove }: { asset: Asset; projectId: string; preview: () => void; remove: () => Promise<void> }) {
    const url = projectDownloadUrl(projectId, "asset", asset.id);
    return <div className="group overflow-hidden rounded-lg border border-white/8 bg-black/20"><div className="flex aspect-video items-center justify-center bg-black/40">{asset.type === "image" ? <button type="button" className="relative h-full w-full cursor-zoom-in overflow-hidden" title="放大查看原图" onClick={preview}><img className="h-full w-full object-cover transition-transform duration-200 group-hover:scale-[1.02]" src={url} alt={asset.name} /><span className="absolute bottom-2 right-2 inline-flex items-center gap-1.5 rounded-md bg-black/70 px-2.5 py-1.5 text-xs text-white/75 opacity-0 backdrop-blur transition-opacity group-hover:opacity-100"><Maximize2 size={13} />查看原图</span></button> : asset.type === "video" ? <video className="h-full w-full object-cover" src={url} controls /> : asset.type === "audio" ? <audio className="w-[90%]" src={url} controls /> : <FileText className="text-white/25" />}</div><div className="flex items-start gap-2 p-3"><div className="min-w-0 flex-1"><strong className="block truncate text-sm">{asset.name}</strong><p className="text-[11px] uppercase text-white/30">{asset.type} · {asset.role} · {(asset.size_bytes / 1024 / 1024).toFixed(1)}MB</p></div>{asset.type === "image" && <button type="button" className="rounded p-1.5 text-white/30 hover:bg-white/8 hover:text-cyan-200" title="查看原图" onClick={preview}><Maximize2 size={14} /></button>}<button type="button" className="rounded p-1.5 text-white/20 opacity-0 hover:text-red-300 group-hover:opacity-100" title="删除素材" onClick={() => void remove()}><Trash2 size={14} /></button></div></div>;
}

function MediaPreview({ preview, onClose }: { preview: MediaPreviewState; onClose: () => void }) {
    useEffect(() => {
        const closeOnEscape = (event: KeyboardEvent) => event.key === "Escape" && onClose();
        window.addEventListener("keydown", closeOnEscape);
        return () => window.removeEventListener("keydown", closeOnEscape);
    }, [onClose]);
    return <div className="studio-modal z-[120]" role="dialog" aria-modal="true" aria-label={`${preview.name} 原图预览`} onMouseDown={onClose}>
        <section className="flex max-h-[94vh] w-full max-w-7xl flex-col overflow-hidden rounded-2xl border border-white/12 bg-[#0d1017] shadow-2xl" onMouseDown={(event) => event.stopPropagation()}>
            <div className="flex items-center gap-3 border-b border-white/8 px-4 py-3 md:px-5"><div className="min-w-0 flex-1"><strong className="block truncate text-sm">{preview.name}</strong><span className="text-xs text-white/30">原图预览 · 点击右侧按钮可在新窗口打开原始文件</span></div><a className="studio-secondary px-3 py-2 text-xs" href={preview.url} target="_blank" rel="noreferrer"><ExternalLink size={14} />打开原图</a><button type="button" className="rounded-lg p-2 text-white/40 hover:bg-white/8 hover:text-white" aria-label="关闭原图预览" onClick={onClose}><X size={20} /></button></div>
            <div className="min-h-0 flex-1 overflow-auto bg-black/50 p-3 md:p-5"><img className="mx-auto max-h-[76vh] max-w-full object-contain" src={preview.url} alt={preview.name} /></div>
            {preview.description && <div className="max-h-36 overflow-y-auto border-t border-white/8 px-5 py-3"><p className="studio-kicker">IMAGE PROMPT</p><p className="whitespace-pre-wrap text-xs leading-5 text-white/45">{preview.description}</p></div>}
        </section>
    </div>;
}

function JobRow({ job, projectId, cancel }: { job: RenderJob; projectId: string; cancel: () => Promise<void> }) {
    const color = job.status === "completed" ? "text-emerald-300" : job.status === "failed" ? "text-red-300" : ACTIVE_JOBS.has(job.status) ? "text-cyan-200" : "text-white/40";
    const params = (job.input_snapshot?.h3_parameters || {}) as Record<string, unknown>;
    const parameterText = params.width ? `${params.width}×${params.height} · ${params.frames}帧 · ${params.steps}步 · ${params.scheduler}` : "历史任务默认参数";
    return <div className="rounded-lg border border-white/8 bg-black/15 p-3"><div className="flex items-center gap-3"><span className={`w-24 text-xs font-semibold uppercase ${color}`}>{job.status}</span><div className="h-1.5 flex-1 overflow-hidden rounded bg-white/8"><div className="h-full bg-cyan-300" style={{ width: `${Math.round(job.progress * 100)}%` }} /></div><span className="w-10 text-right font-mono text-xs text-white/35">{Math.round(job.progress * 100)}%</span>{job.status === "completed" && <a className="rounded p-1.5 text-cyan-200 hover:bg-cyan-300/10" href={projectDownloadUrl(projectId, "job", job.id)}><Download size={15} /></a>}{ACTIVE_JOBS.has(job.status) && <button className="rounded p-1.5 text-white/30 hover:text-red-300" onClick={() => void cancel()}><Trash2 size={14} /></button>}</div>{job.error && <p className="mt-2 text-xs text-red-300/80">{job.error}</p>}<p className="mt-1 text-[11px] text-white/25">{job.mode?.toUpperCase() || "H3"} · {parameterText} · seed {job.seed} · {job.elapsed_seconds ? `${job.elapsed_seconds.toFixed(0)}s` : "等待计时"}</p></div>;
}

function DeliveryCard({ delivery, projectId }: { delivery: Delivery; projectId: string }) {
    const url = projectDownloadUrl(projectId, "delivery", delivery.id);
    return <div className="rounded-xl border border-white/8 bg-black/20 p-4"><video className="mb-4 aspect-video w-full rounded-lg bg-black" src={delivery.preview_path ? projectDownloadUrl(projectId, "preview", delivery.id) : url} controls /><div className="flex items-center justify-between"><div><strong>最终成片</strong><p className="text-xs text-white/35">{delivery.duration_seconds.toFixed(1)} 秒 · QC {delivery.qc_report?.passed === false ? "有警告" : "通过"}</p></div><div className="flex gap-2">{delivery.preview_path && <a className="studio-secondary" href={projectDownloadUrl(projectId, "preview", delivery.id)}><Download size={15} />预览版</a>}<a className="studio-primary" href={url}><Download size={15} />原片</a></div></div></div>;
}

function Field({ label, hint, wide, children }: { label: string; hint?: string; wide?: boolean; children: React.ReactNode }) { return <label className={wide ? "md:col-span-2" : ""}><span className="studio-label">{label}</span><div className="studio-field">{children}</div>{hint && <span className="mt-1.5 block text-[11px] leading-5 text-white/32">{hint}</span>}</label>; }
function toggle(values: string[], value: string) { return values.includes(value) ? values.filter((item) => item !== value) : [...values, value]; }
type Action = (key: string, task: () => Promise<unknown>, success: string, reload?: boolean) => Promise<void>;
