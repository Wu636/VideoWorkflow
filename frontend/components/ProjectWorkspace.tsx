"use client";

/* eslint-disable @next/next/no-img-element */
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import {
    ArrowLeft,
    BookCopy,
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
    Scissors,
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
    applySceneProfiles,
    analyzeCharacterReferences,
    analyzeProjectBrief,
    analyzeProjectStyle,
    cancelRenderJob,
    comfyPreflight,
    confirmShotSplit,
    deleteProjectAsset,
    deleteRenderJob,
    deleteShot,
    enqueueRender,
    exportKeyframes,
    finalizeProject,
    generateH3Prompts,
    generateCharacterReferencesWithOptions,
    generateMissingCharacterReferences,
    generateSceneProfiles,
    generateSeedancePrompts,
    generateKeyframes,
    generateStoryboard,
    getKeyframeMaterials,
    getSeedanceMaterials,
    importStoryboard,
    insertShotWithAi,
    generateSubtitles,
    getProject,
    getH3PromptSkills,
    getSeedanceCatalog,
    estimateSeedance,
    getRenderJobs,
    planRender,
    previewShotSplit,
    projectDownloadUrl,
    projectInlineUrl,
    reorderStoryboard,
    reviseShotWithAi,
    rewriteProjectScript,
    seedancePreflight,
    selectRenderJob,
    saveProjectAsSeries,
    storyboardCsvUrl,
    updateProject,
    updateKeyframePrompt,
    updateShot,
    uploadKeyframe,
    uploadProjectAsset,
    uploadProjectScript,
} from "@/lib/api";
import { isH3PromptBlocked, isKeyframeBusy } from "@/lib/production-busy";
import SceneProfileCard from "@/components/SceneProfileCard";
import type { Asset, AssetRole, CharacterProfile, Delivery, H3PromptSkill, KeyframeMaterialDiagnostics, Project, ProjectAnalysisDraft, ProjectBundle, RenderJob, ScriptRewriteDraft, SeedanceCatalog, SeedanceEstimate, SeedanceMaterialDiagnostics, Shot, ShotSplitPreview, StyleAnalysisDraft } from "@/types";

type Tab = "brief" | "storyboard" | "assets" | "production" | "review" | "delivery";

const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: "brief", label: "需求与角色", icon: <FileText size={16} /> },
    { id: "storyboard", label: "分镜设计", icon: <LayoutList size={16} /> },
    { id: "assets", label: "参考素材", icon: <ImageIcon size={16} /> },
    { id: "production", label: "视频生成", icon: <Film size={16} /> },
    { id: "review", label: "客户确认", icon: <Users size={16} /> },
    { id: "delivery", label: "剪辑交付", icon: <PackageCheck size={16} /> },
];

const ACTIVE_JOBS = new Set(["queued", "submitting", "running", "cancel_requested"]);

function selectedSceneProfileIds(shot: Shot): string[] {
    return Array.from(new Set([
        ...(shot.scene_profile_ids || []),
        ...(shot.scene_profile_id ? [shot.scene_profile_id] : []),
    ])).slice(0, 2);
}

function sceneProfileSelectionPatch(
    shot: Shot,
    position: "start" | "destination",
    profileId: string,
): Pick<Shot, "use_scene_profile" | "scene_profile_id" | "scene_profile_ids"> {
    const current = selectedSceneProfileIds(shot);
    if (position === "start") {
        if (!profileId) return { use_scene_profile: false, scene_profile_id: null, scene_profile_ids: [] };
        const destination = current[1] && current[1] !== profileId ? current[1] : null;
        const ids = [profileId, ...(destination ? [destination] : [])];
        return { use_scene_profile: true, scene_profile_id: profileId, scene_profile_ids: ids };
    }
    const start = current[0];
    if (!start) return { use_scene_profile: false, scene_profile_id: null, scene_profile_ids: [] };
    const ids = profileId && profileId !== start ? [start, profileId] : [start];
    return { use_scene_profile: true, scene_profile_id: start, scene_profile_ids: ids };
}

function applyAnalysisConfirmationDefaults(draft: ProjectAnalysisDraft): ProjectAnalysisDraft {
    const characters = [...draft.characters];
    const knownNames = new Set(characters.map((character) => character.name.trim().toLocaleLowerCase()));
    let changed = false;
    const analysisNotes = draft.analysis_notes.map((note) => {
        const match = note.match(/([^，。；：]{1,36}?)(?:是否需要|是否要|需不需要)(?:单独|独立)(?:设定|设计)(?:形象|角色)?/);
        if (!match) return note;
        let name = match[1].trim();
        for (const prefix of ["需确认", "确认", "剧中出现的", "剧中", "画面中的", "出现的", "中的"]) {
            if (name.includes(prefix)) name = name.split(prefix).at(-1)?.trim() || name;
        }
        name = name.replace(/(?:角色|形象)$/, "").trim();
        if (!name || name.length > 20) return note;
        if (!knownNames.has(name.toLocaleLowerCase())) {
            characters.push({
                character_id: null,
                name,
                description: `根据剧本为${name}建立独立、稳定、可跨镜复用的固定形象；可在当前角色卡直接补充识别特征。`,
                wardrobe: "无服装；如有项圈、挂件或其他固定配饰，可在当前角色卡补充。",
                voice_description: "无台词；如有叫声或拟人台词，可在当前角色卡补充。",
                reference_observations: `由确认项自动建档：${note}`,
            });
            knownNames.add(name.toLocaleLowerCase());
        }
        changed = true;
        return `已默认将${name}作为独立固定形象建档；如不需要，可在上方角色草稿中删除。`;
    });
    return changed ? { ...draft, characters, analysis_notes: analysisNotes } : draft;
}

export default function ProjectWorkspace({ projectId }: { projectId: string }) {
    const [bundle, setBundle] = useState<ProjectBundle | null>(null);
    const [tab, setTab] = useState<Tab>("brief");
    const [loading, setLoading] = useState(true);
    const [busy, setBusy] = useState<Set<string>>(() => new Set());
    const [error, setError] = useState("");
    const [notice, setNotice] = useState("");
    const refreshSequence = useRef(0);

    const refresh = useCallback(async () => {
        const sequence = ++refreshSequence.current;
        try {
            const next = await getProject(projectId);
            if (sequence === refreshSequence.current) {
                setBundle(next);
                setError("");
            }
        } catch (caught) {
            if (sequence === refreshSequence.current) {
                setError(caught instanceof Error ? caught.message : String(caught));
            }
        } finally {
            if (sequence === refreshSequence.current) setLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        void refresh();
    }, [refresh]);

    const jobsActive = bundle?.jobs.some((job) => ACTIVE_JOBS.has(job.status)) || false;
    const sceneReferencesActive = Array.from(busy).some((key) => key.startsWith("scene-ref-")) || bundle?.project.scene_profiles.some((profile) => ["generating", "downloading"].includes(profile.reference_status)) || false;
    useEffect(() => {
        if (!sceneReferencesActive) return;
        let cancelled = false;
        let inFlight = false;
        const timer = window.setInterval(async () => {
            if (inFlight) return;
            inFlight = true;
            const next = await getProject(projectId).catch(() => null);
            if (next && !cancelled) setBundle((current) => current ? {
                ...current, assets: next.assets,
                project: { ...current.project, scene_profiles: next.project.scene_profiles },
            } : current);
            inFlight = false;
        }, 2500);
        return () => { cancelled = true; window.clearInterval(timer); };
    }, [sceneReferencesActive, projectId]);
    useEffect(() => {
        if (!jobsActive) return;
        const timer = window.setInterval(async () => {
            const jobs = await getRenderJobs(projectId).catch(() => null);
            if (!jobs) return;
            setBundle((current) => current ? { ...current, jobs } : current);
            if (!jobs.some((job) => ACTIVE_JOBS.has(job.status))) void refresh();
        }, 2500);
        return () => window.clearInterval(timer);
    }, [jobsActive, projectId, refresh]);

    const action = async (key: string, task: () => Promise<unknown>, success: string, reload = true) => {
        setBusy((current) => new Set(current).add(key));
        setError("");
        setNotice("");
        try {
            await task();
            if (reload) await refresh();
            setNotice(success);
        } catch (caught) {
            if (reload) await refresh();
            setError(caught instanceof Error ? caught.message : String(caught));
        } finally {
            setBusy((current) => {
                const next = new Set(current);
                next.delete(key);
                return next;
            });
        }
    };
    const busyFor = (...prefixes: string[]) => (
        Array.from(busy).find((key) => prefixes.some((prefix) => key === prefix || key.startsWith(`${prefix}-`))) || ""
    );

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
                    {busy.size > 0 && <span className="hidden items-center gap-2 rounded-lg border border-cyan-300/15 bg-cyan-300/[.04] px-3 py-2 text-xs text-cyan-100/65 md:inline-flex"><Loader2 className="animate-spin" size={14} />后台并行处理中 {busy.size} 项</span>}
                    <Link href="/settings" className="studio-secondary px-3" title="模型设置"><Settings2 size={15} /></Link><Link href="/logs" className="studio-secondary px-3" title="运行日志"><FileClock size={15} /></Link><button className="studio-secondary" onClick={() => void refresh()}><RefreshCw size={15} /> <span className="hidden md:inline">刷新</span></button>
                </div>
                <nav className="mx-auto flex max-w-[1700px] overflow-x-auto px-4 md:px-8">
                    {TABS.map((item) => <button key={item.id} className={`studio-tab whitespace-nowrap ${tab === item.id ? "studio-tab-active" : ""}`} onClick={() => setTab(item.id)}>{item.icon}{item.label}</button>)}
                </nav>
            </header>

            <div className="mx-auto max-w-[1700px] px-4 py-6 md:px-8">
                {tab !== "production" && error && <div className="studio-error mb-4">{error}</div>}
                {tab !== "production" && notice && <div className="studio-notice mb-4">{notice}</div>}
                {tab === "brief" && <BriefPanel project={project} assets={bundle.assets} busy={busy} action={action} setLocal={(next) => setBundle({ ...bundle, project: next })} save={(next) => action("save-project", () => updateProject(next), "项目需求与角色设定已保存")} />}
                {tab === "storyboard" && <StoryboardPanel bundle={bundle} busy={busyFor("storyboard", "storyboard-import", "ai-revise", "reorder", "blank-shot", "shot", "delete")} action={action} refresh={refresh} updateLocal={(shots) => setBundle({ ...bundle, shots })} />}
                {tab === "assets" && <AssetsPanel project={project} assets={bundle.assets} shots={bundle.shots} busy={busy} action={action} />}
                {tab === "production" && <ProductionPanel bundle={bundle} busyTasks={busy} busy={busyFor("preflight", "keyframes", "h3-prompts", "seedance-prompts", "export-keyframes", "plan", "render", "h3", "seedance", "preset", "upload-keyframe", "keyframe-prompt", "cancel", "delete")} error={error} notice={notice} action={action} refresh={refresh} notify={(message) => { setError(""); setNotice(message); }} fail={(message) => { setNotice(""); setError(message); }} />}
                {tab === "review" && <ReviewPanel bundle={bundle} busy={busyFor("approve", "changes")} action={action} />}
                {tab === "delivery" && <DeliveryPanel bundle={bundle} busy={busyFor("subtitle", "finalize")} action={action} />}
            </div>
        </main>
    );
}

function BriefPanel({ project, assets, busy, action, setLocal, save }: { project: Project; assets: Asset[]; busy: BusyState; action: Action; setLocal: (project: Project) => void; save: (project: Project) => Promise<void> }) {
    const [analysis, setAnalysis] = useState<ProjectAnalysisDraft | null>(null);
    const [styleDraft, setStyleDraft] = useState<StyleAnalysisDraft | null>(null);
    const [styleReferenceId, setStyleReferenceId] = useState("");
    const [rewriteOpen, setRewriteOpen] = useState(false);
    const [rewriteMode, setRewriteMode] = useState<"auto" | "expand" | "shorten">("auto");
    const [rewriteSuggestions, setRewriteSuggestions] = useState("");
    const [rewriteDraft, setRewriteDraft] = useState<ScriptRewriteDraft | null>(null);
    const [analysisBusy, setAnalysisBusy] = useState("");
    const [analysisError, setAnalysisError] = useState("");
    const [analysisNotice, setAnalysisNotice] = useState("");
    const [seriesOpen, setSeriesOpen] = useState(false);
    const [seriesName, setSeriesName] = useState(project.brief.title);
    const [seriesDescription, setSeriesDescription] = useState("");
    const [selected, setSelected] = useState<Record<string, boolean>>({ visual_style: !project.series_id, pacing: true, audience: true, style_bible: !project.series_id, negative_prompt: !project.series_id, delivery_notes: true, shot_count: true, characters: true });
    useEffect(() => {
        if (!analysis) return;
        const normalized = applyAnalysisConfirmationDefaults(analysis);
        if (normalized !== analysis) setAnalysis(normalized);
    }, [analysis]);
    const changeBrief = (key: keyof Project["brief"], value: string | number) => setLocal({ ...project, brief: { ...project.brief, [key]: value } });
    const changeAspect = (aspectRatio: string) => {
        const [width, height] = aspectRatio === "9:16" ? [768, 1344] : aspectRatio === "1:1" ? [1024, 1024] : [1344, 768];
        setLocal({ ...project, brief: { ...project.brief, aspect_ratio: aspectRatio, width, height } });
    };
    const updateCharacter = (id: string, key: keyof CharacterProfile, value: string | string[]) => setLocal({ ...project, characters: project.characters.map((item) => item.id === id ? { ...item, [key]: value } : item) });
    const addCharacter = () => setLocal({ ...project, characters: [...project.characters, { id: `character_${crypto.randomUUID().replaceAll("-", "")}`, name: "新角色", description: "", wardrobe: "", voice_description: "", tts_voice: "", reference_asset_ids: [], appearance_profiles: [] }] });
    const updateAppearance = (characterId: string, appearanceId: string, patch: Partial<CharacterProfile["appearance_profiles"][number]>) => setLocal({
        ...project,
        characters: project.characters.map((character) => character.id === characterId ? { ...character, appearance_profiles: character.appearance_profiles.map((appearance) => appearance.id === appearanceId ? { ...appearance, ...patch } : appearance) } : character),
    });
    const addAppearance = (characterId: string) => setLocal({
        ...project,
        characters: project.characters.map((character) => character.id === characterId ? { ...character, appearance_profiles: [...character.appearance_profiles, { id: `appearance_${crypto.randomUUID().replaceAll("-", "")}`, label: "新时期形象", time_context: "", description: "", wardrobe: "", reference_asset_ids: [], approved: false }] } : character),
    });
    const styleReferenceAssets = assets.filter((asset) => asset.role === "style" && ["image", "video"].includes(asset.type));
    const activeStyleReferenceId = styleReferenceAssets.some((asset) => asset.id === styleReferenceId) ? styleReferenceId : styleReferenceAssets[0]?.id || "";
    const activeStyleReference = styleReferenceAssets.find((asset) => asset.id === activeStyleReferenceId);
    const analyzeAndApplyStyle = async (assetIds: string[], uploadedFile?: File) => {
        if (!assetIds.length && !uploadedFile) return;
        await action("brief-style-analyze", async () => {
            await updateProject(project);
            let referenceIds = assetIds;
            if (uploadedFile) {
                const asset = await uploadProjectAsset(project.id, uploadedFile, "style");
                referenceIds = [asset.id];
                setStyleReferenceId(asset.id);
            }
            const draft = await analyzeProjectStyle(project.id, referenceIds, true);
            setStyleDraft(draft);
        }, "画风分析已完成，并已自动回填视觉风格、统一风格圣经和负面约束");
    };
    const backfillCharacter = async (character: CharacterProfile, appearanceId?: string) => {
        const appearance = character.appearance_profiles.find((item) => item.id === appearanceId);
        const referenceIds = appearance ? appearance.reference_asset_ids : character.reference_asset_ids;
        if (!referenceIds.length) return;
        const key = `character-analyze-${character.id}-${appearanceId || "base"}`;
        if (busy.has(key)) return;
        await action(key, async () => {
            await updateProject(project);
            await analyzeCharacterReferences(project.id, character.id, referenceIds, { appearanceProfileId: appearanceId || null, appearanceLabel: appearance?.label || "" });
        }, `已根据参考图回填“${character.name}”${appearance ? `的“${appearance.label}”时期` : "基础设定"}。`);
    };
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
    const syncSeries = async (createNew = false) => {
        await action("series-sync", async () => {
            await updateProject(project);
            const result = await saveProjectAsSeries(
                project.id,
                createNew ? seriesName.trim() : "",
                createNew ? seriesDescription.trim() : "",
            );
            setLocal(result.project);
            setSeriesOpen(false);
        }, createNew ? "系列资料已建立，首页现在可以直接创建续集。" : "系列资料已更新，之后创建的续集会使用当前画风和角色资料。");
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
                const inheritedFromSeries = current.reference_asset_ids.some((assetId) => assets.some((asset) => asset.id === assetId && asset.tags.some((tag) => tag.startsWith("series:"))));
                return draft && !inheritedFromSeries ? { ...current, name: draft.name, description: draft.description, wardrobe: draft.wardrobe, voice_description: draft.voice_description } : current;
            });
            const existingIds = new Set(characters.map((item) => item.id));
            const existingNames = new Set(characters.map((item) => item.name));
            for (const draft of drafts) if ((!draft.character_id || !existingIds.has(draft.character_id)) && !existingNames.has(draft.name)) characters.push({ id: `character_${crypto.randomUUID().replaceAll("-", "")}`, name: draft.name, description: draft.description, wardrobe: draft.wardrobe, voice_description: draft.voice_description, tts_voice: "", reference_asset_ids: [], appearance_profiles: [] });
        }
        const next: Project = { ...project, brief, style_bible: selected.style_bible ? analysis.style_bible : project.style_bible, characters, ai_recommended_shot_count: selected.shot_count ? analysis.recommended_shot_count : project.ai_recommended_shot_count };
        setLocal(next); setAnalysisBusy("apply");
        try { await save(next); setAnalysis(null); setAnalysisNotice("已应用所选分析结果并保存。你仍可手动调整任何字段。"); }
        catch (caught) { setAnalysisError(caught instanceof Error ? caught.message : String(caught)); }
        finally { setAnalysisBusy(""); }
    };
    const updateAnalysisCharacter = (index: number, key: keyof ProjectAnalysisDraft["characters"][number], value: string) => {
        if (!analysis) return;
        setAnalysis({
            ...analysis,
            characters: analysis.characters.map((character, characterIndex) => characterIndex === index ? { ...character, [key]: value } : character),
        });
    };
    const addAnalysisCharacter = () => {
        if (!analysis) return;
        setAnalysis({
            ...analysis,
            characters: [...analysis.characters, { character_id: null, name: "新角色", description: "", wardrobe: "", voice_description: "", reference_observations: "用户在确认阶段新增" }],
        });
        setSelected({ ...selected, characters: true });
    };
    const removeAnalysisCharacter = (index: number) => {
        if (!analysis) return;
        setAnalysis({ ...analysis, characters: analysis.characters.filter((_, characterIndex) => characterIndex !== index) });
    };
    return <div className="grid gap-5 xl:grid-cols-[1.3fr_.7fr]">
        <section className="studio-panel">
            <div className="mb-5 flex flex-wrap items-center justify-between gap-3"><div><p className="studio-kicker">CLIENT BRIEF</p><h2 className="text-xl font-semibold">客户需求</h2></div><div className="flex flex-wrap gap-2"><label className="studio-secondary cursor-pointer"><input className="hidden" type="file" accept=".txt,.md,.markdown,.docx,.pdf" disabled={!!analysisBusy} onChange={(event) => event.target.files?.[0] && void uploadScript(event.target.files[0])} />{analysisBusy === "upload" ? <Loader2 className="animate-spin" size={15} /> : <Upload size={15} />}上传剧本</label><button className="studio-secondary" disabled={!!analysisBusy || !project.brief.story.trim()} onClick={() => { setRewriteOpen(true); setRewriteDraft(null); }}><FilePenLine size={15} />AI 扩写/缩写</button><button className="studio-secondary" disabled={!!analysisBusy || !project.brief.story.trim()} onClick={() => void runAnalysis()}>{analysisBusy === "analyze" ? <Loader2 className="animate-spin" size={15} /> : <WandSparkles size={15} />}AI 分析并回填</button><button className="studio-secondary" disabled={busy.has("series-sync")} onClick={() => project.series_id ? void syncSeries(false) : setSeriesOpen(true)}>{busy.has("series-sync") ? <Loader2 className="animate-spin" size={15} /> : <BookCopy size={15} />}{project.series_id ? "更新系列资料" : "建立系列资料"}</button><button className="studio-primary" disabled={busy.has("save-project")} onClick={() => void save(project)}>{busy.has("save-project") ? <Loader2 className="animate-spin" size={16} /> : <Save size={16} />} 保存</button></div></div>
            {analysisError && <div className="studio-error mb-4">{analysisError}</div>}{analysisNotice && <div className="studio-notice mb-4">{analysisNotice}</div>}
            {project.series_id && <div className="mb-4 rounded-lg border border-cyan-300/15 bg-cyan-300/[.035] px-4 py-3 text-sm text-cyan-100/70"><BookCopy className="mr-2 inline" size={15} />本项目属于系列第 {project.episode_number || 1} 集。AI 分析默认保留继承的画风、风格圣经和负面约束。</div>}
            <div className="grid gap-4 md:grid-cols-2">
                <Field label="项目名称"><input value={project.brief.title} onChange={(event) => changeBrief("title", event.target.value)} /></Field>
                <Field label="客户名称"><input value={project.brief.client_name} onChange={(event) => changeBrief("client_name", event.target.value)} /></Field>
                <Field label="目标时长（秒）"><input type="number" min={1} value={project.brief.target_duration_seconds} onChange={(event) => changeBrief("target_duration_seconds", Number(event.target.value))} /></Field>
                <Field label="画幅"><select value={project.brief.aspect_ratio} onChange={(event) => changeAspect(event.target.value)}><option>16:9</option><option>9:16</option><option>1:1</option></select><p className="mt-1 text-[11px] text-white/25">{project.brief.width}×{project.brief.height}</p></Field>
                <Field label="剧情/故事脚本" wide><textarea rows={10} value={project.brief.story} onChange={(event) => changeBrief("story", event.target.value)} /></Field>
                <div className="rounded-xl border border-cyan-300/12 bg-cyan-300/[.025] p-4 md:col-span-2">
                    <div className="flex flex-wrap items-start justify-between gap-3"><div><p className="studio-kicker">STYLE REFERENCE</p><h3 className="font-semibold">用一张风格参考图 AI 回填</h3><p className="mt-1 text-xs leading-5 text-white/38">可直接上传图片，也可选择素材库中已归档为“画风截图 / 参考视频”的素材。完成后会自动保存并更新下方三个字段以及已有分镜 Prompt。</p></div>{project.style_profile?.approved && <span className="studio-status text-emerald-200">已启用：{project.style_profile.name}</span>}</div>
                    <div className="mt-4 grid gap-3 lg:grid-cols-[140px_1fr_auto]">
                        <div className="aspect-video overflow-hidden rounded-lg border border-white/8 bg-black/25">{activeStyleReference?.type === "image" ? <img className="h-full w-full object-cover" src={projectInlineUrl(project.id, "asset", activeStyleReference.id)} alt={activeStyleReference.name} /> : <div className="flex h-full items-center justify-center text-xs text-white/28">{activeStyleReference ? "参考视频" : "尚未选择"}</div>}</div>
                        <div><span className="studio-label">已上传的画风素材</span><select className="studio-input" value={activeStyleReferenceId} onChange={(event) => setStyleReferenceId(event.target.value)}><option value="">请选择</option>{styleReferenceAssets.map((asset) => <option key={asset.id} value={asset.id}>{asset.name}</option>)}</select><p className="mt-1 text-[11px] text-white/28">素材库中共有 {styleReferenceAssets.length} 个可分析画风素材</p></div>
                        <div className="flex flex-col gap-2 lg:pt-6"><button type="button" className="studio-primary whitespace-nowrap" disabled={busy.has("brief-style-analyze") || !activeStyleReferenceId} onClick={() => void analyzeAndApplyStyle([activeStyleReferenceId])}>{busy.has("brief-style-analyze") ? <Loader2 className="animate-spin" size={15} /> : <WandSparkles size={15} />}分析所选并回填</button><label className="studio-secondary cursor-pointer whitespace-nowrap"><input className="hidden" type="file" accept="image/*,video/*" disabled={busy.has("brief-style-analyze")} onChange={(event) => { const file = event.target.files?.[0]; if (file) void analyzeAndApplyStyle([], file); event.target.value = ""; }} /><Upload size={15} />上传并立即分析</label></div>
                    </div>
                </div>
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
                    <input className="studio-input mb-2" value={character.tts_voice || ""} onChange={(event) => updateCharacter(character.id, "tts_voice", event.target.value)} placeholder="可选：专属 TTS Voice ID；留空按年龄、性别、音色和情绪自动选择" />
                    <div className="mb-2 mt-3 flex items-center justify-between gap-2"><p className="text-xs text-white/40">基础形象参考图</p><button type="button" className="studio-secondary px-2 py-1.5 text-[11px]" disabled={busy.has(`character-analyze-${character.id}-base`) || character.reference_asset_ids.length === 0} onClick={() => void backfillCharacter(character)}>{busy.has(`character-analyze-${character.id}-base`) ? <Loader2 className="animate-spin" size={12} /> : <WandSparkles size={12} />}按所选图 AI 回填</button></div>
                    <div className="max-h-32 space-y-1 overflow-y-auto rounded-lg border border-white/8 p-2">{assets.filter((asset) => asset.type === "image" && asset.role === "character").map((asset) => <label key={asset.id} className="flex items-center gap-2 text-xs text-white/60"><input type="checkbox" checked={character.reference_asset_ids.includes(asset.id)} onChange={() => updateCharacter(character.id, "reference_asset_ids", toggle(character.reference_asset_ids, asset.id))} />{asset.name}</label>)}</div>
                    <div className="mb-2 mt-4 flex items-center justify-between gap-2"><div><p className="text-xs font-semibold text-white/60">不同时间 / 年龄 / 体态形象</p><p className="mt-1 text-[11px] leading-5 text-white/30">基础角色保持身份不变；每个时期可单独设置胖瘦、年龄、妆造和参考图，并在逐镜选择。</p></div><button type="button" className="studio-secondary shrink-0 px-2 py-1.5 text-[11px]" onClick={() => addAppearance(character.id)}><Plus size={12} />时期</button></div>
                    <div className="space-y-3">{character.appearance_profiles.map((appearance) => <div key={appearance.id} className="rounded-lg border border-cyan-300/10 bg-cyan-300/[.025] p-3">
                        <div className="mb-2 flex gap-2"><input className="studio-input font-semibold" value={appearance.label} onChange={(event) => updateAppearance(character.id, appearance.id, { label: event.target.value })} placeholder="如：少女时期 / 中年发福 / 晚年" /><button type="button" className="rounded px-2 text-white/25 hover:text-red-300" onClick={() => setLocal({ ...project, characters: project.characters.map((item) => item.id === character.id ? { ...item, appearance_profiles: item.appearance_profiles.filter((entry) => entry.id !== appearance.id) } : item) })}><Trash2 size={13} /></button></div>
                        <input className="studio-input mb-2" value={appearance.time_context} onChange={(event) => updateAppearance(character.id, appearance.id, { time_context: event.target.value })} placeholder="剧情时间与体态：20 岁偏瘦 / 35 岁丰腴 / 70 岁…" />
                        <textarea className="studio-input mb-2" rows={2} value={appearance.description} onChange={(event) => updateAppearance(character.id, appearance.id, { description: event.target.value })} placeholder="这个时期的面部、发型、体型特征" />
                        <textarea className="studio-input mb-2" rows={2} value={appearance.wardrobe} onChange={(event) => updateAppearance(character.id, appearance.id, { wardrobe: event.target.value })} placeholder="这个时期的固定服装与配饰" />
                        <p className="mb-1 text-[11px] text-white/35">该时期专用参考图</p><div className="max-h-24 space-y-1 overflow-y-auto rounded border border-white/8 p-2">{assets.filter((asset) => asset.type === "image" && asset.role === "character").map((asset) => <label key={asset.id} className="flex items-center gap-2 text-[11px] text-white/55"><input type="checkbox" checked={appearance.reference_asset_ids.includes(asset.id)} onChange={() => updateAppearance(character.id, appearance.id, { reference_asset_ids: toggle(appearance.reference_asset_ids, asset.id) })} />{asset.name}</label>)}</div>
                        <button type="button" className="studio-secondary mt-2 w-full px-2 py-1.5 text-[11px]" disabled={busy.has(`character-analyze-${character.id}-${appearance.id}`) || appearance.reference_asset_ids.length === 0} onClick={() => void backfillCharacter(character, appearance.id)}>{busy.has(`character-analyze-${character.id}-${appearance.id}`) ? <Loader2 className="animate-spin" size={12} /> : <WandSparkles size={12} />}按所选图 AI 回填此时期</button>
                    </div>)}</div>
                </div>)}
            </div>
        </section>
        {styleDraft && <StyleAnalysisResultDialog draft={styleDraft} onClose={() => setStyleDraft(null)} />}
        {seriesOpen && <div className="studio-modal" onMouseDown={() => setSeriesOpen(false)}><section className="studio-dialog max-w-xl" onMouseDown={(event) => event.stopPropagation()}><p className="studio-kicker">SERIES IDENTITY</p><h2 className="mb-2 text-2xl font-semibold">建立系列资料</h2><p className="mb-5 text-sm leading-6 text-white/45">保存本项目已经确认的画风、统一风格圣经、负面约束、角色设定和人物参考图。以后可从首页直接创建续集。</p><div className="space-y-4"><Field label="系列名称"><input value={seriesName} onChange={(event) => setSeriesName(event.target.value)} placeholder="例如：张小差安全科普系列" /></Field><Field label="系列说明（可选）"><textarea rows={3} value={seriesDescription} onChange={(event) => setSeriesDescription(event.target.value)} placeholder="系列定位、固定受众或其他长期约定…" /></Field></div><div className="mt-6 flex justify-end gap-2"><button className="studio-secondary" onClick={() => setSeriesOpen(false)}>取消</button><button className="studio-primary" disabled={!seriesName.trim() || busy.has("series-sync")} onClick={() => void syncSeries(true)}>{busy.has("series-sync") ? <Loader2 className="animate-spin" size={15} /> : <BookCopy size={15} />}保存系列资料</button></div></section></div>}
        {rewriteOpen && <div className="studio-modal" onMouseDown={closeRewrite}><section className="studio-dialog max-w-6xl" onMouseDown={(event) => event.stopPropagation()}><p className="studio-kicker">AI SCRIPT REWRITE</p><div className="mb-5 flex items-start justify-between gap-4"><div><h2 className="text-2xl font-semibold">按目标时长扩写或缩写剧本</h2><p className="mt-1 text-sm leading-6 text-white/45">目标成片 {project.brief.target_duration_seconds} 秒。AI 先生成预览，确认后才会覆盖并保存当前剧本。</p></div><button className="studio-secondary" disabled={!!analysisBusy} onClick={closeRewrite}>关闭</button></div>{!rewriteDraft ? <div className="space-y-5"><div className="grid gap-4 md:grid-cols-2"><Field label="改写方式"><select value={rewriteMode} onChange={(event) => setRewriteMode(event.target.value as typeof rewriteMode)}><option value="auto">AI 根据时长自动判断</option><option value="shorten">只缩写 · 保留核心剧情</option><option value="expand">只扩写 · 补足目标时长</option></select></Field><div className="rounded-lg border border-cyan-300/10 bg-cyan-300/[.035] px-4 py-3"><p className="text-xs text-white/35">当前约束</p><p className="mt-1 text-sm text-cyan-100/75">原稿 {project.brief.story.length} 字符 · 目标 {project.brief.target_duration_seconds} 秒 · 单镜 ≤15 秒</p></div></div><label><span className="studio-label">给 AI 的改写建议（可选）</span><textarea className="studio-input min-h-40 resize-y" maxLength={4000} value={rewriteSuggestions} onChange={(event) => setRewriteSuggestions(event.target.value)} placeholder="例如：保留所有关键问答，但合并重复流程；重点突出人物冲突和结尾反思。或者：增加开场铺垫、人物动机和两个情绪转折，不改变原结局……" autoFocus /></label><div className="flex items-center justify-between text-xs text-white/30"><span>建议会作为本次改写的高优先级要求</span><span>{rewriteSuggestions.length}/4000</span></div><div className="flex justify-end gap-2"><button className="studio-secondary" onClick={closeRewrite}>取消</button><button className="studio-primary" disabled={analysisBusy === "rewrite"} onClick={() => void runRewrite()}>{analysisBusy === "rewrite" ? <Loader2 className="animate-spin" size={16} /> : <WandSparkles size={16} />}生成改写预览</button></div></div> : <div><div className="mb-4 grid gap-3 sm:grid-cols-3"><div className="rounded-lg border border-white/8 bg-black/15 p-3"><p className="text-xs text-white/35">改写方向</p><strong className="mt-1 block">{rewriteDraft.rewrite_mode === "expand" ? "扩写" : rewriteDraft.rewrite_mode === "shorten" ? "缩写" : "平衡改写"}</strong></div><div className="rounded-lg border border-white/8 bg-black/15 p-3"><p className="text-xs text-white/35">预计可实现时长</p><strong className="mt-1 block text-cyan-200">{rewriteDraft.estimated_duration_seconds.toFixed(0)} 秒 / 目标 {project.brief.target_duration_seconds} 秒</strong></div><div className="rounded-lg border border-white/8 bg-black/15 p-3"><p className="text-xs text-white/35">文本长度变化</p><strong className="mt-1 block">{project.brief.story.length} → {rewriteDraft.rewritten_story.length} 字符</strong></div></div><div className="mb-4 rounded-lg border border-cyan-300/10 bg-cyan-300/[.03] px-4 py-3"><p className="text-xs text-white/35">改动摘要</p><p className="mt-1 text-sm leading-6 text-white/70">{rewriteDraft.change_summary || "AI 未提供摘要"}</p></div>{rewriteDraft.feasibility_notes.length > 0 && <div className="mb-4 rounded-lg border border-amber-300/15 bg-amber-300/[.04] px-4 py-3"><p className="text-xs font-semibold text-amber-100/70">制作提醒</p><ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-white/55">{rewriteDraft.feasibility_notes.map((note) => <li key={note}>{note}</li>)}</ul></div>}<div className="grid gap-4 lg:grid-cols-2"><label><span className="studio-label">原始剧本（不会直接修改）</span><textarea className="studio-input min-h-[42vh] resize-y text-white/45" value={project.brief.story} readOnly /></label><label><span className="studio-label">AI 改写预览（应用前仍可手动调整）</span><textarea className="studio-input min-h-[42vh] resize-y" value={rewriteDraft.rewritten_story} onChange={(event) => setRewriteDraft({ ...rewriteDraft, rewritten_story: event.target.value })} /></label></div><div className="mt-5 flex flex-wrap justify-end gap-2"><button className="studio-secondary" disabled={!!analysisBusy} onClick={() => setRewriteDraft(null)}>返回调整建议</button><button className="studio-primary" disabled={analysisBusy === "apply-rewrite" || !rewriteDraft.rewritten_story.trim()} onClick={() => void applyRewrite()}>{analysisBusy === "apply-rewrite" ? <Loader2 className="animate-spin" size={16} /> : <Check size={16} />}应用并保存新剧本</button></div></div>}</section></div>}
        {analysis && <div className="studio-modal" onMouseDown={() => setAnalysis(null)}><section className="studio-dialog max-w-5xl" onMouseDown={(event) => event.stopPropagation()}><p className="studio-kicker">AI SCRIPT ANALYSIS</p><div className="mb-5 flex items-start justify-between gap-4"><div><h2 className="text-2xl font-semibold">选择要回填的分析结果</h2><p className="mt-1 text-sm text-white/40">当前表单不会立刻被覆盖；勾选后再应用并保存。</p></div><button className="studio-secondary" onClick={() => setAnalysis(null)}>关闭</button></div><div className="grid max-h-[65vh] gap-3 overflow-y-auto pr-1 md:grid-cols-2">{([
            ["visual_style", "视觉风格", analysis.visual_style], ["pacing", "叙事节奏", analysis.pacing], ["audience", "目标受众", analysis.audience], ["style_bible", "统一风格圣经", analysis.style_bible], ["negative_prompt", "负面提示词", analysis.negative_prompt], ["delivery_notes", "交付备注", analysis.delivery_notes],
        ] as [string, string, string][]).map(([key, label, value]) => <label key={key} className="rounded-lg border border-white/8 bg-black/15 p-4"><div className="mb-2 flex items-center gap-2"><input type="checkbox" checked={selected[key]} onChange={(event) => setSelected({ ...selected, [key]: event.target.checked })} /><strong>{label}</strong></div><p className="whitespace-pre-wrap text-sm leading-6 text-white/55">{value || "AI 未提供"}</p></label>)}<label className="rounded-lg border border-cyan-300/15 bg-cyan-300/[.03] p-4"><div className="mb-2 flex items-center gap-2"><input type="checkbox" checked={selected.shot_count} onChange={(event) => setSelected({ ...selected, shot_count: event.target.checked })} /><strong>AI 推荐分镜数：{analysis.recommended_shot_count} 镜</strong></div><p className="text-sm leading-6 text-white/55">{analysis.shot_count_reason}</p></label><div className="rounded-lg border border-cyan-300/15 bg-cyan-300/[.03] p-4 md:col-span-2"><div className="mb-2 flex flex-wrap items-center justify-between gap-2"><label className="flex items-center gap-2"><input type="checkbox" checked={selected.characters} onChange={(event) => setSelected({ ...selected, characters: event.target.checked })} /><strong>角色一致性草稿（{analysis.characters.length} 个角色）</strong></label><button type="button" className="studio-secondary px-3 py-1.5 text-xs" onClick={addAnalysisCharacter}><Plus size={13} />添加角色 / 动物</button></div><p className="mb-3 text-xs leading-5 text-white/40">这里可以直接修改 AI 推荐档案；人物、动物、鬼魂等需要跨镜一致的主体都会保存为独立角色。</p><div className="grid gap-3 md:grid-cols-2">{analysis.characters.map((character, index) => <div key={`${character.character_id}-${index}`} className="rounded border border-white/8 p-3"><div className="mb-2 flex gap-2"><input className="studio-input font-semibold" value={character.name} onChange={(event) => updateAnalysisCharacter(index, "name", event.target.value)} placeholder="角色名称，例如：老黄狗" /><button type="button" className="rounded px-2 text-white/25 hover:text-red-300" onClick={() => removeAnalysisCharacter(index)} title="删除此角色草稿"><Trash2 size={14} /></button></div><textarea className="studio-input mb-2" rows={3} value={character.description} onChange={(event) => updateAnalysisCharacter(index, "description", event.target.value)} placeholder="固定外貌；动物请写品种、体型、毛色斑纹、耳尾和眼睛" /><textarea className="studio-input mb-2" rows={2} value={character.wardrobe} onChange={(event) => updateAnalysisCharacter(index, "wardrobe", event.target.value)} placeholder="固定服装、项圈、配饰或无服装" /><textarea className="studio-input" rows={2} value={character.voice_description} onChange={(event) => updateAnalysisCharacter(index, "voice_description", event.target.value)} placeholder="声音设定；不说话可填写无台词" />{character.reference_observations && <p className="mt-2 text-[11px] leading-5 text-cyan-100/45">分析依据：{character.reference_observations}</p>}</div>)}</div></div>{analysis.analysis_notes.length > 0 && <div className="rounded-lg border border-amber-300/15 bg-amber-300/[.03] p-4 md:col-span-2"><strong className="text-amber-100/80">AI 已先采用推荐方案</strong><p className="mt-1 text-xs leading-5 text-white/40">这些是需要你留意的推断，推荐值已经写入上方对应字段或角色卡；满意即可直接应用，也可以在本窗口修改。</p><ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-white/50">{analysis.analysis_notes.map((note) => <li key={note}>{note}</li>)}</ul></div>}</div><div className="mt-5 flex justify-end gap-2"><button className="studio-secondary" onClick={() => setAnalysis(null)}>暂不应用</button><button className="studio-primary" disabled={analysisBusy === "apply" || (selected.characters && analysis.characters.some((character) => !character.name.trim()))} onClick={() => void applyAnalysis()}>{analysisBusy === "apply" ? <Loader2 className="animate-spin" size={15} /> : <Check size={15} />}应用所选并保存</button></div></section></div>}
    </div>;
}

function StoryboardPanel({ bundle, busy, action, refresh, updateLocal }: { bundle: ProjectBundle; busy: string; action: Action; refresh: () => Promise<void>; updateLocal: (shots: Shot[]) => void }) {
    const [count, setCount] = useState(bundle.project.manual_shot_count || bundle.shots.length || Math.max(1, Math.ceil(bundle.project.brief.target_duration_seconds / 8)));
    const [countMode, setCountMode] = useState<"manual" | "ai">(bundle.project.storyboard_count_mode || "ai");
    const [open, setOpen] = useState<string | null>(null);
    const [showGenerate, setShowGenerate] = useState(false);
    const [userSuggestions, setUserSuggestions] = useState("");
    const savedPromptTargets = bundle.project.preferred_prompt_targets || [];
    const initialPromptTargets: ("h3" | "seedance")[] = savedPromptTargets.includes("seedance") ? ["seedance"] : savedPromptTargets.includes("h3") ? ["h3"] : [];
    const [promptTargets, setPromptTargets] = useState<("h3" | "seedance")[]>(initialPromptTargets);
    const [reviseShot, setReviseShot] = useState<Shot | null>(null);
    const [reviseSuggestions, setReviseSuggestions] = useState("");
    const [importOpen, setImportOpen] = useState(false);
    const [importFile, setImportFile] = useState<File | null>(null);
    const [importSuggestions, setImportSuggestions] = useState("");
    const [insertOpen, setInsertOpen] = useState(false);
    const [insertAfterShotId, setInsertAfterShotId] = useState("");
    const [insertSuggestions, setInsertSuggestions] = useState("");
    const [splitShot, setSplitShot] = useState<Shot | null>(null);
    const [splitSuggestions, setSplitSuggestions] = useState("");
    const [splitCount, setSplitCount] = useState<"" | 2 | 3 | 4>("");
    const [splitPreview, setSplitPreview] = useState<ShotSplitPreview | null>(null);
    const [splitBusy, setSplitBusy] = useState<"" | "preview" | "confirm">("");
    const [splitError, setSplitError] = useState("");
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
            () => generateStoryboard(bundle.project.id, countMode === "manual" ? count : undefined, countMode, userSuggestions.trim(), promptTargets),
            isRedo ? "AI 已根据本次建议重新生成分镜草稿" : "AI 已根据本次建议生成详细分镜草稿",
        );
    };
    const submitShotRevision = () => {
        if (!reviseShot || !reviseSuggestions.trim()) return;
        const target = reviseShot;
        setReviseShot(null);
        void action(
            `ai-revise-${target.id}`,
            () => reviseShotWithAi(bundle.project.id, target.id, reviseSuggestions.trim(), promptTargets),
            `镜头 ${target.ordinal} 已按建议重做，时长和镜号保持不变`,
        ).then(refresh);
    };
    const submitImport = () => {
        if (!importFile) return;
        const file = importFile;
        setImportOpen(false);
        void action(
            "storyboard-import",
            () => importStoryboard(bundle.project.id, file, importSuggestions.trim(), promptTargets),
            `已导入 ${file.name}，并按剧本与角色设定补全缺失字段`,
        );
    };
    const submitInsert = () => {
        if (!insertSuggestions.trim()) return;
        const afterShot = insertAfterShotId ? bundle.shots.find((shot) => shot.id === insertAfterShotId) : bundle.shots.at(-1);
        const afterId = afterShot?.id || null;
        const insertionLabel = afterId === bundle.shots.at(-1)?.id ? "结尾" : `镜头 ${afterShot?.ordinal || 0} 后`;
        setInsertOpen(false);
        void action(
            "insert-shot",
            () => insertShotWithAi(bundle.project.id, afterId, insertSuggestions.trim(), promptTargets),
            `已在${insertionLabel}新增 1 个完整分镜，原有镜头和生成结果均已保留`,
        ).then(refresh);
    };
    const openSplit = (shot: Shot) => {
        setSplitShot(shot);
        setSplitSuggestions("");
        setSplitCount("");
        setSplitPreview(null);
        setSplitError("");
    };
    const closeSplit = () => {
        if (splitBusy) return;
        setSplitShot(null);
        setSplitPreview(null);
        setSplitError("");
    };
    const generateSplitPreview = async () => {
        if (!splitShot) return;
        setSplitBusy("preview");
        setSplitError("");
        try {
            setSplitPreview(await previewShotSplit(bundle.project.id, splitShot.id, {
                userSuggestions: splitSuggestions.trim(),
                segmentCount: splitCount || null,
            }));
        } catch (caught) {
            setSplitError(caught instanceof Error ? caught.message : String(caught));
        } finally {
            setSplitBusy("");
        }
    };
    const patchSplitSegment = (index: number, patch: Partial<ShotSplitPreview["segments"][number]>) => {
        setSplitPreview((current) => {
            if (!current) return current;
            const segments = current.segments.map((segment, segmentIndex) => segmentIndex === index ? { ...segment, ...patch } : segment);
            return { ...current, segments, proposed_duration_seconds: Number(segments.reduce((sum, segment) => sum + segment.duration_seconds, 0).toFixed(2)) };
        });
    };
    const applySplit = async () => {
        if (!splitShot || !splitPreview) return;
        setSplitBusy("confirm");
        setSplitError("");
        try {
            await confirmShotSplit(bundle.project.id, splitShot.id, splitPreview, promptTargets);
            setSplitShot(null);
            setSplitPreview(null);
            await refresh();
        } catch (caught) {
            setSplitError(caught instanceof Error ? caught.message : String(caught));
        } finally {
            setSplitBusy("");
        }
    };
    return <div>
        <section className="studio-panel mb-5 flex flex-wrap items-center justify-between gap-4">
            <div><p className="studio-kicker">STORYBOARD V{bundle.project.storyboard_version}</p><h2 className="text-xl font-semibold">详细分镜设计表</h2><p className="mt-1 text-sm text-white/40">{bundle.shots.length} 镜 · 合计 {duration.toFixed(1)} 秒 · 每镜强制 ≤15 秒</p></div>
            <div className="flex flex-wrap gap-2"><select className="studio-input w-36" value={countMode} onChange={(event) => setCountMode(event.target.value as "manual" | "ai")}><option value="ai">AI 判断镜头数</option><option value="manual">手动指定镜头数</option></select>{countMode === "manual" ? <input className="studio-input w-20" type="number" min={1} max={500} value={count} onChange={(event) => setCount(Number(event.target.value))} /> : <span className="inline-flex items-center rounded-lg border border-cyan-300/10 bg-cyan-300/[.035] px-3 text-xs text-cyan-100/60">{bundle.project.ai_recommended_shot_count ? `已推荐 ${bundle.project.ai_recommended_shot_count} 镜` : "将按剧情节奏自动判断"}</span>}<button className="studio-primary" disabled={!!busy} onClick={() => setShowGenerate(true)}>{busy === "storyboard" ? <Loader2 className="animate-spin" size={16} /> : <WandSparkles size={16} />} {isRedo ? "AI 按建议重做" : "AI 生成分镜"}</button><button className="studio-secondary" disabled={!!busy} onClick={() => { setInsertAfterShotId(""); setInsertSuggestions(""); setInsertOpen(true); }}><Plus size={15} />单独新增分镜</button><button className="studio-secondary" disabled={!!busy} onClick={() => { setImportOpen(true); setImportFile(null); }}><Upload size={15} />导入分镜表</button><a className="studio-secondary" href={storyboardCsvUrl(bundle.project.id)}><Download size={15} />导出表格</a></div>
        </section>
        {(bundle.project.storyboard_warnings?.length || 0) > 0 && <div className="mb-4 rounded-lg border border-amber-300/15 bg-amber-300/[.04] px-4 py-3 text-sm leading-6 text-amber-100/75"><strong>分镜已保留，并有待补充项：</strong><ul className="mt-1 list-disc pl-5">{bundle.project.storyboard_warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></div>}
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
                        <Field label="首帧完整度" hint="仅影响 Seedance 自动参考策略。完整表示首帧已包含本镜需要保持的全部人物、场景和固定物品。"><select value={shot.first_frame_completeness || "unknown"} onChange={(event) => patchShot(shot.id, { first_frame_completeness: event.target.value as Shot["first_frame_completeness"], seedance_reference_mode: "auto" })}><option value="unknown">未确认 · 默认全模态</option><option value="complete">完整 · 自动严格首帧</option><option value="incomplete">不完整 · 自动全模态</option></select></Field>
                        <Field label="起始场景模板（按需）"><select value={shot.use_scene_profile ? (selectedSceneProfileIds(shot)[0] || "") : ""} onChange={(event) => patchShot(shot.id, sceneProfileSelectionPatch(shot, "start", event.target.value))}><option value="">不使用场景档案 · 保持原链路</option>{bundle.project.scene_profiles.map((profile) => <option key={profile.id} value={profile.id}>{profile.name}{profile.approved ? " · 已有场景母版" : " · 待生成母版"}</option>)}</select><span className="mt-1 block text-[11px] leading-5 text-white/35">首帧和转场前的空间；普通单场景镜头只选这一项。</span></Field>
                        <Field label="转场后场景模板（可选）"><select disabled={!shot.use_scene_profile || !selectedSceneProfileIds(shot)[0]} value={selectedSceneProfileIds(shot)[1] || ""} onChange={(event) => patchShot(shot.id, sceneProfileSelectionPatch(shot, "destination", event.target.value))}><option value="">无 · 本镜不跨场景</option>{bundle.project.scene_profiles.filter((profile) => profile.id !== selectedSceneProfileIds(shot)[0]).map((profile) => <option key={profile.id} value={profile.id}>{profile.name}{profile.approved ? " · 已有场景母版" : " · 待生成母版"}</option>)}</select><span className="mt-1 block text-[11px] leading-5 text-white/35">跨空间镜头按“起始→目标”执行，两套环境不会同时混在同一阶段。</span></Field>
                        <Field label="Seedance 镜头衔接"><select value={shot.continuity_mode} onChange={(event) => patchShot(shot.id, { continuity_mode: event.target.value as Shot["continuity_mode"], continuity_source_shot_id: event.target.value === "continuous" ? (bundle.shots[index - 1]?.id || null) : null })}><option value="independent">独立镜头 · 正常切镜</option><option value="same_scene">同场景切镜 · 不硬接尾帧</option><option value="continuous" disabled={index === 0}>连续长镜头 · 上一镜尾帧续接</option></select><span className="mt-1 block text-[11px] leading-5 text-white/35">“连续长镜头”会把上一镜真实尾帧作为本镜严格首帧；普通对话换机位选“同场景切镜”。</span></Field>
                        <Field label="Seedance 参考策略"><select value={shot.seedance_reference_mode} onChange={(event) => patchShot(shot.id, { seedance_reference_mode: event.target.value as Shot["seedance_reference_mode"] })}><option value="auto">自动 · 按首帧完整度选择</option><option value="multimodal_reference">手动全模态参考</option><option value="strict_first_frame">手动严格首帧</option></select><span className="mt-1 block text-[11px] leading-5 text-white/35">自动模式会读取下方首帧完整度；跨场景或首帧信息不足时保留全模态。</span></Field>
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
                        <Field label="本镜头角色与时期形象" wide><div className="grid gap-2 rounded-lg border border-white/8 p-3 sm:grid-cols-2 lg:grid-cols-3">{bundle.project.characters.length === 0 ? <span className="text-xs text-white/30">请先在“需求与角色”中建立角色。</span> : bundle.project.characters.map((character) => { const checked = shot.character_ids.includes(character.id); const appearance = character.appearance_profiles.find((item) => item.id === shot.character_appearance_ids[character.id]); const referenceCount = appearance ? appearance.reference_asset_ids.length : character.reference_asset_ids.length; return <div key={character.id} className="rounded-md border border-white/6 bg-black/15 p-2"><label className="flex items-center gap-2 text-xs text-white/60"><input type="checkbox" checked={checked} onChange={() => { const nextIds = toggle(shot.character_ids, character.id); const nextAppearances = { ...shot.character_appearance_ids }; if (!nextIds.includes(character.id)) delete nextAppearances[character.id]; patchShot(shot.id, { character_ids: nextIds, character_appearance_ids: nextAppearances }); }} /><span className="min-w-0 flex-1">{character.name}</span><span className={referenceCount > 0 ? "text-cyan-200/45" : "text-amber-200/55"}>{referenceCount > 0 ? `${referenceCount} 图` : "缺图"}</span></label>{checked && character.appearance_profiles.length > 0 && <select className="studio-input mt-2 py-1.5 text-[11px]" value={shot.character_appearance_ids[character.id] || ""} onChange={(event) => { const next = { ...shot.character_appearance_ids }; if (event.target.value) next[character.id] = event.target.value; else delete next[character.id]; patchShot(shot.id, { character_appearance_ids: next }); }}><option value="">基础形象</option>{character.appearance_profiles.map((appearance) => <option key={appearance.id} value={appearance.id}>{appearance.label}{appearance.time_context ? ` · ${appearance.time_context}` : ""}</option>)}</select>}</div>; })}</div><span className="mt-1 block text-[11px] leading-5 text-white/35">系统会把 Prompt 中识别到的完整角色和括号别名自动补勾，并同步对应参考图；右侧“缺图”表示该角色还需要在“需求与角色”绑定或生成人设图。</span></Field>
                        <Field label="通用分镜 Prompt（本地编译）" wide><textarea rows={5} value={shot.visual_prompt} onChange={(event) => patchShot(shot.id, { visual_prompt: event.target.value })} /><span className="mt-1 block text-[11px] leading-5 text-white/35">由事件、首帧状态、角色和风格在本地编译；分镜模型不再重复输出这一大段文本。</span></Field>
                        <Field label="首帧图 Prompt" wide><textarea rows={5} value={shot.keyframe_prompt} onChange={(event) => patchShot(shot.id, { keyframe_prompt: event.target.value })} /><span className="mt-1 block text-[11px] leading-5 text-white/35">只描述视频开始时的静态画面，生成分镜图时实际使用这一项。</span></Field>
                        <Field label={`MiniMax H3 Prompt · ${shot.h3_prompt_skill_id || "h3-prompt-writing"}`} wide><textarea rows={6} value={shot.video_prompt} onChange={(event) => patchShot(shot.id, { video_prompt: event.target.value })} /><span className="mt-1 block text-[11px] leading-5 text-white/35">可手动微调；在“视频生成”页可选择官方风格 Skill 批量重新生成。</span></Field>
                        <Field label={`Seedance 2.0 全模态 Prompt${shot.seedance_prompt_version ? ` · ${shot.seedance_prompt_version}` : ""}`} wide><textarea rows={7} value={shot.seedance_prompt} onChange={(event) => patchShot(shot.id, { seedance_prompt: event.target.value })} placeholder="在“视频生成”页点击编译 Seedance Prompt 后自动生成，也可在这里手动微调。" /><span className="mt-1 block text-[11px] leading-5 text-white/35">使用图片1/视频1/音频1编号，与方舟 content 数组顺序严格一致；不会覆盖 H3 Prompt。</span></Field>
                    </div>
                    <div className="mt-4 flex flex-wrap justify-end gap-2"><button className="studio-secondary" disabled={!!busy || !!splitBusy} onClick={() => openSplit(shot)}><Scissors size={14} />AI 拆分本镜</button><button className="studio-secondary" disabled={!!busy || !!splitBusy} onClick={() => { setReviseShot(shot); setReviseSuggestions(""); }}><WandSparkles size={14} />AI 重做本镜</button><button className="studio-secondary px-3" disabled={index === 0 || !!busy || !!splitBusy} onClick={() => moveShot(index, -1)}><ChevronUp size={14} />上移</button><button className="studio-secondary px-3" disabled={index === bundle.shots.length - 1 || !!busy || !!splitBusy} onClick={() => moveShot(index, 1)}><ChevronDown size={14} />下移</button><button className="studio-danger" disabled={!!splitBusy} onClick={() => void action(`delete-${shot.id}`, () => deleteShot(bundle.project.id, shot.id), `镜头 ${shot.ordinal} 已删除`)}><Trash2 size={14} />删除</button><button className="studio-primary" disabled={busy === `shot-${shot.id}` || !!splitBusy} onClick={() => void action(`shot-${shot.id}`, () => updateShot(shot), `镜头 ${shot.ordinal} 已保存且三个下游 Prompt 已同步`, false).then(refresh)}>{busy === `shot-${shot.id}` ? <Loader2 className="animate-spin" size={15} /> : <Save size={15} />}保存镜头</button></div>
                </div>}
            </article>)}
        </div>}
        {splitShot && <div className="studio-modal" onMouseDown={closeSplit}>
            <section className="studio-dialog max-h-[92vh] max-w-5xl overflow-y-auto" onMouseDown={(event) => event.stopPropagation()}>
                <p className="studio-kicker">AI SPLIT ONE SHOT</p>
                <div className="mb-5 flex items-start justify-between gap-4">
                    <div className="flex items-start gap-3"><span className="rounded-xl bg-cyan-300/10 p-3 text-cyan-200"><Scissors size={22} /></span><div><h2 className="text-2xl font-semibold">拆分镜头 {splitShot.ordinal}</h2><p className="mt-1 text-sm leading-6 text-white/45">{splitPreview ? `AI 已生成 ${splitPreview.segments.length} 镜预览；确认前可直接修改标题、剧情、首帧和时长。` : "AI 先生成可编辑预览；未点击确认时不会修改现有分镜。"}</p></div></div>
                    <button className="studio-secondary px-3" disabled={!!splitBusy} onClick={closeSplit}><X size={15} />关闭</button>
                </div>
                {splitError && <div className="studio-error mb-4">{splitError}</div>}
                {!splitPreview ? <div className="space-y-5">
                    <div className="grid gap-4 md:grid-cols-[220px_1fr]">
                        <Field label="拆成几镜"><select value={splitCount} onChange={(event) => setSplitCount(event.target.value ? Number(event.target.value) as 2 | 3 | 4 : "")}><option value="">AI 根据节拍判断（2–4 镜）</option><option value="2">固定拆成 2 镜</option><option value="3">固定拆成 3 镜</option><option value="4">固定拆成 4 镜</option></select></Field>
                        <div className="rounded-lg border border-amber-300/15 bg-amber-300/[.04] px-4 py-3 text-sm leading-6 text-amber-100/70"><strong>原镜 {splitShot.duration_seconds} 秒</strong><span className="ml-2 text-white/45">{splitShot.title || splitShot.narrative}</span><p className="mt-1 text-xs text-white/40">拆分后使用全新镜头 ID；原镜已有首帧和成片不会误绑到新镜头。</p></div>
                    </div>
                    <label><span className="studio-label">给 AI 的拆分要求（可选）</span><textarea className="studio-input min-h-40 resize-y" maxLength={4000} value={splitSuggestions} onChange={(event) => setSplitSuggestions(event.target.value)} placeholder="例如：第一镜只保留人物发现异常，第二镜改成道具特写，第三镜再呈现人物反应；总时长保持不变。" autoFocus /></label>
                    <div className="flex justify-end gap-2"><button className="studio-secondary" disabled={!!splitBusy} onClick={closeSplit}>取消</button><button className="studio-primary" disabled={!!splitBusy} onClick={() => void generateSplitPreview()}>{splitBusy === "preview" ? <Loader2 className="animate-spin" size={15} /> : <WandSparkles size={15} />}生成拆分预览</button></div>
                </div> : <div>
                    <div className="mb-4 grid gap-3 sm:grid-cols-3"><div className="rounded-lg border border-white/8 bg-black/15 p-3"><p className="text-xs text-white/35">原镜时长</p><strong className="mt-1 block">{splitPreview.original_duration_seconds.toFixed(2)} 秒</strong></div><div className="rounded-lg border border-white/8 bg-black/15 p-3"><p className="text-xs text-white/35">拆分后</p><strong className="mt-1 block text-cyan-200">{splitPreview.segments.length} 镜 · {splitPreview.proposed_duration_seconds.toFixed(2)} 秒</strong></div><div className="rounded-lg border border-white/8 bg-black/15 p-3"><p className="text-xs text-white/35">时长变化</p><strong className="mt-1 block">{(splitPreview.proposed_duration_seconds - splitPreview.original_duration_seconds).toFixed(2)} 秒</strong></div></div>
                    {splitPreview.rationale && <div className="mb-4 rounded-lg border border-cyan-300/10 bg-cyan-300/[.03] px-4 py-3 text-sm leading-6 text-white/55"><strong className="text-cyan-100/70">拆分依据：</strong>{splitPreview.rationale}</div>}
                    <div className="space-y-4">{splitPreview.segments.map((segment, index) => <article key={index} className="rounded-xl border border-white/8 bg-black/15 p-4">
                        <div className="mb-3 flex items-center gap-3"><span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-cyan-300/10 font-mono text-xs text-cyan-200">{splitPreview.source_ordinal + index}</span><strong className="min-w-0 flex-1">拆分镜头 {index + 1}</strong><span className="text-xs text-white/30">{segment.character_names.join("、") || "无固定角色"}</span></div>
                        <div className="grid gap-4 md:grid-cols-[1fr_150px]"><Field label="镜头标题"><input value={segment.title} onChange={(event) => patchSplitSegment(index, { title: event.target.value })} /></Field><Field label="时长（0.25–15秒）"><input type="number" min={0.25} max={15} step={0.25} value={segment.duration_seconds} onChange={(event) => patchSplitSegment(index, { duration_seconds: Math.min(15, Math.max(.25, Number(event.target.value) || .25)) })} /></Field></div>
                        <div className="mt-4 grid gap-4 md:grid-cols-2"><Field label="剧情内容 / 本镜事件"><textarea rows={4} value={segment.narrative} onChange={(event) => patchSplitSegment(index, { narrative: event.target.value })} /></Field><Field label="0 秒首帧状态"><textarea rows={4} value={segment.scene_description} onChange={(event) => patchSplitSegment(index, { scene_description: event.target.value })} /></Field></div>
                        <details className="mt-3 rounded-lg border border-white/8 px-3 py-2 text-xs text-white/45"><summary className="cursor-pointer">查看 AI 镜头设计</summary><div className="mt-2 grid gap-2 sm:grid-cols-2"><span>场景：{segment.scene_profile_name || "沿用原镜"}</span><span>景别 / 机位：{segment.shot_size} · {segment.camera_angle}</span><span>镜头 / 运镜：{segment.lens} · {segment.camera_motion}</span><span>节拍：{segment.visual_beats.length} 段动作 · {segment.voice_events.length} 段声音</span></div></details>
                    </article>)}</div>
                    <div className="mt-5 rounded-lg border border-amber-300/15 bg-amber-300/[.04] px-4 py-3 text-xs leading-5 text-amber-100/70">确认后，原镜会被这 {splitPreview.segments.length} 个新镜头原位替换，后续镜号自动顺延。新镜头会重新编译首帧、H3 和 Seedance Prompt，不继承原镜的已生成媒体。</div>
                    <div className="mt-5 flex flex-wrap justify-end gap-2"><button className="studio-secondary" disabled={!!splitBusy} onClick={() => { setSplitPreview(null); setSplitError(""); }}>返回重新生成</button><button className="studio-primary" disabled={!!splitBusy || splitPreview.segments.some((segment) => !segment.narrative.trim())} onClick={() => void applySplit()}>{splitBusy === "confirm" ? <Loader2 className="animate-spin" size={15} /> : <Check size={15} />}确认替换为 {splitPreview.segments.length} 镜</button></div>
                </div>}
            </section>
        </div>}
        {insertOpen && <div className="studio-modal" onMouseDown={() => setInsertOpen(false)}><section className="studio-dialog max-w-2xl" onMouseDown={(event) => event.stopPropagation()}><p className="studio-kicker">AI INSERT ONE SHOT</p><div className="mb-5 flex items-start gap-3"><span className="rounded-xl bg-cyan-300/10 p-3 text-cyan-200"><Plus size={22} /></span><div><h2 className="text-2xl font-semibold">单独新增一个完整分镜</h2><p className="mt-1 text-sm leading-6 text-white/45">只生成并插入这一镜；现有分镜、首帧、H3 Prompt、Seedance Prompt 和视频结果都保留。</p></div></div><div className="grid gap-4"><Field label="插入位置"><select value={insertAfterShotId} onChange={(event) => setInsertAfterShotId(event.target.value)}><option value="">作为新的结尾镜头</option>{bundle.shots.slice(0, -1).map((shot) => <option key={shot.id} value={shot.id}>插在镜头 {shot.ordinal} 后</option>)}</select></Field><label><span className="studio-label">新增镜头要求</span><textarea className="studio-input min-h-44 resize-y" maxLength={4000} value={insertSuggestions} onChange={(event) => setInsertSuggestions(event.target.value)} placeholder="例如：结尾切回地府大厅，人挤人排着长队，队伍延伸到画面深处；用大全景揭示规模，形成荒诞喜剧反差，不新增主角对白。" autoFocus /></label><div className="rounded-lg border border-cyan-300/10 bg-cyan-300/[.03] p-4"><span className="studio-label">同时生成哪些视频 Prompt</span><div className="mt-2 flex flex-wrap gap-5 text-sm text-white/65"><label className="flex items-center gap-2"><input type="checkbox" checked={promptTargets.includes("h3")} onChange={() => setPromptTargets(toggle(promptTargets, "h3") as ("h3" | "seedance")[])} />MiniMax H3（按需）</label><label className="flex items-center gap-2"><input type="checkbox" checked={promptTargets.includes("seedance")} onChange={() => setPromptTargets(toggle(promptTargets, "seedance") as ("h3" | "seedance")[])} />Seedance 2.x（本地编译）</label></div></div></div><div className="mt-6 flex justify-end gap-2"><button className="studio-secondary" onClick={() => setInsertOpen(false)}>取消</button><button className="studio-primary" disabled={!insertSuggestions.trim() || !!busy} onClick={submitInsert}>{busy === "insert-shot" ? <Loader2 className="animate-spin" size={15} /> : <WandSparkles size={15} />}生成并插入这一镜</button></div></section></div>}
        {showGenerate && <div className="studio-modal" onMouseDown={() => setShowGenerate(false)}><section className="studio-dialog max-w-2xl" onMouseDown={(event) => event.stopPropagation()}><p className="studio-kicker">AI STORYBOARD DIRECTION</p><div className="mb-5 flex items-start gap-3"><span className="rounded-xl bg-cyan-300/10 p-3 text-cyan-200"><MessageSquareText size={22} /></span><div><h2 className="text-2xl font-semibold">{isRedo ? "让 AI 按建议重新修改分镜" : "生成分镜前补充你的建议"}</h2><p className="mt-1 text-sm leading-6 text-white/45">你的文字会和剧情脚本、角色设定、视觉风格一起交给当前分镜模型，并作为本次生成的高优先级要求。</p></div></div>{isRedo && <div className="mb-4 rounded-lg border border-amber-300/15 bg-amber-300/[.04] px-4 py-3 text-sm leading-6 text-amber-100/70">本次会重新生成并替换当前 {bundle.shots.length} 个镜头。需要保留的剧情、镜头或对白，请在建议中明确写出。</div>}<div className="mb-4 rounded-lg border border-cyan-300/10 bg-cyan-300/[.03] p-4"><span className="studio-label">同时生成哪些视频 Prompt</span><div className="mt-2 flex flex-wrap gap-5 text-sm text-white/65"><label className="flex items-center gap-2"><input type="checkbox" checked={promptTargets.includes("h3")} onChange={() => setPromptTargets(toggle(promptTargets, "h3") as ("h3" | "seedance")[])} />MiniMax H3（按需）</label><label className="flex items-center gap-2"><input type="checkbox" checked={promptTargets.includes("seedance")} onChange={() => setPromptTargets(toggle(promptTargets, "seedance") as ("h3" | "seedance")[])} />Seedance 2.x（本地编译）</label></div><p className="mt-2 text-[11px] text-white/35">默认只编译 Seedance；H3 可在这里勾选，或到“视频生成”页只为选中的镜头补生成。</p></div><label><span className="studio-label">给 AI 的本次建议（可选）</span><textarea className="studio-input min-h-40 resize-y" maxLength={4000} value={userSuggestions} onChange={(event) => setUserSuggestions(event.target.value)} placeholder={isRedo ? "例如：保留前 3 镜的剧情；中段减少对白、增加动作；结尾改成角色回头的近景，并让节奏更紧凑……" : "例如：前 3 秒必须有强钩子；人物多用近景；减少旁白、用动作推进；整体保持压抑悬疑感……"} autoFocus /></label><div className="mt-2 flex flex-wrap items-center justify-between gap-3 text-xs text-white/30"><span>{countMode === "ai" ? "AI 会结合这份建议重新判断镜头数" : `本次固定生成 ${count} 个镜头`}</span><span>{userSuggestions.length}/4000</span></div><div className="mt-6 flex flex-wrap justify-end gap-2"><button className="studio-secondary" onClick={() => setShowGenerate(false)}>取消</button><button className="studio-primary" onClick={submitGeneration}><WandSparkles size={16} />{userSuggestions.trim() ? (isRedo ? "按建议重新生成" : "按建议生成分镜") : (isRedo ? "不填建议直接重做" : "不填建议直接生成")}</button></div></section></div>}
        {reviseShot && <div className="studio-modal" onMouseDown={() => setReviseShot(null)}><section className="studio-dialog max-w-2xl" onMouseDown={(event) => event.stopPropagation()}><p className="studio-kicker">AI REDO ONE SHOT</p><h2 className="text-2xl font-semibold">AI 重做镜头 {reviseShot.ordinal}</h2><p className="mt-2 text-sm leading-6 text-white/45">镜头编号与时长 {reviseShot.duration_seconds} 秒固定；人物、对白、场景、动作和构图都可根据你的建议变化。保存后首帧、H3 与 Seedance Prompt 会同步更新。</p><label className="mt-5 block"><span className="studio-label">本镜修改建议</span><textarea className="studio-input min-h-44 resize-y" maxLength={4000} value={reviseSuggestions} onChange={(event) => setReviseSuggestions(event.target.value)} placeholder="例如：让 D 改成画外旁白，场景切到走廊；或者保留人物但重写对白和动作……" autoFocus /></label><div className="mt-5 flex justify-end gap-2"><button className="studio-secondary" onClick={() => setReviseShot(null)}>取消</button><button className="studio-primary" disabled={!reviseSuggestions.trim() || !!busy} onClick={submitShotRevision}><WandSparkles size={15} />重做并同步所选 Prompt</button></div></section></div>}
        {importOpen && <div className="studio-modal" onMouseDown={() => setImportOpen(false)}><section className="studio-dialog max-w-2xl" onMouseDown={(event) => event.stopPropagation()}><p className="studio-kicker">IMPORT STORYBOARD</p><h2 className="text-2xl font-semibold">导入客户分镜表</h2><p className="mt-2 text-sm leading-6 text-white/45">支持 CSV、XLSX、XLSM。表格有的内容会原样保留；系统缺少的角色、景别、运镜、声音、首帧及视频 Prompt 会结合当前剧本和角色设定由 AI 补全，行数不会改变。</p><label className="studio-empty mt-5 min-h-32 cursor-pointer"><input className="hidden" type="file" accept=".csv,.xlsx,.xlsm" onChange={(event) => setImportFile(event.target.files?.[0] || null)} /><Upload className="text-cyan-300" /><strong>{importFile?.name || "选择分镜表文件"}</strong><span>首行为表头；可使用中文列名</span></label><label className="mt-4 block"><span className="studio-label">给 AI 的补全建议（可选）</span><textarea className="studio-input min-h-32 resize-y" value={importSuggestions} onChange={(event) => setImportSuggestions(event.target.value)} placeholder="例如：客户表中对白和时长不可改；缺失镜头语言按古装纪录片风格补齐；只生成 Seedance Prompt……" /></label><div className="mt-4 rounded-lg border border-cyan-300/10 bg-cyan-300/[.03] p-3"><span className="studio-label">同时补生成视频 Prompt</span><div className="mt-2 flex gap-5 text-sm text-white/60"><label className="flex items-center gap-2"><input type="checkbox" checked={promptTargets.includes("h3")} onChange={() => setPromptTargets(toggle(promptTargets, "h3") as ("h3" | "seedance")[])} />MiniMax H3（按需）</label><label className="flex items-center gap-2"><input type="checkbox" checked={promptTargets.includes("seedance")} onChange={() => setPromptTargets(toggle(promptTargets, "seedance") as ("h3" | "seedance")[])} />Seedance（本地编译）</label></div><p className="mt-2 text-[11px] text-white/35">默认仅编译 Seedance，H3 可在这里单独选择。</p></div><div className="mt-5 flex justify-end gap-2"><button className="studio-secondary" onClick={() => setImportOpen(false)}>取消</button><button className="studio-primary" disabled={!importFile || !!busy} onClick={submitImport}><WandSparkles size={15} />导入并 AI 补全</button></div></section></div>}
    </div>;
}

function StyleAnalysisResultDialog({ draft, onClose }: { draft: StyleAnalysisDraft; onClose: () => void }) {
    return <div className="studio-modal" onMouseDown={onClose}><section className="studio-dialog max-w-3xl" onMouseDown={(event) => event.stopPropagation()}><div className="flex flex-wrap items-start justify-between gap-3"><div><p className="studio-kicker">STYLE ANALYSIS · SAVED</p><h2 className="text-2xl font-semibold">{draft.name}</h2></div><span className="studio-status text-emerald-200">已自动回填并保存</span></div><p className="mt-3 text-sm leading-6 text-white/55">{draft.analysis_summary}</p><div className="mt-4 grid gap-3 sm:grid-cols-2"><div className="rounded-lg border border-white/8 p-3 text-sm"><strong>媒介 / 质感</strong><p className="mt-1 leading-6 text-white/45">{draft.medium}；{draft.texture}</p></div><div className="rounded-lg border border-white/8 p-3 text-sm"><strong>色彩 / 光线</strong><p className="mt-1 leading-6 text-white/45">{draft.palette}；{draft.lighting}</p></div><div className="rounded-lg border border-white/8 p-3 text-sm"><strong>镜头 / 构图</strong><p className="mt-1 leading-6 text-white/45">{draft.camera_language}；{draft.composition}</p></div><div className="rounded-lg border border-white/8 p-3 text-sm"><strong>动作 / 剪辑</strong><p className="mt-1 leading-6 text-white/45">{draft.motion_language}</p></div></div><div className="mt-3 rounded-lg border border-rose-300/10 bg-rose-300/[.025] p-3 text-sm"><strong className="text-rose-100/70">避免项</strong><p className="mt-1 leading-6 text-white/45">{draft.negative_constraints}</p></div>{draft.observations.length > 0 && <details className="mt-3 rounded-lg border border-white/8 p-3 text-sm"><summary className="cursor-pointer text-white/55">查看 AI 的画面观察依据（{draft.observations.length} 条）</summary><ul className="mt-2 list-disc space-y-1 pl-5 text-xs leading-5 text-white/40">{draft.observations.map((observation, index) => <li key={`${index}-${observation}`}>{observation}</li>)}</ul></details>}<p className="mt-4 rounded-lg border border-cyan-300/10 bg-cyan-300/[.03] px-3 py-2 text-xs leading-5 text-cyan-100/50">本次结论已写入“需求与角色 → 视觉风格 / 统一风格圣经 / 负面提示词”，并同步刷新已有首帧、H3 和 Seedance Prompt。关闭后仍可在参考素材页的“ACTIVE STYLE”查看。</p><div className="mt-5 flex justify-end"><button className="studio-primary" onClick={onClose}><Check size={15} />完成</button></div></section></div>;
}

function AssetsPanel({ project, assets, shots, busy, action }: { project: Project; assets: Asset[]; shots: Shot[]; busy: BusyState; action: Action }) {
    const [role, setRole] = useState<AssetRole>("character");
    const [preview, setPreview] = useState<MediaPreviewState | null>(null);
    const [styleDraft, setStyleDraft] = useState<StyleAnalysisDraft | null>(null);
    const [selectedAssetIds, setSelectedAssetIds] = useState<string[]>([]);
    const [showAll, setShowAll] = useState(false);
    const [sceneHelpOpen, setSceneHelpOpen] = useState(false);
    const [characterAction, setCharacterAction] = useState<"generate" | "analyze" | null>(null);
    const [targetCharacterId, setTargetCharacterId] = useState(project.characters[0]?.id || "");
    const [targetAppearanceId, setTargetAppearanceId] = useState("");
    const [characterSuggestions, setCharacterSuggestions] = useState("");
    const [characterReferenceStrategy, setCharacterReferenceStrategy] = useState<"identity" | "project_style">("project_style");
    const [batchCharacterSuggestions, setBatchCharacterSuggestions] = useState("");
    const [propBindingAssetId, setPropBindingAssetId] = useState<string | null>(null);
    const [propBindingShotIds, setPropBindingShotIds] = useState<string[]>([]);
    const upload = (files: File[]) => action("upload", () => Promise.all(files.map((file) => uploadProjectAsset(project.id, file, role))), `已上传 ${files.length} 个参考素材`);
    const selectedAssets = assets.filter((asset) => selectedAssetIds.includes(asset.id));
    const characterImageIds = new Set(assets.filter((asset) => asset.type === "image" && asset.role === "character").map((asset) => asset.id));
    const charactersWithReferences = project.characters.filter((character) => character.reference_asset_ids.some((assetId) => characterImageIds.has(assetId)) || assets.some((asset) => asset.type === "image" && asset.role === "character" && asset.character_id === character.id));
    const missingCharacterCount = Math.max(0, project.characters.length - charactersWithReferences.length);
    const uploadedCharacterImages = assets.filter((asset) => asset.type === "image" && asset.role === "character");
    const boundCharacterImageIds = new Set(project.characters.flatMap((character) => character.reference_asset_ids));
    const selectedCharacterImageIds = selectedAssets.filter((asset) => asset.type === "image" && asset.role === "character" && boundCharacterImageIds.has(asset.id)).map((asset) => asset.id);
    const unboundCharacterImageCount = uploadedCharacterImages.filter((asset) => !boundCharacterImageIds.has(asset.id) && !asset.character_id).length;
    const characterBatchBusy = busy.has("character-batch-script-style") || busy.has("character-batch-partial");
    const roleAssets = assets.filter((asset) => asset.role === role);
    const visibleAssets = showAll ? assets : roleAssets;
    const styleAssets = (selectedAssets.length ? selectedAssets : assets).filter((asset) => ["style", "motion"].includes(asset.role) && ["image", "video"].includes(asset.type));
    const hasProjectStyle = !!(project.style_profile?.approved || project.brief.visual_style.trim() || project.style_bible.trim() || assets.some((asset) => asset.role === "style" && asset.type === "image"));
    const mappedSceneShotIds = new Set(project.scene_profiles.flatMap((profile) => profile.source_shot_ids));
    const mappedSceneShotCount = shots.filter((shot) => mappedSceneShotIds.has(shot.id)).length;
    const appliedSceneShotCount = shots.filter((shot) => shot.use_scene_profile && mappedSceneShotIds.has(shot.id)).length;
    const sceneApplyBusy = busy.has("scene-apply");
    const analyzeStyle = () => action("style-analyze", async () => {
        const draft = await analyzeProjectStyle(project.id, styleAssets.map((asset) => asset.id), true);
        setStyleDraft(draft);
        return draft;
    }, "画风分析已完成，并已自动回填需求与角色中的视觉风格和统一风格圣经");
    const generateMissingCharacters = (mode: "script_style" | "complete_missing") => {
        const taskKey = mode === "script_style" ? "character-batch-script-style" : "character-batch-partial";
        const referenceIds = mode === "complete_missing"
            ? selectedCharacterImageIds
            : [];
        void action(taskKey, () => generateMissingCharacterReferences(project.id, {
            mode,
            userSuggestions: batchCharacterSuggestions,
            referenceAssetIds: referenceIds,
        }), mode === "script_style"
            ? "已重新扫描剧本，并按统一风格为缺失角色生成人物参考图"
            : "已重新扫描剧本，并结合现有人物图补齐其他缺失角色");
    };
    const recommendedCharacterStrategy = (characterId: string, appearanceId = ""): "identity" | "project_style" => {
        const character = project.characters.find((item) => item.id === characterId);
        if (!character) return "project_style";
        const appearance = character.appearance_profiles.find((item) => item.id === appearanceId);
        const referenceIds = appearance ? appearance.reference_asset_ids : character.reference_asset_ids;
        const inheritedFromSeries = referenceIds.some((assetId) => assets.some((asset) => asset.id === assetId && asset.tags.some((tag) => tag.startsWith("series:"))));
        if (project.series_id && !inheritedFromSeries) return "project_style";
        return referenceIds.length > 0 ? "identity" : "project_style";
    };
    const openCharacterAction = (mode: "generate" | "analyze") => {
        const firstCharacterId = project.characters[0]?.id || "";
        setCharacterAction(mode);
        setTargetCharacterId(firstCharacterId);
        setTargetAppearanceId("");
        setCharacterSuggestions("");
        setCharacterReferenceStrategy(recommendedCharacterStrategy(firstCharacterId));
    };
    const submitCharacterAction = () => {
        if (!characterAction) return;
        const character = project.characters.find((item) => item.id === targetCharacterId);
        if (!character) return;
        const selectedImageIds = selectedAssets.filter((asset) => asset.type === "image").map((asset) => asset.id);
        const appearance = character.appearance_profiles.find((item) => item.id === targetAppearanceId);
        const fallbackIds = appearance ? appearance.reference_asset_ids : character.reference_asset_ids;
        const referenceIds = characterAction === "generate" && characterReferenceStrategy === "project_style"
            ? selectedImageIds
            : selectedImageIds.length ? selectedImageIds : fallbackIds;
        const taskKey = `character-${characterAction}-${character.id}-${targetAppearanceId || "base"}`;
        if (busy.has(taskKey)) return;
        setCharacterAction(null);
        if (characterAction === "analyze") {
            void action(taskKey, () => analyzeCharacterReferences(project.id, character.id, referenceIds, { appearanceProfileId: targetAppearanceId || null, appearanceLabel: appearance?.label || "", userSuggestions: characterSuggestions }), `已根据所选参考图回填“${character.name}”角色设定`);
        } else {
            void action(taskKey, () => generateCharacterReferencesWithOptions(project.id, { characterIds: [character.id], referenceAssetIds: referenceIds, appearanceProfileId: targetAppearanceId || null, userSuggestions: characterSuggestions, referenceStrategy: characterReferenceStrategy }), characterReferenceStrategy === "project_style" ? `已按同项目/系列角色画风重做“${character.name}”，旧图仍保留在素材库` : `已根据本角色参考图生成“${character.name}”形象补全图`);
        }
    };
    const characterGenerateCount = Array.from(busy).filter((key) => key.startsWith("character-generate-")).length;
    const characterAnalyzeCount = Array.from(busy).filter((key) => key.startsWith("character-analyze-")).length;
    const targetCharacterTaskKey = characterAction ? `character-${characterAction}-${targetCharacterId}-${targetAppearanceId || "base"}` : "";
    const targetCharacterBusy = !!targetCharacterTaskKey && busy.has(targetCharacterTaskKey);
    const bindingProp = propBindingAssetId ? assets.find((asset) => asset.id === propBindingAssetId && asset.role === "prop") : undefined;
    const openPropBinding = (asset: Asset) => {
        setPropBindingAssetId(asset.id);
        setPropBindingShotIds(shots.filter((shot) => shot.reference_asset_ids.includes(asset.id)).map((shot) => shot.id));
    };
    const savePropBinding = () => {
        if (!bindingProp) return;
        const targetIds = new Set(propBindingShotIds);
        const changedShots = shots.filter((shot) => {
            const shouldInclude = targetIds.has(shot.id);
            return shot.reference_asset_ids.includes(bindingProp.id) !== shouldInclude
                || (shot.keyframe_reference_asset_ids !== null && shot.keyframe_reference_asset_ids.includes(bindingProp.id) !== shouldInclude)
                || (shot.video_reference_asset_ids !== null && shot.video_reference_asset_ids.includes(bindingProp.id) !== shouldInclude);
        });
        const asset = bindingProp;
        setPropBindingAssetId(null);
        void action(`prop-bind-${asset.id}`, () => Promise.all(changedShots.map((shot) => updateShot({
            ...shot,
            reference_asset_ids: targetIds.has(shot.id)
                ? Array.from(new Set([...shot.reference_asset_ids, asset.id]))
                : shot.reference_asset_ids.filter((assetId) => assetId !== asset.id),
            keyframe_reference_asset_ids: shot.keyframe_reference_asset_ids === null
                ? null
                : targetIds.has(shot.id)
                    ? Array.from(new Set([...shot.keyframe_reference_asset_ids, asset.id]))
                    : shot.keyframe_reference_asset_ids.filter((assetId) => assetId !== asset.id),
            video_reference_asset_ids: shot.video_reference_asset_ids === null
                ? null
                : targetIds.has(shot.id)
                    ? Array.from(new Set([...shot.video_reference_asset_ids, asset.id]))
                    : shot.video_reference_asset_ids.filter((assetId) => assetId !== asset.id),
        }))), `已把固定物品“${asset.name}”应用到 ${targetIds.size} 个镜头；首帧、H3 与 Seedance Prompt 已同步`);
    };
    return <div className="grid gap-5 xl:grid-cols-[360px_1fr]">
        <section className="studio-panel h-fit">
            <p className="studio-kicker">REFERENCE LIBRARY</p>
            <h2 className="mb-5 text-xl font-semibold">上传参考素材</h2>
            <Field label="素材用途">
                <select value={role} onChange={(event) => { setRole(event.target.value as AssetRole); setShowAll(false); }}><option value="character">人物形象</option><option value="prop">固定物品 / 道具</option><option value="style">画风截图 / 参考视频</option><option value="scene">场景参考</option><option value="motion">动作/运镜视频</option><option value="voice">声音参考</option><option value="music">背景音乐</option><option value="sound_effect">音效</option><option value="other">其他</option></select>
                <p className="mt-1 text-[11px] text-cyan-100/45">右侧已切换为当前分类，共 {roleAssets.length} 项</p>
            </Field>
            <label className="studio-empty mt-4 min-h-44 cursor-pointer"><input className="hidden" type="file" multiple accept="image/*,video/*,audio/*,.srt,.vtt" disabled={busy.has("upload")} onChange={(event) => { const files = Array.from(event.target.files || []); if (files.length) void upload(files); event.target.value = ""; }} />{busy.has("upload") ? <Loader2 className="animate-spin text-cyan-300" /> : <Upload className="text-cyan-300" />}<strong>批量选择图片、视频或音频</strong><span>可一次多选；全部按上方用途归档</span></label>
            <div className="mt-4 grid gap-2">
                <button className="studio-primary" disabled={busy.has("style-analyze") || styleAssets.length === 0} onClick={() => void analyzeStyle()}>{busy.has("style-analyze") ? <Loader2 className="animate-spin" size={15} /> : <WandSparkles size={15} />}分析并回填 {styleAssets.length} 个画风素材</button>
                <p className="rounded-lg border border-cyan-300/10 bg-cyan-300/[.025] px-3 py-2 text-[11px] leading-5 text-cyan-100/45">完成后会自动保存到“需求与角色”的视觉风格、统一风格圣经和负面提示词；分析详情会立即弹出，并长期显示在本页下方。</p>
                <div className="rounded-xl border border-cyan-300/12 bg-cyan-300/[.025] p-3">
                    <div className="mb-3 flex items-start justify-between gap-2"><div><p className="text-xs font-semibold text-cyan-100/70">缺失人物参考图批量补齐</p><p className="mt-1 text-[11px] leading-5 text-white/35">每次都会先重新扫描剧本角色，再只生成缺图人物；成图统一为“大正脸＋正/侧/背全身”的横向四视图设定板。</p></div><span className="studio-status whitespace-nowrap">{project.characters.length} 角色 · {charactersWithReferences.length} 有图 · {missingCharacterCount} 缺图</span></div>
                    <label className="mb-3 block"><span className="studio-label">本批次生成建议（可选）</span><textarea className="studio-input min-h-24 resize-y" maxLength={4000} value={batchCharacterSuggestions} onChange={(event) => setBatchCharacterSuggestions(event.target.value)} placeholder="例如：所有人都保持电力工装体系；主角更沉稳利落，配角之间的脸型和体型要有明显区分。" /><span className="mt-1 flex justify-between text-[10px] text-white/25"><span>会应用到本批次所有缺图角色</span><span>{batchCharacterSuggestions.length}/4000</span></span></label>
                    <div className="grid gap-2"><button className="studio-secondary justify-start text-left" disabled={characterBatchBusy || !hasProjectStyle || !project.brief.story.trim() || uploadedCharacterImages.length > 0} onClick={() => generateMissingCharacters("script_style")}>{busy.has("character-batch-script-style") ? <Loader2 className="animate-spin" size={15} /> : <Sparkles size={15} />}完全无人设图：按剧本＋统一风格生成</button><button className="studio-secondary justify-start text-left" disabled={characterBatchBusy || !hasProjectStyle || !project.brief.story.trim() || charactersWithReferences.length === 0} onClick={() => generateMissingCharacters("complete_missing")}>{busy.has("character-batch-partial") ? <Loader2 className="animate-spin" size={15} /> : <Users size={15} />}已有部分人设图：参考现有角色补齐其他人</button></div>
                    <p className="mt-2 text-[11px] leading-5 text-white/30">第二种模式会自动使用已绑定人物图；右侧勾选可进一步缩小参考范围。其他角色图只继承画风、比例和服饰语言，不绑定或复制身份。</p>
                    {!hasProjectStyle && <p className="mt-2 rounded border border-amber-300/10 bg-amber-300/[.03] px-2 py-1.5 text-[11px] leading-5 text-amber-100/55">请先上传画风参考图并点击上方“分析并回填”，或者在“需求与角色”填写视觉风格。</p>}
                    {unboundCharacterImageCount > 0 && <p className="mt-2 rounded border border-amber-300/10 bg-amber-300/[.03] px-2 py-1.5 text-[11px] leading-5 text-amber-100/55">还有 {unboundCharacterImageCount} 张客户人物图未绑定角色。请先到“需求与角色 → 基础形象参考图”勾到对应人物，再使用第二种模式。</p>}
                </div>
                <button className="studio-secondary" disabled={project.characters.length === 0} onClick={() => openCharacterAction("generate")}>{characterGenerateCount > 0 ? <Loader2 className="animate-spin" size={15} /> : <Users size={15} />}单个角色生成 / 补全{characterGenerateCount > 0 ? ` · 后台 ${characterGenerateCount} 项` : ""}</button>
                <button className="studio-secondary" disabled={project.characters.length === 0 || selectedAssets.filter((asset) => asset.type === "image").length === 0} onClick={() => openCharacterAction("analyze")}>{characterAnalyzeCount > 0 ? <Loader2 className="animate-spin" size={15} /> : <WandSparkles size={15} />}按所选图 AI 回填角色{characterAnalyzeCount > 0 ? ` · 后台 ${characterAnalyzeCount} 项` : ""}</button>
                <button className="studio-secondary" disabled={busy.has("scene-profiles") || shots.length === 0} onClick={() => { setSceneHelpOpen(true); void action("scene-profiles", () => generateSceneProfiles(project.id), "已识别可选场景文字档案；可继续逐个生成母版图"); }}>{busy.has("scene-profiles") ? <Loader2 className="animate-spin" size={15} /> : <ImageIcon size={15} />}AI 识别场景文字档案</button>
            </div>
            <p className="mt-4 text-xs leading-5 text-white/35">{shots.length === 0 ? "请先生成或导入分镜。场景档案需要从已有分镜中归纳重复空间。" : "第一步识别空间文字规范；第二步生成并修正场景母版；最后在下方一键应用到所有已归类镜头，不需要逐镜勾选。"}</p>
        </section>
        <section className="studio-panel"><div className="mb-5 flex flex-wrap items-center justify-between gap-3"><div><p className="studio-kicker">{visibleAssets.length} / {assets.length} ASSETS</p><h2 className="text-xl font-semibold">项目素材库 · {assetRoleLabel(role)}</h2><p className="mt-1 text-xs text-white/35">人物、场景和固定物品都可成为逐镜参考；物品卡片可批量设置出现范围。已选 {selectedAssetIds.length} 项。</p></div><button className="studio-secondary" onClick={() => setShowAll(!showAll)}>{showAll ? "只看当前分类" : "显示全部素材"}</button></div>{visibleAssets.length === 0 ? <div className="studio-empty"><ImageIcon className="text-cyan-300" /><strong>当前分类还没有素材</strong><span>左侧上传后会立即出现在这里</span></div> : <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{visibleAssets.map((asset) => <div key={asset.id} className="min-w-0"><AssetCard asset={asset} projectId={project.id} selected={selectedAssetIds.includes(asset.id)} select={() => setSelectedAssetIds(toggle(selectedAssetIds, asset.id))} preview={() => setPreview({ name: asset.name, url: projectDownloadUrl(project.id, "asset", asset.id), description: asset.description })} remove={() => action(`asset-${asset.id}`, () => deleteProjectAsset(project.id, asset.id), "素材记录已删除")} />{asset.role === "prop" && <button type="button" className="studio-secondary mt-2 w-full text-xs" onClick={() => openPropBinding(asset)}><LayoutList size={13} />已绑定 {shots.filter((shot) => shot.reference_asset_ids.includes(asset.id)).length} 镜 · 设置出现范围</button>}</div>)}</div>}</section>
        {(project.style_profile || project.scene_profiles.length > 0) && <section className="studio-panel xl:col-span-2">
            {project.style_profile && <div className="mb-5 rounded-xl border border-cyan-300/12 bg-cyan-300/[.025] p-4"><div className="flex flex-wrap items-start justify-between gap-2"><div><p className="studio-kicker">ACTIVE STYLE · 已自动回填</p><strong>{project.style_profile.name}</strong></div><span className="studio-status text-emerald-200">项目统一风格生效中</span></div><p className="mt-2 text-sm leading-6 text-white/55">{project.style_profile.analysis_summary}</p><div className="mt-3 grid gap-2 text-xs text-white/40 md:grid-cols-2"><p><span className="text-white/60">媒介 / 质感：</span>{project.style_profile.medium}；{project.style_profile.texture}</p><p><span className="text-white/60">色彩 / 光影：</span>{project.style_profile.palette}；{project.style_profile.lighting}</p><p><span className="text-white/60">镜头 / 构图：</span>{project.style_profile.camera_language}；{project.style_profile.composition}</p><p><span className="text-white/60">动作 / 剪辑：</span>{project.style_profile.motion_language}</p></div><p className="mt-3 text-xs leading-5 text-rose-100/45"><span className="text-rose-100/65">避免项：</span>{project.style_profile.negative_constraints}</p></div>}
            <div className="mb-4 flex flex-wrap items-start justify-between gap-3"><div><p className="studio-kicker">SCENE CONTINUITY ASSETS</p><h3 className="text-lg font-semibold">场景档案（文字规范）与场景母版图</h3><p className="mt-1 text-xs leading-5 text-white/38">修正档案和母版后，可按已经识别的镜头归属一次性同步首帧与 Seedance Prompt；不会调用图片或 H3 接口。</p></div><button className="studio-primary shrink-0" disabled={sceneApplyBusy || mappedSceneShotCount === 0} onClick={() => void action("scene-apply", () => applySceneProfiles(project.id), `已把 ${project.scene_profiles.length} 个场景档案批量应用到 ${mappedSceneShotCount} 个镜头；原 H3 和已有图片均已保留`)}>{sceneApplyBusy ? <Loader2 className="animate-spin" size={15} /> : <WandSparkles size={15} />}{appliedSceneShotCount === mappedSceneShotCount && mappedSceneShotCount > 0 ? `重新同步 ${mappedSceneShotCount} 镜` : `一键应用到 ${mappedSceneShotCount} 镜`}</button></div>
            {mappedSceneShotCount > 0 && <p className="mb-4 rounded-lg border border-cyan-300/10 bg-cyan-300/[.025] px-3 py-2 text-[11px] leading-5 text-cyan-100/50">当前已应用 {appliedSceneShotCount}/{mappedSceneShotCount} 镜。场景档案负责本镜环境、光线与固定陈设；项目统一风格只提供绘制媒介。已有首帧图保留，可随后只重生画面确实有问题的镜头。</p>}
            <div className="grid gap-4 lg:grid-cols-2">{project.scene_profiles.length === 0 ? <p className="text-sm text-white/35">没有需要固定的重复场景。</p> : project.scene_profiles.map((profile) => <SceneProfileCard key={profile.id} projectId={project.id} profile={profile} assets={assets} ordinals={shots.filter((shot) => profile.source_shot_ids.includes(shot.id)).map((shot) => shot.ordinal)} busy={busy} action={action} preview={(asset) => setPreview({ name: asset.name, url: projectInlineUrl(project.id, "asset", asset.id), description: asset.description })} />)}</div>
        </section>}
        {styleDraft && <StyleAnalysisResultDialog draft={styleDraft} onClose={() => setStyleDraft(null)} />}
        {characterAction && <div className="studio-modal" onMouseDown={() => setCharacterAction(null)}>
            <section className="studio-dialog max-w-2xl" onMouseDown={(event) => event.stopPropagation()}>
                <p className="studio-kicker">CHARACTER REFERENCES</p>
                <h2 className="text-2xl font-semibold">{characterAction === "analyze" ? "按参考图 AI 回填角色设定" : "按参考图生成 / 补全人物形象"}</h2>
                <p className="mt-2 text-sm leading-6 text-white/45">{characterAction === "generate" ? "生成横向四视图设定板：左侧大正脸，右侧为同一角色的全身正面、侧面和背面。可以沿用本角色身份，也可以忽略错误旧图、按同系列已确认角色画风重做。" : "优先使用右侧已勾选的图片；未勾选时使用角色设定中已绑定的基础或时期参考图。"}提交后会转入后台，可马上继续选择其他角色或时期。</p>
                <div className="mt-5 grid gap-4 sm:grid-cols-2"><Field label="目标角色"><select value={targetCharacterId} onChange={(event) => { const nextId = event.target.value; setTargetCharacterId(nextId); setTargetAppearanceId(""); if (characterAction === "generate") setCharacterReferenceStrategy(recommendedCharacterStrategy(nextId)); }}>{project.characters.map((character) => <option key={character.id} value={character.id}>{character.name}</option>)}</select></Field><Field label="目标时期形象"><select value={targetAppearanceId} onChange={(event) => { const nextAppearanceId = event.target.value; setTargetAppearanceId(nextAppearanceId); if (characterAction === "generate") setCharacterReferenceStrategy(recommendedCharacterStrategy(targetCharacterId, nextAppearanceId)); }}><option value="">基础形象</option>{project.characters.find((item) => item.id === targetCharacterId)?.appearance_profiles.map((appearance) => <option key={appearance.id} value={appearance.id}>{appearance.label}</option>)}</select></Field></div>
                {characterAction === "generate" && <div className="mt-4"><span className="studio-label">参考策略</span><div className="mt-2 grid gap-2 sm:grid-cols-2"><label className={`cursor-pointer rounded-lg border p-3 text-sm ${characterReferenceStrategy === "identity" ? "border-cyan-300/35 bg-cyan-300/[.06]" : "border-white/8 bg-black/15"}`}><span className="flex items-center gap-2 font-medium"><input type="radio" name="character-reference-strategy" checked={characterReferenceStrategy === "identity"} onChange={() => setCharacterReferenceStrategy("identity")} />沿用本角色身份</span><span className="mt-1 block text-xs leading-5 text-white/40">使用右侧所选图，未选择时沿用该角色已绑定图；适合补角度、微调服装。</span></label><label className={`cursor-pointer rounded-lg border p-3 text-sm ${characterReferenceStrategy === "project_style" ? "border-cyan-300/35 bg-cyan-300/[.06]" : "border-white/8 bg-black/15"}`}><span className="flex items-center gap-2 font-medium"><input type="radio" name="character-reference-strategy" checked={characterReferenceStrategy === "project_style"} onChange={() => setCharacterReferenceStrategy("project_style")} />按同系列画风重做</span><span className="mt-1 block text-xs leading-5 text-white/40">忽略本角色旧图，优先参考本项目/系列其他已确认角色；成功后新图成为绑定图，旧图仍留在素材库。</span></label></div></div>}
                <label className="mt-4 block"><span className="studio-label">{characterAction === "generate" ? "生成建议（可选）" : "回填建议（可选）"}</span><textarea className="studio-input min-h-32 resize-y" maxLength={4000} value={characterSuggestions} onChange={(event) => setCharacterSuggestions(event.target.value)} placeholder={characterAction === "generate" ? "例如：保留参考图脸型与五官；眼神更锐利，衣料增加磨损细节。四视图排版由系统自动保持。" : "例如：识别为 18 岁偏瘦时期，不要沿用盛年妆造。"} /><span className="mt-1 block text-right text-[10px] text-white/25">{characterSuggestions.length}/4000</span></label>
                <div className="mt-4 rounded-lg border border-cyan-300/10 bg-cyan-300/[.03] p-3 text-xs leading-5 text-cyan-100/55">{targetCharacterBusy ? "当前这个角色/时期已经在后台处理中；可切换到其他角色继续提交。" : <>本次右侧已选 {selectedAssets.filter((asset) => asset.type === "image").length} 张图片。{characterAction === "analyze" && selectedAssets.filter((asset) => asset.type === "image").length === 0 ? "回填至少需要一张参考图。" : characterAction === "generate" && characterReferenceStrategy === "project_style" ? "系统会把其他已确认角色图置于普通风格图之前，并明确排除真人照片与 3D 写实人物。" : "生成会同时结合项目剧本和统一视觉风格。"}</>}</div>
                <div className="mt-5 flex justify-end gap-2"><button className="studio-secondary" onClick={() => setCharacterAction(null)}>取消</button><button className="studio-primary" disabled={targetCharacterBusy || !targetCharacterId || (characterAction === "analyze" && selectedAssets.filter((asset) => asset.type === "image").length === 0)} onClick={submitCharacterAction}>{targetCharacterBusy ? <Loader2 className="animate-spin" size={15} /> : <WandSparkles size={15} />}{targetCharacterBusy ? "此形象处理中" : characterAction === "analyze" ? "分析并回填" : "生成四视图"}</button></div>
            </section>
        </div>}
        {sceneHelpOpen && <div className="studio-modal" onMouseDown={() => setSceneHelpOpen(false)}><section className="studio-dialog max-w-2xl" onMouseDown={(event) => event.stopPropagation()}><p className="studio-kicker">SCENE CONTINUITY</p><h2 className="text-2xl font-semibold">场景文字档案与母版图怎么用</h2><ol className="mt-4 list-decimal space-y-3 pl-5 text-sm leading-6 text-white/55"><li>“AI 识别场景文字档案”只分析分镜并生成空间说明，不会调用图片模型，所以这一步得到的是文字。</li><li>文字档案会立即显示在页面下方；点击对应档案的“生成母版图”，才会生成一张无人物环境图并直接显示预览。</li><li>修正所有档案后，点击“一键应用到全部镜头”，系统会按识别好的归属批量启用并重编译本地 Prompt。</li><li>生成首帧时会同时参考对应母版：镜头角度可以变化，房间布局、陈设、材质和本场景光线会保持。</li></ol><p className="mt-4 rounded-lg border border-amber-300/10 bg-amber-300/[.035] p-3 text-xs leading-5 text-amber-100/55">批量应用不会调用生图或 H3 接口，也不会删除已有结果。旧首帧如果已经画出了错误混光，可到视频生成页只选择这些镜头重生。</p><div className="mt-5 flex justify-end"><button className="studio-primary" onClick={() => setSceneHelpOpen(false)}>知道了</button></div></section></div>}
        {bindingProp && <div className="studio-modal" role="dialog" aria-modal="true" aria-labelledby="prop-binding-title" onMouseDown={() => setPropBindingAssetId(null)}><section className="studio-dialog my-0 flex max-h-[calc(100dvh-2rem)] min-h-0 max-w-3xl flex-col overflow-hidden" onMouseDown={(event) => event.stopPropagation()}><div className="shrink-0"><p className="studio-kicker">FIXED PROP CONTINUITY</p><div className="mb-4 flex items-start justify-between gap-3"><div><h2 id="prop-binding-title" className="text-2xl font-semibold">设置“{bindingProp.name}”出现范围</h2><p className="mt-1 text-sm leading-6 text-white/45">勾选后，该物品图会同时进入本镜首帧、Seedance 全模态参考和 H3 R2V 参考，并把物品尺寸约束写进 Prompt。</p></div><button type="button" className="rounded-lg p-2 text-white/35 hover:bg-white/8 hover:text-white" aria-label="关闭" onClick={() => setPropBindingAssetId(null)}><X size={20} /></button></div><div className="mb-3 flex flex-wrap gap-2"><button type="button" className="studio-secondary text-xs" onClick={() => setPropBindingShotIds(shots.map((shot) => shot.id))}>全选</button><button type="button" className="studio-secondary text-xs" onClick={() => setPropBindingShotIds([])}>清空</button><span className="self-center text-xs text-white/35">已选 {propBindingShotIds.length}/{shots.length} 镜</span></div></div><div className="min-h-0 flex-1 space-y-1 overflow-y-auto rounded-xl border border-white/8 bg-black/15 p-2">{shots.map((shot) => <label key={shot.id} className="flex cursor-pointer items-start gap-3 rounded-lg px-3 py-2.5 hover:bg-white/[.035]"><input className="mt-1" type="checkbox" checked={propBindingShotIds.includes(shot.id)} onChange={() => setPropBindingShotIds(toggle(propBindingShotIds, shot.id))} /><span className="min-w-0"><strong className="block text-sm">镜头 {shot.ordinal} · {shot.title.replace(/^镜头\s*\d+\s*[·.:：-]?\s*/, "")}</strong><span className="mt-0.5 line-clamp-2 block text-xs leading-5 text-white/35">{shot.scene_description || shot.narrative}</span></span></label>)}</div><div className="mt-5 flex shrink-0 flex-wrap items-center justify-between gap-3 border-t border-white/8 pt-4"><p className="text-xs text-white/30">只更新素材绑定和本地 Prompt，不会自动调用生图或视频接口。</p><div className="flex gap-2"><button type="button" className="studio-secondary" onClick={() => setPropBindingAssetId(null)}>取消</button><button type="button" className="studio-primary" disabled={busy.has(`prop-bind-${bindingProp.id}`)} onClick={savePropBinding}><Save size={15} />保存出现范围</button></div></div></section></div>}
        {preview && <MediaPreview preview={preview} onClose={() => setPreview(null)} />}
    </div>;
}

type H3Preset = "fast" | "balanced" | "quality";
type KeyframeEditorState = {
    shotId: string;
    prompt: string;
    suggestions: string;
    revisionMode: "fresh" | "iterate";
    referenceAssetIds: string[] | null;
};
type MediaPreviewState = { name: string; url: string; description?: string };
type VideoPreviewState = { name: string; url: string; downloadUrl: string; provider: string };

const H3_SCHEDULERS: Shot["h3_scheduler"][] = ["simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic", "kl_optimal"];
const H3_MODEL_PROFILE_LABELS: Record<Shot["h3_model_profile"], string> = {
    default: "沿用模型设置中心",
    pruned_int8: "剪枝 INT8 · 21GB（当前基线）",
    pruned_fp8: "剪枝 FP8 · 21GB（A/B 实验）",
    full_int8: "完整 INT8 · 34GB（CPU offload）",
    pruned_bf16: "剪枝 BF16 · 40.2GB（高精度实验）",
    full_bf16: "完整 BF16 · 66.3GB（多卡/重度 offload）",
};
const H3_TEXT_ENCODER_LABELS: Record<Shot["h3_text_encoder_profile"], string> = {
    default: "沿用模型设置中心",
    nvfp4: "NVFP4 · 15.7GB（当前基线）",
    int8: "INT8 · 27.1GB（指令增强实验）",
    bf16: "BF16 · 51.5GB（多卡/重度 offload）",
};

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
        h3_steps: preset === "quality" ? 25 : 4,
        h3_scheduler: "simple",
        h3_denoise: 1,
        h3_lora_strength: 1,
        h3_low_vram: false,
        h3_shift_video: 12,
        h3_shift_audio: 3,
    };
}

function ShotKeyframeSummary({ shot, keyframe, projectId, generating, edit, preview }: { shot: Shot; keyframe?: Asset; projectId: string; generating: boolean; edit: () => void; preview: () => void }) {
    const prompt = shot.keyframe_prompt || keyframe?.description || shot.scene_description || shot.visual_prompt;
    return <div className="overflow-hidden rounded-xl border border-white/8 bg-black/20">
        <div className="flex aspect-video items-center justify-center overflow-hidden bg-black/40">
            {keyframe ? <button type="button" className="group relative h-full w-full cursor-zoom-in" onClick={preview}><img className="h-full w-full object-cover transition-transform group-hover:scale-[1.02]" src={projectDownloadUrl(projectId, "asset", keyframe.id)} alt={`镜头 ${shot.ordinal} 首帧`} /><span className="absolute bottom-2 right-2 rounded-md bg-black/70 p-2 text-white/70 opacity-0 group-hover:opacity-100"><Maximize2 size={14} /></span></button> : <div className="flex flex-col items-center gap-2 text-white/25"><ImageIcon size={28} /><span className="text-xs">尚未生成首帧</span></div>}
        </div>
        <div className="p-3"><p className="studio-kicker">KEYFRAME / FIRST FRAME</p><p className="mt-1 line-clamp-3 text-[11px] leading-5 text-white/38">{prompt || "尚未填写首帧 Prompt"}</p>{shot.keyframe_revision_suggestion_draft && <div className="mt-2 rounded-md border border-amber-300/15 bg-amber-300/[.04] px-2.5 py-2 text-[11px] leading-5 text-amber-100/65">上次失败建议已缓存：{shot.keyframe_revision_suggestion_draft}</div>}<div className="mt-3 grid grid-cols-2 gap-2">{keyframe && <button type="button" className="studio-secondary px-2 py-2 text-xs" onClick={preview}><Maximize2 size={13} />预览</button>}<button type="button" className={`${keyframe ? "studio-secondary" : "studio-primary"} px-2 py-2 text-xs`} disabled={generating} onClick={edit}>{generating ? <Loader2 className="animate-spin" size={13} /> : <Sparkles size={13} />}{generating ? "生成中…" : keyframe ? "重生首帧" : "生成首帧"}</button></div></div>
    </div>;
}

type ShotCardMediaProps = {
    projectId: string;
    keyframe?: Asset;
    generatingKeyframe: boolean;
    editKeyframe: () => void;
    previewKeyframe: () => void;
    submit: (shot: Shot) => Promise<void>;
};

function H3ShotCard({ shot, project, assets, checked, busy, directorVersion, toggleChecked, save, projectId, keyframe, generatingKeyframe, editKeyframe, previewKeyframe, submit }: { shot: Shot; project: Project; assets: Asset[]; checked: boolean; busy: boolean; directorVersion: string; toggleChecked: () => void; save: (shot: Shot) => Promise<void> } & ShotCardMediaProps) {
    const [draft, setDraft] = useState(shot);
    const promptNeedsReview = !!shot.h3_prompt_skill_output.trim() && (
        shot.h3_prompt_source_revision !== shot.content_revision
        || (!!directorVersion && shot.h3_director_version !== directorVersion)
    );
    const width = draft.h3_width || project.brief.width;
    const height = draft.h3_height || project.brief.height;
    const baseline = Math.max(1, project.brief.width * project.brief.height * 124 * 4);
    const relativeWork = width * height * draft.render_frames * draft.h3_steps / baseline;
    const patch = (values: Partial<Shot>) => setDraft((current) => ({ ...current, ...values }));
    const applyPreset = (preset: H3Preset) => patch(h3PresetPatch(preset, project));
    const dialogueGuidedR2v = !!draft.dialogue.trim() && !!(draft.keyframe_asset_id || draft.image_path || draft.reference_asset_ids.length);
    const effectiveMode = draft.generation_mode === "auto"
        ? (dialogueGuidedR2v ? "r2v" : draft.keyframe_asset_id || draft.image_path ? "i2v" : draft.reference_asset_ids.length ? "r2v" : "i2v")
        : draft.generation_mode;
    const imageReferenceCount = draft.reference_asset_ids.filter((assetId) => assets.some((asset) => asset.id === assetId && asset.type === "image")).length;
    const tuningWarnings = [
        draft.generation_mode === "r2v" && !!(draft.keyframe_asset_id || draft.image_path) ? "R2V 会把完整分镜图作为 Picture 1 构图锚点，但它不是硬锁首帧；对白镜头会同时使用干净 TTS 参考音频驱动指定说话人。" : "",
        effectiveMode === "r2v" && imageReferenceCount > 1 && !(draft.keyframe_asset_id || draft.image_path) ? `当前 R2V 只有 ${imageReferenceCount} 张独立人物图，没有完整群像构图锚点；建议先生成分镜首帧。` : "",
        effectiveMode === "i2v" && !(draft.keyframe_asset_id || draft.image_path) ? "当前会走 I2V，但还没有首帧；请先点“生成所选首帧”。" : "",
        draft.ref_image_size === "max" ? "参考图尺寸为 max；会明显增加参考编码开销。官方模板默认 match，除非强制 R2V 且身份仍不稳，否则建议 match。" : "",
        draft.h3_turbo && draft.h3_steps !== 4 ? `当前启用了 4-step Turbo，但步数是 ${draft.h3_steps}；LoRA 与步数不匹配会让画面发糊、过锐或动作不稳。` : "",
        !draft.h3_turbo && draft.h3_steps < 20 ? `当前走官方原生采样，但只有 ${draft.h3_steps} 步；最终成片至少 20 步，本工作台清晰优先使用 25 步。` : "",
        draft.h3_scheduler !== "simple" ? `当前 Scheduler 是 ${draft.h3_scheduler}；4-step Turbo 建议使用 simple。` : "",
        draft.h3_denoise !== 1 ? `Denoise 当前为 ${draft.h3_denoise}；低于 1 可能出现去噪不足、画面灰糊或动作偏弱。` : "",
        draft.h3_turbo && draft.h3_lora_strength !== 1 ? `Turbo LoRA 强度当前为 ${draft.h3_lora_strength}；匹配值是 1.0。` : "",
        draft.h3_low_vram ? "低显存是旧版自定义链路参数；新版官方采样链路不会提交它。RTX 5090 32GB 也不需要开启。" : "",
        ["full_int8", "pruned_bf16", "full_bf16"].includes(draft.h3_model_profile) ? "所选 diffusion 权重超过 RTX 5090 32GB 显存，需要 ComfyUI CPU offload；会变慢，但这是验证剪枝/量化是否影响质量的正确 A/B 路径。" : "",
        ["int8", "bf16"].includes(draft.h3_text_encoder_profile) ? "更高精度文本编码器主要影响复杂指令、角色关系和对白归属，不等于自动修复口型；需要先在服务器安装对应文件。" : "",
        width * height < 900_000 ? `当前仅 ${(width * height / 1_000_000).toFixed(2)}MP，适合预览；最终成片建议使用 1344×768（约 0.98MP）。` : "",
    ].filter(Boolean);
    return <article className="rounded-xl border border-white/8 bg-black/15 p-4">
        <div className="flex flex-wrap items-start gap-3">
            <input type="checkbox" checked={checked} onChange={toggleChecked} />
            <div className="min-w-48 flex-1"><strong>#{shot.ordinal} {shot.title}</strong><p className="line-clamp-1 text-xs text-white/35">{shot.video_prompt}</p></div>
            <span className="rounded bg-white/[.05] px-2 py-1 text-[10px] text-white/45" title={draft.h3_prompt_skill_version ? `Skill version ${draft.h3_prompt_skill_version}` : "尚未通过风格 Skill 重新生成"}>{draft.h3_prompt_skill_id || "h3-prompt-writing"}{draft.h3_prompt_skill_version ? ` · v${draft.h3_prompt_skill_version}` : ""}</span>
            <span className="studio-status uppercase">{draft.generation_mode === "auto" ? `AUTO→${effectiveMode}` : effectiveMode}</span>
            <span className="rounded bg-cyan-300/8 px-2 py-1 font-mono text-[11px] text-cyan-100/70">{width}×{height} · {draft.render_frames}帧 · {draft.h3_turbo ? "Turbo" : "原生"}{draft.h3_steps}步 · 约 {relativeWork.toFixed(1)}×算力</span>
        </div>
        <div className="mt-4 grid gap-4 xl:grid-cols-[360px_minmax(0,1fr)]"><ShotKeyframeSummary shot={shot} keyframe={keyframe} projectId={projectId} generating={generatingKeyframe} edit={editKeyframe} preview={previewKeyframe} /><label><span className="studio-label">MiniMax H3 Prompt</span><textarea className="studio-input min-h-64 resize-y" value={draft.video_prompt} onChange={(event) => patch({ video_prompt: event.target.value })} /><span className="mt-1.5 block text-[11px] text-white/32">首帧与本镜 H3 Prompt 放在同一卡片；提交本镜前可一起核对。</span></label></div>
        {promptNeedsReview && <p className="mt-3 rounded-lg border border-amber-300/20 bg-amber-300/[.04] px-3 py-2.5 text-xs leading-5 text-amber-100/80">H3 原文已保留 · 剧情、参考素材或导演版本已变更，待核对。这里仍是之前保存的 Prompt，不是默认模板；请检查旧素材编号与当前首帧/参考图是否一致，需要时再重新生成。</p>}
        {tuningWarnings.length > 0 && <div className="mt-3 rounded-lg border border-amber-300/15 bg-amber-300/[.04] px-3 py-2.5"><strong className="text-xs text-amber-100/80">当前参数有 {tuningWarnings.length} 项画质提醒</strong><ul className="mt-1 list-disc space-y-1 pl-4 text-[11px] leading-5 text-amber-50/55">{tuningWarnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></div>}
        <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <Field label="速度/质量预设" hint="快速和均衡使用官方 4 步加速；清晰优先改用原生 25 步并关闭 Turbo LoRA，给脸、手、服装和运动更多收敛空间。"><select defaultValue="custom" onChange={(event) => event.target.value !== "custom" && applyPreset(event.target.value as H3Preset)}><option value="custom">当前/自定义</option><option value="fast">快速预览 · 608长边 / Turbo 4步</option><option value="balanced">均衡 · 项目分辨率 / Turbo 4步</option><option value="quality">清晰优先 · 1344长边 / 原生25步</option></select></Field>
            <Field label="生成宽度" hint="决定横向细节；越高越清晰，也越慢、越占显存。必须是 32 的倍数。"><input type="number" min={32} max={4096} step={32} value={width} onChange={(event) => patch({ h3_width: Number(event.target.value) || null })} /></Field>
            <Field label="生成高度" hint="决定纵向细节；需与宽度保持目标画幅。16:9 成片推荐 1344×768。"><input type="number" min={32} max={4096} step={32} value={height} onChange={(event) => patch({ h3_height: Number(event.target.value) || null })} /></Field>
            <Field label="采样质量链路" hint="Turbo 4步适合预览；原生链路绕过加速 LoRA。本工作台最终成片使用 25 步，通常比 20 步更稳但约慢 25%。"><select value={draft.h3_turbo ? "turbo" : "native"} onChange={(event) => patch(event.target.value === "turbo" ? { h3_turbo: true, h3_steps: 4 } : { h3_turbo: false, h3_steps: 25, h3_lora_strength: 1 })}><option value="turbo">Turbo · 4步快速预览</option><option value="native">原生 · 25步质量优先</option></select></Field>
            <Field label="H3 Diffusion 权重" hint="任务提交时会冻结真实文件名。先用同一首帧/Prompt/Seed 对比完整 INT8 或剪枝 BF16，才能判断当前变形是否来自剪枝与量化。"><select value={draft.h3_model_profile} onChange={(event) => patch({ h3_model_profile: event.target.value as Shot["h3_model_profile"] })}>{Object.entries(H3_MODEL_PROFILE_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></Field>
            <Field label="H3 文本编码器" hint="决定模型理解人物、动作、镜头和对白关系的能力。当前 NVFP4 最省显存；多人关系错误可用 INT8 做受控对比。"><select value={draft.h3_text_encoder_profile} onChange={(event) => patch({ h3_text_encoder_profile: event.target.value as Shot["h3_text_encoder_profile"] })}>{Object.entries(H3_TEXT_ENCODER_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></Field>
            <Field label="采样步数" hint={draft.h3_turbo ? "Turbo LoRA 与 4 步成套匹配，请保持 4。" : "原生链路 20 步是官方基线；25 步会更慢，但复杂人物和动作通常有更多收敛机会。"}><input type="number" min={1} max={100} value={draft.h3_steps} onChange={(event) => patch({ h3_steps: Math.max(1, Number(event.target.value) || 1) })} /></Field>
            <Field label="H3 时长（5–15秒）" hint="越长帧数越多、生成越慢，也更容易人物漂移。复杂动作建议拆成 5–8 秒小镜头。"><input type="number" min={5} max={15} step={0.25} value={draft.duration_seconds} onChange={(event) => { const duration = Math.max(5, Math.min(15, Number(event.target.value) || 5)); patch({ duration_seconds: duration, render_frames: h3FramesForSeconds(duration) }); }} /></Field>
            <Field label="固定 Seed（留空则随机）" hint="固定后可用同一构图比较不同参数；换 Seed 会改变人物姿态、构图和细节，但不代表质量一定更高。"><input type="number" min={1} max={2147483647} value={draft.h3_seed ?? ""} placeholder="每次随机" onChange={(event) => patch({ h3_seed: event.target.value ? Number(event.target.value) : null })} /></Field>
            <Field label="完整分镜图 / 构图锚点" hint="无对白时作为 I2V 硬首帧；对白 R2V 时自动成为 Picture 1，保留群像构图，同时让干净 TTS 驱动指定说话人。"><select value={draft.keyframe_asset_id || ""} onChange={(event) => patch({ keyframe_asset_id: event.target.value || null })}><option value="">未绑定</option>{assets.filter((asset) => asset.type === "image").map((asset) => <option key={asset.id} value={asset.id}>{asset.name}</option>)}</select></Field>
            <div className="md:col-span-2"><span className="studio-label">全能参考（R2V，可多选）</span><div className="max-h-28 space-y-1 overflow-y-auto rounded-lg border border-white/8 p-2">{assets.filter((asset) => ["image", "video", "audio"].includes(asset.type)).map((asset) => <label key={asset.id} className="flex items-center gap-2 text-xs text-white/60"><input type="checkbox" checked={draft.reference_asset_ids.includes(asset.id)} onChange={() => patch({ reference_asset_ids: toggle(draft.reference_asset_ids, asset.id) })} />{asset.type.toUpperCase()} · {asset.name}</label>)}</div><p className="mt-1.5 text-[11px] leading-5 text-white/32">只绑定本镜真正出现的人物、动作或声音。无关或互相冲突的参考越多，模型越容易混脸、串衣服、构图失控。</p></div>
        </div>
        <details className="mt-4 rounded-lg border border-white/8 bg-white/[.02] p-3">
            <summary className="cursor-pointer text-xs font-semibold text-white/55">高级参数（已对齐当前官方 ComfyUI H3 模板）</summary>
            <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                <Field label="Sampler / Scheduler" hint="实际 Sampler 固定为官方 res_multistep；这里控制噪声日程。官方模板使用 simple，其他选项可能改变对比度、锐度与运动稳定性。"><select value={draft.h3_scheduler} onChange={(event) => patch({ h3_scheduler: event.target.value as Shot["h3_scheduler"] })}>{H3_SCHEDULERS.map((scheduler) => <option key={scheduler} value={scheduler}>{scheduler}{scheduler === "simple" ? " · 官方默认" : " · 实验"}</option>)}</select></Field>
                <Field label="Denoise（去噪强度）" hint="1.0 是完整生成。调低会减少变化，但在这套工作流中容易去噪不足、画面灰糊或动作弱；建议保持 1。"><input type="number" min={0} max={1} step={0.01} value={draft.h3_denoise} onChange={(event) => patch({ h3_denoise: Number(event.target.value) })} /></Field>
                <Field label="Turbo LoRA 强度" hint={draft.h3_turbo ? "只在 Turbo 链路生效。1.0 是 4 步模型匹配值；偏离会产生未收敛、锐化、纹理噪点或重影。" : "原生链路已绕过 Turbo LoRA，因此本项不参与生成。"}><input disabled={!draft.h3_turbo} type="number" min={-10} max={10} step={0.05} value={draft.h3_lora_strength} onChange={(event) => patch({ h3_lora_strength: Number(event.target.value) })} /></Field>
                <Field label="参考图尺寸" hint="只影响 R2V。match 是官方模板默认值；max 保留更大参考编码，但不修复多人重影，反而更慢、更占显存。"><select value={draft.ref_image_size} onChange={(event) => patch({ ref_image_size: event.target.value })}><option value="match">match · 官方默认</option><option value="max">max · 高显存实验</option></select></Field>
            </div>
            <p className="mt-3 text-[11px] leading-5 text-amber-100/55">最终成片推荐：原生 25 步 + res_multistep + simple + Denoise 1 + 1344×768 + 清晰完整构图锚点。对白镜头会把同一条干净 TTS 同时交给 H3 和最终音轨，以对齐口型并消除原生波浪杂音。</p>
        </details>
        <div className="mt-4 flex flex-wrap justify-end gap-2"><button className="studio-secondary" disabled={busy} onClick={() => void save(draft)}>{busy ? <Loader2 className="animate-spin" size={14} /> : <Save size={14} />}保存本镜参数</button><button className="studio-primary" disabled={busy || !draft.video_prompt.trim()} onClick={() => void submit(draft)}>{busy ? <Loader2 className="animate-spin" size={14} /> : <Play size={14} />}提交本镜</button></div>
    </article>;
}

function SeedanceShotCard({ shot, project, assets, checked, busy, toggleChecked, save, projectId, keyframe, generatingKeyframe, editKeyframe, previewKeyframe, submit, canSubmit }: { shot: Shot; project: Project; assets: Asset[]; checked: boolean; busy: boolean; toggleChecked: () => void; save: (shot: Shot) => Promise<void>; canSubmit: boolean } & ShotCardMediaProps) {
    const [draft, setDraft] = useState(shot);
    const [diagnostics, setDiagnostics] = useState<SeedanceMaterialDiagnostics | null>(null);
    useEffect(() => {
        let cancelled = false;
        void getSeedanceMaterials(projectId, shot.id).then((result) => {
            if (!cancelled) setDiagnostics(result);
        }).catch(() => {
            if (!cancelled) setDiagnostics(null);
        });
        return () => { cancelled = true; };
    }, [projectId, shot.id, shot.updated_at, shot.seedance_reference_mode]);
    const assetIds = [shot.keyframe_asset_id, ...shot.reference_asset_ids].filter((value): value is string => !!value);
    const refs = [...new Set(assetIds)].map((id) => assets.find((asset) => asset.id === id)).filter((asset): asset is Asset => !!asset && ["image", "video", "audio"].includes(asset.type));
    const availableMaterials = diagnostics?.available_materials || assets.filter((asset) => ["image", "video", "audio"].includes(asset.type) && asset.role !== "output");
    const effectiveVideoReferenceIds = draft.video_reference_asset_ids ?? diagnostics?.materials.map((item) => item.id) ?? refs.map((item) => item.id);
    const manuallySelectedMaterials = effectiveVideoReferenceIds.map((id) => availableMaterials.find((item) => item.id === id) || assets.find((item) => item.id === id)).filter((item) => !!item);
    const resolvedMaterials = draft.video_reference_asset_ids === null || draft.video_reference_asset_ids === undefined
        ? diagnostics?.materials || refs
        : diagnostics?.resolved_mode === "strict_first_frame"
            ? diagnostics.materials
            : manuallySelectedMaterials;
    const materialCounters = { image: 0, video: 0, audio: 0 };
    const numberedMaterials = resolvedMaterials.map((asset) => {
        const kind = asset.type as "image" | "video" | "audio";
        materialCounters[kind] += 1;
        return { ...asset, label: `${kind === "image" ? "图片" : kind === "video" ? "视频" : "音频"}${materialCounters[kind]}` };
    });
    const counts = {
        image: resolvedMaterials.filter((asset) => asset.type === "image").length,
        video: resolvedMaterials.filter((asset) => asset.type === "video").length,
        audio: resolvedMaterials.filter((asset) => asset.type === "audio").length,
    };
    return <article className="rounded-xl border border-white/8 bg-black/15 p-4">
        <div className="flex flex-wrap items-start gap-3">
            <input type="checkbox" checked={checked} onChange={toggleChecked} />
            <div className="min-w-48 flex-1"><strong>#{shot.ordinal} {shot.title}</strong><p className="mt-1 text-xs text-white/35">{shot.duration_seconds.toFixed(2)} 秒 · 图片 {counts.image} / 视频 {counts.video} / 音频 {counts.audio}</p></div>
            <span className="studio-status">{draft.seedance_prompt_version || "待编译"}</span>
        </div>
        {resolvedMaterials.length === 0 && <div className="mt-3 rounded-lg border border-amber-300/15 bg-amber-300/[.04] px-3 py-2 text-xs leading-5 text-amber-50/60">本镜还没有分镜图或参考素材；Seedance 提交前会拦截，请先生成首帧或在“参考素材”页绑定素材。</div>}
        {diagnostics && <details className="mt-3 rounded-lg border border-cyan-300/10 bg-cyan-300/[.025] px-3 py-2 text-xs leading-5 text-cyan-50/55"><summary className="cursor-pointer font-semibold">{diagnostics.resolved_mode === "strict_first_frame" ? "实际提交：严格首帧" : "实际提交：全模态参考"} · {numberedMaterials.length} 项素材</summary><p className="mt-2 text-cyan-100/55">首帧完整度：{diagnostics.first_frame_completeness === "complete" ? "已确认完整" : diagnostics.first_frame_completeness === "incomplete" ? "已标记不完整" : "未确认"} · {diagnostics.reference_mode_reason}</p><div className="mt-2 space-y-1">{numberedMaterials.map((item) => <div key={item.id} className="flex items-center gap-2"><span className="rounded bg-cyan-300/10 px-1.5 py-0.5 font-mono text-cyan-100/70">{item.label}</span><span className="min-w-0 flex-1 truncate">{item.name}</span><span className="text-white/25">{assetRoleLabel(item.role)}</span></div>)}{numberedMaterials.length === 0 && <p className="text-amber-100/65">当前没有视频参考素材。</p>}</div><div className="mt-3 flex items-center justify-between gap-2"><strong className="text-white/50">选择或取消本镜视频参考</strong><button type="button" className="studio-secondary px-2 py-1 text-[11px]" onClick={() => setDraft((current) => ({ ...current, video_reference_asset_ids: null }))}>恢复自动选择</button></div><div className="mt-2 max-h-40 space-y-1 overflow-y-auto rounded-md border border-white/8 p-2">{availableMaterials.map((asset) => <label key={asset.id} className="flex items-center gap-2 rounded px-1 py-0.5 text-white/55 hover:bg-white/[.03]"><input type="checkbox" checked={effectiveVideoReferenceIds.includes(asset.id)} onChange={() => setDraft((current) => ({ ...current, video_reference_asset_ids: toggle(current.video_reference_asset_ids ?? effectiveVideoReferenceIds, asset.id) }))} /><span className="min-w-0 flex-1 truncate">{asset.name}</span><span className="text-white/25">{assetRoleLabel(asset.role)}</span></label>)}</div><p className="mt-2 text-[11px] text-white/30">{draft.video_reference_asset_ids == null ? "自动选择中" : "已切换为本镜手动清单；保存或提交时会按新编号重编译 Prompt。"}</p>{diagnostics.warnings.map((warning) => <p key={warning} className="mt-1 text-amber-100/75">{warning}</p>)}</details>}
        <div className="mt-3 grid gap-3 md:grid-cols-2"><Field label="首帧完整度" hint="完整表示首帧已经包含本镜所需人物、场景和固定物品；自动模式会据此选择提交方式。"><select value={draft.first_frame_completeness || "unknown"} onChange={(event) => setDraft((current) => ({ ...current, first_frame_completeness: event.target.value as Shot["first_frame_completeness"], seedance_reference_mode: "auto" }))}><option value="unknown">未确认 · 默认全模态</option><option value="complete">完整 · 自动严格首帧</option><option value="incomplete">不完整 · 自动全模态</option></select></Field><Field label="Seedance 参考策略" hint="严格首帧只提交当前首帧；全模态会同时提交场景、人物、固定物品等选定素材。"><select value={draft.seedance_reference_mode} onChange={(event) => setDraft((current) => ({ ...current, seedance_reference_mode: event.target.value as Shot["seedance_reference_mode"] }))}><option value="auto">自动 · 按首帧完整度选择</option><option value="multimodal_reference">手动全模态参考</option><option value="strict_first_frame">手动严格首帧</option></select></Field></div>
        <div className="mt-4 grid gap-4 xl:grid-cols-[360px_minmax(0,1fr)]"><ShotKeyframeSummary shot={shot} keyframe={keyframe} projectId={projectId} generating={generatingKeyframe} edit={editKeyframe} preview={previewKeyframe} /><div><label className="block"><span className="studio-label">Seedance 2.0 Prompt</span><textarea className="studio-input min-h-64 resize-y" value={draft.seedance_prompt} onChange={(event) => setDraft((current) => ({ ...current, seedance_prompt: event.target.value }))} placeholder="点击上方“编译所选 Prompt”，系统会按图片1/视频1/音频1的真实发送顺序生成 Seedance 专用提示词。" /></label><div className="mt-3 grid gap-3 md:grid-cols-2"><Field label="起始场景模板（可选）" hint="首帧与转场前只使用这一场景；单场景镜头只需选择这里。"><select value={draft.use_scene_profile ? selectedSceneProfileIds(draft)[0] || "" : ""} onChange={(event) => setDraft((current) => ({ ...current, ...sceneProfileSelectionPatch(current, "start", event.target.value) }))}><option value="">关闭 · 保持原链路</option>{project.scene_profiles.map((profile) => <option key={profile.id} value={profile.id}>{profile.name}{profile.approved ? " · 已有母版" : " · 待生成母版"}</option>)}</select></Field><Field label="转场后场景模板（可选）" hint="角色完成穿门、穿墙或传送后只使用目标场景，系统会按顺序提交第二张场景母版。"><select disabled={!draft.use_scene_profile || !selectedSceneProfileIds(draft)[0]} value={selectedSceneProfileIds(draft)[1] || ""} onChange={(event) => setDraft((current) => ({ ...current, ...sceneProfileSelectionPatch(current, "destination", event.target.value) }))}><option value="">无 · 本镜不跨场景</option>{project.scene_profiles.filter((profile) => profile.id !== selectedSceneProfileIds(draft)[0]).map((profile) => <option key={profile.id} value={profile.id}>{profile.name}{profile.approved ? " · 已有母版" : " · 待生成母版"}</option>)}</select></Field><Field label="镜头连续方式" hint="连续续接会读取上一镜已采用版本的尾帧作为构图起点，并与场景图、人物形象图一起提交。"><select value={draft.continuity_mode} onChange={(event) => setDraft((current) => ({ ...current, continuity_mode: event.target.value as Shot["continuity_mode"], continuity_source_shot_id: event.target.value === "continuous" ? current.continuity_source_shot_id : null }))}><option value="independent">独立镜头</option><option value="same_scene">同场景但允许切机位</option><option value="continuous" disabled={shot.ordinal === 1}>连续长镜头 · 接上一镜尾帧</option></select></Field></div>{selectedSceneProfileIds(draft).length > 1 && <p className="mt-2 rounded-lg border border-cyan-300/12 bg-cyan-300/[.03] px-3 py-2 text-xs leading-5 text-cyan-50/55">跨场景模式已启用：首帧只按起始场景生成；视频提交时同时携带两张母版，并明确要求转场前后分别使用，禁止把两套房间和光影拼在一起。</p>}{draft.continuity_mode === "continuous" && <p className="mt-2 rounded-lg border border-cyan-300/12 bg-cyan-300/[.03] px-3 py-2 text-xs leading-5 text-cyan-50/55">本镜提交时会等待并读取上一镜已采用版本的尾帧；若尚未手动采用，则使用上一镜最近一次成功结果。按顺序批量提交最稳；单独提交前请确保上一镜已完成。</p>}</div></div>
        <div className="mt-3 flex flex-wrap items-center justify-between gap-3"><p className="text-[11px] leading-5 text-white/30">编号与提交给方舟的同类素材顺序共用同一编译器，不会覆盖 MiniMax H3 Prompt。</p><div className="flex gap-2"><button type="button" className="studio-secondary" disabled={busy || !draft.seedance_prompt.trim()} onClick={() => void save(draft)}>{busy ? <Loader2 className="animate-spin" size={14} /> : <Save size={14} />}保存本镜</button><button type="button" className="studio-primary" disabled={busy || !canSubmit || !draft.seedance_prompt.trim()} title={canSubmit ? "提交后会产生方舟费用" : "请先配置方舟并载入模型目录"} onClick={() => void submit(draft)}>{busy ? <Loader2 className="animate-spin" size={14} /> : <Play size={14} />}提交本镜</button></div></div>
    </article>;
}

function ProductionPanel({ bundle, busy, busyTasks, error, notice, action, refresh, notify, fail }: { bundle: ProjectBundle; busy: string; busyTasks: BusyState; error: string; notice: string; action: Action; refresh: () => Promise<void>; notify: (message: string) => void; fail: (message: string) => void }) {
    const [videoEngine, setVideoEngine] = useState<"seedance" | "h3">("seedance");
    const [preflight, setPreflight] = useState<Record<string, unknown> | null>(null);
    const [selected, setSelected] = useState<string[]>([]);
    const [seedanceCatalog, setSeedanceCatalog] = useState<SeedanceCatalog | null>(null);
    const [seedanceModel, setSeedanceModel] = useState("doubao-seedance-2-0-mini-260615");
    const [seedanceResolution, setSeedanceResolution] = useState("480p");
    const [seedanceGenerateAudio, setSeedanceGenerateAudio] = useState(true);
    const [seedanceEstimate, setSeedanceEstimate] = useState<SeedanceEstimate | null>(null);
    const [seedanceEstimateError, setSeedanceEstimateError] = useState("");
    const [promptSkills, setPromptSkills] = useState<H3PromptSkill[]>([]);
    const [promptSkillId, setPromptSkillId] = useState(bundle.shots[0]?.h3_prompt_skill_id || "h3-prompt-writing");
    const [promptSkillSuggestions, setPromptSkillSuggestions] = useState("");
    const [promptSkillError, setPromptSkillError] = useState("");
    const [h3DirectorVersion, setH3DirectorVersion] = useState("");
    const [h3PromptTargetCount, setH3PromptTargetCount] = useState(0);
    const [keyframeBoardOpen, setKeyframeBoardOpen] = useState(false);
    const [seedancePromptsOpen, setSeedancePromptsOpen] = useState(false);
    const [keyframeEditor, setKeyframeEditor] = useState<KeyframeEditorState | null>(null);
    const [keyframeMaterials, setKeyframeMaterials] = useState<KeyframeMaterialDiagnostics | null>(null);
    const [keyframePreview, setKeyframePreview] = useState<MediaPreviewState | null>(null);
    const [videoPreview, setVideoPreview] = useState<VideoPreviewState | null>(null);
    const [pendingKeyframes, setPendingKeyframes] = useState<string[]>([]);
    const pendingKeyframeIds = useRef(new Set<string>());
    const savingKeyframeIds = useRef(new Set<string>());
    const h3PromptRequest = useRef(false);
    const h3PromptsBusy = busyTasks.has("h3-prompts");
    const h3PromptBlocked = isH3PromptBlocked(busyTasks);
    const keyframeBusy = (shot: Shot) => isKeyframeBusy(shot, busyTasks, new Set(pendingKeyframes));
    const availableKeyframeIds = bundle.shots.filter((shot) => selected.includes(shot.id) && !keyframeBusy(shot)).map((shot) => shot.id);
    const active = bundle.jobs.filter((job) => ACTIVE_JOBS.has(job.status));
    const cost = bundle.jobs.reduce((sum, job) => sum + (job.estimated_cost || 0), 0);
    const keyframeAssets = new Map(bundle.assets.map((asset) => [asset.id, asset]));
    const generatedShotIds = bundle.shots.filter((shot) => shot.keyframe_asset_id && keyframeAssets.has(shot.keyframe_asset_id)).map((shot) => shot.id);
    const generatedKeyframes = generatedShotIds.length;
    const selectedKeyframes = generatedShotIds.filter((shotId) => selected.includes(shotId));
    const allGeneratedSelected = generatedShotIds.length > 0 && selectedKeyframes.length === generatedShotIds.length;
    const editorShot = keyframeEditor ? bundle.shots.find((shot) => shot.id === keyframeEditor.shotId) : undefined;
    const editorKeyframe = editorShot?.keyframe_asset_id ? keyframeAssets.get(editorShot.keyframe_asset_id) : undefined;
    const effectiveKeyframeReferenceIds = keyframeEditor?.referenceAssetIds
        ?? keyframeMaterials?.materials.map((item) => item.id)
        ?? [];
    const keyframeMaterialOptions = keyframeMaterials?.available_materials
        ?? bundle.assets.filter((asset) => asset.type === "image" && ["character", "prop", "scene", "style"].includes(asset.role));
    const selectedKeyframeMaterials = effectiveKeyframeReferenceIds
        .map((id) => keyframeMaterialOptions.find((item) => item.id === id) || bundle.assets.find((item) => item.id === id))
        .filter((item) => !!item);
    const selectedPromptSkill = promptSkills.find((skill) => skill.id === promptSkillId);
    const selectedSeedanceModel = seedanceCatalog?.models.find((model) => model.id === seedanceModel);
    const missingH3ShotIds = bundle.shots.filter((shot) => (
        !shot.h3_prompt_skill_output.trim()
        || shot.h3_prompt_source_revision !== shot.content_revision
        || (!!h3DirectorVersion && shot.h3_director_version !== h3DirectorVersion)
    )).map((shot) => shot.id);
    const readyH3PromptCount = bundle.shots.length - missingH3ShotIds.length;
    const savedH3PromptCount = bundle.shots.filter((shot) => shot.h3_prompt_skill_output.trim()).length;
    const absentH3PromptCount = bundle.shots.length - savedH3PromptCount;
    const staleH3PromptCount = savedH3PromptCount - readyH3PromptCount;
    useEffect(() => {
        let active = true;
        void getH3PromptSkills().then((payload) => {
            if (!active) return;
            setPromptSkills(payload.skills);
            setH3DirectorVersion(payload.director_version);
            setPromptSkillId((current) => payload.skills.some((skill) => skill.id === current) ? current : payload.default_skill_id);
            setPromptSkillError("");
        }).catch((reason: unknown) => {
            if (active) setPromptSkillError(reason instanceof Error ? reason.message : "H3 Prompt Skill 列表加载失败");
        });
        void getSeedanceCatalog().then((payload) => {
            if (!active) return;
            setSeedanceCatalog(payload);
            setSeedanceModel(payload.default_model);
            const model = payload.models.find((item) => item.id === payload.default_model);
            const preferredResolution = payload.default_model.includes("mini") && model?.resolutions.includes("480p")
                ? "480p"
                : model?.resolutions.includes(payload.default_resolution)
                    ? payload.default_resolution
                    : model?.resolutions[0] || "480p";
            setSeedanceResolution(preferredResolution);
        }).catch((reason: unknown) => {
            if (active) setSeedanceEstimateError(reason instanceof Error ? reason.message : "Seedance 模型目录加载失败");
        });
        return () => { active = false; };
    }, []);
    useEffect(() => {
        if (videoEngine !== "seedance" || !seedanceCatalog || selected.length === 0) {
            setSeedanceEstimate(null);
            return;
        }
        let active = true;
        setSeedanceEstimateError("");
        const timer = window.setTimeout(() => {
            void estimateSeedance(bundle.project.id, selected, seedanceModel, seedanceResolution).then((payload) => {
                if (active) setSeedanceEstimate(payload);
            }).catch((reason: unknown) => {
                if (active) {
                    setSeedanceEstimate(null);
                    setSeedanceEstimateError(reason instanceof Error ? reason.message : "费用估算失败");
                }
            });
        }, 180);
        return () => { active = false; window.clearTimeout(timer); };
    }, [bundle.project.id, seedanceCatalog, seedanceModel, seedanceResolution, selected, videoEngine]);
    useEffect(() => {
        if (pendingKeyframes.length === 0 && !h3PromptsBusy && !bundle.shots.some((shot) => shot.image_status === "processing")) return;
        const timer = window.setInterval(() => { void refresh(); }, 3000);
        return () => window.clearInterval(timer);
    }, [pendingKeyframes.length, h3PromptsBusy, bundle.shots, refresh]);
    const changeSeedanceModel = (modelId: string) => {
        setSeedanceModel(modelId);
        const model = seedanceCatalog?.models.find((item) => item.id === modelId);
        if (modelId.includes("mini") && model?.resolutions.includes("480p")) {
            setSeedanceResolution("480p");
        } else if (model && !model.resolutions.includes(seedanceResolution)) {
            setSeedanceResolution(model.resolutions[0]);
        }
        setPreflight(null);
    };
    const runPreflight = () => action("preflight", async () => {
        setPreflight(videoEngine === "seedance" ? await seedancePreflight(bundle.project.id, seedanceModel, seedanceResolution) : await comfyPreflight(bundle.project.id));
    }, videoEngine === "seedance" ? "方舟配置检查完成；本次检查没有创建付费任务。" : "服务器检查完成；结果显示在当前按钮下方。", false);
    const generateSelectedKeyframes = () => startKeyframeGeneration(availableKeyframeIds, {}, `所选 ${availableKeyframeIds.length} 镜首帧`, "所选镜头的分镜首帧已处理；成功图片显示在下方并自动绑定到 I2V 首帧。");
    const generateH3PromptSet = async (shotIds: string[], label: string) => {
        if (shotIds.length === 0 || h3PromptBlocked || h3PromptRequest.current) return;
        h3PromptRequest.current = true;
        setH3PromptTargetCount(shotIds.length);
        try {
            await action(
                "h3-prompts",
                () => generateH3Prompts(bundle.project.id, shotIds, promptSkillId, promptSkillSuggestions.trim()),
                `已用“${selectedPromptSkill?.name || promptSkillId}”生成并持久化保存 ${label}。`,
            );
        } finally {
            h3PromptRequest.current = false;
            setH3PromptTargetCount(0);
        }
    };
    const generateSelectedH3Prompts = () => generateH3PromptSet(selected, `${selected.length} 个所选镜头的 H3 Prompt`);
    const generateMissingH3Prompts = () => generateH3PromptSet(missingH3ShotIds, `${missingH3ShotIds.length} 个待补镜头的 H3 Prompt`);
    const generateSelectedSeedancePrompts = () => action(
        "seedance-prompts",
        () => generateSeedancePrompts(bundle.project.id, selected),
        `已按方舟官方主体定义、素材编号、镜头时序与声音规则编译并保存 ${selected.length} 个 Seedance Prompt。`,
    );
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
        setKeyframeMaterials(null);
        setKeyframeEditor({
            shotId: shot.id,
            prompt: shot.keyframe_prompt || currentKeyframe?.description || shot.scene_description || shot.visual_prompt,
            suggestions: shot.keyframe_revision_suggestion_draft || "",
            revisionMode: shot.keyframe_revision_suggestion_draft
                ? shot.keyframe_revision_mode
                : currentKeyframe ? "iterate" : "fresh",
            referenceAssetIds: shot.keyframe_reference_asset_ids ?? null,
        });
        void getKeyframeMaterials(bundle.project.id, shot.id)
            .then(setKeyframeMaterials)
            .catch(() => setKeyframeMaterials(null));
    };
    const toggleKeyframeMaterial = (assetId: string) => {
        setKeyframeEditor((current) => {
            if (!current) return current;
            const base = current.referenceAssetIds ?? effectiveKeyframeReferenceIds;
            return { ...current, referenceAssetIds: toggle(base, assetId) };
        });
    };
    const saveKeyframePrompt = async () => {
        if (!editorShot || !keyframeEditor?.prompt.trim() || keyframeBusy(editorShot) || savingKeyframeIds.current.has(editorShot.id)) return;
        savingKeyframeIds.current.add(editorShot.id);
        let completed = false;
        await action(
            `keyframe-prompt-${editorShot.id}`,
            async () => {
                await updateKeyframePrompt(bundle.project.id, editorShot.id, {
                    keyframe_prompt: keyframeEditor.prompt.trim(),
                    keyframe_reference_asset_ids: keyframeEditor.referenceAssetIds,
                });
                completed = true;
            },
            `镜头 ${editorShot.ordinal} 的首帧 Prompt 已保存。`,
        );
        savingKeyframeIds.current.delete(editorShot.id);
        if (completed) setKeyframeEditor(null);
    };
    const startKeyframeGeneration = (shotIds: string[], options: { revisionMode?: "fresh" | "iterate"; userSuggestions?: string }, label: string, doneMessage: string) => {
        const targets = [...new Set(shotIds)].filter((id) => {
            const shot = bundle.shots.find((item) => item.id === id);
            return shot && !keyframeBusy(shot) && !pendingKeyframeIds.current.has(id);
        });
        if (targets.length === 0) {
            notify("这些镜头的首帧已在后台生成中，完成后可再次重做。");
            return;
        }
        targets.forEach((id) => pendingKeyframeIds.current.add(id));
        setPendingKeyframes(Array.from(pendingKeyframeIds.current));
        notify(`${label}已在后台开始生成，可关闭弹窗继续其他操作，完成后新图自动绑定。`);
        void (async () => {
            try {
                await generateKeyframes(bundle.project.id, targets, options);
                await refresh();
                notify(doneMessage);
            } catch (caught) {
                await refresh();
                fail(caught instanceof Error ? caught.message : String(caught));
            } finally {
                targets.forEach((id) => pendingKeyframeIds.current.delete(id));
                setPendingKeyframes(Array.from(pendingKeyframeIds.current));
            }
        })();
    };
    const regenerateKeyframe = async () => {
        if (!editorShot || !keyframeEditor?.prompt.trim() || keyframeBusy(editorShot) || savingKeyframeIds.current.has(editorShot.id)) return;
        const shot = editorShot;
        const prompt = keyframeEditor.prompt.trim();
        const suggestions = keyframeEditor.suggestions.trim();
        const revisionMode = keyframeEditor.revisionMode;
        const normalizedPrompt = prompt.replace(/[\s，。；：、,.!?！？:;'"“”‘’（）()【】\[\]]+/g, "").toLocaleLowerCase();
        const normalizedSuggestions = suggestions.replace(/[\s，。；：、,.!?！？:;'"“”‘’（）()【】\[\]]+/g, "").toLocaleLowerCase();
        const suggestionToSend = normalizedSuggestions === normalizedPrompt ? "" : suggestions;
        let saved = false;
        savingKeyframeIds.current.add(shot.id);
        await action(`keyframe-prompt-${shot.id}`, async () => {
            await updateKeyframePrompt(bundle.project.id, shot.id, {
                keyframe_prompt: prompt,
                keyframe_revision_suggestion_draft: suggestionToSend,
                keyframe_revision_mode: revisionMode,
                keyframe_reference_asset_ids: keyframeEditor.referenceAssetIds,
            });
            saved = true;
        }, `镜头 ${shot.ordinal} 的首帧设置已保存。`, false);
        savingKeyframeIds.current.delete(shot.id);
        if (!saved) return;
        setKeyframeEditor(null);
        startKeyframeGeneration(
            [shot.id],
            { revisionMode, userSuggestions: suggestionToSend },
            `镜头 ${shot.ordinal} 的首帧`,
            `镜头 ${shot.ordinal} 的分镜首帧已生成并自动绑定。`,
        );
    };
    const saveShot = (shot: Shot) => action(`h3-${shot.id}`, () => updateShot(shot), `镜头 ${shot.ordinal} 的 H3 参数已保存`);
    const uploadKeyframeFile = (shot: Shot, file: File) => action(`upload-keyframe-${shot.id}`, () => uploadKeyframe(bundle.project.id, shot.id, file), `镜头 ${shot.ordinal} 的首帧已替换为本地图片，并绑定为 I2V 首帧。`);
    const saveSeedanceShot = (shot: Shot) => action(`seedance-${shot.id}`, () => updateShot(shot), `镜头 ${shot.ordinal} 的 Seedance Prompt 已保存`);
    const submitOneShot = (engine: "seedance" | "h3", shot: Shot) => action(
        `render-${engine}-${shot.id}`,
        async () => {
            await updateShot(shot);
            await enqueueRender(
                bundle.project.id,
                [shot.id],
                engine === "seedance"
                    ? { provider: "ark_seedance", modelId: seedanceModel, resolution: seedanceResolution, generateAudio: seedanceGenerateAudio }
                    : {},
            );
        },
        engine === "seedance"
            ? `镜头 ${shot.ordinal} 已保存并加入 Seedance 队列。`
            : `镜头 ${shot.ordinal} 已保存并加入 H3 队列。`,
    );
    const applyBulkPreset = (preset: H3Preset) => action(`preset-${preset}`, () => Promise.all(bundle.shots.filter((shot) => selected.includes(shot.id)).map((shot) => updateShot({ ...shot, ...h3PresetPatch(preset, bundle.project) }))), `已将${preset === "fast" ? "快速预览" : preset === "balanced" ? "均衡" : "清晰优先"}应用到 ${selected.length} 个镜头`);
    const submitSelected = () => action(
        "render",
        () => enqueueRender(bundle.project.id, selected, videoEngine === "seedance" ? { provider: "ark_seedance", modelId: seedanceModel, resolution: seedanceResolution, generateAudio: seedanceGenerateAudio } : {}),
        videoEngine === "seedance" ? "Seedance 任务已加入本地持久队列；后台将逐镜提交、轮询并下载成片。" : "任务已加入本地持久队列；可在本页底部查看进度与成片。",
    );
    const busyText = busy === "preflight" ? (videoEngine === "seedance" ? "正在检查方舟 API Key 与所选模型配置（不会创建任务）…" : "正在检查当前 H3 通道配置…")
        : busy === "keyframes" ? `正在调用分镜图模型生成 ${selected.length} 张首帧，请稍候…`
            : busy === "h3-prompts" ? `正在按“${selectedPromptSkill?.name || promptSkillId}”为 ${h3PromptTargetCount} 个镜头生成 H3 Prompt；每镜完成后立即保存…`
            : busy === "seedance-prompts" ? `正在为 ${selected.length} 个镜头编译 Seedance 2.0 全模态 Prompt…`
            : busy === "plan" ? "正在编译所有镜头的 H3 模式、参考素材、帧数和参数…"
                : busy === "render" ? `正在把 ${selected.length} 个${videoEngine === "seedance" ? " Seedance" : " H3"}镜头写入本地持久队列…`
                    : busy.startsWith("preset-") ? "正在保存所选镜头的批量预设…"
                        : busy.startsWith("keyframe-prompt-") ? "正在保存首帧 Prompt…"
                            : busy.startsWith("upload-keyframe-") ? "正在上传本地首帧并替换绑定…"
                                : busy.startsWith("delete-") ? "正在删除渲染任务记录…"
                            : busy.startsWith("keyframe-") ? "正在生成这个镜头的首帧…" : "";
    return <div className="space-y-5">
        <section className="studio-panel">
            <div className="mb-5 flex flex-wrap items-center justify-between gap-3"><div><p className="studio-kicker">VIDEO ENGINE</p><h2 className="text-xl font-semibold">视频生成控制台</h2><p className="mt-1 text-sm text-white/40">Seedance API 与 MiniMax H3 的自建、MetaSo、Atlas 通道都完整保留；进入页面时默认不勾选任何分镜，避免误提交。</p></div><div className="flex rounded-xl border border-white/10 bg-black/20 p-1"><button type="button" className={`rounded-lg px-4 py-2 text-sm ${videoEngine === "seedance" ? "bg-cyan-300 text-black" : "text-white/45 hover:text-white"}`} onClick={() => { setVideoEngine("seedance"); setPreflight(null); }}>Seedance 2.0 API</button><button type="button" className={`rounded-lg px-4 py-2 text-sm ${videoEngine === "h3" ? "bg-cyan-300 text-black" : "text-white/45 hover:text-white"}`} onClick={() => { setVideoEngine("h3"); setPreflight(null); }}>MiniMax H3</button></div></div>
            <div className="flex flex-wrap items-center justify-between gap-4"><div><p className="studio-kicker">{videoEngine === "seedance" ? "VOLCENGINE ARK · FULL MODAL" : "MINIMAX H3"}</p><h3 className="font-semibold">{videoEngine === "seedance" ? selectedSeedanceModel?.label || "Seedance 2.0" : "当前 H3 通道"}</h3><p className="mt-1 text-xs leading-5 text-white/38">{videoEngine === "seedance" ? "使用分镜图与角色素材做全模态参考；按官方主体定义和图片/视频/音频编号生成。" : "具体使用自建、MetaSo 或 Atlas，由模型设置中的 H3 生成通道决定；默认保留 H3 原声。"}</p></div><div className="flex flex-wrap gap-2"><button className="studio-secondary" disabled={!!busy || (videoEngine === "seedance" && !seedanceCatalog)} title="只检查配置，不创建视频任务" onClick={() => void runPreflight()}>{busy === "preflight" ? <Loader2 className="animate-spin" size={15} /> : <Server size={15} />}检查配置</button><button className="studio-secondary" disabled={availableKeyframeIds.length === 0} title="调用分镜图服务，可与 H3 Prompt 并行；自动跳过正在处理首帧的镜头" onClick={() => void generateSelectedKeyframes()}>{selected.some((id) => pendingKeyframes.includes(id)) ? <Loader2 className="animate-spin" size={15} /> : <Sparkles size={15} />}生成所选首帧</button>{videoEngine === "seedance" ? <button className="studio-secondary" disabled={!!busy || selected.length === 0} title="只保存 Seedance 专用 Prompt，不调用付费视频接口" onClick={() => void generateSelectedSeedancePrompts()}>{busy === "seedance-prompts" ? <Loader2 className="animate-spin" size={15} /> : <WandSparkles size={15} />}编译所选 Prompt</button> : <button className="studio-secondary" disabled={!!busy || bundle.shots.length === 0} title="只整理参数和参考绑定，不生成图片或视频" onClick={() => void action("plan", () => planRender(bundle.project.id), "编译完成：H3 模式、帧数、参数和参考标签已保存。")}>{busy === "plan" ? <Loader2 className="animate-spin" size={15} /> : <Settings2 size={15} />}编译计划</button>}<button className="studio-primary" disabled={!!busy || selected.length === 0 || (videoEngine === "seedance" && (!seedanceCatalog?.configured || !seedanceEstimate))} title={videoEngine === "seedance" ? "提交后会产生实际方舟费用" : "提交后会使用模型设置中选定的 H3 通道"} onClick={() => void submitSelected()}>{busy === "render" ? <Loader2 className="animate-spin" size={15} /> : <Play size={15} />}提交 {selected.length} 镜</button></div></div>
            <div className="mt-5 grid gap-2 text-xs sm:grid-cols-2 xl:grid-cols-4">
                <div className="rounded-lg border border-white/8 bg-white/[.02] p-3"><strong className="text-white/75">1. 检查配置</strong><p className="mt-1 leading-5 text-white/35">{videoEngine === "seedance" ? "验证 API Key、模型与清晰度；不会创建任务或计费。" : "验证当前 H3 通道、鉴权或本地模型；不创建视频任务。"}</p></div>
                <div className="rounded-lg border border-white/8 bg-white/[.02] p-3"><strong className="text-white/75">2. 生成所选首帧</strong><p className="mt-1 leading-5 text-white/35">调用“模型设置 → 分镜首帧生成”；产物既是分镜图，也是 I2V 的第一帧。</p></div>
                <div className="rounded-lg border border-white/8 bg-white/[.02] p-3"><strong className="text-white/75">3. 编译 Prompt</strong><p className="mt-1 leading-5 text-white/35">{videoEngine === "seedance" ? "生成方舟专用素材编号、镜头时序、说话人和声音指令。" : "整理 H3 I2V/R2V、Prompt、帧数和参数。"}</p></div>
                <div className="rounded-lg border border-white/8 bg-white/[.02] p-3"><strong className="text-white/75">4. 提交镜头</strong><p className="mt-1 leading-5 text-white/35">{videoEngine === "seedance" ? "加入方舟异步队列；这是产生实际费用的一步。" : "加入 H3 持久队列，并按模型设置选择自建或线上通道。"}</p></div>
            </div>
            {videoEngine === "seedance" ? <div className="mt-4 rounded-xl border border-cyan-300/15 bg-cyan-300/[.025] p-4">
                <div className="flex flex-wrap items-start justify-between gap-3"><div><p className="studio-kicker">SEEDANCE MODEL & COST</p><h3 className="font-semibold">模型、清晰度、声音与实时费用</h3><p className="mt-1 max-w-3xl text-xs leading-5 text-white/40">按方舟公开 token 公式估算；时长向上取整且每镜至少 4 秒，最终金额以任务 usage 与账单为准。</p></div><div className="flex gap-3 text-xs"><a className="text-cyan-200/70 underline underline-offset-2" href={seedanceCatalog?.prompt_guide} target="_blank" rel="noreferrer">官方提示词指南</a><a className="text-cyan-200/70 underline underline-offset-2" href={seedanceCatalog?.pricing_source} target="_blank" rel="noreferrer">官方价格</a></div></div>
                <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                    <Field label="视频模型" hint={selectedSeedanceModel?.description}><select value={seedanceModel} disabled={!seedanceCatalog || !!busy} onChange={(event) => changeSeedanceModel(event.target.value)}>{seedanceCatalog?.models.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}</select></Field>
                    <Field label="输出清晰度" hint="2.0 支持 1080p/4K；Fast 与 Mini 最高 720p。"><select value={seedanceResolution} disabled={!selectedSeedanceModel || !!busy} onChange={(event) => { setSeedanceResolution(event.target.value); setPreflight(null); }}>{selectedSeedanceModel?.resolutions.map((resolution) => <option key={resolution} value={resolution}>{resolution}</option>)}</select></Field>
                    <Field label="原生同步声音" hint="开启后模型同时生成对白、环境声和音效；Prompt 会明确说话人。"><select value={seedanceGenerateAudio ? "on" : "off"} disabled={!!busy} onChange={(event) => setSeedanceGenerateAudio(event.target.value === "on")}><option value="on">开启 generate_audio</option><option value="off">关闭，生成无声视频</option></select></Field>
                    <div className="rounded-lg border border-cyan-300/15 bg-black/20 p-3"><p className="text-[11px] uppercase tracking-wider text-white/35">本次所选预估</p>{seedanceEstimate ? <><strong className="mt-1 block text-2xl text-cyan-100">¥{seedanceEstimate.estimated_yuan.toFixed(2)}</strong><p className="mt-1 text-xs text-white/45">{seedanceEstimate.shot_count} 镜 · 请求 {seedanceEstimate.requested_duration_seconds.toFixed(1)}s · 计费 {seedanceEstimate.billed_duration_seconds}s</p><p className="mt-1 text-[11px] text-white/30">约 ¥{seedanceEstimate.estimated_yuan_per_second.toFixed(4)}/秒 + 每任务 ¥{seedanceEstimate.estimated_yuan_per_task.toFixed(4)}</p></> : <p className="mt-2 text-xs text-white/35">选择镜头后自动计算</p>}</div>
                </div>
                {seedanceCatalog && <div className="mt-4 grid gap-2 md:grid-cols-3">{seedanceCatalog.models.map((model) => <div key={`price-${model.id}`} className={`rounded-lg border p-3 ${model.id === seedanceModel ? "border-cyan-300/35 bg-cyan-300/[.04]" : "border-white/8 bg-black/15"}`}><div className="flex items-center justify-between gap-2"><strong className="text-sm text-white/75">{model.label}</strong><span className="text-xs text-cyan-100/70">约 ¥{model.example_720p_yuan_per_second.toFixed(4)}/秒</span></div><p className="mt-1 text-[11px] leading-5 text-white/35">720p 16:9；图/文输入 ¥{model.price_per_million_tokens}/百万 token，含视频输入 ¥{model.video_input_price_per_million_tokens}/百万 token。</p></div>)}</div>}
                {!seedanceCatalog?.configured && <div className="mt-3 rounded-lg border border-amber-300/15 bg-amber-300/[.04] px-3 py-2 text-xs text-amber-50/65">方舟 API Key 尚未配置。先到“模型设置”填写 ARK_API_KEY；在此之前仍可编辑 Prompt、首帧和费用方案，提交按钮保持禁用。</div>}
                {seedanceEstimateError && <p className="mt-3 text-xs text-red-300/80">{seedanceEstimateError}</p>}
            </div> : <div className="mt-4 rounded-xl border border-cyan-300/15 bg-cyan-300/[.025] p-4">
                <div className="flex flex-wrap items-start justify-between gap-3"><div><p className="studio-kicker">OFFICIAL PROMPT SKILLS</p><h3 className="font-semibold">后期一键补生成 MiniMax H3 Prompt</h3><p className="mt-1 max-w-3xl text-xs leading-5 text-white/40">分镜阶段只选 Seedance 也没关系。这里会识别缺失、内容已更新或导演版本过期的镜头，一键补齐；只生成并保存 Prompt，可与首帧生成并行，不会启动视频生成。文字策略按引擎分开：H3 会保留分镜明确写出的画面文字，首帧与 Seedance 仍保持无可读文字。</p></div><div className="flex items-center gap-2"><span className="studio-status text-emerald-200">已保存 {savedH3PromptCount}/{bundle.shots.length} · 待核对 {staleH3PromptCount}</span>{selectedPromptSkill && <a className="text-xs text-cyan-200/70 underline underline-offset-2" href={selectedPromptSkill.source_url} target="_blank" rel="noreferrer">查看官方 Skill v{selectedPromptSkill.version}</a>}</div></div>
                <div className="mt-4 grid gap-3 lg:grid-cols-[minmax(240px,.72fr)_minmax(300px,1.28fr)]">
                    <label><span className="studio-label">风格 Skill</span><select className="studio-input" value={promptSkillId} disabled={h3PromptBlocked || promptSkills.length === 0} onChange={(event) => setPromptSkillId(event.target.value)}>{promptSkills.length === 0 ? <option value={promptSkillId}>正在加载官方风格…</option> : promptSkills.map((skill) => <option key={skill.id} value={skill.id}>{skill.category} · {skill.name}</option>)}</select></label>
                    <label><span className="studio-label">本次 Prompt 建议（可选）</span><input className="studio-input" maxLength={4000} value={promptSkillSuggestions} onChange={(event) => setPromptSkillSuggestions(event.target.value)} placeholder="例如：动作更克制；保留当前群像构图；不要背景音乐……" /></label>
                </div>
                <div className="mt-3 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-white/8 bg-black/15 p-3"><p className="text-xs leading-5 text-white/40">缺失 {absentH3PromptCount} 镜 · 待核对 {staleH3PromptCount} 镜。素材或剧情变化只标记待核对，已生成原文与历史仍会保留。点击补齐/更新才会再次调用文本模型。</p><div className="flex flex-wrap gap-2"><button type="button" className="studio-secondary" disabled={h3PromptBlocked || selected.length === 0 || promptSkills.length === 0} onClick={() => void generateSelectedH3Prompts()}>{h3PromptsBusy ? <Loader2 className="animate-spin" size={15} /> : <WandSparkles size={15} />}重新生成所选 {selected.length} 镜</button><button type="button" className="studio-primary" disabled={h3PromptBlocked || missingH3ShotIds.length === 0 || promptSkills.length === 0} onClick={() => void generateMissingH3Prompts()}>{h3PromptsBusy ? <Loader2 className="animate-spin" size={15} /> : <Sparkles size={15} />}{missingH3ShotIds.length > 0 ? staleH3PromptCount > 0 ? `补齐/更新待核对 ${missingH3ShotIds.length} 镜` : `一键补齐待生成 ${missingH3ShotIds.length} 镜` : "H3 Prompt 已全部生成"}</button></div></div>
                {selectedPromptSkill && <p className="mt-3 text-xs leading-5 text-cyan-50/50"><strong className="text-cyan-100/70">{selectedPromptSkill.name}：</strong>{selectedPromptSkill.summary}</p>}
                {promptSkillError && <p className="mt-3 text-xs text-red-300/80">{promptSkillError}</p>}
            </div>}
            {(busyText || error || notice) && <div aria-live="polite" className={`mt-4 ${error ? "studio-error" : "studio-notice"}`}>{busyText && <span className="inline-flex items-center gap-2"><Loader2 className="animate-spin" size={15} />{busyText}</span>}{!busyText && (error || notice)}{error && <Link className="ml-2 underline underline-offset-2" href="/logs">查看运行日志</Link>}</div>}
        </section>
        {preflight && (videoEngine === "seedance" ? <div className={preflight.ok ? "studio-notice" : "rounded-lg border border-amber-400/20 bg-amber-400/6 px-4 py-3 text-sm text-amber-100/75"}><strong>{String(preflight.message || (preflight.ok ? "方舟配置已就绪" : "方舟配置待补充"))}</strong><p className="mt-1 text-xs opacity-70">{String(preflight.model_label || preflight.model_id || seedanceModel)} · {String(preflight.resolution || seedanceResolution)} · 仅配置检查，未创建付费任务。</p></div> : preflight.provider === "metaso_h3" || preflight.provider === "atlas_h3" ? <div className={preflight.ok ? "studio-notice" : "rounded-lg border border-amber-400/20 bg-amber-400/6 px-4 py-3 text-sm text-amber-100/75"}><strong>{String(preflight.message || (preflight.ok ? "线上 H3 配置已就绪" : "线上 H3 配置待补充"))}</strong><p className="mt-1 text-xs opacity-70">{String(preflight.model || "MiniMax H3")} · {String(preflight.resolution || "-")} · {String(preflight.ratio || "adaptive")}{preflight.provider === "metaso_h3" ? ` · Context IR ${preflight.context_ir_enabled ? "已开启" : "已关闭"}` : ""} · 本次只查询配置，未创建视频任务。</p>{preflight.error ? <p className="mt-1 text-xs text-red-200/80">{String(preflight.error)}</p> : null}</div> : <div className={preflight.online ? "studio-notice" : "rounded-lg border border-amber-400/20 bg-amber-400/6 px-4 py-3 text-sm text-amber-100/75"}>{preflight.online ? <div className="space-y-1.5"><strong>服务器在线 · 默认模型检查：{preflight.ok ? "通过" : "有缺失"}</strong><p className="text-xs opacity-70">Diffusion：{String(preflight.model_profile || "-")} · 文本编码器：{String(preflight.text_encoder_profile || "-")}</p>{Array.isArray(preflight.missing_models) && preflight.missing_models.length > 0 && <p className="text-xs text-amber-100">缺少模型：{preflight.missing_models.map(String).join("、")}</p>}{Array.isArray(preflight.missing_nodes) && preflight.missing_nodes.length > 0 && <p className="text-xs text-amber-100">缺少节点：{preflight.missing_nodes.map(String).join("、")}</p>}<p className="text-[11px] opacity-55">已发现 diffusion {(preflight.available_diffusion_models as unknown[] | undefined)?.length || 0} 个、文本编码器 {(preflight.available_text_encoders as unknown[] | undefined)?.length || 0} 个；提交单镜时还会按该镜所选文件再次拦截检查。</p></div> : <span className="inline-flex items-center gap-2"><CloudOff size={15} />服务器处于关闭状态，参数仍可在本地编辑保存。</span>} {preflight.error ? String(preflight.error) : ""}</div>)}
        <section className="studio-panel">
            <div className={`flex flex-wrap items-center justify-between gap-3 ${keyframeBoardOpen ? "mb-4" : ""}`}><div><p className="studio-kicker">KEYFRAME BOARD</p><h3 className="font-semibold">分镜图 / I2V 首帧</h3><p className="mt-1 text-xs text-white/35">已生成 {generatedKeyframes}/{bundle.shots.length}。{keyframeBoardOpen ? "可全选已生成图片并打包导出；也可在卡片中单独生成或重做。" : "当前已收起，展开后可查看、选择和生成分镜图。"}</p></div><div className="flex flex-wrap items-center gap-2"><span className="studio-status">已选 {selected.length} 镜 · 可导出 {selectedKeyframes.length} 张</span><button type="button" className="studio-secondary px-3" disabled={generatedShotIds.length === 0} onClick={toggleAllGeneratedKeyframes}><Check size={15} />{allGeneratedSelected ? "取消全选" : `全选已生成 ${generatedShotIds.length} 张`}</button><button type="button" className="studio-secondary px-3" disabled={busyTasks.has("export-keyframes") || selectedKeyframes.length === 0} onClick={() => void exportSelectedKeyframes()}>{busyTasks.has("export-keyframes") ? <Loader2 className="animate-spin" size={15} /> : <Download size={15} />}导出选中 {selectedKeyframes.length} 张</button><Link href="/settings" className="studio-secondary px-3">分镜图模型设置</Link><button type="button" className="studio-secondary px-3" aria-expanded={keyframeBoardOpen} aria-controls="keyframe-board-content" onClick={() => setKeyframeBoardOpen((open) => !open)}>{keyframeBoardOpen ? <ChevronUp size={15} /> : <ChevronDown size={15} />}{keyframeBoardOpen ? "收起" : "展开"}</button></div></div>
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
                const prompt = shot.keyframe_prompt || keyframe?.description || shot.scene_description || shot.visual_prompt;
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
                            <button type="button" className="studio-secondary px-3 py-2 text-xs" disabled={keyframeBusy(shot)} onClick={() => openKeyframeEditor(shot)}><FilePenLine size={13} />Prompt 详情</button>
                            <label className={`${keyframe ? "col-span-2 " : ""}studio-secondary cursor-pointer px-3 py-2 text-xs ${keyframeBusy(shot) ? "pointer-events-none opacity-50" : ""}`} title="上传本地图片，替换本镜 I2V 首帧绑定"><input type="file" accept="image/png,image/jpeg,image/webp,image/bmp" className="hidden" disabled={keyframeBusy(shot)} onChange={(event) => { const file = event.target.files?.[0]; event.target.value = ""; if (file && !keyframeBusy(shot)) void uploadKeyframeFile(shot, file); }} />{busyTasks.has(`upload-keyframe-${shot.id}`) ? <Loader2 className="animate-spin" size={13} /> : <Upload size={13} />}{keyframe ? "上传/替换本地图" : "上传本地首帧"}</label>
                            <button type="button" className={`${keyframe ? "studio-secondary" : "studio-primary"} col-span-2 px-3 py-2 text-xs`} disabled={keyframeBusy(shot)} onClick={() => openKeyframeEditor(shot)}>{pendingKeyframes.includes(shot.id) || shot.image_status === "processing" ? <Loader2 className="animate-spin" size={13} /> : <Sparkles size={13} />}{pendingKeyframes.includes(shot.id) || shot.image_status === "processing" ? "后台生成中…" : keyframe || shot.image_status === "failed" ? "输入建议并重新生成" : "编辑 Prompt 并生成本镜"}</button>
                        </div>
                    </div>
                </article>;
            })}</div>}</div>}
        </section>
        {videoEngine === "seedance" ? <section className="studio-panel"><div className={`flex flex-wrap items-center justify-between gap-3 ${seedancePromptsOpen ? "mb-4" : ""}`}><div><p className="studio-kicker">SEEDANCE SHOTS</p><h3 className="font-semibold">逐镜首帧 + Seedance 全模态 Prompt</h3><p className="mt-1 text-xs text-white/35">{seedancePromptsOpen ? "每个分镜内可核对首帧、Prompt、场景连续性并直接提交本镜。" : `当前已收起 ${bundle.shots.length} 个完整分镜，渲染队列显示在下方。`}</p></div><div className="flex flex-wrap gap-2">{seedancePromptsOpen && <><button className="studio-secondary" disabled={!selected.length || !!busy} onClick={() => void generateSelectedSeedancePrompts()}>{busy === "seedance-prompts" ? <Loader2 className="animate-spin" size={15} /> : <WandSparkles size={15} />}重新编译所选 {selected.length} 镜</button><button className="studio-secondary" disabled={bundle.shots.length === 0} onClick={() => setSelected(selected.length === bundle.shots.length ? [] : bundle.shots.map((shot) => shot.id))}>{bundle.shots.length > 0 && selected.length === bundle.shots.length ? "取消全选" : "全选"}</button></>}<button type="button" className="studio-secondary px-3" aria-expanded={seedancePromptsOpen} aria-controls="seedance-prompts-content" onClick={() => setSeedancePromptsOpen((open) => !open)}>{seedancePromptsOpen ? <ChevronUp size={15} /> : <ChevronDown size={15} />}{seedancePromptsOpen ? "整体收起" : "整体展开"}</button></div></div>{seedancePromptsOpen && <div id="seedance-prompts-content">{bundle.shots.length === 0 ? <p className="rounded-lg border border-dashed border-white/10 p-5 text-sm text-white/35">请先在“分镜设计”中生成或导入分镜。</p> : <div className="space-y-3">{bundle.shots.map((shot) => { const keyframe = shot.keyframe_asset_id ? keyframeAssets.get(shot.keyframe_asset_id) : undefined; return <SeedanceShotCard key={`seedance-${shot.id}-${shot.updated_at}`} shot={shot} project={bundle.project} assets={bundle.assets} checked={selected.includes(shot.id)} busy={h3PromptsBusy || keyframeBusy(shot)} toggleChecked={() => setSelected(toggle(selected, shot.id))} save={saveSeedanceShot} projectId={bundle.project.id} keyframe={keyframe} generatingKeyframe={keyframeBusy(shot)} editKeyframe={() => openKeyframeEditor(shot)} previewKeyframe={() => keyframe && setKeyframePreview({ name: `镜头 ${shot.ordinal} 首帧`, url: projectDownloadUrl(bundle.project.id, "asset", keyframe.id), description: shot.keyframe_prompt || keyframe.description })} submit={(draft) => submitOneShot("seedance", draft)} canSubmit={!!seedanceCatalog?.configured} />; })}</div>}</div>}</section> : <section className="studio-panel"><div className="mb-4 flex flex-wrap items-center justify-between gap-3"><div><h3 className="font-semibold">逐镜首帧 + H3 Prompt、参数与参考绑定</h3><p className="text-xs text-white/35">每个分镜可同时核对首帧和 H3 Prompt，并保存或直接提交本镜。</p></div><div className="flex flex-wrap gap-2"><button className="studio-secondary" title="608 长边、Turbo 4 步，适合低成本测试构图与动作" disabled={!selected.length || !!busy} onClick={() => void applyBulkPreset("fast")}>批量快速</button><button className="studio-secondary" title="使用项目设定分辨率和 Turbo 4 步参数" disabled={!selected.length || !!busy} onClick={() => void applyBulkPreset("balanced")}>批量均衡</button><button className="studio-secondary" title="至少 1344 长边、原生 25 步，适合最终成片" disabled={!selected.length || !!busy} onClick={() => void applyBulkPreset("quality")}>批量清晰</button><button className="studio-secondary" title="切换全部镜头的勾选状态，不修改参数" disabled={bundle.shots.length === 0} onClick={() => setSelected(selected.length === bundle.shots.length ? [] : bundle.shots.map((shot) => shot.id))}>{bundle.shots.length > 0 && selected.length === bundle.shots.length ? "取消全选" : "全选"}</button></div></div>
            <details open className="mb-4 rounded-xl border border-cyan-300/12 bg-cyan-300/[.025] p-4"><summary className="cursor-pointer text-sm font-semibold text-cyan-100/80">视频质量太差时，按画面问题这样调</summary><div className="mt-3 grid gap-2 text-xs sm:grid-cols-2 xl:grid-cols-4"><div className="rounded-lg border border-white/8 bg-black/15 p-3"><strong className="text-white/75">画面模糊、细节少</strong><p className="mt-1 leading-5 text-white/38">选择“清晰优先”：1344×768、原生25步；使用清晰完整构图锚点。Turbo 4步只用于预览。</p></div><div className="rounded-lg border border-white/8 bg-black/15 p-3"><strong className="text-white/75">人物脸或服装漂移</strong><p className="mt-1 leading-5 text-white/38">无对白优先 I2V 硬首帧；对白自动用完整分镜图作为 Picture 1，再只绑定本镜真实出场人物与发言者音频。</p></div><div className="rounded-lg border border-white/8 bg-black/15 p-3"><strong className="text-white/75">动作崩、拖影、重影</strong><p className="mt-1 leading-5 text-white/38">每个生成段只保留一个主动作，复杂镜头内部续帧；可保留群像、双人和过肩构图，不强制切单人近景。</p></div><div className="rounded-lg border border-white/8 bg-black/15 p-3"><strong className="text-white/75">生成太慢</strong><p className="mt-1 leading-5 text-white/38">先用 608×352 / Turbo4步预览；确认 Seed、Prompt、构图后，再切清晰优先生成最终版。</p></div></div></details>
            {bundle.shots.length === 0 ? <p className="rounded-lg border border-dashed border-white/10 p-5 text-sm text-white/35">请先在“分镜设计”中生成或导入分镜。</p> : <div className="space-y-3">{bundle.shots.map((shot) => { const keyframe = shot.keyframe_asset_id ? keyframeAssets.get(shot.keyframe_asset_id) : undefined; return <H3ShotCard key={`${shot.id}-${shot.updated_at}`} directorVersion={h3DirectorVersion} shot={shot} project={bundle.project} assets={bundle.assets} checked={selected.includes(shot.id)} busy={h3PromptsBusy || keyframeBusy(shot)} toggleChecked={() => setSelected(toggle(selected, shot.id))} save={saveShot} projectId={bundle.project.id} keyframe={keyframe} generatingKeyframe={keyframeBusy(shot)} editKeyframe={() => openKeyframeEditor(shot)} previewKeyframe={() => keyframe && setKeyframePreview({ name: `镜头 ${shot.ordinal} 首帧`, url: projectDownloadUrl(bundle.project.id, "asset", keyframe.id), description: shot.keyframe_prompt || keyframe.description })} submit={(draft) => submitOneShot("h3", draft)} />; })}</div>}</section>}
        <section className="studio-panel"><div className="mb-4 flex items-center justify-between"><div><h3 className="font-semibold">渲染任务</h3><p className="text-xs text-white/35">活动 {active.length} · 已记录估算成本 ¥{cost.toFixed(2)}</p></div><button className="studio-secondary" onClick={() => void refresh()}><RefreshCw size={14} />刷新</button></div>{bundle.jobs.length === 0 ? <p className="text-sm text-white/35">尚未提交任务。</p> : <div className="space-y-2">{bundle.jobs.map((job) => {
            const shot = bundle.shots.find((item) => item.id === job.shot_id);
            const shotTitle = shot?.title.replace(/^镜头\s*\d+\s*[·.:：-]?\s*/, "");
            const name = shot ? `镜头 ${shot.ordinal}${shotTitle ? ` · ${shotTitle}` : ""}` : `生成视频 ${job.id.slice(-6)}`;
            const seedance = (job.input_snapshot?.seedance || {}) as Record<string, unknown>;
            const provider = job.provider === "ark_seedance" ? String(seedance.model_label || "Seedance 2.0") : "MiniMax H3";
            const selected = shot?.selected_video_job_id === job.id;
            const currentOutput = !!shot?.video_path && shot.video_path === job.output_path;
            return <JobRow key={job.id} job={job} projectId={bundle.project.id} shotLabel={shot ? `镜头 ${shot.ordinal}` : "未绑定镜头"} selected={selected} currentOutput={currentOutput} preview={() => setVideoPreview({ name, url: projectInlineUrl(bundle.project.id, "job", job.id), downloadUrl: projectDownloadUrl(bundle.project.id, "job", job.id), provider })} adopt={shot ? () => action(`select-${job.id}`, () => selectRenderJob(bundle.project.id, job.id), `镜头 ${shot.ordinal} 已采用此版本；对应尾帧已锁定。`) : undefined} cancel={() => action(`cancel-${job.id}`, () => cancelRenderJob(bundle.project.id, job.id), "已发送取消请求")} remove={() => action(`delete-${job.id}`, () => deleteRenderJob(bundle.project.id, job.id), "已删除该渲染任务记录")} />;
        })}</div>}</section>
        {keyframeEditor && editorShot && <div className="studio-modal" role="dialog" aria-modal="true" aria-labelledby="keyframe-editor-title" onMouseDown={() => setKeyframeEditor(null)}>
            <section className="studio-dialog my-0 flex max-h-[calc(100dvh-2rem)] min-h-0 max-w-5xl flex-col overflow-hidden" onMouseDown={(event) => event.stopPropagation()}>
                <div className="shrink-0">
                    <div className="mb-5 flex items-start justify-between gap-4">
                    <div><p className="studio-kicker">KEYFRAME REVISION</p><h2 id="keyframe-editor-title" className="text-2xl font-semibold">镜头 {editorShot.ordinal} · 首帧 Prompt 与重新生成</h2><p className="mt-1 text-sm leading-6 text-white/45">可先修改完整 Prompt，再补充本次修改建议，并决定是否把当前首帧作为视觉参考。</p></div>
                    <button type="button" className="rounded-lg p-2 text-white/35 hover:bg-white/8 hover:text-white" aria-label="关闭" onClick={() => setKeyframeEditor(null)}><X size={20} /></button>
                    </div>
                </div>
                <div className="min-h-0 flex-1 overflow-y-auto pr-1">
                    <div className="grid gap-5 lg:grid-cols-[minmax(260px,.72fr)_minmax(0,1.28fr)]">
                    <div>
                        <div className="flex aspect-video items-center justify-center overflow-hidden rounded-xl border border-white/10 bg-black/35">
                            {editorKeyframe ? <img className="h-full w-full object-contain" src={projectDownloadUrl(bundle.project.id, "asset", editorKeyframe.id)} alt={`镜头 ${editorShot.ordinal} 当前首帧`} /> : <div className="flex flex-col items-center gap-2 text-white/25"><ImageIcon size={30} /><span className="text-xs">当前还没有首帧图</span></div>}
                        </div>
                        {editorKeyframe && <button type="button" className="studio-secondary mt-3 w-full" onClick={() => setKeyframePreview({ name: `镜头 ${editorShot.ordinal} 当前首帧`, url: projectDownloadUrl(bundle.project.id, "asset", editorKeyframe.id), description: editorKeyframe.description || keyframeEditor.prompt })}><Maximize2 size={14} />查看当前原图</button>}
                        <div className="mt-4 rounded-lg border border-cyan-300/12 bg-cyan-300/[.03] p-3 text-xs leading-5 text-cyan-50/50">首帧参考与视频参考分别保存。本次首帧生成只提交右侧勾选的场景母版、人物形象图和固定物品图；“基于当前图修改”还会把当前首帧置于参考图 1。</div>
                    </div>
                    <div className="space-y-4">
                        <div className="rounded-xl border border-cyan-300/12 bg-cyan-300/[.025] p-3">
                            <div className="flex flex-wrap items-center justify-between gap-2"><div><span className="studio-label">本次首帧实际参考素材与编号</span><p className="text-[11px] leading-5 text-white/35">编号就是提交给图片模型的顺序；取消勾选后不会发送该图。</p></div><button type="button" className="studio-secondary px-2.5 py-1.5 text-xs" onClick={() => setKeyframeEditor((current) => current ? { ...current, referenceAssetIds: null } : current)}>恢复自动选择</button></div>
                            <div className="mt-2 space-y-1.5 rounded-lg border border-white/8 bg-black/15 p-2.5 text-xs">
                                {keyframeEditor.revisionMode === "iterate" && editorKeyframe && <div className="flex items-center gap-2 text-cyan-100/70"><span className="rounded bg-cyan-300/10 px-1.5 py-0.5 font-mono">参考图 1</span><span>当前首帧 · 仅“基于当前图修改”时提交</span></div>}
                                {selectedKeyframeMaterials.map((asset, index) => <div key={asset.id} className="flex items-center gap-2 text-white/60"><span className="rounded bg-white/6 px-1.5 py-0.5 font-mono">参考图 {index + 1 + (keyframeEditor.revisionMode === "iterate" && editorKeyframe ? 1 : 0)}</span><span className="min-w-0 flex-1 truncate">{asset.name}</span><span className="text-white/25">{assetRoleLabel(asset.role)}</span></div>)}
                                {selectedKeyframeMaterials.length === 0 && !(keyframeEditor.revisionMode === "iterate" && editorKeyframe) && <p className="text-amber-100/60">当前不提交任何图片参考，只使用文字 Prompt。</p>}
                            </div>
                            <div className="mt-3 max-h-44 space-y-1 overflow-y-auto rounded-lg border border-white/8 p-2">
                                {keyframeMaterialOptions.map((asset) => <label key={asset.id} className="flex items-center gap-2 rounded-md px-1 py-1 text-xs text-white/60 hover:bg-white/[.03]"><input type="checkbox" checked={effectiveKeyframeReferenceIds.includes(asset.id)} onChange={() => toggleKeyframeMaterial(asset.id)} /><span className="min-w-0 flex-1 truncate">{asset.name}</span><span className="text-white/25">{assetRoleLabel(asset.role)}</span></label>)}
                                {!keyframeMaterials && <p className="px-1 py-2 text-white/30">正在读取该镜头的自动参考清单…</p>}
                            </div>
                            <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]"><span className="studio-status">{keyframeEditor.referenceAssetIds === null ? "自动选择" : "手动选择"}</span>{keyframeMaterials?.warnings.map((warning) => <span key={warning} className="text-amber-100/65">{warning}</span>)}</div>
                        </div>
                        <label><span className="studio-label">首帧图详细 Prompt（可直接修改）</span><textarea className="studio-input min-h-44 resize-y" maxLength={12000} value={keyframeEditor.prompt} onChange={(event) => setKeyframeEditor((current) => current ? { ...current, prompt: event.target.value } : current)} placeholder="描述人物、动作瞬间、环境、构图、机位、光线、色彩和材质…" /></label>
                        <div className="flex justify-end text-[11px] text-white/25">{keyframeEditor.prompt.length}/12000</div>
                        <label><span className="studio-label">本次修改建议（可选，只写变化）</span><textarea className="studio-input min-h-28 resize-y" maxLength={4000} value={keyframeEditor.suggestions} onChange={(event) => setKeyframeEditor((current) => current ? { ...current, suggestions: event.target.value } : current)} placeholder="只填写相对上方 Prompt 需要改变的内容，不要重复粘贴完整 Prompt。例如：人物表情更克制；门口增加逆光；去掉右侧多余人物……" /></label>
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
                </div>
                <div className="mt-6 flex shrink-0 flex-wrap items-center justify-between gap-3 border-t border-white/8 pt-5">
                    <p className="text-xs text-white/30">点击生成后可直接关闭弹窗继续操作；后台生成成功后会保留旧图文件，并把新图自动绑定为本镜 I2V 首帧。</p>
                    <div className="flex flex-wrap gap-2"><button type="button" className="studio-secondary" disabled={keyframeBusy(editorShot) || !keyframeEditor.prompt.trim()} onClick={() => void saveKeyframePrompt()}>{busyTasks.has(`keyframe-prompt-${editorShot.id}`) ? <Loader2 className="animate-spin" size={15} /> : <Save size={15} />}仅保存 Prompt</button><button type="button" className="studio-primary" disabled={keyframeBusy(editorShot) || !keyframeEditor.prompt.trim()} onClick={() => void regenerateKeyframe()}>{pendingKeyframes.includes(editorShot.id) ? <Loader2 className="animate-spin" size={15} /> : <Sparkles size={15} />}{pendingKeyframes.includes(editorShot.id) ? "后台生成中…" : editorKeyframe ? "按以上设置重新生成" : "按以上设置生成首帧"}</button></div>
                </div>
            </section>
        </div>}
        {keyframePreview && <MediaPreview preview={keyframePreview} onClose={() => setKeyframePreview(null)} />}
        {videoPreview && <VideoPreview preview={videoPreview} onClose={() => setVideoPreview(null)} />}
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

function AssetCard({ asset, projectId, selected = false, select, preview, remove }: { asset: Asset; projectId: string; selected?: boolean; select?: () => void; preview: () => void; remove: () => Promise<void> }) {
    const url = projectDownloadUrl(projectId, "asset", asset.id);
    return <div className={`group relative overflow-hidden rounded-lg border bg-black/20 ${selected ? "border-cyan-300/55 ring-1 ring-cyan-300/25" : "border-white/8"}`}>{select && <button type="button" className={`absolute left-2 top-2 z-20 flex h-7 w-7 items-center justify-center rounded-md border backdrop-blur ${selected ? "border-cyan-200 bg-cyan-300 text-black" : "border-white/20 bg-black/65 text-white/55"}`} title={selected ? "取消选择" : "选择素材"} onClick={select}>{selected ? <Check size={15} /> : <Plus size={14} />}</button>}<div className="flex aspect-video items-center justify-center bg-black/40">{asset.type === "image" ? <button type="button" className="relative h-full w-full cursor-zoom-in overflow-hidden" title="放大查看原图" onClick={preview}><img className="h-full w-full object-cover transition-transform duration-200 group-hover:scale-[1.02]" src={url} alt={asset.name} /><span className="absolute bottom-2 right-2 inline-flex items-center gap-1.5 rounded-md bg-black/70 px-2.5 py-1.5 text-xs text-white/75 opacity-0 backdrop-blur transition-opacity group-hover:opacity-100"><Maximize2 size={13} />查看原图</span></button> : asset.type === "video" ? <video className="h-full w-full object-cover" src={url} controls /> : asset.type === "audio" ? <audio className="w-[90%]" src={url} controls /> : <FileText className="text-white/25" />}</div><div className="flex items-start gap-2 p-3"><div className="min-w-0 flex-1"><strong className="block truncate text-sm">{asset.name}</strong><p className="text-[11px] uppercase text-white/30">{asset.type} · {asset.role} · {(asset.size_bytes / 1024 / 1024).toFixed(1)}MB</p></div>{asset.type === "image" && <button type="button" className="rounded p-1.5 text-white/30 hover:bg-white/8 hover:text-cyan-200" title="查看原图" onClick={preview}><Maximize2 size={14} /></button>}<button type="button" className="rounded p-1.5 text-white/20 opacity-0 hover:text-red-300 group-hover:opacity-100" title="删除素材" onClick={() => void remove()}><Trash2 size={14} /></button></div></div>;
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

function VideoPreview({ preview, onClose }: { preview: VideoPreviewState; onClose: () => void }) {
    useEffect(() => {
        const closeOnEscape = (event: KeyboardEvent) => event.key === "Escape" && onClose();
        window.addEventListener("keydown", closeOnEscape);
        return () => window.removeEventListener("keydown", closeOnEscape);
    }, [onClose]);
    return <div className="studio-modal z-[120]" role="dialog" aria-modal="true" aria-label={`${preview.name} 视频预览`} onMouseDown={onClose}>
        <section className="flex max-h-[94vh] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-white/12 bg-[#0d1017] shadow-2xl" onMouseDown={(event) => event.stopPropagation()}>
            <div className="flex items-center gap-3 border-b border-white/8 px-4 py-3 md:px-5"><div className="min-w-0 flex-1"><strong className="block truncate text-sm">{preview.name}</strong><span className="text-xs text-white/30">{preview.provider} · 在线预览</span></div><a className="studio-secondary px-3 py-2 text-xs" href={preview.downloadUrl}><Download size={14} />下载原片</a><button type="button" className="rounded-lg p-2 text-white/40 hover:bg-white/8 hover:text-white" aria-label="关闭视频预览" onClick={onClose}><X size={20} /></button></div>
            <div className="flex min-h-0 flex-1 items-center justify-center bg-black p-2 md:p-4"><video className="max-h-[78vh] w-full rounded-lg bg-black object-contain" src={preview.url} controls autoPlay playsInline preload="metadata">当前浏览器未显示内嵌播放器，请使用“下载原片”。</video></div>
        </section>
    </div>;
}

function formatJobTime(iso: string | null | undefined): string {
    if (!iso) return "-";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "-";
    return date.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

function JobRow({ job, projectId, shotLabel, selected, currentOutput, preview, adopt, cancel, remove }: { job: RenderJob; projectId: string; shotLabel: string; selected: boolean; currentOutput: boolean; preview: () => void; adopt?: () => Promise<void>; cancel: () => Promise<void>; remove: () => Promise<void> }) {
    const color = job.status === "completed" ? "text-emerald-300" : job.status === "failed" ? "text-red-300" : ACTIVE_JOBS.has(job.status) ? "text-cyan-200" : "text-white/40";
    const params = (job.input_snapshot?.h3_parameters || {}) as Record<string, unknown>;
    const seedance = (job.input_snapshot?.seedance || {}) as Record<string, unknown>;
    const parameterText = job.provider === "ark_seedance"
        ? `${seedance.model_label || seedance.model_id || "Seedance"} · ${seedance.resolution || "720p"} · ${seedance.duration || "-"}s · ${seedance.generate_audio === false ? "无声" : "同步声音"}`
        : params.width ? `${params.width}×${params.height} · ${params.frames}帧 · ${params.steps}步 · ${params.scheduler} · ${params.model_profile || "旧模型"}+${params.text_encoder_profile || "旧编码器"}` : "历史任务默认参数";
    const providerLabel = job.provider === "ark_seedance" ? "SEEDANCE API" : job.provider === "atlas_h3" ? "ATLAS H3 API" : job.provider === "metaso_h3" ? "METASO H3 API" : job.mode?.toUpperCase() || "H3";
    return <div className={`rounded-lg border p-3 ${selected ? "border-emerald-300/25 bg-emerald-300/[.035]" : "border-white/8 bg-black/15"}`}><div className="flex flex-wrap items-center gap-3"><span className={`w-24 text-xs font-semibold uppercase ${color}`}>{job.status}</span>{selected ? <span className="inline-flex items-center gap-1 rounded-md bg-emerald-300/12 px-2 py-1 text-[11px] font-semibold text-emerald-200"><Check size={12} />已采用</span> : currentOutput ? <span className="rounded-md bg-cyan-300/10 px-2 py-1 text-[11px] text-cyan-100/65">当前自动版本</span> : null}<div className="h-1.5 min-w-32 flex-1 overflow-hidden rounded bg-white/8"><div className="h-full bg-cyan-300" style={{ width: `${Math.round(job.progress * 100)}%` }} /></div><span className="w-10 text-right font-mono text-xs text-white/35">{Math.round(job.progress * 100)}%</span>{job.status === "completed" && <div className="flex gap-1.5"><button type="button" className="studio-secondary px-2.5 py-1.5 text-xs" onClick={preview}><Play size={13} />预览</button><a className="studio-secondary px-2.5 py-1.5 text-xs" href={projectDownloadUrl(projectId, "job", job.id)}><Download size={13} />下载</a>{!selected && adopt && <button type="button" className="studio-primary px-2.5 py-1.5 text-xs" onClick={() => void adopt()}><PackageCheck size={13} />{currentOutput ? "锁定此版" : "采用此版本"}</button>}</div>}{ACTIVE_JOBS.has(job.status) && <button className="rounded p-1.5 text-white/30 hover:text-red-300" title="取消该任务" onClick={() => void cancel()}><Trash2 size={14} /></button>}{!ACTIVE_JOBS.has(job.status) && !selected && <button className="rounded p-1.5 text-white/30 hover:text-red-300" title="删除该任务记录" onClick={() => void remove()}><Trash2 size={14} /></button>}</div>{job.error && <p className={`mt-2 text-xs ${ACTIVE_JOBS.has(job.status) ? "text-cyan-100/55" : "text-red-300/80"}`}>{job.error}</p>}<p className="mt-1 text-[11px] text-white/25">{shotLabel} · {providerLabel} · {parameterText} · seed {job.seed} · 预估 ¥{(job.estimated_cost || 0).toFixed(2)} · {job.elapsed_seconds ? `${job.elapsed_seconds.toFixed(0)}s` : "等待计时"} · 开始生成 {formatJobTime(job.started_at || job.created_at)}</p></div>;
}

function DeliveryCard({ delivery, projectId }: { delivery: Delivery; projectId: string }) {
    const url = projectDownloadUrl(projectId, "delivery", delivery.id);
    return <div className="rounded-xl border border-white/8 bg-black/20 p-4"><video className="mb-4 aspect-video w-full rounded-lg bg-black" src={delivery.preview_path ? projectDownloadUrl(projectId, "preview", delivery.id) : url} controls /><div className="flex items-center justify-between"><div><strong>最终成片</strong><p className="text-xs text-white/35">{delivery.duration_seconds.toFixed(1)} 秒 · QC {delivery.qc_report?.passed === false ? "有警告" : "通过"}</p></div><div className="flex gap-2">{delivery.preview_path && <a className="studio-secondary" href={projectDownloadUrl(projectId, "preview", delivery.id)}><Download size={15} />预览版</a>}<a className="studio-primary" href={url}><Download size={15} />原片</a></div></div></div>;
}

function Field({ label, hint, wide, children }: { label: string; hint?: string; wide?: boolean; children: React.ReactNode }) { return <label className={wide ? "md:col-span-2" : ""}><span className="studio-label">{label}</span><div className="studio-field">{children}</div>{hint && <span className="mt-1.5 block text-[11px] leading-5 text-white/32">{hint}</span>}</label>; }
function assetRoleLabel(role: AssetRole) {
    return ({ character: "人物形象", prop: "固定物品", style: "画风参考", scene: "场景参考", keyframe: "首帧", last_frame: "尾帧", motion: "动作/运镜", voice: "声音参考", music: "背景音乐", sound_effect: "音效", output: "输出", other: "其他" } as Record<AssetRole, string>)[role] || role;
}
function toggle(values: string[], value: string) { return values.includes(value) ? values.filter((item) => item !== value) : [...values, value]; }
type BusyState = ReadonlySet<string>;
type Action = (key: string, task: () => Promise<unknown>, success: string, reload?: boolean) => Promise<void>;
