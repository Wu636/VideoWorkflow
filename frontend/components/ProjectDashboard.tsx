"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { ArchiveRestore, Clapperboard, Clock3, FileClock, Film, Loader2, Plus, Server, Settings2, Sparkles, Trash2 } from "lucide-react";

import { createProject, deleteProject, importLegacySession, listLegacySessions, listProjects, type LegacySession } from "@/lib/api";
import type { Project } from "@/types";

const STATUS_LABEL: Record<string, string> = {
    brief_draft: "需求草稿",
    storyboard_draft: "分镜草稿",
    storyboard_review: "客户审阅",
    storyboard_approved: "分镜已确认",
    keyframes_review: "分镜图审阅",
    render_plan_approved: "待生成视频",
    rendering: "生成中",
    clips_review: "视频审阅",
    editing: "剪辑中",
    final_review: "成片审阅",
    delivered: "已交付",
};

export default function ProjectDashboard() {
    const router = useRouter();
    const [projects, setProjects] = useState<Project[]>([]);
    const [loading, setLoading] = useState(true);
    const [creating, setCreating] = useState(false);
    const [showCreate, setShowCreate] = useState(false);
    const [showImport, setShowImport] = useState(false);
    const [legacy, setLegacy] = useState<LegacySession[]>([]);
    const [error, setError] = useState("");
    const [form, setForm] = useState({
        title: "",
        client_name: "",
        story: "",
        visual_style: "",
        target_duration_seconds: 30,
        aspect_ratio: "16:9",
    });

    const reload = async () => {
        try {
            setProjects(await listProjects());
            setError("");
        } catch (caught) {
            setError(caught instanceof Error ? caught.message : String(caught));
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void reload();
    }, []);

    const activeCount = useMemo(() => projects.filter((item) => item.status !== "delivered").length, [projects]);

    const submit = async () => {
        if (!form.title.trim() || !form.story.trim()) return;
        setCreating(true);
        try {
            const project = await createProject({
                ...form,
            });
            router.push(`/projects/${project.id}`);
        } catch (caught) {
            setError(caught instanceof Error ? caught.message : String(caught));
        } finally {
            setCreating(false);
        }
    };

    const openImport = async () => {
        setShowImport(true);
        try {
            setLegacy(await listLegacySessions());
        } catch (caught) {
            setError(caught instanceof Error ? caught.message : String(caught));
        }
    };

    const importSession = async (sessionId: string) => {
        setCreating(true);
        try {
            const project = await importLegacySession(sessionId);
            router.push(`/projects/${project.id}`);
        } catch (caught) {
            setError(caught instanceof Error ? caught.message : String(caught));
        } finally {
            setCreating(false);
        }
    };

    const remove = async (project: Project) => {
        if (!window.confirm(`确定删除项目「${project.brief.title}」及数据库记录吗？`)) return;
        await deleteProject(project.id);
        await reload();
    };

    return (
        <main className="min-h-screen bg-[#080a0f] text-white">
            <header className="border-b border-white/8 bg-[#0d1017]/95 px-5 py-4 md:px-10">
                <div className="mx-auto flex max-w-[1500px] items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="rounded-xl bg-cyan-400 p-2 text-black"><Clapperboard size={23} /></div>
                        <div>
                            <h1 className="text-lg font-bold tracking-wide">VideoWorkflow Studio</h1>
                            <p className="text-xs text-white/45">AI 视频项目生产与 MiniMax H3 调度台</p>
                        </div>
                    </div>
                    <div className="flex gap-2"><Link href="/settings" className="studio-secondary px-3"><Settings2 size={16} /><span className="hidden xl:inline">模型设置</span></Link><Link href="/logs" className="studio-secondary px-3"><FileClock size={16} /><span className="hidden xl:inline">运行日志</span></Link><button className="studio-secondary" onClick={() => void openImport()}><ArchiveRestore size={16} /> <span className="hidden md:inline">导入旧项目</span></button><button className="studio-primary" onClick={() => setShowCreate(true)}><Plus size={17} /> 新建项目</button></div>
                </div>
            </header>

            <section className="mx-auto max-w-[1500px] px-5 py-8 md:px-10">
                <div className="mb-8 grid gap-4 md:grid-cols-3">
                    <Metric icon={<Film size={20} />} label="项目总数" value={String(projects.length)} />
                    <Metric icon={<Clock3 size={20} />} label="进行中" value={String(activeCount)} />
                    <Metric icon={<Server size={20} />} label="云端渲染" value="按需开机" detail="开发期间可保持关机" />
                </div>

                <div className="mb-5 flex items-end justify-between">
                    <div>
                        <p className="studio-kicker">PROJECTS</p>
                        <h2 className="text-2xl font-semibold">制作项目</h2>
                    </div>
                    <Link href="/legacy" className="text-sm text-white/45 hover:text-cyan-300">保留的旧版工作台 →</Link>
                </div>

                {error && <div className="studio-error mb-5">{error}</div>}
                {loading ? (
                    <div className="flex min-h-64 items-center justify-center text-white/50"><Loader2 className="mr-2 animate-spin" />载入项目…</div>
                ) : projects.length === 0 ? (
                    <button onClick={() => setShowCreate(true)} className="studio-empty w-full">
                        <Sparkles size={30} className="text-cyan-300" />
                        <strong>创建第一个 AI 视频项目</strong>
                        <span>录入客户故事、风格和时长，从分镜设计开始完整生产</span>
                    </button>
                ) : (
                    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                        {projects.map((project) => (
                            <article key={project.id} className="studio-card group">
                                <Link href={`/projects/${project.id}`} className="block">
                                    <div className="mb-6 flex items-start justify-between gap-3">
                                        <span className="studio-status">{STATUS_LABEL[project.status] || project.status}</span>
                                        <span className="font-mono text-[11px] text-white/30">V{project.storyboard_version}</span>
                                    </div>
                                    <h3 className="mb-1 truncate text-xl font-semibold group-hover:text-cyan-200">{project.brief.title}</h3>
                                    <p className="mb-5 text-sm text-white/45">{project.brief.client_name || "未填写客户"}</p>
                                    <p className="line-clamp-3 min-h-[63px] text-sm leading-5 text-white/65">{project.brief.story}</p>
                                    <div className="mt-6 flex gap-4 border-t border-white/8 pt-4 text-xs text-white/45">
                                        <span>{project.brief.target_duration_seconds}s</span>
                                        <span>{project.brief.aspect_ratio}</span>
                                        <span>{project.brief.width}×{project.brief.height}</span>
                                    </div>
                                </Link>
                                <button className="absolute bottom-3 right-3 rounded p-2 text-white/20 opacity-0 hover:bg-red-500/10 hover:text-red-300 group-hover:opacity-100" onClick={() => void remove(project)} aria-label="删除项目">
                                    <Trash2 size={15} />
                                </button>
                            </article>
                        ))}
                    </div>
                )}
            </section>

            {showCreate && (
                <div className="studio-modal" onMouseDown={() => setShowCreate(false)}>
                    <section className="studio-dialog" onMouseDown={(event) => event.stopPropagation()}>
                        <p className="studio-kicker">NEW PRODUCTION</p>
                        <h2 className="mb-6 text-2xl font-semibold">建立客户项目</h2>
                        <div className="grid gap-4 md:grid-cols-2">
                            <Field label="项目名称"><input value={form.title} onChange={(event) => setForm({ ...form, title: event.target.value })} placeholder="例如：品牌微电影《回家》" /></Field>
                            <Field label="客户名称"><input value={form.client_name} onChange={(event) => setForm({ ...form, client_name: event.target.value })} placeholder="选填" /></Field>
                            <Field label="目标时长（秒）"><input type="number" min={1} value={form.target_duration_seconds} onChange={(event) => setForm({ ...form, target_duration_seconds: Number(event.target.value) })} /></Field>
                            <Field label="画幅"><select value={form.aspect_ratio} onChange={(event) => setForm({ ...form, aspect_ratio: event.target.value })}><option>16:9</option><option>9:16</option><option>1:1</option></select></Field>
                            <Field label="客户故事/剧情脚本" wide><textarea rows={7} value={form.story} onChange={(event) => setForm({ ...form, story: event.target.value })} placeholder="粘贴客户的初步剧情、人物关系、重要台词和必须出现的内容…" /></Field>
                            <Field label="期望画风" wide><textarea rows={3} value={form.visual_style} onChange={(event) => setForm({ ...form, visual_style: event.target.value })} placeholder="例如：东方奇幻、电影级光影、写实人物、冷青橙调…" /></Field>
                        </div>
                        <div className="mt-7 flex justify-end gap-3">
                            <button className="studio-secondary" onClick={() => setShowCreate(false)}>取消</button>
                            <button className="studio-primary" disabled={creating || !form.title.trim() || !form.story.trim()} onClick={() => void submit()}>{creating ? <Loader2 size={16} className="animate-spin" /> : <Plus size={16} />} 创建并进入</button>
                        </div>
                    </section>
                </div>
            )}
            {showImport && (
                <div className="studio-modal" onMouseDown={() => setShowImport(false)}>
                    <section className="studio-dialog max-w-2xl" onMouseDown={(event) => event.stopPropagation()}>
                        <p className="studio-kicker">LEGACY MIGRATION</p><h2 className="mb-2 text-2xl font-semibold">导入旧版输出项目</h2><p className="mb-5 text-sm text-white/40">自动读取 outputs/*/script.json，并复制已有分镜图和视频到新项目结构。</p>
                        <div className="max-h-[55vh] space-y-2 overflow-y-auto">{legacy.length === 0 ? <p className="rounded-lg border border-dashed border-white/10 p-6 text-center text-sm text-white/35">没有发现可导入的旧项目</p> : legacy.map((item) => <button key={item.session_id} disabled={creating} className="flex w-full items-center gap-4 rounded-lg border border-white/8 bg-black/15 p-4 text-left hover:border-cyan-300/25" onClick={() => void importSession(item.session_id)}><ArchiveRestore size={18} className="text-cyan-300" /><div className="min-w-0 flex-1"><strong className="block truncate">{item.topic}</strong><span className="text-xs text-white/35">{item.session_id} · {item.scene_count} 镜 · {item.has_images ? "有分镜图" : "无分镜图"} · {item.has_videos ? "有视频" : "无视频"}</span></div><span className="text-sm text-cyan-200">导入</span></button>)}</div>
                        <div className="mt-5 flex justify-end"><button className="studio-secondary" onClick={() => setShowImport(false)}>关闭</button></div>
                    </section>
                </div>
            )}
        </main>
    );
}

function Metric({ icon, label, value, detail }: { icon: React.ReactNode; label: string; value: string; detail?: string }) {
    return <div className="studio-metric"><span className="text-cyan-300">{icon}</span><div><p className="text-xs text-white/40">{label}</p><strong className="text-xl">{value}</strong>{detail && <p className="text-[11px] text-white/30">{detail}</p>}</div></div>;
}

function Field({ label, wide, children }: { label: string; wide?: boolean; children: React.ReactNode }) {
    return <label className={wide ? "md:col-span-2" : ""}><span className="studio-label">{label}</span><div className="studio-field">{children}</div></label>;
}
