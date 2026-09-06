"use client";

/* eslint-disable @next/next/no-img-element */
import { useEffect, useState } from "react";
import { Check, Clapperboard, Loader2, MessageSquareText, RefreshCw } from "lucide-react";

import { getPublicReview, projectDownloadUrl, submitPublicReview } from "@/lib/api";
import type { ApprovalStatus, ProjectBundle } from "@/types";

export default function PublicReview({ token }: { token: string }) {
    const [bundle, setBundle] = useState<ProjectBundle | null>(null);
    const [loading, setLoading] = useState(true);
    const [sending, setSending] = useState("");
    const [reviewer, setReviewer] = useState("");
    const [comment, setComment] = useState("");
    const [message, setMessage] = useState("");

    useEffect(() => {
        getPublicReview(token).then(setBundle).catch((error) => setMessage(String(error))).finally(() => setLoading(false));
    }, [token]);

    const submit = async (decision: ApprovalStatus, targetType?: "storyboard" | "shot" | "delivery", targetId?: string) => {
        if (!bundle) return;
        const delivery = bundle.deliveries[0];
        const resolvedTargetType = targetType || (delivery ? "delivery" : "storyboard");
        const resolvedTargetId = targetId || delivery?.id || bundle.project.id;
        setSending(`${resolvedTargetType}-${resolvedTargetId}-${decision}`);
        try {
            await submitPublicReview(token, {
                target_type: resolvedTargetType,
                target_id: resolvedTargetId,
                decision,
                comment,
                reviewer,
            });
            setMessage(decision === "approved" ? "反馈已提交：确认通过。" : "修改意见已提交给制作方。" );
            setComment("");
            setBundle(await getPublicReview(token));
        } catch (error) {
            setMessage(error instanceof Error ? error.message : String(error));
        } finally {
            setSending("");
        }
    };

    if (loading) return <div className="flex min-h-screen items-center justify-center bg-[#080a0f] text-white/50"><Loader2 className="mr-2 animate-spin" />载入审片页…</div>;
    if (!bundle) return <div className="min-h-screen bg-[#080a0f] p-8 text-white"><div className="studio-error">{message || "审片链接不存在"}</div></div>;

    const { project, shots, assets, jobs, deliveries } = bundle;
    const reviewingDelivery = deliveries.length > 0;
    return <main className="min-h-screen bg-[#080a0f] text-white">
        <header className="border-b border-white/8 bg-[#0d1017] px-5 py-4"><div className="mx-auto flex max-w-6xl items-center gap-3"><div className="rounded-lg bg-cyan-300 p-2 text-black"><Clapperboard size={20} /></div><div><h1 className="font-semibold">{project.brief.title}</h1><p className="text-xs text-white/40">客户审片 · 分镜 V{project.storyboard_version}</p></div></div></header>
        <div className="mx-auto max-w-6xl px-5 py-8">
            <section className="studio-panel mb-5"><p className="studio-kicker">PROJECT BRIEF</p><h2 className="mb-2 text-2xl font-semibold">{project.brief.title}</h2><p className="whitespace-pre-wrap text-sm leading-7 text-white/65">{project.brief.story}</p><div className="mt-4 flex gap-5 text-xs text-white/35"><span>目标 {project.brief.target_duration_seconds}s</span><span>{project.brief.aspect_ratio}</span><span>{project.brief.visual_style}</span></div></section>
            <div className="space-y-4">{shots.map((shot) => {
                const keyframe = assets.find((asset) => asset.id === shot.keyframe_asset_id);
                const job = jobs.find((item) => item.id === shot.selected_video_job_id && item.status === "completed")
                    || jobs.find((item) => item.shot_id === shot.id && item.status === "completed" && item.output_path === shot.video_path)
                    || jobs.find((item) => item.shot_id === shot.id && item.status === "completed");
                return <article key={shot.id} className="studio-panel"><div className="grid gap-5 md:grid-cols-[280px_1fr]">
                    <div className="flex aspect-video items-center justify-center overflow-hidden rounded-lg bg-black/40">{job ? <video className="h-full w-full object-contain" src={projectDownloadUrl(project.id, "job", job.id)} controls /> : keyframe ? <img className="h-full w-full object-cover" src={projectDownloadUrl(project.id, "asset", keyframe.id)} alt={shot.title} /> : <span className="text-sm text-white/25">等待分镜图/视频</span>}</div>
                    <div><div className="mb-3 flex items-center gap-3"><span className="rounded bg-cyan-300/10 px-2 py-1 font-mono text-sm text-cyan-200">#{shot.ordinal}</span><h3 className="font-semibold">{shot.title}</h3><span className="ml-auto text-xs text-white/35">{shot.duration_seconds}s</span></div><p className="mb-2 text-sm leading-6 text-white/65">{shot.narrative}</p>{shot.dialogue && <p className="mb-3 rounded border-l-2 border-cyan-300/35 bg-cyan-300/5 px-3 py-2 text-sm text-white/55">对白：{shot.dialogue}</p>}<p className="line-clamp-3 text-xs leading-5 text-white/30">{shot.video_prompt}</p><div className="mt-4 flex justify-end gap-2"><button className="studio-secondary" disabled={!!sending} onClick={() => void submit("changes_requested", "shot", shot.id)}><RefreshCw size={14} />此镜需修改</button><button className="studio-primary" disabled={!!sending} onClick={() => void submit("approved", "shot", shot.id)}><Check size={14} />此镜确认</button></div></div>
                </div></article>;
            })}</div>
            {deliveries.length > 0 && <section className="studio-panel mt-5"><p className="studio-kicker">FINAL CUT</p><h2 className="mb-4 text-xl font-semibold">最终成片</h2><video className="aspect-video w-full rounded-lg bg-black" src={projectDownloadUrl(project.id, deliveries[0].preview_path ? "preview" : "delivery", deliveries[0].id)} controls /></section>}
            <section className="studio-panel sticky bottom-4 mt-5 border-cyan-300/15 shadow-2xl shadow-black"><p className="mb-3 text-xs font-medium text-cyan-200/70">{reviewingDelivery ? "最终成片确认" : "分镜方案整体确认"}</p><div className="grid gap-3 md:grid-cols-[180px_1fr_auto]"><input className="studio-input" value={reviewer} onChange={(event) => setReviewer(event.target.value)} placeholder="您的称呼" /><textarea className="studio-input" rows={2} value={comment} onChange={(event) => setComment(event.target.value)} placeholder="整体意见或需要调整的细节…" /><div className="flex gap-2"><button className="studio-secondary" disabled={!!sending} onClick={() => void submit("changes_requested")}><MessageSquareText size={15} />需要修改</button><button className="studio-primary" disabled={!!sending} onClick={() => void submit("approved")}>{sending ? <Loader2 className="animate-spin" size={15} /> : <Check size={15} />}整体确认</button></div></div>{message && <p className="mt-3 text-sm text-cyan-200">{message}</p>}</section>
        </div>
    </main>;
}
