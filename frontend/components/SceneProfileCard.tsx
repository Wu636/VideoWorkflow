"use client";

/* eslint-disable @next/next/no-img-element */
import { useState } from "react";
import { FilePenLine, ImageIcon, Loader2, Maximize2, RefreshCw, Save, X } from "lucide-react";
import { generateSceneReference, getSceneReferencePrompt, previewSceneReferencePrompt, projectInlineUrl, retrySceneReferenceDownload, updateSceneProfile } from "@/lib/api";
import type { Asset, SceneProfile } from "@/types";

type Props = {
    projectId: string;
    profile: SceneProfile;
    assets: Asset[];
    ordinals: number[];
    busy: ReadonlySet<string>;
    action: (key: string, task: () => Promise<unknown>, success: string, reload?: boolean) => Promise<void>;
    preview: (asset: Asset) => void;
};

export default function SceneProfileCard({ projectId, profile, assets, ordinals, busy, action, preview }: Props) {
    const [draft, setDraft] = useState<SceneProfile | null>(null);
    const [error, setError] = useState("");
    const [rebuilding, setRebuilding] = useState(false);
    const [imageFailed, setImageFailed] = useState(false);
    const [imageRetry, setImageRetry] = useState(0);
    const taskKey = `scene-ref-${profile.id}`;
    const saving = busy.has(`scene-save-${profile.id}`);
    const active = busy.has(taskKey) || ["generating", "downloading"].includes(profile.reference_status);
    const asset = [...profile.reference_asset_ids].reverse().map((id) => assets.find((item) => item.id === id)).find((item) => item?.type === "image");
    const status = profile.reference_status === "downloading" ? "服务商已出图，正在下载保存" : active ? "正在生成母版图" : profile.reference_status === "download_failed" ? "结果待取回" : profile.reference_status === "failed" ? "上次处理失败" : asset ? "母版图已保存" : "文字档案";

    const openEditor = () => void action(`scene-open-${profile.id}`, async () => {
        const result = await getSceneReferencePrompt(projectId, profile.id);
        setDraft({ ...result.profile, reference_prompt: result.prompt });
        setError("");
    }, "已打开场景档案与完整生图 Prompt", false);

    const submit = (generate: boolean) => {
        if (!draft || saving || active || rebuilding) return;
        const snapshot = draft;
        setError("");
        void action(generate ? taskKey : `scene-save-${profile.id}`, async () => {
            try {
                const saved = await updateSceneProfile(projectId, snapshot);
                setDraft(saved);
                if (generate) {
                    setDraft(null);
                    await generateSceneReference(projectId, saved.id, saved.reference_prompt, saved.version);
                    setImageFailed(false);
                }
            } catch (caught) {
                setError(caught instanceof Error ? caught.message : String(caught));
                throw caught;
            }
        }, generate ? `已生成并保存 ${snapshot.name} 场景母版图` : "场景档案与 Prompt 已保存；已有图片和 H3 原文保持不变");
    };

    const rebuild = async () => {
        if (!draft) return;
        setRebuilding(true);
        setError("");
        try {
            const result = await previewSceneReferencePrompt(projectId, draft);
            setDraft((current) => current ? { ...current, reference_prompt: result.prompt } : current);
        } catch (caught) {
            setError(caught instanceof Error ? caught.message : String(caught));
        } finally { setRebuilding(false); }
    };

    return <article className="overflow-hidden rounded-xl border border-white/8 bg-black/15">
        {asset ? <div className="relative aspect-video overflow-hidden bg-black/30">
            <button type="button" className="block h-full w-full" onClick={() => preview(asset)}>
                <img key={`${asset.id}-${imageRetry}`} className="h-full w-full object-contain" src={`${projectInlineUrl(projectId, "asset", asset.id)}&retry=${imageRetry}`} alt={`${profile.name} 场景母版`} onError={() => setImageFailed(true)} onLoad={() => setImageFailed(false)} />
                <span className="absolute bottom-2 right-2 inline-flex items-center gap-1 rounded bg-black/70 px-2 py-1 text-xs"><Maximize2 size={12} />查看母版图</span>
            </button>
            {imageFailed && <button className="studio-secondary absolute left-3 top-3 text-xs" onClick={() => { setImageFailed(false); setImageRetry((value) => value + 1); }}>图片加载失败，点击重新加载</button>}
        </div> : <div className="flex aspect-video items-center justify-center border-b border-dashed border-white/8 text-center text-sm leading-6 text-white/40">
            <span>{active ? <Loader2 className="mx-auto mb-2 animate-spin text-cyan-200" /> : <ImageIcon className="mx-auto mb-2" />} {status}<br />{active ? "刷新页面后仍可查看进度" : profile.reference_status === "download_failed" ? "无需重复付费生图，请重试取回结果" : "编辑档案并确认 Prompt 后再生成"}</span>
        </div>}
        <div className="p-4">
            <div className="flex items-start justify-between gap-3"><strong className="text-sm">{profile.name}</strong><span className="studio-status">{status}</span></div>
            <p className="mt-1 text-xs text-cyan-100/40">适用镜头：{ordinals.length ? ordinals.map((n) => `#${n}`).join("、") : "当前分镜未绑定"}</p>
            <p className="mt-3 whitespace-pre-wrap text-xs leading-5 text-white/50">固定空间：{profile.description || "暂无描述"}</p>
            <p className="mt-2 whitespace-pre-wrap text-xs leading-5 text-white/40">连续性规则：{profile.continuity_notes || "暂无规则"}</p>
            {profile.reference_error && <p role="alert" className="studio-error mt-3 break-words text-xs">{profile.reference_error}</p>}
            {profile.reference_status === "download_failed" && <button className="studio-primary mt-3 w-full text-xs" disabled={active} onClick={() => void action(taskKey, () => retrySceneReferenceDownload(projectId, profile.id), "已有生图结果已取回并保存，没有重新提交生成")}><RefreshCw size={14} />取回已有结果 · 不重新生图</button>}
            <button className="studio-secondary mt-4 w-full text-xs" disabled={active || busy.has(`scene-open-${profile.id}`)} onClick={openEditor}><FilePenLine size={14} />编辑档案 / 查看与修改生图 Prompt</button>
            {asset && <p className="mt-2 text-[11px] text-white/30">修改档案或 Prompt 后，已生成的图片仍保留；确认后可另行生成新版。</p>}
        </div>
        {draft && <div className="studio-modal" onMouseDown={() => { if (!saving && !active && !rebuilding) setDraft(null); }}>
            <section className="studio-dialog max-w-3xl" style={{ maxHeight: "calc(100dvh - 2rem)", overflowY: "auto", marginBlock: 0 }} role="dialog" aria-modal="true" aria-label="编辑场景档案与母版图 Prompt" onMouseDown={(event) => event.stopPropagation()}>
                <div className="flex items-center justify-between gap-3"><h2 className="text-xl font-semibold">场景档案与母版图 Prompt</h2><button aria-label="关闭场景编辑" disabled={saving || active || rebuilding} onClick={() => setDraft(null)}><X size={18} /></button></div>
                <p className="mt-2 text-xs leading-5 text-white/45">先编辑场景，再核对完整 Prompt。实际提交以此框原文为准，不额外追加全局灯光、构图或人物设定。</p>
                {error && <p role="alert" className="studio-error mt-3">{error}</p>}
                <fieldset disabled={saving || active || rebuilding} className="mt-4 space-y-3">
                    <label className="block"><span className="studio-label">场景名称</span><input className="studio-input" maxLength={200} value={draft.name} onChange={(event) => setDraft({ ...draft, name: event.target.value })} /></label>
                    <label className="block"><span className="studio-label">固定空间 / 场景描述（含本场景光影）</span><textarea className="studio-input" rows={4} maxLength={12000} value={draft.description} onChange={(event) => setDraft({ ...draft, description: event.target.value })} /></label>
                    <label className="block"><span className="studio-label">连续性规则</span><textarea className="studio-input" rows={3} maxLength={12000} value={draft.continuity_notes} onChange={(event) => setDraft({ ...draft, continuity_notes: event.target.value })} /></label>
                    <div className="flex items-center justify-between gap-2"><span className="studio-label">最终生图 Prompt（可自由修改）</span><button type="button" className="studio-secondary text-xs" onClick={() => void rebuild()}><RefreshCw size={12} />按当前档案重建 Prompt · 免费</button></div>
                    <label className="block"><span className="sr-only">最终生图 Prompt</span><textarea className="studio-input" rows={10} maxLength={12000} value={draft.reference_prompt} onChange={(event) => setDraft({ ...draft, reference_prompt: event.target.value })} /></label>
                    <p className="text-[11px] leading-5 text-white/35">修改上方档案后，可点击“重建 Prompt”同步到下方；重建会替换编辑框内的 Prompt。仅保存不调用生图 API。</p>
                </fieldset>
                <div className="mt-5 flex flex-wrap justify-end gap-2"><button className="studio-secondary" disabled={saving || active || rebuilding} onClick={() => setDraft(null)}>关闭</button><button className="studio-secondary" disabled={saving || active || rebuilding || !draft.name.trim() || !draft.description.trim()} onClick={() => submit(false)}><Save size={14} />仅保存档案和 Prompt</button><button className="studio-primary" disabled={saving || active || rebuilding || !draft.name.trim() || !draft.description.trim() || !draft.reference_prompt.trim()} onClick={() => submit(true)}><ImageIcon size={14} />保存并{asset ? "生成新版母版图" : "生成母版图"}</button></div>
            </section>
        </div>}
    </article>;
}
