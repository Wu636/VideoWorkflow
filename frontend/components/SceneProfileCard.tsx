"use client";

/* eslint-disable @next/next/no-img-element */
import { useState } from "react";
import { FilePenLine, ImageIcon, Loader2, Maximize2, RefreshCw, Save, Trash2, Upload, X } from "lucide-react";
import { clearSceneProfileReference, generateSceneReference, getSceneReferencePrompt, previewSceneReferencePrompt, projectInlineUrl, retrySceneReferenceDownload, reviseSceneReference, updateSceneProfile, uploadSceneProfileReference } from "@/lib/api";
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
    const [revisionOpen, setRevisionOpen] = useState(false);
    const [revisionSuggestions, setRevisionSuggestions] = useState("");
    const [revisionError, setRevisionError] = useState("");
    const taskKey = `scene-ref-${profile.id}`;
    const uploadKey = `scene-upload-${profile.id}`;
    const clearKey = `scene-clear-${profile.id}`;
    const saving = busy.has(`scene-save-${profile.id}`);
    const active = busy.has(taskKey) || ["generating", "downloading"].includes(profile.reference_status);
    const uploading = busy.has(uploadKey);
    const clearing = busy.has(clearKey);
    const asset = [...profile.reference_asset_ids].reverse().map((id) => assets.find((item) => item.id === id)).find((item) => item?.type === "image");
    const status = profile.reference_status === "downloading" ? "服务商已出图，正在下载保存" : active ? "正在生成母版图" : profile.reference_status === "download_failed" ? "结果待取回" : profile.reference_status === "failed" ? "上次处理失败" : asset && profile.reference_source === "upload" ? "本地场景图已绑定" : asset ? "AI 母版图已保存" : "文字档案";

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

    const upload = (file: File) => {
        void action(uploadKey, () => uploadSceneProfileReference(projectId, profile.id, file, true), "本地场景图已上传并绑定；后续首帧与视频 Prompt 会优先使用这张图");
    };

    const clearReference = () => {
        if (!asset || active || uploading || clearing) return;
        void action(clearKey, () => clearSceneProfileReference(projectId, profile.id), "已清除场景档案的图片绑定；原文件仍保留在项目素材库");
    };

    const reviseReference = () => {
        if (!asset || !revisionSuggestions.trim() || active) return;
        const sourceAssetId = asset.id;
        const suggestions = revisionSuggestions.trim();
        setRevisionError("");
        void action(taskKey, async () => {
            try {
                await reviseSceneReference(projectId, profile.id, sourceAssetId, suggestions, profile.version);
                setRevisionOpen(false);
                setRevisionSuggestions("");
                setImageFailed(false);
            } catch (caught) {
                setRevisionError(caught instanceof Error ? caught.message : String(caught));
                throw caught;
            }
        }, "已根据建议生成场景图新版本；上一张图保留在素材库");
    };

    return <article className="overflow-hidden rounded-xl border border-white/8 bg-black/15">
        {asset ? <div className="relative aspect-video overflow-hidden bg-black/30">
            <button type="button" className="block h-full w-full" onClick={() => preview(asset)}>
                <img key={`${asset.id}-${imageRetry}`} className="h-full w-full object-contain" src={`${projectInlineUrl(projectId, "asset", asset.id)}&retry=${imageRetry}`} alt={`${profile.name} 场景母版`} onError={() => setImageFailed(true)} onLoad={() => setImageFailed(false)} />
                <span className="absolute bottom-2 right-2 inline-flex items-center gap-1 rounded bg-black/70 px-2 py-1 text-xs"><Maximize2 size={12} />查看母版图</span>
            </button>
            {imageFailed && <button className="studio-secondary absolute left-3 top-3 text-xs" onClick={() => { setImageFailed(false); setImageRetry((value) => value + 1); }}>图片加载失败，点击重新加载</button>}
        </div> : <div className="flex aspect-video items-center justify-center border-b border-dashed border-white/8 text-center text-sm leading-6 text-white/40">
            <span>{active ? <Loader2 className="mx-auto mb-2 animate-spin text-cyan-200" /> : <ImageIcon className="mx-auto mb-2" />} {status}<br />{active ? "刷新页面后仍可查看进度" : profile.reference_status === "download_failed" ? "无需重复付费生图，请重试取回结果" : "可上传本地场景图，也可编辑档案后再调用 AI 生图"}</span>
        </div>}
        <div className="p-4">
            <div className="flex items-start justify-between gap-3"><strong className="text-sm">{profile.name}</strong><span className="studio-status">{status}</span></div>
            <p className="mt-1 text-xs text-cyan-100/40">适用镜头：{ordinals.length ? ordinals.map((n) => `#${n}`).join("、") : "当前分镜未绑定"}</p>
            <p className="mt-3 whitespace-pre-wrap text-xs leading-5 text-white/50">固定空间：{profile.description || "暂无描述"}</p>
            <p className="mt-2 whitespace-pre-wrap text-xs leading-5 text-white/40">连续性规则：{profile.continuity_notes || "暂无规则"}</p>
            {profile.reference_error && <p role="alert" className="studio-error mt-3 break-words text-xs">{profile.reference_error}</p>}
            {profile.reference_status === "download_failed" && <button className="studio-primary mt-3 w-full text-xs" disabled={active} onClick={() => void action(taskKey, () => retrySceneReferenceDownload(projectId, profile.id), "已有生图结果已取回并保存，没有重新提交生成")}><RefreshCw size={14} />取回已有结果 · 不重新生图</button>}
            <div className="mt-4 grid gap-2 sm:grid-cols-2">
                <label className="studio-secondary cursor-pointer justify-center text-xs"><input className="hidden" type="file" accept="image/png,image/jpeg,image/webp,image/bmp" disabled={active || uploading || clearing} onChange={(event) => { const file = event.target.files?.[0]; if (file) upload(file); event.target.value = ""; }} />{uploading ? <Loader2 className="animate-spin" size={14} /> : <Upload size={14} />}上传本地场景图</label>
                {asset && <button type="button" className="studio-secondary justify-center text-xs" disabled={active || uploading || clearing} onClick={clearReference}>{clearing ? <Loader2 className="animate-spin" size={14} /> : <Trash2 size={14} />}清除图片绑定</button>}
            </div>
            {asset && profile.reference_source === "ai" && <button type="button" className="studio-secondary mt-2 w-full justify-center text-xs" disabled={active || uploading || clearing} onClick={() => { setRevisionError(""); setRevisionOpen(true); }}><RefreshCw size={14} />按建议修改上一张图</button>}
            <button className="studio-secondary mt-4 w-full text-xs" disabled={active || busy.has(`scene-open-${profile.id}`)} onClick={openEditor}><FilePenLine size={14} />编辑档案 / 查看与修改生图 Prompt</button>
            <p className="mt-2 text-[11px] text-white/35">AI 生图后会检查是否有人物；检出时自动纠错一次，可能增加一次生图费用。复查仍有人物的图片不绑定为母版。</p>
            {asset && <p className="mt-2 text-[11px] text-white/30">上传新图会替换当前场景档案绑定；旧文件仍保留在素材库。修改档案或 Prompt 后，已生成的图片也会保留。</p>}
        </div>
        {draft && <div className="studio-modal" onMouseDown={() => { if (!saving && !active && !rebuilding) setDraft(null); }}>
            <section className="studio-dialog max-w-3xl" style={{ maxHeight: "calc(100dvh - 2rem)", overflowY: "auto", marginBlock: 0 }} role="dialog" aria-modal="true" aria-label="编辑场景档案与母版图 Prompt" onMouseDown={(event) => event.stopPropagation()}>
                <div className="flex items-center justify-between gap-3"><h2 className="text-xl font-semibold">场景档案与母版图 Prompt</h2><button aria-label="关闭场景编辑" disabled={saving || active || rebuilding} onClick={() => setDraft(null)}><X size={18} /></button></div>
                <p className="mt-2 text-xs leading-5 text-white/45">生成时会剔除人物及站位片段，追加“纯环境、绝对无人”约束，并在出图后检查画面是否有人；不会追加全局人物设定。</p>
                {error && <p role="alert" className="studio-error mt-3">{error}</p>}
                <fieldset disabled={saving || active || rebuilding} className="mt-4 space-y-3">
                    <label className="block"><span className="studio-label">场景名称</span><input className="studio-input" maxLength={200} value={draft.name} onChange={(event) => setDraft({ ...draft, name: event.target.value })} /></label>
                    <label className="block"><span className="studio-label">固定空间 / 场景描述（含本场景光影）</span><textarea className="studio-input" rows={4} maxLength={12000} value={draft.description} onChange={(event) => setDraft({ ...draft, description: event.target.value })} /></label>
                    <label className="block"><span className="studio-label">连续性规则</span><textarea className="studio-input" rows={3} maxLength={12000} value={draft.continuity_notes} onChange={(event) => setDraft({ ...draft, continuity_notes: event.target.value })} /></label>
                    <div className="flex items-center justify-between gap-2"><span className="studio-label">场景生图基础 Prompt（可修改）</span><button type="button" className="studio-secondary text-xs" onClick={() => void rebuild()}><RefreshCw size={12} />按当前档案重建 Prompt · 免费</button></div>
                    <label className="block"><span className="sr-only">场景生图基础 Prompt</span><textarea className="studio-input" rows={10} maxLength={12000} value={draft.reference_prompt} onChange={(event) => setDraft({ ...draft, reference_prompt: event.target.value })} /></label>
                    <p className="text-[11px] leading-5 text-white/35">修改上方档案后，可点击“重建 Prompt”同步到下方；重建会替换编辑框内的 Prompt。仅保存不调用生图 API。</p>
                </fieldset>
                <div className="mt-5 flex flex-wrap justify-end gap-2"><button className="studio-secondary" disabled={saving || active || rebuilding} onClick={() => setDraft(null)}>关闭</button><button className="studio-secondary" disabled={saving || active || rebuilding || !draft.name.trim() || !draft.description.trim()} onClick={() => submit(false)}><Save size={14} />仅保存档案和 Prompt</button><button className="studio-primary" disabled={saving || active || rebuilding || !draft.name.trim() || !draft.description.trim() || !draft.reference_prompt.trim()} onClick={() => submit(true)}><ImageIcon size={14} />保存并{asset ? "生成新版母版图" : "生成母版图"}</button></div>
            </section>
        </div>}
        {revisionOpen && asset && <div className="studio-modal" onMouseDown={() => { if (!active) setRevisionOpen(false); }}>
            <section className="studio-dialog max-w-xl" role="dialog" aria-modal="true" aria-label="按建议修改上一张场景图" onMouseDown={(event) => event.stopPropagation()}>
                <div className="flex items-center justify-between gap-3"><h2 className="text-xl font-semibold">按建议修改上一张场景图</h2><button aria-label="关闭场景图修改" disabled={active} onClick={() => setRevisionOpen(false)}><X size={18} /></button></div>
                <p className="mt-2 text-xs leading-5 text-white/45">以上一张 AI 场景图为参考，只修改你提出的部分。若旧图含人物，生成时会要求移除人物并补全背景；旧图保留在素材库。</p>
                <img className="mt-4 max-h-52 w-full rounded-lg bg-black/30 object-contain" src={projectInlineUrl(projectId, "asset", asset.id)} alt="上一张场景图" />
                {revisionError && <p role="alert" className="studio-error mt-3">{revisionError}</p>}
                <label className="mt-4 block"><span className="studio-label">这次要修改什么</span><textarea className="studio-input" rows={5} maxLength={4000} disabled={active} placeholder="例如：保留原有建筑结构，把黄昏改成清晨；清除画面中的所有人员。" value={revisionSuggestions} onChange={(event) => setRevisionSuggestions(event.target.value)} /></label>
                <div className="mt-5 flex justify-end gap-2"><button className="studio-secondary" disabled={active} onClick={() => setRevisionOpen(false)}>取消</button><button className="studio-primary" disabled={active || !revisionSuggestions.trim()} onClick={reviseReference}>{active ? <Loader2 className="animate-spin" size={14} /> : <RefreshCw size={14} />}生成修改版</button></div>
            </section>
        </div>}
    </article>;
}
