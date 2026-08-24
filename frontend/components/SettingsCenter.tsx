"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, GitBranch, KeyRound, Loader2, LockKeyhole, Save, Trash2 } from "lucide-react";

import UtilityHeader from "@/components/UtilityHeader";
import { getRuntimeSettings, updateRuntimeSettings } from "@/lib/api";
import type { RuntimeSettingField, RuntimeSettingsPayload } from "@/types";

export default function SettingsCenter() {
    const [payload, setPayload] = useState<RuntimeSettingsPayload | null>(null);
    const [draft, setDraft] = useState<Record<string, unknown>>({});
    const [clearKeys, setClearKeys] = useState<string[]>([]);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState("");
    const [notice, setNotice] = useState("");

    const load = async () => {
        try {
            const result = await getRuntimeSettings();
            setPayload(result);
            setDraft(Object.fromEntries(result.groups.flatMap((group) => group.fields.map((field) => [field.key, field.value]))));
            setError("");
        } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); }
    };
    useEffect(() => { void load(); }, []);

    const save = async () => {
        setBusy(true); setError(""); setNotice("");
        try {
            const result = await updateRuntimeSettings(draft, clearKeys);
            setPayload(result); setClearKeys([]);
            setDraft(Object.fromEntries(result.groups.flatMap((group) => group.fields.map((field) => [field.key, field.value]))));
            setNotice("配置已保存并即时生效，H3 队列连接也已刷新。新密钥不会回显到浏览器。");
        } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); }
        finally { setBusy(false); }
    };

    return <main className="min-h-screen bg-[#080a0f] text-white">
        <UtilityHeader title="模型与接口设置" subtitle="集中管理剧本分析、分镜图、外部视频与 MiniMax H3 接口" />
        <div className="mx-auto max-w-[1500px] px-5 py-8 md:px-10">
            {error && <div className="studio-error mb-5">{error}</div>}
            {notice && <div className="studio-notice mb-5">{notice}</div>}
            {!payload ? <div className="flex min-h-64 items-center justify-center text-white/45"><Loader2 className="mr-2 animate-spin" />读取配置…</div> : <>
                <div className="mb-5 flex flex-wrap items-center justify-between gap-3"><div><p className="studio-kicker">GLOBAL PROVIDERS</p><h2 className="text-2xl font-semibold">统一配置中心</h2><p className="mt-1 text-sm text-white/35">保存在 {payload.path}；密钥字段只显示配置状态。</p></div><button className="studio-primary" disabled={busy} onClick={() => void save()}>{busy ? <Loader2 className="animate-spin" size={16} /> : <Save size={16} />}保存全部配置</button></div>
                <section className="studio-panel mb-5">
                    <div className="mb-5 flex items-start gap-3"><span className="rounded-lg bg-cyan-300/10 p-2 text-cyan-200"><GitBranch size={18} /></span><div><p className="studio-kicker">FUNCTION ROUTING</p><h3 className="text-xl font-semibold">各功能当前优先配置</h3><p className="mt-1 text-sm text-white/35">这里修改的是实际功能路由；选择后点击右上角“保存全部配置”即时生效。</p></div></div>
                    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">{payload.routes.map((route, index) => {
                        const selectedValue = route.setting_key ? String(draft[route.setting_key] ?? route.selected) : route.selected;
                        const changed = route.setting_key ? selectedValue !== route.selected : false;
                        return <article key={route.id} className="rounded-xl border border-white/8 bg-black/15 p-4">
                            <div className="mb-3 flex items-start justify-between gap-3"><span className="font-mono text-[10px] text-cyan-300/45">{String(index + 1).padStart(2, "0")}</span><span className={route.configured ? "inline-flex items-center gap-1 text-[11px] text-emerald-300/70" : "inline-flex items-center gap-1 text-[11px] text-amber-300/75"}>{route.configured ? <CheckCircle2 size={12} /> : <AlertTriangle size={12} />}{route.configured ? "已配置" : "缺少密钥"}</span></div>
                            <h4 className="font-semibold">{route.label}</h4><p className="mt-1 min-h-10 text-xs leading-5 text-white/35">{route.description}</p>
                            <div className="studio-field mt-3">{route.setting_key ? <select value={selectedValue} onChange={(event) => setDraft({ ...draft, [route.setting_key as string]: event.target.value })}>{route.options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select> : <div className="flex h-[42px] items-center gap-2 rounded-lg border border-white/8 bg-white/[.025] px-3 text-sm text-white/55"><LockKeyhole size={13} />{route.effective_label}</div>}</div>
                            <div className="mt-3 rounded-lg border border-cyan-300/8 bg-cyan-300/[.025] p-3"><p className="text-[10px] tracking-wide text-white/30">{changed ? "保存后切换" : "当前优先使用"}</p><strong className={changed ? "mt-1 block text-sm text-amber-200/80" : "mt-1 block text-sm text-cyan-100/80"}>{changed ? route.options.find((item) => item.value === selectedValue)?.label || selectedValue : route.effective_label}</strong><p className="mt-1 break-all font-mono text-[10px] text-white/25">{changed ? "等待保存后解析具体模型" : route.model}</p></div>
                            {route.priority.length > 1 && <details className="mt-3"><summary className="cursor-pointer text-[11px] text-white/35">查看自动优先顺序</summary><ol className="mt-2 space-y-1 text-[11px] text-white/35">{route.priority.map((item, priorityIndex) => <li key={item}>{priorityIndex + 1}. {item}</li>)}</ol></details>}
                        </article>;
                    })}</div>
                </section>
                <div className="mb-3"><p className="studio-kicker">ADVANCED PARAMETERS</p><h3 className="text-lg font-semibold">服务地址、模型名称与密钥</h3></div>
                <div className="grid gap-5 xl:grid-cols-2">{payload.groups.map((group) => <section key={group.id} className="studio-panel"><p className="studio-kicker">{group.id.toUpperCase()}</p><h3 className="mb-5 text-lg font-semibold">{group.label}</h3><div className="grid gap-4 md:grid-cols-2">{group.fields.map((field) => <SettingField key={field.key} field={field} value={draft[field.key]} clear={clearKeys.includes(field.key)} onChange={(value) => { setDraft({ ...draft, [field.key]: value }); setClearKeys(clearKeys.filter((key) => key !== field.key)); }} onClear={() => setClearKeys(clearKeys.includes(field.key) ? clearKeys.filter((key) => key !== field.key) : [...clearKeys, field.key])} />)}</div></section>)}</div>
            </>}
        </div>
    </main>;
}

function SettingField({ field, value, clear, onChange, onClear }: { field: RuntimeSettingField; value: unknown; clear: boolean; onChange: (value: unknown) => void; onClear: () => void }) {
    const isBool = typeof value === "boolean";
    const isNumber = typeof value === "number";
    return <label><span className="studio-label flex items-center justify-between"><span>{field.label}</span>{field.secret && <span className={field.configured && !clear ? "text-emerald-300/70" : "text-white/25"}>{field.configured && !clear ? <span className="inline-flex items-center gap-1"><CheckCircle2 size={11} />{field.masked}</span> : "未配置"}</span>}</span><div className="studio-field">
        {field.secret ? <div className="flex gap-2"><div className="relative min-w-0 flex-1"><KeyRound className="absolute left-3 top-3 text-white/20" size={14} /><input className="!pl-9" type="password" value={String(value || "")} placeholder={clear ? "保存后清除密钥" : "留空则保持原密钥"} onChange={(event) => onChange(event.target.value)} /></div><button type="button" className={clear ? "studio-danger px-3" : "studio-secondary px-3"} title="清除已保存密钥" onClick={onClear}><Trash2 size={14} /></button></div>
            : field.options.length ? <select value={String(value ?? "")} onChange={(event) => onChange(event.target.value)}>{field.options.map((option) => <option key={option}>{option}</option>)}</select>
            : isBool ? <label className="flex h-[42px] items-center gap-2 rounded-lg border border-white/10 bg-black/25 px-3 text-sm text-white/70"><input type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(event.target.checked)} />{value ? "开启" : "关闭"}</label>
            : <input type={isNumber ? "number" : "text"} step={isNumber ? "any" : undefined} value={String(value ?? "")} onChange={(event) => onChange(isNumber ? Number(event.target.value) : event.target.value)} />}
    </div>{field.description && <p className="mt-1 text-[11px] text-white/25">{field.description}</p>}</label>;
}
