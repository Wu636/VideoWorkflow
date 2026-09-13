"use client";

import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Cable, CheckCircle2, ChevronDown, Database, GitBranch, KeyRound, Loader2, Plus, RefreshCw, RotateCcw, Save, Search, Trash2, Wifi, X } from "lucide-react";

import UtilityHeader from "@/components/UtilityHeader";
import { addModelToConnection, createModelConnection, deleteModelConnection, discoverModelConnection, getModelRegistry, getRuntimeSettings, testModelConnection, updateModelConnection, updateModelRoute, updateRuntimeSettings } from "@/lib/api";
import type { ModelRegistryPayload, ProviderConnection, ProviderModel, RuntimeSettingField, RuntimeSettingsPayload } from "@/types";

type Tab = "routes" | "connections" | "models" | "advanced";

export default function SettingsCenter() {
    const [tab, setTab] = useState<Tab>("routes");
    const [runtime, setRuntime] = useState<RuntimeSettingsPayload | null>(null);
    const [registry, setRegistry] = useState<ModelRegistryPayload | null>(null);
    const [draft, setDraft] = useState<Record<string, unknown>>({});
    const [clearKeys, setClearKeys] = useState<string[]>([]);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState("");
    const [notice, setNotice] = useState("");

    const load = async () => {
        setBusy(true);
        try {
            const [runtimeResult, registryResult] = await Promise.all([getRuntimeSettings(), getModelRegistry()]);
            setRuntime(runtimeResult);
            setDraft(Object.fromEntries(runtimeResult.groups.flatMap((group) => group.fields.map((field) => [field.key, field.value]))));
            setRegistry(registryResult);
            setError("");
        } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); }
        finally { setBusy(false); }
    };
    useEffect(() => { void load(); }, []);

    const saveAdvanced = async () => {
        setBusy(true); setError(""); setNotice("");
        try {
            const result = await updateRuntimeSettings(draft, clearKeys);
            setRuntime(result); setClearKeys([]);
            setDraft(Object.fromEntries(result.groups.flatMap((group) => group.fields.map((field) => [field.key, field.value]))));
            setNotice("高级配置已保存并即时生效。密钥只保存状态，不会回显原文。");
        } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); }
        finally { setBusy(false); }
    };

    const resetAdvanced = () => {
        if (!runtime) return;
        setDraft(Object.fromEntries(runtime.groups.flatMap((group) => group.fields.map((field) => [field.key, field.value]))));
        setClearKeys([]);
        setNotice("已恢复未保存修改。");
        setError("");
    };

    const refreshRegistry = async () => {
        try { setRegistry(await getModelRegistry()); setNotice("模型目录已刷新。"); setError(""); }
        catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); }
    };

    const changeRoute = async (routeId: string, value: string) => {
        const [connectionId, ...modelParts] = value.split("::");
        const modelId = modelParts.join("::");
        if (!connectionId || !modelId) return;
        setBusy(true); setError(""); setNotice("");
        try { setRegistry(await updateModelRoute(routeId, connectionId, modelId)); setNotice("功能路由已切换并即时生效。"); }
        catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); }
        finally { setBusy(false); }
    };

    const mutateRegistry = (result: ModelRegistryPayload, message: string) => { setRegistry(result); setNotice(message); setError(""); };

    return <main className="min-h-screen bg-[#080a0f] text-white">
        <UtilityHeader title="模型与接口设置" subtitle="把接口、模型目录和功能路由集中到一个可维护的控制台" />
        <div className="mx-auto max-w-[1500px] px-5 py-8 md:px-10">
            {error && <div className="studio-error mb-5 flex items-center gap-2"><AlertTriangle size={16} />{error}</div>}
            {notice && <div className="studio-notice mb-5 flex items-center gap-2"><CheckCircle2 size={16} />{notice}</div>}
            {(!runtime || !registry) ? <div className="flex min-h-64 items-center justify-center text-white/45"><Loader2 className="mr-2 animate-spin" />读取模型配置…</div> : <>
                <div className="mb-6 flex flex-wrap items-end justify-between gap-4"><div><p className="studio-kicker">MODEL CONTROL PLANE</p><h2 className="text-2xl font-semibold">模型管理</h2><p className="mt-1 text-sm text-white/35">内置接口可直接切换；自定义接口支持模型发现，遇到非标准接口也能手动补录模型。</p></div><button className="studio-secondary" onClick={() => void refreshRegistry()} disabled={busy}><RefreshCw size={15} className={busy ? "animate-spin" : ""} />刷新目录</button></div>
                <div className="mb-6 flex flex-wrap gap-2 border-b border-white/8 pb-3"><TabButton active={tab === "routes"} icon={<GitBranch size={15} />} onClick={() => setTab("routes")}>功能路由</TabButton><TabButton active={tab === "connections"} icon={<Cable size={15} />} onClick={() => setTab("connections")}>接口连接 <span className="ml-1 text-[10px] opacity-60">{registry.connections.length}</span></TabButton><TabButton active={tab === "models"} icon={<Database size={15} />} onClick={() => setTab("models")}>模型目录 <span className="ml-1 text-[10px] opacity-60">{registry.models.length}</span></TabButton><TabButton active={tab === "advanced"} icon={<Save size={15} />} onClick={() => setTab("advanced")}>高级参数</TabButton></div>
                {tab === "routes" && <RoutesPanel registry={registry} onChange={changeRoute} busy={busy} />}
                {tab === "connections" && <ConnectionsPanel registry={registry} onChange={mutateRegistry} onError={setError} />}
                {tab === "models" && <ModelsPanel registry={registry} onChange={mutateRegistry} onError={setError} />}
                {tab === "advanced" && <AdvancedPanel runtime={runtime} draft={draft} setDraft={setDraft} clearKeys={clearKeys} setClearKeys={setClearKeys} busy={busy} onSave={() => void saveAdvanced()} onReset={resetAdvanced} />}
            </>}
        </div>
    </main>;
}

function TabButton({ active, icon, onClick, children }: { active: boolean; icon: React.ReactNode; onClick: () => void; children: React.ReactNode }) { return <button onClick={onClick} className={active ? "inline-flex items-center rounded-lg border border-cyan-300/30 bg-cyan-300/10 px-4 py-2 text-sm text-cyan-100" : "inline-flex items-center rounded-lg border border-transparent px-4 py-2 text-sm text-white/45 hover:border-white/10 hover:text-white/75"}>{icon}<span className="ml-2">{children}</span></button>; }

function RoutesPanel({ registry, onChange, busy }: { registry: ModelRegistryPayload; onChange: (routeId: string, value: string) => void; busy: boolean }) {
    const groups = [
        { title: "理解与编排", hint: "剧本、镜头规划、分镜 Prompt 和参考图分析", ids: ["brief_analysis", "shot_count", "storyboard", "reference_vision"] },
        { title: "视觉与视频生成", hint: "角色/场景/首帧/封面生图，以及 Seedance 与 H3 视频", ids: ["image_generation", "seedance_video", "h3_video"] },
        { title: "声音输出", hint: "对白独立音轨；不需要时可切换为静音并保留 H3 原声", ids: ["dialogue_audio"] },
    ];
    return <div className="space-y-8">{groups.map((group) => <section key={group.title}><div className="mb-3"><h3 className="text-lg font-semibold">{group.title}</h3><p className="mt-1 text-xs text-white/35">{group.hint}</p></div><div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">{registry.routes.filter((route) => group.ids.includes(route.id)).map((route) => <RouteCard key={route.id} route={route} onChange={onChange} busy={busy} />)}</div></section>)}</div>;
}

function RouteCard({ route, onChange, busy }: { route: ModelRegistryPayload["routes"][number]; onChange: (routeId: string, value: string) => void; busy: boolean }) {
    const selectedOption = route.options.find((option) => option.value === route.selected);
    return <article className="studio-panel !p-5"><div className="mb-4 flex items-start justify-between"><span className="font-mono text-[10px] text-cyan-300/50">{route.id.replace(/_.*/, "").toUpperCase()}</span><span className="rounded-full bg-cyan-300/10 px-2 py-1 text-[10px] text-cyan-100/65">{route.capability === "vision" ? "图片理解" : route.capability === "image" ? "生图" : route.capability === "video" ? "视频" : route.capability === "audio" ? "音频" : "文本"}</span></div><h3 className="text-lg font-semibold">{route.label}</h3><p className="mt-1 min-h-10 text-xs leading-5 text-white/35">{route.description || "按功能单独切换，模型和服务可以分别维护。"}</p><ModelPicker route={route} selectedOption={selectedOption} onChange={onChange} busy={busy} /><div className="mt-3 rounded-lg border border-white/8 bg-black/20 p-3"><p className="text-[10px] text-white/30">当前模型</p><p className="mt-1 break-all font-mono text-xs text-cyan-100/80">{route.selected_model_id}</p><p className="mt-1 text-[10px] text-white/30">{route.selected_connection_id}</p></div></article>;
}

function ModelPicker({ route, selectedOption, onChange, busy }: { route: ModelRegistryPayload["routes"][number]; selectedOption?: ModelRegistryPayload["routes"][number]["options"][number]; onChange: (routeId: string, value: string) => void; busy: boolean }) {
    const [open, setOpen] = useState(false);
    const [query, setQuery] = useState("");
    const normalizedQuery = query.trim().toLowerCase();
    const filtered = route.options.filter((option) => !normalizedQuery || `${option.label} ${option.value}`.toLowerCase().includes(normalizedQuery));
    const visible = filtered.slice(0, 60);
    return <div className="relative mt-4">
        <button type="button" className="studio-field flex w-full items-center justify-between gap-3 text-left" disabled={busy} onClick={() => { setOpen(!open); setQuery(""); }}>
            <span className="min-w-0 truncate">{selectedOption?.label || route.selected}</span><ChevronDown size={15} className="shrink-0 text-white/40" />
        </button>
        {open && <div className="absolute inset-x-0 top-full z-30 mt-2 overflow-hidden rounded-xl border border-white/15 bg-[#11151d] shadow-2xl">
            <label className="flex items-center gap-2 border-b border-white/10 px-3 py-2"><Search size={14} className="text-white/30" /><input autoFocus value={query} className="min-w-0 flex-1 border-0 bg-transparent p-1 text-xs outline-none" placeholder="搜索模型或接口" onChange={(event) => setQuery(event.target.value)} /></label>
            <div className="max-h-64 overflow-y-auto p-1">{visible.map((option) => <button type="button" key={option.value} className={`block w-full rounded-lg px-3 py-2 text-left text-xs hover:bg-cyan-300/10 ${option.value === route.selected ? "bg-cyan-300/10 text-cyan-100" : "text-white/70"}`} onClick={() => { onChange(route.id, option.value); setOpen(false); }}>{option.label}</button>)}{!visible.length && <p className="px-3 py-4 text-center text-xs text-white/35">没有匹配模型</p>}{filtered.length > visible.length && <p className="px-3 py-2 text-[10px] text-white/30">已显示前 {visible.length} 个，请继续搜索缩小范围</p>}</div>
        </div>}
    </div>;
}

function ConnectionsPanel({ registry, onChange, onError }: { registry: ModelRegistryPayload; onChange: (result: ModelRegistryPayload, message: string) => void; onError: (message: string) => void }) {
    const [showAdd, setShowAdd] = useState(false); const [editing, setEditing] = useState<ProviderConnection | null>(null); const [form, setForm] = useState({ name: "", base_url: "", api_key: "", models_url: "", protocol: "openai_chat" }); const [busyId, setBusyId] = useState("");
    const submit = async () => { try { onChange(await createModelConnection(form), "自定义接口已添加。现在可以去模型目录发现或手动添加模型。"); setForm({ name: "", base_url: "", api_key: "", models_url: "", protocol: "openai_chat" }); setShowAdd(false); } catch (caught) { onError(caught instanceof Error ? caught.message : String(caught)); } };
    const saveEdit = async () => { if (!editing) return; setBusyId(editing.id); try { onChange(await updateModelConnection(editing.id, form), "接口配置已更新。"); setEditing(null); } catch (caught) { onError(caught instanceof Error ? caught.message : String(caught)); } finally { setBusyId(""); } };
    const discover = async (connection: ProviderConnection) => { setBusyId(connection.id); try { const result = await discoverModelConnection(connection.id); onChange(result.registry, result.supported ? `已发现 ${result.count || 0} 个模型。` : (result.message || "该接口未提供标准模型列表，请手动添加。")); } catch (caught) { onError(caught instanceof Error ? caught.message : String(caught)); } finally { setBusyId(""); } };
    const test = async (connection: ProviderConnection) => { setBusyId(connection.id); try { const result = await testModelConnection(connection.id); onChange(result.registry, result.ok ? "接口连通测试成功。" : (result.message || "接口测试未通过。")); } catch (caught) { onError(caught instanceof Error ? caught.message : String(caught)); } finally { setBusyId(""); } };
    const toggle = async (connection: ProviderConnection) => { setBusyId(connection.id); try { onChange(await updateModelConnection(connection.id, { enabled: !connection.enabled }), connection.enabled ? "接口已停用。" : "接口已启用。"); } catch (caught) { onError(caught instanceof Error ? caught.message : String(caught)); } finally { setBusyId(""); } };
    const remove = async (connection: ProviderConnection) => { if (!window.confirm(`删除接口“${connection.name}”？`)) return; setBusyId(connection.id); try { onChange(await deleteModelConnection(connection.id), "自定义接口已删除。"); } catch (caught) { onError(caught instanceof Error ? caught.message : String(caught)); } finally { setBusyId(""); } };
    return <section><div className="mb-4 flex items-center justify-between"><div><h3 className="text-xl font-semibold">接口连接</h3><p className="mt-1 text-sm text-white/35">统一维护 GRSAI、OpenLux、Claude 中转、火山方舟及自定义 OpenAI 兼容地址。</p></div><button className="studio-primary" onClick={() => { setShowAdd(!showAdd); setEditing(null); }}>{showAdd ? <X size={15} /> : <Plus size={15} />}{showAdd ? "取消" : "添加接口"}</button></div>{(showAdd || editing) && <div className="studio-panel mb-4 grid gap-4 md:grid-cols-2 xl:grid-cols-3"><Field label="接口名称"><input value={form.name} placeholder="例如：我的 Claude 中转" onChange={(event) => setForm({ ...form, name: event.target.value })} /></Field><Field label="Base URL"><input value={form.base_url} placeholder="https://example.com/v1" onChange={(event) => setForm({ ...form, base_url: event.target.value })} /></Field><Field label="模型列表 URL（可选）"><input value={form.models_url} placeholder="留空则尝试 Base URL/models" onChange={(event) => setForm({ ...form, models_url: event.target.value })} /></Field><Field label="API Key"><input type="password" value={form.api_key} placeholder={editing ? "留空则保持原密钥" : "仅保存到本地配置"} onChange={(event) => setForm({ ...form, api_key: event.target.value })} /></Field><Field label="协议"><select value={form.protocol} onChange={(event) => setForm({ ...form, protocol: event.target.value })}><option value="openai_chat">OpenAI Chat Completions</option><option value="anthropic_messages">Anthropic Messages（目录占位）</option><option value="gemini">Gemini（目录占位）</option></select></Field><div className="flex items-end gap-2"><button className="studio-primary" onClick={() => void (editing ? saveEdit() : submit())}><Save size={15} />{editing ? "保存修改" : "保存接口"}</button>{editing && <button className="studio-secondary" onClick={() => setEditing(null)}>取消</button>}</div></div>}<div className="grid gap-3">{registry.connections.map((connection) => <article key={connection.id} className="studio-panel flex flex-wrap items-center justify-between gap-4 !p-4"><div className="flex min-w-0 items-center gap-3"><span className={connection.enabled ? "rounded-lg bg-emerald-300/10 p-2 text-emerald-200" : "rounded-lg bg-white/5 p-2 text-white/25"}><Cable size={18} /></span><div className="min-w-0"><div className="flex flex-wrap items-center gap-2"><h4 className="font-semibold">{connection.name}</h4>{connection.builtin && <span className="rounded-full bg-cyan-300/10 px-2 py-0.5 text-[10px] text-cyan-100/65">内置</span>}{!connection.enabled && <span className="rounded-full bg-white/8 px-2 py-0.5 text-[10px] text-white/40">已停用</span>}</div><p className="mt-1 truncate font-mono text-xs text-white/35">{connection.base_url}</p><div className="mt-1 space-y-0.5 text-[11px] text-white/35"><p>{connection.configured ? `Key ${connection.masked}` : "未配置 Key"}</p><p><span className="text-white/25">模型发现：</span>{getDiscoverySummary(connection)}</p><p><span className="text-white/25">连通测试：</span>{getTestSummary(connection)}</p></div></div></div><div className="flex flex-wrap gap-2"><button className="studio-secondary" onClick={() => void discover(connection)} disabled={busyId === connection.id}><RefreshCw size={14} />发现模型</button><button className="studio-secondary" onClick={() => void test(connection)} disabled={busyId === connection.id}><Wifi size={14} />测试</button><button className="studio-secondary" onClick={() => void toggle(connection)} disabled={busyId === connection.id}>{connection.enabled ? "停用" : "启用"}</button>{!connection.builtin && <button className="studio-secondary" onClick={() => { setEditing(connection); setShowAdd(false); setForm({ name: connection.name, base_url: connection.base_url, api_key: "", models_url: connection.models_url, protocol: connection.protocol }); }} disabled={busyId === connection.id}>编辑</button>}{!connection.builtin && <button className="studio-danger" onClick={() => void remove(connection)} disabled={busyId === connection.id}><Trash2 size={14} /></button>}</div></article>)}</div></section>;
}

function getDiscoverySummary(connection: ProviderConnection): string {
    if (connection.last_discovery_status === "ok") return connection.last_discovery_count == null ? "完成" : `已发现 ${connection.last_discovery_count} 个模型`;
    if (connection.last_discovery_status === "manual") return connection.last_discovery_error || "接口未提供列表，请手动添加";
    if (connection.last_discovery_status === "discovery_failed") return connection.last_discovery_error || "发现失败";
    if (connection.last_error?.startsWith("模型列表接口")) return `上次发现：${connection.last_error}`;
    return "尚未发现";
}

function getTestSummary(connection: ProviderConnection): string {
    if (connection.last_test_status === "ok") return "通过";
    if (connection.last_test_status === "test_failed") return connection.last_test_error || "失败";
    if (connection.last_status === "ok") return "最近测试通过";
    if (connection.last_status === "test_failed") return connection.last_error || "测试失败";
    return "尚未测试";
}

function ModelsPanel({ registry, onChange, onError }: { registry: ModelRegistryPayload; onChange: (result: ModelRegistryPayload, message: string) => void; onError: (message: string) => void }) {
    const [query, setQuery] = useState(""); const [connectionId, setConnectionId] = useState("all"); const [showAdd, setShowAdd] = useState(false); const [form, setForm] = useState({ connection_id: "", model_id: "", label: "", capabilities: "text,json" });
    const rows = useMemo(() => registry.models.filter((model) => (!query || `${model.model_id} ${model.label}`.toLowerCase().includes(query.toLowerCase())) && (connectionId === "all" || model.connection_id === connectionId)), [registry.models, query, connectionId]);
    const submit = async () => { try { onChange(await addModelToConnection(form.connection_id, { model_id: form.model_id, label: form.label || form.model_id, capabilities: form.capabilities.split(",").map((item) => item.trim()).filter(Boolean) }), "手动模型已添加到目录。"); setForm({ ...form, model_id: "", label: "" }); setShowAdd(false); } catch (caught) { onError(caught instanceof Error ? caught.message : String(caught)); } };
    return <section><div className="mb-4 flex flex-wrap items-end justify-between gap-3"><div><h3 className="text-xl font-semibold">模型目录</h3><p className="mt-1 text-sm text-white/35">模型是独立条目，路由只引用目录中的模型；价格为每百万 Token 的参考值。</p></div><button className="studio-primary" onClick={() => { setShowAdd(!showAdd); if (!form.connection_id) setForm({ ...form, connection_id: registry.connections.find((item) => !item.builtin)?.id || "grsai" }); }}>{showAdd ? <X size={15} /> : <Plus size={15} />}{showAdd ? "取消" : "手动添加模型"}</button></div>{showAdd && <div className="studio-panel mb-4 grid gap-4 md:grid-cols-2 xl:grid-cols-4"><Field label="所属接口"><select value={form.connection_id} onChange={(event) => setForm({ ...form, connection_id: event.target.value })}>{registry.connections.map((connection) => <option key={connection.id} value={connection.id}>{connection.name}</option>)}</select></Field><Field label="模型 ID"><input value={form.model_id} placeholder="例如：claude-sonnet-5" onChange={(event) => setForm({ ...form, model_id: event.target.value })} /></Field><Field label="显示名称"><input value={form.label} placeholder="可留空，默认使用模型 ID" onChange={(event) => setForm({ ...form, label: event.target.value })} /></Field><Field label="能力（逗号分隔）"><input value={form.capabilities} onChange={(event) => setForm({ ...form, capabilities: event.target.value })} placeholder="text,json,vision" /></Field><div className="flex items-end"><button className="studio-primary" onClick={() => void submit()}><Save size={15} />保存模型</button></div></div>}<div className="mb-4 flex flex-wrap gap-3"><label className="studio-field flex min-w-64 flex-1 items-center gap-2"><Search size={15} className="text-white/30" /><input className="!border-0 !bg-transparent !p-0" value={query} placeholder="搜索模型 ID 或名称" onChange={(event) => setQuery(event.target.value)} /></label><select className="studio-field min-w-52" value={connectionId} onChange={(event) => setConnectionId(event.target.value)}><option value="all">全部接口</option>{registry.connections.map((connection) => <option key={connection.id} value={connection.id}>{connection.name}</option>)}</select></div><div className="overflow-hidden rounded-xl border border-white/8"><div className="grid grid-cols-[minmax(180px,1.3fr)_minmax(150px,1fr)_140px_140px_140px] gap-3 bg-white/[.035] px-4 py-3 text-[11px] text-white/40"><span>模型</span><span>接口</span><span>能力</span><span>输入 / M</span><span>输出 / M</span></div>{rows.map((model) => <ModelRow key={`${model.connection_id}:${model.model_id}`} model={model} connection={registry.connections.find((item) => item.id === model.connection_id)} />)}{!rows.length && <div className="p-8 text-center text-sm text-white/35">没有匹配的模型。若接口未提供列表，可以使用“手动添加模型”。</div>}</div></section>;
}

function ModelRow({ model, connection }: { model: ProviderModel; connection?: ProviderConnection }) { return <div className="grid grid-cols-[minmax(180px,1.3fr)_minmax(150px,1fr)_140px_140px_140px] gap-3 border-t border-white/6 px-4 py-3 text-xs"><div><p className="font-medium text-white/85">{model.label || model.model_id}</p><p className="mt-1 break-all font-mono text-[10px] text-white/30">{model.model_id}</p></div><span className="text-white/55">{connection?.name || model.connection_id}</span><span className="flex flex-wrap gap-1">{model.capabilities.map((capability) => <span key={capability} className="rounded bg-cyan-300/10 px-1.5 py-0.5 text-[10px] text-cyan-100/65">{capability}</span>)}</span><span className="text-white/55">{model.input_price_per_million == null ? "—" : `${model.currency === "CNY" ? "¥" : "$"}${model.input_price_per_million}`}</span><span className="text-white/55">{model.output_price_per_million == null ? "—" : `${model.currency === "CNY" ? "¥" : "$"}${model.output_price_per_million}`}</span></div>; }

const ADVANCED_SECTION_DEFS = [
    { id: "generation", title: "常用生成设置", description: "每天最常调整的画面、视频和长镜头参数。", expertOnly: false },
    { id: "performance", title: "性能与队列", description: "请求超时、轮询频率和并发行为。", expertOnly: false },
    { id: "stability", title: "稳定性与网络", description: "公网素材、代理和响应落盘配置。", expertOnly: false },
    { id: "audio", title: "音频与后处理", description: "配音声音和最终音轨参数。", expertOnly: false },
    { id: "local", title: "自部署与服务调优", description: "MetaSo、Atlas、ComfyUI 等服务的专属参数。", expertOnly: true },
    { id: "compatibility", title: "旧版兼容参数", description: "地址、密钥、模型 ID 和旧版路由；日常切换请使用上面的三个管理页。", expertOnly: true },
] as const;

const ADVANCED_FIELD_SECTION: Record<string, { section: string; expert?: boolean }> = {
    GRSAI_IMAGE_SIZE: { section: "generation" },
    IMAGE_STYLE_REFERENCE_MODE: { section: "generation" },
    SEEDANCE_DEFAULT_RESOLUTION: { section: "generation" },
    SEEDANCE_RENDER_CONCURRENCY: { section: "generation" },
    METASO_H3_RESOLUTION: { section: "generation" },
    METASO_H3_RATIO: { section: "generation" },
    METASO_H3_CONTEXT_IR_ENABLED: { section: "generation" },
    H3_AUTO_SEGMENT_COMPLEX_SHOTS: { section: "generation" },
    H3_MAX_SEGMENT_SECONDS: { section: "generation" },
    H3_AUDIO_MODE: { section: "generation" },
    PROMPT_OPTIMIZER_MODEL: { section: "generation" },
    OPENLUX_REQUEST_TIMEOUT_SECONDS: { section: "performance", expert: true },
    OPENLUX_STREAM: { section: "performance", expert: true },
    OPENLUX_SAVE_RAW_RESPONSES: { section: "performance", expert: true },
    SEEDANCE_POLL_INTERVAL_SECONDS: { section: "performance", expert: true },
    SEEDANCE_JOB_TIMEOUT_SECONDS: { section: "performance", expert: true },
    METASO_H3_POLL_INTERVAL_SECONDS: { section: "performance", expert: true },
    METASO_H3_JOB_TIMEOUT_SECONDS: { section: "performance", expert: true },
    ATLASCLOUD_JOB_TIMEOUT_SECONDS: { section: "performance", expert: true },
    COMFYUI_REQUEST_TIMEOUT_SECONDS: { section: "performance", expert: true },
    COMFYUI_POLL_INTERVAL_SECONDS: { section: "performance", expert: true },
    COMFYUI_JOB_TIMEOUT_SECONDS: { section: "performance", expert: true },
    SEEDANCE_PUBLIC_ASSET_BASE_URL: { section: "stability", expert: true },
    COMFYUI_VERIFY_TLS: { section: "stability", expert: true },
    COMFYUI_TRUST_ENV: { section: "stability", expert: true },
    TTS_DEFAULT_FEMALE_VOICE: { section: "audio", expert: true },
    TTS_DEFAULT_MALE_VOICE: { section: "audio", expert: true },
    TTS_DEFAULT_MATURE_FEMALE_VOICE: { section: "audio", expert: true },
    TTS_DEFAULT_MATURE_MALE_VOICE: { section: "audio", expert: true },
    H3_POSTPROCESS_AUDIO_BITRATE: { section: "audio", expert: true },
    METASO_H3_API_KEY: { section: "local", expert: true },
    METASO_H3_BASE_URL: { section: "local", expert: true },
    ATLASCLOUD_API_KEY: { section: "local", expert: true },
    ATLASCLOUD_BASE_URL: { section: "local", expert: true },
    COMFYUI_BASE_URL: { section: "local", expert: true },
    COMFYUI_API_TOKEN: { section: "local", expert: true },
    COMFYUI_HOURLY_RATE: { section: "local", expert: true },
    H3_MODEL_PROFILE: { section: "local", expert: true },
    H3_TEXT_ENCODER_PROFILE: { section: "local", expert: true },
    H3_PROVIDER: { section: "compatibility", expert: true },
    TTS_PROVIDER: { section: "compatibility", expert: true },
};

const ADVANCED_FIELD_CONSTRAINTS: Record<string, { unit?: string; min?: number; max?: number; step?: number; recommended?: string }> = {
    SEEDANCE_RENDER_CONCURRENCY: { unit: "路并发", min: 1, max: 8, step: 1, recommended: "账号配额允许时建议 2–3" },
    SEEDANCE_POLL_INTERVAL_SECONDS: { unit: "秒", min: 1, max: 60, step: 1 },
    SEEDANCE_JOB_TIMEOUT_SECONDS: { unit: "秒", min: 60, step: 30 },
    OPENLUX_REQUEST_TIMEOUT_SECONDS: { unit: "秒", min: 30, step: 30 },
    METASO_H3_POLL_INTERVAL_SECONDS: { unit: "秒", min: 1, max: 60, step: 1 },
    METASO_H3_JOB_TIMEOUT_SECONDS: { unit: "秒", min: 60, step: 30 },
    ATLASCLOUD_JOB_TIMEOUT_SECONDS: { unit: "秒", min: 60, step: 30 },
    COMFYUI_REQUEST_TIMEOUT_SECONDS: { unit: "秒", min: 30, step: 30 },
    COMFYUI_POLL_INTERVAL_SECONDS: { unit: "秒", min: 1, max: 60, step: 1 },
    COMFYUI_JOB_TIMEOUT_SECONDS: { unit: "秒", min: 60, step: 30 },
    H3_MAX_SEGMENT_SECONDS: { unit: "秒", min: 4, max: 15, step: 1, recommended: "6–8 秒" },
    COMFYUI_HOURLY_RATE: { unit: "元/小时", min: 0, step: 0.01 },
    H3_POSTPROCESS_AUDIO_BITRATE: { unit: "kbps", min: 32, step: 8 },
};

function advancedFieldMeta(field: RuntimeSettingField, groupId: string) {
    return ADVANCED_FIELD_SECTION[field.key] || { section: groupId === "audio" ? "audio" : groupId === "comfyui" || groupId === "h3_api" ? "local" : "compatibility", expert: true };
}

function advancedFieldVisible(field: RuntimeSettingField, draft: Record<string, unknown>, search: string) {
    const q = search.trim().toLowerCase();
    const matches = !q || `${field.key} ${field.label} ${field.description}`.toLowerCase().includes(q);
    if (!matches) return false;
    if (field.key.startsWith("METASO_") && draft.H3_PROVIDER !== "metaso_h3") return Boolean(q);
    if (field.key.startsWith("ATLASCLOUD_") && draft.H3_PROVIDER !== "atlas_h3") return Boolean(q);
    if (field.key.startsWith("H3_ATLAS_") && draft.H3_PROVIDER !== "atlas_h3") return Boolean(q);
    if ((field.key.startsWith("COMFYUI_") || field.key === "H3_MODEL_PROFILE" || field.key === "H3_TEXT_ENCODER_PROFILE") && draft.H3_PROVIDER !== "comfyui_h3") return Boolean(q);
    if (field.key.startsWith("TTS_DEFAULT_") || field.key === "H3_POSTPROCESS_AUDIO_BITRATE") return draft.H3_AUDIO_MODE === "clean_tts" || Boolean(q);
    return true;
}

function AdvancedPanel({ runtime, draft, setDraft, clearKeys, setClearKeys, busy, onSave, onReset }: { runtime: RuntimeSettingsPayload; draft: Record<string, unknown>; setDraft: (value: Record<string, unknown>) => void; clearKeys: string[]; setClearKeys: (value: string[]) => void; busy: boolean; onSave: () => void; onReset: () => void }) {
    const [mode, setMode] = useState<"common" | "expert">("common");
    const [search, setSearch] = useState("");
    const [expanded, setExpanded] = useState<Record<string, boolean>>({ generation: true, performance: true, stability: false, audio: false, local: false, compatibility: false });
    const allFields = runtime.groups.flatMap((group) => group.fields.map((field) => ({ field, groupId: group.id })));
    const dirtyKeys = allFields.filter(({ field }) => clearKeys.includes(field.key) || (field.secret ? Boolean(draft[field.key]) : String(draft[field.key] ?? "") !== String(field.value ?? ""))).map(({ field }) => field.key);
    const visibleSections = ADVANCED_SECTION_DEFS.map((section) => ({ ...section, fields: allFields.filter(({ field, groupId }) => { const meta = advancedFieldMeta(field, groupId); return meta.section === section.id && (mode === "expert" || !meta.expert || Boolean(search.trim())) && advancedFieldVisible(field, draft, search); }) })).filter((section) => section.fields.length > 0);
    const updateField = (key: string, value: unknown) => { setDraft({ ...draft, [key]: value }); setClearKeys(clearKeys.filter((item) => item !== key)); };
    return <section>
        <div className="mb-5 flex flex-wrap items-end justify-between gap-3"><div><h3 className="text-xl font-semibold">高级参数</h3><p className="mt-1 text-sm text-white/35">这里负责运行行为和性能调优；接口、模型和功能路由分别在其他页面维护。</p></div><button className="studio-primary" disabled={busy} onClick={onSave}>{busy ? <Loader2 className="animate-spin" size={15} /> : <Save size={15} />}保存高级配置</button></div>
        <div className="mb-5 flex flex-wrap items-center gap-3"><label className="studio-field flex min-w-64 flex-1 items-center gap-2"><Search size={15} className="text-white/30" /><input className="!border-0 !bg-transparent !p-0" value={search} placeholder="搜索参数名称、环境变量或说明" onChange={(event) => setSearch(event.target.value)} /></label><div className="flex rounded-lg border border-white/10 bg-black/20 p-1"><button className={mode === "common" ? "rounded-md bg-cyan-300/15 px-3 py-2 text-xs text-cyan-100" : "rounded-md px-3 py-2 text-xs text-white/45"} onClick={() => setMode("common")}>常用模式</button><button className={mode === "expert" ? "rounded-md bg-cyan-300/15 px-3 py-2 text-xs text-cyan-100" : "rounded-md px-3 py-2 text-xs text-white/45"} onClick={() => setMode("expert")}>专家模式</button></div><button className="studio-secondary" onClick={onReset} disabled={!dirtyKeys.length}><RotateCcw size={14} />恢复未保存</button></div>
        <div className="grid gap-6 lg:grid-cols-[210px_minmax(0,1fr)]"><nav className="hidden lg:block"><div className="sticky top-5 space-y-1">{visibleSections.map((section) => <button key={section.id} className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-xs text-white/55 hover:bg-white/5 hover:text-white/85" onClick={() => { setExpanded({ ...expanded, [section.id]: true }); document.getElementById(`advanced-${section.id}`)?.scrollIntoView({ behavior: "smooth", block: "start" }); }}><span>{section.title}</span><span className="text-[10px] text-white/25">{section.fields.length}</span></button>)}</div></nav><div className="min-w-0 space-y-4">{visibleSections.map((section) => <section id={`advanced-${section.id}`} key={section.id} className="studio-panel !p-0"><button className="flex w-full items-start justify-between gap-4 p-5 text-left" onClick={() => setExpanded({ ...expanded, [section.id]: !expanded[section.id] })}><span><span className="studio-kicker">{section.id.toUpperCase()}</span><span className="mt-1 block text-lg font-semibold">{section.title}</span><span className="mt-1 block text-xs text-white/35">{section.description}</span></span><span className="mt-1 flex shrink-0 items-center gap-2 text-[10px] text-white/30"><span>{section.fields.length} 项</span><ChevronDown size={16} className={expanded[section.id] ? "rotate-180 transition-transform" : "transition-transform"} /></span></button>{expanded[section.id] && <div className="grid gap-4 border-t border-white/8 p-5 md:grid-cols-2">{section.fields.map(({ field }) => <SettingField key={field.key} field={field} value={draft[field.key]} clear={clearKeys.includes(field.key)} onChange={(value) => updateField(field.key, value)} onClear={() => setClearKeys(clearKeys.includes(field.key) ? clearKeys.filter((key) => key !== field.key) : [...clearKeys, field.key])} />)}</div>}</section>)}{!visibleSections.length && <div className="studio-panel p-10 text-center text-sm text-white/35">没有匹配的参数。切换到专家模式可以查看全部兼容字段。</div>}</div></div>
        {dirtyKeys.length > 0 && <div className="sticky bottom-4 z-20 mt-5 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-cyan-300/20 bg-[#111820]/95 px-4 py-3 text-xs shadow-2xl backdrop-blur"><span className="text-cyan-100/80">已修改 {dirtyKeys.length} 项，保存后立即生效</span><div className="flex gap-2"><button className="studio-secondary" onClick={onReset}>放弃修改</button><button className="studio-primary" disabled={busy} onClick={onSave}><Save size={14} />保存并生效</button></div></div>}
    </section>;
}

function SettingField({ field, value, clear, onChange, onClear }: { field: RuntimeSettingField; value: unknown; clear: boolean; onChange: (value: unknown) => void; onClear: () => void }) { const isBool = typeof value === "boolean"; const isNumber = typeof value === "number"; const hint = ADVANCED_FIELD_CONSTRAINTS[field.key]; return <div><div className="studio-label flex items-center justify-between gap-3"><span>{field.label}{hint?.unit && <span className="ml-1 text-[10px] text-white/25">({hint.unit})</span>}</span>{field.secret && <span className={field.configured && !clear ? "text-emerald-300/70" : "text-white/25"}>{field.configured && !clear ? <span className="inline-flex items-center gap-1"><KeyRound size={11} />{field.masked}</span> : "未配置"}</span>}</div><div className="studio-field">{field.secret ? <div className="flex gap-2"><div className="relative min-w-0 flex-1"><KeyRound className="absolute left-3 top-3 text-white/20" size={14} /><input className="!pl-9" type="password" value={String(value || "")} placeholder={clear ? "保存后清除密钥" : "留空则保持原密钥"} onChange={(event) => onChange(event.target.value)} /></div><button type="button" className={clear ? "studio-danger px-3" : "studio-secondary px-3"} title="清除已保存密钥" onClick={onClear}><Trash2 size={14} /></button></div> : field.options.length ? <select value={String(value ?? "")} onChange={(event) => onChange(event.target.value)}>{field.options.map((option) => <option key={option}>{option}</option>)}</select> : isBool ? <div className="flex h-[42px] items-center gap-2 rounded-lg border border-white/10 bg-black/25 px-3 text-sm text-white/70"><input type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(event.target.checked)} />{value ? "开启" : "关闭"}</div> : <input type={isNumber ? "number" : "text"} min={hint?.min} max={hint?.max} step={hint?.step ?? (isNumber ? "any" : undefined)} value={String(value ?? "")} onChange={(event) => onChange(isNumber ? Number(event.target.value) : event.target.value)} />}</div>{field.description && <p className="mt-1 text-[11px] text-white/25">{field.description}</p>}{hint?.recommended && <p className="mt-1 text-[10px] text-cyan-100/40">推荐：{hint.recommended}</p>}</div>; }

function Field({ label, children }: { label: string; children: React.ReactNode }) { return <label><span className="studio-label">{label}</span><div className="studio-field">{children}</div></label>; }
