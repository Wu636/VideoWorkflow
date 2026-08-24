"use client";

import { useCallback, useEffect, useState } from "react";
import { Download, Loader2, Pause, Play, RefreshCw, Trash2 } from "lucide-react";

import UtilityHeader from "@/components/UtilityHeader";
import { clearRuntimeLogs, getRuntimeLogs, runtimeLogsDownloadUrl } from "@/lib/api";
import type { RuntimeLogRecord } from "@/types";

export default function LogsCenter() {
    const [records, setRecords] = useState<RuntimeLogRecord[]>([]);
    const [level, setLevel] = useState("INFO");
    const [search, setSearch] = useState("");
    const [live, setLive] = useState(true);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");
    const load = useCallback(async () => { try { setRecords((await getRuntimeLogs({ level, search, limit: 1000 })).records); setError(""); } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); } finally { setLoading(false); } }, [level, search]);
    useEffect(() => { void load(); if (!live) return; const timer = window.setInterval(() => void load(), 2500); return () => window.clearInterval(timer); }, [live, load]);
    const clear = async () => { if (!window.confirm("清空当前运行日志和日志文件？")) return; await clearRuntimeLogs(); await load(); };
    return <main className="min-h-screen bg-[#080a0f] text-white"><UtilityHeader title="运行日志中心" subtitle="查看 API、分镜、图片、H3 队列与成片处理错误" /><div className="mx-auto max-w-[1700px] px-5 py-8 md:px-10">
        {error && <div className="studio-error mb-5">{error}</div>}
        <section className="studio-panel mb-4 flex flex-wrap items-end gap-3"><label><span className="studio-label">最低级别</span><select className="studio-input" value={level} onChange={(event) => setLevel(event.target.value)}><option>DEBUG</option><option>INFO</option><option>WARNING</option><option>ERROR</option><option>CRITICAL</option></select></label><label className="min-w-60 flex-1"><span className="studio-label">搜索日志</span><input className="studio-input" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="任务 ID、项目 ID、异常关键词…" /></label><button className="studio-secondary" onClick={() => setLive(!live)}>{live ? <Pause size={14} /> : <Play size={14} />}{live ? "暂停刷新" : "恢复刷新"}</button><button className="studio-secondary" onClick={() => void load()}><RefreshCw size={14} />刷新</button><a className="studio-secondary" href={runtimeLogsDownloadUrl()}><Download size={14} />下载</a><button className="studio-danger" onClick={() => void clear()}><Trash2 size={14} />清空</button></section>
        <section className="overflow-hidden rounded-xl border border-white/8 bg-[#090b10]"><div className="flex justify-between border-b border-white/8 px-4 py-3 text-xs text-white/35"><span>{records.length} 条记录</span><span>{live ? "LIVE · 每 2.5 秒刷新" : "PAUSED"}</span></div>{loading ? <div className="flex min-h-64 items-center justify-center text-white/35"><Loader2 className="mr-2 animate-spin" />读取日志…</div> : records.length === 0 ? <div className="flex min-h-64 items-center justify-center text-sm text-white/30">当前筛选条件下没有日志</div> : <div className="max-h-[70vh] overflow-auto font-mono text-xs">{records.map((record, index) => <LogRow key={`${record.timestamp}-${index}`} record={record} />)}</div>}</section>
    </div></main>;
}

function LogRow({ record }: { record: RuntimeLogRecord }) {
    const color = record.level === "ERROR" || record.level === "CRITICAL" ? "text-red-300" : record.level === "WARNING" ? "text-amber-300" : "text-cyan-200";
    return <div className="grid gap-2 border-b border-white/[.04] px-4 py-2.5 hover:bg-white/[.025] md:grid-cols-[170px_80px_220px_1fr]"><span className="text-white/30">{new Date(record.timestamp).toLocaleString("zh-CN")}</span><strong className={color}>{record.level}</strong><span className="truncate text-white/35" title={record.logger}>{record.logger}</span><div className="min-w-0 whitespace-pre-wrap break-words text-white/70">{record.message}{record.exception && <pre className="mt-2 whitespace-pre-wrap text-red-200/70">{record.exception}</pre>}</div></div>;
}
