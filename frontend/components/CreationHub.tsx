"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Upload, Wand2, Video, Sparkles, Loader2, Zap, Terminal, Cpu } from "lucide-react";
import { createSession } from "@/lib/api";
import { VIRAL_TEMPLATES } from "@/types";

export default function CreationHub() {
    const router = useRouter();
    const [topic, setTopic] = useState("");
    const [loading, setLoading] = useState(false);
    const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null);
    const [refImage, setRefImage] = useState<string | null>(null);
    const [dragActive, setDragActive] = useState(false);

    const handleFile = (file: File) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            setRefImage(reader.result as string);
        };
        reader.readAsDataURL(file);
    };

    const handleDrag = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    };

    const handleSubmit = async () => {
        if (!topic) return;
        setLoading(true);
        try {
            const res = await createSession({
                topic,
                reference_image: refImage || undefined,
                template: selectedTemplate || undefined,
                count: 5
            });
            router.push(`/workspace/${res.session_id}/script`);
        } catch (e) {
            console.error(e);
            alert("Creation failed: " + e);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="w-full max-w-7xl mx-auto z-10 flex flex-col lg:flex-row gap-12 items-center lg:items-start pt-10 px-4">

            {/* Left Column: Cyberpunk Hero */}
            <div className="flex-1 space-y-8 pt-10 text-center lg:text-left">
                <div className="inline-flex items-center space-x-2 px-4 py-1 border-l-2 border-neon-cyan bg-neon-cyan/5 text-neon-cyan text-sm font-mono tracking-widest uppercase">
                    <Terminal className="w-4 h-4" />
                    <span>System Ready // V.2.0.45</span>
                </div>

                <h1 className="text-6xl md:text-8xl font-black tracking-tighter leading-none text-white uppercase ml-[-4px]">
                    Video <br />
                    <span className="text-neon-cyan bg-clip-text text-transparent bg-gradient-to-r from-neon-cyan to-white relative">
                        Synthesis
                        <span className="absolute -inset-1 bg-neon-cyan/20 blur-xl -z-10 opacity-50 animate-pulse-slow"></span>
                    </span>
                </h1>

                <p className="text-xl text-gray-400 max-w-lg leading-relaxed font-mono border-l border-white/10 pl-6 mx-auto lg:mx-0 text-left">
                    <span className="text-neon-purple opacity-70">&gt;&gt;</span> Initialize neural rendering...<br />
                    <span className="text-neon-purple opacity-70">&gt;&gt;</span> Transform ideas into cinematic reality.
                </p>

                <div className="flex items-center justify-center lg:justify-start space-x-8 pt-8 opacity-60 hover:opacity-100 transition-opacity">
                    <div className="flex flex-col">
                        <span className="text-2xl font-bold font-mono text-white">4.2s</span>
                        <span className="text-xs text-gray-500 uppercase tracking-widest">Latency</span>
                    </div>
                    <div className="w-px h-10 bg-white/20"></div>
                    <div className="flex flex-col">
                        <span className="text-2xl font-bold font-mono text-neon-green">99.9%</span>
                        <span className="text-xs text-gray-500 uppercase tracking-widest">Uptime</span>
                    </div>
                </div>
            </div>

            {/* Right Column: Console Interface */}
            <div className="flex-1 w-full max-w-xl">
                <div className="glass-panel-cyber p-1 space-y-0">

                    {/* Console Header */}
                    <div className="h-8 bg-black/80 flex items-center justify-between px-4 border-b border-white/10">
                        <div className="flex space-x-2">
                            <div className="w-2 h-2 rounded-full bg-red-500/50"></div>
                            <div className="w-2 h-2 rounded-full bg-yellow-500/50"></div>
                            <div className="w-2 h-2 rounded-full bg-green-500/50"></div>
                        </div>
                        <div className="text-[10px] text-gray-500 font-mono tracking-widest">TERMINAL_RELAY_01</div>
                    </div>

                    <div className="p-8 space-y-8 bg-black/20">
                        {/* Input Field */}
                        <div className="space-y-3">
                            <label className="text-xs font-bold text-neon-cyan uppercase tracking-widest flex items-center gap-2">
                                <Cpu className="w-4 h-4" />
                                Input Directive
                            </label>
                            <div className="relative group/input">
                                <input
                                    type="text"
                                    value={topic}
                                    onChange={(e) => setTopic(e.target.value)}
                                    placeholder="Enter generation parameters..."
                                    className="w-full input-cyber p-4 pl-12 text-sm font-mono h-14"
                                />
                                <span className="absolute left-4 top-1/2 -translate-y-1/2 text-neon-purple font-mono text-lg animate-pulse">&gt;</span>
                            </div>
                        </div>

                        {/* Split Grid */}
                        <div className="grid grid-cols-2 gap-4">
                            {/* Dropzone */}
                            <div
                                className={`relative h-40 border border-white/10 flex flex-col items-center justify-center text-center p-4 transition-all cursor-pointer overflow-hidden bg-black/40 group
                                    ${dragActive ? 'border-neon-purple bg-neon-purple/5' : 'hover:border-neon-cyan/50'}`}
                                onDragEnter={handleDrag}
                                onDragLeave={handleDrag}
                                onDragOver={handleDrag}
                                onDrop={handleDrop}
                            >
                                <input
                                    type="file"
                                    className="absolute inset-0 opacity-0 cursor-pointer z-20"
                                    onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                                    accept="image/*"
                                />
                                {/* Grid Background */}
                                <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:20px_20px]"></div>

                                {refImage ? (
                                    <>
                                        <img src={refImage} alt="Ref" className="absolute inset-0 w-full h-full object-cover opacity-60 grayscale group-hover:grayscale-0 transition-all duration-500" />
                                        <div className="absolute inset-0 bg-black/40 z-10 flex items-center justify-center pointer-events-none">
                                            <p className="text-neon-cyan font-mono text-xs uppercase tracking-widest">[REPLACE_DATA]</p>
                                        </div>
                                    </>
                                ) : (
                                    <>
                                        <Upload className="w-6 h-6 text-gray-600 mb-2 group-hover:text-neon-cyan transition-colors" />
                                        <p className="text-[10px] text-gray-500 font-mono uppercase tracking-widest">Upload Reference</p>
                                    </>
                                )}
                            </div>

                            {/* Template Selector */}
                            <div className="h-40 overflow-y-auto space-y-1 pr-1 custom-scrollbar border border-white/10 bg-black/40 relative">
                                <div className="sticky top-0 bg-black/90 z-10 p-2 border-b border-white/10 backdrop-blur-sm">
                                    <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Select Mode</div>
                                </div>
                                {VIRAL_TEMPLATES.map((t) => (
                                    <button
                                        key={t.id}
                                        onClick={() => setSelectedTemplate(selectedTemplate === t.name ? null : t.name)}
                                        className={`w-full text-left p-2 text-xs font-mono transition-all border-l-2 ${selectedTemplate === t.name
                                            ? 'bg-neon-cyan/10 border-neon-cyan text-white'
                                            : 'bg-transparent border-transparent text-gray-500 hover:text-white hover:border-white/20'
                                            }`}
                                    >
                                        {t.name}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* CTA */}
                        <button
                            onClick={handleSubmit}
                            disabled={!topic || loading}
                            className={`w-full h-16 btn-cyber group/btn flex items-center justify-center space-x-3
                                ${!topic || loading
                                    ? 'grayscale opacity-50 cursor-not-allowed'
                                    : ''
                                }`}
                        >
                            {loading ? (
                                <>
                                    <Loader2 className="animate-spin w-5 h-5" />
                                    <span>PROCESSING...</span>
                                </>
                            ) : (
                                <>
                                    <Zap className="w-5 h-5 group-hover:fill-current" />
                                    <span>INITIATE_SEQUENCE</span>
                                </>
                            )}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
