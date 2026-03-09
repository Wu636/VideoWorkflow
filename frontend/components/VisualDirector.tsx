"use client";

import { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { Play, RotateCcw, CheckCircle, Loader2, Upload, X, Check } from "lucide-react";
import { Storyboard, Scene } from "@/types";
import { getScript, generateVideos, generateImages, uploadFile, updateScript } from "@/lib/api";

interface Props {
    sessionId: string;
}

const API_BASE = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001/api").replace("/api", "");

function getImageUrl(path?: string) {
    if (!path) return "";
    let cleanPath = path;
    if (cleanPath.startsWith("outputs/")) {
        cleanPath = cleanPath.substring("outputs/".length);
    } else if (cleanPath.includes("/outputs/")) {
        const parts = cleanPath.split("/outputs/");
        cleanPath = parts[parts.length - 1];
    }
    // Add timestamp for cache busting
    return `${API_BASE}/static/${cleanPath}?t=${Date.now()}`;
}

export default function VisualDirector({ sessionId }: Props) {
    const router = useRouter();
    const [storyboard, setStoryboard] = useState<Storyboard | null>(null);
    const [loading, setLoading] = useState(true);
    const [generating, setGenerating] = useState(false);
    const [selectedScenes, setSelectedScenes] = useState<Set<number>>(new Set());

    // Regeneration state
    const [regenScene, setRegenScene] = useState<Scene | null>(null);
    const [regenPrompt, setRegenPrompt] = useState("");
    const [regenRefImage, setRegenRefImage] = useState<File | null>(null);
    const [isRegenerating, setIsRegenerating] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        const poll = setInterval(() => {
            getScript(sessionId).then(sb => {
                // Determine if we should update (simple check: if images changed)
                setStoryboard(prev => {
                    if (!prev) return sb;
                    // Only update if not regenerating locally to avoid jumpy UI? 
                    // Actually we want updates.
                    return sb;
                });
            });
        }, 3000);

        getScript(sessionId)
            .then((sb: Storyboard) => {
                setStoryboard(sb);
                // default select all scenes that have images
                const ids = sb.scenes.filter((s) => s.image_path).map((s) => s.id);
                setSelectedScenes(new Set(ids));
            })
            .catch((e) => console.error(e))
            .finally(() => setLoading(false));

        return () => clearInterval(poll);
    }, [sessionId]);

    const toggleSelection = (id: number) => {
        const newSet = new Set(selectedScenes);
        if (newSet.has(id)) {
            newSet.delete(id);
        } else {
            newSet.add(id);
        }
        setSelectedScenes(newSet);
    };

    const handleGenerateVideos = async () => {
        if (selectedScenes.size === 0) {
            alert("Please select at least one scene to generate video for.");
            return;
        }

        setGenerating(true);
        try {
            await generateVideos(sessionId, Array.from(selectedScenes));
            router.push(`/workspace/${sessionId}/cinema`);
        } catch (e) {
            alert("Failed to start video generation: " + e);
            setGenerating(false);
        }
    };

    const openRegenModal = (scene: Scene) => {
        setRegenScene(scene);
        setRegenPrompt(scene.visual_prompt);
        setRegenRefImage(null);
    };

    const closeRegenModal = () => {
        setRegenScene(null);
        setRegenPrompt("");
        setRegenRefImage(null);
    };

    const handleConfirmRegen = async () => {
        if (!regenScene || !storyboard) return;
        setIsRegenerating(true);

        try {
            // 1. Update visual prompt if changed
            if (regenPrompt !== regenScene.visual_prompt) {
                const updatedScenes = storyboard.scenes.map(s =>
                    s.id === regenScene.id ? { ...s, visual_prompt: regenPrompt } : s
                );
                await updateScript(sessionId, { ...storyboard, scenes: updatedScenes });
            }

            // 2. Upload reference image if provided
            let refImagePath = undefined;
            if (regenRefImage) {
                const uploadRes = await uploadFile(regenRefImage);
                refImagePath = uploadRes.path;
            }

            // 3. Call generateImages
            await generateImages(sessionId, [regenScene.id], refImagePath);

            // 4. Refresh script
            const newScript = await getScript(sessionId);
            setStoryboard(newScript);
            closeRegenModal();
        } catch (e) {
            alert("Regeneration failed: " + e);
        } finally {
            setIsRegenerating(false);
        }
    };

    if (loading) return <div className="text-center p-12"><Loader2 className="animate-spin w-8 h-8 mx-auto" /></div>;
    if (!storyboard) return <div className="text-center p-12 text-red-400">Loading Session...</div>;

    return (
        <div className="max-w-6xl mx-auto p-6 space-y-8 relative">
            <div className="flex items-center justify-between sticky top-0 z-10 bg-black/80 backdrop-blur-md py-4 -my-4 px-4 -mx-4 border-b border-white/10">
                <div>
                    <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                        Visual Director
                    </h1>
                    <p className="text-gray-400">Select scenes to generate videos</p>
                </div>
                <div className="flex items-center gap-4">
                    <div className="text-sm text-gray-400">
                        {selectedScenes.size} selected
                    </div>
                    <button
                        onClick={handleGenerateVideos}
                        disabled={generating || selectedScenes.size === 0}
                        className="btn-primary px-6 py-2 flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {generating ? <Loader2 className="animate-spin w-4 h-4" /> : <Play className="w-4 h-4 ml-1 fill-current" />}
                        <span>Generate Videos</span>
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pb-12">
                {storyboard.scenes.map((scene) => (
                    <div
                        key={scene.id}
                        className={`glass-card overflow-hidden group transition-all duration-200 border-2 ${selectedScenes.has(scene.id) ? 'border-blue-500/50 shadow-lg shadow-blue-500/20' : 'border-transparent'}`}
                        onClick={() => toggleSelection(scene.id)}
                    >
                        {/* Status / Selection Indicator */}
                        <div className="absolute top-2 right-2 z-10 pointer-events-none">
                            <div className={`w-6 h-6 rounded-full flex items-center justify-center transition-colors ${selectedScenes.has(scene.id) ? 'bg-blue-500 text-white' : 'bg-black/50 border border-white/20'}`}>
                                {selectedScenes.has(scene.id) && <Check className="w-4 h-4" />}
                            </div>
                        </div>

                        <div className="aspect-video bg-black/50 relative cursor-pointer">
                            {scene.image_path && getImageUrl(scene.image_path) ? (
                                <img
                                    src={getImageUrl(scene.image_path)}
                                    className="w-full h-full object-contain transition-transform duration-700"
                                    alt={`Scene ${scene.id}`}
                                />
                            ) : (
                                <div className="flex items-center justify-center h-full text-gray-500">
                                    <Loader2 className="animate-spin w-6 h-6 mr-2" />
                                    Generating...
                                </div>
                            )}

                            {/* Hover Actions */}
                            <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center" onClick={(e) => e.stopPropagation()}>
                                <button
                                    onClick={() => openRegenModal(scene)}
                                    className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 text-white flex items-center gap-2 backdrop-blur-sm border border-white/10 transition-colors"
                                >
                                    <RotateCcw className="w-4 h-4" />
                                    <span>Regenerate</span>
                                </button>
                            </div>

                            <div className="absolute top-2 left-2 px-2 py-1 bg-black/50 backdrop-blur rounded text-xs font-mono border border-white/10">
                                Scene {scene.id}
                            </div>
                        </div>
                        <div className="p-4 space-y-2 select-none">
                            <p className="text-xs text-gray-400 line-clamp-3 leading-relaxed">{scene.visual_prompt}</p>
                        </div>
                    </div>
                ))}
            </div>

            {/* Regeneration Modal */}
            {regenScene && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
                    <div className="glass-card w-full max-w-lg p-6 space-y-6 animate-in fade-in zoom-in-95 duration-200">
                        <div className="flex items-center justify-between">
                            <h3 className="text-xl font-bold">Regenerate Scene {regenScene.id}</h3>
                            <button onClick={closeRegenModal} disabled={isRegenerating} className="text-gray-400 hover:text-white">
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        <div className="space-y-4">
                            <div className="space-y-2">
                                <label className="text-xs uppercase tracking-wider text-gray-400">Visual Prompt</label>
                                <textarea
                                    className="w-full h-32 input-premium p-3 text-sm resize-none focus:ring-2 focus:ring-blue-500/50"
                                    value={regenPrompt}
                                    onChange={(e) => setRegenPrompt(e.target.value)}
                                />
                            </div>

                            <div className="space-y-2">
                                <label className="text-xs uppercase tracking-wider text-gray-400">Reference Image (Optional)</label>
                                <div
                                    className="border-2 border-dashed border-white/10 rounded-lg p-4 text-center cursor-pointer hover:bg-white/5 transition-colors"
                                    onClick={() => fileInputRef.current?.click()}
                                >
                                    {regenRefImage ? (
                                        <div className="flex items-center justify-center gap-2 text-green-400">
                                            <CheckCircle className="w-4 h-4" />
                                            <span className="text-sm truncate max-w-[200px]">{regenRefImage.name}</span>
                                            <button
                                                onClick={(e) => { e.stopPropagation(); setRegenRefImage(null); }}
                                                className="p-1 hover:text-red-400 ml-2"
                                            >
                                                <X className="w-3 h-3" />
                                            </button>
                                        </div>
                                    ) : (
                                        <div className="flex flex-col items-center gap-2 text-gray-400">
                                            <Upload className="w-6 h-6" />
                                            <span className="text-xs">Click to upload new reference</span>
                                        </div>
                                    )}
                                    <input
                                        type="file"
                                        ref={fileInputRef}
                                        className="hidden"
                                        accept="image/*"
                                        onChange={(e) => e.target.files?.[0] && setRegenRefImage(e.target.files[0])}
                                    />
                                </div>
                            </div>
                        </div>

                        <div className="flex items-center justify-end gap-3 pt-2">
                            <button
                                onClick={closeRegenModal}
                                disabled={isRegenerating}
                                className="btn-secondary px-4 py-2"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleConfirmRegen}
                                disabled={isRegenerating}
                                className="btn-primary px-6 py-2 flex items-center gap-2"
                            >
                                {isRegenerating ? <Loader2 className="animate-spin w-4 h-4" /> : <RotateCcw className="w-4 h-4" />}
                                <span>Regenerate</span>
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
