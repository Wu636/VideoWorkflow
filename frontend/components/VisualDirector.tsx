"use client";

import Image from "next/image";
import { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { Play, RotateCcw, CheckCircle, Loader2, Upload, X, Check, AlertTriangle, Sparkles } from "lucide-react";
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
    const [imageGenerationError, setImageGenerationError] = useState<string | null>(null);
    const imageGenerationStartedRef = useRef(false);

    // Regeneration state
    const [regenScene, setRegenScene] = useState<Scene | null>(null);
    const [regenPrompt, setRegenPrompt] = useState("");
    const [regenRefImage, setRegenRefImage] = useState<File | null>(null);
    const [isRegenerating, setIsRegenerating] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        if (!storyboard || imageGenerationStartedRef.current) {
            return;
        }

        const pendingSessionId = sessionStorage.getItem("video-workflow:pending-image-generation");
        const hasGeneratedImages = storyboard.scenes.some((scene) => !!scene.image_path);
        const hasInFlightScenes = storyboard.scenes.some((scene) => scene.image_status === "processing");

        if (pendingSessionId !== sessionId || hasGeneratedImages || hasInFlightScenes) {
            return;
        }

        imageGenerationStartedRef.current = true;
        setGenerating(true);
        setImageGenerationError(null);

        void generateImages(sessionId)
            .catch((error) => {
                console.error("Image generation failed:", error);
                setImageGenerationError(String(error));
            })
            .finally(async () => {
                sessionStorage.removeItem("video-workflow:pending-image-generation");
                setGenerating(false);
                const refreshed = await getScript(sessionId).catch(() => null);
                if (refreshed) {
                    setStoryboard(refreshed);
                }
            });
    }, [sessionId, storyboard]);

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

    const completedImageCount = storyboard.scenes.filter((scene) => !!scene.image_path).length;
    const failedImageCount = storyboard.scenes.filter((scene) => scene.image_status === "failed").length;
    const processingImageCount = storyboard.scenes.filter((scene) => scene.image_status === "processing").length;

    return (
        <div className="max-w-6xl mx-auto p-6 space-y-8 relative">
            <div className="sticky top-4 z-20 rounded-2xl border border-white/10 bg-black/85 px-5 py-4 shadow-2xl backdrop-blur-md">
                <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                    <div>
                    <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                        Visual Director
                    </h1>
                        <p className="text-gray-400">Select scenes to generate videos (reuse existing keyframes)</p>
                    </div>
                    <div className="flex items-center gap-4 self-end md:self-auto">
                        <div className="text-right text-sm text-gray-400">
                            <div>{selectedScenes.size} selected</div>
                            <div className="text-xs text-gray-500">
                                {completedImageCount}/{storyboard.scenes.length} images ready
                                {failedImageCount > 0 ? `, ${failedImageCount} failed` : ""}
                            </div>
                        </div>
                        <button
                            onClick={handleGenerateVideos}
                            disabled={generating || selectedScenes.size === 0}
                            className="btn-primary px-6 py-2 flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {generating ? <Loader2 className="animate-spin w-4 h-4" /> : <Play className="w-4 h-4 ml-1 fill-current" />}
                            <span>{generating ? "Generating Videos..." : "Generate Videos"}</span>
                        </button>
                    </div>
                </div>
            </div>

            {imageGenerationError && (
                <div className="rounded-2xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
                    Image generation hit an error: {imageGenerationError}
                </div>
            )}

            {(generating || processingImageCount > 0) && (
                <div className="rounded-2xl border border-cyan-400/20 bg-cyan-400/8 px-4 py-3 text-sm text-cyan-100">
                    <div className="flex items-center gap-3">
                        <Loader2 className="h-4 w-4 animate-spin text-cyan-300" />
                        <div>
                            <div className="font-medium">Video generation is running in the background.</div>
                            <div className="text-cyan-100/70">
                                Existing keyframes are reused. You do not need to regenerate images.
                            </div>
                        </div>
                    </div>
                </div>
            )}

            <div className="grid grid-cols-1 gap-6 pb-12 pt-2 md:grid-cols-2 lg:grid-cols-3">
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
                                <Image
                                    src={getImageUrl(scene.image_path)}
                                    alt={`Scene ${scene.id}`}
                                    fill
                                    unoptimized
                                    sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                                    className="h-full w-full object-contain transition-transform duration-700"
                                />
                            ) : (
                                <div className="flex h-full items-center justify-center text-gray-500">
                                    <Loader2 className="mr-2 h-6 w-6 animate-spin" />
                                    Generating...
                                </div>
                            )}

                            {scene.image_status === "processing" && (
                                <div className="pointer-events-none absolute inset-0 z-10 flex flex-col items-center justify-center bg-black/55 backdrop-blur-[2px]">
                                    <Loader2 className="mb-2 h-8 w-8 animate-spin text-cyan-300" />
                                    <div className="text-sm font-medium text-cyan-100">Generating keyframe</div>
                                    <div className="mt-1 text-xs text-cyan-100/70">This card will refresh automatically.</div>
                                </div>
                            )}

                            {scene.image_status === "failed" && !scene.image_path && (
                                <div className="pointer-events-none absolute inset-0 z-10 flex flex-col items-center justify-center bg-red-950/50 px-4 text-center backdrop-blur-[2px]">
                                    <AlertTriangle className="mb-2 h-8 w-8 text-red-300" />
                                    <div className="text-sm font-medium text-red-100">Generation failed</div>
                                    <div className="mt-1 text-xs text-red-100/75">You can regenerate this scene with a new prompt.</div>
                                </div>
                            )}

                            {scene.image_status === "failed" && (
                                <div className="absolute inset-x-0 bottom-4 z-20 flex justify-center" onClick={(e) => e.stopPropagation()}>
                                    <button
                                        onClick={() => openRegenModal(scene)}
                                        className="rounded-lg border border-red-300/30 bg-red-500/20 px-3 py-1.5 text-xs font-medium text-red-100 hover:bg-red-500/30"
                                    >
                                        Regenerate
                                    </button>
                                </div>
                            )}

                            {scene.image_status === "completed" && (
                                <div className="absolute bottom-2 right-2 z-10 rounded-full border border-emerald-400/30 bg-emerald-400/10 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-emerald-200">
                                    Ready
                                </div>
                            )}

                            {scene.image_status === "processing" && (
                                <div className="absolute bottom-2 right-2 z-10 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-200">
                                    Rendering
                                </div>
                            )}

                            {scene.image_status === "failed" && (
                                <div className="absolute bottom-2 right-2 z-10 rounded-full border border-red-400/30 bg-red-400/10 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-red-200">
                                    Failed
                                </div>
                            )}

                            {!scene.image_path && scene.image_status === "pending" && (
                                <div className="absolute bottom-2 right-2 z-10 rounded-full border border-white/15 bg-white/5 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-white/65">
                                    Pending
                                </div>
                            )}

                            {/* Hover Actions */}
                            <div className="absolute inset-0 z-20 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center" onClick={(e) => e.stopPropagation()}>
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
                            <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-white/45">
                                <Sparkles className="h-3.5 w-3.5" />
                                <span>{scene.image_status ?? "pending"}</span>
                            </div>
                            <p className="text-xs text-gray-400 line-clamp-3 leading-relaxed">{scene.visual_prompt}</p>
                            {scene.error_message && (
                                <p className="text-xs text-red-300/90 leading-relaxed">{scene.error_message}</p>
                            )}
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
