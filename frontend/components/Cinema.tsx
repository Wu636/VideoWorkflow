"use client";

import { useEffect, useState, useRef } from "react";
import { Download, Loader2, Film, RotateCcw, Image as ImageIcon, ArrowLeft } from "lucide-react";
import Link from "next/link";
import { Storyboard, Scene } from "@/types";
import { getScript, concatenateVideos, generateVideos } from "@/lib/api";

const API_BASE = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001/api").replace("/api", "");

function getVideoUrl(path?: string, timestamp?: number) {
    if (!path) return "";
    let cleanPath = path;
    if (cleanPath.startsWith("outputs/")) {
        cleanPath = cleanPath.substring("outputs/".length);
    } else if (cleanPath.includes("/outputs/")) {
        const parts = cleanPath.split("/outputs/");
        cleanPath = parts[parts.length - 1];
    }
    // Only add timestamp if provided, or default to empty (no cache busting by default unless requested)
    return timestamp ? `${API_BASE}/static/${cleanPath}?t=${timestamp}` : `${API_BASE}/static/${cleanPath}`;
}

export default function Cinema({ sessionId }: { sessionId: string }) {
    const [storyboard, setStoryboard] = useState<Storyboard | null>(null);
    const [finalVideoPath, setFinalVideoPath] = useState<string | null>(null);
    const [concatenating, setConcatenating] = useState(false);
    const [regeneratingMap, setRegeneratingMap] = useState<Record<number, boolean>>({});

    // Track versions for cache busting
    const [videoVersions, setVideoVersions] = useState<Record<number, number>>({});
    const prevStatuses = useRef<Record<number, string | undefined>>({});

    useEffect(() => {
        // Poll for video completion
        const poll = setInterval(() => {
            getScript(sessionId).then(setStoryboard);
        }, 3000);

        getScript(sessionId).then(setStoryboard);

        return () => clearInterval(poll);
    }, [sessionId]);

    // Monitor status changes to update video versions
    useEffect(() => {
        if (!storyboard) return;

        storyboard.scenes.forEach(scene => {
            const prev = prevStatuses.current[scene.id];
            const current = scene.video_status;

            // If transitioned to completed (or initially loaded as completed), we can allow a refresh
            // But to avoid infinite initial refresh, we might not strictly need it on mounting if browser cache is ok.
            // However, to be safe against previous regenerations:
            if (current === 'COMPLETED' && prev !== 'COMPLETED' && prev !== undefined) {
                setVideoVersions(v => ({ ...v, [scene.id]: Date.now() }));
            }

            prevStatuses.current[scene.id] = current;
        });
    }, [storyboard]);

    const handleConcatenate = async () => {
        setConcatenating(true);
        try {
            const result = await concatenateVideos(sessionId);
            setFinalVideoPath(result.final_video_path);
        } catch (e) {
            alert("Failed to concatenate videos: " + e);
        } finally {
            setConcatenating(false);
        }
    };

    const handleRegenerateVideo = async (sceneId: number) => {
        setRegeneratingMap(prev => ({ ...prev, [sceneId]: true }));
        try {
            await generateVideos(sessionId, [sceneId]);
            // refresh storyboard immediately
            const sb = await getScript(sessionId);
            setStoryboard(sb);
        } catch (e) {
            alert("Failed to regenerate video: " + e);
        } finally {
            setRegeneratingMap(prev => ({ ...prev, [sceneId]: false }));
        }
    };

    if (!storyboard) return <div className="text-center p-12 text-gray-500">Loading...</div>;

    const generatedVideoCount = storyboard.scenes.filter(scene => scene.video_path).length;
    const canConcatenate = generatedVideoCount > 0;

    return (
        <div className="max-w-6xl mx-auto p-6 space-y-8">
            <div className="flex items-center justify-between">
                <Link
                    href={`/workspace/${sessionId}/visuals`}
                    className="flex items-center text-gray-400 hover:text-white transition-colors"
                >
                    <ArrowLeft className="w-4 h-4 mr-2" />
                    Back to Visuals
                </Link>
                <div className="text-center">
                    <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                        Cinema
                    </h1>
                    <p className="text-gray-400">Your final production is ready</p>
                </div>
                <div className="w-24"></div> {/* Spacer for centering */}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {storyboard.scenes.map((scene) => (
                    <div key={scene.id} className="glass-card overflow-hidden group">
                        <div className="aspect-video bg-black/50 relative">
                            {scene.video_path && !regeneratingMap[scene.id] ? (
                                <video
                                    src={getVideoUrl(scene.video_path, videoVersions[scene.id])}
                                    controls
                                    className="w-full h-full object-contain"
                                />
                            ) : (regeneratingMap[scene.id] || scene.video_status === 'PROCESSING') ? (
                                <div className="flex items-center justify-center h-full text-gray-500 flex-col gap-2">
                                    <Loader2 className="animate-spin w-8 h-8" />
                                    {regeneratingMap[scene.id] ? "Regenerating..." : "Rendering..."}
                                </div>
                            ) : (
                                // Fallback to image if not generating video
                                scene.image_path ? (
                                    <img
                                        src={getVideoUrl(scene.image_path)}
                                        className="w-full h-full object-contain"
                                        alt={`Scene ${scene.id}`}
                                    />
                                ) : (
                                    <div className="flex items-center justify-center h-full text-gray-500">
                                        No Content
                                    </div>
                                )
                            )}

                            {/* Hover Actions */}
                            <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity flex gap-2" onClick={(e) => e.stopPropagation()}>
                                <button
                                    onClick={() => handleRegenerateVideo(scene.id)}
                                    disabled={!!regeneratingMap[scene.id]}
                                    className="p-2 rounded-lg bg-black/60 hover:bg-black/80 text-white backdrop-blur-sm border border-white/10"
                                    title="Regenerate Video Only"
                                >
                                    <RotateCcw className="w-4 h-4" />
                                </button>
                                <Link
                                    href={`/workspace/${sessionId}/visuals`}
                                    className="p-2 rounded-lg bg-black/60 hover:bg-black/80 text-white backdrop-blur-sm border border-white/10"
                                    title="Go back to regenerate Keyframe"
                                >
                                    <ImageIcon className="w-4 h-4" />
                                </Link>
                            </div>

                            <div className="absolute top-2 left-2 px-2 py-1 bg-black/50 backdrop-blur rounded text-xs font-mono border border-white/10">
                                Scene {scene.id}
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Final Concatenated Video */}
            {finalVideoPath && (
                <div className="glass-card p-6 space-y-4 animate-in fade-in slide-in-from-bottom-4">
                    <h3 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-green-400 to-blue-400">Final Masterpiece</h3>
                    <div className="aspect-video bg-black rounded-lg overflow-hidden shadow-2xl shadow-blue-900/20 border border-white/10">
                        <video
                            src={getVideoUrl(finalVideoPath)}
                            controls
                            className="w-full h-full"
                        />
                    </div>
                    <a
                        href={`${API_BASE}/api/sessions/download/${sessionId}/final_video.mp4`}
                        className="btn-primary px-6 py-3 flex items-center justify-center space-x-2 w-full bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700"
                    >
                        <Download className="w-5 h-5" />
                        <span>Download Final Video</span>
                    </a>
                </div>
            )}

            {/* Concatenate Button */}
            {!finalVideoPath && (
                <div className="glass-card p-6 border-blue-500/20">
                    <div className="flex items-center justify-between gap-4">
                        <div>
                            <h3 className="text-lg font-bold">Create Final Video</h3>
                            <p className="text-gray-400 text-sm">
                                {canConcatenate
                                    ? `Concatenate ${generatedVideoCount} available video(s)`
                                    : "Waiting for videos to complete..."}
                            </p>
                        </div>
                        <button
                            onClick={handleConcatenate}
                            disabled={!canConcatenate || concatenating}
                            className={`btn-primary px-8 py-3 flex items-center space-x-2 ${!canConcatenate || concatenating ? "opacity-50 cursor-not-allowed" : ""
                                }`}
                        >
                            {concatenating ? (
                                <>
                                    <Loader2 className="animate-spin w-5 h-5" />
                                    <span>Processing...</span>
                                </>
                            ) : (
                                <>
                                    <Film className="w-5 h-5" />
                                    <span>Concatenate Videos</span>
                                </>
                            )}
                        </button>
                    </div>
                </div>
            )}

            {/* Individual Downloads */}
            <div className="glass-card p-6">
                <h3 className="text-lg font-bold mb-4">Download Individual Scenes</h3>
                <div className="flex gap-2 flex-wrap">
                    {storyboard.scenes.map((scene) => scene.video_path && (
                        <a
                            key={scene.id}
                            href={`${API_BASE}/api/sessions/download/${sessionId}/scene_${scene.id}.mp4`}
                            className="px-3 py-2 rounded bg-white/5 hover:bg-white/10 border border-white/10 text-sm flex items-center space-x-2 transition-colors"
                        >
                            <Download className="w-3 h-3" />
                            <span>Scene {scene.id}</span>
                        </a>
                    ))}
                </div>
            </div>
        </div>
    );
}
