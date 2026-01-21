"use client";

import { useEffect, useState } from "react";
import { Download, Loader2, Film } from "lucide-react";
import { Storyboard } from "@/types";
import { getScript, concatenateVideos } from "@/lib/api";

const API_BASE = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001/api").replace("/api", "");

function getVideoUrl(path?: string) {
    if (!path) return "";
    let cleanPath = path;
    if (cleanPath.startsWith("outputs/")) {
        cleanPath = cleanPath.substring("outputs/".length);
    } else if (cleanPath.includes("/outputs/")) {
        const parts = cleanPath.split("/outputs/");
        cleanPath = parts[parts.length - 1];
    }
    return `${API_BASE}/static/${cleanPath}`;
}

export default function Cinema({ sessionId }: { sessionId: string }) {
    const [storyboard, setStoryboard] = useState<Storyboard | null>(null);
    const [finalVideoPath, setFinalVideoPath] = useState<string | null>(null);
    const [concatenating, setConcatenating] = useState(false);

    useEffect(() => {
        // Poll for video completion
        const poll = setInterval(() => {
            getScript(sessionId).then(setStoryboard);
        }, 3000);
        return () => clearInterval(poll);
    }, [sessionId]);

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

    if (!storyboard) return <div className="text-center p-12 text-gray-500">Loading...</div>;

    const allVideosReady = storyboard.scenes.every(scene => scene.video_path);

    return (
        <div className="max-w-6xl mx-auto p-6 space-y-8">
            <div className="text-center space-y-2">
                <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                    Cinema
                </h1>
                <p className="text-gray-400">Your final production is ready</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {storyboard.scenes.map((scene) => (
                    <div key={scene.id} className="glass-card overflow-hidden">
                        <div className="aspect-video bg-black/50 relative">
                            {scene.video_path ? (
                                <video
                                    src={getVideoUrl(scene.video_path)}
                                    controls
                                    className="w-full h-full object-cover"
                                />
                            ) : (
                                <div className="flex items-center justify-center h-full text-gray-500">
                                    <Loader2 className="animate-spin w-6 h-6 mr-2" />
                                    Rendering...
                                </div>
                            )}
                            <div className="absolute top-2 left-2 px-2 py-1 bg-black/50 backdrop-blur rounded text-xs font-mono">
                                Scene {scene.id}
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Final Concatenated Video */}
            {finalVideoPath && (
                <div className="glass-card p-6 space-y-4">
                    <h3 className="text-xl font-bold">Final Concatenated Video</h3>
                    <div className="aspect-video bg-black rounded-lg overflow-hidden">
                        <video
                            src={getVideoUrl(finalVideoPath)}
                            controls
                            className="w-full h-full"
                        />
                    </div>
                    <a
                        href={`${API_BASE}/api/sessions/download/${sessionId}/final_video.mp4`}
                        className="btn-primary px-6 py-3 flex items-center justify-center space-x-2 w-full"
                    >
                        <Download className="w-5 h-5" />
                        <span>Download Final Video</span>
                    </a>
                </div>
            )}

            {/* Concatenate Button */}
            {!finalVideoPath && (
                <div className="glass-card p-6">
                    <div className="flex items-center justify-between">
                        <div>
                            <h3 className="text-lg font-bold">Create Final Video</h3>
                            <p className="text-gray-400 text-sm">
                                {allVideosReady
                                    ? "Concatenate all scenes into a single video"
                                    : "Waiting for all videos to complete..."}
                            </p>
                        </div>
                        <button
                            onClick={handleConcatenate}
                            disabled={!allVideosReady || concatenating}
                            className={`btn-primary px-6 py-3 flex items-center space-x-2 ${!allVideosReady || concatenating ? "opacity-50 cursor-not-allowed" : ""
                                }`}
                        >
                            {concatenating ? (
                                <>
                                    <Loader2 className="animate-spin w-5 h-5" />
                                    <span>Concatenating...</span>
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
            <div className="glass-card p-6 flex items-center justify-between">
                <div>
                    <h3 className="text-lg font-bold">Download Individual Scenes</h3>
                    <p className="text-gray-400 text-sm">Download scene videos separately</p>
                </div>
                <div className="flex gap-2 flex-wrap">
                    {storyboard.scenes.map((scene) => scene.video_path && (
                        <a
                            key={scene.id}
                            href={`${API_BASE}/api/sessions/download/${sessionId}/scene_${scene.id}.mp4`}
                            className="btn-primary px-4 py-2 flex items-center space-x-2 text-sm"
                        >
                            <Download className="w-4 h-4" />
                            <span>Scene {scene.id}</span>
                        </a>
                    ))}
                </div>
            </div>
        </div>
    );
}
