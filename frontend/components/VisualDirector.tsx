"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Play, RotateCcw, CheckCircle, Loader2 } from "lucide-react";
import { Storyboard, Scene } from "@/types";
import { getScript, generateVideos } from "@/lib/api";

interface Props {
    sessionId: string;
}

const API_BASE = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001/api").replace("/api", "");

function getImageUrl(path?: string) {
    if (!path) return "";
    // Static files are mounted at /static/ which points to OUTPUT_DIR (outputs/)
    // So we need to remove the "outputs/" prefix from paths
    let cleanPath = path;
    if (cleanPath.startsWith("outputs/")) {
        cleanPath = cleanPath.substring("outputs/".length);
    } else if (cleanPath.includes("/outputs/")) {
        const parts = cleanPath.split("/outputs/");
        cleanPath = parts[parts.length - 1];
    }
    return `${API_BASE}/static/${cleanPath}`;
}

export default function VisualDirector({ sessionId }: Props) {
    const router = useRouter();
    const [storyboard, setStoryboard] = useState<Storyboard | null>(null);
    const [loading, setLoading] = useState(true);
    const [generating, setGenerating] = useState(false);

    useEffect(() => {
        const poll = setInterval(() => {
            getScript(sessionId).then(sb => {
                setStoryboard(sb);
                // Check if all images are done
                // Simple fallback: checking if image_path exists for all scenes
                // In real app, check 'image_status'
            });
        }, 2000);

        getScript(sessionId)
            .then(setStoryboard)
            .catch((e) => console.error(e))
            .finally(() => setLoading(false));

        return () => clearInterval(poll);
    }, [sessionId]);

    const handleGenerateVideos = async () => {
        setGenerating(true);
        try {
            await generateVideos(sessionId);
            router.push(`/workspace/${sessionId}/cinema`);
        } catch (e) {
            alert("Failed to start video generation: " + e);
            setGenerating(false);
        }
    };

    if (loading) return <div className="text-center p-12"><Loader2 className="animate-spin w-8 h-8 mx-auto" /></div>;
    if (!storyboard) return <div className="text-center p-12 text-red-400">Loading Session...</div>;

    return (
        <div className="max-w-6xl mx-auto p-6 space-y-8">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                        Visual Director
                    </h1>
                    <p className="text-gray-400">Review generated keyframes before filming</p>
                </div>
                <button
                    onClick={handleGenerateVideos}
                    disabled={generating}
                    className="btn-primary px-6 py-2 flex items-center space-x-2"
                >
                    {generating ? <Loader2 className="animate-spin w-4 h-4" /> : <Play className="w-4 h-4" />}
                    <span>Generate Videos</span>
                </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {storyboard.scenes.map((scene) => (
                    <div key={scene.id} className="glass-card overflow-hidden group">
                        <div className="aspect-video bg-black/50 relative">
                            {scene.image_path && getImageUrl(scene.image_path) ? (
                                <img
                                    src={getImageUrl(scene.image_path)}
                                    className="w-full h-full object-cover"
                                    alt={`Scene ${scene.id}`}
                                />
                            ) : (
                                <div className="flex items-center justify-center h-full text-gray-500">
                                    <Loader2 className="animate-spin w-6 h-6 mr-2" />
                                    Generating...
                                </div>
                            )}

                            <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center space-x-4">
                                <button className="p-2 rounded-full bg-white/10 hover:bg-white/20 text-white" title="Regenerate">
                                    <RotateCcw className="w-5 h-5" />
                                </button>
                            </div>

                            <div className="absolute top-2 left-2 px-2 py-1 bg-black/50 backdrop-blur rounded text-xs font-mono">
                                Scene {scene.id}
                            </div>
                        </div>
                        <div className="p-4 space-y-2">
                            <p className="text-xs text-gray-400 line-clamp-2">{scene.visual_prompt}</p>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
