"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Save, Sparkles, Image as ImageIcon, Loader2 } from "lucide-react";
import { Storyboard, Scene } from "@/types";
import { getScript, generateImages } from "@/lib/api";

interface ScriptEditorProps {
    sessionId: string;
}

export default function ScriptEditor({ sessionId }: ScriptEditorProps) {
    const router = useRouter();
    const [storyboard, setStoryboard] = useState<Storyboard | null>(null);
    const [loading, setLoading] = useState(true);
    const [generating, setGenerating] = useState(false);

    useEffect(() => {
        getScript(sessionId)
            .then(setStoryboard)
            .catch((e) => alert("Failed to load script: " + e))
            .finally(() => setLoading(false));
    }, [sessionId]);

    const handleGenerateImages = async () => {
        setGenerating(true);
        try {
            await generateImages(sessionId);
            router.push(`/workspace/${sessionId}/visuals`);
        } catch (e) {
            alert("Failed to start generation: " + e);
            setGenerating(false);
        }
    };

    if (loading) return <div className="text-center p-12"><Loader2 className="animate-spin w-8 h-8 mx-auto" /></div>;
    if (!storyboard) return <div className="text-center p-12 text-red-400">Script not found</div>;

    return (
        <div className="max-w-6xl mx-auto p-6 space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                        Script Studio
                    </h1>
                    <p className="text-gray-400">Review and polish your storyboard before filming</p>
                </div>
                <div className="flex space-x-3">
                    <button className="btn-secondary px-4 py-2 border border-white/10 rounded-lg hover:bg-white/5 flex items-center space-x-2">
                        <Save className="w-4 h-4" />
                        <span>Save Draft</span>
                    </button>
                    <button
                        onClick={handleGenerateImages}
                        disabled={generating}
                        className="btn-primary px-6 py-2 flex items-center space-x-2"
                    >
                        {generating ? <Loader2 className="animate-spin w-4 h-4" /> : <ImageIcon className="w-4 h-4" />}
                        <span>Generate Images</span>
                    </button>
                </div>
            </div>

            <div className="space-y-6">
                {storyboard.scenes.map((scene, idx) => (
                    <div key={scene.id} className="glass-card p-6 flex gap-6">
                        <div className="flex-shrink-0 w-12 h-12 rounded-full bg-blue-600/20 flex items-center justify-center text-blue-400 font-bold border border-blue-500/30">
                            {scene.id}
                        </div>
                        <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-2">
                                <label className="text-xs font-uppercase tracking-wider text-gray-500">Narrative & Dialogue</label>
                                <textarea
                                    className="w-full h-32 input-premium p-3 text-sm resize-none"
                                    value={scene.narrative}
                                    readOnly
                                />
                            </div>
                            <div className="space-y-2">
                                <label className="text-xs font-uppercase tracking-wider text-gray-500">Visual Prompt</label>
                                <textarea
                                    className="w-full h-32 input-premium p-3 text-sm resize-none font-mono text-gray-300"
                                    value={scene.visual_prompt}
                                    readOnly
                                />
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
