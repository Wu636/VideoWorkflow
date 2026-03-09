"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Save, Sparkles, Image as ImageIcon, Loader2, Wand2 } from "lucide-react";
import { Storyboard, Scene } from "@/types";
import { getScript, generateImages, updateScript, reviseScript } from "@/lib/api";

interface ScriptEditorProps {
    sessionId: string;
}

export default function ScriptEditor({ sessionId }: ScriptEditorProps) {
    const router = useRouter();
    const [storyboard, setStoryboard] = useState<Storyboard | null>(null);
    const [loading, setLoading] = useState(true);
    const [generating, setGenerating] = useState(false);
    const [saving, setSaving] = useState(false);
    const [revising, setRevising] = useState(false);
    const [feedback, setFeedback] = useState("");
    const [showReviseInput, setShowReviseInput] = useState(false);

    useEffect(() => {
        getScript(sessionId)
            .then(setStoryboard)
            .catch((e) => alert("Failed to load script: " + e))
            .finally(() => setLoading(false));
    }, [sessionId]);

    const handleSceneChange = (id: number, field: keyof Scene, value: string) => {
        if (!storyboard) return;
        const newScenes = storyboard.scenes.map(scene =>
            scene.id === id ? { ...scene, [field]: value } : scene
        );
        setStoryboard({ ...storyboard, scenes: newScenes });
    };

    const handleSave = async () => {
        if (!storyboard) return;
        setSaving(true);
        try {
            await updateScript(sessionId, storyboard);
        } catch (e) {
            alert("Failed to save: " + e);
        } finally {
            setSaving(false);
        }
    };

    const handleRevise = async () => {
        if (!feedback.trim()) return;
        setRevising(true);
        try {
            const newStoryboard = await reviseScript(sessionId, feedback);
            setStoryboard(newStoryboard);
            setFeedback("");
            setShowReviseInput(false);
        } catch (e) {
            alert("Failed to revise: " + e);
        } finally {
            setRevising(false);
        }
    };

    const handleGenerateImages = async () => {
        setGenerating(true);
        // Ensure we save latest changes before generating
        try {
            if (storyboard) await updateScript(sessionId, storyboard);
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
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                <div>
                    <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                        Script Studio
                    </h1>
                    <p className="text-gray-400">Review and polish your storyboard before filming</p>
                </div>
                <div className="flex space-x-3 w-full md:w-auto">
                    <button
                        onClick={() => setShowReviseInput(!showReviseInput)}
                        className={`btn-secondary px-4 py-2 border border-purple-500/30 rounded-lg hover:bg-purple-500/10 flex items-center space-x-2 text-purple-300 ${showReviseInput ? 'bg-purple-500/20' : ''}`}
                    >
                        <Wand2 className="w-4 h-4" />
                        <span>AI Revise</span>
                    </button>
                    <button
                        onClick={handleSave}
                        disabled={saving}
                        className="btn-secondary px-4 py-2 border border-white/10 rounded-lg hover:bg-white/5 flex items-center space-x-2"
                    >
                        {saving ? <Loader2 className="animate-spin w-4 h-4" /> : <Save className="w-4 h-4" />}
                        <span>{saving ? 'Saving...' : 'Save Draft'}</span>
                    </button>
                    <button
                        onClick={handleGenerateImages}
                        disabled={generating || revising}
                        className="btn-primary px-6 py-2 flex items-center space-x-2"
                    >
                        {generating ? <Loader2 className="animate-spin w-4 h-4" /> : <ImageIcon className="w-4 h-4" />}
                        <span>Generate Images</span>
                    </button>
                </div>
            </div>

            {/* AI Revision Input */}
            {showReviseInput && (
                <div className="glass-card p-6 border-purple-500/30 bg-purple-900/10 animate-in fade-in slide-in-from-top-2">
                    <h3 className="text-lg font-bold text-purple-300 mb-2 flex items-center gap-2">
                        <Sparkles className="w-5 h-5" />
                        AI Script Revision
                    </h3>
                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={feedback}
                            onChange={(e) => setFeedback(e.target.value)}
                            placeholder="e.g. 'Make the ending more dramatic' or 'Change the setting to a cyberpunk city'"
                            className="flex-1 input-premium p-3"
                            onKeyDown={(e) => e.key === 'Enter' && handleRevise()}
                        />
                        <button
                            onClick={handleRevise}
                            disabled={revising || !feedback.trim()}
                            className="btn-primary bg-purple-600 hover:bg-purple-500 px-6"
                        >
                            {revising ? <Loader2 className="animate-spin w-4 h-4" /> : "Apply"}
                        </button>
                    </div>
                </div>
            )}

            <div className="space-y-6">
                {storyboard.scenes.map((scene) => (
                    <div key={scene.id} className="glass-card p-6 flex gap-6">
                        <div className="flex-shrink-0 w-12 h-12 rounded-full bg-blue-600/20 flex items-center justify-center text-blue-400 font-bold border border-blue-500/30">
                            {scene.id}
                        </div>
                        <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-2">
                                <label className="text-xs font-uppercase tracking-wider text-gray-500">Narrative & Dialogue</label>
                                <textarea
                                    className="w-full h-32 input-premium p-3 text-sm resize-none focus:ring-2 focus:ring-blue-500/50 outline-none transition-all"
                                    value={scene.narrative}
                                    onChange={(e) => handleSceneChange(scene.id, 'narrative', e.target.value)}
                                    placeholder="Enter narrative description..."
                                />
                            </div>
                            <div className="space-y-2">
                                <label className="text-xs font-uppercase tracking-wider text-gray-500">Visual Prompt</label>
                                <textarea
                                    className="w-full h-32 input-premium p-3 text-sm resize-none font-mono text-gray-300 focus:ring-2 focus:ring-purple-500/50 outline-none transition-all"
                                    value={scene.visual_prompt}
                                    onChange={(e) => handleSceneChange(scene.id, 'visual_prompt', e.target.value)}
                                    placeholder="Enter visual prompt..."
                                />
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
