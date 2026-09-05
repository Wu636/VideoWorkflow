import type {
    ApprovalStatus,
    Asset,
    AssetRole,
    CharacterProfile,
    Delivery,
    H3PromptSkill,
    ImageProviderOption,
    Project,
    ProjectBrief,
    ProjectBundle,
    ProductionSeries,
    ProjectAnalysisDraft,
    ScriptRewriteDraft,
    SeedanceCatalog,
    SeedanceEstimate,
    SeedanceMaterialDiagnostics,
    RuntimeLogRecord,
    RuntimeSettingsPayload,
    RenderJob,
    Review,
    Shot,
    SceneProfile,
    StyleAnalysisDraft,
    Storyboard,
    TemplateOption,
    VideoProviderOption,
} from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001/api";

async function api<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${path}`, {
        cache: "no-store",
        ...init,
        headers: init?.body instanceof FormData
            ? init.headers
            : { "Content-Type": "application/json", ...(init?.headers || {}) },
    });
    if (!response.ok) {
        let message = await response.text();
        try {
            const parsed = JSON.parse(message) as { detail?: string };
            message = parsed.detail || message;
        } catch {
            // Keep the backend response body.
        }
        throw new Error(message || `Request failed (${response.status})`);
    }
    return response.json() as Promise<T>;
}

export function projectDownloadUrl(projectId: string, kind: "asset" | "job" | "delivery" | "preview", itemId: string) {
    return `${API_BASE}/projects/${projectId}/download/${kind}/${itemId}`;
}

export function projectInlineUrl(projectId: string, kind: "asset" | "job" | "delivery" | "preview", itemId: string) {
    return `${projectDownloadUrl(projectId, kind, itemId)}?inline=true`;
}

export function storyboardCsvUrl(projectId: string) {
    return `${API_BASE}/projects/${projectId}/storyboard.csv`;
}

export async function listProjects(): Promise<Project[]> {
    return api<Project[]>("/projects");
}

export async function createProject(brief: Partial<ProjectBrief> & Pick<ProjectBrief, "title" | "story">): Promise<Project> {
    return api<Project>("/projects", { method: "POST", body: JSON.stringify(normalizeProjectBrief(brief)) });
}

function normalizeProjectBrief(brief: Partial<ProjectBrief> & Pick<ProjectBrief, "title" | "story">): ProjectBrief {
    const portrait = brief.aspect_ratio === "9:16";
    const square = brief.aspect_ratio === "1:1";
    return {
        title: brief.title,
        client_name: brief.client_name || "",
        story: brief.story,
        target_duration_seconds: brief.target_duration_seconds || 30,
        aspect_ratio: brief.aspect_ratio || "16:9",
        width: brief.width || (portrait ? 768 : square ? 1024 : 1344),
        height: brief.height || (portrait ? 1344 : square ? 1024 : 768),
        fps: brief.fps || 24,
        language: brief.language || "zh-CN",
        visual_style: brief.visual_style || "",
        pacing: brief.pacing || "",
        audience: brief.audience || "",
        negative_prompt: brief.negative_prompt || "",
        delivery_notes: brief.delivery_notes || "",
    };
}

export async function listProductionSeries(): Promise<ProductionSeries[]> {
    return api<ProductionSeries[]>("/projects/series");
}

export async function saveProjectAsSeries(projectId: string, name = "", description = ""): Promise<{ series: ProductionSeries; project: Project }> {
    return api(`/projects/${projectId}/series`, {
        method: "POST",
        body: JSON.stringify({ name, description }),
    });
}

export async function createSeriesEpisode(
    seriesId: string,
    brief: Partial<ProjectBrief> & Pick<ProjectBrief, "title" | "story">,
    characterIds: string[],
): Promise<Project> {
    return api(`/projects/series/${seriesId}/episodes`, {
        method: "POST",
        body: JSON.stringify({ brief: normalizeProjectBrief(brief), character_ids: characterIds }),
    });
}

export async function getProject(projectId: string): Promise<ProjectBundle> {
    return api<ProjectBundle>(`/projects/${projectId}`);
}

export async function updateProject(project: Project): Promise<Project> {
    return api<Project>(`/projects/${project.id}`, { method: "PUT", body: JSON.stringify(project) });
}

export async function deleteProject(projectId: string): Promise<{ deleted: boolean }> {
    return api(`/projects/${projectId}`, { method: "DELETE" });
}

export interface LegacySession {
    session_id: string;
    topic: string;
    scene_count: number;
    has_images: boolean;
    has_videos: boolean;
}

export async function listLegacySessions(): Promise<LegacySession[]> {
    return api("/projects/legacy/sessions");
}

export async function importLegacySession(sessionId: string): Promise<Project> {
    return api(`/projects/legacy/import/${encodeURIComponent(sessionId)}`, { method: "POST", body: "{}" });
}

export async function generateStoryboard(
    projectId: string,
    shotCount?: number,
    countMode: "manual" | "ai" = "manual",
    userSuggestions = "",
    promptTargets?: ("h3" | "seedance")[],
    h3SkillId = "h3-prompt-writing",
): Promise<{ shots: Shot[]; project: Project }> {
    return api(`/projects/${projectId}/storyboard/generate`, {
        method: "POST",
        body: JSON.stringify({ shot_count: shotCount || null, count_mode: countMode, user_suggestions: userSuggestions, prompt_targets: promptTargets, h3_skill_id: h3SkillId }),
    });
}

export async function analyzeProjectStyle(projectId: string, assetIds?: string[], apply = true): Promise<StyleAnalysisDraft> {
    return api(`/projects/${projectId}/style/analyze`, {
        method: "POST",
        body: JSON.stringify({ asset_ids: assetIds || null, apply }),
    });
}

export async function generateSceneProfiles(projectId: string, userSuggestions = ""): Promise<SceneProfile[]> {
    return api(`/projects/${projectId}/scene-profiles/generate`, {
        method: "POST",
        body: JSON.stringify({ user_suggestions: userSuggestions }),
    });
}

export interface ApplySceneProfilesResult {
    applied_count: number;
    shot_ids: string[];
    shot_ordinals: number[];
    scene_count: number;
    h3_preserved_count: number;
    keyframes_preserved_count: number;
}

export async function applySceneProfiles(projectId: string, sceneProfileIds?: string[]): Promise<ApplySceneProfilesResult> {
    return api(`/projects/${projectId}/scene-profiles/apply`, {
        method: "POST",
        body: JSON.stringify({ scene_profile_ids: sceneProfileIds || null }),
    });
}

export async function getSceneReferencePrompt(projectId: string, sceneProfileId: string): Promise<{ prompt: string; default_prompt: string; profile: SceneProfile }> {
    return api(`/projects/${projectId}/scene-profiles/${sceneProfileId}/reference-prompt`);
}

export async function previewSceneReferencePrompt(projectId: string, profile: SceneProfile): Promise<{ prompt: string }> {
    return api(`/projects/${projectId}/scene-profiles/${profile.id}/reference-prompt`, {
        method: "POST",
        body: JSON.stringify({ ...profile, expected_version: profile.version }),
    });
}

export async function updateSceneProfile(projectId: string, profile: SceneProfile): Promise<SceneProfile> {
    return api(`/projects/${projectId}/scene-profiles/${profile.id}`, {
        method: "PATCH",
        body: JSON.stringify({ expected_version: profile.version, name: profile.name, description: profile.description,
            continuity_notes: profile.continuity_notes, reference_prompt: profile.reference_prompt }),
    });
}

export async function retrySceneReferenceDownload(projectId: string, sceneProfileId: string): Promise<Asset> {
    return api(`/projects/${projectId}/scene-profiles/${sceneProfileId}/reference/retry-download`, { method: "POST" });
}

export async function generateSceneReference(projectId: string, sceneProfileId: string, prompt: string, expectedVersion: number): Promise<Asset> {
    return api(`/projects/${projectId}/scene-profiles/${sceneProfileId}/reference`, {
        method: "POST",
        body: JSON.stringify({ prompt, expected_version: expectedVersion }),
    });
}

export async function generateCharacterReferences(projectId: string, characterIds?: string[], userSuggestions = ""): Promise<Asset[]> {
    return api(`/projects/${projectId}/characters/references/generate`, {
        method: "POST",
        body: JSON.stringify({ character_ids: characterIds || null, user_suggestions: userSuggestions }),
    });
}

export async function generateCharacterReferencesWithOptions(
    projectId: string,
    options: {
        characterIds?: string[];
        userSuggestions?: string;
        referenceAssetIds?: string[];
        appearanceProfileId?: string | null;
    },
): Promise<Asset[]> {
    return api(`/projects/${projectId}/characters/references/generate`, {
        method: "POST",
        body: JSON.stringify({
            character_ids: options.characterIds || null,
            user_suggestions: options.userSuggestions || "",
            reference_asset_ids: options.referenceAssetIds || null,
            appearance_profile_id: options.appearanceProfileId || null,
        }),
    });
}

export interface MissingCharacterReferencesResult {
    assets: Asset[];
    generated_character_ids: string[];
    character_count: number;
    missing_count: number;
    message: string;
}

export async function generateMissingCharacterReferences(
    projectId: string,
    options: {
        mode: "script_style" | "complete_missing";
        userSuggestions?: string;
        referenceAssetIds?: string[];
    },
): Promise<MissingCharacterReferencesResult> {
    return api(`/projects/${projectId}/characters/references/generate-missing`, {
        method: "POST",
        body: JSON.stringify({
            mode: options.mode,
            user_suggestions: options.userSuggestions || "",
            reference_asset_ids: options.referenceAssetIds || null,
        }),
    });
}

export async function analyzeCharacterReferences(
    projectId: string,
    characterId: string,
    referenceAssetIds: string[],
    options: { appearanceProfileId?: string | null; appearanceLabel?: string; userSuggestions?: string } = {},
): Promise<CharacterProfile> {
    return api(`/projects/${projectId}/characters/references/analyze`, {
        method: "POST",
        body: JSON.stringify({
            character_id: characterId,
            reference_asset_ids: referenceAssetIds,
            appearance_profile_id: options.appearanceProfileId || null,
            appearance_label: options.appearanceLabel || "",
            user_suggestions: options.userSuggestions || "",
        }),
    });
}

export async function importStoryboard(
    projectId: string,
    file: File,
    userSuggestions = "",
    promptTargets: ("h3" | "seedance")[] = [],
): Promise<{ shots: Shot[]; project: Project; imported_rows: number }> {
    const data = new FormData();
    data.append("file", file);
    data.append("user_suggestions", userSuggestions);
    data.append("prompt_targets", promptTargets.join(","));
    return api(`/projects/${projectId}/storyboard/import`, { method: "POST", body: data });
}

export async function analyzeProjectBrief(projectId: string): Promise<ProjectAnalysisDraft> {
    return api(`/projects/${projectId}/brief/analyze`, { method: "POST", body: "{}" });
}

export async function rewriteProjectScript(projectId: string, mode: "auto" | "expand" | "shorten", userSuggestions = ""): Promise<ScriptRewriteDraft> {
    return api(`/projects/${projectId}/brief/rewrite`, {
        method: "POST",
        body: JSON.stringify({ mode, user_suggestions: userSuggestions }),
    });
}

export async function uploadProjectScript(projectId: string, file: File): Promise<{ asset: Asset; project: Project; extracted_characters: number }> {
    const data = new FormData();
    data.append("file", file);
    return api(`/projects/${projectId}/script/upload`, { method: "POST", body: data });
}

export async function getRuntimeSettings(): Promise<RuntimeSettingsPayload> {
    return api("/settings");
}

export async function updateRuntimeSettings(values: Record<string, unknown>, clearKeys: string[] = []): Promise<RuntimeSettingsPayload> {
    return api("/settings", { method: "PUT", body: JSON.stringify({ values, clear_keys: clearKeys }) });
}

export async function getRuntimeLogs(options: { level?: string; search?: string; limit?: number } = {}): Promise<{ records: RuntimeLogRecord[]; file: string }> {
    const query = new URLSearchParams();
    if (options.level) query.set("level", options.level);
    if (options.search) query.set("search", options.search);
    query.set("limit", String(options.limit || 500));
    return api(`/logs?${query}`);
}

export async function clearRuntimeLogs(): Promise<{ cleared: boolean }> {
    return api("/logs", { method: "DELETE" });
}

export function runtimeLogsDownloadUrl() {
    return `${API_BASE}/logs/download`;
}

export async function approveStoryboard(projectId: string, approved: boolean, comment = "", reviewer = ""): Promise<{ project: Project; review: Review }> {
    return api(`/projects/${projectId}/storyboard/approve`, {
        method: "POST",
        body: JSON.stringify({ approved, comment, reviewer }),
    });
}

export async function updateShot(shot: Shot): Promise<Shot> {
    return api<Shot>(`/projects/${shot.project_id}/shots/${shot.id}`, {
        method: "PUT",
        body: JSON.stringify(shot),
    });
}

export async function updateKeyframePrompt(
    projectId: string,
    shotId: string,
    patch: Pick<Shot, "keyframe_prompt"> & Partial<Pick<Shot, "keyframe_revision_suggestion_draft" | "keyframe_revision_mode">>,
): Promise<Shot> {
    return api<Shot>(`/projects/${projectId}/shots/${shotId}/keyframe-prompt`, {
        method: "PATCH",
        body: JSON.stringify(patch),
    });
}

export async function reviseShotWithAi(
    projectId: string,
    shotId: string,
    userSuggestions: string,
    promptTargets: ("h3" | "seedance")[],
    h3SkillId = "h3-prompt-writing",
): Promise<Shot> {
    return api(`/projects/${projectId}/shots/${shotId}/ai-revise`, {
        method: "POST",
        body: JSON.stringify({ user_suggestions: userSuggestions, prompt_targets: promptTargets, h3_skill_id: h3SkillId }),
    });
}

export async function insertShotWithAi(
    projectId: string,
    afterShotId: string | null,
    userSuggestions: string,
    promptTargets: ("h3" | "seedance")[],
    h3SkillId = "h3-prompt-writing",
): Promise<Shot> {
    return api(`/projects/${projectId}/shots/insert-ai`, {
        method: "POST",
        body: JSON.stringify({ after_shot_id: afterShotId, user_suggestions: userSuggestions, prompt_targets: promptTargets, h3_skill_id: h3SkillId }),
    });
}

export async function getSeedanceMaterials(projectId: string, shotId: string): Promise<SeedanceMaterialDiagnostics> {
    return api(`/projects/${projectId}/shots/${shotId}/seedance-materials`);
}

export async function createShot(projectId: string, shot: Shot): Promise<Shot> {
    return api<Shot>(`/projects/${projectId}/shots`, { method: "POST", body: JSON.stringify(shot) });
}

export async function createBlankShot(projectId: string): Promise<Shot> {
    return api<Shot>(`/projects/${projectId}/shots/blank`, { method: "POST", body: "{}" });
}

export async function reorderStoryboard(projectId: string, shotIds: string[]): Promise<Shot[]> {
    return api<Shot[]>(`/projects/${projectId}/storyboard/reorder`, {
        method: "POST",
        body: JSON.stringify({ shot_ids: shotIds }),
    });
}

export async function deleteShot(projectId: string, shotId: string): Promise<{ deleted: boolean }> {
    return api(`/projects/${projectId}/shots/${shotId}`, { method: "DELETE" });
}

export async function uploadProjectAsset(
    projectId: string,
    file: File,
    role: AssetRole,
    options: { name?: string; characterId?: string; description?: string } = {},
): Promise<Asset> {
    const data = new FormData();
    data.append("file", file);
    data.append("role", role);
    data.append("name", options.name || file.name);
    if (options.characterId) data.append("character_id", options.characterId);
    if (options.description) data.append("description", options.description);
    return api<Asset>(`/projects/${projectId}/assets`, { method: "POST", body: data });
}

export async function deleteProjectAsset(projectId: string, assetId: string): Promise<{ deleted: boolean }> {
    return api(`/projects/${projectId}/assets/${assetId}`, { method: "DELETE" });
}

export async function generateKeyframes(
    projectId: string,
    shotIds?: string[],
    options: { revisionMode?: "fresh" | "iterate"; userSuggestions?: string } = {},
): Promise<Shot[]> {
    return api(`/projects/${projectId}/keyframes/generate`, {
        method: "POST",
        body: JSON.stringify({
            shot_ids: shotIds || null,
            revision_mode: options.revisionMode || "fresh",
            user_suggestions: options.userSuggestions || "",
        }),
    });
}

export async function getH3PromptSkills(): Promise<{ default_skill_id: string; director_version: string; skills: H3PromptSkill[] }> {
    return api("/projects/h3-prompt-skills");
}

export async function uploadKeyframe(projectId: string, shotId: string, file: File): Promise<Shot> {
    const data = new FormData();
    data.append("file", file);
    return api(`/projects/${projectId}/shots/${shotId}/keyframe/upload`, { method: "POST", body: data });
}

export async function generateH3Prompts(
    projectId: string,
    shotIds: string[],
    skillId: string,
    userSuggestions = "",
): Promise<Shot[]> {
    return api(`/projects/${projectId}/h3-prompts/generate`, {
        method: "POST",
        body: JSON.stringify({
            shot_ids: shotIds,
            skill_id: skillId,
            user_suggestions: userSuggestions,
        }),
    });
}

export async function generateSeedancePrompts(projectId: string, shotIds: string[]): Promise<Shot[]> {
    return api(`/projects/${projectId}/seedance-prompts/generate`, {
        method: "POST",
        body: JSON.stringify({ shot_ids: shotIds }),
    });
}

export async function getSeedanceCatalog(): Promise<SeedanceCatalog> {
    return api("/projects/seedance/catalog");
}

export async function estimateSeedance(
    projectId: string,
    shotIds: string[],
    modelId: string,
    resolution: string,
): Promise<SeedanceEstimate> {
    return api(`/projects/${projectId}/seedance/estimate`, {
        method: "POST",
        body: JSON.stringify({ shot_ids: shotIds, model_id: modelId, resolution }),
    });
}

export async function exportKeyframes(projectId: string, shotIds: string[]): Promise<{ blob: Blob; filename: string }> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/keyframes/export`, {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ shot_ids: shotIds }),
    });
    if (!response.ok) {
        let message = await response.text();
        try {
            const parsed = JSON.parse(message) as { detail?: string };
            message = parsed.detail || message;
        } catch {
            // Keep the backend response body.
        }
        throw new Error(message || `Export failed (${response.status})`);
    }
    const disposition = response.headers.get("Content-Disposition") || "";
    const encoded = disposition.match(/filename\*=utf-8''([^;]+)/i)?.[1];
    const quoted = disposition.match(/filename="?([^";]+)"?/i)?.[1];
    const filename = encoded ? decodeURIComponent(encoded) : quoted || `keyframes-${projectId}.zip`;
    return { blob: await response.blob(), filename };
}

export async function planRender(projectId: string): Promise<Shot[]> {
    return api(`/projects/${projectId}/render/plan`, { method: "POST", body: "{}" });
}

export async function enqueueRender(projectId: string, shotIds?: string[], options: {
    provider?: "comfyui_h3" | "metaso_h3" | "atlas_h3" | "ark_seedance";
    modelId?: string;
    resolution?: string;
    generateAudio?: boolean;
} = {}): Promise<RenderJob[]> {
    return api(`/projects/${projectId}/render`, {
        method: "POST",
        body: JSON.stringify({
            shot_ids: shotIds || null,
            provider: options.provider || "comfyui_h3",
            model_id: options.modelId || null,
            resolution: options.resolution || null,
            generate_audio: options.generateAudio ?? true,
        }),
    });
}

export async function getRenderJobs(projectId: string): Promise<RenderJob[]> {
    return api(`/projects/${projectId}/jobs`);
}

export async function cancelRenderJob(projectId: string, jobId: string): Promise<RenderJob> {
    return api(`/projects/${projectId}/jobs/${jobId}/cancel`, { method: "POST", body: "{}" });
}

export async function deleteRenderJob(projectId: string, jobId: string): Promise<{ deleted: boolean }> {
    return api(`/projects/${projectId}/jobs/${jobId}`, { method: "DELETE" });
}

export async function comfyPreflight(projectId: string): Promise<Record<string, unknown> & { online: boolean; ok: boolean; error?: string }> {
    return api(`/projects/${projectId}/comfyui/preflight`);
}

export async function seedancePreflight(projectId: string, modelId: string, resolution: string): Promise<Record<string, unknown> & { online: boolean; ok: boolean; message?: string }> {
    const query = new URLSearchParams({ model_id: modelId, resolution });
    return api(`/projects/${projectId}/seedance/preflight?${query}`);
}

export async function generateSubtitles(projectId: string): Promise<Asset> {
    return api(`/projects/${projectId}/subtitles/generate`, { method: "POST", body: "{}" });
}

export async function finalizeProject(projectId: string, options: {
    shot_ids?: string[] | null;
    output_name?: string;
    crossfade_seconds?: number;
    burn_subtitles?: boolean;
    subtitle_asset_id?: string | null;
    background_music_asset_id?: string | null;
    background_music_volume?: number;
    normalize_audio?: boolean;
    preview?: boolean;
}): Promise<Delivery> {
    return api(`/projects/${projectId}/finalize`, { method: "POST", body: JSON.stringify(options) });
}

export async function getPublicReview(token: string): Promise<ProjectBundle> {
    return api(`/review/${token}`);
}

export async function submitPublicReview(token: string, request: {
    target_type: "storyboard" | "shot" | "delivery";
    target_id: string;
    decision: ApprovalStatus;
    comment?: string;
    reviewer?: string;
}): Promise<Review> {
    return api(`/review/${token}`, { method: "POST", body: JSON.stringify(request) });
}


export async function createSession(data: {
    topic: string;
    reference_image?: string; // file path from upload
    template?: string;
    count?: number;
    include_dialogue?: boolean;
    character_description?: string;
    image_style?: string;
    image_provider?: string;
    image_model?: string;
    video_provider?: string;
    video_model?: string;
    video_aspect_ratio?: string;
}) {
    const res = await fetch(`${API_BASE}/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function getScript(sessionId: string): Promise<Storyboard> {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/script`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function getTemplates(): Promise<TemplateOption[]> {
    const res = await fetch(`${API_BASE}/sessions/templates`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function getImageOptions(): Promise<ImageProviderOption[]> {
    const res = await fetch(`${API_BASE}/sessions/image-options`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function getVideoOptions(): Promise<VideoProviderOption[]> {
    const res = await fetch(`${API_BASE}/sessions/video-options`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}


export async function updateScript(sessionId: string, storyboard: Storyboard): Promise<Storyboard> {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/script`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(storyboard),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function reviseScript(sessionId: string, feedback: string, referenceImage?: string) {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/script/revise`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ feedback, reference_image: referenceImage }),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function uploadFile(file: File) {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API_BASE}/files/upload`, {
        method: "POST",
        body: formData,
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json(); // { path: string, filename: string }
}

export async function analyzeImage(file: File): Promise<{ character: string | null; style: string | null }> {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API_BASE}/files/analyze`, {
        method: "POST",
        body: formData,
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function generateImages(sessionId: string, sceneIds?: number[], referenceImage?: string) {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/images`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scene_ids: sceneIds, reference_image: referenceImage }),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function generateVideos(
    sessionId: string,
    sceneIds?: number[],
    retryFailedOnly: boolean = true,
    videoProvider: string = "ark",
) {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/videos`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            scene_ids: sceneIds,
            retry_failed_only: retryFailedOnly,
            video_provider: videoProvider,
        }),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function concatenateVideos(sessionId: string) {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/concatenate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}
