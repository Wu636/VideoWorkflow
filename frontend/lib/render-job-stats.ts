import type { RenderJob, Shot } from "@/types";

export interface RenderJobStats {
    taskCount: number;
    totalSeconds: number;
}

function positiveNumber(value: unknown): number | null {
    const parsed = typeof value === "number"
        ? value
        : typeof value === "string" && value.trim()
            ? Number(value)
            : Number.NaN;
    return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function snapshotDurationSeconds(job: RenderJob): number | null {
    const snapshot = job.input_snapshot || {};
    const frozenDuration = positiveNumber(snapshot.duration_seconds);
    if (frozenDuration !== null) return frozenDuration;

    const seedance = snapshot.seedance;
    if (seedance && typeof seedance === "object" && !Array.isArray(seedance)) {
        const duration = positiveNumber((seedance as Record<string, unknown>).duration);
        if (duration !== null) return duration;
    }

    const segmentPlan = snapshot.segment_plan;
    if (Array.isArray(segmentPlan) && segmentPlan.length > 0) {
        const durations = segmentPlan.map((segment) => (
            segment && typeof segment === "object" && !Array.isArray(segment)
                ? positiveNumber((segment as Record<string, unknown>).duration)
                : null
        ));
        if (durations.every((duration): duration is number => duration !== null)) {
            return durations.reduce((sum, duration) => sum + duration, 0);
        }
    }
    return null;
}

function legacyH3DurationSeconds(job: RenderJob): number | null {
    const parameters = job.input_snapshot?.h3_parameters;
    if (!parameters || typeof parameters !== "object" || Array.isArray(parameters)) return null;
    const frames = positiveNumber((parameters as Record<string, unknown>).frames);
    return frames === null ? null : frames / 24;
}

export function summarizeRenderJobs(jobs: RenderJob[], shots: Shot[]): RenderJobStats {
    const shotDurations = new Map(shots.map((shot) => [shot.id, shot.duration_seconds]));
    const videoJobs = jobs.filter((job) => job.type === "video");
    const totalSeconds = videoJobs.reduce((sum, job) => {
        const duration = snapshotDurationSeconds(job)
            ?? (job.shot_id ? positiveNumber(shotDurations.get(job.shot_id)) : null)
            ?? legacyH3DurationSeconds(job)
            ?? 0;
        return sum + duration;
    }, 0);
    return { taskCount: videoJobs.length, totalSeconds };
}
