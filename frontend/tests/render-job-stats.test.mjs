import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

// Compile the pure TS helper in memory so these tests also run on Node 20.
const source = readFileSync(new URL("../lib/render-job-stats.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } });
const { summarizeRenderJobs } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);

const job = (overrides = {}) => ({
    id: "job-1",
    type: "video",
    shot_id: "shot-1",
    input_snapshot: {},
    ...overrides,
});

test("counts only video jobs and sums frozen and provider snapshot durations", () => {
    const result = summarizeRenderJobs([
        job({ id: "frozen", input_snapshot: { duration_seconds: 6 } }),
        job({ id: "seedance", input_snapshot: { seedance: { duration: "4.5" } } }),
        job({ id: "segments", shot_id: null, input_snapshot: { segment_plan: [{ duration: 3 }, { duration: 2 }] } }),
        job({ id: "image", type: "image", input_snapshot: { duration_seconds: 99 } }),
    ], [{ id: "shot-1", duration_seconds: 20 }]);

    assert.deepEqual(result, { taskCount: 3, totalSeconds: 15.5 });
});

test("backfills legacy jobs from their shot, then H3 frames when the shot is gone", () => {
    const result = summarizeRenderJobs([
        job({ id: "current-shot" }),
        job({ id: "deleted-shot", shot_id: "missing", input_snapshot: { h3_parameters: { frames: 240 } } }),
    ], [{ id: "shot-1", duration_seconds: 7 }]);

    assert.deepEqual(result, { taskCount: 2, totalSeconds: 17 });
});

test("keeps a queued job's frozen duration when its shot is later edited", () => {
    const result = summarizeRenderJobs([
        job({ input_snapshot: { duration_seconds: 5 } }),
    ], [{ id: "shot-1", duration_seconds: 12 }]);

    assert.deepEqual(result, { taskCount: 1, totalSeconds: 5 });
});
