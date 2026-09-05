import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

// Compile the pure TS policy in memory so these tests also run on Node 20.
const source = readFileSync(new URL("../lib/production-busy.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } });
const { isKeyframeBusy, isH3PromptBlocked } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);
const shot = { id: "shot-1", image_status: "pending" };

test("H3 generation leaves first-frame editing, upload and generation available", () => {
    assert.equal(isKeyframeBusy(shot, new Set(["h3-prompts"]), new Set()), false);
});

test("same-shot image tasks lock only their own shot, even when H3 started first", () => {
    for (const task of ["keyframe-prompt", "upload-keyframe"]) {
        const busy = new Set(["h3-prompts", `${task}-shot-1`]);
        assert.equal(isKeyframeBusy(shot, busy, new Set()), true);
        assert.equal(isKeyframeBusy({ ...shot, id: "shot-2" }, busy, new Set()), false);
    }
});

test("queued/server-side and local image tasks prevent duplicate generation", () => {
    assert.equal(isKeyframeBusy(shot, new Set(), new Set([shot.id])), true);
    assert.equal(isKeyframeBusy({ ...shot, image_status: "processing" }, new Set(), new Set()), true);
    assert.equal(isKeyframeBusy({ ...shot, image_status: "failed" }, new Set(), new Set()), false);
    assert.equal(isKeyframeBusy(shot, new Set(), new Set(["shot-2"])), false);
});

test("keyframe saving and uploading leave H3 generation available", () => {
    assert.equal(isH3PromptBlocked(new Set(["keyframe-prompt-shot-1", "upload-keyframe-shot-2"])), false);
    assert.equal(isH3PromptBlocked(new Set()), false);
});

test("H3 stays guarded regardless of which parallel task started first", () => {
    assert.equal(isH3PromptBlocked(new Set(["keyframe-prompt-shot-1", "h3-prompts"])), true);
    assert.equal(isH3PromptBlocked(new Set(["h3-prompts", "keyframe-prompt-shot-1"])), true);
});

test("conflicting whole-shot writers retain their guard", () => {
    for (const task of ["h3-shot-1", "seedance-shot-1", "plan", "preset-fast"]) {
        assert.equal(isKeyframeBusy(shot, new Set([task]), new Set()), true);
        assert.equal(isH3PromptBlocked(new Set([task])), true);
    }
});
