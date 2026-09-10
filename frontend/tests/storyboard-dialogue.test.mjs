import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

const source = readFileSync(new URL("../lib/storyboard-dialogue.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } });
const { buildStoryboardDialoguePatch, getStoryboardDialogueText } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);

const systemVoice = {
    kind: "system_vo",
    speaker_id: null,
    speaker_name: "系统",
    text: "第一关开始。",
    start_seconds: 0.5,
    end_seconds: 2.5,
    lip_sync: false,
};

test("shows generated structured voice text when the legacy dialogue field is empty", () => {
    assert.equal(getStoryboardDialogueText({
        dialogue: "",
        dialogue_turns: [],
        voice_events: [systemVoice],
    }), "第一关开始。");
});

test("keeps the structured speaker and timing when editing one generated voice event", () => {
    const patch = buildStoryboardDialoguePatch({
        dialogue: "",
        dialogue_turns: [],
        voice_events: [systemVoice],
    }, "第一关现在开始。");

    assert.equal(patch.dialogue, "");
    assert.equal(patch.voice_events[0].text, "第一关现在开始。");
    assert.equal(patch.voice_events[0].kind, "system_vo");
    assert.equal(patch.voice_events[0].start_seconds, 0.5);
});

test("renders and edits multiple voice events one per line without losing speakers", () => {
    const characterVoice = {
        kind: "character",
        speaker_id: "character-1",
        speaker_name: "陆峥",
        text: "立即停止作业。",
        start_seconds: 3,
        end_seconds: 5,
        lip_sync: true,
    };
    const shot = { dialogue: "", dialogue_turns: [], voice_events: [systemVoice, characterVoice] };
    assert.equal(getStoryboardDialogueText(shot), "第一关开始。\n立即停止作业。");

    const patch = buildStoryboardDialoguePatch(shot, "规则播报完毕。\n马上停止作业。");
    assert.deepEqual(patch.voice_events.map((event) => event.text), ["规则播报完毕。", "马上停止作业。"]);
    assert.equal(patch.voice_events[1].speaker_id, "character-1");
    assert.deepEqual(patch.dialogue_turns, [{ speaker_id: "character-1", text: "马上停止作业。" }]);
});

test("keeps an explicit legacy dialogue value authoritative", () => {
    assert.equal(getStoryboardDialogueText({
        dialogue: "使用人工填写的对白",
        dialogue_turns: [],
        voice_events: [systemVoice],
    }), "使用人工填写的对白");
});
