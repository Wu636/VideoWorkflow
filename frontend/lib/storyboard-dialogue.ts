import type { Shot, VoiceEvent } from "@/types";

type DialogueShot = Pick<Shot, "dialogue" | "dialogue_turns" | "voice_events">;
type DialoguePatch = Pick<Shot, "dialogue" | "dialogue_turns" | "voice_events">;

function dialogueTurns(events: VoiceEvent[]) {
    return events
        .filter((event) => event.kind === "character" && event.text.trim())
        .map((event) => ({ speaker_id: event.speaker_id, text: event.text }));
}

export function getStoryboardDialogueText(shot: DialogueShot): string {
    if (shot.dialogue.trim()) return shot.dialogue;
    return shot.voice_events.map((event) => event.text.trim()).join("\n").trim();
}

export function buildStoryboardDialoguePatch(shot: DialogueShot, value: string): DialoguePatch {
    const events = shot.voice_events;
    if (!value.trim()) {
        return { dialogue: "", dialogue_turns: [], voice_events: [] };
    }
    if (events.length === 1) {
        const voiceEvents = [{ ...events[0], text: value }];
        return {
            dialogue: shot.dialogue.trim() ? value : "",
            dialogue_turns: dialogueTurns(voiceEvents),
            voice_events: voiceEvents,
        };
    }
    const lines = value.replaceAll("\r\n", "\n").split("\n");
    if (events.length > 1 && lines.length === events.length) {
        const voiceEvents = events.map((event, index) => ({ ...event, text: lines[index] }));
        return {
            dialogue: shot.dialogue.trim() ? value : "",
            dialogue_turns: dialogueTurns(voiceEvents),
            voice_events: voiceEvents,
        };
    }
    return { dialogue: value, dialogue_turns: [], voice_events: [] };
}
