type KeyframeShot = { id: string; image_status: string };

// H3 prompt generation and image generation use separate resources. Only the
// same shot's image work (or a conflicting whole-shot save) locks its keyframe.
export function isKeyframeBusy(shot: KeyframeShot, busy: ReadonlySet<string>, pending: ReadonlySet<string>): boolean {
    return shot.image_status === "processing" || pending.has(shot.id)
        || ["keyframe-prompt", "upload-keyframe", "h3", "seedance", "shot", "ai-revise", "delete", "render-h3", "render-seedance"]
            .some((prefix) => busy.has(`${prefix}-${shot.id}`))
        || Array.from(busy).some((key) => key === "plan" || key === "storyboard" || key.startsWith("preset-"));
}

export function isH3PromptBlocked(busy: ReadonlySet<string>): boolean {
    // Keep existing exclusions for other writers, but allow independent image
    // edits, uploads and exports regardless of which task started first.
    return Array.from(busy).some((key) => !(
        key.startsWith("keyframe-prompt-") || key.startsWith("upload-keyframe-")
        || key === "export-keyframes" || key === "preflight"
    ));
}
