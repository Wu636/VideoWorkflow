"use client";

/* eslint-disable @next/next/no-img-element */
import { useEffect, useMemo, useRef, useState } from "react";
import { AudioLines, Triangle } from "lucide-react";

import type { Asset } from "@/types";

export type PromptReferenceMaterial = Pick<Asset, "id" | "type" | "name"> & { label: string };

type PromptReferenceEditorProps = {
    value: string;
    materials: PromptReferenceMaterial[];
    getAssetUrl: (asset: Pick<Asset, "id">) => string;
    onChange: (value: string) => void;
    onReferenceClick?: (material: PromptReferenceMaterial) => void;
    placeholder?: string;
    className?: string;
};

type PromptToken = {
    raw: string;
    label: string;
};

const TOKEN_PATTERN = /@(图片|视频|音频)\s*(\d+)|<(Picture|Video|Audio)\s+(\d+)>/g;

function tokenFromMatch(match: RegExpExecArray): PromptToken {
    if (match[1]) {
        const label = `${match[1]}${match[2]}`;
        return { raw: match[0], label };
    }
    const kind = match[3] === "Picture" ? "图片" : match[3] === "Video" ? "视频" : "音频";
    const label = `${kind}${match[4]}`;
    return { raw: match[0], label };
}

function materialSignature(materials: PromptReferenceMaterial[]): string {
    return materials.map((item) => `${item.id}:${item.label}:${item.name}:${item.type}`).join("|");
}

function isBlockElement(element: Element): boolean {
    return ["DIV", "P", "LI", "PRE", "BLOCKQUOTE"].includes(element.tagName);
}

function serializeNode(node: Node): string {
    if (node.nodeType === Node.TEXT_NODE) return node.textContent || "";
    if (node.nodeType !== Node.ELEMENT_NODE) return "";
    const element = node as HTMLElement;
    const referenceToken = element.dataset.referenceToken;
    if (referenceToken) return referenceToken;
    if (element.tagName === "BR") return "\n";
    let text = Array.from(element.childNodes).map(serializeNode).join("");
    if (isBlockElement(element) && !text.endsWith("\n")) text += "\n";
    return text;
}

function serializeEditor(editor: HTMLElement): string {
    return Array.from(editor.childNodes)
        .map(serializeNode)
        .join("")
        .replace(/\n+$/, "");
}

function appendText(parent: Node, text: string): void {
    if (text) parent.appendChild(document.createTextNode(text));
}

function removePromptReferenceHovers(): void {
    document.querySelectorAll<HTMLElement>('[data-prompt-reference-hover="true"]').forEach((element) => element.remove());
}

function positionPromptReferenceHover(hover: HTMLElement, anchor: HTMLElement): void {
    const anchorRect = anchor.getBoundingClientRect();
    const margin = 8;
    const left = Math.min(Math.max(margin, anchorRect.left), window.innerWidth - hover.offsetWidth - margin);
    const above = anchorRect.top - hover.offsetHeight - margin;
    const top = above >= margin
        ? above
        : Math.min(window.innerHeight - hover.offsetHeight - margin, anchorRect.bottom + margin);
    hover.style.left = `${Math.max(margin, left)}px`;
    hover.style.top = `${Math.max(margin, top)}px`;
}

function renderPrompt(
    editor: HTMLElement,
    value: string,
    materials: PromptReferenceMaterial[],
    getAssetUrl: (asset: Pick<Asset, "id">) => string,
    onReferenceClickRef: { current?: (material: PromptReferenceMaterial) => void },
): void {
    removePromptReferenceHovers();
    editor.replaceChildren();
    const materialMap = new Map(materials.map((item) => [item.label, item]));
    const pattern = new RegExp(TOKEN_PATTERN.source, "g");
    let cursor = 0;
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(value)) !== null) {
        appendText(editor, value.slice(cursor, match.index));
        const token = tokenFromMatch(match);
        const material = materialMap.get(token.label);
        if (!material) {
            appendText(editor, token.raw);
        } else {
            const chip = document.createElement("span");
            chip.contentEditable = "false";
            chip.dataset.referenceToken = token.raw;
            chip.className = "group relative mx-0.5 inline-flex select-none items-center gap-1 rounded-md border border-cyan-300/25 bg-cyan-300/[.10] px-1 py-0.5 align-baseline text-[.88em] text-cyan-50/90";
            chip.title = `${token.label} · ${material.name}`;
            chip.setAttribute("role", "button");
            chip.tabIndex = 0;
            const openReference = () => onReferenceClickRef.current
                ? onReferenceClickRef.current(material)
                : window.open(getAssetUrl(material), "_blank", "noopener,noreferrer");
            chip.addEventListener("click", openReference);
            chip.addEventListener("keydown", (event) => {
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    openReference();
                }
            });

            if (material.type === "image") {
                const image = document.createElement("img");
                image.src = getAssetUrl(material);
                image.alt = material.name;
                image.className = "h-5 w-7 rounded object-cover";
                chip.appendChild(image);
            } else {
                const icon = document.createElement("span");
                icon.className = "inline-flex h-5 w-5 items-center justify-center rounded bg-black/25 text-cyan-100/70";
                icon.textContent = material.type === "video" ? "▶" : "♫";
                chip.appendChild(icon);
            }
            const label = document.createElement("span");
            label.textContent = token.label;
            chip.appendChild(label);
            const hover = document.createElement("span");
            hover.dataset.promptReferenceHover = "true";
            hover.className = "pointer-events-none fixed z-[200] hidden w-48 overflow-hidden rounded-lg border border-white/15 bg-[#11151e] p-1.5 text-left shadow-2xl";
            if (material.type === "image") {
                const largeImage = document.createElement("img");
                largeImage.src = getAssetUrl(material);
                largeImage.alt = "";
                largeImage.className = "max-h-36 w-full rounded object-contain";
                hover.appendChild(largeImage);
            } else if (material.type === "video") {
                const video = document.createElement("video");
                video.src = getAssetUrl(material);
                video.muted = true;
                video.preload = "metadata";
                video.className = "max-h-36 w-full rounded bg-black object-contain";
                hover.appendChild(video);
            }
            const materialName = document.createElement("span");
            materialName.textContent = material.name;
            materialName.className = "mt-1 block truncate px-1 text-[10px] text-white/65";
            hover.appendChild(materialName);
            const showHover = () => {
                document.body.appendChild(hover);
                hover.classList.remove("hidden");
                positionPromptReferenceHover(hover, chip);
            };
            const hideHover = () => {
                hover.remove();
            };
            chip.addEventListener("mouseenter", showHover);
            chip.addEventListener("mouseleave", hideHover);
            chip.addEventListener("focus", showHover);
            chip.addEventListener("blur", hideHover);
            editor.appendChild(chip);
        }
        cursor = match.index + match[0].length;
    }
    appendText(editor, value.slice(cursor));
}

export function PromptMaterialThumbnail({ material, getAssetUrl, onClick }: { material: PromptReferenceMaterial; getAssetUrl: (asset: Pick<Asset, "id">) => string; onClick?: () => void }) {
    const [showPreview, setShowPreview] = useState(false);
    const url = getAssetUrl(material);
    const content = material.type === "image"
        ? <img className="h-7 w-10 rounded object-cover" src={url} alt={material.name} />
        : material.type === "video"
            ? <span className="relative inline-flex h-7 w-10 items-center justify-center overflow-hidden rounded bg-black/35 text-cyan-100/75"><video className="h-full w-full object-cover opacity-70" src={url} muted preload="metadata" /><span className="absolute inset-0 flex items-center justify-center bg-black/25 text-[10px]">▶</span></span>
            : <span className="inline-flex h-7 w-10 items-center justify-center rounded bg-cyan-300/10 text-cyan-100/70"><AudioLines size={14} /></span>;
    const preview = material.type === "image"
        ? <span className="pointer-events-none absolute bottom-full left-0 z-50 mb-2 hidden w-48 overflow-hidden rounded-lg border border-white/15 bg-[#11151e] p-1.5 shadow-2xl group-hover:block"><img className="max-h-36 w-full rounded object-contain" src={url} alt="" /><span className="mt-1 block truncate px-1 text-[10px] text-white/65">{material.name}</span></span>
        : material.type === "video"
            ? <span className="pointer-events-none absolute bottom-full left-0 z-50 mb-2 hidden w-56 overflow-hidden rounded-lg border border-white/15 bg-[#11151e] p-1.5 shadow-2xl group-hover:block"><video className="max-h-36 w-full rounded bg-black object-contain" src={url} muted preload="metadata" /><span className="mt-1 block truncate px-1 text-[10px] text-white/65">{material.name}</span></span>
            : <span className="pointer-events-none absolute bottom-full left-0 z-50 mb-2 hidden whitespace-nowrap rounded-lg border border-white/15 bg-[#11151e] px-2.5 py-2 text-[10px] text-white/65 shadow-2xl group-hover:block">{material.name}</span>;
    const body = <>{content}{preview}</>;
    const open = onClick ?? (() => setShowPreview(true));
    return <>
        <button type="button" className="group relative inline-flex shrink-0 cursor-zoom-in rounded-md outline-none focus-visible:ring-1 focus-visible:ring-cyan-200/80" title={`预览 ${material.label} · ${material.name}`} onClick={open}>{body}</button>
        {showPreview && <div className="studio-modal z-[130]" role="dialog" aria-modal="true" aria-label={`${material.name} 素材预览`} onMouseDown={() => setShowPreview(false)}>
            <section className="flex max-h-[94vh] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-white/12 bg-[#0d1017] shadow-2xl" onMouseDown={(event) => event.stopPropagation()}>
                <div className="flex items-center gap-3 border-b border-white/8 px-4 py-3"><div className="min-w-0 flex-1"><strong className="block truncate text-sm">{material.name}</strong><span className="text-xs text-white/30">{material.label} · 参考素材预览</span></div><button type="button" className="rounded-lg px-3 py-1.5 text-xs text-white/50 hover:bg-white/8 hover:text-white" onClick={() => setShowPreview(false)}>关闭</button></div>
                <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto bg-black/60 p-4">
                    {material.type === "image" ? <img className="max-h-[76vh] max-w-full object-contain" src={url} alt={material.name} /> : material.type === "video" ? <video className="max-h-[76vh] w-full object-contain" src={url} controls autoPlay playsInline preload="metadata" /> : <audio className="w-full max-w-xl" src={url} controls autoPlay />}
                </div>
            </section>
        </div>}
    </>;
}

function serializeSelection(selection: Selection, editor: HTMLElement): string {
    if (!selection.rangeCount || !editor.contains(selection.anchorNode)) return "";
    const range = selection.getRangeAt(0);
    const fragment = range.cloneContents();
    return Array.from(fragment.childNodes).map(serializeNode).join("");
}

export default function PromptReferenceEditor({ value, materials, getAssetUrl, onChange, onReferenceClick, placeholder = "输入 Prompt…", className = "" }: PromptReferenceEditorProps) {
    const editorRef = useRef<HTMLDivElement>(null);
    const renderedSignature = useRef("");
    const referenceClickRef = useRef(onReferenceClick);
    const materialsRef = useRef(materials);
    const getAssetUrlRef = useRef(getAssetUrl);
    const signature = useMemo(() => materialSignature(materials), [materials]);

    useEffect(() => {
        referenceClickRef.current = onReferenceClick;
    }, [onReferenceClick]);

    useEffect(() => {
        materialsRef.current = materials;
        getAssetUrlRef.current = getAssetUrl;
    }, [getAssetUrl, materials]);

    useEffect(() => {
        const editor = editorRef.current;
        if (!editor) return;
        const current = serializeEditor(editor);
        if (current !== value || renderedSignature.current !== signature) {
            renderPrompt(editor, value, materialsRef.current, getAssetUrlRef.current, referenceClickRef);
            renderedSignature.current = signature;
        }
    }, [signature, value]);

    const emitChange = () => {
        const editor = editorRef.current;
        if (!editor) return;
        const next = serializeEditor(editor);
        onChange(next);
    };

    const pastePlainText = (event: React.ClipboardEvent<HTMLDivElement>) => {
        event.preventDefault();
        const text = event.clipboardData.getData("text/plain");
        const selection = window.getSelection();
        if (!selection || !selection.rangeCount || !editorRef.current) return;
        const range = selection.getRangeAt(0);
        range.deleteContents();
        const node = document.createTextNode(text);
        range.insertNode(node);
        range.setStartAfter(node);
        range.collapse(true);
        selection.removeAllRanges();
        selection.addRange(range);
        emitChange();
    };

    const copyPlainText = (event: React.ClipboardEvent<HTMLDivElement>) => {
        const editor = editorRef.current;
        const selection = window.getSelection();
        if (!editor || !selection || !selection.rangeCount || !editor.contains(selection.anchorNode)) return;
        const text = serializeSelection(selection, editor);
        if (!text) return;
        event.preventDefault();
        event.clipboardData.setData("text/plain", text);
    };

    const normalizeTokens = () => {
        const editor = editorRef.current;
        if (!editor) return;
        const current = serializeEditor(editor);
        renderPrompt(editor, current, materialsRef.current, getAssetUrlRef.current, referenceClickRef);
        renderedSignature.current = signature;
    };

    return <div className={`relative ${className}`}>
        <div
            ref={editorRef}
            contentEditable
            role="textbox"
            aria-multiline="true"
            suppressContentEditableWarning
            spellCheck={false}
            className="studio-input min-h-64 max-h-[34rem] w-full overflow-y-auto whitespace-pre-wrap break-words rounded-lg border border-white/10 bg-black/25 px-3 py-2.5 text-sm text-white outline-none focus:border-cyan-300/45"
            onInput={emitChange}
            onBlur={normalizeTokens}
            onPaste={pastePlainText}
            onCopy={copyPlainText}
        />
        {!value && <span className="pointer-events-none absolute left-3 top-3 text-sm text-white/25">{placeholder}</span>}
        <div className="mt-1.5 flex items-center gap-1 text-[11px] text-white/30">
            <Triangle size={10} className="rotate-180 text-cyan-300/55" />
            素材引用会显示缩略图；保存和提交时仍使用原始 Prompt 标签
        </div>
    </div>;
}
