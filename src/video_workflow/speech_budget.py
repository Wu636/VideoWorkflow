from __future__ import annotations

import math
import re
from typing import Any


# Normal Mandarin TTS is usually intelligible around four visible Chinese
# characters per second.  Reserving roughly 15% of the shot for breath,
# reactions and ambience keeps H3 from producing machine-gun delivery.
SPEECH_CHARACTERS_PER_SECOND = 3.4
SPEECH_WINDOW_CHARACTERS_PER_SECOND = 3.9
SPEECH_EVENT_GAP_SECONDS = 0.15


def spoken_character_count(value: str) -> int:
    """Count the characters that materially consume spoken time."""
    text = re.sub(r"[\s【】\[\]<>]", "", value or "")
    return len(text)


def speech_budget_for_duration(duration_seconds: float) -> int:
    """Return the total comfortable spoken-character budget for one shot."""
    duration = max(0.25, float(duration_seconds))
    return max(4, int(math.floor(duration * SPEECH_CHARACTERS_PER_SECOND)))


def _clean_spoken_text(value: str) -> str:
    text = re.sub(r"\s+", "", str(value or "").strip())
    text = re.sub(r"^[【\[](.+)[】\]]$", r"\1", text)
    return text.strip("【】[]“”\"' ")


def _shorten_common_phrases(value: str) -> str:
    replacements = (
        ("副本任务：", "任务："),
        ("副本等级：", "等级："),
        ("核心铁律：", "铁律："),
        ("失败惩罚：", "惩罚："),
        ("高危隐患预警：", "预警："),
        ("必须全方位", "须全面"),
        ("必须全面", "须全面"),
        ("欢迎试炼者", "试炼者"),
        ("登入高危电力副本", "进入高危电力副本"),
        ("双人配合登杆", "双人登杆"),
        ("一人上杆并完成转位", "一人作业"),
        ("一人专职地面监护", "一人地面监护"),
        ("任何细节违规", "违规"),
        ("核验安全装备", "检查装备"),
        ("即刻抹杀", "即抹杀"),
        ("登杆前须全面检查杆身、杆基、拉线", "登杆前全面检查"),
    )
    result = value
    for source, target in replacements:
        result = result.replace(source, target)
    result = re.sub(
        r"试炼者([A-Za-z0-9_-]+)，进入高危电力副本：[^，。！？；]+",
        r"试炼者\1进入高危电力副本",
        result,
    )
    result = re.sub(r"([，。！？；：])\1+", r"\1", result)
    return result


def compact_spoken_text(value: str, limit: int) -> str:
    """Shorten an over-budget utterance into a complete, speakable phrase.

    This is a last-resort guard for providers that ignore the prompt budget.
    Full punctuation-delimited clauses are preferred; a clipped fragment is
    used only when a single clause is itself longer than the available window.
    """
    limit = max(2, int(limit))
    text = _clean_spoken_text(value)
    if spoken_character_count(text) <= limit:
        return text
    text = _shorten_common_phrases(text)
    if spoken_character_count(text) <= limit:
        return text

    prefix = ""
    body = text
    prefix_match = re.match(r"^([^：:]{1,8})[：:]", text)
    if prefix_match:
        prefix = f"{prefix_match.group(1)}："
        body = text[prefix_match.end():]
        if spoken_character_count(prefix) >= limit - 2:
            prefix = ""

    clauses = [item for item in re.findall(r"[^，。！？；]+[，。！？；]?", body) if item]
    result = prefix
    for clause in clauses:
        candidate = f"{result}{clause}"
        if spoken_character_count(candidate) <= limit:
            result = candidate
            continue
        remaining = limit - spoken_character_count(result)
        if remaining > 1 and not result.rstrip("，。！？；："):
            result += clause[:remaining]
        elif remaining > 2 and result == prefix:
            clipped = clause[:remaining].rstrip("，。！？；：、的与和并在从为把将")
            result += clipped or clause[:remaining]
        break

    result = result.rstrip("，；：、的与和并在从为把将")
    if result and result[-1] not in "。！？!?":
        if spoken_character_count(result) < limit:
            result += "。"
        else:
            result = result[:-1].rstrip("，；：、的与和并在从为把将") + "。"
    return result or text[:limit]


def _allocate_event_budgets(lengths: list[int], total_budget: int) -> list[int]:
    if not lengths:
        return []
    if sum(lengths) <= total_budget:
        return lengths
    count = len(lengths)
    base = min(6, max(2, total_budget // max(1, count * 2)))
    allocations = [min(length, base) for length in lengths]
    remaining = max(0, total_budget - sum(allocations))
    while remaining:
        candidates = [index for index, length in enumerate(lengths) if allocations[index] < length]
        if not candidates:
            break
        weights = [math.sqrt(max(1, lengths[index] - allocations[index])) for index in candidates]
        weight_total = sum(weights)
        progressed = False
        for index, weight in zip(candidates, weights, strict=False):
            share = max(1, int(round(remaining * weight / weight_total)))
            addition = min(share, lengths[index] - allocations[index], remaining)
            if addition:
                allocations[index] += addition
                remaining -= addition
                progressed = True
            if not remaining:
                break
        if not progressed:
            break
    return allocations


def fit_voice_event_payloads(
    events: list[dict[str, Any]],
    duration_seconds: float,
) -> list[dict[str, Any]]:
    """Fit timed voice events to a comfortable shot-level speech budget."""
    duration = max(0.25, float(duration_seconds))
    cleaned: list[dict[str, Any]] = []
    for event in events:
        text = _clean_spoken_text(str(event.get("text") or ""))
        if not text:
            continue
        cleaned.append({**event, "text": text})
    if not cleaned:
        return []

    while len(cleaned) > 4:
        merge_candidates = [
            index
            for index in range(len(cleaned) - 1)
            if (
                cleaned[index].get("kind"),
                cleaned[index].get("speaker_id") or cleaned[index].get("speaker_name"),
            )
            == (
                cleaned[index + 1].get("kind"),
                cleaned[index + 1].get("speaker_id") or cleaned[index + 1].get("speaker_name"),
            )
        ]
        if not merge_candidates:
            break
        index = min(
            merge_candidates,
            key=lambda item: spoken_character_count(str(cleaned[item]["text"]))
            + spoken_character_count(str(cleaned[item + 1]["text"])),
        )
        left = cleaned[index]
        right = cleaned[index + 1]
        separator = "" if str(left["text"]).endswith(("。", "！", "？", ".", "!", "?")) else "。"
        try:
            merged_end = max(float(left.get("end_seconds", 0.0)), float(right.get("end_seconds", 0.0)))
        except (TypeError, ValueError):
            merged_end = duration
        cleaned[index:index + 2] = [{
            **left,
            "text": f"{left['text']}{separator}{right['text']}",
            "end_seconds": merged_end,
        }]

    total_budget = speech_budget_for_duration(duration)
    source_texts = [str(event["text"]) for event in cleaned]
    lengths = [spoken_character_count(text) for text in source_texts]
    allocations = _allocate_event_budgets(lengths, total_budget)
    for event, allocation in zip(cleaned, allocations, strict=False):
        event["text"] = compact_spoken_text(str(event["text"]), allocation)

    # Clause-safe compaction can leave a few characters unused. Reassign that
    # space to still-truncated events so important second clauses survive.
    for _ in range(3):
        used = [spoken_character_count(str(event["text"])) for event in cleaned]
        spare = total_budget - sum(used)
        candidates = [index for index, length in enumerate(lengths) if used[index] < length]
        candidates.sort(
            key=lambda index: bool(
                re.search(r"违规|失败|抹杀|死亡|禁止|警告|危险|通关", source_texts[index])
            ),
            reverse=True,
        )
        if spare <= 0 or not candidates:
            break
        progressed = False
        for index in candidates:
            expanded = compact_spoken_text(
                source_texts[index],
                min(lengths[index], used[index] + spare),
            )
            expanded_size = spoken_character_count(expanded)
            if expanded_size > used[index] and expanded_size - used[index] <= spare:
                cleaned[index]["text"] = expanded
                spare -= expanded_size - used[index]
                progressed = True
            if spare <= 0:
                break
        if not progressed:
            break

    # Keep voice events sequential and make every declared window large enough
    # for normal-speed delivery.  Existing later cues remain preferred when
    # there is room, but overlapping speech is removed.
    needed_windows = [
        max(
            0.7,
            spoken_character_count(str(event["text"])) / SPEECH_WINDOW_CHARACTERS_PER_SECOND + 0.1,
        )
        for event in cleaned
    ]
    cursor = 0.2
    latest_end = max(0.25, duration - 0.2)
    for index, (event, needed) in enumerate(zip(cleaned, needed_windows, strict=False)):
        try:
            requested_start = float(event.get("start_seconds", 0.35))
        except (TypeError, ValueError):
            requested_start = 0.35
        remaining_events = len(cleaned) - index - 1
        remaining_windows = sum(needed_windows[index + 1:])
        reserved_tail = remaining_windows + remaining_events * SPEECH_EVENT_GAP_SECONDS
        latest_start = max(cursor, latest_end - needed - reserved_tail)
        start = min(max(cursor, requested_start), latest_start)
        end = min(latest_end - reserved_tail, start + needed)
        if end <= start:
            end = min(latest_end, start + 0.5)
        event["start_seconds"] = round(max(0.0, start), 2)
        event["end_seconds"] = round(max(start + 0.05, end), 2)
        if str(event.get("kind") or "") != "character":
            event["lip_sync"] = False
        cursor = end + SPEECH_EVENT_GAP_SECONDS
    return cleaned
