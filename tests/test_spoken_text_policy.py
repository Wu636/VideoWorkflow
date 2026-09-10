from __future__ import annotations

from src.video_workflow.domain import ProjectBrief, SpokenTextPolicy
from src.video_workflow.generators.llm import _normalize_storyboard_payload
from src.video_workflow.speech_budget import fit_voice_event_payloads


def test_verbatim_policy_keeps_over_budget_event_text_and_event_count() -> None:
    source = "这是完整对白，包含数字123、责任范围和结尾行动号召，任何一部分都需要进入音频。"
    events = [
        {
            "kind": "narration",
            "speaker_name": "旁白",
            "text": source,
            "start_seconds": 0.2,
            "end_seconds": 1.0,
        },
        {
            "kind": "inner_monologue",
            "speaker_name": "林晓",
            "text": "我必须把这句话也完整说完。",
            "start_seconds": 1.0,
            "end_seconds": 1.5,
        },
    ]

    fitted = fit_voice_event_payloads(
        events,
        1.5,
        characters_per_second=3.4,
        preserve_spoken_text=True,
    )

    assert [event["text"] for event in fitted] == [source, "我必须把这句话也完整说完。"]
    assert [event["kind"] for event in fitted] == ["narration", "inner_monologue"]


def test_storyboard_normalizer_does_not_compact_verbatim_voice_events() -> None:
    source = "长旁白第一句，长旁白第二句，长旁白第三句，必须完整保留。"
    payload = {
        "topic": "测试",
        "scenes": [
            {
                "duration": 4,
                "event": "场景",
                "voice_events": [
                    {
                        "kind": "narration",
                        "speaker_name": "旁白",
                        "text": source,
                        "start_seconds": 0.2,
                        "end_seconds": 1.0,
                    }
                ],
            }
        ],
    }

    normalized = _normalize_storyboard_payload(
        payload,
        include_dialogue=True,
        characters_per_second=3.4,
        preserve_spoken_text=True,
    )

    assert normalized["scenes"][0]["voice_events"][0]["text"] == source


def test_project_brief_defaults_to_adaptive_and_supports_verbatim() -> None:
    brief = ProjectBrief(title="测试", story="对白")
    assert brief.spoken_text_policy is SpokenTextPolicy.ADAPTIVE
    assert ProjectBrief(title="测试", story="对白", spoken_text_policy="verbatim").spoken_text_policy is SpokenTextPolicy.VERBATIM
