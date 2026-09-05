from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from src.video_workflow.config import settings
from src.video_workflow.domain import CharacterProfile, Project, Shot
from src.video_workflow.services.projects import ProjectService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TTSVoiceProfile:
    voice: str
    rate_percent: int
    pitch_hz: int
    volume_percent: int
    voice_anchor: str
    selection_reason: str

    @property
    def rate(self) -> str:
        return f"{self.rate_percent:+d}%"

    @property
    def pitch(self) -> str:
        return f"{self.pitch_hz:+d}Hz"

    @property
    def volume(self) -> str:
        return f"{self.volume_percent:+d}%"


class DialogueAudioService:
    """Generate a clean dialogue stem independently from MiniMax H3 audio."""

    @staticmethod
    def speaker(project: Project, shot: Shot) -> CharacterProfile | None:
        return next((item for item in project.characters if item.id == shot.dialogue_speaker_id), None)

    @classmethod
    def voice_for(cls, project: Project, shot: Shot) -> str:
        return cls.profile_for(project, shot).voice

    @classmethod
    def profile_for(cls, project: Project, shot: Shot) -> TTSVoiceProfile:
        speaker = cls.speaker(project, shot)
        voice_description = (speaker.voice_description if speaker else "").strip()
        visual_description = (speaker.description if speaker else "").strip()
        combined = f"{voice_description} {visual_description}".lower()
        external_kinds = {
            event.kind for event in shot.voice_events if event.kind != "character"
        }
        age_match = re.search(r"(\d{1,2})\s*岁", visual_description)
        age = int(age_match.group(1)) if age_match else None
        is_male = bool(re.search(r"男性|男声|男音|male|男主|男生", combined))
        is_female = bool(re.search(r"女性|女声|女音|female|女主|女生", combined))
        forceful = bool(re.search(r"毛躁|急躁|不耐烦|语调冲|激动|愤怒|强硬|高亢|有力|音量偏大", voice_description))
        youthful_lively = bool(re.search(r"清亮|偏脆|活泼|明快|急切|焦急", voice_description))
        mature = bool(age is not None and age >= 35) or bool(re.search(r"中年|成熟|资深|威严", combined))

        if speaker and speaker.tts_voice.strip():
            voice = speaker.tts_voice.strip()
            selection_reason = "character_explicit_voice_id"
        elif "system_vo" in external_kinds and speaker is None:
            voice = settings.TTS_DEFAULT_MATURE_MALE_VOICE
            selection_reason = "system_voice_professional_reliable"
        elif is_male and forceful and not mature:
            # Edge labels Yunjian as a passionate male voice, which is a much
            # closer base for impatient/forceful young men than Yunxi's
            # lively-sunshine personality.
            voice = "zh-CN-YunjianNeural"
            selection_reason = "young_forceful_male"
        elif is_male and mature:
            voice = settings.TTS_DEFAULT_MATURE_MALE_VOICE
            selection_reason = "mature_male"
        elif is_male:
            voice = settings.TTS_DEFAULT_MALE_VOICE
            selection_reason = "young_or_neutral_male"
        elif is_female and mature:
            voice = settings.TTS_DEFAULT_MATURE_FEMALE_VOICE
            selection_reason = "mature_female"
        elif is_female and youthful_lively:
            voice = "zh-CN-XiaoyiNeural"
            selection_reason = "young_lively_female"
        else:
            voice = settings.TTS_DEFAULT_FEMALE_VOICE
            selection_reason = "neutral_or_narration"

        semantic_rate = 0
        if re.search(r"语速(?:很快|快)|节奏(?:很快|快)", voice_description):
            semantic_rate = 6
        elif re.search(r"语速偏快|节奏偏快", voice_description):
            semantic_rate = 3
        elif re.search(r"稍快|略快", voice_description):
            semantic_rate = 2
        elif re.search(r"语速(?:很慢|慢)|节奏(?:很慢|慢)", voice_description):
            semantic_rate = -8
        elif re.search(r"语速偏慢|节奏偏慢|语速偏缓|节奏偏缓", voice_description):
            semantic_rate = -5
        rate_percent = max(-50, min(100, int(shot.dialogue_rate_percent) + semantic_rate))

        pitch_hz = 0
        if re.search(r"低沉|浑厚|沙哑|冷调", voice_description):
            pitch_hz = -10 if mature else -6
        elif is_male and forceful:
            pitch_hz = -4
        elif re.search(r"清亮|清冽|偏脆|明亮", voice_description):
            pitch_hz = 4
        volume_percent = 3 if forceful or re.search(r"急切|焦急|音量偏大", voice_description) else 0
        voice_anchor = (
            ProjectService.character_voice_anchor(speaker)
            if speaker
            else "冷静清晰的中性电子播报音，音高稳定、吐字精准、无真人口语感"
            if "system_vo" in external_kinds
            else "清晰自然的画外旁白，音色稳定、吐字从容"
        )
        return TTSVoiceProfile(
            voice=voice,
            rate_percent=rate_percent,
            pitch_hz=pitch_hz,
            volume_percent=volume_percent,
            voice_anchor=voice_anchor,
            selection_reason=selection_reason,
        )

    @classmethod
    async def synthesize(cls, project: Project, shot: Shot, destination: Path) -> dict[str, object]:
        text = ProjectService.spoken_dialogue_text(shot.dialogue)
        if not text:
            return {"generated": False, "reason": "no_dialogue", "path": None}
        if settings.TTS_PROVIDER == "disabled":
            return {"generated": False, "reason": "provider_disabled", "path": None}
        if settings.TTS_PROVIDER != "edge":
            raise ValueError(f"Unsupported TTS provider: {settings.TTS_PROVIDER}")

        try:
            import edge_tts
        except ImportError as exc:  # pragma: no cover - covered by deployment smoke test
            raise RuntimeError("edge-tts dependency is not installed") from exc

        destination.parent.mkdir(parents=True, exist_ok=True)
        profile = cls.profile_for(project, shot)
        communicate = edge_tts.Communicate(
            text=text,
            voice=profile.voice,
            rate=profile.rate,
            pitch=profile.pitch,
            volume=profile.volume,
        )
        await communicate.save(str(destination))
        if not destination.exists() or destination.stat().st_size == 0:
            raise RuntimeError("TTS returned an empty audio file")
        speaker = cls.speaker(project, shot)
        return {
            "generated": True,
            "provider": "edge",
            "voice": profile.voice,
            "speaker": speaker.name if speaker else "旁白",
            "voice_anchor": profile.voice_anchor,
            "voice_selection_reason": profile.selection_reason,
            "rate": profile.rate,
            "pitch": profile.pitch,
            "volume": profile.volume,
            "characters": len(text),
            "path": str(destination),
        }
