from __future__ import annotations

import logging
import re
from pathlib import Path

from src.video_workflow.config import settings
from src.video_workflow.domain import CharacterProfile, Project, Shot
from src.video_workflow.services.projects import ProjectService

logger = logging.getLogger(__name__)


class DialogueAudioService:
    """Generate a clean dialogue stem independently from MiniMax H3 audio."""

    @staticmethod
    def speaker(project: Project, shot: Shot) -> CharacterProfile | None:
        return next((item for item in project.characters if item.id == shot.dialogue_speaker_id), None)

    @classmethod
    def voice_for(cls, project: Project, shot: Shot) -> str:
        speaker = cls.speaker(project, shot)
        if speaker and speaker.tts_voice.strip():
            return speaker.tts_voice.strip()
        voice_description = (speaker.voice_description if speaker else "").lower()
        if re.search(r"男|male|低沉|浑厚|青年男|中年男", voice_description):
            if not re.search(r"青年|年轻", voice_description) and re.search(r"中年|成熟|沉稳|威严|低沉|浑厚", voice_description):
                return settings.TTS_DEFAULT_MATURE_MALE_VOICE
            return settings.TTS_DEFAULT_MALE_VOICE
        if re.search(r"中年|成熟|严肃|庄重", voice_description):
            return settings.TTS_DEFAULT_MATURE_FEMALE_VOICE
        return settings.TTS_DEFAULT_FEMALE_VOICE

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
        voice = cls.voice_for(project, shot)
        rate = f"{shot.dialogue_rate_percent:+d}%"
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
        await communicate.save(str(destination))
        if not destination.exists() or destination.stat().st_size == 0:
            raise RuntimeError("TTS returned an empty audio file")
        speaker = cls.speaker(project, shot)
        return {
            "generated": True,
            "provider": "edge",
            "voice": voice,
            "speaker": speaker.name if speaker else "旁白",
            "rate": rate,
            "characters": len(text),
            "path": str(destination),
        }
