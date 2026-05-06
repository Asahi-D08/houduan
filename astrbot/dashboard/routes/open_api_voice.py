import mimetypes
import os
import uuid
from pathlib import Path

from quart import request, send_file

from astrbot.core import logger
from astrbot.core.core_lifecycle import AstrBotCoreLifecycle
from astrbot.core.provider.provider import STTProvider, TTSProvider
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

from .route import Response, Route, RouteContext


class OpenApiVoiceRoute(Route):
    def __init__(
        self,
        context: RouteContext,
        core_lifecycle: AstrBotCoreLifecycle,
    ) -> None:
        super().__init__(context)
        self.core_lifecycle = core_lifecycle
        self.routes = {
            "/v1/voice/capabilities": ("GET", self.capabilities),
            "/v1/voice/speech": ("POST", self.speech),
            "/v1/voice/transcriptions": ("POST", self.transcriptions),
        }
        self.register_routes()

    def _get_stt_provider(self) -> STTProvider | None:
        context = self.core_lifecycle.plugin_manager.context
        provider_manager = context.provider_manager
        provider = context.get_using_stt_provider()
        if provider:
            return provider

        if not self.config["provider_stt_settings"].get("enable"):
            return None

        provider_id = self.config["provider_stt_settings"].get("provider_id")
        if provider_id:
            provider = provider_manager.inst_map.get(provider_id)
            if isinstance(provider, STTProvider):
                return provider

        provider = provider_manager.curr_stt_provider_inst
        if isinstance(provider, STTProvider):
            return provider
        return (
            provider_manager.stt_provider_insts[0]
            if provider_manager.stt_provider_insts
            else None
        )

    def _get_tts_provider(self) -> TTSProvider | None:
        context = self.core_lifecycle.plugin_manager.context
        provider_manager = context.provider_manager
        provider = context.get_using_tts_provider()
        if provider:
            return provider

        if not self.config["provider_tts_settings"].get("enable"):
            return None

        provider_id = self.config["provider_tts_settings"].get("provider_id")
        if provider_id:
            provider = provider_manager.inst_map.get(provider_id)
            if isinstance(provider, TTSProvider):
                return provider

        provider = provider_manager.curr_tts_provider_inst
        if isinstance(provider, TTSProvider):
            return provider
        return (
            provider_manager.tts_provider_insts[0]
            if provider_manager.tts_provider_insts
            else None
        )

    @staticmethod
    def _serialize_provider(provider: STTProvider | TTSProvider | None) -> dict | None:
        if not provider:
            return None
        try:
            meta = provider.meta()
            return {
                "id": meta.id,
                "type": meta.type,
                "model": meta.model,
            }
        except Exception:
            return {
                "id": provider.provider_config.get("id", ""),
                "type": provider.provider_config.get("type", ""),
                "model": provider.get_model(),
            }

    async def capabilities(self):
        stt_provider = self._get_stt_provider()
        tts_provider = self._get_tts_provider()
        return (
            Response()
            .ok(
                data={
                    "can_transcribe": stt_provider is not None,
                    "can_speak": tts_provider is not None,
                    "stt_provider": self._serialize_provider(stt_provider),
                    "tts_provider": self._serialize_provider(tts_provider),
                }
            )
            .__dict__
        )

    async def speech(self):
        post_data = await request.get_json(silent=True) or {}
        text = str(post_data.get("text", "")).strip()
        if not text:
            return Response().error("Missing key: text").__dict__

        tts_provider = self._get_tts_provider()
        if not tts_provider:
            return (
                Response().error("Text-to-speech provider is not configured").__dict__
            )

        try:
            audio_path = await tts_provider.get_audio(text)
        except Exception as e:
            logger.error("Open API voice speech failed: %s", e, exc_info=True)
            return Response().error(f"Text-to-speech failed: {e}").__dict__

        if not audio_path or not os.path.exists(audio_path):
            return Response().error("Text-to-speech returned no audio file").__dict__

        suffix = Path(audio_path).suffix.lower()
        mimetype = (
            "audio/wav" if suffix == ".wav" else mimetypes.guess_type(audio_path)[0]
        )
        if not mimetype:
            mimetype = "application/octet-stream"
        return await send_file(audio_path, mimetype=mimetype)

    async def transcriptions(self):
        files = await request.files
        if "file" not in files:
            return Response().error("Missing key: file").__dict__

        stt_provider = self._get_stt_provider()
        if not stt_provider:
            return (
                Response().error("Speech-to-text provider is not configured").__dict__
            )

        file = files["file"]
        suffix = Path(file.filename or "").suffix or ".wav"
        temp_dir = Path(get_astrbot_temp_path())
        temp_dir.mkdir(parents=True, exist_ok=True)
        audio_path = temp_dir / f"open_api_voice_{uuid.uuid4().hex}{suffix}"

        try:
            await file.save(str(audio_path))
            text = await stt_provider.get_text(str(audio_path))
        except Exception as e:
            logger.error("Open API voice transcription failed: %s", e, exc_info=True)
            return Response().error(f"Speech-to-text failed: {e}").__dict__
        finally:
            audio_path.unlink(missing_ok=True)

        return Response().ok(data={"text": text or ""}).__dict__
