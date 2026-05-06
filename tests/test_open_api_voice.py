import asyncio
import os
import uuid
from io import BytesIO
from types import SimpleNamespace

import pytest
import pytest_asyncio
from quart import Quart
from werkzeug.datastructures import FileStorage

from astrbot.core import LogBroker
from astrbot.core.core_lifecycle import AstrBotCoreLifecycle
from astrbot.core.db.sqlite import SQLiteDatabase
from astrbot.core.provider.provider import STTProvider, TTSProvider
from astrbot.dashboard.server import AstrBotDashboard


class FakeSTTProvider(STTProvider):
    def __init__(self):
        super().__init__(
            {"id": "fake-stt", "type": "fake_stt"},
            {},
        )
        self.set_model("fake-stt-model")
        self.audio_paths: list[str] = []

    async def get_text(self, audio_url: str) -> str:
        self.audio_paths.append(audio_url)
        return "识别文本"

    def meta(self):
        return SimpleNamespace(
            id="fake-stt",
            type="fake_stt",
            model=self.get_model(),
        )


class FakeTTSProvider(TTSProvider):
    def __init__(self, audio_path: str):
        super().__init__(
            {"id": "fake-tts", "type": "fake_tts"},
            {},
        )
        self.audio_path = audio_path
        self.texts: list[str] = []
        self.set_model("fake-tts-model")

    async def get_audio(self, text: str) -> str:
        self.texts.append(text)
        return self.audio_path

    def meta(self):
        return SimpleNamespace(
            id="fake-tts",
            type="fake_tts",
            model=self.get_model(),
        )


async def _create_api_key(
    app: Quart,
    authenticated_header: dict,
    *,
    scopes: list[str],
    name_prefix: str = "voice-test",
) -> str:
    test_client = app.test_client()
    create_res = await test_client.post(
        "/api/apikey/create",
        json={"name": f"{name_prefix}-{uuid.uuid4().hex[:8]}", "scopes": scopes},
        headers=authenticated_header,
    )
    assert create_res.status_code == 200
    create_data = await create_res.get_json()
    assert create_data["status"] == "ok"
    return create_data["data"]["api_key"]


@pytest_asyncio.fixture(scope="module")
async def core_lifecycle_td(tmp_path_factory):
    tmp_db_path = tmp_path_factory.mktemp("data") / "test_data_voice_api.db"
    db = SQLiteDatabase(str(tmp_db_path))
    log_broker = LogBroker()
    core_lifecycle = AstrBotCoreLifecycle(log_broker, db)
    await core_lifecycle.initialize()
    try:
        yield core_lifecycle
    finally:
        try:
            stop_result = core_lifecycle.stop()
            if asyncio.iscoroutine(stop_result):
                await stop_result
        except Exception:
            pass


@pytest.fixture(scope="module")
def app(core_lifecycle_td: AstrBotCoreLifecycle):
    shutdown_event = asyncio.Event()
    server = AstrBotDashboard(core_lifecycle_td, core_lifecycle_td.db, shutdown_event)
    return server.app


@pytest_asyncio.fixture(scope="module")
async def authenticated_header(app: Quart, core_lifecycle_td: AstrBotCoreLifecycle):
    test_client = app.test_client()
    response = await test_client.post(
        "/api/auth/login",
        json={
            "username": core_lifecycle_td.astrbot_config["dashboard"]["username"],
            "password": core_lifecycle_td.astrbot_config["dashboard"]["password"],
        },
    )
    data = await response.get_json()
    token = data["data"]["token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_open_api_voice_capabilities_use_chat_scope(
    app: Quart,
    authenticated_header: dict,
    core_lifecycle_td: AstrBotCoreLifecycle,
):
    raw_key = await _create_api_key(
        app,
        authenticated_header,
        scopes=["chat"],
        name_prefix="voice-capabilities-key",
    )
    stt_provider = FakeSTTProvider()
    tts_provider = FakeTTSProvider(__file__)
    provider_manager = core_lifecycle_td.plugin_manager.context.provider_manager
    original_stt = provider_manager.curr_stt_provider_inst
    original_tts = provider_manager.curr_tts_provider_inst
    provider_manager.curr_stt_provider_inst = stt_provider
    provider_manager.curr_tts_provider_inst = tts_provider

    try:
        res = await app.test_client().get(
            "/api/v1/voice/capabilities",
            headers={"X-API-Key": raw_key},
        )
    finally:
        provider_manager.curr_stt_provider_inst = original_stt
        provider_manager.curr_tts_provider_inst = original_tts

    assert res.status_code == 200
    data = await res.get_json()
    assert data["status"] == "ok"
    assert data["data"]["can_transcribe"] is True
    assert data["data"]["can_speak"] is True
    assert data["data"]["stt_provider"]["id"] == "fake-stt"
    assert data["data"]["tts_provider"]["id"] == "fake-tts"


@pytest.mark.asyncio
async def test_open_api_voice_speech_returns_provider_audio(
    app: Quart,
    authenticated_header: dict,
    core_lifecycle_td: AstrBotCoreLifecycle,
    tmp_path,
):
    audio_path = tmp_path / "reply.wav"
    audio_path.write_bytes(b"RIFFfake-wave")
    raw_key = await _create_api_key(
        app,
        authenticated_header,
        scopes=["chat"],
        name_prefix="voice-speech-key",
    )
    tts_provider = FakeTTSProvider(str(audio_path))
    provider_manager = core_lifecycle_td.plugin_manager.context.provider_manager
    original_tts = provider_manager.curr_tts_provider_inst
    provider_manager.curr_tts_provider_inst = tts_provider

    try:
        res = await app.test_client().post(
            "/api/v1/voice/speech",
            json={"text": "你好"},
            headers={"X-API-Key": raw_key},
        )
    finally:
        provider_manager.curr_tts_provider_inst = original_tts

    assert res.status_code == 200
    assert res.mimetype == "audio/wav"
    assert await res.get_data() == b"RIFFfake-wave"
    assert tts_provider.texts == ["你好"]


@pytest.mark.asyncio
async def test_open_api_voice_transcriptions_uses_uploaded_file(
    app: Quart,
    authenticated_header: dict,
    core_lifecycle_td: AstrBotCoreLifecycle,
):
    raw_key = await _create_api_key(
        app,
        authenticated_header,
        scopes=["chat"],
        name_prefix="voice-transcription-key",
    )
    stt_provider = FakeSTTProvider()
    provider_manager = core_lifecycle_td.plugin_manager.context.provider_manager
    original_stt = provider_manager.curr_stt_provider_inst
    provider_manager.curr_stt_provider_inst = stt_provider

    try:
        res = await app.test_client().post(
            "/api/v1/voice/transcriptions",
            files={
                "file": FileStorage(
                    stream=BytesIO(b"RIFFfake-wave"),
                    filename="input.wav",
                    content_type="audio/wav",
                )
            },
            headers={"X-API-Key": raw_key},
        )
    finally:
        provider_manager.curr_stt_provider_inst = original_stt

    assert res.status_code == 200
    data = await res.get_json()
    assert data["status"] == "ok"
    assert data["data"]["text"] == "识别文本"
    assert len(stt_provider.audio_paths) == 1
    assert os.path.exists(stt_provider.audio_paths[0]) is False
