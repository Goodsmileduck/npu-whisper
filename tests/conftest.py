"""Shared fixtures for dictation engine tests."""
import os
import sys
import tempfile
import threading
from time import monotonic
import warnings
from pathlib import Path

import pytest

_THREAD_JOIN_SECONDS = 2.0

# Importing the package must never move the user's pre-rename folders, nor
# create model or cache folders in the user's shared MODELS_DIR.
os.environ["DEBORA_WHISPER_NO_MIGRATION"] = "1"
os.environ["MODELS_DIR"] = tempfile.mkdtemp(prefix="debora-models-")

# Add project root to path so we can import debora_whisper
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True, scope="session")
def _private_logs(tmp_path_factory):
    """Tests log to a temporary folder, never to the user's app.log, and
    never record a lost NPU or a hardware inventory for the real app."""
    from debora_whisper import dictation_engine as engine
    home = tmp_path_factory.mktemp("debora")
    engine.LOG_DIR = home / "logs"
    engine.LOG_FILE = engine.LOG_DIR / "app.log"
    engine.TELEMETRY_LOG = engine.LOG_DIR / "telemetry.log"
    engine.TTS_SERVER_LOG = engine.LOG_DIR / "tts_server.log"
    engine.LLM_SERVER_LOG = engine.LOG_DIR / "llm_server.log"
    engine.NPU_LOST_FILE = home / "npu_lost.json"
    engine.HARDWARE_INVENTORY = home / "hardware-inventory.json"


@pytest.fixture(autouse=True)
def _no_npu_loss_record():
    from debora_whisper import dictation_engine as engine
    engine.NPU_LOST_FILE.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def _private_last_recording(tmp_path, monkeypatch):
    """A DictationApp deletes or overwrites last_recording.wav; never the
    user's real one."""
    from debora_whisper import dictation_engine as engine
    monkeypatch.setattr(engine, "LAST_RECORDING", tmp_path / "last_recording.wav")


@pytest.fixture(autouse=True)
def _private_silero_model(tmp_path, monkeypatch):
    """Never use the operator's cached VAD, even if MODEL_DIR is rebound.

    VAD tests may populate this private path and inject a stub session.
    """
    from debora_whisper import dictation_engine as engine
    model_path = tmp_path / "silero_vad" / "silero_vad.onnx"
    monkeypatch.setattr(engine, "silero_vad_model_path", lambda: model_path)


@pytest.fixture(autouse=True)
def _reset_allocation_reporter():
    from debora_whisper.hardware_allocation import allocation
    with allocation.lock:
        allocation.rows = {}
        allocation.owners = {}
        allocation.config = allocation.log = allocation.last = None
    yield
    with allocation.lock:
        allocation.config = allocation.log = None


@pytest.fixture(autouse=True)
def _reset_device_failure_latch():
    """The fatal device latch is process-wide; isolate it per test."""
    from debora_whisper import dictation_engine
    dictation_engine._reset_device_failure_for_tests()
    yield
    dictation_engine._reset_device_failure_for_tests()


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "shipped_defaults: sees DEFAULT_CONFIG as users get it")
    config.addinivalue_line(
        "markers", "allow_thread_leak: allow surviving threads (explain why in a comment)")


@pytest.fixture(autouse=True)
def _thread_guard(request, monkeypatch):
    """Clean up owned services before monkeypatch teardown; report all survivors."""
    from debora_whisper import dictation_engine as engine, voice_chat, device_inventory
    from tqdm import tqdm
    # Progress accounting is synchronous; tests need no background UI refresh.
    monkeypatch.setattr(tqdm, "monitor_interval", 0)
    before = set(threading.enumerate())
    apps = []
    initialize = engine.DictationApp.__init__

    def track_app(self, *args, **kwargs):
        apps.append(self)
        initialize(self, *args, **kwargs)

    monkeypatch.setattr(engine.DictationApp, "__init__", track_app)
    yield
    for app in apps:
        if hasattr(app, "_stopping"):
            app.stop_monitoring()
    voice_chat.stop_tts_health_polls()
    device_inventory.stop_inventory_workers()
    # One shared budget, including children spawned while their parent exits.
    deadline = monotonic() + _THREAD_JOIN_SECONDS
    while True:
        surviving = [t for t in threading.enumerate() if t not in before and t.is_alive()]
        if not surviving or monotonic() >= deadline:
            break
        for worker in surviving:
            worker.join(max(0, deadline - monotonic()))
    if not surviving:
        return
    details = ", ".join(f"{t.name} (ident={t.ident}, daemon={t.daemon})" for t in surviving)
    message = f"{request.node.nodeid}: new threads still alive after teardown: {details}"
    if request.node.get_closest_marker("allow_thread_leak"):
        warnings.warn(message, pytest.PytestWarning, stacklevel=1)
    elif any(not t.daemon for t in surviving):
        pytest.fail(message, pytrace=False)
    else:
        warnings.warn(message, pytest.PytestWarning, stacklevel=1)


@pytest.fixture(autouse=True)
def _dictation_by_default(request, monkeypatch):
    """Users get voice chat on by default; tests exercise dictation unless
    they turn it on, and never start the real LLM or TTS server by accident."""
    if request.node.get_closest_marker("shipped_defaults"):
        return
    from debora_whisper import dictation_engine
    monkeypatch.setitem(dictation_engine.DEFAULT_CONFIG, "voice_chat", False)


@pytest.fixture(autouse=True)
def _no_real_uia(monkeypatch):
    """Block UIA for every provider instance; UIA unit tests inject a fake API."""
    from debora_whisper.platforms import windows_uia
    monkeypatch.setattr(windows_uia, "comtypes", None)


@pytest.fixture(autouse=True)
def _reset_tts_capabilities(monkeypatch):
    from debora_whisper import voice_chat
    monkeypatch.setattr(voice_chat, "_tts_capabilities_cache", {})


@pytest.fixture(autouse=True)
def _unknown_audio_output(monkeypatch):
    """The developer's headphones must not switch the echo filter off."""
    from debora_whisper import platforms
    monkeypatch.setattr(platforms, "_current", None)
    provider = platforms.current()
    monkeypatch.setattr(provider, "default_output_name", lambda: None)
    monkeypatch.setattr(provider, "output_is_headphones", lambda name=None: None)
    # Engine tests must not inspect the developer's focused text box.
    monkeypatch.setattr(provider, "text_before_caret", lambda count: None)
    # Engine tests mock the keyboard package; on Linux the provider would
    # hand out the evdev module, which reads the developer's real keyboard.
    import importlib
    monkeypatch.setattr(provider, "keyboard", lambda: importlib.import_module("keyboard"))
    # Nor ask KWin which window has focus.
    if hasattr(provider, "_focused"):
        monkeypatch.setattr(provider, "_focused", lambda: None)


@pytest.fixture
def windows_text_provider(monkeypatch):
    """The Windows text path on any OS: tests of the clipboard/keyboard
    drafts mock its keys and UIA, so they run on Linux too."""
    from debora_whisper import platforms
    from debora_whisper.platforms.windows import WindowsPlatform
    provider = WindowsPlatform()
    monkeypatch.setattr(platforms, "_current", provider)
    monkeypatch.setattr(provider, "default_output_name", lambda: None)
    monkeypatch.setattr(provider, "output_is_headphones", lambda name=None: None)
    monkeypatch.setattr(provider, "text_before_caret", lambda count: None)
    monkeypatch.setattr(provider, "keyboard", lambda: __import__("keyboard"))
    return provider
