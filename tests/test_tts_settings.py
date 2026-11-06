"""Settings and managed TTS switching, without a display, servers or models."""
import threading
import time
import weakref
from unittest.mock import MagicMock

import pytest

from debora_whisper import dictation_engine as de
from debora_whisper import voice_chat as vc
from debora_whisper.ui.settings import (
    SettingsWindow, _DEFAULT_VOICE, _TTS_ENGINES, _TTS_ENGINE_KEYS, _make_wide_dropdown,
)


def test_version_label_follows_background_results():
    window = SettingsWindow.__new__(SettingsWindow)
    window._win = MagicMock()
    window._version_label = MagicMock()
    text = "Débora Whisper 0.1.0 · ⬆ atualização disponível"
    window.update_version(text)
    assert window._version_text == text
    window._version_label.configure.assert_called_once_with(text=text)
    window._version_label.reset_mock()
    window._win.winfo_exists.return_value = False
    window.update_version("closed")
    window._version_label.configure.assert_not_called()


@pytest.mark.parametrize("backend,label", [
    ("chatterbox", "Chatterbox (clones the voice)"),
    ("qwen3", "Qwen3-TTS (experimental, clones the voice)"),
    ("kokoro", "Kokoro (fastest, pt-BR voice pf_dora)"),
    ("xtts", "XTTS-v2 (fast, clones the voice, non-commercial)"),
])
@pytest.mark.parametrize("voice", [_DEFAULT_VOICE, "isabel"])
def test_settings_apply_engine_and_voice(backend, label, voice, monkeypatch):
    assert _TTS_ENGINES[backend] == label
    assert _TTS_ENGINE_KEYS[label] == backend
    window = SettingsWindow.__new__(SettingsWindow)
    window._config = dict(de.DEFAULT_CONFIG)
    for attr, value in {
        "_model_radio_var": window._config["model_size"],
        "_hotkey_var": window._config["hotkey"],
        "_beep_var": True, "_enter_var": False, "_inline_drafts_var": False,
        "_voice_chat_var": True, "_backend_var": "Local (Qwen)",
        "_harness_new_session_var": True, "_harness_session_name_var": "Debora",
        "_voice_var": voice, "_balloon_var": True, "_font_size_var": "16",
        "_tts_engine_var": label,
    }.items():
        setattr(window, attr, MagicMock(get=MagicMock(return_value=value)))
    window._harness_cwd = None
    window._get_selected_lang_code = lambda: "pt"
    window._on_apply = MagicMock()
    window.update_status = MagicMock()
    monkeypatch.setattr(de, "detect_devices", lambda: ["CPU"])
    monkeypatch.setattr(de, "select_device", lambda *args: "CPU")
    original = dict(window._config)
    window._apply()
    if backend in ("qwen3", "xtts") and voice == _DEFAULT_VOICE:
        window._on_apply.assert_not_called()
        assert window._config == original
        name = "Qwen3" if backend == "qwen3" else "XTTS-v2"
        assert f"{name} requires a reference voice" in window.update_status.call_args.args[0]
        return
    saved = window._on_apply.call_args.args[0]
    assert saved["tts_backend"] == backend
    assert saved["tts_voice"] == (None if voice == _DEFAULT_VOICE else voice)


def test_kokoro_disables_voice_and_other_engines_restore_it():
    window = SettingsWindow.__new__(SettingsWindow)
    window._voice_dropdown = MagicMock()
    for backend in ("kokoro", "qwen3", "kokoro", "xtts", "chatterbox"):
        window._on_tts_engine_change(_TTS_ENGINES[backend])
        window._voice_dropdown.configure.assert_called_with(
            state="disabled" if backend == "kokoro" else "normal")


def test_disabled_wide_dropdown_does_not_open():
    menu = MagicMock()
    menu.cget.return_value = "disabled"
    _make_wide_dropdown(menu, ["voice"], None)
    menu._open_dropdown_menu()
    menu.winfo_rootx.assert_not_called()


@pytest.fixture
def managed_tts(monkeypatch):
    monkeypatch.setattr(vc, "_tts_process", None)
    monkeypatch.setattr(vc, "_tts_server_key", None)
    monkeypatch.setattr(vc, "_tts_retiring_url", None)
    monkeypatch.setattr(vc, "_tts_stopping", threading.Event())
    monkeypatch.setattr(vc.shutil, "which", lambda name: "uv.exe")
    monkeypatch.setattr(vc, "_check_tts_health", MagicMock(return_value="unavailable"))
    monkeypatch.setattr(vc.atexit, "register", MagicMock())
    kill = MagicMock()
    monkeypatch.setattr(vc, "kill_tree", kill)
    processes = [MagicMock(), MagicMock()]
    for process in processes:
        process.poll.return_value = None
    popen = MagicMock(side_effect=processes)
    monkeypatch.setattr(vc.subprocess, "Popen", popen)
    try:
        yield popen, kill, processes
    finally:
        with vc._tts_health_cache_lock:
            workers = [worker for _, worker in vc._tts_health_polls.values()]
        vc.stop_tts_server()
        for worker in workers:
            worker.join(3)


@pytest.mark.parametrize("backend", ["qwen3", "kokoro", "xtts"])
def test_backend_change_stops_old_server_before_starting_new(managed_tts, backend):
    popen, kill, processes = managed_tts
    calls = MagicMock()
    calls.attach_mock(popen, "start")
    calls.attach_mock(kill, "stop")
    vc.ensure_tts_server(de.DEFAULT_CONFIG)
    vc.ensure_tts_server({**de.DEFAULT_CONFIG, "tts_backend": backend})
    kill.assert_called_once_with(processes[0])
    processes[0].wait.assert_called_once_with(5)
    assert [call[0] for call in calls.mock_calls] == ["start", "stop", "start"]
    assert str(vc.TTS_SERVER_SCRIPTS[backend]) in popen.call_args.args[0]
    assert vc._tts_process is processes[1]
    assert not vc._tts_stopping.is_set()


@pytest.mark.parametrize("backend", ["chatterbox", "qwen3", "kokoro", "xtts"])
def test_same_backend_and_command_reuse_server(managed_tts, backend):
    popen, kill, _ = managed_tts
    config = {**de.DEFAULT_CONFIG, "tts_backend": backend}
    vc.ensure_tts_server(config)
    vc.ensure_tts_server({**config, "show_balloon": False})
    popen.assert_called_once()
    kill.assert_not_called()


@pytest.mark.parametrize("backend", ["qwen3", "xtts"])
def test_startup_voice_change_restarts_clone(managed_tts, backend):
    popen, kill, processes = managed_tts
    config = {**de.DEFAULT_CONFIG, "tts_backend": backend}
    vc.ensure_tts_server(config)
    vc.ensure_tts_server({**config, "tts_voice": "isabel"})
    kill.assert_called_once_with(processes[0])
    assert popen.call_count == 2
    command = popen.call_args.args[0]
    assert command[command.index("--voice") + 1].endswith("isabel.wav")


def test_kokoro_ignores_voice_change(managed_tts):
    popen, kill, _ = managed_tts
    config = {**de.DEFAULT_CONFIG, "tts_backend": "kokoro"}
    vc.ensure_tts_server(config)
    vc.ensure_tts_server({**config, "tts_voice": "isabel"})
    popen.assert_called_once()
    kill.assert_not_called()


@pytest.mark.parametrize("backend", ["chatterbox", "qwen3", "kokoro", "xtts"])
def test_language_change_does_not_restart_managed_server(managed_tts, backend):
    popen, kill, _ = managed_tts
    config = {**de.DEFAULT_CONFIG, "tts_backend": backend, "language": "pt"}
    vc.ensure_tts_server(config)
    vc.ensure_tts_server({**config, "language": "en"})
    popen.assert_called_once()
    kill.assert_not_called()


def test_chatterbox_voice_and_file_changes_do_not_restart_server(managed_tts, tmp_path):
    popen, kill, _ = managed_tts
    voice = tmp_path / "voice.wav"
    config = {**de.DEFAULT_CONFIG, "tts_backend": "chatterbox", "tts_voice": None}
    vc.ensure_tts_server(config)
    config["tts_voice"] = str(voice)
    vc.ensure_tts_server(config)
    voice.touch()
    assert "--voice" in vc.tts_command(config)
    vc.ensure_tts_server(config)
    voice.unlink()
    assert "--voice" not in vc.tts_command(config)
    vc.ensure_tts_server(config)
    popen.assert_called_once()
    kill.assert_not_called()


@pytest.mark.parametrize("change", [
    {"tts_server_command": ["custom-server", "--port", "9999"]},
    {"tts_url": "http://127.0.0.1:9999"},
])
def test_process_settings_restart_managed_server(managed_tts, change):
    popen, kill, processes = managed_tts
    vc.ensure_tts_server(de.DEFAULT_CONFIG)
    vc.ensure_tts_server({**de.DEFAULT_CONFIG, **change})
    kill.assert_called_once_with(processes[0])
    assert popen.call_count == 2


@pytest.mark.parametrize("health", ["ready", "loading", vc.TTSStartupError("still stopping")])
def test_restart_waits_for_child_to_stop_answering(managed_tts, monkeypatch, health):
    popen, kill, processes = managed_tts
    vc.ensure_tts_server(de.DEFAULT_CONFIG)
    monkeypatch.setattr(vc.time, "sleep", MagicMock())
    vc._check_tts_health.side_effect = [health, health, "unavailable", "unavailable"]
    vc.ensure_tts_server({**de.DEFAULT_CONFIG, "tts_backend": "kokoro"})
    assert vc._tts_process is processes[1]
    assert vc._tts_retiring_url is None
    assert popen.call_count == 2
    assert vc.time.sleep.call_count == 2


def test_retiring_child_is_not_adopted_after_shutdown_timeout(managed_tts, monkeypatch):
    popen, kill, processes = managed_tts
    vc.ensure_tts_server(de.DEFAULT_CONFIG)
    vc._check_tts_health.return_value = "ready"
    monkeypatch.setattr(vc.time, "monotonic", MagicMock(side_effect=[0, 6]))
    config = {**de.DEFAULT_CONFIG, "tts_backend": "kokoro"}
    vc.ensure_tts_server(config)
    assert vc._tts_process is None and vc._tts_server_key is None
    assert vc._tts_retiring_url == de.DEFAULT_CONFIG["tts_url"]
    assert popen.call_count == 1
    vc.time.monotonic.side_effect = [7]
    vc._check_tts_health.return_value = "unavailable"
    vc.ensure_tts_server(config, managed_only=True)
    assert vc._tts_process is processes[1]
    assert vc._tts_retiring_url is None
    kill.assert_called_once_with(processes[0])


@pytest.mark.parametrize("backend", ["qwen3", "xtts"])
@pytest.mark.parametrize("command", [None, ["custom-server", "--voice", "own.wav"]])
def test_clone_only_null_voice_is_valid_at_startup(command, backend):
    # A saved config must still load; Settings and the server's /health
    # explain the missing voice instead.
    de.validate_config({**de.DEFAULT_CONFIG, "tts_backend": backend,
                        "tts_voice": None, "tts_server_command": command})


@pytest.mark.parametrize("backend", ["qwen3", "xtts"])
def test_clone_only_custom_command_can_supply_its_own_voice_in_settings(backend):
    config = {**de.DEFAULT_CONFIG, "tts_backend": backend, "tts_voice": None,
              "tts_server_command": ["custom-server", "--voice", "own.wav"]}
    window = SettingsWindow.__new__(SettingsWindow)
    window._get_new_config = lambda: config
    window._on_apply = MagicMock()
    window.update_status = MagicMock()
    window._apply()
    window._on_apply.assert_called_once_with(config)
    window.update_status.assert_not_called()


def test_external_server_is_kept_and_backend_change_logged(managed_tts):
    popen, kill, _ = managed_tts
    vc._check_tts_health.return_value = "ready"
    logs = []
    vc.ensure_tts_server(de.DEFAULT_CONFIG, logs.append)
    vc.ensure_tts_server({**de.DEFAULT_CONFIG, "tts_backend": "kokoro"}, logs.append)
    popen.assert_not_called()
    kill.assert_not_called()
    assert vc._tts_process is None
    assert any("kokoro" in line and "external server" in line for line in logs)


def test_engine_loader_restarts_tts_off_the_calling_thread(managed_tts, monkeypatch):
    popen, kill, _ = managed_tts
    vc.ensure_tts_server(de.DEFAULT_CONFIG)
    threads = []
    kill.side_effect = lambda process: threads.append(threading.get_ident())
    monkeypatch.setattr(de, "ensure_tts_server", lambda config, *args: vc.ensure_tts_server(config))
    engine = de.DictationApp.__new__(de.DictationApp)
    engine.config = {**de.DEFAULT_CONFIG, "tts_backend": "kokoro",
                     "voice_chat_backend": "claude", "voice_chat": False}
    engine._voice_chat_loading = threading.Lock()
    engine.voice_chat = MagicMock()
    done = threading.Event()
    # Signal after the entire loader has returned, before fixtures are restored.
    load = engine._load_voice_chat

    def load_and_signal():
        try:
            load()
        finally:
            done.set()

    engine._load_voice_chat = load_and_signal
    engine._warm_up_voice_chat()
    assert done.wait(5)
    assert len(threads) == 1 and threads[0] != threading.get_ident()
    assert popen.call_count == 2
    engine.voice_chat.warm_tts.assert_called_once()


@pytest.fixture
def engine_startup():
    return de.DictationApp.start_background, de.DictationApp._warm_up_voice_chat


@pytest.fixture
def settings_gui(monkeypatch, engine_startup):
    """Use real engine lifecycle and history, with hardware and GUI stubbed."""
    from debora_whisper.app import GUIApp

    monkeypatch.setattr(de, "AudioRecorder", MagicMock())
    monkeypatch.setattr(de, "ChimePlayer", MagicMock())
    monkeypatch.setattr(de.DictationApp, "start_background", MagicMock())
    monkeypatch.setattr(de.DictationApp, "_warm_up_voice_chat", MagicMock())
    monkeypatch.setattr(vc.VoiceChat, "interrupt", MagicMock())
    monkeypatch.setattr(vc.VoiceChat, "warm_tts", MagicMock())
    monkeypatch.setattr("keyboard.unhook_all", MagicMock())
    monkeypatch.setitem(GUIApp._on_settings_apply.__globals__, "save_config", MagicMock())
    gui = GUIApp.__new__(GUIApp)
    gui._config = {**de.DEFAULT_CONFIG, "voice_chat": True,
                   "voice_chat_backend": "local", "voice_chat_echo_filter": "off"}
    gui._engine = de.DictationApp(gui._config)
    gui._overlay, gui._tray = MagicMock(), MagicMock()
    gui._settings_status, gui._settings_set_apply = MagicMock(), MagicMock()
    yield gui
    gui._engine.stop()
    gui._engine._resource_thread.join(2)
    assert not gui._engine._resource_thread.is_alive()


def test_repeated_audio_changes_dispose_models_before_replacement(settings_gui, monkeypatch):
    """Exercise Apply with the real telemetry thread and idle shutdown path."""
    gui = settings_gui
    models, retired = [], []

    class Model:
        def __init__(self):
            # Model wrappers can contain cycles: dropping whisper alone is
            # insufficient to reclaim their accelerator allocation promptly.
            self.cycle = self

    def load(engine, **kwargs):
        assert all(model() is None for model in models)
        engine.whisper = Model()
        models.append(weakref.ref(engine.whisper))

    monkeypatch.setattr(de.DictationApp, "start_background", load)
    load(gui._engine)
    try:
        for change in ({"sample_rate": 48000}, {"max_record_seconds": 30},
                       {"sample_rate": 16000}, {"max_record_seconds": 60}):
            previous = gui._engine
            retired.append(previous)  # Keep engines alive like pending callbacks.
            assert previous._resource_thread.is_alive()
            gui._on_settings_apply({**gui._config, **change})
            assert gui._engine is not previous
            assert previous.whisper is None
            previous._resource_thread.join(2)
            assert not previous._resource_thread.is_alive()
            assert sum(model() is not None for model in models) == 1
    finally:
        gui._engine.stop_if_idle()
        for engine in retired:
            engine.stop()
            engine._resource_thread.join(2)
    assert all(model() is None for model in models)


def test_idle_shutdown_releases_fallback_but_keeps_failed_model(settings_gui, monkeypatch):
    gui = settings_gui
    quarantine = []
    monkeypatch.setattr(de, "_quarantined_models", quarantine)
    failed = MagicMock()
    gui._engine.whisper = failed
    gui._engine._quarantine_lost_model()
    gui._engine.whisper = MagicMock()  # Healthy fallback on another device.
    fallback = weakref.ref(gui._engine.whisper)
    assert gui._engine.stop_if_idle() is None
    assert fallback() is None
    assert quarantine == [failed]
    assert gui._engine._quarantined_models == [failed]


@pytest.mark.parametrize("warm_voice_chat", [False, True])
@pytest.mark.parametrize("download", [False, True])
@pytest.mark.parametrize("change", ["backend", "voice"])
def test_apply_reconciles_managed_tts_in_dictation(
        settings_gui, managed_tts, engine_startup, monkeypatch,
        warm_voice_chat, download, change):
    gui = settings_gui
    popen, kill, processes = managed_tts
    gui._config.update(voice_chat=False, voice_chat_backend="claude",
                       tts_backend="qwen3", tts_voice="debora")
    previous = gui._engine
    previous.warm_voice_chat = warm_voice_chat
    vc.ensure_tts_server(gui._config)
    changed = {**gui._config}
    changed.update({"tts_backend": "kokoro"} if change == "backend"
                   else {"tts_voice": "isabel"})
    if download:
        changed["model_size"] = next(
            key for key in de.MODEL_REGISTRY if key != gui._config["model_size"])
    onboarding = MagicMock()
    gui._root = MagicMock()
    monkeypatch.setitem(gui._on_settings_apply.__globals__, "is_model_downloaded", lambda _: False)
    monkeypatch.setitem(gui._on_settings_apply.__globals__, "OnboardingWindow", onboarding)
    for name, method in zip(("start_background", "_warm_up_voice_chat"), engine_startup):
        monkeypatch.setattr(de.DictationApp, name, method)
    for name in ("_start_loader", "_start_segment_consumer", "_start_hotkeys"):
        monkeypatch.setattr(de.DictationApp, name, MagicMock())
    monkeypatch.setattr(vc.VoiceChat, "warm_tts", MagicMock())
    # Run the real background paths and join them before restoring fixtures.
    thread_class = threading.Thread
    workers, stop_threads = [], []

    def thread_factory(*args, **kwargs):
        worker = thread_class(*args, **kwargs)
        workers.append(worker)
        return worker

    monkeypatch.setattr(de.threading, "Thread", thread_factory)
    kill.side_effect = lambda _: stop_threads.append(threading.get_ident())
    try:
        gui._on_settings_apply(changed)
        if download:
            assert gui._engine is previous
            onboarding.call_args.kwargs["on_done"]()
    finally:
        # Engine services and TTS health polling live until fixture teardown;
        # only startup/reconciliation workers must have finished at this point.
        services = (gui._engine._resource_thread, gui._engine._hotkey_worker)
        for worker in workers:
            if worker not in services and worker.name != "tts-health":
                worker.join(5)
    # Reconciliation can append a poller while the parent worker is being joined.
    assert all(not worker.is_alive() for worker in workers
               if worker not in services and worker.name != "tts-health")
    assert (gui._engine is not previous) is download
    if not download:
        assert gui._engine.warm_voice_chat is warm_voice_chat
    assert gui._config["voice_chat"] is False
    kill.assert_called_once_with(processes[0])
    assert len(stop_threads) == 1 and stop_threads[0] != threading.get_ident()
    assert popen.call_count == 2
    assert vc._tts_process is processes[1]
    command = popen.call_args.args[0]
    assert str(vc.TTS_SERVER_SCRIPTS[changed["tts_backend"]]) in command
    if change == "voice":
        assert command[command.index("--voice") + 1].endswith("isabel.wav")


@pytest.mark.parametrize("external", [False, True])
def test_managed_only_reconciliation_does_not_start_or_probe_unmanaged_tts(managed_tts, external):
    popen, kill, _ = managed_tts
    vc._check_tts_health.return_value = "ready" if external else "unavailable"
    vc.ensure_tts_server({**de.DEFAULT_CONFIG, "tts_backend": "kokoro"}, managed_only=True)
    vc._check_tts_health.assert_not_called()
    popen.assert_not_called()
    kill.assert_not_called()


@pytest.fixture
def tts_updates(monkeypatch):
    """Run real daemon workers, but never contact a server."""
    monkeypatch.setattr(vc, "ensure_tts_server", MagicMock())
    workers = []
    thread_class = threading.Thread

    def thread_factory(*args, **kwargs):
        worker = thread_class(*args, **kwargs)
        if getattr(kwargs.get("target"), "__name__", "") == "reconfigure_tts":
            workers.append(worker)
        return worker

    monkeypatch.setattr(threading, "Thread", thread_factory)
    yield workers
    for worker in workers:
        worker.join(5)
        assert not worker.is_alive()


@pytest.mark.parametrize("change", [{"tts_backend": "kokoro"}, {"tts_voice": "isabel"}])
@pytest.mark.parametrize("busy", ["recording", "reply", "device_failure"])
def test_tts_apply_keeps_engine_and_live_state(settings_gui, tts_updates, monkeypatch, change, busy):
    gui = settings_gui
    previous = gui._engine
    model = previous.whisper = object()
    previous.warm_voice_chat = True
    previous._history = [{"text": "ditado antigo"}]
    previous.voice_chat._history = [{"role": "user", "content": "oi"}]
    if busy == "recording":
        previous.is_recording = previous._continuous = True
    elif busy == "reply":
        previous._voice_worker = object()
    else:
        monkeypatch.setitem(gui._on_settings_apply.__globals__, "device_failure", lambda: {"device": "NPU"})
    stop = MagicMock(side_effect=AssertionError("TTS must not stop the engine"))
    monkeypatch.setattr(previous, "stop_if_idle", stop)
    calls = []
    vc.ensure_tts_server.side_effect = lambda *args, **kwargs: calls.append(("ensure", threading.get_ident()))
    previous.voice_chat.warm_tts.side_effect = lambda: calls.append(("warm", threading.get_ident()))
    gui._on_settings_apply({**gui._config, **change, "show_balloon": False})
    worker = tts_updates[0]
    worker.join(5)
    assert not worker.is_alive() and worker.daemon
    assert [name for name, _ in calls] == ["ensure", "warm"]
    assert all(ident != threading.get_ident() for _, ident in calls)
    vc.ensure_tts_server.assert_called_once_with(gui._config, de.log, de.TTS_SERVER_LOG, managed_only=False)
    previous.voice_chat.interrupt.assert_called_once()
    previous.start_background.assert_not_called()
    assert gui._engine is previous and previous.whisper is model
    assert previous.config is gui._config
    assert previous.warm_voice_chat is True
    assert not previous._stopping.is_set()
    assert gui._config["show_balloon"] is False
    assert all(gui._config[key] == value for key, value in change.items())
    assert previous.history == [{"text": "ditado antigo"}]
    assert previous.voice_chat._history == [{"role": "user", "content": "oi"}]
    assert gui._settings_status.call_args.args[0] == "Settings saved."


def test_tts_apply_during_reply_keeps_finalized_history(settings_gui, tts_updates, monkeypatch):
    gui = settings_gui
    previous = gui._engine
    spoken, finalize = threading.Event(), threading.Event()
    answer = {"role": "assistant", "content": "Tudo bem!"}

    def respond(text, on_reply, stop):
        previous.voice_chat._history.append({"role": "user", "content": text})
        on_reply(answer["content"])
        spoken.set()
        # Simulate playback/cancellation finishing before respond saves the answer.
        assert finalize.wait(5)
        previous.voice_chat._history.append(answer)
        previous.voice_chat._last_turn = time.time()
        return answer["content"]

    monkeypatch.setattr(previous.voice_chat, "respond", respond)
    previous._voice_chat_turn("oi", [], True)
    worker = previous._voice_worker
    try:
        assert spoken.wait(5)
        gui._on_settings_apply({**gui._config, "tts_backend": "kokoro"})
        tts_updates[-1].join(5)
        assert gui._engine is previous
        assert gui._config["tts_backend"] == "kokoro"
        assert not previous._stopping.is_set()
        previous.voice_chat.interrupt.assert_called_once()
        previous.start_background.assert_not_called()
    finally:
        finalize.set()
        worker.join(5)
    assert not worker.is_alive()

    assert gui._engine.history[-1]["text"] == "oi\n→ Tudo bem!"
    assert answer in gui._engine.voice_chat._messages("e você?")


@pytest.mark.parametrize("warm", [False, True])
def test_live_tts_in_dictation_uses_warm_policy(settings_gui, tts_updates, warm):
    gui = settings_gui
    gui._config["voice_chat"] = False
    gui._engine.warm_voice_chat = warm
    gui._on_settings_apply({**gui._config, "tts_backend": "kokoro"})
    tts_updates[0].join(5)
    vc.ensure_tts_server.assert_called_once_with(
        gui._config, de.log, de.TTS_SERVER_LOG, managed_only=not warm)
    gui._engine.voice_chat.warm_tts.assert_called_once()


@pytest.mark.parametrize("rebuild", [False, True])
def test_disabling_local_chat_forgets_conversation_with_tts_change(settings_gui, tts_updates, rebuild):
    gui = settings_gui
    previous = gui._engine
    previous._history = [{"text": "ditado antigo"}]
    previous.voice_chat._history = [
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": "Tudo bem!"},
    ]
    previous.voice_chat._last_turn = time.time()
    changed = {**gui._config, "voice_chat": False, "tts_backend": "kokoro"}
    if rebuild:
        changed["sample_rate"] = 48000
    gui._on_settings_apply(changed)
    assert (gui._engine is not previous) is rebuild
    if not rebuild:
        tts_updates[0].join(5)
        assert gui._engine.history == [{"text": "ditado antigo"}]
    assert gui._engine.voice_chat._history == []

    # Re-enabling immediately must not resume the conversation before timeout.
    gui._on_settings_apply({**gui._config, "voice_chat": True})
    messages = gui._engine.voice_chat._messages("nova conversa")
    assert [m for m in messages if m["role"] != "system"] == [
        {"role": "user", "content": "nova conversa"},
    ]


@pytest.mark.parametrize("value", [True, False])
def test_tts_stream_bool_config(value):
    assert de.DEFAULT_CONFIG["tts_stream"] is True
    de.validate_config({**de.DEFAULT_CONFIG, "tts_stream": value})


@pytest.mark.parametrize("value", [0, 1, None, "true", "false", []])
def test_tts_stream_rejects_non_bool(value):
    with pytest.raises(ValueError, match="tts_stream must be a bool"):
        de.validate_config({**de.DEFAULT_CONFIG, "tts_stream": value})


@pytest.mark.parametrize("value", [True, 0, -1, "900", None, float("nan"), float("inf"), 1e7])
def test_tts_reply_timeout_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="tts_reply_timeout_seconds"):
        de.validate_config({**de.DEFAULT_CONFIG, "tts_reply_timeout_seconds": value})


@pytest.mark.parametrize("value", [0.5, 900, 3600, 86400])
def test_tts_reply_timeout_accepts_finite_positive_values(value):
    assert de.DEFAULT_CONFIG["tts_reply_timeout_seconds"] == 900
    de.validate_config({**de.DEFAULT_CONFIG, "tts_reply_timeout_seconds": value})
