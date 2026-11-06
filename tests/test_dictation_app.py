"""Tests for DictationApp transcription threading."""
import threading
import time
from unittest.mock import MagicMock, patch
import pytest

from debora_whisper.dictation_engine import AppState, DictationApp, DEFAULT_CONFIG


@pytest.mark.parametrize("action", ["interrupt", "start", "continuous"])
def test_hotkey_hands_slow_actions_to_worker(action):
    app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False})
    app._voice_active = object() if action == "interrupt" else None
    app._continuous = action == "continuous"
    entered, release = threading.Event(), threading.Event()
    calls = []
    name = {"interrupt": "_interrupt_voice_reply", "start": "_start_recording",
            "continuous": "_end_continuous"}[action]

    def slow(*args):
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(2)

    with patch.object(app, name, side_effect=slow), patch.object(app, "_watch_key"):
        try:
            app.toggle_recording()
            assert entered.wait(2)
            app.toggle_recording()  # auto-repeat while the action is blocked
            assert len(calls) == 1
            assert calls[0] != threading.get_ident()
        finally:
            release.set()
            app.stop_monitoring()


def test_tap_duration_excludes_delayed_audio_start():
    app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False})
    app._hotkey_held = True
    app._hotkey_released_at = 100.1
    starter = MagicMock()
    with patch.object(app, "_begin_continuous") as begin, \
            patch.object(app, "_finish_recording") as finish:
        app._watch_key(100.0, starter)
    starter.join.assert_called_once()
    begin.assert_called_once_with(100.0)
    finish.assert_not_called()
    assert not app._hotkey_held


def test_native_hotkeys_start_once_and_stop_before_workers():
    app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False})
    provider, keyboard = MagicMock(), MagicMock()
    native = provider.native_hotkeys.return_value
    stop_monitoring = app.stop_monitoring

    def stop():
        native.stop.assert_called_once_with()
        stop_monitoring()

    with patch("debora_whisper.dictation_engine.platforms.current", return_value=provider):
        try:
            app._start_hotkeys(keyboard)
            app._start_hotkeys(keyboard)
            native.start.assert_called_once_with()
            keyboard.add_hotkey.assert_not_called()
            keyboard.hook.assert_not_called()
            with patch.object(app, "stop_monitoring", side_effect=stop):
                app.stop()
            app._start_hotkeys(keyboard)
            assert provider.native_hotkeys.call_count == 1
            assert not app._hotkey_worker.is_alive()
        finally:
            stop_monitoring()


def test_shutdown_waits_for_inflight_native_registration():
    app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False})
    entered, release = threading.Event(), threading.Event()
    provider = MagicMock()
    native = provider.native_hotkeys.return_value

    def start():
        entered.set()
        assert release.wait(2)

    native.start.side_effect = start
    with patch("debora_whisper.dictation_engine.platforms.current", return_value=provider):
        registration = threading.Thread(target=app._start_hotkeys, args=(MagicMock(),))
        shutdown = threading.Thread(target=app.stop)
        registration.start()
        try:
            assert entered.wait(2)
            shutdown.start()
            assert app._stopping.wait(2)
            native.stop.assert_not_called()
        finally:
            release.set()
            registration.join(2)
            if shutdown.ident is not None:
                shutdown.join(2)
            app.stop_monitoring()
        assert not registration.is_alive()
        assert not shutdown.is_alive()
        native.stop.assert_called_once_with()


class TestTranscriptionThreading:
    """Verify transcription runs in background thread, not hotkey thread."""

    def test_stop_removes_backend_hotkeys_before_joining_monitor(self):
        app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False})
        app._hotkey_removers.append(MagicMock())
        stop_monitoring = app.stop_monitoring
        with patch("keyboard.unhook_all") as unhook:
            def stop():
                unhook.assert_called_once_with()
                stop_monitoring()

            with patch.object(app, "stop_monitoring", side_effect=stop):
                app.stop()

    def test_toggle_recording_stop_does_not_block(self):
        """Stopping recording must return quickly (transcription in background)."""
        config = {**DEFAULT_CONFIG, "beep_on_start": False}
        app = DictationApp(config)

        # Mock the whisper model with a slow transcribe
        app.whisper = MagicMock()
        app.whisper.transcribe = MagicMock(side_effect=lambda *a, **kw: (time.sleep(0.5) or "hello"))
        app._model_ready.set()  # Mark model as loaded

        # Mock recorder to return some audio
        import numpy as np
        app.recorder = MagicMock()
        app.recorder.stop = MagicMock(return_value=np.zeros(16000, dtype=np.float32))

        # Simulate: start recording, then stop
        app.is_recording = True

        finished = threading.Event()
        app.add_callback(lambda s, d: finished.set() if s.value == 'ready' else None)
        with patch('keyboard.is_pressed', return_value=False), patch('debora_whisper.dictation_engine.type_text'):
            start = time.time()
            app.toggle_recording()
            elapsed = time.time() - start
            assert finished.wait(2), 'Transcription worker did not finish'

        # toggle_recording should return quickly (< 200ms), not wait for transcription
        assert elapsed < 0.3, f"toggle_recording blocked for {elapsed:.2f}s — transcription must run in background"

    @pytest.mark.parametrize('shutdown', [False, True])
    def test_inflight_transcription_is_serialized_and_respects_shutdown(self, shutdown):
        import numpy as np
        app = DictationApp({**DEFAULT_CONFIG, 'beep_on_start': False})
        app.recorder = MagicMock()
        app.recorder.stop.return_value = np.zeros(16000, dtype=np.float32)
        app.is_recording = True
        app._model_ready.set()
        entered = threading.Event()
        release = threading.Event()

        def transcribe(*args, **kwargs):
            entered.set()
            assert release.wait(2), 'Test did not release transcription'
            return 'completed speech'

        app.whisper = MagicMock()
        app.whisper.transcribe.side_effect = transcribe
        states = []
        app.add_callback(lambda state, data: states.append(state))
        with patch('debora_whisper.dictation_engine.type_text') as paste, patch('keyboard.unhook_all'):
            worker = threading.Thread(target=app._finish_recording)
            worker.start()
            try:
                assert entered.wait(2)
                # Neither another stop nor another hotkey may start parallel work.
                app._finish_recording()
                app.toggle_recording()
                app.recorder.stop.assert_called_once()
                app.recorder.start.assert_not_called()
                if shutdown:
                    app.stop()
            finally:
                release.set()
                worker.join(2)
            assert not worker.is_alive()
            assert not app._transcribing
            if shutdown:
                paste.assert_not_called()
                assert not app.history
                assert states == [AppState.PROCESSING]
            else:
                paste.assert_called_once_with('completed speech ', auto_enter=False)
                assert states == [AppState.PROCESSING, AppState.READY]


@pytest.mark.parametrize("listening", [False, True])
@pytest.mark.parametrize("text", [None, "Finished turn"])
def test_ready_passes_continuous_listening_to_overlay(listening, text):
    from debora_whisper.app import GUIApp

    gui = GUIApp.__new__(GUIApp)
    gui._root = MagicMock()
    gui._tray = MagicMock()
    gui._overlay = MagicMock()
    gui._engine = MagicMock(continuous_active=listening)
    gui._stop_audio_polling = MagicMock()
    gui._settings_status = MagicMock()
    gui._settings_set_apply = MagicMock()
    gui._update_ui(AppState.READY, {"text": text})
    if text:
        gui._overlay.show_result.assert_called_once_with(text, continuous_listening=listening)
    else:
        gui._overlay.show_ready.assert_called_once_with(continuous_listening=listening)


def test_voice_chat_delivers_both_sides_to_the_single_overlay():
    import numpy as np
    from debora_whisper.app import GUIApp
    from debora_whisper.ui.overlay import OverlayWindow

    gui = GUIApp.__new__(GUIApp)
    gui._config = {**DEFAULT_CONFIG, "beep_on_start": False, "voice_chat": True}
    gui._root = MagicMock()
    gui._tray = MagicMock()
    gui._stop_audio_polling = MagicMock()
    gui._settings_status = MagicMock()
    gui._settings_set_apply = MagicMock()
    with patch.object(OverlayWindow, "_build"), \
            patch.object(OverlayWindow, "_get_scale", return_value=1.0):
        gui._overlay = OverlayWindow(gui._root)
    gui._overlay._update_layout = MagicMock()
    engine = gui._engine = DictationApp(gui._config)
    engine.recorder = MagicMock()
    engine.add_callback(gui._update_ui)
    during_speech = []

    def respond(text, on_reply, stop):
        assert gui._overlay._conversation_lines()[0][1] == "Você: abre o log"
        on_reply("O log mostra.")
        on_reply("O log mostra. O TTS demorou.")
        during_speech.extend(gui._overlay._conversation_lines())
        assert gui._overlay._balloon_id is None
        return "O log mostra. O TTS demorou."

    with patch.object(engine.voice_chat, "respond", side_effect=respond):
        engine._voice_chat_turn("abre o log", np.zeros(160), True)
        worker = engine._voice_worker
        if worker is not None:
            worker.join(3)
            assert not worker.is_alive()
    assert [line[1] for line in during_speech] == [
        "Você: abre o log", "Débora: O log mostra. O TTS demorou."]
    assert gui._overlay._conversation_lines() == during_speech
    assert gui._overlay._balloon_id is not None
    gui._update_ui(AppState.READY, {"notice": "Voice chat ready"})
    assert gui._overlay._conversation_lines() == during_speech
    gui._tray.update_state.assert_called_with("ready", "Débora Whisper — Voice chat ready")


@pytest.mark.parametrize("width", [None, 200, 360, 420.5])
def test_balloon_width_config_round_trip(tmp_path, monkeypatch, width):
    from debora_whisper import dictation_engine as de
    from debora_whisper.app import GUIApp

    monkeypatch.setattr(de, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(de, "CONFIG_FILE", tmp_path / "config.json")
    gui = GUIApp.__new__(GUIApp)
    gui._config = DEFAULT_CONFIG.copy()
    gui._on_width_changed(width)
    config = de.load_config()
    de.validate_config(config)
    assert config["balloon_width"] == width
    gui._on_pos_changed(200, 1042)
    assert de.load_config()["balloon_width"] == width


@pytest.mark.parametrize("width", [True, False, 0, -1, 199, "360", float("inf"), float("nan")])
def test_invalid_balloon_width_is_rejected(width):
    from debora_whisper.dictation_engine import validate_config
    with pytest.raises(ValueError, match="balloon_width"):
        validate_config({**DEFAULT_CONFIG, "balloon_width": width})
