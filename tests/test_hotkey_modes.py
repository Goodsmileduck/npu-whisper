"""Hotkey gestures: hold = push-to-talk, tap = continuous listening."""
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from debora_whisper.dictation_engine import (AppState, AudioRecorder, DEFAULT_CONFIG,
                                          DictationApp, validate_config)


def _wait(cond, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.005)
    return cond()


def _app(**config):
    app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False, **config})
    app.whisper = MagicMock()
    app.whisper.transcribe = MagicMock(return_value="hello world")
    app._model_ready.set()
    app.recorder = MagicMock()
    app.recorder.stop.return_value = np.zeros(16000, dtype=np.float32)
    app.recorder.last_speech_time = 0.0
    # Wide margins: a slow CI runner must not turn a tap into a hold.
    app.HOLD_SECONDS = 0.2
    states = []
    app.add_callback(lambda s, d: states.append(s))
    return app, states


def _press(app, hold_for=0.0, until=None):
    """One hotkey press, released after hold_for seconds. `until` is the
    outcome to wait for while the patches are still active."""
    released_at = time.time() + hold_for
    with patch("keyboard.is_pressed", side_effect=lambda _k: time.time() < released_at), \
            patch("debora_whisper.dictation_engine.type_text"):
        app.toggle_recording()
        assert _wait(lambda: not app._hotkey_held)
        if until is not None:
            assert _wait(until)
        assert _wait(lambda: not app._transcribing)


def test_tap_starts_continuous_listening():
    app, states = _app()
    _press(app)
    assert _wait(lambda: app.continuous_active)
    app.recorder.start.assert_called_once()
    app.recorder.begin_continuous.assert_called_once()
    app.recorder.stop.assert_not_called()  # push-to-talk audio is not transcribed
    app.whisper.transcribe.assert_not_called()
    assert _wait(lambda: states[-1] == AppState.RECORDING)
    assert app.is_recording


def test_second_tap_stops_continuous_listening():
    app, states = _app()
    _press(app)
    assert _wait(lambda: app.continuous_active)
    _press(app)
    assert _wait(lambda: not app.continuous_active)
    app.recorder.end_continuous.assert_called_once()
    assert _wait(lambda: states[-1] == AppState.READY)
    assert not app.is_recording
    app.recorder.start.assert_called_once()  # the stopping tap starts nothing


def test_tap_stops_continuous_even_while_a_segment_transcribes():
    app, _ = _app(continuous_listening=True)
    app.is_recording = True
    app._transcribing = True
    with patch("keyboard.is_pressed", return_value=False):
        app.toggle_recording()
        assert _wait(lambda: not app.continuous_active)
    app.recorder.end_continuous.assert_called_once()


def test_hold_is_push_to_talk():
    app, states = _app()
    _press(app, hold_for=0.5, until=lambda: app.whisper.transcribe.called
           and states[-1] == AppState.READY)
    assert not app.continuous_active
    app.recorder.begin_continuous.assert_not_called()
    app.recorder.stop.assert_called_once()
    app.whisper.transcribe.assert_called_once()
    assert AppState.PROCESSING in states and states[-1] == AppState.READY


def test_toggle_tap_action_keeps_the_recording_open():
    app, states = _app(tap_action="toggle")
    _press(app)
    assert app.is_recording and not app.continuous_active
    app.recorder.begin_continuous.assert_not_called()
    # The next tap stops and transcribes.
    _press(app, until=lambda: app.whisper.transcribe.called and states[-1] == AppState.READY)
    app.whisper.transcribe.assert_called_once()
    assert states[-1] == AppState.READY


def test_invalid_tap_action_is_rejected():
    with pytest.raises(ValueError):
        validate_config({**DEFAULT_CONFIG, "tap_action": "double"})


def test_recorder_switches_between_push_to_talk_and_vad():
    rec = AudioRecorder(sample_rate=16000, config={})
    rec._ensure_vad_thread = MagicMock()
    rec._write_pos = 32000
    rec.recording = True
    rec._frames = [np.zeros((10, 1), dtype=np.float32)]

    rec.begin_continuous(rewind_seconds=0.5)
    assert rec.continuous and not rec.paused and not rec.recording
    assert rec._frames == []
    assert rec._read_pos == 32000 - 8000  # VAD re-reads the tap
    rec._ensure_vad_thread.assert_called_once()

    rec.end_continuous()
    assert rec.paused and not rec.continuous


def test_continuous_from_startup_stops_on_a_tap():
    app, states = _app(continuous_listening=True)
    app.is_recording = True
    _press(app, until=lambda: not app.continuous_active)
    app.recorder.end_continuous.assert_called_once()
    app.recorder.start.assert_not_called()
    assert _wait(lambda: states and states[-1] == AppState.READY)


def test_stop_while_the_final_transcribes_ends_ready_not_recording():
    """Regression: the final used the listening state from before
    transcription and put the island back to RECORDING with the mic off."""
    app, states = _app(continuous_listening=True)
    app.is_recording = True
    started, release = threading.Event(), threading.Event()

    def slow_transcribe(*_a, **_k):
        started.set()
        release.wait(2)
        return "hello world"
    app.whisper.transcribe.side_effect = slow_transcribe
    with patch("debora_whisper.dictation_engine.type_text"), \
            patch("debora_whisper.dictation_engine.get_input_target", return_value=None):
        worker = threading.Thread(target=app._finish_recording,
                                  kwargs={"audio": np.zeros(16000, dtype=np.float32)})
        worker.start()
        assert started.wait(2)
        app._end_continuous()
        release.set()
        worker.join(2)
    assert not app.continuous_active and not app.is_recording
    assert states[-1] == AppState.READY


def test_failed_switch_to_continuous_rolls_back():
    app, states = _app()
    app.is_recording = True
    app.recorder.begin_continuous.side_effect = RuntimeError("no VAD")
    app._begin_continuous(time.time())
    assert not app.continuous_active and not app.is_recording
    app.recorder.stop.assert_called_once()
    assert states[-1] == AppState.READY


def test_continuous_stops_after_idle_time():
    app, states = _app()
    app._continuous, app.is_recording = True, True
    app._continuous_since = time.time() - 10
    app._stop_continuous_when_idle(app._continuous_since, idle_seconds=1, poll=0.01)
    assert not app.continuous_active
    app.recorder.end_continuous.assert_called_once()
    assert states[-1] == AppState.READY


def test_recent_speech_keeps_continuous_on():
    app, _ = _app()
    app._continuous, app.is_recording = True, True
    app._continuous_since = time.time() - 10
    app.recorder.last_speech_time = time.time()
    watcher = threading.Thread(target=app._stop_continuous_when_idle,
                               args=(app._continuous_since, 5), kwargs={"poll": 0.01})
    watcher.start()
    time.sleep(0.1)
    assert app.continuous_active
    app._end_continuous()  # a tap; the watcher sees it and exits
    watcher.join(1)
    assert not watcher.is_alive()
    app.recorder.end_continuous.assert_called_once()


def test_vad_stays_paused_during_push_to_talk():
    """The VAD thread starts on the tap; until begin_continuous it must not
    cut segments, or speech is typed and then typed again from the rewind."""
    rec = AudioRecorder(sample_rate=16000, config={})
    rec._ensure_vad_thread = MagicMock()
    rec.recording = True
    rec.prepare_vad()
    assert rec.paused
    rec.begin_continuous(rewind_seconds=0.5)
    assert not rec.paused
    rec.prepare_vad()  # already continuous: stays live
    assert not rec.paused


def test_failed_switch_leaves_the_vad_paused():
    app, _ = _app()
    app.is_recording = True
    app.recorder.begin_continuous.side_effect = RuntimeError("no VAD")
    app._begin_continuous(time.time())
    app.recorder.end_continuous.assert_called_once()


def test_idle_watcher_does_not_stop_a_newer_session():
    app, _ = _app()
    app._continuous, app.is_recording = True, True
    app._continuous_since = time.time()
    app._end_continuous(session=app._continuous_since - 5)  # a stale watcher
    assert app.continuous_active
    app.recorder.end_continuous.assert_not_called()
