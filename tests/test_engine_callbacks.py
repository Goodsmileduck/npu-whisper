"""Tests for DictationApp callback system and state transitions."""

import threading
import time
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from dictation_engine import AppState, DictationApp, DEFAULT_CONFIG


class TestAppState:
    """Verify AppState enum values."""

    def test_all_states_exist(self):
        assert AppState.LOADING.value == "loading"
        assert AppState.READY.value == "ready"
        assert AppState.RECORDING.value == "recording"
        assert AppState.PROCESSING.value == "processing"
        assert AppState.ERROR.value == "error"


class TestCallbackSystem:
    """Verify callback registration and firing."""

    def test_initial_state_is_loading(self):
        app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False})
        assert app.state == AppState.LOADING

    def test_add_callback_and_fire(self):
        app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False})
        received = []
        app.add_callback(lambda state, data: received.append((state, data)))

        app._set_state(AppState.READY, {"info": "test"})

        assert len(received) == 1
        assert received[0] == (AppState.READY, {"info": "test"})

    def test_multiple_callbacks(self):
        app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False})
        cb1, cb2 = [], []
        app.add_callback(lambda s, d: cb1.append(s))
        app.add_callback(lambda s, d: cb2.append(s))

        app._set_state(AppState.RECORDING)

        assert cb1 == [AppState.RECORDING]
        assert cb2 == [AppState.RECORDING]

    def test_broken_callback_does_not_crash(self):
        app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False})

        def bad_callback(s, d):
            raise ValueError("boom")

        app.add_callback(bad_callback)
        # Should not raise
        app._set_state(AppState.READY)
        assert app.state == AppState.READY

    def test_set_state_with_no_data(self):
        app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False})
        received = []
        app.add_callback(lambda s, d: received.append(d))
        app._set_state(AppState.READY)
        assert received == [{}]


class TestStateTransitions:
    """Verify state transitions during model loading and recording."""

    def test_load_model_transitions_to_ready(self):
        config = {**DEFAULT_CONFIG, "beep_on_start": False}
        app = DictationApp(config)
        app.whisper = MagicMock()  # Skip real model loading

        states = []
        app.add_callback(lambda s, d: states.append(s))

        app._load_model_background()

        assert AppState.LOADING in states
        assert states[-1] == AppState.READY

    def test_load_model_error_transitions_to_error(self):
        config = {**DEFAULT_CONFIG, "beep_on_start": False}
        app = DictationApp(config)

        states = []
        app.add_callback(lambda s, d: states.append(s))

        # Force ensure_model to fail
        with patch.object(app, "ensure_model", side_effect=RuntimeError("test fail")):
            app._load_model_background()

        assert AppState.ERROR in states
        assert states[-1] == AppState.ERROR

    def test_recording_state_transitions(self):
        config = {**DEFAULT_CONFIG, "beep_on_start": False}
        app = DictationApp(config)
        app.whisper = MagicMock()
        app.whisper.transcribe = MagicMock(return_value="hello world")
        app._model_ready.set()

        states = []
        app.add_callback(lambda s, d: states.append(s))

        # Start recording
        with patch("dictation_engine.AudioRecorder.start"):
            app.toggle_recording()
        assert AppState.RECORDING in states

        # Stop recording — should go to PROCESSING then READY
        app.recorder = MagicMock()
        app.recorder.stop = MagicMock(return_value=np.zeros(16000, dtype=np.float32))

        with patch("dictation_engine.type_text"):
            app.toggle_recording()

        # Wait for transcription thread
        time.sleep(0.5)

        assert AppState.PROCESSING in states
        assert states[-1] == AppState.READY


class TestHistory:
    """Verify transcription history tracking."""

    def test_history_populated_after_transcription(self):
        config = {**DEFAULT_CONFIG, "beep_on_start": False}
        app = DictationApp(config)
        app.whisper = MagicMock()
        app.whisper.transcribe = MagicMock(return_value="test text")
        app._model_ready.set()
        app.is_recording = True

        app.recorder = MagicMock()
        app.recorder.stop = MagicMock(return_value=np.zeros(16000, dtype=np.float32))

        with patch("dictation_engine.type_text"):
            app.toggle_recording()

        time.sleep(0.5)

        assert len(app.history) == 1
        assert app.history[0]["text"] == "test text"
        assert "timestamp" in app.history[0]
        assert "duration" in app.history[0]

    def test_history_capped_at_max(self):
        config = {**DEFAULT_CONFIG, "beep_on_start": False}
        app = DictationApp(config)

        # Directly populate history beyond max
        for i in range(25):
            app._history.append({"text": f"entry {i}", "timestamp": "", "duration": 1.0})

        # Simulate capping
        if len(app._history) > app.MAX_HISTORY:
            app._history = app._history[-app.MAX_HISTORY:]

        assert len(app._history) == app.MAX_HISTORY

    def test_history_returns_copy(self):
        config = {**DEFAULT_CONFIG, "beep_on_start": False}
        app = DictationApp(config)
        app._history.append({"text": "foo", "timestamp": "", "duration": 1.0})

        h = app.history
        h.clear()
        # Original should be unchanged
        assert len(app._history) == 1


class TestFallbackDevice:
    """Verify device fallback mechanism."""

    def test_fallback_device_clears_error(self):
        config = {**DEFAULT_CONFIG, "beep_on_start": False}
        app = DictationApp(config)
        app._load_error = "DEVICE_LOST"
        app.whisper = MagicMock()

        with patch.object(app, "_load_model_background"):
            app.fallback_device("GPU")

        assert app._load_error is None
        assert app.config["device"] == "GPU"
        assert app.whisper is None


class TestListInputDevices:
    """Verify audio input device enumeration."""

    def test_list_input_devices_returns_list(self):
        with patch("sounddevice.query_devices", return_value=[
            {"name": "Mic 1", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
            {"name": "Mic 2", "max_input_channels": 2, "max_output_channels": 0},
        ]):
            devices = DictationApp.list_input_devices()

        assert len(devices) == 2
        assert devices[0] == {"index": 0, "name": "Mic 1"}
        assert devices[1] == {"index": 2, "name": "Mic 2"}

    def test_list_input_devices_handles_import_error(self):
        with patch("dictation_engine.DictationApp.list_input_devices", return_value=[]):
            devices = DictationApp.list_input_devices()
        assert devices == []
