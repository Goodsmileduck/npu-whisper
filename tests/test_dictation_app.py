"""Tests for DictationApp transcription threading."""
import threading
import time
from unittest.mock import MagicMock, patch
import pytest

from dictation_engine import DictationApp, DEFAULT_CONFIG


class TestTranscriptionThreading:
    """Verify transcription runs in background thread, not hotkey thread."""

    def test_toggle_recording_stop_does_not_block(self):
        """Stopping recording must return quickly (transcription in background)."""
        config = {**DEFAULT_CONFIG, "beep_on_start": False}
        app = DictationApp(config)

        # Mock the whisper model with a slow transcribe
        app.whisper = MagicMock()
        app.whisper.transcribe = MagicMock(side_effect=lambda *a, **kw: (time.sleep(0.5) or "hello"))

        # Mock recorder to return some audio
        import numpy as np
        app.recorder = MagicMock()
        app.recorder.stop = MagicMock(return_value=np.zeros(16000, dtype=np.float32))

        # Simulate: start recording, then stop
        app.is_recording = True

        start = time.time()
        app.toggle_recording()
        elapsed = time.time() - start

        # toggle_recording should return quickly (< 200ms), not wait for transcription
        assert elapsed < 0.3, f"toggle_recording blocked for {elapsed:.2f}s — transcription must run in background"
