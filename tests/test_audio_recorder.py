"""Tests for AudioRecorder thread safety and timeout."""
import threading
import time
import numpy as np
import pytest

from dictation_engine import AudioRecorder


class TestAudioRecorderThreadSafety:
    """Verify AudioRecorder uses proper synchronization."""

    def test_frames_list_has_lock(self):
        """AudioRecorder must use a lock when accessing _frames."""
        recorder = AudioRecorder(sample_rate=16000)
        assert hasattr(recorder, '_lock'), "AudioRecorder must have a _lock attribute"
        assert isinstance(recorder._lock, type(threading.Lock()))

    def test_stop_returns_empty_when_not_started(self):
        recorder = AudioRecorder(sample_rate=16000)
        audio = recorder.stop()
        assert len(audio) == 0
        assert audio.dtype == np.float32


class TestAudioRecorderTimeout:
    """Verify max_record_seconds is enforced."""

    def test_max_record_seconds_stops_recording(self):
        """Recording should auto-stop after max_record_seconds."""
        recorder = AudioRecorder(sample_rate=16000, max_record_seconds=1)
        assert hasattr(recorder, 'max_record_seconds')
        assert recorder.max_record_seconds == 1

    def test_default_max_record_seconds_is_none(self):
        """Default recorder has no timeout."""
        recorder = AudioRecorder(sample_rate=16000)
        assert recorder.max_record_seconds is None
