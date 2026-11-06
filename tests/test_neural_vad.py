"""Silero VAD wrapper and the per-block speech decision."""
from unittest.mock import MagicMock

import numpy as np

from debora_whisper.dictation_engine import AudioRecorder, NeuralVAD


class FakeSession:
    """onnxruntime session boundary: records inputs, returns a fixed score."""

    def __init__(self, prob=0.9):
        self.inputs = []
        self.prob = prob

    def run(self, _outputs, feeds):
        self.inputs.append(feeds["input"].copy())
        return np.array([[self.prob]], np.float32), feeds["state"] + 1


def _vad(session):
    vad = NeuralVAD.__new__(NeuralVAD)
    vad.sample_rate = 16000
    vad.session = session
    vad.reset_state()
    return vad


def test_each_block_is_prefixed_with_previous_64_samples():
    session = FakeSession()
    vad = _vad(session)
    first = np.arange(512, dtype=np.float32)
    second = np.arange(512, 1024, dtype=np.float32)

    vad.process(first)
    vad.process(second)

    assert session.inputs[0].shape == (1, 576)
    np.testing.assert_array_equal(session.inputs[0][0, :64], np.zeros(64))
    np.testing.assert_array_equal(session.inputs[1][0, :64], first[-64:])
    np.testing.assert_array_equal(session.inputs[1][0, 64:], second)


def test_reset_clears_context_and_state():
    session = FakeSession()
    vad = _vad(session)
    vad.process(np.ones(512, np.float32))

    vad.reset_state()
    assert not vad.state.any()
    vad.process(np.ones(512, np.float32))

    np.testing.assert_array_equal(session.inputs[1][0, :64], np.zeros(64))


def _recorder(prob, **config):
    recorder = AudioRecorder(sample_rate=16000, config=config)
    recorder.neural_vad = MagicMock()
    recorder.neural_vad.process.return_value = prob
    return recorder


BLOCK = np.full(512, 0.05, np.float32)


def test_speech_starts_above_threshold_and_ends_below_hysteresis():
    assert not _recorder(0.45)._block_is_speech(BLOCK, is_speaking=False)
    assert _recorder(0.55)._block_is_speech(BLOCK, is_speaking=False)
    # Once speaking, a soft syllable (0.4) still counts as speech...
    assert _recorder(0.40)._block_is_speech(BLOCK, is_speaking=True)
    # ...but real silence ends it.
    assert not _recorder(0.30)._block_is_speech(BLOCK, is_speaking=True)


def test_loud_noise_is_not_speech_when_silero_says_no():
    # The old hybrid rule counted any block with RMS > 0.01 as speech.
    loud = (np.random.default_rng(0).standard_normal(512) * 0.2).astype(np.float32)
    assert not _recorder(0.01)._block_is_speech(loud, is_speaking=False)


def test_threshold_is_configurable():
    recorder = _recorder(0.3, vad_speech_threshold=0.25)
    assert recorder._block_is_speech(BLOCK, is_speaking=False)


def test_rms_fallback_without_silero():
    recorder = AudioRecorder(sample_rate=16000)
    assert recorder.neural_vad is None
    assert recorder._block_is_speech(BLOCK, is_speaking=False)
    assert not recorder._block_is_speech(np.zeros(512, np.float32), is_speaking=False)
