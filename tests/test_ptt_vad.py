"""Push-to-talk finals require speech when cached Silero is available."""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from debora_whisper import dictation_engine as engine


@pytest.fixture
def app(monkeypatch):
    app = engine.DictationApp({**engine.DEFAULT_CONFIG, "beep_on_start": False,
                              "save_last_recording": False})
    app.whisper = MagicMock()
    app.whisper.transcribe.return_value = "E aí"
    app._model_ready.set()
    app.is_recording = True
    app.recorder.stop = MagicMock(return_value=np.zeros(80000, np.float32))
    monkeypatch.setattr(engine, "type_text", MagicMock())
    monkeypatch.setattr(engine, "get_input_target", lambda: None)
    monkeypatch.setattr(engine, "log", MagicMock())
    return app


def set_vad(app, probability):
    vad = engine.NeuralVAD.__new__(engine.NeuralVAD)
    vad.sample_rate = 16000
    vad.session = MagicMock()
    vad.session.run.side_effect = lambda _, feeds: (
        np.array([[probability]], np.float32), feeds["state"] + 1)
    vad.reset_state()
    app.recorder.neural_vad = vad
    return vad


@pytest.mark.parametrize("noise", [False, True])
def test_no_speech_is_not_typed(app, noise):
    if noise:
        app.recorder.stop.return_value = np.random.default_rng(0).normal(
            0, 0.2, 80000).astype(np.float32)
    set_vad(app, 0.01)

    app._finish_recording()

    engine.type_text.assert_not_called()
    app.whisper.transcribe.assert_not_called()
    engine.log.assert_any_call("PTT dropped: no speech detected (duration=5.00s).")
    assert app._state == engine.AppState.READY
    assert not app._transcribing


def test_short_speech_is_typed_without_changing_live_vad_state(app):
    app.recorder.stop.return_value = np.zeros(6400, np.float32)
    vad = set_vad(app, 0.9)
    vad.state.fill(7)
    vad.context.fill(0.25)

    app._finish_recording()

    engine.type_text.assert_called_once_with("E aí ", auto_enter=False)
    assert np.all(vad.state == 7)
    assert np.all(vad.context == 0.25)


@pytest.mark.parametrize("user_cache_exists", [False, True])
def test_missing_vad_keeps_existing_behavior_without_download(
        app, monkeypatch, tmp_path, user_cache_exists):
    # Simulate MODEL_DIR being rebound to a populated user cache. The suite's
    # private Silero lookup must still take precedence, independent of order.
    user_models = tmp_path / "user-models"
    monkeypatch.setattr(engine, "MODEL_DIR", user_models)
    if user_cache_exists:
        cached = user_models / "silero_vad" / "silero_vad.onnx"
        cached.parent.mkdir(parents=True)
        cached.touch()
    constructor = MagicMock(side_effect=AssertionError("must not load or download"))
    monkeypatch.setattr(engine, "NeuralVAD", constructor)

    app._finish_recording()

    engine.type_text.assert_called_once_with("E aí ", auto_enter=False)
    constructor.assert_not_called()


@pytest.mark.parametrize("cached", [False, True])
def test_constructor_uses_private_cache_without_download(monkeypatch, cached):
    model_path = engine.silero_vad_model_path()
    session = MagicMock()
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=session))
    download = MagicMock(side_effect=AssertionError("must not download"))
    monkeypatch.setattr("urllib.request.urlretrieve", download)
    if cached:
        model_path.parent.mkdir()
        model_path.touch()
        engine.NeuralVAD(allow_download=False)
        session.assert_called_once_with(str(model_path), providers=["CPUExecutionProvider"])
    else:
        with pytest.raises(FileNotFoundError):
            engine.NeuralVAD(allow_download=False)
        session.assert_not_called()
    download.assert_not_called()


def test_voice_chat_ptt_without_speech_resumes_reply(app):
    app.config["voice_chat"] = True
    app.voice_chat = MagicMock()
    app._voice_chat_turn = MagicMock()
    set_vad(app, 0.01)

    app._finish_recording()

    app._voice_chat_turn.assert_not_called()
    app.voice_chat.resume.assert_called_once_with("no speech")
    engine.type_text.assert_not_called()


def test_cached_vad_loads_without_download(app, monkeypatch):
    vad = set_vad(app, 0.01)
    app.recorder.neural_vad = None
    model_path = engine.silero_vad_model_path()
    model_path.parent.mkdir()
    model_path.touch()
    constructor = MagicMock(return_value=vad)
    monkeypatch.setattr(engine, "NeuralVAD", constructor)
    monkeypatch.setattr(engine, "cpu_target", lambda: "CPU")

    app._finish_recording()

    constructor.assert_called_once_with(sample_rate=16000, allow_download=False)
    engine.type_text.assert_not_called()


@pytest.mark.parametrize("failure", ["load", "inference"])
def test_vad_failure_keeps_existing_behavior(app, monkeypatch, failure):
    if failure == "load":
        model_path = engine.silero_vad_model_path()
        model_path.parent.mkdir()
        model_path.touch()
        monkeypatch.setattr(engine, "NeuralVAD", MagicMock(side_effect=RuntimeError("load failed")))
    else:
        vad = set_vad(app, 0.01)
        vad.session.run.side_effect = RuntimeError("inference failed")

    app._finish_recording()

    engine.type_text.assert_called_once_with("E aí ", auto_enter=False)
    assert not app._transcribing


@pytest.mark.parametrize("frames, expected", [(512, False), (3199, False), (3200, True)])
def test_speech_minimum_counts_real_samples_not_padding(app, frames, expected):
    set_vad(app, 0.3)
    app.recorder.vad_threshold = 0.25
    app.recorder.vad_neg_threshold = 0.1

    assert app.recorder.ptt_has_speech(np.zeros(frames, np.float32)) is expected


@pytest.mark.parametrize("probabilities, expected", [
    ([0.6] + [0.4] * 6, True),   # onset, then a soft tail above the lower threshold
    ([0.4] * 7, False),          # never crosses the onset threshold
])
def test_speech_keeps_listener_hysteresis(app, probabilities, expected):
    vad = set_vad(app, 0.0)
    scores = iter(probabilities)
    vad.session.run.side_effect = lambda _, feeds: (
        np.array([[next(scores)]], np.float32), feeds["state"])
    app.recorder.vad_threshold = 0.5
    app.recorder.vad_neg_threshold = 0.35

    assert app.recorder.ptt_has_speech(np.zeros(512 * 7, np.float32)) is expected


def test_continuous_final_does_not_run_ptt_check(app):
    app.recorder.ptt_has_speech = MagicMock(return_value=False)

    app._finish_recording(audio=np.zeros(8000, np.float32))

    app.recorder.ptt_has_speech.assert_not_called()
    engine.type_text.assert_called_once_with("E aí ", auto_enter=False)
