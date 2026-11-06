"""Deterministic audio lifecycle tests; never open real audio devices."""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from debora_whisper import dictation_engine as engine


@pytest.fixture
def audio_backend(monkeypatch):
    streams = []
    flags = SimpleNamespace(input_overflow=False, output_underflow=False)

    class Status:
        input_overflow = False
        output_underflow = False

        def __bool__(self):
            return self.input_overflow or self.output_underflow

    class Stream:
        def __init__(self, direction, **kwargs):
            self.direction = direction
            self.callback = kwargs['callback']
            self.samplerate = kwargs['samplerate']
            self.channels = kwargs['channels']
            self.blocksize = kwargs['blocksize']
            self.device = 0
            self.latency = 0.01
            self.active = False
            self.closed = False
            self.deliver_on_start = True
            streams.append(self)

        def start(self):
            self.active = True
            if self.deliver_on_start:
                for _ in range(3):
                    if self.direction == 'input':
                        self.feed(np.zeros(160, dtype=np.float32))
                    else:
                        self.render(160)

        def feed(self, data, adc=0.0, current=0.0, status=None):
            data = np.asarray(data, dtype=np.float32).reshape(-1, self.channels)
            self.callback(data, len(data), SimpleNamespace(
                inputBufferAdcTime=adc, currentTime=current), status or Status())

        def render(self, frames):
            data = np.full((frames, self.channels), np.nan, dtype=np.float32)
            self.callback(data, frames, SimpleNamespace(), flags)
            return data

        def stop(self):
            self.active = False

        def close(self):
            self.active = False
            self.closed = True

    sd = SimpleNamespace(
        InputStream=MagicMock(side_effect=lambda **kw: Stream('input', **kw)),
        OutputStream=MagicMock(side_effect=lambda **kw: Stream('output', **kw)),
        query_devices=MagicMock(return_value=dict(name='Test device', hostapi=0,
                                default_samplerate=48000, max_output_channels=2)),
        query_hostapis=MagicMock(return_value={'name': 'Test API'}),
        play=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, 'sounddevice', sd)
    # keyboard.unhook_all() initializes real listener threads even without hooks.
    monkeypatch.setitem(sys.modules, 'keyboard', MagicMock())
    monkeypatch.setattr(engine, 'log', MagicMock())
    return SimpleNamespace(sd=sd, streams=streams, Status=Status)


def test_variable_blocks_preserve_exact_lookback_and_live_audio(audio_backend):
    recorder = engine.AudioRecorder()
    recorder.warmup()
    stream = audio_backend.streams[0]
    # More than 1.5 seconds, split into unequal blocks and wrapping the ring.
    audio = np.arange(30000, dtype=np.float32)
    for block in np.split(audio, [700, 5500, 17999]):
        stream.feed(block)
    recorder.start()
    stream.feed([30000, 30001, 30002])
    result = recorder.stop()
    np.testing.assert_array_equal(result, np.arange(6000, 30003, dtype=np.float32))
    assert recorder.telemetry['live_frames'] == 3
    assert stream.active
    audio_backend.sd.InputStream.assert_called_once()
    assert audio_backend.sd.InputStream.call_args.kwargs['blocksize'] == 0
    recorder.close()


def test_block_larger_than_ring_and_next_recording(audio_backend):
    recorder = engine.AudioRecorder()
    recorder.warmup()
    stream = audio_backend.streams[0]
    stream.feed(np.arange(50000, dtype=np.float32))
    recorder.start()
    np.testing.assert_array_equal(recorder.stop(), np.arange(26000, 50000, dtype=np.float32))
    stream.feed([10, 11, 12])
    recorder.start()
    stream.feed([13, 14])
    np.testing.assert_array_equal(recorder.stop(), [10, 11, 12, 13, 14])
    recorder.close()


def test_telemetry_measures_callback_and_adc_gaps_without_mixing_clocks(audio_backend, monkeypatch):
    recorder = engine.AudioRecorder()
    recorder.warmup()
    stream = audio_backend.streams[0]
    # PortAudio and perf_counter deliberately use unrelated clock origins.
    monkeypatch.setattr(engine.time, 'perf_counter', lambda: 500.0)
    stream.feed(np.zeros(160), adc=10.0, current=10.02)
    recorder.start()
    monkeypatch.setattr(engine.time, 'perf_counter', lambda: 500.8)
    status = audio_backend.Status()
    status.input_overflow = True
    stream.feed(np.zeros(160), adc=10.81, current=10.84, status=status)
    recorder.stop()
    assert recorder.telemetry['input_overflows'] == 1
    assert recorder.telemetry['max_callback_gap'] == pytest.approx(0.8)
    assert recorder.telemetry['max_adc_gap'] == pytest.approx(0.8)
    assert recorder.telemetry['max_delivery_delay'] == pytest.approx(0.03)
    recorder.close()


def test_microphone_timeout_closes_stream_and_allows_retry(audio_backend):
    factory = audio_backend.sd.InputStream.side_effect

    def silent_stream(**kwargs):
        stream = factory(**kwargs)
        stream.deliver_on_start = False
        return stream

    audio_backend.sd.InputStream.side_effect = silent_stream
    recorder = engine.AudioRecorder()
    with pytest.raises(RuntimeError, match='stable audio callbacks'):
        recorder.warmup(timeout=0)
    assert audio_backend.streams[0].closed
    assert recorder._stream is None
    audio_backend.sd.InputStream.side_effect = factory
    recorder.warmup(timeout=0)
    recorder.wait_ready(timeout=0)  # Silence is valid captured audio.
    recorder.close()


def test_stale_microphone_is_not_ready(audio_backend, monkeypatch):
    recorder = engine.AudioRecorder()
    recorder.warmup()
    later = recorder._last_callback + 1
    monkeypatch.setattr(engine.time, 'perf_counter', lambda: later)
    with pytest.raises(RuntimeError, match='stalled'):
        recorder.wait_ready(timeout=0)
    recorder.close()


def test_chimes_reuse_stream_and_fill_unused_output_with_silence(audio_backend):
    player = engine.ChimePlayer()
    player.warmup()
    stream = audio_backend.streams[0]
    assert not stream.render(200).any()
    for name in ('start', 'stop', 'warning', 'start'):
        expected = player._tones[name].copy()
        player.play(name)
        # Exercise continuation and the partial final callback.
        first = stream.render(137)
        rest = stream.render(len(expected) - 137 + 20)
        actual = np.concatenate([first, rest])
        np.testing.assert_array_equal(actual[:len(expected), 0], expected)
        np.testing.assert_array_equal(actual[:, 0], actual[:, 1])
        assert not actual[len(expected):].any()
        assert not stream.render(100).any()
    player.warmup()
    audio_backend.sd.OutputStream.assert_called_once()
    audio_backend.sd.play.assert_not_called()
    player.close()
    player.close()
    player.play('start')
    assert stream.closed
    assert not player._pending


def test_newest_chime_replaces_pending_tone(audio_backend):
    player = engine.ChimePlayer()
    player.warmup()
    player.play('start')
    player.play('stop')
    output = audio_backend.streams[0].render(len(player._tones['stop']))
    np.testing.assert_array_equal(output[:, 0], player._tones['stop'])
    player.close()


def test_output_failure_does_not_prevent_capture(audio_backend):
    audio_backend.sd.OutputStream.side_effect = RuntimeError('No speaker')
    player = engine.ChimePlayer()
    player.warmup()
    player.play('start')
    recorder = engine.AudioRecorder()
    recorder.warmup()
    recorder.wait_ready(timeout=0)
    assert player._stream is None
    recorder.close()


def test_shutdown_closes_both_streams_even_when_input_stop_fails(audio_backend):
    app = engine.DictationApp(dict(engine.DEFAULT_CONFIG))
    app.chimes.warmup()
    app.recorder.warmup()
    input_stream = app.recorder._stream
    output_stream = app.chimes._stream
    input_stream.stop = MagicMock(side_effect=RuntimeError('Device lost'))
    with pytest.raises(RuntimeError, match='Device lost'):
        app.stop()
    assert input_stream.closed
    assert output_stream.closed
    app.stop()


def test_load_does_not_claim_ready_with_only_lookback_audio(audio_backend, monkeypatch):
    app = engine.DictationApp({**engine.DEFAULT_CONFIG, 'beep_on_start': False})
    app.whisper = MagicMock()
    # Startup callbacks populate the lookback, but no new callback arrives
    # during the warmup recording. Do not let old samples prove readiness.
    monkeypatch.setattr(app._stopping, 'wait', lambda timeout: False)
    app._load_model_background()
    assert app.state == engine.AppState.ERROR
    assert 'no new audio frames' in app._load_error
    app.whisper.transcribe.assert_not_called()
    app.stop()


def test_shutdown_during_model_load_never_reopens_audio(audio_backend, monkeypatch):
    app = engine.DictationApp(dict(engine.DEFAULT_CONFIG))
    monkeypatch.setattr(app, 'ensure_model', app.stop)
    app._load_model_background()
    assert all(stream.closed for stream in audio_backend.streams)
    assert not app._model_ready.is_set()
    audio_backend.sd.InputStream.assert_called_once()
    audio_backend.sd.OutputStream.assert_called_once()


def test_timeout_transcribes_captured_audio_once(audio_backend, monkeypatch):
    monkeypatch.setattr(engine.threading, 'Timer', MagicMock())
    paste = MagicMock()
    monkeypatch.setattr(engine, 'type_text', paste)
    app = engine.DictationApp({**engine.DEFAULT_CONFIG, 'beep_on_start': False})
    app.whisper = MagicMock()
    app.whisper.transcribe.return_value = 'captured speech'
    app.recorder.warmup()
    app.recorder.start()
    app.is_recording = True
    speech = np.linspace(-0.1, 0.1, 8000, dtype=np.float32)
    app.recorder._stream.feed(speech)
    generation = app.recorder._recording_generation
    app._model_ready.set()
    app.recorder._timeout_stop(generation)
    app.recorder._timeout_stop(generation)
    app._finish_recording()
    app.whisper.transcribe.assert_called_once()
    np.testing.assert_array_equal(app.whisper.transcribe.call_args.args[0][-8000:], speech)
    paste.assert_called_once_with('captured speech ', auto_enter=False)
    assert app.state == engine.AppState.READY
    assert not app.is_recording
    assert not app._transcribing
    app.stop()


def test_previous_timer_cannot_stop_new_recording(audio_backend, monkeypatch):
    monkeypatch.setattr(engine.threading, 'Timer', MagicMock())
    recorder = engine.AudioRecorder(max_record_seconds=1)
    recorder.warmup()
    recorder.start()
    old_generation = recorder._recording_generation
    recorder.stop()
    recorder.start()
    recorder._stream.feed([1, 2, 3])
    recorder._timeout_stop(old_generation)
    assert recorder.recording
    np.testing.assert_array_equal(recorder.stop(), [1, 2, 3])
    recorder.close()


def test_audio_captured_during_reply_reaches_asr_without_skipping(audio_backend, monkeypatch):
    import threading

    app = engine.DictationApp({**engine.DEFAULT_CONFIG, "voice_chat": True,
                               "beep_on_start": False, "continuous_listening": True,
                               "vad_lookback_seconds": 0, "vad_trailing_seconds": 0})
    recorder = app.recorder
    # Use deterministic RMS classification, with no model/device loading.
    monkeypatch.setattr(engine, "NeuralVAD", MagicMock(side_effect=RuntimeError("test RMS")))
    monkeypatch.setattr(app, "ensure_model", lambda: None)
    monkeypatch.setattr(engine, "ensure_devices_usable", lambda: None)
    app.whisper = MagicMock()
    app.whisper.transcribe.return_value = "quero continuar minha ideia."
    app._model_ready.set()
    started, received, release = threading.Event(), threading.Event(), threading.Event()
    turns = []

    def respond(text, on_reply, stop):
        turns.append(text)
        if len(turns) == 1:
            started.set()
            assert release.wait(4)
        else:
            received.set()
        return "Certo."

    monkeypatch.setattr(app.voice_chat, "respond", respond)
    recorder.warmup()
    app.is_recording = True
    app._start_segment_consumer()
    try:
        app._voice_chat_turn("sobre o projeto...", np.zeros(16000), True)
        assert started.wait(2)
        # Exact 512-frame blocks, all captured while the first reply waits.
        speech = np.full(512 * 20, 0.1, dtype=np.float32)
        silence = np.zeros(512 * 32, dtype=np.float32)
        audio_backend.streams[0].feed(np.concatenate([speech, silence]))
        # ASR must finish before we allow the first reply to return.
        assert app._voice_stop.wait(2)
        # A draft at the pause, then the final: the last call is the whole segment.
        assert app.whisper.transcribe.call_count >= 1
        captured = app.whisper.transcribe.call_args.args[0]
        assert np.count_nonzero(captured) == len(speech)
        assert len(app.history) == 2
        release.set()
        assert received.wait(2)
        assert turns == ["sobre o projeto...", "quero continuar minha ideia."]
    finally:
        release.set()
        worker = app._voice_worker
        if worker is not None:
            worker.join(3)
        app.stop()
