"""Regression tests for the fail-closed accelerator recovery policy (issue #11).

A user hit OpenVINO GPU CL_OUT_OF_RESOURCES after 16.3 s of audio. OpenVINO
warns that later OpenCL calls may hang, yet the GUI reloaded the same GPU.
These tests drive the real engine (WhisperNPU / DictationApp) into the real
GUI consumer (GUIApp._on_state_change / _update_ui). Only the hardware
boundaries are faked: the openvino_genai pipeline, the microphone recorder,
the Tk root/tray/overlay widgets and the modal dialog.
"""
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from debora_whisper import dictation_engine as de
from debora_whisper.dictation_engine import AppState, DictationApp, DEFAULT_CONFIG
from tests.test_app_shutdown import GUIApp

CL_ERROR = (
    "Exception from src/inference/src/cpp/infer_request.cpp:223:\n"
    "[GPU] clWaitForEvents, error code: -5 CL_OUT_OF_RESOURCES. "
    "The OpenCL context may be in an unrecoverable state; "
    "subsequent OpenCL calls may hang."
)


class FakeGenAI:
    """Stand-in for the openvino_genai module (the GPU/NPU boundary)."""

    def __init__(self, generate_error=None, load_errors=None):
        self.generate_error = generate_error
        self.load_errors = dict(load_errors or {})
        self.loads = []
        self.load_kwargs = []
        self.generate_calls = 0
        self.on_generate = None  # test hook to pause inside inference
        fake = self

        class WhisperPipeline:
            def __init__(self, path, device, **kwargs):
                fake.loads.append(device)
                fake.load_kwargs.append(kwargs)
                if device in fake.load_errors:
                    raise RuntimeError(fake.load_errors[device])

            def get_generation_config(self):
                return SimpleNamespace()

            def generate(self, audio, config):
                fake.generate_calls += 1
                if fake.on_generate:
                    fake.on_generate()
                if fake.generate_error:
                    raise RuntimeError(fake.generate_error)
                return "hello"

        self.module = types.ModuleType("openvino_genai")
        self.module.WhisperPipeline = WhisperPipeline


@pytest.fixture
def genai(monkeypatch, tmp_path):
    fake = FakeGenAI(generate_error=CL_ERROR)
    monkeypatch.setitem(sys.modules, "openvino_genai", fake.module)
    fake_ov = types.ModuleType("openvino")
    class Core:
        available_devices = ["CPU", "GPU", "NPU"]

        def get_property(self, device, prop):
            return f"Fake {device}"
    fake_ov.Core = Core
    monkeypatch.setitem(sys.modules, "openvino", fake_ov)

    fake_fw = types.ModuleType("faster_whisper")
    class WhisperModel:
        def __init__(self, model_size_or_path, device, compute_type):
            pass
        def transcribe(self, audio, **kwargs):
            return [SimpleNamespace(text="hello")], {}
    fake_fw.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

    # Fallback follows device_priority over the present devices; pin them so
    # results do not depend on whether this machine has an RTX.
    monkeypatch.setattr(de, "has_nvidia_gpu", lambda return_name=False: None if return_name else False)
    monkeypatch.setattr(de, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(de, "setup_model", lambda config: tmp_path / "whisper-base")
    return fake


@pytest.fixture(autouse=True)
def isolate_desktop():
    with patch("keyboard.is_pressed", return_value=False), \
            patch("keyboard.unhook_all"), patch("debora_whisper.dictation_engine.type_text") as paste:
        yield paste


class FakeRecorder:
    """Microphone boundary: returns 16.3 s of audio like the user's report."""
    _recording_generation = 0
    telemetry = {"live_frames": 16000}

    def __init__(self):
        self.starts = 0

    def warmup(self):
        pass

    def wait_ready(self, timeout=3.0):
        pass

    def start(self):
        self.starts += 1

    def stop(self):
        return np.zeros(int(16000 * 16.3), dtype=np.float32)

    def ptt_has_speech(self, audio):
        return None

    def close(self):
        pass


class Root:
    def after(self, delay, fn, *args):
        fn(*args)
        return "after-id"

    def after_cancel(self, _id):
        pass


def _engine(config=None):
    app = DictationApp({**DEFAULT_CONFIG, "device": "GPU", "beep_on_start": False,
                        **(config or {})})
    app.recorder = FakeRecorder()
    return app


def _gui(engine):
    gui = GUIApp.__new__(GUIApp)
    gui._config = engine.config
    gui._root = Root()
    gui._engine = engine
    gui._tray = MagicMock()
    gui._overlay = MagicMock()
    gui._settings_win = None
    gui._audio_poll_id = None
    gui._alert_error = MagicMock()
    engine.add_callback(gui._on_state_change)
    return gui


def _transcribe_once(app):
    app.is_recording = True
    app._finish_recording()


def test_gpu_failure_during_transcription_falls_back(genai, isolate_desktop):
    app = _engine()
    app.ensure_model()
    app._model_ready.set()
    gui = _gui(app)
    states = []
    app.add_callback(lambda state, data: states.append((state, data)))

    with patch.object(app, "fallback_device", wraps=app.fallback_device) as fallback, \
            patch.object(app, "_load_model_background") as reload:
        _transcribe_once(app)

        # The GUI caught the device lost and swapped to NPU
        fallback.assert_called_once_with("NPU")
        reload.assert_called_once()
    assert genai.loads == ["GPU"]
    assert genai.generate_calls == 1

    # App no longer latches GPU failures
    assert de.device_failure() is None


def test_gpu_failure_during_warmup_falls_back_to_npu(genai):
    app = _engine()
    gui = _gui(app)
    states = []
    app.add_callback(lambda state, data: states.append((state, data)))

    with patch.object(app._stopping, "wait", return_value=False), \
            patch.object(app, "fallback_device") as fallback:
        app._load_model_background()

    # The app should fall back to NPU rather than failing closed
    fallback.assert_called_once_with("NPU")
    assert not any(d.get("restart_required") for _, d in states if isinstance(d, dict))
    assert de.device_failure() is None


def test_settings_apply_after_gpu_failure_reloads_normally(genai):
    app = _engine()
    app.ensure_model()
    app._model_ready.set()
    gui = _gui(app)
    gui._settings_status = MagicMock()
    
    with patch.object(app, "_load_model_background"):
        _transcribe_once(app)
        _join_daemons()

    factory = MagicMock()
    with patch.dict(GUIApp._on_settings_apply.__globals__, {
        "DictationApp": factory, "save_config": MagicMock(),
    }) as _:
        gui._on_settings_apply({**gui._config, "device": "CPU"})
        
    # It should successfully instantiate a new engine and reload
    factory.assert_called_once()
    assert gui._config["device"] == "CPU"


def _install_fake_parakeet_runtime(monkeypatch, tmp_path, compile_model):
    """OpenVINO/onnxruntime boundary for a real ParakeetNPU._load_pipeline."""
    class Model:
        def reshape(self, shapes):
            pass

    class Core:
        def read_model(self, path):
            return Model()

        def compile_model(self, model, device, cfg=None):
            return compile_model(device)

    ov = types.ModuleType("openvino")
    ov.Core = Core
    ort = types.ModuleType("onnxruntime")
    ort.InferenceSession = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, "openvino", ov)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    (tmp_path / "nemo128.onnx").write_bytes(b"")
    monkeypatch.setattr(de.ParakeetNPU, "_load_vocab", lambda self: None)
    monkeypatch.setattr(de, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(de, "setup_model", lambda config: tmp_path)
    monkeypatch.delenv("PARAKEET_LATENCY_OVERRIDE", raising=False)





def test_parakeet_benign_gpu_fallback_failure_still_reaches_cpu(monkeypatch, tmp_path):
    compiles = []

    def compile_model(device):
        compiles.append(device)
        if device in ("NPU", "GPU"):
            raise RuntimeError(f"[{device}] unsupported layer Foo")
        raise RuntimeError("stop here: CPU reached")

    _install_fake_parakeet_runtime(monkeypatch, tmp_path, compile_model)
    with pytest.raises(RuntimeError, match="CPU reached"):
        de.ParakeetNPU(tmp_path, device="NPU")
    assert compiles == ["NPU", "GPU", "CPU"]
    assert de.device_failure() is None


def _join_daemons():
    for t in threading.enumerate():
        if t is not threading.current_thread() and t.daemon:
            t.join(5)


def test_stop_waits_for_paste_in_progress(genai, isolate_desktop):
    """stop() during the final paste returns only after it finished, so
    nothing can be pasted after stop() has returned."""
    genai.generate_error = None
    app = _engine()
    app.ensure_model()
    app._model_ready.set()
    events, entered, release = [], threading.Event(), threading.Event()
    paste_timed_out = threading.Event()

    def paste(text, auto_enter=False):
        assert app._output_lock.locked(), "paste ran outside _output_lock"
        events.append("paste_start")
        entered.set()
        released = release.wait(5)
        if not released:
            paste_timed_out.set()
        assert released, "paste timed out waiting for the test to release it"
        events.append("paste_end")

    isolate_desktop.side_effect = paste
    worker = threading.Thread(target=_transcribe_once, args=(app,), daemon=True)
    worker.start()
    assert entered.wait(5)

    def stopper():
        app.stop()
        events.append("stop_returned")

    stop_thread = threading.Thread(target=stopper, daemon=True)
    stop_thread.start()
    try:
        stop_thread.join(0.3)
        assert not paste_timed_out.is_set(), (
            "test thread resumed after the paste timed out; "
            "this does not demonstrate stop() bypassing _output_lock"
        )
        assert app._output_lock.locked(), "paste released _output_lock too early"
        assert stop_thread.is_alive(), "stop() returned while a paste was in progress"
    finally:
        release.set()
        stop_thread.join(5)
        worker.join(5)
    assert events == ["paste_start", "paste_end", "stop_returned"]
    assert len(app.history) == 1


def test_stop_at_output_boundary_discards_text(genai, isolate_desktop):
    """A stop that completes after inference but right before the final
    check: the text is neither pasted nor stored."""
    genai.generate_error = None
    app = _engine()
    app.ensure_model()
    app._model_ready.set()
    states = []
    app.add_callback(lambda state, data: states.append(state))
    real_lock = app._output_lock

    class StopFirst:
        def __enter__(self):
            app._output_lock = real_lock
            app.stop()  # completes fully before the final check runs
            return real_lock.__enter__()

        def __exit__(self, *exc):
            return real_lock.__exit__(*exc)

    app._output_lock = StopFirst()
    _transcribe_once(app)
    assert genai.generate_calls == 1
    isolate_desktop.assert_not_called()
    assert app.history == []
    assert AppState.READY not in states


def test_settings_rebuild_refused_while_old_inference_runs(genai, isolate_desktop):
    """Old engine is mid-inference on the GPU when Settings asks for a rebuild:
    no config change, no new engine, no stop; after it finishes, Apply works."""
    genai.generate_error = None
    app = _engine()
    app.ensure_model()
    app._model_ready.set()
    gui = _gui(app)
    gui._settings_status = MagicMock()
    gui._settings_set_apply = MagicMock()
    inside, release = threading.Event(), threading.Event()

    def pause():
        inside.set()
        assert release.wait(5)

    genai.on_generate = pause
    worker = threading.Thread(target=_transcribe_once, args=(app,), daemon=True)
    worker.start()
    assert inside.wait(5)

    before = dict(gui._config)
    factory, save = MagicMock(), MagicMock()
    with patch.dict(GUIApp._on_settings_apply.__globals__, {
        "DictationApp": factory, "save_config": save,
    }):
        gui._on_settings_apply({**gui._config, "device": "CPU"})
        factory.assert_not_called()
        save.assert_not_called()
        gui._tray.update_info.assert_not_called()
        assert gui._config == before  # shared with the running engine
        assert not app._stopping.is_set()
        message = gui._settings_status.call_args.args[0]
        assert "transcription in progress" in message and "Apply again" in message

        release.set()
        worker.join(5)
        assert not worker.is_alive()
        isolate_desktop.assert_called_once()
        assert len(app.history) == 1

        gui._on_settings_apply({**gui._config, "device": "CPU"})
        factory.assert_called_once_with(gui._config)
        assert app._stopping.is_set()
    assert gui._config["device"] == "CPU"
    assert genai.loads == ["GPU"]


def test_stop_if_idle_refuses_recording_and_loading():
    app = _engine()
    app.is_recording = True
    assert app.stop_if_idle() == "recording"
    assert not app._stopping.is_set()
    app.is_recording = False

    entered, gate = threading.Event(), threading.Event()

    def slow_load():
        entered.set()
        assert gate.wait(5)

    with patch.object(app, "_load_model_background_inner", side_effect=slow_load):
        app._start_loader()
        assert entered.wait(5)
        assert app.stop_if_idle() == "model loading"
        assert not app._stopping.is_set()
        gate.set()
        _join_daemons()
    assert app.stop_if_idle() is None
    assert app._stopping.is_set()


def test_npu_loss_fallback_loader_stays_busy_after_first_loader_returns(
        genai, monkeypatch, tmp_path):
    """The NPU-loss ERROR callback synchronously starts the GPU retry fallback
    loader before the first loader has returned. Once the first loader is
    done, the second (held pending) must still report 'model loading' and
    refuse a Settings rebuild."""
    genai.generate_error = None
    genai.load_errors = {"NPU": "[NPU] ZE_RESULT_ERROR_DEVICE_LOST"}
    entered, gate = threading.Event(), threading.Event()

    def setup_model(config):
        if not getattr(setup_model, 'first_done', False):
            setup_model.first_done = True
        else:
            # Second load (the fallback to GPU)
            entered.set()
            assert gate.wait(5)
        return tmp_path / "whisper-base"
    setup_model.first_done = False

    monkeypatch.setattr(de, "setup_model", setup_model)
    app = _engine({"device": "NPU"})
    gui = _gui(app)  # synchronous Root: ERROR -> _update_ui -> fallback_device
    states = []
    app.add_callback(lambda state, data: states.append((state, data)))

    with patch.object(app._stopping, "wait", return_value=False), \
         patch.object(gui, "_schedule_npu_recovery"):
        app._load_model_background()  # first loader, returns here
        assert entered.wait(5), "GPU fallback loader never started"
        assert any(d.get("device_failure") == "NPU" for _, d in states)
        assert app.busy_reason() == "model loading"
        assert app.stop_if_idle() == "model loading"
        assert not app._stopping.is_set()
        gate.set()
        _join_daemons()

    # The NPU failed first time, then the second load was instantly attempted on GPU or CUDA
    if app.config["device"] == "CUDA":
        assert genai.loads == ["NPU"]
    else:
        assert genai.loads == ["NPU", "GPU"]
    assert states[-1][0] == AppState.READY
    assert app.busy_reason() is None
    assert de.device_failure() is None
    gui._alert_error.assert_not_called()


def test_loader_thread_start_failure_releases_reservation():
    app = _engine()
    with patch.object(threading.Thread, "start",
                      side_effect=RuntimeError("can't start new thread")):
        with pytest.raises(RuntimeError, match="start new thread"):
            app._start_loader()
    assert app.busy_reason() is None
    assert app.stop_if_idle() is None


def test_ready_callback_may_stop_engine_without_deadlock(genai):
    """READY used to be emitted under _audio_lifecycle_lock; a callback that
    stops the engine (or waits on a thread that does) deadlocked."""
    genai.generate_error = None
    app = _engine()
    stopped = []

    def on_state(state, data):
        if state == AppState.READY:
            app.stop()
            stopped.append(True)

    app.add_callback(on_state)
    with patch.object(app._stopping, "wait", return_value=False):
        loader = threading.Thread(target=app._load_model_background, daemon=True)
        loader.start()
        loader.join(5)
    assert not loader.is_alive(), "READY callback deadlocked against the loader"
    assert stopped == [True]
    assert app._stopping.is_set()
    assert app.busy_reason() is None





def test_gpu_load_failure_does_not_fall_back_to_cpu(genai, tmp_path):
    genai.load_errors = {"GPU": CL_ERROR}
    with pytest.raises(de.DeviceFailureError) as info:
        de.WhisperNPU(tmp_path, device="GPU")
    assert info.value.device == "GPU"
    assert CL_ERROR in str(info.value.__cause__)
    assert genai.loads == ["GPU"]


def test_benign_load_failure_keeps_cpu_fallback(genai, tmp_path):
    genai.load_errors = {"GPU": "[GPU] unsupported operation Foo"}
    model = de.WhisperNPU(tmp_path, device="GPU")
    assert model.device == "CPU"
    assert genai.loads == ["GPU", "CPU"]
    assert de.device_failure() is None


def test_npu_device_lost_falls_back_to_gpu_and_schedules_npu_recovery(genai):
    genai.generate_error = "[NPU] ZE_RESULT_ERROR_DEVICE_LOST"
    app = _engine({"device": "NPU"})
    app.ensure_model()
    app._model_ready.set()
    gui = _gui(app)
    
    with patch.object(app, "_load_model_background") as reload, \
         patch.object(gui, "_schedule_npu_recovery") as recovery:
        _transcribe_once(app)
        _join_daemons()

    reload.assert_called_once()
    assert app.config["device"] in ("GPU", "CUDA")
    recovery.assert_called_once()
    assert de.device_failure() is None
    gui._alert_error.assert_not_called()


def _probe_gui(genai):
    genai.generate_error = None
    app = _engine({"device": "CUDA"})
    gui = _gui(app)
    return app, gui


def test_npu_recovery_probe_rejects_silent_cpu_fallback(genai):
    # Real log: the lost NPU's quarantined model keeps the cache blob open,
    # the probe's NPU compile fails benignly and WhisperNPU lands on CPU.
    genai.load_errors = {"NPU": "remove: The process cannot access the file "
                                "because it is being used by another process."}
    app, gui = _probe_gui(genai)

    with patch.object(app, "inject_recovered_model") as inject, \
         patch.object(gui, "_schedule_npu_recovery") as recovery, \
         patch("debora_whisper.app.probe_npu"):
        gui._run_npu_recovery_probe()

    assert genai.loads == ["NPU", "CPU"]
    inject.assert_not_called()
    recovery.assert_called_once()
    assert app.config["device"] == "CUDA"


def test_npu_recovery_probe_swaps_when_npu_loads(genai):
    app, gui = _probe_gui(genai)
    de.remember_npu_loss("DEVICE_LOST")

    with patch.object(app, "inject_recovered_model") as inject, \
         patch.object(gui, "_schedule_npu_recovery") as recovery, \
         patch("debora_whisper.app.probe_npu") as probe:
        gui._run_npu_recovery_probe()

    probe.assert_called_once()
    assert not de.NPU_LOST_FILE.exists()

    assert genai.loads == ["NPU"]
    inject.assert_called_once()
    assert inject.call_args.args[0] == "NPU"
    recovery.assert_not_called()
    # An uncached NPU compile holds the GIL for minutes and froze the app.
    assert "CACHE_DIR" in genai.load_kwargs[0]


def test_failed_npu_probe_never_loads_on_the_npu_in_the_app(genai):
    """The probe process said no (or hung and was killed): the app itself
    must not touch the NPU, or a hung load freezes it."""
    app, gui = _probe_gui(genai)

    with patch.object(app, "inject_recovered_model") as inject, \
         patch.object(gui, "_schedule_npu_recovery") as recovery, \
         patch("debora_whisper.app.probe_npu", side_effect=RuntimeError("no answer in 300s")):
        gui._run_npu_recovery_probe()

    assert genai.loads == []
    inject.assert_not_called()
    recovery.assert_called_once()


def test_hung_probe_process_is_killed_after_the_timeout():
    from debora_whisper import npu_probe
    import time
    start = time.time()
    with pytest.raises(RuntimeError, match="no answer in 1s"):
        npu_probe.probe_npu({}, None, timeout=1,
                            command=[sys.executable, "-c", "import time; time.sleep(60)"])
    assert time.time() - start < 10


def test_probe_process_failure_reports_its_last_line():
    from debora_whisper import npu_probe
    with pytest.raises(RuntimeError, match="^probe model loaded on CPU, not NPU$"):
        npu_probe.probe_npu({}, None, command=[
            sys.executable, "-c",
            "print('Loading...'); print('probe model loaded on CPU, not NPU'); exit(1)"])
    npu_probe.probe_npu({}, None, command=[sys.executable, "-c", "pass"])


def test_probe_process_runs_the_model_on_the_npu(genai, tmp_path):
    from debora_whisper import npu_probe
    genai.generate_error = None
    assert npu_probe.run(str(tmp_path), "whisper", "base", "pt", 16000) == 0
    assert genai.loads == ["NPU"]


def test_probe_process_fails_when_the_model_lands_elsewhere(genai, tmp_path, capsys):
    from debora_whisper import npu_probe
    genai.load_errors = {"NPU": "remove: The process cannot access the file"}
    assert npu_probe.run(str(tmp_path), "whisper", "base", "pt", 16000) == 1
    assert capsys.readouterr().out.strip().endswith("probe model loaded on CPU, not NPU")


def test_probe_command_runs_the_probe_module():
    from debora_whisper import npu_probe
    command = npu_probe.probe_command({**DEFAULT_CONFIG, "model_size": "turbo",
                                       "language": "pt"}, "D:/m")
    assert command[:3] == [sys.executable, "-m", "debora_whisper.npu_probe"]
    assert command[3:] == ["D:/m", "whisper", "turbo", "pt", str(DEFAULT_CONFIG["sample_rate"])]


class QueuedRoot(Root):
    """Tk root whose after() callbacks run only when the test says so."""

    def __init__(self):
        self.pending = []

    def after(self, delay, fn, *args):
        self.pending.append((delay, fn, args))
        return "after-id"

    def run_pending(self):
        pending, self.pending = self.pending, []
        for _delay, fn, args in pending:
            fn(*args)


def test_recovered_npu_swap_retries_until_engine_is_idle(genai):
    app, gui = _probe_gui(genai)
    gui._root = QueuedRoot()
    recovered = object()

    with patch.object(app, "inject_recovered_model", side_effect=[False, True]) as inject:
        gui._swap_to_recovered_npu(app, recovered)
        assert app.config["device"] == "CUDA"
        assert [d for d, _, _ in gui._root.pending] == [gui.SWAP_RETRY_MS]

        gui._root.run_pending()

    assert inject.call_count == 2
    assert app.config["device"] == "NPU"
    assert gui._root.pending == []


def test_recovered_npu_swap_dropped_after_engine_rebuild(genai):
    app, gui = _probe_gui(genai)
    gui._engine = _engine({"device": "CUDA"})

    with patch.object(app, "inject_recovered_model") as inject:
        gui._swap_to_recovered_npu(app, object())

    inject.assert_not_called()


def test_continuous_listening_swap_not_blocked_by_open_microphone():
    app = _engine({"device": "CUDA", "continuous_listening": True})
    app.is_recording = True

    app._transcribing = True
    assert app.inject_recovered_model("NPU", object()) is False

    app._transcribing = False
    recovered = object()
    assert app.inject_recovered_model("NPU", recovered) is True
    assert app.whisper is recovered
    assert app.config["device"] == "NPU"


def test_push_to_talk_swap_waits_for_recording_to_end():
    app = _engine({"device": "CUDA"})
    app.is_recording = True

    assert app.inject_recovered_model("NPU", object()) is False


def test_classification_uses_backend_devices_not_config():
    lost = RuntimeError("DEVICE_LOST")
    # Configured NPU, but Parakeet's decoder ran on GPU: fail closed.
    parakeet = de.ParakeetNPU.__new__(de.ParakeetNPU)
    parakeet.device, parakeet.dec_device = "NPU", "GPU"
    assert de.classify_device_failure(lost, parakeet.active_devices()) == "UNKNOWN"
    parakeet.dec_device = "CPU"
    assert de.classify_device_failure(lost, parakeet.active_devices()) == "NPU"
    # Whisper that fell back to CPU at load is not blamed on a GPU/NPU name.
    assert de.classify_device_failure(RuntimeError(CL_ERROR), {"CPU"}) == "GPU"
    assert de.classify_device_failure(RuntimeError("bad input shape"), {"GPU"}) is None
    # The cause chain is searched, not only the outermost message.
    try:
        try:
            raise RuntimeError(CL_ERROR)
        except RuntimeError as inner:
            raise ValueError("wrapped") from inner
    except ValueError as outer:
        assert de.classify_device_failure(outer, {"NPU"}) == "GPU"


def test_parakeet_inference_failure_is_attributed_to_gpu_decoder():
    parakeet = de.ParakeetNPU.__new__(de.ParakeetNPU)
    parakeet.device, parakeet.dec_device = "NPU", "GPU"
    parakeet._preprocess = lambda audio: (np.zeros((1, 128, 100), np.float32), None)
    parakeet.enc_compiled = {b: (lambda inputs: {
        "outputs": np.zeros((1, 1024, 10), np.float32),
        "encoded_lengths": np.array([10]),
    }) for b in de.ParakeetNPU.MEL_BUCKETS}

    def decoder(inputs):
        raise RuntimeError("[GPU] DEVICE_LOST while executing decoder")

    parakeet.dec_compiled = decoder
    parakeet.vocab = {}
    with pytest.raises(de.DeviceFailureError) as info:
        parakeet.transcribe(np.zeros(16000, np.float32))
    assert info.value.device == "GPU"


# -- A lost NPU across runs ----------------------------------------------------
# Loading on an NPU lost in an earlier run hung inside the driver while holding
# the GIL, which froze the next run (console and Ctrl+C included) until a reboot.

def _priority_config(device="NPU"):
    return {**DEFAULT_CONFIG, "device": device, "model_size": "turbo",
            "device_priority": ["CUDA", "NPU", "GPU", "CPU"]}


def _lose_npu(engine_device="NPU"):
    """An NPU DEVICE_LOST on the engine's real error path."""
    app = _engine({"device": engine_device})
    return app._error_payload(RuntimeError(
        "L0 zeFenceHostSynchronize result: ZE_RESULT_ERROR_DEVICE_LOST"))


def test_npu_loss_is_remembered_until_windows_restarts(monkeypatch):
    monkeypatch.setattr(de, "_boot_time", lambda: 1000.0)
    assert _lose_npu()["device_failure"] == "NPU"
    assert de.npu_lost_this_boot()["boot"] == 1000.0

    monkeypatch.setattr(de, "_boot_time", lambda: 5000.0)  # rebooted
    assert de.npu_lost_this_boot() is None
    assert not de.NPU_LOST_FILE.exists()


def test_gpu_loss_is_not_remembered():
    app = _engine({"device": "GPU"})
    assert app._error_payload(RuntimeError(CL_ERROR))["device_failure"] == "GPU"
    assert not de.NPU_LOST_FILE.exists()


def test_next_run_skips_a_lost_npu(monkeypatch):
    monkeypatch.setattr(de, "_boot_time", lambda: 1000.0)
    monkeypatch.setattr(de, "detect_devices", lambda: {"NPU", "GPU", "CPU"})
    _lose_npu()

    config = _priority_config()
    de.avoid_lost_npu(config)
    assert config["device"] == "GPU"


def test_healthy_npu_is_kept(monkeypatch):
    monkeypatch.setattr(de, "_boot_time", lambda: 1000.0)
    config = _priority_config()
    de.avoid_lost_npu(config)
    assert config["device"] == "NPU"


def test_large_logs_are_rotated(tmp_path, monkeypatch):
    # Its own folder: threads of other tests may still write the shared one.
    for name in ("LOG_FILE", "TELEMETRY_LOG", "TTS_SERVER_LOG", "LLM_SERVER_LOG"):
        monkeypatch.setattr(de, name, tmp_path / getattr(de, name).name)
    de.LOG_FILE.write_bytes(b"x" * (de.LOG_MAX_BYTES + 1))
    de.TELEMETRY_LOG.write_bytes(b"small")
    de.rotate_logs()
    assert not de.LOG_FILE.exists()
    assert (tmp_path / "app.log.1").stat().st_size == de.LOG_MAX_BYTES + 1
    assert de.TELEMETRY_LOG.read_bytes() == b"small"


def test_lines_logged_from_many_threads_stay_whole(tmp_path, monkeypatch):
    import threading
    monkeypatch.setattr(de, "LOG_FILE", tmp_path / "app.log")
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)

    def write(n):
        for i in range(50):
            de.log(f"thread {n} line {i} " + "x" * (n * 7))

    threads = [threading.Thread(target=write, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = de.LOG_FILE.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 400
    assert all(line.startswith("[") and " thread " in line for line in lines)


def test_tests_never_write_the_real_app_log():
    from debora_whisper import paths
    assert paths.LOG_DIR not in de.LOG_FILE.parents


def test_startup_after_an_npu_loss_probes_it_in_the_background(monkeypatch):
    from debora_whisper import app as app_module
    monkeypatch.setattr(de, "_boot_time", lambda: 1000.0)
    monkeypatch.setattr(de, "detect_devices", lambda: {"NPU", "GPU", "CPU"})
    de.remember_npu_loss("DEVICE_LOST")
    created = []

    class FakeGUI:
        def __init__(self, config):
            self.config, self.scheduled = config, 0
            self._engine = types.SimpleNamespace()
            created.append(self)

        def _schedule_npu_recovery(self):
            self.scheduled += 1

        def run(self):
            pass

    monkeypatch.setattr(app_module, "GUIApp", FakeGUI)
    monkeypatch.setattr(app_module, "_claim_single_instance", lambda: True)
    monkeypatch.setattr(app_module, "load_config", lambda: _priority_config())
    # This tests recovery scheduling, not the independent inventory worker.
    # A real worker can outlive this test and call a later test's patched planner.
    inventory = MagicMock()
    monkeypatch.setattr(app_module, "log_hardware_inventory", inventory)
    monkeypatch.setattr(sys, "argv", ["debora", "--device", "NPU"])
    app_module.main()

    inventory.assert_called_once()
    assert created[0].config["device"] == "GPU"
    assert created[0].scheduled == 1
