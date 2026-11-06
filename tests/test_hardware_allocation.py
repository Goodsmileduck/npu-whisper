"""Actual placements reach the main console, including worker identities."""
import os
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from debora_whisper import hardware_allocation as hw
from debora_whisper import dictation_engine as de


class Core:
    available_devices = ["CPU", "GPU.0", "GPU.1", "NPU"]

    def get_property(self, device, prop):
        if prop == "DEVICE_ID":
            return "0"
        return {"CPU": "Intel CPU", "GPU.0": "Intel Arc (iGPU)",
                "GPU.1": "NVIDIA RTX (dGPU)", "NPU": "Intel AI Boost"}[device]


def test_summary_devices_pids_residency_and_mode_changes(monkeypatch):
    monkeypatch.setattr(hw, "cpu_target", lambda: "CPU Intel CPU")
    config, logs = {"voice_chat": False}, []
    reporter = hw.AllocationReporter()
    reporter.configure(config, logs.append)
    reporter.update("STT", "turbo (faster-whisper) -> CUDA:0 NVIDIA RTX  pid 101")
    assert "Hardware allocation (mode=dictation):" in logs[-1]
    assert "LLM  inactive in dictation" in logs[-1]
    assert "DRAFT STT  shares STT model and pid" in logs[-1]
    reporter.update("TTS", "Chatterbox -> CUDA:0 NVIDIA RTX  server pid 202")
    assert "inactive in dictation; resident: Chatterbox" in logs[-1]
    count = len(logs)
    reporter.emit()
    reporter.update("TTS", "Chatterbox -> CUDA:0 NVIDIA RTX  server pid 202")
    assert len(logs) == count
    config.update(voice_chat=True, voice_chat_backend="claude")
    reporter.emit()
    assert len(logs) == count + 1
    assert "mode=chat_claude" in logs[-1]
    assert "external Claude Code" in logs[-1]
    assert "server pid 202" in logs[-1]


def test_openvino_alias_and_loaded_graph_disambiguate_gpus():
    assert hw.ov_target(Core(), "GPU") == "OV:GPU.0 Intel Arc (iGPU)"
    graph = SimpleNamespace(get_property=lambda prop: ["GPU.1"])
    assert hw.ov_target(Core(), "GPU", graph) == "OV:GPU.1 NVIDIA RTX (dGPU)"


def test_missing_identity_is_explicit_not_guessed():
    core = SimpleNamespace(get_property=Mock(side_effect=RuntimeError("unsupported")))
    assert "unavailable" in hw.ov_target(core, "GPU")
    assert "GPU.0" not in hw.ov_target(core, "GPU")


def test_load_time_fallback_uses_actual_whisper_device(monkeypatch, tmp_path):
    calls = []

    def pipeline(path, device, **kwargs):
        calls.append(device)
        if device == "NPU":
            raise RuntimeError("unsupported operation")
        return object()

    monkeypatch.setitem(sys.modules, "openvino_genai", SimpleNamespace(WhisperPipeline=pipeline))
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=Core))
    monkeypatch.setattr(de, "CACHE_DIR", tmp_path)
    model = de.WhisperNPU(tmp_path, "NPU")
    detail = hw.model_description(model, "turbo", "NPU")
    assert calls == ["NPU", "CPU"]
    assert "-> OV:CPU Intel CPU" in detail
    assert "planned NPU, fell back to OV:CPU Intel CPU: NPU: unsupported operation" in detail
    assert f"pid {os.getpid()}" in detail


def test_runtime_device_loss_then_reload_keeps_original_reason(monkeypatch):
    config = {**de.DEFAULT_CONFIG, "device": "NPU"}
    logs = []
    hw.allocation.configure(config, logs.append)
    app = de.DictationApp.__new__(de.DictationApp)
    app.config = config
    app.whisper = SimpleNamespace(device="NPU")
    app._model_lock = threading.RLock()
    app._model_ready = threading.Event()
    app._quarantined_models = []
    app._lost_device = None
    app._load_error = None
    monkeypatch.setattr(app, "_start_loader", lambda: None)
    monkeypatch.setattr(de, "setup_model", lambda config: Path("model"))
    monkeypatch.setattr(de, "create_model", lambda *a, **k: SimpleNamespace(
        device="GPU", hardware_target="OV:GPU.0 Intel Arc (iGPU)",
        hardware_backend="openvino-genai", hardware_fallbacks=[]))
    app._error_payload(de.DeviceFailureError("NPU", RuntimeError("ZE_RESULT_ERROR_DEVICE_LOST")))
    assert "device lost/quarantined" in logs[-1]
    app.fallback_device("GPU")
    app.ensure_model()
    assert "planned NPU, fell back to OV:GPU.0" in logs[-1]
    assert "ZE_RESULT_ERROR_DEVICE_LOST" in logs[-1]
    count = len(logs)
    app.ensure_model()
    assert len(logs) == count


def test_optional_device_identity_failure_does_not_reload_whisper(monkeypatch, tmp_path):
    pipeline = Mock(return_value=object())
    monkeypatch.setitem(sys.modules, "openvino_genai", SimpleNamespace(WhisperPipeline=pipeline))
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=Mock(side_effect=RuntimeError("query failed"))))
    monkeypatch.setattr(de, "CACHE_DIR", tmp_path)
    model = de.WhisperNPU(tmp_path, "GPU")
    assert model.device == "GPU"
    assert "identity unreported" in model.hardware_target
    pipeline.assert_called_once()


@pytest.mark.parametrize("cuda_available,identity_fails", [(False, False), (True, False), (True, True)])
def test_tts_loader_reports_its_own_device_and_fallback(monkeypatch, cuda_available, identity_fails):
    from debora_whisper import tts_server
    from types import ModuleType
    model = SimpleNamespace(conds=None)
    torch = ModuleType("torch")
    torch._dynamo = ModuleType("torch._dynamo")
    torch.cuda = SimpleNamespace(is_available=lambda: cuda_available, current_device=lambda: 1,
        get_device_name=Mock(side_effect=RuntimeError("query failed")) if identity_fails else lambda i: "RTX worker GPU")
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch._dynamo", torch._dynamo)
    monkeypatch.setitem(sys.modules, "chatterbox", ModuleType("chatterbox"))
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts",
                        SimpleNamespace(ChatterboxMultilingualTTS=SimpleNamespace(from_pretrained=factory)))
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setattr(tts_server, "_stub_sklearn", lambda: None)
    tts_server.load_model("cuda", None)
    assert model._debora_allocation["pid"] == os.getpid()
    if not cuda_available:
        assert model._debora_device == "CPU"
        assert model._debora_allocation["device_id"] == "CPU"
        assert "CUDA unavailable" in model._debora_allocation["fallback_reason"]
    elif identity_fails:
        assert model._debora_device == "CUDA"
        assert "unreported" in model._debora_allocation["device_id"]
    else:
        assert model._debora_allocation["device_id"] == "CUDA:1"
        assert model._debora_allocation["device_name"] == "RTX worker GPU"
    factory.assert_called_once_with(device="cuda" if cuda_available else "cpu")


def test_parakeet_reports_mixed_encoder_buckets_and_decoder_fallback(monkeypatch, tmp_path):
    import numpy as np

    class Graph:
        def __init__(self, device):
            self.device = device

        def get_property(self, prop):
            return ["GPU.1" if self.device == "GPU" else self.device]

        def __call__(self, inputs):
            return {"outputs": np.zeros((1, 1024, 3))}

    class Runtime(Core):
        encoder_loads = 0

        def read_model(self, path):
            return SimpleNamespace(decoder="decoder" in path, reshape=lambda shape: None)

        def compile_model(self, model, device, config=None):
            if model.decoder and device == "GPU":
                raise RuntimeError("decoder unsupported")
            if not model.decoder and device == "NPU":
                self.encoder_loads += 1
                if self.encoder_loads > 1:
                    raise RuntimeError("bucket unsupported")
            return Graph(device)

    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=Runtime))
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=lambda *a, **k: object()))
    monkeypatch.setattr(de.ParakeetNPU, "_load_vocab", lambda self: None)
    monkeypatch.setattr(de.ParakeetNPU, "_validate_vocab_against_decoder", lambda self: None)
    monkeypatch.setattr(de.ParakeetNPU, "MEL_BUCKETS", (10, 20))
    monkeypatch.setattr(de, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "nemo128.onnx").touch()
    model = de.ParakeetNPU(tmp_path, "NPU")
    detail = hw.model_description(model, "parakeet", "NPU")
    assert "bucket 10=OV:NPU Intel AI Boost" in detail
    assert "bucket 20=OV:GPU.1 NVIDIA RTX (dGPU)" in detail
    assert "decoder OV:CPU Intel CPU" in detail
    assert "mel preprocessor CPU" in detail
    assert "bucket unsupported" in detail and "decoder unsupported" in detail


@pytest.mark.parametrize("adapter", [False, True])
def test_tts_health_uses_server_pid_not_launcher_and_is_deduplicated(monkeypatch, adapter):
    from debora_whisper import tts_server, voice_chat
    model = SimpleNamespace(sr=24000, conds=None, _debora_device="CPU",
        _debora_allocation={"pid": 4321, "device_id": "CPU", "device_name": "Intel CPU",
                            "requested_device": "CUDA", "fallback_reason": "CUDA unavailable in TTS worker"})
    if adapter:
        model.package = "torch"
        model.model_id = "XTTS-v2"
        model.precision = "float32"
        model.generate_stream = lambda *a, **kw: iter(())
    server = ThreadingHTTPServer(("127.0.0.1", 0), tts_server.make_handler(model, "pt", adapter=adapter))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = {"voice_chat": True, "tts_url": f"http://127.0.0.1:{server.server_port}"}
    logs = []
    hw.allocation.configure(config, logs.append)
    monkeypatch.setattr(voice_chat, "_tts_process", SimpleNamespace(pid=9999))
    try:
        assert voice_chat.tts_server_up(config)
        assert ("XTTS-v2" if adapter else "ChatterboxMultilingualTTS") in logs[-1]
        assert voice_chat._tts_streams(config) is adapter
        assert voice_chat._tts_capabilities(config)["ok"] is True
        assert not voice_chat._tts_streams({**config, "tts_stream": False})
        assert "server pid 4321" in logs[-1]
        assert "9999" not in logs[-1]
        assert "planned CUDA, fell back to CPU: CUDA unavailable in TTS worker" in logs[-1]
        assert voice_chat.tts_server_up(config)
        assert len(logs) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)


def test_external_tts_never_borrows_local_hardware():
    logs = []
    config = {"voice_chat": True, "tts_url": "https://voice.example"}
    hw.allocation.configure(config, logs.append)
    hw.report_tts(config, None)
    assert "external endpoint=https://voice.example; device/model/pid unreported" in logs[-1]
    assert "CUDA" not in logs[-1]


def test_claude_protocol_reports_actual_external_model_and_harness_pid():
    import io
    import queue
    from debora_whisper import harness
    logs = []
    hw.allocation.configure({"voice_chat": True, "voice_chat_backend": "claude"}, logs.append)
    session = harness.HarnessSession.__new__(harness.HarnessSession)
    session._allocation_owner = hw.allocation.claim("CLAUDE")
    session.process = SimpleNamespace(pid=7654, stdout=io.StringIO(
        '{"type":"system","subtype":"init","model":"claude-test"}\n'))
    session._events = queue.Queue()
    session.error = None
    session.log = logs.append
    session._read()
    assert "external Claude Code; model=claude-test" in logs[0]
    assert "harness pid 7654 (CPU orchestration)" in logs[0]
    assert "endpoint=Claude Code managed" in logs[0]
    assert "harness exited" in logs[-1]


def test_summary_console_and_app_log_match(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(de, "LOG_DIR", tmp_path)
    monkeypatch.setattr(de, "LOG_FILE", tmp_path / "app.log")
    block = hw.format_summary({}, {"STT": "turbo -> CUDA:0 NVIDIA RTX  pid 1234"})
    de.log(block)
    console = capsys.readouterr().out
    assert block in console
    assert console == de.LOG_FILE.read_text(encoding="utf-8")


def test_closed_console_does_not_drop_app_log(tmp_path, monkeypatch):
    monkeypatch.setattr(de, "LOG_DIR", tmp_path)
    monkeypatch.setattr(de, "LOG_FILE", tmp_path / "app.log")
    monkeypatch.setattr("builtins.print", Mock(side_effect=OSError("closed")))
    de.log("Hardware allocation (mode=dictation):")
    assert "Hardware allocation" in de.LOG_FILE.read_text(encoding="utf-8")


def test_npu_probe_forwards_temporary_worker_allocation(monkeypatch):
    import json
    from debora_whisper import npu_probe
    logs = []
    detail = "turbo (openvino-genai) -> OV:NPU Intel AI Boost  pid 8765"
    process = SimpleNamespace(returncode=0, pid=8765,
        communicate=lambda **kw: (("@allocation " + json.dumps(detail) + "\n").encode(), None))
    monkeypatch.setattr(npu_probe.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(de, "log", logs.append)
    npu_probe.probe_npu({}, "model", command=["fake"])
    assert "mode=npu_probe" in logs[-1]
    assert detail in logs[-1]
    assert "temporary worker now exited" in logs[-1]


@pytest.mark.parametrize("component", ["LLM", "CLAUDE"])
def test_delayed_worker_reader_cannot_replace_or_remove_current_allocation(monkeypatch, tmp_path, component):
    import io
    import json
    from debora_whisper import harness, voice_chat

    logs = []
    config = {**de.DEFAULT_CONFIG, "voice_chat": True, "harness_cwd": str(tmp_path)}
    hw.allocation.configure(config, logs.append)
    real_thread = threading.Thread
    readers = []

    def track_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        readers.append(thread)
        return thread

    monkeypatch.setattr(threading, "Thread", track_thread)
    monkeypatch.setattr(harness, "_run_sessions", {})
    monkeypatch.setattr(harness.paths, "CONFIG_DIR", tmp_path)
    releases = []
    workers = []

    def start(pid):
        ready, release = threading.Event(), threading.Event()
        releases.append(release)
        message = ({"ready": "CPU", "allocation": {"target": "CPU test CPU"}}
                   if component == "LLM" else
                   {"type": "system", "subtype": "init", "model": f"claude-{pid}"})

        def stdout():
            yield json.dumps(message)
            ready.set()
            assert release.wait(10)
            # Buffered startup output can also arrive after the replacement.
            yield json.dumps(message)

        process = SimpleNamespace(pid=pid, stdout=stdout(), stderr=io.StringIO(),
                                  stdin=io.StringIO(), wait=Mock(return_value=0), poll=lambda: None)
        if component == "LLM":
            monkeypatch.setattr(voice_chat.subprocess, "Popen", lambda *a, **kw: process)
            worker = voice_chat.LLMProcess((f"model-{pid}", "CPU"), ["fake"], logs.append)
        else:
            worker = harness.HarnessSession(config, logs.append,
                process_factory=lambda *a, **kw: process,
                command_factory=lambda *a: ["fake"], session_file=tmp_path / "session.json")
        workers.append(worker)
        assert ready.wait(5)
        return worker

    try:
        old = start(1111)
        old.stop()  # Process wait completes while stdout's reader is still blocked.
        old_readers = list(readers)
        current = start(2222)
        expected = hw.allocation.rows[component]
        assert "2222" in expected
        count = len(logs)
        releases[0].set()
        for reader in old_readers:
            reader.join(5)
            assert not reader.is_alive()
        assert hw.allocation.rows[component] == expected
        assert not any(line.startswith("Hardware allocation") for line in logs[count:])
        old.stop()  # A delayed stop/removal must also respect ownership.
        assert hw.allocation.rows[component] == expected
        current.stop()
        assert component not in hw.allocation.rows
        releases[1].set()
        for reader in readers:
            reader.join(5)
        assert component not in hw.allocation.rows  # Stopped readers cannot resurrect rows.
    finally:
        for release in releases:
            release.set()
        for reader in readers:
            reader.join(5)
        for worker in workers:
            worker.stop()


@pytest.mark.parametrize("loaded,status", [(True, "ok"), (False, "ok"), (True, "unknown")])
def test_dictation_startup_reports_only_confirmed_resident_tts(monkeypatch, loaded, status):
    from debora_whisper import device_inventory, voice_chat

    config = {**de.DEFAULT_CONFIG, "voice_chat": False}
    logs = []
    hw.allocation.configure(config, logs.append)
    observed = {"tts": {"schema_version": 1, "status": status, "loaded": loaded,
        "device": "CUDA", "device_id": "CUDA:0", "device_name": "Resident GPU",
        "model": "Resident Chatterbox", "pid": 4321, "worker_pid": 9876, "versions": {}}}
    inventory = {**device_inventory.make_inventory(observed),
                 "timed_out": False, "elapsed_ms": 0.0}
    monkeypatch.setattr(device_inventory, "probe_inventory", lambda *a, **kw: inventory)
    monkeypatch.setattr(de, "log", logs.append)
    ensure = Mock(side_effect=AssertionError("Inventory must not start TTS"))
    monkeypatch.setattr(voice_chat, "ensure_tts_server", ensure)
    thread = de.log_hardware_inventory(config)
    thread.join(5)
    assert not thread.is_alive()
    assert not any("failed:" in line.lower() for line in logs)
    summaries = [line for line in logs if line.startswith("Hardware allocation")]
    assert "mode=dictation" in summaries[-1]
    if loaded and status == "ok":
        assert "TTS  inactive in dictation; resident: Resident Chatterbox" in summaries[-1]
        assert "CUDA:0 Resident GPU server pid 4321" in summaries[-1]
        assert "9876" not in hw.allocation.rows["TTS"]
    else:
        assert "TTS" not in hw.allocation.rows
    ensure.assert_not_called()


@pytest.mark.parametrize("interleaved_log", [False, True])
@pytest.mark.parametrize("inventory", [None, {"loaded": True, "model": "Chatterbox", "pid": 4321}])
@pytest.mark.parametrize("endpoint,origin", [
    ("https://user-secret:pass-secret@voice.example:8443/path-secret?token=query-secret#fragment-secret",
     "https://voice.example:8443"),
    ("http://user-secret:pass-secret@[::1]:8765/path-secret?token=query-secret#fragment-secret",
     "http://[::1]:8765"),
    ("https://voice.example:port-secret/path-secret", "invalid endpoint"),
    ("https://[invalid-secret/path-secret", "invalid endpoint"),
])
def test_tts_endpoint_secrets_never_reach_console_or_app_log(monkeypatch, tmp_path, capsys, inventory,
                                                           endpoint, origin, interleaved_log):
    monkeypatch.setattr(de, "LOG_DIR", tmp_path)
    monkeypatch.setattr(de, "LOG_FILE", tmp_path / "app.log")
    config = {"voice_chat": True, "tts_url": endpoint}
    hw.allocation.configure(config, de.log)
    hw.allocation.emit()  # Before any successful health check.
    hw.report_tts(config, inventory)
    output = capsys.readouterr().out
    if interleaved_log:
        # Deterministically reproduce a log arriving after console capture.
        de.log("Unrelated background activity")
    file_output = de.LOG_FILE.read_text(encoding="utf-8")
    # Other threads may log between capture and read; only compare our rows.
    def endpoint_lines(text):
        return [line.split("] ", 1)[-1] for line in text.splitlines() if "endpoint=" in line]

    assert endpoint_lines(output) == endpoint_lines(file_output)
    for text in (output, file_output):
        assert text.count(f"endpoint={origin}") == 2
        assert "secret" not in text
