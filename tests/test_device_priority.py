"""config["device_priority"] picks the startup device and the fallback."""
import pytest

from debora_whisper import dictation_engine as de

ALL = {"CUDA", "NPU", "GPU", "CPU"}


def _cfg(model="turbo", priority=("CUDA", "NPU", "GPU", "CPU")):
    return {**de.DEFAULT_CONFIG, "model_size": model, "device_priority": list(priority)}


def test_default_priority_puts_rtx_first():
    assert de.DEFAULT_CONFIG["device_priority"][:2] == ["CUDA", "NPU"]


def test_first_present_device_wins():
    assert de.select_device(_cfg(), ALL) == "CUDA"
    assert de.select_device(_cfg(), {"NPU", "GPU", "CPU"}) == "NPU"


def test_failed_device_falls_back_to_next_in_priority():
    assert de.select_device(_cfg(), ALL, exclude={"CUDA"}) == "NPU"
    assert de.select_device(_cfg(), ALL, exclude={"NPU"}) == "CUDA"


def test_user_order_is_respected():
    assert de.select_device(_cfg(priority=["NPU", "CUDA"]), ALL) == "NPU"


def test_parakeet_skips_cuda():
    assert de.select_device(_cfg(model="parakeet"), ALL) == "NPU"


def test_nothing_usable_returns_none():
    assert de.select_device(_cfg(priority=["CUDA"]), {"CPU"}) is None


def test_apply_device_priority_sets_device(monkeypatch):
    monkeypatch.setattr(de, "detect_devices", lambda: {"NPU", "CPU"})
    config = _cfg()
    de.apply_device_priority(config)
    assert config["device"] == "NPU"


@pytest.mark.parametrize("priority", [[], ["RTX"], "CUDA"])
def test_invalid_priority_is_rejected(priority):
    config = {**_cfg(), "device_priority": priority}
    with pytest.raises(ValueError):
        de.validate_config(config)


@pytest.mark.parametrize("has_backend", [True, False])
def test_cuda_requires_the_cuda_extra(monkeypatch, has_backend):
    # A plain install on an NVIDIA box has no faster-whisper: CUDA must not be
    # offered, or the default priority would pick it and fail to load.
    import importlib.util
    import sys
    import types
    monkeypatch.setattr(de, "has_nvidia_gpu", lambda return_name=False: True)
    monkeypatch.setitem(sys.modules, "openvino", types.SimpleNamespace(
        Core=lambda: types.SimpleNamespace(available_devices=["CPU", "NPU"])))
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: (
        (object() if has_backend else None) if name == "faster_whisper"
        else real_find_spec(name, *a)))
    monkeypatch.setattr(de, "import_faster_whisper", lambda: object)
    devices = de.detect_devices()
    assert ("CUDA" in devices) is has_backend
    assert de.select_device(_cfg(), devices) == ("CUDA" if has_backend else "NPU")


def _nvidia_with_faster_whisper(monkeypatch):
    import importlib.util
    import sys
    import types
    monkeypatch.setattr(de, "has_nvidia_gpu", lambda return_name=False: True)
    monkeypatch.setitem(sys.modules, "openvino", types.SimpleNamespace(
        Core=lambda: types.SimpleNamespace(available_devices=["CPU", "NPU"])))
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: (
        object() if name == "faster_whisper" else real_find_spec(name, *a)))


def test_cuda_backend_that_cannot_load_falls_back(monkeypatch):
    # Installed but blocked (e.g. Windows Application Control on a DLL):
    # start on the next device instead of failing to load CUDA.
    _nvidia_with_faster_whisper(monkeypatch)

    def blocked():
        raise RuntimeError("faster-whisper cannot be loaded: DLL load failed")
    monkeypatch.setattr(de, "import_faster_whisper", blocked)
    devices = de.detect_devices()
    assert "CUDA" not in devices
    assert de.select_device(_cfg(), devices) == "NPU"


def test_faster_whisper_loads_without_pyav(monkeypatch):
    # PyAV only decodes audio files; a blocked PyAV must not disable CUDA.
    import sys
    import types
    monkeypatch.setitem(sys.modules, "av", None)  # `import av` raises ImportError
    model = object()
    monkeypatch.setitem(sys.modules, "faster_whisper",
                        types.SimpleNamespace(WhisperModel=model))
    assert de.import_faster_whisper() is model
    assert isinstance(sys.modules["av"], types.ModuleType)


def test_faster_whisper_load_error_names_the_cause(monkeypatch):
    import sys
    import types
    monkeypatch.setitem(sys.modules, "av", types.ModuleType("av"))
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(RuntimeError, match="cannot be loaded"):
        de.import_faster_whisper()


# Startup inventory is observational: none of these capabilities feed select_device.
from debora_whisper import device_inventory as inventory
from debora_whisper import device_queries as queries


def _observed():
    return {
        "system": {"devices": [inventory._cpu()]},
        "nvidia": {"status": "ok", "devices": [
            {"id": "CUDA:0", "kind": "CUDA", "index": 0, "physical_id": "GPU-aaa",
             "name": "RTX", "driver": "560.1", "memory_total": 8000, "memory_free": 6000}]},
        "openvino": {"status": "ok", "devices": [
            {"id": "OV:CPU", "kind": "CPU", "index": "CPU", "name": "CPU", "driver": None},
            {"id": "OV:NPU", "kind": "NPU", "index": "NPU", "physical_id": "npu-id",
             "name": "AI Boost", "driver": "32.1", "memory_total": None, "memory_free": None}]},
        "environment": {"versions": {"openvino": "1", "openvino-genai": "1",
                                       "faster-whisper": "1", "ctranslate2": "1", "onnxruntime": "1"},
                        "artifacts": {
                            "openvino": {"model": "OpenVINO/whisper-turbo-int8-ov", "precision": "int8",
                                         "revision": "a", "files": [["weights.bin", 100, 42]], "complete": True},
                            "cuda": {"model": "vendor/whisper-turbo", "precision": "int8_float16", "complete": True},
                            "llm": {"model": "Qwen-int4-ov", "precision": "int4", "complete": True}},
                        "npu_loss": None, "loss_checked": True},
        "tts": {"status": "unknown", "versions": {}, "device": None},
    }


def _row(result, component, device, backend=None):
    return next(r for r in result["capabilities"] if r["component"] == component
                and r["device"] == device and (backend is None or r["backend"] == backend))


def _validated_cache(observed, now=100):
    cached = inventory.make_inventory(observed, now=now)
    row = _row(cached, "stt.whisper", "OV:NPU", "openvino-genai")
    cached["validations"][row["id"]] = {"state": "verified", "checked_at": now}
    cached["measurements"] = {row["id"]: {"peak_bytes": 1024, "latency_ms": 10}}
    return cached


def test_inventory_nvidia_multiple_gpus_one_query(monkeypatch):
    calls = []
    # CI runners have no nvidia-smi anywhere.
    monkeypatch.setattr(queries, "nvidia_smi", lambda: "nvidia-smi.exe")

    def output(args, **kwargs):
        calls.append((args, kwargs))
        return (b'0, GPU-aaa, "NVIDIA, RTX", 560.1, 8192, 4096\n'
                b'1, GPU-bbb, RTX second, N/A, 16384, [Not Supported]\n')

    monkeypatch.setattr(queries.subprocess, "check_output", output)
    result = queries.nvidia_inventory(0.2)
    first, second = result["devices"]
    assert first["id"] == "CUDA:0" and second["id"] == "CUDA:1"
    assert first["physical_id"] == "GPU-aaa" and second["physical_id"] == "GPU-bbb"
    assert first["name"] == "NVIDIA, RTX"
    assert first["memory_free"] == 4096 * 1024 * 1024
    assert second["driver"] is None and second["memory_free"] is None
    assert len(calls) == 1 and calls[0][1]["timeout"] == 0.2
    assert "--query-gpu=index,uuid,name,driver_version,memory.total,memory.free" in calls[0][0]


@pytest.mark.parametrize("output", [None, "", "garbage", "not-an-index,u,n,d,1,2"])
def test_inventory_missing_or_invalid_nvidia_is_unknown(monkeypatch, output):
    monkeypatch.setattr(queries, "nvidia_output", lambda *a, **kw: output)
    assert queries.nvidia_inventory(0.1)["devices"] == []


def test_inventory_openvino_preserves_indices_and_unknown_properties(monkeypatch):
    import copy

    class Core:
        available_devices = ["CPU", "GPU.0", "GPU.1", "NPU"]

        def get_property(self, device, prop):
            if prop == "FULL_DEVICE_NAME":
                return "name " + device
            raise RuntimeError("unsupported")

    monkeypatch.setattr(queries, "openvino_core", Core)
    updates = []
    queries.openvino_inventory(lambda value: updates.append(copy.deepcopy(value)))
    assert [d["id"] for d in updates[0]["devices"]] == ["OV:CPU", "OV:GPU.0", "OV:GPU.1", "OV:NPU"]
    assert updates[0]["devices"][-1]["name"] == "NPU"  # published before properties
    assert updates[-1]["devices"][-1]["name"] == "name NPU"
    assert all(d["driver"] is None for d in updates[-1]["devices"])


@pytest.mark.parametrize("contents", ["not json", "[]", "null", '{"schema_version": 0}',
                                      '{"schema_version": 1, "created_at": "yesterday"}'])
def test_inventory_corrupt_or_old_cache_is_disposable(tmp_path, contents):
    path = tmp_path / "inventory.json"
    path.write_text(contents, encoding="utf-8")
    assert inventory.read_cache(path) is None


def test_inventory_cache_roundtrip_and_schema_invalidation(tmp_path):
    import json
    path = tmp_path / "inventory.json"
    original = _validated_cache(_observed())
    inventory.write_cache(path, original)
    assert inventory.read_cache(path) == original
    assert not list(tmp_path.glob("*.tmp"))
    original["schema_version"] += 1
    path.write_text(json.dumps(original), encoding="utf-8")
    assert inventory.read_cache(path) is None


@pytest.mark.parametrize("change", ["driver", "identity", "package", "revision", "precision", "files", "tts"])
def test_inventory_changed_identity_invalidates_validation_and_measurements(change):
    import copy
    observed = _observed()
    cached = _validated_cache(observed)
    current = copy.deepcopy(observed)
    if change == "driver":
        current["openvino"]["devices"][-1]["driver"] = "new"
    elif change == "identity":
        current["nvidia"]["devices"][0]["physical_id"] = "GPU-new"
    elif change == "package":
        current["environment"]["versions"]["ctranslate2"] = "new"
    elif change in ("revision", "precision", "files"):
        current["environment"]["artifacts"]["openvino"][change] = "new"
    else:
        current["tts"]["versions"]["torch"] = "different environment"
    result = inventory.make_inventory(current, cached, now=101)
    assert result["cache_status"] == "miss"
    assert result["validations"] == result["measurements"] == {}
    assert not _row(result, "stt.whisper", "OV:NPU", "openvino-genai")["ready"]


def test_inventory_policy_version_invalidates_cache(monkeypatch):
    observed = _observed()
    cached = _validated_cache(observed)
    monkeypatch.setattr(inventory, "POLICY_VERSION", inventory.POLICY_VERSION + 1)
    assert inventory.make_inventory(observed, cached, now=101)["cache_status"] == "miss"


def test_inventory_unknown_driver_validation_expires_and_does_not_slide():
    observed = _observed()
    observed["openvino"]["devices"][-1]["driver"] = None
    cached = _validated_cache(observed)
    recent = inventory.make_inventory(observed, cached, now=101)
    assert _row(recent, "stt.whisper", "OV:NPU", "openvino-genai")["ready"]
    assert recent["created_at"] == 100
    expired = inventory.make_inventory(observed, recent, now=100 + inventory.UNKNOWN_DRIVER_TTL)
    assert expired["validations"] == expired["measurements"] == {}
    assert not _row(expired, "stt.whisper", "OV:NPU", "openvino-genai")["ready"]


def test_inventory_never_reuses_cached_free_memory_or_presence():
    import copy
    observed = _observed()
    cached = _validated_cache(observed)
    current = copy.deepcopy(observed)
    current["nvidia"]["devices"][0]["memory_free"] = 50
    result = inventory.make_inventory(current, cached, now=101)
    assert result["cache_status"] == "hit"
    assert next(d for d in result["devices"] if d["id"] == "CUDA:0")["memory_free"] == 50
    del current["nvidia"]  # timed-out query: cached presence is not evidence
    result = inventory.make_inventory(current, cached, now=102)
    assert all(d["kind"] != "CUDA" for d in result["devices"])


def test_inventory_npu_loss_overrides_cached_health():
    observed = _observed()
    cached = _validated_cache(observed)
    observed["environment"]["npu_loss"] = {"time": 101, "detail": "DEVICE_LOST"}
    result = inventory.make_inventory(observed, cached, now=102)
    row = _row(result, "stt.whisper", "OV:NPU", "openvino-genai")
    assert row["validation"] == "quarantined" and not row["ready"]
    assert result["validations"] == result["measurements"] == {}


@pytest.mark.parametrize("stt_cuda,tts_device", [(True, "CPU"), (False, "CUDA")])
def test_inventory_stt_and_tts_environments_are_independent(stt_cuda, tts_device):
    observed = _observed()
    if not stt_cuda:
        observed["environment"]["versions"]["faster-whisper"] = None
    observed["tts"] = {"status": "ok", "device": tts_device, "loaded": True,
                       "versions": {"torch": "2.6.0+cu124", "chatterbox-tts": "0.1.7"}}
    result = inventory.make_inventory(observed)
    assert _row(result, "stt.whisper", "CUDA:0", "faster-whisper")["ready"] is stt_cuda
    assert _row(result, "tts", "TTS:" + tts_device)["ready"]
    assert _row(result, "tts", "TTS:" + tts_device)["physical_id"] is None
    other = "CPU" if tts_device == "CUDA" else "CUDA"
    assert not _row(result, "tts", "TTS:" + other)["ready"]


def test_inventory_parakeet_and_llm_have_no_cuda_backend():
    observed = _observed()
    observed["environment"]["artifacts"]["openvino"]["model"] = "vendor/parakeet-onnx"
    result = inventory.make_inventory(observed)
    for component in ("stt.parakeet.encoder", "stt.parakeet.decoder", "llm"):
        assert not _row(result, component, "CUDA:0")["supported"]
    assert not _row(result, "stt.parakeet.decoder", "OV:NPU")["supported"]
    assert _row(result, "stt.parakeet.encoder", "OV:NPU")["supported"]
    assert not _row(result, "stt.parakeet.encoder", "OV:NPU")["ready"]


def test_inventory_support_does_not_imply_local_artifacts():
    observed = _observed()
    observed["environment"]["artifacts"]["cuda"]["complete"] = False
    row = _row(inventory.make_inventory(observed), "stt.whisper", "CUDA:0", "faster-whisper")
    assert row["supported"] and row["dependencies_installed"]
    assert not row["ready"]


def test_inventory_artifact_metadata_detects_incomplete_weights(tmp_path):
    (tmp_path / "model.xml").write_text("graph", encoding="utf-8")
    required = ["model.xml", "model.bin"]
    before = inventory._artifact(tmp_path, required, model="export", precision="int8")
    assert not before["complete"]
    (tmp_path / "model.bin").write_bytes(b"weights")
    after = inventory._artifact(tmp_path, required, model="export", precision="int8")
    assert after["complete"] and before["files"] != after["files"]


def test_inventory_nothing_installed(monkeypatch, tmp_path):
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "openvino", None)
    monkeypatch.setattr(inventory, "_versions", lambda names: dict.fromkeys(names))
    monkeypatch.setattr(inventory, "_local_snapshot", lambda model: None)
    monkeypatch.setattr(inventory, "_faster_whisper_repo", lambda model: None)
    request = {"config": _cfg(), "model_info": de.MODEL_REGISTRY["turbo"],
               "model_dir": str(tmp_path), "npu_loss_file": str(tmp_path / "lost.json")}
    observed = {"environment": inventory._environment(request)}
    result = inventory.make_inventory(observed)
    assert [d["kind"] for d in result["devices"]] == ["CPU"]
    assert not any(row["ready"] for row in result["capabilities"])
    assert sys.modules["torch"] is None


@pytest.mark.parametrize("config", [{"tts_url": "https://tts.example"},
                                     {"tts_url": "http://localhost:8765", "tts_server_command": ["custom"]}])
def test_inventory_external_tts_stays_unknown(config, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("must not start a TTS server or resolve packages")
    monkeypatch.setattr(inventory.subprocess, "Popen", forbidden)
    result = inventory._tts(config)
    assert result["status"] == "unknown" and result["external"]


def test_inventory_tts_health_reports_actual_server_environment(monkeypatch):
    import importlib.metadata
    import json
    from debora_whisper import tts_server
    versions = {"torch": "server-torch", "chatterbox-tts": "server-chatterbox"}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    model = type("Model", (), {"sr": 24000, "conds": None, "_debora_device": "CPU"})()
    handler = tts_server.make_handler(model, "pt").__new__(tts_server.make_handler(model, "pt"))
    handler.path = "/health"
    replies = []
    handler._reply = lambda status, body: replies.append(json.loads(body))
    handler.do_GET()
    assert replies[0]["inventory"]["versions"] == versions
    assert replies[0]["inventory"]["device"] == "CPU"
    assert replies[0]["ok"]


@pytest.mark.parametrize("hung", ["openvino", "cache"])
def test_inventory_deadline_survives_hung_query_or_cache(monkeypatch, tmp_path, hung):
    import threading
    import time
    release = threading.Event()
    written = threading.Event()
    observed = _observed()

    def query(source, request, deadline, events, cancelled):
        if source == hung:
            # Native calls live in a child; this models the reader waiting
            # forever for that child, without running a real driver in tests.
            release.wait(5)
            return
        events.put((source, observed[source]))
        events.put((source, None))

    def read(path):
        if hung == "cache":
            release.wait(5)
        return None

    monkeypatch.setattr(inventory, "_query_process", query)
    monkeypatch.setattr(inventory, "read_cache", read)
    monkeypatch.setattr(inventory, "write_cache", lambda *args: written.set())
    start = time.monotonic()
    try:
        result = inventory.probe_inventory({}, tmp_path / "inventory.json", timeout=0.15)
        assert time.monotonic() - start < 0.75
        assert result["timed_out"]
        assert any(d["id"] == "CUDA:0" for d in result["devices"])
        assert written.wait(1)
    finally:
        release.set()


def test_inventory_slow_process_creation_and_cleanup_are_off_startup_thread(monkeypatch, tmp_path):
    import io
    import subprocess
    import threading
    import time
    from debora_whisper import processes
    release = threading.Event()
    killed = threading.Event()
    written = threading.Event()
    calls = []

    class HungProcess:
        stdin = io.BytesIO()
        stdout = io.BytesIO(b'{"devices": []}\n')

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("probe", timeout)

    def popen(command, **kwargs):
        calls.append(command)
        release.wait(5)  # slow process creation also consumes the common budget
        return HungProcess()

    def kill(process):
        killed.set()

    monkeypatch.setattr(inventory, "SOURCES", ("openvino",))
    monkeypatch.setattr(inventory.subprocess, "Popen", popen)
    monkeypatch.setattr(processes, "python_executable", lambda: "python.exe")
    monkeypatch.setattr(processes, "kill_tree", kill)
    monkeypatch.setattr(inventory, "write_cache", lambda *args: written.set())
    start = time.monotonic()
    result = inventory.probe_inventory({}, tmp_path / "inventory.json", timeout=0.1)
    assert time.monotonic() - start < 0.75
    assert result["timed_out"]
    assert calls[0][1:] == ["-m", "debora_whisper.device_inventory", "openvino"]
    release.set()
    assert killed.wait(1) and written.wait(1)


def test_inventory_logging_is_one_line_and_does_not_select(monkeypatch, tmp_path):
    import copy
    config = _cfg()
    before = copy.deepcopy(config)
    result = {**inventory.make_inventory(_observed()), "elapsed_ms": 123.4, "timed_out": False}
    monkeypatch.setattr(inventory, "probe_inventory", lambda *args, **kwargs: result)
    monkeypatch.setattr(de, "HARDWARE_INVENTORY", tmp_path / "inventory.json")
    logs = []
    monkeypatch.setattr(de, "log", logs.append)
    monkeypatch.setattr(de, "detect_devices", lambda: pytest.fail("inventory must not select"))
    de.log_hardware_inventory(config).join(5)
    assert config == before
    assert len([line for line in logs if line.startswith("Hardware inventory:")]) == 1
    assert all("\n" not in line for line in logs)
    assert "CUDA:0=RTX" in logs[0] and "probe=123.4ms" in logs[0]


def test_nvidia_smi_never_runs_from_the_current_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("SystemRoot", str(tmp_path / "no-windows"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(queries.shutil, "which", lambda name: str(tmp_path / "nvidia-smi.exe"))
    assert queries.nvidia_smi() is None
    elsewhere = tmp_path / "bin" / "nvidia-smi.exe"
    monkeypatch.setattr(queries.shutil, "which", lambda name: str(elsewhere))
    assert queries.nvidia_smi() == str(elsewhere.resolve())
