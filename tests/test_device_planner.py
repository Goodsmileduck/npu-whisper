"""Pure policy examples and resource admission, without hardware or user files."""

import copy
import threading

import pytest

from debora_whisper import device_inventory, device_planner, device_queries
from debora_whisper.device_planner import plan


def _config(**changes):
    return {"model_size": "turbo", "device": "auto", "llm_device": "auto", "tts_device": "auto",
            "device_priority": ["CUDA", "NPU", "GPU", "CPU"], "llm_model": "Qwen-8B-int4", **changes}


def _inventory(kinds=("CUDA", "NPU", "GPU"), *, cuda_free=14336, ram_free=49152, parakeet=False):
    def device(kind):
        return {"id": "CUDA:0" if kind == "CUDA" else "CPU" if kind == "CPU" else "OV:" + kind,
                "kind": kind, "index": 0 if kind == "CUDA" else kind, "physical_id": kind + "-uuid",
                "name": kind, "driver": "1", "memory_shared": kind != "CUDA",
                "memory_total": (16384 if kind == "CUDA" else 65536) * 2**20,
                "memory_free": None if kind == "CUDA" and cuda_free is None else (
                    cuda_free if kind == "CUDA" else ram_free) * 2**20}

    observed = {
        "system": {"devices": [device("CPU")]},
        "nvidia": {"devices": [device("CUDA")] if "CUDA" in kinds else []},
        "openvino": {"devices": [{**device("CPU"), "id": "OV:CPU"}] +
                     [device(k) for k in kinds if k != "CUDA"]},
        "environment": {"versions": dict.fromkeys(("openvino", "openvino-genai", "onnxruntime",
                                                    "faster-whisper", "ctranslate2"), "1"),
                        "loss_checked": True, "artifacts": {
                            "openvino": {"model": "parakeet" if parakeet else "whisper-turbo", "complete": True},
                            "cuda": {"model": "whisper-turbo", "complete": True},
                            "llm": {"model": "Qwen-8B-int4", "complete": True}}},
        "tts": {"status": "ok", "device": "CUDA" if "CUDA" in kinds else "CPU",
                "physical_id": "CUDA-uuid" if "CUDA" in kinds else None,
                "model": "ChatterboxMultilingualTTS", "versions": {"torch": "1", "chatterbox-tts": "1"}},
    }
    inventory = device_inventory.make_inventory(observed, now=100)
    for row in inventory["capabilities"]:
        if row["prerequisites"]:
            row.update(validation="verified", ready=True)
        # Synthetic evidence of CPU TTS readiness as well as CUDA readiness.
        # Today's live probe reports only the server's current device.
        if row["device"] == "TTS:CPU":
            row.update(prerequisites=True, artifacts_complete=True, ready=True, validation="verified")
    return inventory


def _chain(entry):
    return ([entry["kind"]] if entry["kind"] else []) + [d["kind"] for d in entry["fallback_details"]]


@pytest.mark.parametrize("kinds,mode,stt,llm,tts", [
    (("CUDA", "NPU", "GPU"), "dictation", ["CUDA", "NPU", "GPU", "CPU"], [], []),
    (("CUDA", "NPU", "GPU"), "chat_local", ["NPU", "GPU", "CUDA", "CPU"], ["GPU", "CPU"], ["CUDA", "CPU"]),
    (("CUDA", "NPU", "GPU"), "chat_claude", ["NPU", "GPU", "CUDA", "CPU"], [], ["CUDA", "CPU"]),
    (("NPU", "GPU"), "dictation", ["NPU", "GPU", "CPU"], [], []),
    (("NPU", "GPU"), "chat_local", ["NPU", "GPU", "CPU"], ["GPU", "CPU"], ["CPU"]),
    (("NPU", "GPU"), "chat_claude", ["NPU", "GPU", "CPU"], [], ["CPU"]),
    ((), "dictation", ["CPU"], [], []),
    ((), "chat_local", ["CPU"], ["CPU"], ["CPU"]),
    ((), "chat_claude", ["CPU"], [], ["CPU"]),
])
def test_documented_examples(kinds, mode, stt, llm, tts):
    result = plan(_inventory(kinds), mode=mode, config=_config())["components"]
    assert _chain(result["stt"]) == stt
    assert _chain(result["llm"]) == llm
    assert _chain(result["tts"]) == tts
    if mode == "dictation":
        assert result["llm"]["state"] == result["tts"]["state"] == "inactive"
    if mode == "chat_claude":
        assert result["llm"]["state"] == "disabled"
        assert result["llm"]["reservation_mib"] is None


@pytest.mark.parametrize("free", [6500, None])
def test_insufficient_or_unknown_vram_blocks_cuda_sharing(free):
    # 6500 - 15% of 16384 fits TTS, but not TTS + Whisper.
    result = plan(_inventory(("CUDA",), cuda_free=free), mode="chat_local", config=_config())["components"]
    assert result["tts"]["kind"] == "CUDA"
    assert _chain(result["stt"]) == ["CPU"]


def test_unknown_memory_allows_one_accelerator_load():
    result = plan(_inventory(("CUDA",), cuda_free=None), mode="dictation", config=_config())["components"]["stt"]
    assert result["kind"] == "CUDA"
    assert result["confidence"] == "unknown"


@pytest.mark.parametrize("mode", ["dictation", "chat_local"])
def test_live_openvino_unknown_memory_does_not_inherit_ram_certainty(monkeypatch, mode):
    class Core:
        available_devices = ["CPU", "GPU.0"]

        def get_property(self, device, prop):
            raise RuntimeError("unsupported")

    monkeypatch.setattr(device_queries, "openvino_core", Core)
    updates = []
    device_queries.openvino_inventory(updates.append)
    observed = _inventory(("GPU",))["observed"]
    observed["openvino"] = updates[-1]
    inventory = device_inventory.make_inventory(observed, now=100)
    gpu = next(d for d in inventory["devices"] if d["kind"] == "GPU")
    assert all(gpu[field] is None for field in ("memory_shared", "memory_total", "memory_free"))
    result = plan(inventory, mode=mode, config=_config())["components"]
    component = "stt" if mode == "dictation" else "llm"
    assert result[component]["device"] == "OV:GPU.0"
    assert result[component]["confidence"] == "unknown"
    if mode == "chat_local":
        assert _chain(result["stt"]) == ["CPU"]


@pytest.mark.parametrize("unknown_field", [None, "memory_shared", "memory_total", "memory_free"])
def test_accelerator_sharing_requires_known_capacity_and_sharing(unknown_field):
    inventory = _inventory(("GPU",))
    gpu = next(d for d in inventory["devices"] if d["kind"] == "GPU")
    if unknown_field:
        gpu[unknown_field] = None
    result = plan(inventory, mode="chat_local", config=_config())["components"]
    assert result["llm"]["kind"] == "GPU"
    assert result["llm"]["confidence"] == ("unknown" if unknown_field else "estimated")
    assert result["stt"]["kind"] == ("CPU" if unknown_field else "GPU")
    if unknown_field:
        assert "GPU" not in _chain(result["stt"])


@pytest.mark.parametrize("kinds,expected", [(("CUDA", "NPU", "GPU"), "CUDA"), (("NPU", "GPU"), None)])
def test_shared_ram_is_one_budget(kinds, expected):
    inventory = _inventory(kinds, ram_free=14000 if "CUDA" not in kinds else 11000)
    for d in inventory["devices"]:
        if d["memory_shared"]:
            d["memory_total"] = 16384 * 2**20
    result = plan(inventory, mode="chat_local", config=_config())["components"]
    assert result["llm"]["kind"] == "GPU"
    assert result["stt"]["kind"] == expected
    assert "NPU" not in _chain(result["stt"])
    assert "GPU" not in _chain(result["stt"])
    if expected is None:
        assert result["stt"]["state"] == "unavailable"


@pytest.mark.parametrize("changes,component,expected", [
    ({"device": "gpu"}, "stt", "GPU"),
    ({"device": "CPU"}, "stt", "CPU"),
    ({"llm_device": "cpu"}, "llm", "CPU"),
    ({"tts_device": "CPU"}, "tts", "CPU"),
])
def test_explicit_override_wins(changes, component, expected):
    result = plan(_inventory(), mode="chat_local", config=_config(**changes))["components"][component]
    assert result["kind"] == expected
    assert "explicit" in result["reason"]


@pytest.mark.parametrize("preference", ["GPU.1", "gpu.1", "OV:GPU.1", "ov:gpu.1"])
@pytest.mark.parametrize("model,key,components", [
    ("turbo", "device", ["stt"]),
    ("turbo", "llm_device", ["llm"]),
    ("parakeet", "device", ["stt.parakeet.encoder", "stt.parakeet.decoder"]),
])
def test_explicit_backend_index_selects_second_gpu(preference, model, key, components):
    observed = _inventory(("GPU",), parakeet=model == "parakeet")["observed"]
    gpu = next(d for d in observed["openvino"]["devices"] if d["kind"] == "GPU")
    observed["openvino"]["devices"] = [d for d in observed["openvino"]["devices"] if d["kind"] != "GPU"] + [
        {**gpu, "id": f"OV:GPU.{index}", "index": f"GPU.{index}", "physical_id": f"gpu-{index}"}
        for index in range(2)]
    inventory = device_inventory.make_inventory(observed, now=100)
    results = plan(inventory, mode="chat_local", config=_config(model_size=model, **{key: preference}))["components"]
    for component in components:
        result = results[component]
        assert result["device"] == "OV:GPU.1"
        assert result["index"] == "GPU.1"
        assert result["physical_id"] == "gpu-1"
        assert result["reason"] == "explicit device choice"
        assert "OV:GPU.0" in result["fallbacks"]


@pytest.mark.parametrize("preference", ["CPU", "cpu", "OV:CPU", "ov:cpu"])
@pytest.mark.parametrize("model", ["turbo", "parakeet"])
def test_cpu_aliases_apply_to_every_override(preference, model):
    results = plan(_inventory(parakeet=model == "parakeet"), mode="chat_local",
                   config=_config(model_size=model, device=preference, llm_device=preference,
                                  tts_device=preference))["components"]
    for result in results.values():
        assert result["kind"] == "CPU"
        assert result["reason"] == "explicit device choice"


def test_explicit_stt_reserves_before_automatic_tts():
    result = plan(_inventory(("CUDA",), cuda_free=6500), mode="chat_local",
                  config=_config(device="CUDA"))["components"]
    assert result["stt"]["kind"] == "CUDA"
    assert result["tts"]["kind"] == "CPU"


def test_auto_is_case_insensitive_and_priority_breaks_ties():
    result = plan(_inventory(), mode="dictation", config=_config(device="AuTo", device_priority=["GPU", "NPU", "CUDA"]))
    assert _chain(result["components"]["stt"]) == ["GPU", "NPU", "CUDA", "CPU"]


@pytest.mark.parametrize("mode", ["dictation", "chat_local", "chat_claude"])
@pytest.mark.parametrize("device", ["auto", "CPU", "CUDA"])
def test_parakeet_has_encoder_decoder_and_never_cuda(mode, device):
    result = plan(_inventory(parakeet=True), mode=mode, config=_config(model_size="parakeet", device=device))["components"]
    assert "stt" not in result
    encoder, decoder = result["stt.parakeet.encoder"], result["stt.parakeet.decoder"]
    assert encoder["kind"] == ("CPU" if device == "CPU" else "NPU")
    assert decoder["kind"] == ("CPU" if device == "CPU" else "GPU")
    assert "CUDA" not in _chain(encoder) + _chain(decoder)
    assert "NPU" not in _chain(decoder)
    assert encoder["model"] == decoder["model"] == "parakeet"
    assert encoder["backend"] == decoder["backend"] == "openvino"


def test_parakeet_cpu_only():
    result = plan(_inventory((), parakeet=True), mode="dictation", config=_config(model_size="parakeet"))["components"]
    assert result["stt.parakeet.encoder"]["kind"] == result["stt.parakeet.decoder"]["kind"] == "CPU"


@pytest.mark.parametrize("state,expected", [("unverified", ["GPU", "CPU", "NPU"]),
                                            ("quarantined", ["GPU", "CPU"])])
def test_npu_requires_validation_even_with_override(state, expected):
    inventory = _inventory(("NPU", "GPU"))
    for row in inventory["capabilities"]:
        if row["device"] == "OV:NPU":
            row.update(validation=state, ready=False)
    result = plan(inventory, mode="chat_local", config=_config(device="NPU"))["components"]["stt"]
    assert _chain(result) == expected
    if state == "unverified":
        assert result["fallback_details"][-1]["requires_validation"]


@pytest.mark.parametrize("field,value", [("supported", False), ("dependencies_installed", False),
                                        ("artifacts_complete", False), ("prerequisites", False),
                                        ("validation", "quarantined")])
def test_capability_filters_apply_to_preferred_and_fallbacks(field, value):
    inventory = _inventory()
    for row in inventory["capabilities"]:
        if row["device"] == "CUDA:0":
            row[field] = value
    result = plan(inventory, mode="dictation", config=_config())["components"]["stt"]
    assert _chain(result) == ["NPU", "GPU", "CPU"]


def test_absent_device_is_not_a_candidate():
    inventory = _inventory()
    inventory["devices"] = [d for d in inventory["devices"] if d["kind"] != "NPU"]
    assert "NPU" not in _chain(plan(inventory, mode="dictation", config=_config())["components"]["stt"])


@pytest.mark.parametrize("inventory", [None, device_inventory.make_inventory({}, now=0)])
def test_missing_inventory_is_conservative(inventory):
    result = plan(inventory, mode="chat_local", config=_config(device="NPU"), generation=7)
    assert result["id"] == result["generation"] == 7
    for entry in result["components"].values():
        assert entry["kind"] == "CPU"
        assert entry["confidence"] == "unknown"


def test_timeout_does_not_authorize_known_missing_artifacts():
    inventory = _inventory(())
    inventory.update(timed_out=True)
    for row in inventory["capabilities"]:
        row["artifacts_complete"] = False
    result = plan(inventory, mode="dictation", config=_config())["components"]["stt"]
    assert result["state"] == "unavailable"


def test_partial_inventory_keeps_observed_facts_with_low_confidence():
    inventory = _inventory(("GPU",))
    inventory.update(timed_out=True, pending=["nvidia"])
    result = plan(inventory, mode="dictation", config=_config())["components"]["stt"]
    assert _chain(result) == ["GPU", "CPU"]
    assert result["confidence"] == "unknown"


def test_measured_latency_beats_priority_and_measured_peak_is_used():
    inventory = _inventory()
    for row in inventory["capabilities"]:
        if row["component"] == "stt.whisper":
            inventory["measurements"][row["id"]] = {
                "peak_bytes": 1200 * 2**20, "latency_ms": 10 if row["device"] == "OV:GPU" else 20}
    result = plan(inventory, mode="dictation", config=_config())["components"]["stt"]
    assert result["kind"] == "GPU"
    assert result["reservation_mib"] == 1200
    assert result["confidence"] == "measured"


def test_resident_model_is_not_reserved_again_and_breaks_ties():
    inventory = _inventory(cuda_free=2800)
    loaded = {"stt": {"device": "CUDA:0", "model": "turbo"}}
    result = plan(inventory, mode="dictation", config=_config(device_priority=["GPU", "CUDA", "NPU"]), loaded=loaded)
    assert result["components"]["stt"]["kind"] == "CUDA"
    assert "already resident" in result["components"]["stt"]["reason"]
    loaded["stt"]["model"] = "small"
    assert plan(inventory, mode="dictation", config=_config(), loaded=loaded)["components"]["stt"]["kind"] == "NPU"


def test_inactive_resident_tts_blocks_unknown_cuda_sharing():
    result = plan(_inventory(cuda_free=None), mode="dictation", config=_config(),
                  loaded={"tts": {"device": "CUDA:0", "model": "ChatterboxMultilingualTTS"}})["components"]
    assert result["stt"]["kind"] == "NPU"
    assert "CUDA" not in _chain(result["stt"])
    assert result["tts"]["state"] == "inactive"


def test_resident_tts_free_memory_is_not_charged_twice():
    result = plan(_inventory(("CUDA",), cuda_free=4500), mode="chat_claude", config=_config(),
                  loaded={"tts": {"device": "CUDA:0", "model": "ChatterboxMultilingualTTS"}})["components"]
    assert result["tts"]["kind"] == result["stt"]["kind"] == "CUDA"


def test_cpu_also_requires_memory_headroom():
    inventory = _inventory((), ram_free=1800)
    inventory["devices"][0]["memory_total"] = 2048 * 2**20
    result = plan(inventory, mode="dictation", config=_config())["components"]["stt"]
    assert result["state"] == "unavailable"  # 1800 - minimum margin 512 < 1500.


def test_tts_unknown_uuid_never_assumes_first_cuda_card():
    inventory = _inventory(("CUDA",))
    for row in inventory["capabilities"]:
        if row["component"] == "tts":
            row["physical_id"] = None
    result = plan(inventory, mode="chat_claude", config=_config())["components"]
    assert result["tts"]["device"] == "TTS:CUDA"
    assert result["tts"]["physical_id"] is None
    assert result["stt"]["kind"] == "CPU"


@pytest.mark.parametrize("second_free,expected", [(None, "CUDA:1"), (4500, "CUDA:1"), (3500, "CPU")])
@pytest.mark.parametrize("verified_npu", [False, True])
def test_multiple_cuda_cards_keep_identity_and_separate_reservations(second_free, expected, verified_npu):
    inventory = _inventory(("CUDA", "NPU") if verified_npu else ("CUDA",), cuda_free=None)
    first = next(d for d in inventory["devices"] if d["kind"] == "CUDA")
    # 4500 MiB minus the margin fits STT alone, not STT + TTS; 3500 cannot fit STT.
    inventory["devices"].append({**first, "id": "CUDA:1", "index": 1, "physical_id": "second-uuid",
                                 "memory_free": None if second_free is None else second_free * 2**20})
    # The matrix also contains an unsupported OpenVINO row for CUDA:0.
    row = next(r for r in inventory["capabilities"] if r["component"] == "stt.whisper"
               and r["device"] == "CUDA:0" and r["backend"] == "faster-whisper")
    assert row["supported"] and row["prerequisites"]
    inventory["capabilities"].append({**row, "id": row["id"] + "-second", "device": "CUDA:1"})
    result = plan(inventory, mode="chat_claude", config=_config())["components"]
    assert result["tts"]["device"] == "CUDA:0"
    assert result["stt"]["device"] == ("OV:NPU" if verified_npu else expected)
    assert "CUDA:0" not in result["stt"]["fallbacks"]
    if expected == "CUDA:1":
        cuda = next(d for d in [result["stt"], *result["stt"]["fallback_details"]] if d["device"] == "CUDA:1")
        assert cuda["physical_id"] == "second-uuid"
        assert cuda["index"] == 1
        assert cuda["backend"] == "faster-whisper"
    else:
        assert "CUDA:1" not in result["stt"]["fallbacks"]


def test_incompatible_llm_override_falls_back_without_changing_model():
    config = _config(llm_device="CUDA")
    result = plan(_inventory(), mode="chat_local", config=config)["components"]["llm"]
    assert _chain(result) == ["GPU", "CPU"]
    assert result["model"] == config["llm_model"]
    assert "requested device unavailable" in result["reason"]


@pytest.mark.parametrize("changes", [{"tts_server_command": ["custom-server"]}, {"tts_url": "https://voice.example/tts"}])
def test_external_tts_is_not_repositioned(changes):
    result = plan(_inventory(), mode="chat_claude", config=_config(**changes))["components"]["tts"]
    assert result["state"] == "external"
    assert result["device"] is None and result["reservation_mib"] is None


def test_plan_is_deterministic_and_does_not_mutate_inputs():
    inventory, config = _inventory(), _config()
    loaded = {"stt": {"device": "OV:NPU", "model": "turbo"}}
    before = copy.deepcopy((inventory, config, loaded))
    first = plan(inventory, mode="chat_local", config=config, loaded=loaded, generation=2)
    assert first == plan(inventory, mode="chat_local", config=config, loaded=loaded, generation=2)
    assert (inventory, config, loaded) == before
    assert first["components"]["stt"]["physical_id"] == "NPU-uuid"


@pytest.mark.parametrize("mode", ["dictation", "chat_local", "chat_claude"])
def test_logging_hook_runs_in_background_without_changing_config(monkeypatch, mode):
    from debora_whisper import dictation_engine as engine

    inventory = {**_inventory(), "elapsed_ms": 1.0, "timed_out": False}
    config = _config(voice_chat=mode != "dictation", voice_chat_backend="claude" if mode == "chat_claude" else "local")
    original = copy.deepcopy(config)
    calls, logs = [], []
    def capture(*args, **kwargs):
        calls.append(threading.current_thread())
        return plan(*args, **kwargs)
    monkeypatch.setattr(device_inventory, "probe_inventory", lambda *args, **kwargs: inventory)
    monkeypatch.setattr(device_planner, "plan", capture)
    monkeypatch.setattr(engine, "log", logs.append)
    thread = engine.log_hardware_inventory(config)
    thread.join(5)
    assert not thread.is_alive()
    assert calls == [thread]
    assert config == original
    lines = [line for line in logs if line.startswith("Device plan 1 ")]
    assert len(lines) == 3
    assert all(f"mode={mode}" in line and "confidence=" in line and "reservation=" in line for line in lines)
    stt_line = next(line for line in lines if "component=stt " in line)
    assert "GPU,GPU" not in stt_line
    assert "OV:GPU" in stt_line


@pytest.mark.parametrize("failure", ["probe", "planner"])
def test_logging_failure_is_contained(monkeypatch, failure):
    from debora_whisper import dictation_engine as engine

    logs = []
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(engine, "log", logs.append)
    monkeypatch.setattr(device_inventory, "probe_inventory", fail if failure == "probe" else lambda *args, **kwargs: _inventory())
    if failure == "planner":
        monkeypatch.setattr(device_planner, "plan", fail)
    thread = engine.log_hardware_inventory(_config())
    thread.join(5)
    assert not thread.is_alive()
    if failure == "planner":
        assert any("Device plan failed:" in line for line in logs)
    else:
        assert any(line.startswith("Device plan 1 ") and "confidence=unknown" in line for line in logs)


def test_logging_uses_one_snapshot_while_caller_changes_config(monkeypatch):
    from debora_whisper import dictation_engine as engine

    config = _config(voice_chat=True, voice_chat_backend="claude")
    original = copy.deepcopy(config)
    started, release = threading.Event(), threading.Event()
    requests, calls, results, logs = [], [], [], []

    def blocked_probe(request, cache_path, **kwargs):
        requests.append(request)
        started.set()
        assert release.wait(5)
        return {**_inventory(), "elapsed_ms": 1.0, "timed_out": False}

    def capture(inventory, **kwargs):
        calls.append(kwargs)
        result = plan(inventory, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(device_inventory, "probe_inventory", blocked_probe)
    monkeypatch.setattr(device_planner, "plan", capture)
    monkeypatch.setattr(engine, "log", logs.append)
    thread = engine.log_hardware_inventory(config)
    try:
        assert started.wait(5)
        # Startup priority resolution and settings edits both mutate this object.
        config.update(device="CUDA", model_size="small", voice_chat=False,
                      voice_chat_backend="local", llm_model="different-model")
        config["device_priority"][:] = ["CPU"]
    finally:
        release.set()
        thread.join(5)

    assert not thread.is_alive()
    assert len(calls) == 1
    assert requests[0]["config"] is calls[0]["config"]
    assert calls[0]["config"] == original
    assert requests[0]["model_info"] == engine.MODEL_REGISTRY[original["model_size"]]
    assert calls[0]["mode"] == "chat_claude"
    assert results[0]["components"]["stt"]["model"] == "turbo"
    assert "explicit device choice" not in results[0]["components"]["stt"]["reason"]
    assert not any("failed:" in line for line in logs)


@pytest.mark.parametrize("kind,physical_id,expected", [
    ("CPU", None, "CPU"),
    ("CUDA", "CUDA-uuid", "CUDA:0"),
    ("CUDA", "second-uuid", "CUDA:1"),
    ("CUDA", None, "TTS:CUDA"),
    ("CUDA", "unmatched-uuid", "TTS:CUDA"),
])
def test_logging_reuses_live_tts_allocation(monkeypatch, kind, physical_id, expected):
    from debora_whisper import dictation_engine as engine

    # Rebuild from observations so only the loaded server's device is ready,
    # exactly as in the live capability matrix (no synthetic CPU fallback).
    observed = copy.deepcopy(_inventory(
        ("CUDA",) if kind == "CUDA" else (), cuda_free=4500, ram_free=12000)["observed"])
    observed["tts"].update(loaded=True, schema_version=1, external=False,
                           device=kind, physical_id=physical_id, index=0)
    if expected == "CUDA:1":
        first = observed["nvidia"]["devices"][0]
        observed["nvidia"]["devices"].append({
            **first, "id": "CUDA:1", "index": 1, "physical_id": "second-uuid"})
    inventory = {**device_inventory.make_inventory(observed, now=100),
                 "elapsed_ms": 1.0, "timed_out": False}
    ready = [r["device"] for r in inventory["capabilities"] if r["component"] == "tts" and r["ready"]]
    assert ready == ["TTS:" + kind]
    results, calls, logs = [], [], []

    def capture(inventory, **kwargs):
        calls.append(kwargs)
        result = plan(inventory, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(device_inventory, "probe_inventory", lambda *args, **kwargs: inventory)
    monkeypatch.setattr(device_planner, "plan", capture)
    monkeypatch.setattr(engine, "log", logs.append)
    config = _config(voice_chat=True, voice_chat_backend="claude")
    thread = engine.log_hardware_inventory(config)
    thread.join(5)
    assert not thread.is_alive()
    assert len(results) == 1
    assert calls[0]["loaded"] == {"tts": {"device": expected, "model": observed["tts"]["model"]}}
    components = results[0]["components"]
    assert components["tts"]["state"] == "planned"
    assert components["tts"]["device"] == expected
    assert "already resident" in components["tts"]["reason"]
    assert components["stt"]["state"] == "planned"
    assert components["stt"]["kind"] == ("CPU" if kind == "CUDA" and physical_id is None else kind)
    assert not any("failed:" in line for line in logs)


@pytest.mark.parametrize("report", [
    {}, {"status": "unknown", "loaded": True},
    {"status": "ok", "loaded": False}, {"status": "ok"},
    {"status": "ok", "loaded": True, "external": True},
])
def test_unobserved_tts_is_not_resident(report):
    inventory = _inventory()
    inventory["observed"]["tts"].update(report)
    if "loaded" not in report:
        inventory["observed"]["tts"].pop("loaded", None)
    assert device_planner.resident_tts(inventory) == {}
