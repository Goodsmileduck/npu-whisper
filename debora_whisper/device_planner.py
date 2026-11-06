"""Deterministic, observational device policy. No probing or model loading."""

import math
from urllib.parse import urlparse


def _normalize_device(value):
    """Use OpenVINO backend names for both inventory IDs and config aliases."""
    return str(value).upper().removeprefix("OV:")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _pool(device):
    # OpenVINO often cannot report whether memory is shared. Assume RAM then.
    if device["kind"] == "CPU" or device.get("memory_shared") is not False and device["kind"] != "CUDA":
        return "RAM"
    return device.get("physical_id") or device["id"]


def _accelerator_memory_known(device):
    # A RAM budget does not establish an accelerator's capacity or sharing mode.
    return (isinstance(device.get("memory_shared"), bool)
            and _number(device.get("memory_total")) and _number(device.get("memory_free")))


def _estimate(component, model):
    # Initial peak MiB (weights + workspace), to refine with measurements.
    # Parakeet includes resident buckets; LLM includes context/KV workspace.
    if component == "stt":
        return {"tiny": 512, "base": 768, "small": 1024, "medium": 2048,
                "turbo": 1500, "large-v3": 4096}.get(model, 4096)
    return {"stt.parakeet.encoder": 3072, "stt.parakeet.decoder": 512,
            "llm": 8192, "tts": 3072}[component]


def _tts_device(devices, kind, physical_id):
    device = next((d for d in devices.values() if d["kind"] == kind and (
        kind == "CPU" or physical_id and d.get("physical_id") == physical_id)), None)
    # The server's CUDA index belongs to its own environment, not ours.
    if device is None and kind == "CUDA":
        device = {"id": "TTS:CUDA", "kind": kind, "physical_id": physical_id,
                  "memory_shared": False}
    return device


def resident_tts(inventory):
    """Translate a live server report into allocations already in free memory."""
    inventory = inventory or {}
    report = inventory.get("observed", {}).get("tts", {})
    if report.get("status") != "ok" or report.get("loaded") is not True or report.get("external"):
        return {}
    devices = {d["id"]: d for d in inventory.get("devices", [])}
    devices.setdefault("CPU", {"id": "CPU", "kind": "CPU"})
    device = _tts_device(devices, report.get("device"), report.get("physical_id"))
    if device is None:
        return {}
    return {"tts": {"device": device["id"], "model": report.get("model")}}


def plan(inventory, *, mode, config, loaded=None, generation=0):
    """Return preferences only; never mutate inputs or change model/precision.

    Inventory memory and measurement ``peak_bytes`` are bytes. Reservations
    are MiB. Measurements are keyed by capability ID; ``latency_ms`` must
    describe comparable useful-text latency for this inventory/model.
    ``loaded`` maps component names to {device: inventory ID, model: config
    model}. Those allocations are already reflected in free memory, including
    inactive components. Only matching models can reuse a resident allocation.
    Fallback details retain backend/export/precision and physical identity.
    """
    if mode not in {"dictation", "chat_local", "chat_claude"}:
        raise ValueError(f"Unknown device plan mode: {mode}")
    inventory = inventory or {}
    loaded = loaded or {}
    observed = inventory.get("observed", {})
    partial = not inventory or bool(inventory.get("timed_out") or inventory.get("pending"))
    devices = {d["id"]: dict(d) for d in inventory.get("devices", [])}
    devices.setdefault("CPU", {"id": "CPU", "kind": "CPU", "memory_shared": True})
    measurements = inventory.get("measurements", {})
    priority = list(dict.fromkeys(str(d).upper() for d in config.get(
        "device_priority", ["CUDA", "NPU", "GPU", "CPU"])))
    priority += [d for d in ("CUDA", "NPU", "GPU", "CPU") if d not in priority]
    parakeet = "parakeet" in config.get("model_size", "turbo").lower()
    stt = ["stt.parakeet.encoder", "stt.parakeet.decoder"] if parakeet else ["stt"]
    components = stt + ["llm", "tts"]
    models = {c: config.get("model_size", "turbo") for c in stt}
    models.update(llm=config.get("llm_model", "OpenVINO/Qwen3-8B-int4-cw-ov"),
                  tts=config.get("tts_model", "ChatterboxMultilingualTTS"))
    candidates = {c: [] for c in components}
    for row in inventory.get("capabilities", []):
        component = "stt" if row["component"] == "stt.whisper" else row["component"]
        if component not in candidates or not all(row.get(k) for k in (
                "supported", "dependencies_installed", "artifacts_complete", "prerequisites")):
            continue
        if row.get("validation") in {"quarantined", "incompatible", "failed"}:
            continue
        device = devices.get(row["device"])
        if row["device"].startswith("TTS:"):
            kind = row["device"].split(":")[1]
            device = _tts_device(devices, kind, row.get("physical_id"))
            if device is not None:
                devices[device["id"]] = device
        if device is None or device.get("validation") == "quarantined":
            continue
        kinds = ({"CPU", "CUDA"} if component == "tts" else {"CPU", "GPU"} if component in (
            "llm", "stt.parakeet.decoder") else {"CPU", "GPU", "NPU"} if parakeet else
            {"CPU", "GPU", "NPU", "CUDA"})
        if device["kind"] not in kinds:
            continue
        candidates[component].append({**row, "target": device})

    # Only absent observations permit an assumed CPU. Known missing packages
    # or exports must still produce unavailable, even after a probe timeout.
    for component in components:
        source = "tts" if component == "tts" else "environment"
        if source not in observed and not candidates[component]:
            partial = True
            candidates[component].append({"id": component + "/assumed-cpu", "target": devices["CPU"],
                "backend": "chatterbox" if component == "tts" else "openvino" if parakeet and component in stt
                else "openvino-genai", "validation": "unknown", "assumed": True})

    limits = {}
    for device in devices.values():
        total, free = device.get("memory_total"), device.get("memory_free")
        if _number(total) and _number(free):
            limit = max(0, free / 2**20 - max(512, total / 2**20 * 0.15))
            pool = _pool(device)
            limits[pool] = min(limits.get(pool, limit), limit)

    selected = {}

    def resident(component, candidate):
        current = loaded.get(component, {})
        return current.get("device") == candidate["target"]["id"] and current.get("model") == models[component]

    def reservation(component, candidate):
        peak = measurements.get(candidate["id"], {}).get("peak_bytes")
        return math.ceil(peak / 2**20) if _number(peak) else _estimate(component, models[component])

    def occupants(component, device):
        others = {(name, c["target"]["id"]): c["target"] for name, c in selected.items() if name != component}
        for name, current in loaded.items():
            if (name != component or current.get("model") != models[component]) and current.get("device") in devices:
                others[name, current["device"]] = devices[current["device"]]
        # A TTS CUDA device with unknown identity could share any CUDA card.
        return [d for d in others.values() if d["id"] == device["id"] or (
            d["kind"] == device["kind"] == "CUDA" and (not d.get("physical_id") or
                not device.get("physical_id") or d["physical_id"] == device["physical_id"]))]

    def fits(component, candidate):
        device = candidate["target"]
        if resident(component, candidate):
            return True
        pool = _pool(device)
        used = sum(reservation(name, c) for name, c in selected.items()
                   if name != component and _pool(c["target"]) == pool and not resident(name, c))
        others = occupants(component, device)
        if device["kind"] != "CPU" and others and (not _accelerator_memory_known(device) or any(
                not _accelerator_memory_known(d) or _pool(d) != pool for d in others)):
            return False
        return pool not in limits or used + reservation(component, candidate) <= limits[pool]

    def override(component):
        key = "device" if component in stt else component + "_device"
        value = _normalize_device(config.get(key, "auto"))
        # Encoder choices do not force an unsupported NPU decoder.
        if component == "stt.parakeet.decoder" and value.split(".")[0] not in {"CPU", "GPU"}:
            return "AUTO"
        return value

    def explicit(component, candidate):
        device = candidate["target"]
        return override(component) in {_normalize_device(value) for value in (
            device["id"], device["kind"], device.get("index", ""))}

    def ordered(component):
        available = [c for c in candidates[component] if fits(component, c)]
        accelerated = [c for c in available if c["target"]["kind"] != "CPU" and (
            c["target"]["kind"] != "NPU" or c.get("validation") == "verified")]
        comparable = bool(accelerated) and all(_number(measurements.get(c["id"], {}).get("latency_ms"))
                                              for c in accelerated)

        def key(candidate):
            device = candidate["target"]
            kind = device["kind"]
            unverified = kind == "NPU" and candidate.get("validation") != "verified"
            policy = 0
            if mode != "dictation" and component in stt:
                policy = {"NPU": 0, "GPU": 1, "CUDA": 1 if comparable else 2, "CPU": 3}[kind]
            latency = measurements[candidate["id"]]["latency_ms"] if comparable and candidate in accelerated else 0
            return (unverified, not explicit(component, candidate), kind == "CPU", policy,
                    latency, not resident(component, candidate), len(occupants(component, device)),
                    priority.index(kind), device["id"], candidate["id"])

        return sorted(available, key=key), comparable

    url = config.get("tts_url")
    external_tts = bool(config.get("tts_server_command") or observed.get("tts", {}).get("external") or
                        url and urlparse(url).hostname not in {"localhost", "127.0.0.1"})
    # Reserve voice output and local generation before placing STT in chat.
    allocation_order = ["tts", "llm"] + stt if mode != "dictation" else stt
    allocation_order.sort(key=lambda c: override(c) == "AUTO")
    for component in allocation_order:
        if component == "llm" and mode == "chat_claude" or component == "tts" and external_tts:
            continue
        choices, _ = ordered(component)
        selected[component] = next((c for c in choices if c["target"]["kind"] != "NPU" or
                                    c.get("validation") == "verified"), None)
        if selected[component] is None:
            del selected[component]

    def detail(component, candidate):
        device = candidate["target"]
        return {"device": device["id"], "kind": device["kind"], "physical_id": device.get("physical_id"),
                "index": device.get("index"), "backend": candidate["backend"],
                "artifact_model": candidate.get("model"), "precision": candidate.get("precision"),
                "reservation_mib": reservation(component, candidate),
                "requires_validation": device["kind"] == "NPU" and candidate.get("validation") != "verified"}

    result = {}
    for component in components:
        state = "planned"
        reason = "device priority"
        if component == "llm" and mode == "chat_claude":
            state, reason = "disabled", "Claude uses an external LLM"
        elif component in {"llm", "tts"} and mode == "dictation":
            state, reason = "inactive", "inactive in dictation"
        elif component == "tts" and external_tts:
            state, reason = "external", "TTS placement is externally managed"
        chosen = selected.get(component)
        backend = "chatterbox" if component == "tts" else "openvino" if parakeet and component in stt else "openvino-genai"
        entry = {"state": state, "model": models[component], "backend": backend, "device": None,
                 "kind": None, "physical_id": None, "index": None, "fallbacks": [], "fallback_details": [],
                 "reservation_mib": None, "reason": reason, "confidence": "unknown"}
        if state == "planned":
            choices, comparable = ordered(component)
            alternatives = [c for c in choices if c != chosen]
            entry["fallbacks"] = [c["target"]["id"] for c in alternatives]
            entry["fallback_details"] = [detail(component, c) for c in alternatives]
            if chosen:
                entry.update(detail(component, chosen))
                if explicit(component, chosen):
                    reason = "explicit device choice"
                elif mode != "dictation":
                    reason = "chat placement; prefer separate accelerators"
                elif comparable:
                    reason = "lowest measured useful-text latency"
                if resident(component, chosen):
                    reason += "; already resident, no new reservation"
                if override(component) != "AUTO" and not explicit(component, chosen):
                    reason += "; requested device unavailable or unsafe"
                device = chosen["target"]
                known = _pool(device) in limits and (device["kind"] == "CPU" or _accelerator_memory_known(device))
                peak = measurements.get(chosen["id"], {}).get("peak_bytes")
                entry["confidence"] = "unknown" if partial or not known or chosen.get("assumed") else (
                    "measured" if _number(peak) else "estimated")
                entry["reason"] = reason + ("; partial inventory" if partial else "")
            else:
                entry.update(state="unavailable", reason="no compatible validated device fits memory")
        result[component] = entry
    return {"id": generation, "generation": generation, "mode": mode, "components": result}
