"""Observed allocations, separate from the advisory device planner.

Loaders supply facts; rendering never probes hardware or starts a service.
Workers publish through their existing JSON/HTTP protocols to the main logger.
"""
import os
import platform
import threading
from urllib.parse import urlsplit


def one_line(value):
    return " ".join(str(value).split())


def format_endpoint(value):
    """Only expose an endpoint's origin, never credentials or URL payloads."""
    if not value:
        return "not configured"
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        if parsed.scheme not in ("http", "https") or not host:
            return "invalid endpoint"
        if ":" in host:
            host = f"[{host}]"
        port = f":{parsed.port}" if parsed.port is not None else ""
        return f"{parsed.scheme}://{host}{port}"
    except (ValueError, TypeError, AttributeError):
        return "invalid endpoint"


def cpu_name():
    """The CPU's model name; platform.processor() is empty on most Linux."""
    name = platform.processor()
    if name and name != platform.machine():
        return name
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return name or "name unavailable"


def cpu_target():
    return "CPU " + cpu_name()


def cuda_target(index=0, fallback_name=None):
    """Ask the loaded CUDA driver, respecting this process's device ordering."""
    try:
        import ctypes
        driver = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
        device, name = ctypes.c_int(), ctypes.create_string_buffer(256)
        if driver.cuDeviceGet(ctypes.byref(device), index) != 0:
            raise RuntimeError("CUDA device identity unavailable")
        if driver.cuDeviceGetName(name, len(name), device) != 0:
            raise RuntimeError("CUDA device name unavailable")
        return f"CUDA:{index} {name.value.decode('utf-8', 'replace')}"
    except Exception:
        # nvidia-smi lists ALL cards; never attribute that list to CUDA:0.
        name = fallback_name if isinstance(fallback_name, str) and "\n" not in fallback_name else None
        return f"CUDA:{index} {name or 'name unavailable'}"


def ov_target(core, device, compiled=None):
    """Resolve GPU aliases using the loaded graph, then plugin identity.

    GenAI does not expose a compiled graph. For its explicit device loads,
    query that plugin's DEVICE_ID; never assume GPU means the first GPU.
    """
    if compiled is not None:
        try:
            devices = compiled.get_property("EXECUTION_DEVICES")
            if isinstance(devices, (list, tuple)) and devices:
                return " + ".join(ov_target(core, d) for d in devices)
        except Exception:
            pass
    resolved = device
    try:
        if device == "GPU":
            index = core.get_property(device, "DEVICE_ID")
            if isinstance(index, str) and all(part.isdigit() for part in index.split(".")):
                resolved = "GPU." + index
        name = core.get_property(resolved, "FULL_DEVICE_NAME")
        if not isinstance(name, str):
            raise ValueError("missing device name")
        if resolved == "GPU":
            matches = [d for d in core.available_devices if d.startswith("GPU.")
                       and core.get_property(d, "FULL_DEVICE_NAME") == name]
            if len(matches) == 1:
                resolved = matches[0]
        suffix = " (index unreported)" if resolved == "GPU" else ""
        return f"OV:{resolved} {name}{suffix}"
    except Exception:
        return f"OV:{resolved} (name/index unavailable)"


def model_description(model, model_name, planned, reason=None):
    target = getattr(model, "hardware_target", None)
    if not isinstance(target, str):
        target = f"{getattr(model, 'device', planned)} (hardware identity unreported)"
    backend = getattr(model, "hardware_backend", type(model).__name__)
    detail = f"{model_name} ({backend}) -> {target}  pid {os.getpid()}"
    reasons = list(getattr(model, "hardware_fallbacks", []))
    if reason:
        reasons.insert(0, reason)
    if reasons:
        detail += f"; planned {planned}, fell back to {target}: " + "; ".join(map(one_line, reasons))
    return detail


def format_summary(config, rows):
    mode = ("chat_claude" if config.get("voice_chat_backend") == "claude" else "chat_local") \
        if config.get("voice_chat") else "dictation"
    lines = [f"Hardware allocation (mode={mode}):",
             f"  APP  UI/audio/orchestration -> {cpu_target()}  pid {os.getpid()}",
             "  STT  " + rows.get("STT", "not loaded"),
             "  DRAFT STT  shares STT model and pid (no separate allocation)",
             "  VAD  " + rows.get("VAD", "not loaded")]
    if rows.get("STT RECOVERY"):
        lines.append("  STT RECOVERY  " + rows["STT RECOVERY"])
    for component in ("LLM", "CLAUDE", "TTS"):
        active = mode != "dictation" and (component == "TTS" or
                 (component == "CLAUDE") == (mode == "chat_claude"))
        detail = rows.get(component)
        if not active:
            suffix = (("; resident: " if " -> " in detail else "; ") + detail) if detail else ""
            detail = f"inactive in {mode}" + suffix
        elif detail is None:
            if component == "CLAUDE":
                detail = f"external Claude Code; model={config.get('harness_model') or 'CLI default'}; not started"
            elif component == "TTS":
                detail = f"not loaded/confirmed; endpoint={format_endpoint(config.get('tts_url'))}"
            else:
                detail = f"{config.get('llm_model', 'unknown')} not loaded"
        lines.append(f"  {component}  {one_line(detail)}")
    if rows.get("HELPERS"):
        lines.append("  HELPERS  " + rows["HELPERS"])
    return "\n".join(lines)


class AllocationReporter:
    def __init__(self):
        self.lock = threading.RLock()
        self.config = None
        self.log = None
        self.rows = {}
        self.owners = {}
        self.last = None

    def configure(self, config, log):
        with self.lock:
            self.config, self.log = config, log
            # Workers can survive an engine rebuild; STT and VAD cannot.
            for key in ("STT", "STT RECOVERY", "VAD"):
                self.rows.pop(key, None)
            self.last = None

    def claim(self, component):
        """Supersede earlier workers before starting a new reader thread."""
        with self.lock:
            owner = object()
            self.owners[component] = owner
            self.rows.pop(component, None)
            return owner

    def update(self, component, detail, *, emit=True, owner=None):
        with self.lock:
            if owner is not None and self.owners.get(component) is not owner:
                return
            if detail is None:
                self.rows.pop(component, None)
                # Revoke callbacks still queued by a stopped reader.
                self.owners.pop(component, None)
            else:
                self.rows[component] = one_line(detail)
            if emit:
                self.emit()

    def emit(self):
        with self.lock:
            if self.config is None or self.log is None:
                return
            block = format_summary(self.config, self.rows)
            if block != self.last:
                self.last = block
                self.log(block)


allocation = AllocationReporter()


def report_tts(config, inventory):
    """A server's PID/device belong to its environment, never to uv's PID."""
    endpoint = format_endpoint(config.get("tts_url"))
    if not isinstance(inventory, dict) or inventory.get("loaded") is not True:
        detail = f"external endpoint={endpoint}; device/model/pid unreported"
    else:
        from urllib.parse import urlparse
        service = "external" if (config.get("tts_server_command") or
                  urlparse(endpoint).hostname not in ("localhost", "127.0.0.1")) else "worker"
        target = inventory.get("device_name") or "hardware name unreported"
        device = inventory.get("device_id") or inventory.get("device") or "device unreported"
        detail = (f"{inventory.get('model', 'unknown')} ({service} endpoint={endpoint}) -> "
                  f"{device} {target}  server pid {inventory.get('pid', 'unreported')}")
        if inventory.get("fallback_reason"):
            detail += (f"; planned {inventory.get('requested_device')}, fell back to {device}: "
                       f"{inventory['fallback_reason']}")
    allocation.update("TTS", detail)
