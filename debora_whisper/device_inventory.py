"""Observational startup inventory; never selects a device or loads a model.

Native queries run in disposable interpreters, not threads in the UI process.
The threads here only launch/read/kill those processes or access cache files.
Even slow process creation, cache I/O and cleanup are outside the caller's wait.
Unknown facts stay unknown; this is not a benchmark or proof of driver health.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

SCHEMA_VERSION = 1
POLICY_VERSION = 1
# The probe runs beside startup, never in front of it: importing OpenVINO in
# a fresh process alone takes a few seconds on a laptop.
PROBE_BUDGET = 15.0
QUERY_TIMEOUT = 5.0
UNKNOWN_DRIVER_TTL = 24 * 60 * 60
MAX_CACHE_BYTES = 1_000_000
SOURCES = ("system", "environment", "nvidia", "openvino", "tts")

_workers_lock = threading.Lock()
_workers = {}


def start_inventory_worker(target, args=(), *, cancelled=None, name="device-inventory"):
    """Retain diagnostic workers until they exit, including query children."""
    if cancelled is None:
        cancelled = threading.Event()

    def run():
        try:
            if not cancelled.is_set():
                target(*args)
        finally:
            with _workers_lock:
                _workers.pop(threading.current_thread(), None)

    worker = threading.Thread(target=run, name=name, daemon=True)
    with _workers_lock:
        _workers[worker] = cancelled
        try:
            worker.start()
        except BaseException:
            _workers.pop(worker, None)
            raise
    return worker


def stop_inventory_workers(timeout=2.0):
    """Cancel diagnostics and join them with one bounded shutdown budget."""
    deadline = time.perf_counter() + timeout
    while True:
        with _workers_lock:
            workers = list(_workers)
            for cancelled in _workers.values():
                cancelled.set()
        workers = [worker for worker in workers if worker is not threading.current_thread()]
        if not workers or time.perf_counter() >= deadline:
            return
        for worker in workers:
            worker.join(max(0, deadline - time.perf_counter()))


def _cpu():
    return {"id": "CPU", "kind": "CPU", "index": None, "physical_id": None,
            "name": "CPU", "driver": None, "memory_total": None,
            "memory_free": None, "memory_shared": True}


def _versions(names):
    from importlib.metadata import PackageNotFoundError, version
    result = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _artifact(path, required, *, model, precision):
    """Metadata only, no weight hashing or downloads. Missing is not ready."""
    files = []
    complete = path is not None
    if path is not None:
        for name in required:
            try:
                stat = (path / name).stat()
                complete &= (path / name).is_file() and stat.st_size > 0
                files.append([name, stat.st_size, stat.st_mtime_ns])
            except OSError:
                complete = False
        # Include sidecars/shards in the key, without reading large files.
        for file in sorted(path.glob("*")):
            if file.is_file() and file.name not in required:
                stat = file.stat()
                files.append([file.name, stat.st_size, stat.st_mtime_ns])
    return {"model": model, "precision": precision, "path": str(path) if path else None,
            "revision": path.name if path else None, "files": files, "complete": bool(complete)}


def _local_snapshot(model):
    """Resolve only known local metadata, never call snapshot_download."""
    path = Path(model)
    if path.is_dir():
        return path
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        repo = Path(HF_HUB_CACHE) / ("models--" + model.replace("/", "--"))
        revision = (repo / "refs" / "main").read_text(encoding="utf-8").strip()
        if revision and Path(revision).name == revision:
            snapshot = repo / "snapshots" / revision
            return snapshot if snapshot.is_dir() else None
    except (ImportError, OSError):
        pass
    return None


def _faster_whisper_repo(model):
    """Read the installed backend's alias table without loading native DLLs."""
    import ast
    from importlib.metadata import distribution, PackageNotFoundError
    try:
        source = distribution("faster-whisper").locate_file("faster_whisper/utils.py")
        for node in ast.parse(source.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "_MODELS" for target in node.targets):
                return ast.literal_eval(node.value).get(model)
    except (OSError, ValueError, SyntaxError, PackageNotFoundError):
        pass
    return None


def _environment(request):
    from debora_whisper import dictation_engine as de
    versions = _versions(("openvino", "openvino-genai", "onnxruntime", "ctranslate2",
                          "faster-whisper", "nvidia-cublas-cu12", "nvidia-cudnn-cu12"))
    info = request["model_info"]
    path = Path(request["model_dir"]) / info["local_dir"]
    parakeet = info["backend"] == "parakeet"
    required = (["encoder-model.onnx", "decoder_joint-model.onnx", "nemo128.onnx", "vocab.txt"]
                if parakeet else ["openvino_encoder_model.xml", "openvino_encoder_model.bin",
                                     "openvino_decoder_model.xml", "openvino_decoder_model.bin",
                                     "openvino_tokenizer.xml", "openvino_tokenizer.bin",
                                     "openvino_detokenizer.xml", "openvino_detokenizer.bin", "config.json"])
    ov = _artifact(path, required, model=info["ov_repo"],
                   precision="unknown" if parakeet else "int8")
    ov["complete"] &= de.model_files_complete(path)
    # The faster-whisper aliases used by FasterWhisperCUDA, without importing
    # that backend (and its native dependencies) just to find local files.
    size = request["config"]["model_size"]
    cuda_repo = _faster_whisper_repo(size) if not parakeet else None
    cuda = _artifact(_local_snapshot(cuda_repo) if cuda_repo else None,
                     ["model.bin", "config.json", "tokenizer.json"],
                     model=cuda_repo or info["repo"], precision="int8_float16")
    llm_model = request["config"].get("llm_model", "")
    llm = _artifact(_local_snapshot(llm_model) if llm_model else None,
                    ["openvino_model.xml", "openvino_model.bin", "config.json",
                     "openvino_tokenizer.xml", "openvino_tokenizer.bin",
                     "openvino_detokenizer.xml", "openvino_detokenizer.bin"],
                    model=llm_model, precision="unknown")
    # Read the existing loss format without deleting it, unlike the selector's
    # reboot cleanup. Unknown boot information remains conservatively lost.
    loss = None
    loss_checked = True
    try:
        record = json.loads(Path(request["npu_loss_file"]).read_text(encoding="utf-8"))
        boot = de._boot_time()
        if boot is None or record.get("boot") is None or abs(record["boot"] - boot) < 120:
            loss = record
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError, AttributeError):
        loss_checked = False
    return {"versions": versions, "artifacts": {"openvino": ov, "cuda": cuda, "llm": llm},
            "npu_loss": loss, "loss_checked": loss_checked, "python": sys.executable}


def _tts(config):
    """Ask an already-running managed server; never invoke uv or its script."""
    from urllib.parse import urlparse
    from urllib.request import ProxyHandler, build_opener
    url = config.get("tts_url", "")
    unknown = {"status": "unknown", "endpoint": url, "versions": {}, "device": None}
    if config.get("tts_server_command") or urlparse(url).hostname not in ("localhost", "127.0.0.1"):
        return {**unknown, "external": True}
    try:
        with build_opener(ProxyHandler({})).open(url.rstrip("/") + "/health", timeout=0.2) as response:
            report = json.loads(response.read(16_384)).get("inventory")
        if (isinstance(report, dict) and report.get("schema_version") == 1
                and report.get("device") in ("CPU", "CUDA")
                and isinstance(report.get("versions"), dict) and report.get("loaded") is True):
            return {**report, "status": "ok", "endpoint": url, "external": False}
    except Exception:
        pass
    return unknown


def _source(source, request, emit):
    if source == "system":
        import platform
        cpu = _cpu()
        cpu["name"] = platform.processor() or "CPU"
        try:
            import psutil
            memory = psutil.virtual_memory()
            cpu.update(memory_total=memory.total, memory_free=memory.available)
        except ImportError:
            pass
        emit({"devices": [cpu], "logical_cpus": os.cpu_count()})
    elif source == "environment":
        emit(_environment(request))
    elif source == "tts":
        emit(_tts(request["config"]))
    elif source == "nvidia":
        from debora_whisper.device_queries import nvidia_inventory
        emit(nvidia_inventory(QUERY_TIMEOUT))
    elif source == "openvino":
        from debora_whisper.device_queries import openvino_inventory
        openvino_inventory(emit)


def capability_matrix(observed):
    """Support is a backend contract; ready additionally needs fresh local facts.

    No native library is loaded here. 'ready' means prerequisites observed,
    not successful model execution; NPU additionally needs cached validation.
    """
    env = observed.get("environment", {})
    versions = env.get("versions", {})
    artifacts = env.get("artifacts", {})
    devices = observed.get("system", {}).get("devices", [_cpu()])[:1]
    devices = devices + observed.get("nvidia", {}).get("devices", [])
    devices += [d for d in observed.get("openvino", {}).get("devices", []) if d["kind"] != "CPU"]
    ov_ids = {d["id"] for d in observed.get("openvino", {}).get("devices", [])}
    rows = []
    specs = (("stt.whisper", "openvino-genai", "openvino", {"CPU", "GPU", "NPU"}, ("openvino", "openvino-genai")),
             ("stt.whisper", "faster-whisper", "cuda", {"CUDA"}, ("faster-whisper", "ctranslate2")),
             ("stt.parakeet.encoder", "openvino", "openvino", {"CPU", "GPU", "NPU"}, ("openvino", "onnxruntime")),
             ("stt.parakeet.decoder", "openvino", "openvino", {"CPU", "GPU"}, ("openvino", "onnxruntime")),
             ("llm", "openvino-genai", "llm", {"CPU", "GPU"}, ("openvino", "openvino-genai")))
    for component, backend, artifact_key, kinds, dependencies in specs:
        artifact = artifacts.get(artifact_key, {})
        model = artifact.get("model")
        model_matches = (component == "llm" or not model or
                         ("parakeet" in model.lower()) == component.startswith("stt.parakeet"))
        for device in devices:
            supported = device["kind"] in kinds and model_matches
            installed = all(versions.get(package) for package in dependencies)
            present = (device["id"] in ov_ids or device["id"] == "CPU" and "OV:CPU" in ov_ids
                       if backend.startswith("openvino") else device["kind"] == "CUDA")
            state = "unverified"
            if device["kind"] == "NPU" and (env.get("npu_loss") or not env.get("loss_checked")):
                state = "quarantined" if env.get("npu_loss") else "unknown"
            prerequisites = bool(supported and installed and present and artifact.get("complete"))
            rows.append({"id": component + "/" + backend + "/" + device["id"],
                         "component": component, "backend": backend, "device": device["id"],
                         "model": model, "precision": artifact.get("precision"),
                         "supported": bool(supported), "dependencies_installed": bool(installed),
                         "artifacts_complete": bool(artifact.get("complete")),
                         "prerequisites": prerequisites, "validation": state,
                         "ready": prerequisites and device["kind"] != "NPU"})
    tts = observed.get("tts", {})
    # A service's CUDA index is in its own environment. Without a reported
    # UUID, do not associate it with the first NVIDIA card in the STT process.
    for kind in ("CPU", "CUDA", "GPU", "NPU"):
        reported = tts.get("status") == "ok" and tts.get("device") == kind
        deps = all(tts.get("versions", {}).get(p) for p in ("torch", "chatterbox-tts"))
        rows.append({"id": "tts/chatterbox/" + kind, "component": "tts", "backend": "chatterbox",
                     "device": "TTS:" + kind, "physical_id": tts.get("physical_id") if reported else None,
                     "model": tts.get("model"), "precision": tts.get("precision"),
                     "supported": kind in ("CPU", "CUDA"),
                     "dependencies_installed": bool(deps) if tts.get("status") == "ok" else None,
                     "artifacts_complete": bool(reported), "prerequisites": bool(reported and deps),
                     "validation": "loaded" if reported else "unknown", "ready": bool(reported and deps)})
    return devices, rows


def cache_key(observed):
    """Free memory/presence are refreshed; only stable facts key validation."""
    stable = json.loads(json.dumps(observed))
    for report in stable.values():
        report.pop("worker_pid", None)
    stable.get("tts", {}).pop("pid", None)
    for source in ("system", "nvidia", "openvino"):
        for device in stable.get(source, {}).get("devices", []):
            device.pop("memory_free", None)
        if "devices" in stable.get(source, {}):
            stable[source]["devices"].sort(key=lambda d: d["id"])
    stable.get("environment", {}).pop("npu_loss", None)
    data = json.dumps([POLICY_VERSION, stable], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def read_cache(path):
    """All malformed/old caches are disposable. Called off the startup thread."""
    try:
        with Path(path).open("rb") as file:
            raw = file.read(MAX_CACHE_BYTES + 1)
        if len(raw) > MAX_CACHE_BYTES:
            return None
        data = json.loads(raw)
        if (not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION
                or data.get("policy_version") != POLICY_VERSION
                or not isinstance(data.get("key"), str)
                or not isinstance(data.get("observed"), dict)
                or not isinstance(data.get("validations"), dict)
                or not isinstance(data.get("measurements"), dict)
                or not isinstance(data.get("created_at"), (int, float))
                or not math.isfinite(data["created_at"])
                or any(not isinstance(v, dict) or v.get("state") != "verified"
                       or not isinstance(v.get("checked_at"), (int, float))
                       or not math.isfinite(v["checked_at"]) for v in data["validations"].values())):
            return None
        return data
    except (OSError, ValueError, TypeError, RecursionError):
        return None


def make_inventory(observed, cached=None, *, now=None):
    now = time.time() if now is None else now
    devices, rows = capability_matrix(observed)
    key = cache_key(observed)
    unknown_driver = any(d["kind"] != "CPU" and not d.get("driver") for d in devices)
    cache_valid = bool(cached and cached["key"] == key and 0 <= now - cached["created_at"]
                       and (not unknown_driver or now - cached["created_at"] < UNKNOWN_DRIVER_TTL))
    validations = cached["validations"].copy() if cache_valid else {}
    for row in rows:
        check = validations.get(row["id"])
        if (row["validation"] in ("quarantined", "unknown") or
                check and (check["checked_at"] > now or unknown_driver and
                           now - check["checked_at"] >= UNKNOWN_DRIVER_TTL)):
            validations.pop(row["id"], None)
            check = None
        if check:
            row["validation"] = "verified"
            row["ready"] = row["prerequisites"]
    return {"schema_version": SCHEMA_VERSION, "policy_version": POLICY_VERSION,
            "key": key, "created_at": cached["created_at"] if cache_valid else now,
            "observed_at": now, "observed": observed, "devices": devices,
            "capabilities": rows, "validations": validations,
            "measurements": cached["measurements"] if cache_valid and not observed.get("environment", {}).get("npu_loss") else {},
            "cache_status": "hit" if cache_valid else "miss"}


def write_cache(path, inventory):
    """Atomic replacement; no partial JSON, no parent-thread disk writes."""
    import tempfile
    temporary = None
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(inventory, file, allow_nan=False)
        os.replace(temporary, path)
    except (OSError, ValueError):
        pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _query_process(source, request, deadline, events, cancelled):
    from debora_whisper.processes import NO_WINDOW, kill_tree, python_executable
    process = None
    reader = None
    try:
        if cancelled.is_set() or time.monotonic() >= deadline:
            return
        process = subprocess.Popen([python_executable(), "-m", __name__, source],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, creationflags=NO_WINDOW,
                                   # Not the launch folder: "-m" puts the
                                   # working directory first on sys.path.
                                   cwd=Path(__file__).resolve().parents[1])
        # A separate reader preserves completed enumeration if a later driver
        # property hangs. It only handles small JSON packets, never native code.
        def read():
            try:
                for line in process.stdout:
                    if len(line) <= MAX_CACHE_BYTES:
                        packet = json.loads(line)
                        if isinstance(packet, dict) and time.monotonic() < deadline:
                            events.put((source, {**packet, "worker_pid": getattr(process, "pid", None)}))
            except (OSError, ValueError):
                pass
            finally:
                process.stdout.close()
        # The reader must run even if cancelled, to close/drain the pipe after
        # process cleanup. Its parent stays registered until it is joined.
        reader = threading.Thread(target=read, name=f"inventory-reader-{source}", daemon=True)
        reader.start()
        if cancelled.is_set() or time.monotonic() >= deadline:
            return
        process.stdin.write(json.dumps(request).encode())
        process.stdin.close()
        while not cancelled.is_set() and time.monotonic() < deadline:
            try:
                process.wait(timeout=min(0.1, max(0, deadline - time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                continue
        while reader.is_alive() and not cancelled.is_set() and time.monotonic() < deadline:
            reader.join(timeout=min(0.1, max(0, deadline - time.monotonic())))
        if not reader.is_alive():
            events.put((source, None))
    except (OSError, subprocess.TimeoutExpired):
        pass
    finally:
        if process is not None:
            # kill_tree may stall in taskkill; stop_inventory_workers bounds
            # its join of this daemon worker during shutdown.
            kill_tree(process)
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            if reader is not None:
                reader.join(timeout=1.0)


def _collect(request, cache_path, deadline, state, done, cancelled):
    events = queue.Queue()
    observed = {}
    cached = None
    pending = set(SOURCES) | {"cache"}

    def cache_reader():
        events.put(("cache", read_cache(cache_path)))

    start_inventory_worker(cache_reader, cancelled=cancelled, name="inventory-cache")
    for source in SOURCES:
        start_inventory_worker(_query_process, (source, request, deadline, events, cancelled),
                               cancelled=cancelled, name=f"inventory-query-{source}")
    try:
        while pending and not cancelled.is_set() and time.monotonic() < deadline:
            try:
                source, value = events.get(timeout=min(0.1, max(0, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if source == "cache":
                cached = value
                pending.discard(source)
            elif value is None:
                pending.discard(source)
            else:
                observed = {**observed, source: value}
            state[0] = make_inventory(observed, cached)
        state[0] = {**state[0], "pending": sorted(pending | (set(SOURCES) - observed.keys()))}
    finally:
        done.set()
    # The cache never authorizes presence or free memory: only this session's
    # observed data is returned. Writing is also outside the startup wait.
    if not cancelled.is_set():
        write_cache(cache_path, state[0])


def probe_inventory(request, cache_path, *, timeout=PROBE_BUDGET, cancelled=None):
    started = time.monotonic()
    # Leave a little room for returning a snapshot; no OS hard-real-time promise.
    deadline = started + max(0, min(timeout, PROBE_BUDGET) - 0.010)
    state = [make_inventory({})]
    done = threading.Event()
    if cancelled is None:
        cancelled = threading.Event()
    start_inventory_worker(_collect, (request, cache_path, deadline, state, done, cancelled),
                           cancelled=cancelled)
    while not cancelled.is_set() and time.monotonic() < deadline:
        if done.wait(min(0.1, max(0, deadline - time.monotonic()))):
            break
    result = {**state[0]}
    result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 1)
    result["timed_out"] = not done.is_set() or bool(result.get("pending"))
    return result


def inventory_summary(inventory):
    names = ",".join(d["id"] + "=" + " ".join(str(d["name"]).split())[:80]
                     for d in inventory["devices"])
    ready = sum(row["ready"] for row in inventory["capabilities"])
    return (f"Hardware inventory: {names}; ready={ready}/{len(inventory['capabilities'])} "
            f"tts={inventory['observed'].get('tts', {}).get('status', 'unknown')} "
            f"cache={inventory['cache_status']} partial={inventory['timed_out']} "
            f"probe={inventory['elapsed_ms']:.1f}ms")


if __name__ == "__main__":
    request = json.load(sys.stdin)
    def emit(value):
        print(json.dumps(value), flush=True)
    try:
        _source(sys.argv[1], request, emit)
    except Exception:
        # Missing packages, unavailable tools and driver errors are unknown
        # observations, never startup errors or installation requests.
        emit({"status": "unknown"})
