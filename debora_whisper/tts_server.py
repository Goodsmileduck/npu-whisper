# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "chatterbox-tts==0.1.7", "torch==2.6.0", "torchaudio==2.6.0",
#     # resemble-perth (Chatterbox's watermark) imports pkg_resources,
#     # which setuptools 81 dropped.
#     "setuptools==80.9.0",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu124"
# url = "https://download.pytorch.org/whl/cu124"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu124" }
# torchaudio = { index = "pytorch-cu124" }
# ///
"""Local TTS server for debora-whisper's voice chat, backed by Chatterbox
Multilingual on an NVIDIA GPU.

It never runs in the app's environment: chatterbox-tts pins torch 2.6 (CUDA)
and numpy 1.26. The block above lets uv build its own environment on the first
run (a few GB, cached afterwards):

    uv run --script debora_whisper/tts_server.py --voice ref.wav

The app starts it this way by itself in voice chat mode, with the voice
from "tts_voice" in config.json, and stops it on exit.

Endpoints (127.0.0.1 only):
    GET  /health -> {"ok": true, "sample_rate": ...}
    POST /tts    {"text": "...", "language": "pt", "voice": "C:/voices/ana.wav"}
                 -> audio/wav (16-bit mono). "voice" is optional: "" is
                 Chatterbox's own voice, no "voice" the one from --voice.

Log lines look like app.log's: "[YYYY-MM-DD HH:MM:SS] message".
"""
import argparse
import io
import json
import logging
import os
import select
import socket
import sys
import threading
import time
import types
import warnings
import wave
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if __package__:
    from .download_progress import DownloadProgress, track_hf_downloads
else:  # uv run --script: sibling modules are available without the app.
    from download_progress import DownloadProgress, track_hf_downloads

MAX_TEXT_CHARS = 1000
MAX_BODY_BYTES = 64_000
STREAM_WRITE_TIMEOUT_SECONDS = 5.0
STREAM_INACTIVITY_SECONDS = 120.0
VOICE_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg")
# Prepared voices kept in memory (a few MB each on the GPU).
MAX_CACHED_VOICES = 8

log = logging.getLogger("tts_server").info


def setup_logging():
    """app.log's line format, without the libraries' noise: deprecation
    warnings, the HF token hint, per-sentence progress bars and Chatterbox's
    "forcing EOS token" notes."""
    # Libraries' INFO lines (each HTTP request to the hub, frame rates) stay
    # out; their warnings and errors stay in.
    logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.WARNING, stream=sys.stdout, force=True)
    logging.getLogger("tts_server").setLevel(logging.INFO)
    for name in ("huggingface_hub", "transformers", "diffusers", "chatterbox"):
        logging.getLogger(name).setLevel(logging.ERROR)
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.captureWarnings(True)
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


# Smart App Control blocks one of scikit-learn's DLLs, and nothing here calls
# it: while torch/transformers load it must look absent (they probe its spec),
# and librosa's later imports get empty stand-ins without DLLs.
def _unavailable(*a, **k):
    raise RuntimeError("scikit-learn is stubbed out in the TTS server")


def _stub_attr(attr):
    if attr.startswith("__"):
        raise AttributeError(attr)
    return _unavailable


def _stub_sklearn():
    for name in ("sklearn", "sklearn.decomposition", "sklearn.cluster",
                 "sklearn.feature_extraction", "sklearn.neighbors", "sklearn.metrics"):
        mod = types.ModuleType(name)
        mod.__path__ = []
        mod.__getattr__ = _stub_attr
        sys.modules[name] = mod
        if "." in name:
            setattr(sys.modules["sklearn"], name.split(".", 1)[1], mod)


# Chatterbox Multilingual reads the attention of these layers to keep speech
# aligned with the text (AlignmentStreamAnalyzer's LLAMA_ALIGNED_HEADS).
SPIED_LAYERS = (9, 12, 13)


def drop_attention_spies(model):
    """Each generate() adds 3 forward hooks that are never removed, each
    copying its layer's attention to the CPU on every token: remove the old
    ones before the next generate() adds its own. Unmeasurable after 20
    sentences (93 hooks), but they pile up for as long as the server runs."""
    for idx in SPIED_LAYERS:
        model.t3.tfmr.layers[idx].self_attn._forward_hooks.clear()


def cpu_name():
    """hardware_allocation.cpu_name(), which this environment cannot import."""
    import platform
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


def preload_cuda_libraries():
    """Load the pip-installed cuDNN (nvidia/cudnn/lib) globally, on Linux.

    cuDNN 9 opens its sub-libraries (libcudnn_graph...) by name and finds
    them through its own $ORIGIN. A library in LD_PRELOAD that wraps dlopen,
    such as NoMachine's libnxegl.so, becomes the caller instead and the
    lookup fails: torch aborted on its first cuDNN call. A library already
    loaded is found by name whoever asks."""
    if not sys.platform.startswith("linux"):
        return
    import ctypes
    import glob
    import site
    for root in site.getsitepackages():
        # cuDNN only: the rest is linked with torch's RUNPATH, and libnvblas
        # would take over the CPU's BLAS calls.
        pending = sorted(glob.glob(os.path.join(root, "nvidia", "cudnn", "lib", "lib*.so*")))
        # Dependencies first: retry what failed until no progress is made.
        while pending:
            failed = []
            for lib in pending:
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    failed.append(lib)
            if len(failed) == len(pending):
                break
            pending = failed


def load_model(device: str, voice: str | None):
    sys.modules["sklearn"] = None
    import torch
    import torch._dynamo  # noqa: F401
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    requested_device = device
    if device == "cuda" and not torch.cuda.is_available():
        log("CUDA not available; using the CPU (much slower).")
        device = "cpu"
    start = time.time()
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    # Report the device actually loaded, without another CUDA probe. The UI's
    # STT environment cannot infer this server's PyTorch capabilities.
    model._debora_device = device.upper()
    model._debora_allocation = {
        "pid": os.getpid(), "requested_device": requested_device.upper(),
        "device_id": "CPU", "device_name": cpu_name(),
        "fallback_reason": "CUDA unavailable in TTS worker" if device != requested_device else None,
    }
    if device == "cuda":
        model._debora_allocation.update(device_id="CUDA (index unreported)", device_name="name unavailable")
        try:
            index = torch.cuda.current_device()
            model._debora_allocation.update(device_id=f"CUDA:{index}",
                                            device_name=torch.cuda.get_device_name(index))
        except Exception:
            pass  # Optional identity queries must not invalidate a loaded model.
    _stub_sklearn()
    model.builtin_conds = model.conds
    if voice:
        model.prepare_conditionals(voice)
    log(f"Chatterbox loaded on {device} in {time.time() - start:.1f}s")
    return model


def to_wav(samples, sample_rate: int) -> bytes:
    import numpy as np
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return out.getvalue()


class Voices:
    """Chatterbox's conditionals per reference file, so each request can pick
    its voice without reloading the model. Call under the generation lock."""

    def __init__(self, model):
        self.model = model
        self.startup = model.conds
        self.cache = {}  # (path, mtime) -> conditionals

    def use(self, voice: str | None):
        """Point the model at voice: None keeps the startup voice, "" is
        Chatterbox's own, anything else a reference file."""
        if voice is None:
            conds = self.startup
        elif voice == "":
            conds = self.model.builtin_conds
        else:
            path = os.path.abspath(voice)
            key = (path, os.path.getmtime(path))
            conds = self.cache.get(key)
            if conds is None:
                start = time.time()
                self.model.prepare_conditionals(path)
                conds = self.model.conds
                if len(self.cache) >= MAX_CACHED_VOICES:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[key] = conds
                log(f"Voice {path} prepared in {time.time() - start:.1f}s")
        if conds is None:
            raise ValueError("no voice: pass a reference file")
        self.model.conds = conds


def check_voice(voice) -> str | None:
    """The request's "voice", or ValueError: a reference audio file, never
    anything else on disk."""
    if voice is None or voice == "":
        return voice
    if not isinstance(voice, str) or not voice.lower().endswith(VOICE_SUFFIXES):
        raise ValueError(f"voice must be one of {', '.join(VOICE_SUFFIXES)} files")
    # Opening \\host\share\x.wav would make Windows log in to that host with
    # the user's credentials (NTLM).
    if voice.startswith(("\\\\", "//")) or os.path.abspath(voice).startswith(("\\\\", "//")):
        raise ValueError("voice must be a local file, not a network path")
    if not os.path.isfile(voice):
        raise ValueError(f"voice {voice!r} not found")
    return voice


class SynthesisCancelled(Exception):
    pass


def check_connection(connection):
    try:
        readable, _, _ = select.select([connection], [], [], 0)
        if not readable or connection.recv(1, socket.MSG_PEEK):
            return
    except OSError:
        pass
    raise SynthesisCancelled()


@contextmanager
def cancellable_synthesis(model, connection, modules=None, *, check_after=True):
    """Check between speech tokens and flow steps; always remove our hooks."""
    last_check = 0.0
    def check(*args):
        nonlocal last_check
        # Adapters hook leaf modules, including callers that use forward()
        # directly. Bound socket polling overhead to once every 20 ms.
        now = time.monotonic()
        if modules is not None and args and now - last_check < 0.02:
            return
        last_check = now
        check_connection(connection)

    hooks = []
    try:
        check()
        if modules is not None:
            for module in modules:
                hooks.append(module.register_forward_pre_hook(check))
        # Hook inner modules: Chatterbox calls the transformer/estimator's
        # forward directly, bypassing hooks on those outer modules.
        for path in ("t3.tfmr.layers.0", "s3gen.flow.decoder.estimator.time_mlp"):
            try:
                module = model
                for name in path.split("."):
                    module = module[int(name)] if name.isdigit() else getattr(module, name)
                hooks.append(module.register_forward_pre_hook(check))
            except (AttributeError, IndexError, TypeError):
                pass  # another Chatterbox layout: synthesize without cancelling
        yield
        if check_after:
            check()
    finally:
        for hook in hooks:
            hook.remove()


def make_handler(model, default_language: str, *, adapter=False, error=None):
    # One generation at a time: the GPU is shared and Chatterbox is not
    # thread-safe.
    lock = threading.Lock()
    voices = model if adapter else Voices(model)
    from importlib.metadata import PackageNotFoundError, version
    versions = {}
    for package in (("torch", model.package) if adapter else ("torch", "chatterbox-tts")):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    inventory = {"schema_version": 1, "device": getattr(model, "_debora_device", None),
                 "versions": versions, "loaded": error is None,
                 "model": model.model_id if adapter else "ChatterboxMultilingualTTS",
                 "precision": getattr(model, "precision", "unknown") if adapter else "unknown"}
    inventory.update(getattr(model, "_debora_allocation", {}))

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _reply(self, code, body: bytes, content_type="application/json"):
            self.close_connection = True
            try:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass  # the client interrupted this request

        def _error(self, code, message):
            self._reply(code, json.dumps({"error": message}).encode("utf-8"))

        def do_GET(self):
            if self.path != "/health":
                return self._error(404, "not found")
            if error:
                return self._reply(503, json.dumps({"ok": False, "error": error,
                                                   "inventory": inventory}).encode())
            self._reply(200, json.dumps({"ok": True, "sample_rate": model.sr,
                                        "stream": callable(getattr(model, "generate_stream", None)),
                                        "inventory": inventory}).encode())

        def _stream(self, text, language, start):
            import numpy as np
            # Slow CPU synthesis may take minutes while still making progress.
            # Bound inactivity, including blocked generation and socket writes.
            deadline = time.monotonic() + STREAM_INACTIVITY_SECONDS
            progress = threading.Condition()
            finished = False

            def bound_write():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SynthesisCancelled()
                # On Windows, shutdown from another thread may not wake the
                # select inside sendall; its own timeout must obey the deadline.
                self.connection.settimeout(min(STREAM_WRITE_TIMEOUT_SECONDS, remaining))

            def shutdown():
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

            def watch():
                with progress:
                    while not finished:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            shutdown()
                            return
                        progress.wait(remaining)

            watchdog = threading.Thread(target=watch, daemon=True)
            watchdog.start()
            chunks = None
            total = 0
            first = None
            try:
                bound_write()
                chunks = model.generate_stream(text, language_id=language)
                for chunk in chunks:
                    if time.monotonic() >= deadline:
                        raise SynthesisCancelled()
                    check_connection(self.connection)
                    if hasattr(chunk, "cpu"):
                        chunk = chunk.float().cpu().numpy()
                    samples = np.asarray(chunk).reshape(-1)
                    if not len(samples):
                        continue
                    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
                    bound_write()
                    if first is None:
                        first = time.time() - start
                        self.stream_headers = True
                        try:
                            self.send_response(200)
                            self.send_header("Content-Type", f"audio/L16; rate={model.sr}; channels=1")
                            self.send_header("X-Sample-Rate", str(model.sr))
                            self.send_header("Transfer-Encoding", "chunked")
                            self.send_header("Connection", "close")
                            self.end_headers()
                        except OSError as e:
                            raise SynthesisCancelled() from e
                    try:
                        bound_write()
                        self.wfile.write(f"{len(pcm):X}\r\n".encode() + pcm + b"\r\n")
                        self.wfile.flush()
                    except OSError as e:
                        raise SynthesisCancelled() from e
                    total += len(samples)
                    with progress:
                        deadline = time.monotonic() + STREAM_INACTIVITY_SECONDS
                if first is None:
                    raise ValueError("generation returned no audio")
                check_connection(self.connection)
                bound_write()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (OSError, SynthesisCancelled) as e:
                shutdown()
                raise SynthesisCancelled() from e
            finally:
                try:
                    if hasattr(chunks, "close"):
                        chunks.close()
                finally:
                    with progress:
                        finished = True
                        progress.notify_all()
                    watchdog.join()
            log(f"{total / model.sr:.1f}s of audio in {time.time() - start:.1f}s "
                f"(first chunk {first:.3f}s): {text[:300]!r}")

        def do_POST(self):
            self.close_connection = True
            self.stream_headers = False
            if self.path != "/tts":
                return self._error(404, "not found")
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._error(400, "invalid Content-Length")
            if length <= 0 or length > MAX_BODY_BYTES:
                return self._error(413, "body missing or too large")
            # Read the body even when refusing it: closing a socket with
            # unread data makes Windows reset the connection, and the client
            # sees that instead of the 403.
            body = self.rfile.read(length)
            # Web pages can POST to 127.0.0.1 too: browsers send an Origin,
            # and a JSON body would need a CORS preflight this server never
            # answers.
            if self.headers.get("Origin") is not None or not (
                    self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._error(403, "only local JSON clients")
            if error:
                return self._error(503, error)
            try:
                request = json.loads(body.decode("utf-8"))
                text = str(request["text"]).strip()[:MAX_TEXT_CHARS]
                language = str(request.get("language") or default_language)
                voice = check_voice(request.get("voice"))
            except Exception as e:
                return self._error(400, f"bad request: {e}")
            if not text:
                return self._error(400, "empty text")
            import torch
            with lock:
                start = time.time()
                try:
                    check_connection(self.connection)
                    voices.use(voice)
                except SynthesisCancelled:
                    log("Cancelled queued synthesis (client disconnected)")
                    return
                except Exception as e:
                    log(f"Voice {voice!r} unusable: {type(e).__name__}: {e}")
                    return self._error(400, f"voice unusable: {e}")
                try:
                    if not adapter:
                        drop_attention_spies(model)
                    modules = model.cancel_modules if adapter else None
                    streaming = (request.get("stream") is True
                                 and callable(getattr(model, "generate_stream", None)))
                    # _stream checks before sending EOF. A client may close as
                    # soon as it sees EOF, which is successful completion.
                    with cancellable_synthesis(model, self.connection, modules,
                                               check_after=not streaming), torch.inference_mode():
                        # Reference encoding also needs cancellation on new voices.
                        if adapter:
                            model.prepare_voice()
                        if streaming:
                            self._stream(text, language, start)
                        else:
                            wav = model.generate(text, language_id=language)
                    if streaming:
                        return
                except SynthesisCancelled:
                    log(f"Cancelled synthesis after {time.time() - start:.1f}s "
                        "(client disconnected or stream timed out)")
                    return
                except Exception as e:
                    # Chatterbox fails on text too short to speak ("OK").
                    log(f"Failed after {time.time() - start:.1f}s on {text[:300]!r}: "
                        f"{type(e).__name__}: {e}")
                    if not self.stream_headers:
                        self._error(500, f"generation failed: {e}")
                    return
            samples = wav.squeeze(0).float().cpu().numpy()
            log(f"{len(samples) / model.sr:.1f}s of audio in {time.time() - start:.1f}s: "
                f"{text[:300]!r}")
            self._reply(200, to_wav(samples, model.sr), "audio/wav")

        def log_message(self, fmt, *args):
            pass

    return Handler


def loading_server(address, loader, language, *, adapter=True):
    """Bind before loading; atomically publish the finished request handler."""
    progress = DownloadProgress()

    class LoadingHandler(BaseHTTPRequestHandler):
        def reply(self, body):
            raw = json.dumps(body).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path != "/health":
                return self.send_error(404)
            self.reply({"ok": False, "state": "loading", "download": progress.snapshot()})

        def do_POST(self):
            if self.path != "/tts":
                return self.send_error(404)
            self.reply({"error": "loading"})

        def log_message(self, *args):
            pass

    state = types.SimpleNamespace(handler=LoadingHandler)

    def dispatch(*args, **kwargs):
        state.handler(*args, **kwargs)

    server = ThreadingHTTPServer(address, dispatch)

    def load():
        try:
            preload_cuda_libraries()
            with track_hf_downloads(progress):
                model, error = loader()
            handler = make_handler(model, language, adapter=adapter, error=error)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log(f"TTS startup failed: {error}")
            model = types.SimpleNamespace(package="unknown", model_id="unknown")
            handler = make_handler(model, language, adapter=True, error=error)
        state.handler = handler

    threading.Thread(target=load, daemon=True).start()
    return server


def serve(server):
    log(f"Listening on http://127.0.0.1:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Stopped.")
    finally:
        server.server_close()


def backend_main(loader, package, model_id, argv=None, notice=None):
    """Shared entry point for isolated uv scripts; no app imports required.
    notice (license terms, for example) ends --help."""
    parser = argparse.ArgumentParser(description=f"Local TTS: {model_id}", epilog=notice)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--voice", help="Local reference WAV (Qwen requires its .txt sidecar)")
    parser.add_argument("--language", default="pt")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args(argv)
    setup_logging()
    serve(loading_server(("127.0.0.1", args.port),
                         lambda: load_backend(loader, args.device, args.voice, package, model_id),
                         args.language))


def load_backend(loader, device, voice, package, model_id):
    """Keep /health available with an actionable error after startup failure."""
    try:
        start = time.time()
        model = loader(device, voice)
        log(f"{model_id} loaded on {model._debora_device} in {time.time() - start:.1f}s")
        return model, None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log(f"TTS startup failed: {error}")
        return types.SimpleNamespace(package=package, model_id=model_id), error


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--voice", help="Reference audio to clone (wav/flac, ~10 s)")
    parser.add_argument("--language", default="pt", help="Default language id")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    setup_logging()
    log(f"Starting (pid {os.getpid()}, voice {args.voice or 'default'}, "
        f"language {args.language})")
    serve(loading_server(("127.0.0.1", args.port),
                         lambda: (load_model(args.device, args.voice), None),
                         args.language, adapter=False))


if __name__ == "__main__":
    main()
