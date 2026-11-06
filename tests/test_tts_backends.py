"""Backend contracts: no model packages or weights are downloaded."""
import importlib.util
import http.client
import io
import json
import os
from pathlib import Path
import runpy
import socket
import sys
import threading
import time
import urllib.error
from types import SimpleNamespace
from unittest.mock import Mock
import wave
from contextlib import nullcontext
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from debora_whisper import tts_server as server
from debora_whisper import tts_server_kokoro as kokoro
from debora_whisper import tts_server_qwen3 as qwen
from debora_whisper import tts_server_xtts as xtts
from debora_whisper import voice_chat as vc
from debora_whisper.dictation_engine import DEFAULT_CONFIG, validate_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def streaming_server(fake_torch):
    running = []

    def start(chunks=None):
        backend = SimpleNamespace(sr=24000, package="fake-tts", model_id="fake",
                                  cancel_modules=[], use=Mock(), prepare_voice=Mock(),
                                  generate=lambda *a, **kw: FakeTensor([[0.25, -0.25]]))
        if chunks is not None:
            backend.generate_stream = chunks
        srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(backend, "pt", adapter=True))
        srv.daemon_threads = False
        worker = threading.Thread(target=srv.serve_forever)
        worker.start()
        running.append((srv, worker))
        return srv, backend

    yield start
    for srv, worker in running:
        srv.shutdown()
        srv.server_close()
        worker.join()


def test_streaming_http_contract(streaming_server):
    chunks = [np.array([0.25, -0.25]), np.array([1.0, -1.0, 0.0])]
    srv, backend = streaming_server(lambda *a, **kw: iter(chunks))
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=3)
    try:
        conn.request("GET", "/health")
        health = conn.getresponse()
        assert json.loads(health.read())["stream"] is True
        conn.request("POST", "/tts", json.dumps({"text": "Olá.", "stream": True}),
                     {"Content-Type": "application/json"})
        reply = conn.getresponse()
        assert reply.status == 200 and reply.chunked
        assert reply.getheader("Content-Type") == "audio/L16; rate=24000; channels=1"
        assert reply.getheader("X-Sample-Rate") == "24000"
        # Inspect framing too: exactly one HTTP chunk per generated chunk.
        for samples in chunks:
            pcm = (samples * 32767).astype("<i2").tobytes()
            assert reply.fp.readline() == f"{len(pcm):X}\r\n".encode()
            assert reply.fp.read(len(pcm) + 2) == pcm + b"\r\n"
        assert reply.fp.read() == b"0\r\n\r\n"
        reply.close()
        backend.use.assert_called_once_with(None)
        backend.prepare_voice.assert_called_once()
    finally:
        conn.close()


def test_stream_flag_falls_back_to_wav(streaming_server):
    srv, _ = streaming_server()
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=3)
    try:
        conn.request("GET", "/health")
        assert json.loads(conn.getresponse().read())["stream"] is False
        conn.request("POST", "/tts", '{"text":"Olá", "stream":true}'.encode(),
                     {"Content-Type": "application/json"})
        reply = conn.getresponse()
        assert reply.status == 200 and reply.getheader("Content-Type") == "audio/wav"
        with wave.open(io.BytesIO(reply.read())) as wav:
            assert wav.getnframes() == 2
    finally:
        conn.close()


def test_successful_stream_does_not_log_client_close_as_cancellation(streaming_server, monkeypatch):
    logs = []
    complete, finished = threading.Event(), threading.Event()
    srv, _ = streaming_server(lambda *a, **kw: iter([np.ones(2400, np.float32)]))
    original_stream = srv.RequestHandlerClass._stream
    original_finish = srv.RequestHandlerClass.finish
    original_check = server.check_connection

    def stream(handler, *args):
        original_stream(handler, *args)
        complete.set()  # client can disconnect immediately after receiving EOF

    def check(connection):
        if complete.is_set():
            raise server.SynthesisCancelled()
        original_check(connection)

    def finish(handler):
        try:
            original_finish(handler)
        finally:
            finished.set()

    monkeypatch.setattr(srv.RequestHandlerClass, "_stream", stream)
    monkeypatch.setattr(srv.RequestHandlerClass, "finish", finish)
    monkeypatch.setattr(server, "check_connection", check)
    monkeypatch.setattr(server, "log", logs.append)
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=3)
    try:
        conn.request("POST", "/tts", '{"text":"Hello", "stream":true}',
                     {"Content-Type": "application/json"})
        assert len(conn.getresponse().read()) == 4800
        assert finished.wait(3) and complete.is_set()
        assert not any("Cancelled" in line for line in logs)
    finally:
        conn.close()


@pytest.mark.parametrize("after_first", [False, True])
def test_stream_generation_error_framing(streaming_server, after_first):
    def chunks(*args, **kwargs):
        if after_first:
            yield np.ones(100)
        raise ValueError("broken generation")

    srv, _ = streaming_server(chunks)
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=3)
    try:
        conn.request("POST", "/tts", '{"text":"Hello", "stream":true}',
                     {"Content-Type": "application/json"})
        reply = conn.getresponse()
        if after_first:
            assert reply.status == 200
            with pytest.raises(http.client.IncompleteRead):
                reply.read()
        else:
            assert reply.status == 500
            assert "broken generation" in json.loads(reply.read())["error"]
    finally:
        conn.close()


@pytest.mark.parametrize("headers", [{"Origin": "null"}, {"Content-Type": "text/plain"}])
def test_stream_keeps_origin_and_content_type_checks(streaming_server, headers):
    chunks = Mock()
    srv, _ = streaming_server(chunks)
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=3)
    try:
        conn.request("POST", "/tts", '{"text":"Hello", "stream":true}',
                     {"Content-Type": "application/json", **headers})
        reply = conn.getresponse()
        assert reply.status == 403
        reply.read()
        chunks.assert_not_called()
    finally:
        conn.close()


def test_stream_disconnect_closes_generator(streaming_server, monkeypatch):
    resume, closed = threading.Event(), threading.Event()
    produced, logs = [], []
    monkeypatch.setattr(server, "log", logs.append)

    def chunks(*args, **kwargs):
        try:
            for index in range(100):
                produced.append(index)
                yield np.ones(2400)
                assert resume.wait(3)
        finally:
            closed.set()

    srv, _ = streaming_server(chunks)
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=3)
    try:
        conn.connect()
        sock = conn.sock
        conn.request("POST", "/tts", '{"text":"Hello", "stream":true}',
                     {"Content-Type": "application/json"})
        reply = conn.getresponse()
        assert reply.read1(4800)
        sock.shutdown(socket.SHUT_RDWR)
        reply.close()
    finally:
        conn.close()
        resume.set()
    assert closed.wait(3)
    assert len(produced) < 100
    # server_close in the fixture joins the request thread.
    srv.shutdown()
    srv.server_close()
    assert any("Cancelled synthesis" in line for line in logs)


@pytest.mark.parametrize("limit", ["write", "deadline"])
def test_stalled_stream_releases_synthesis_lock(streaming_server, monkeypatch, limit):
    monkeypatch.setattr(server, "STREAM_WRITE_TIMEOUT_SECONDS", 0.2 if limit == "write" else 10)
    monkeypatch.setattr(server, "STREAM_INACTIVITY_SECONDS", 10 if limit == "write" else 0.2)
    began, closed, cleanup = threading.Event(), threading.Event(), threading.Event()

    def chunks(*args, **kwargs):
        try:
            samples = np.ones(512 * 1024, dtype=np.float32)
            while not cleanup.is_set():
                began.set()
                yield samples
        finally:
            closed.set()

    srv, _ = streaming_server(chunks)
    setup = srv.RequestHandlerClass.setup

    def small_buffer(handler):
        setup(handler)
        handler.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)

    monkeypatch.setattr(srv.RequestHandlerClass, "setup", small_buffer)
    stalled = socket.socket()
    stalled.settimeout(3)
    stalled.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    next_request = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=3)
    try:
        stalled.connect(("127.0.0.1", srv.server_port))
        body = b'{"text":"Hello", "stream":true}'
        stalled.sendall((f"POST /tts HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                         f"Content-Length: {len(body)}\r\n\r\n").encode() + body)
        assert began.wait(3)
        next_request.request("POST", "/tts", '{"text":"Next"}', {"Content-Type": "application/json"})
        # The first peer remains connected and never reads, even headers.
        assert closed.wait(3), "stalled socket retained the generator and synthesis lock"
        reply = next_request.getresponse()
        assert reply.status == 200 and reply.read().startswith(b"RIFF")
        # Expiry must close the first connection without a successful chunk terminator.
        received = bytearray()
        while data := stalled.recv(65536):
            received.extend(data)
        assert received and not received.endswith(b"0\r\n\r\n")
    finally:
        cleanup.set()
        stalled.close()
        next_request.close()


def test_stream_inactivity_allows_continuous_progress(streaming_server, monkeypatch):
    monkeypatch.setattr(server, "STREAM_WRITE_TIMEOUT_SECONDS", 10)
    monkeypatch.setattr(server, "STREAM_INACTIVITY_SECONDS", 0.3)
    closed, cleanup = threading.Event(), threading.Event()

    def chunks(*args, **kwargs):
        try:
            for _ in range(25):
                yield np.ones(2400, dtype=np.float32)
                cleanup.wait(0.02)
        finally:
            closed.set()

    srv, _ = streaming_server(chunks)
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=3)
    reply = None
    try:
        began = time.monotonic()
        conn.request("POST", "/tts", '{"text":"Hello", "stream":true}',
                     {"Content-Type": "application/json"})
        reply = conn.getresponse()
        assert reply.status == 200
        assert len(reply.read()) == 25 * 2400 * 2
        assert closed.wait(3)
        assert 0.3 < time.monotonic() - began < 3
        conn.request("POST", "/tts", '{"text":"Next"}', {"Content-Type": "application/json"})
        response = conn.getresponse()
        assert response.status == 200 and response.read().startswith(b"RIFF")
    finally:
        cleanup.set()
        if reply is not None:
            reply.close()
        conn.close()


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def shape(self):
        return self.values.shape

    def reshape(self, *shape):
        return FakeTensor(self.values.reshape(*shape))

    def squeeze(self, axis):
        return FakeTensor(self.values.squeeze(axis))

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeModule:
    def __init__(self):
        self.hooks = []

    def modules(self):
        return [self]

    def children(self):
        return []

    def register_forward_pre_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))

    def __call__(self):
        for hook in self.hooks:
            hook(self, ())


@pytest.fixture
def fake_torch(monkeypatch):
    # The app/test environment intentionally has no torch dependency.
    fake = SimpleNamespace(
        float32="torch.float32", float16="torch.float16", bfloat16="torch.bfloat16",
        cuda=SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False),
        inference_mode=nullcontext, as_tensor=FakeTensor,
        cat=lambda chunks: FakeTensor(np.concatenate([chunk.values for chunk in chunks])))
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


@pytest.mark.parametrize("backend,filename", [
    ("chatterbox", "tts_server.py"), ("qwen3", "tts_server_qwen3.py"),
    ("kokoro", "tts_server_kokoro.py"), ("xtts", "tts_server_xtts.py"),
])
def test_backend_selection(monkeypatch, backend, filename):
    monkeypatch.setattr(vc.shutil, "which", lambda _: "uv.exe")
    config = {**DEFAULT_CONFIG, "tts_backend": backend}
    validate_config(config)
    command = vc.tts_command(config)
    assert command[:3] == ["uv.exe", "run", "--script"]
    assert Path(command[3]).name == filename
    assert Path(command[3]).is_file()
    assert ("--voice" in command) == (backend != "kokoro")


def test_default_and_legacy_configs_keep_chatterbox(monkeypatch):
    assert DEFAULT_CONFIG["tts_backend"] == "chatterbox"
    monkeypatch.setattr(vc.shutil, "which", lambda _: "uv.exe")
    config = {k: v for k, v in DEFAULT_CONFIG.items() if k != "tts_backend"}
    validate_config(config)
    assert Path(vc.tts_command(config)[3]).name == "tts_server.py"


@pytest.mark.parametrize("backend", [None, "qwen3-tts", "", "unknown", 1, [], {}])
def test_invalid_backend_rejected(backend):
    with pytest.raises(ValueError, match="tts_backend"):
        validate_config({**DEFAULT_CONFIG, "tts_backend": backend})


@pytest.mark.parametrize("backend", ["qwen3", "xtts"])
def test_clone_only_missing_voice_is_not_silently_replaced(monkeypatch, tmp_path, backend):
    monkeypatch.setattr(vc.shutil, "which", lambda _: "uv.exe")
    missing = tmp_path / "missing.wav"
    command = vc.tts_command({**DEFAULT_CONFIG, "tts_backend": backend, "tts_voice": str(missing)})
    assert command[command.index("--voice") + 1] == str(missing)


def response(handler_class, path="/health", body=None, headers=None):
    """Exercise HTTP handlers without binding a socket or starting a server."""
    handler = handler_class.__new__(handler_class)
    handler.path = path
    handler._reply = Mock()
    handler.connection = object()
    if body is None:
        handler.do_GET()
    else:
        payload = json.dumps(body).encode()
        handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(payload)),
                           **(headers or {})}
        handler.rfile = io.BytesIO(payload)
        handler.do_POST()
    return handler._reply.call_args.args


@pytest.mark.parametrize("transcript", [None, "", "  \n"])
def test_missing_or_empty_transcript_fails_at_health_before_model_load(tmp_path, monkeypatch, transcript):
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"reference")
    if transcript is not None:
        voice.with_suffix(".txt").write_text(transcript, encoding="utf-8")
    load = Mock(side_effect=AssertionError("must not load a model"))
    monkeypatch.setitem(sys.modules, "qwen_tts", SimpleNamespace(Qwen3TTSModel=SimpleNamespace(from_pretrained=load)))
    model, error = server.load_backend(qwen.QwenBackend, "cpu", str(voice), "qwen-tts", qwen.MODEL_ID)
    code, body = response(server.make_handler(model, "pt", adapter=True, error=error))
    health = json.loads(body)
    assert code == 503 and not health["ok"]
    assert "voice.txt" in health["error"] and "exact words" in health["error"]
    assert not health["inventory"]["loaded"]
    load.assert_not_called()


def test_missing_espeak_fails_at_health_before_model_load(monkeypatch):
    monkeypatch.setitem(sys.modules, "misaki.espeak", SimpleNamespace(
        EspeakG2P=Mock(side_effect=RuntimeError("espeak not installed"))))
    load = Mock(side_effect=AssertionError("must not load a model"))
    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=load))
    model, error = server.load_backend(kokoro.KokoroBackend, "cpu", None, "kokoro", kokoro.MODEL_ID)
    handler = server.make_handler(model, "pt", adapter=True, error=error)
    code, body = response(handler)
    assert code == 503
    assert "espeakng-loader" in json.loads(body)["error"]
    assert "Smart App Control" in json.loads(body)["error"]
    assert response(handler, "/tts", {"text": "Olá"})[0] == 503
    load.assert_not_called()


@pytest.fixture
def qwen_model(monkeypatch, tmp_path, fake_torch):
    inner = FakeModule()
    inner.speech_tokenizer = SimpleNamespace(model=FakeModule())
    model = Mock(model=inner)
    model.generate_voice_clone.return_value = ([np.zeros(2400, dtype=np.float32)], 24000)
    load = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "qwen_tts", SimpleNamespace(Qwen3TTSModel=SimpleNamespace(from_pretrained=load)))
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"reference")
    voice.with_suffix(".txt").write_text("Olá, referência confirmada.", encoding="utf-8")
    backend = qwen.QwenBackend("cpu", str(voice))
    return backend, model, load, voice


def test_qwen_uses_exact_transcript_and_caches_prompt(qwen_model):
    backend, model, load, voice = qwen_model
    assert load.call_args.args == (qwen.MODEL_ID,)
    assert load.call_args.kwargs["attn_implementation"] == "sdpa"
    model.create_voice_clone_prompt.assert_called_once_with(
        ref_audio=str(voice.resolve()), ref_text="Olá, referência confirmada.", x_vector_only_mode=False)
    backend.use(None)
    backend.prepare_voice()
    assert model.create_voice_clone_prompt.call_count == 1
    assert backend.generate("Olá, mundo.", "pt-BR").shape == (1, 2400)
    assert model.generate_voice_clone.call_args.kwargs["language"] == "Portuguese"
    assert model.generate_voice_clone.call_args.kwargs["non_streaming_mode"] is True
    voice.with_suffix(".txt").write_text("Texto corrigido.", encoding="utf-8")
    backend.use(str(voice))
    backend.prepare_voice()
    assert model.create_voice_clone_prompt.call_args.kwargs["ref_text"] == "Texto corrigido."
    with pytest.raises(ValueError, match="requires tts_voice"):
        backend.use("")


@pytest.mark.parametrize("transcript,limit", [
    (b"x" * (qwen.MAX_TRANSCRIPT_BYTES + 1), "bytes"),
    (("á" * (qwen.MAX_TRANSCRIPT_CHARS + 1)).encode(), "characters"),
])
def test_qwen_rejects_oversized_transcripts_before_model_processing(qwen_model, monkeypatch,
                                                                 transcript, limit):
    backend, model, _, voice = qwen_model
    model.reset_mock()
    voice.with_suffix(".txt").write_bytes(transcript)
    monkeypatch.setattr(server, "check_connection", lambda _: None)
    handler = server.make_handler(backend, "pt", adapter=True)
    code, body = response(handler, "/tts", {"text": "Olá", "voice": str(voice)})
    assert code == 400 and limit in json.loads(body)["error"]
    model.create_voice_clone_prompt.assert_not_called()
    model.generate_voice_clone.assert_not_called()
    # The rejected request releases the shared lock and leaves synthesis usable.
    voice.with_suffix(".txt").write_text("Referência corrigida.", encoding="utf-8")
    assert response(handler, "/tts", {"text": "Olá"})[0] == 200


def test_qwen_transcript_read_is_bounded(qwen_model, monkeypatch):
    _, _, _, voice = qwen_model
    stream = io.BytesIO(b"x" * (qwen.MAX_TRANSCRIPT_BYTES + 100))
    read = Mock(wraps=stream.read)
    stream.read = read
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: stream)
    with pytest.raises(ValueError, match="bytes"):
        qwen.reference(str(voice))
    read.assert_called_once_with(qwen.MAX_TRANSCRIPT_BYTES + 1)


def test_qwen_transcript_limits_accept_utf8_bom_at_boundary(qwen_model):
    _, _, _, voice = qwen_model
    text = "á" * qwen.MAX_TRANSCRIPT_CHARS
    voice.with_suffix(".txt").write_text(text, encoding="utf-8-sig")
    assert qwen.reference(str(voice))[1] == text


def test_kokoro_fixed_voice_and_complete_audio(monkeypatch, fake_torch):
    phonemizer = Mock(return_value=Mock(return_value=("ola", None)))
    monkeypatch.setitem(sys.modules, "misaki.espeak", SimpleNamespace(EspeakG2P=phonemizer))
    pipeline = Mock(model=FakeModule())
    pipeline.return_value = [("Olá", "ola", FakeTensor(np.zeros(1200))),
                             ("mundo", "mundo", FakeTensor(np.zeros(2400)))]
    load = Mock(return_value=pipeline)
    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=load))
    backend = kokoro.KokoroBackend("cpu", "ignored.wav")
    phonemizer.assert_called_once_with(language="pt-br")
    load.assert_called_once_with(lang_code="p", repo_id=kokoro.MODEL_ID, device="cpu")
    backend.use("another.wav")
    backend.prepare_voice()
    assert backend.generate("Olá, mundo.", "pt").shape == (1, 3600)
    pipeline.assert_called_once_with("Olá, mundo.", voice="pf_dora")


def fake_coqui(monkeypatch, model, downloads=None):
    """Stand-ins for coqui-tts's modules: nothing is imported or downloaded."""
    manager = Mock()
    manager.download_model.side_effect = (
        lambda name: (downloads if downloads is not None else []).append(name) or ("C:/xtts", None, {}))
    monkeypatch.setattr(xtts, "report_coqui_downloads", lambda: manager)
    config = Mock()
    modules = {
        "TTS": SimpleNamespace(), "TTS.tts": SimpleNamespace(), "TTS.tts.configs": SimpleNamespace(),
        "TTS.tts.models": SimpleNamespace(),
        "TTS.tts.configs.xtts_config": SimpleNamespace(XttsConfig=Mock(return_value=config)),
        "TTS.tts.models.xtts": SimpleNamespace(Xtts=SimpleNamespace(init_from_config=Mock(return_value=model))),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return manager, config


@pytest.fixture
def xtts_model(monkeypatch, tmp_path, fake_torch):
    monkeypatch.delenv("COQUI_TOS_AGREED", raising=False)
    model = Mock()
    model.modules.return_value = [FakeModule()]
    model.get_conditioning_latents.side_effect = lambda audio_path: (f"lat:{audio_path[0]}", "spk")
    model.inference.return_value = {"wav": np.full(2400, 0.1, dtype=np.float32)}
    downloads = []
    _, config = fake_coqui(monkeypatch, model, downloads)
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"reference")
    backend = xtts.XttsBackend("cpu", str(voice))
    return backend, model, voice, downloads, config


def test_xtts_loads_once_clones_and_caches_latents(xtts_model):
    backend, model, voice, downloads, config = xtts_model
    # Choosing the backend accepts the non-commercial CPML: no stdin prompt.
    assert os.environ["COQUI_TOS_AGREED"] == "1"
    assert downloads == [xtts.MODEL_ID]
    config.load_json.assert_called_once_with(os.path.join("C:/xtts", "config.json"))
    model.load_checkpoint.assert_called_once_with(config, checkpoint_dir="C:/xtts", use_deepspeed=False)
    model.cuda.assert_not_called()
    assert backend._debora_device == "CPU"
    model.get_conditioning_latents.assert_called_once_with(audio_path=[str(voice.resolve())])
    backend.use(None)
    backend.prepare_voice()
    backend.use(str(voice))
    backend.prepare_voice()
    assert model.get_conditioning_latents.call_count == 1
    assert backend.generate("Olá, mundo.", "pt-BR").shape == (1, 2400)
    args, kwargs = model.inference.call_args
    assert args == ("Olá, mundo", "pt", f"lat:{voice.resolve()}", "spk")
    assert kwargs == {"temperature": 0.45, "enable_text_splitting": False}
    with pytest.raises(ValueError, match="no voice of its own"):
        backend.use("")


def test_xtts_http_synthesis_returns_24khz_wav(xtts_model, monkeypatch):
    backend, model, voice, _, _ = xtts_model
    monkeypatch.setattr(server, "check_connection", lambda _: None)
    handler = server.make_handler(backend, "pt", adapter=True)
    code, body = response(handler)
    health = json.loads(body)
    assert code == 200 and health["sample_rate"] == 24000
    assert health["inventory"]["model"] == xtts.MODEL_ID
    code, wav, content_type = response(handler, "/tts", {"text": "Olá!", "voice": str(voice)})
    assert code == 200 and content_type == "audio/wav"
    with wave.open(io.BytesIO(wav), "rb") as audio:
        assert audio.getnframes() == 2400 and audio.getframerate() == 24000
    assert model.inference.call_args.args[:2] == ("Olá", "pt")
    code, body = response(handler, "/tts", {"text": "Olá", "voice": ""})
    assert code == 400 and "no voice of its own" in json.loads(body)["error"]
    code, body = response(handler, "/tts", {"text": "Hej", "language": "sv"})
    assert code == 500 and "does not speak language" in json.loads(body)["error"]


def test_xtts_voice_cache_is_bounded_and_follows_file_changes(xtts_model, tmp_path):
    backend, model, _, _, _ = xtts_model
    others = []
    for i in range(xtts.MAX_CACHED_VOICES):
        other = tmp_path / f"other{i}.wav"
        other.write_bytes(b"reference")
        others.append(other)
        backend.use(str(other))
        backend.prepare_voice()
    assert len(backend.cache) == xtts.MAX_CACHED_VOICES
    calls = model.get_conditioning_latents.call_count
    # The startup voice was the least recently used one: evicted, prepared again.
    backend.use(None)
    backend.prepare_voice()
    assert model.get_conditioning_latents.call_count == calls + 1
    # A recently used voice stays cached...
    backend.use(str(others[-1]))
    backend.prepare_voice()
    assert model.get_conditioning_latents.call_count == calls + 1
    # ...until its file changes.
    stat = os.stat(others[-1])
    os.utime(others[-1], ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    backend.use(str(others[-1]))
    backend.prepare_voice()
    assert model.get_conditioning_latents.call_count == calls + 2
    assert len(backend.cache) == xtts.MAX_CACHED_VOICES


@pytest.mark.parametrize("voice", [None, "", "missing.wav"])
def test_xtts_missing_voice_fails_at_health_before_model_load(monkeypatch, tmp_path, voice):
    manager, _ = fake_coqui(monkeypatch, Mock())
    voice = voice and str(tmp_path / voice)
    loaded, error = server.load_backend(xtts.XttsBackend, "cpu", voice, "coqui-tts", xtts.MODEL_ID)
    handler = server.make_handler(loaded, "pt", adapter=True, error=error)
    code, body = response(handler)
    assert code == 503
    assert ("not found" if voice else "reference voice") in json.loads(body)["error"]
    manager.download_model.assert_not_called()


@pytest.mark.parametrize("language,expected", [
    ("pt", "pt"), ("pt-BR", "pt"), ("PT_br", "pt"), ("en", "en"), ("en-US", "en"),
    ("zh", "zh-cn"), ("zh-CN", "zh-cn"), ("ja", "ja"),
])
def test_xtts_language_codes(language, expected):
    assert xtts.xtts_language(language) == expected


@pytest.mark.parametrize("language", ["auto", "sv", "", "xx-pt"])
def test_xtts_unsupported_language_is_a_clear_error(language):
    with pytest.raises(ValueError, match="does not speak language"):
        xtts.xtts_language(language)


@pytest.mark.parametrize("text,expected", [
    ("Olá, mundo.", "Olá, mundo"),
    ("Que bom!", "Que bom"),
    ("Veja isto:  ", "Veja isto"),
    ("Pois é;", "Pois é"),
    ("E então…", "E então"),
    ("Fim... ", "Fim"),
    ("Tudo bem?", "Tudo bem?"),
    ("Sério?!", "Sério?"),
    ("Dr. Silva chegou.", "Dr. Silva chegou"),
    ("Sem pontuação", "Sem pontuação"),
])
def test_xtts_drops_only_final_punctuation(text, expected):
    assert xtts.speakable_text(text) == expected


@pytest.mark.parametrize("text", ["", "   ", ".", "!...", " … "])
def test_xtts_never_sends_empty_text(xtts_model, text):
    backend, model, _, _, _ = xtts_model
    with pytest.raises(ValueError, match="nothing to speak"):
        backend.generate(text, "pt")
    model.inference.assert_not_called()


SR = xtts.SAMPLE_RATE


def tone(seconds, amplitude=0.2):
    t = np.arange(int(SR * seconds)) / SR
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds, level=0.0):
    return np.full(int(SR * seconds), level, dtype=np.float32)


def test_trim_cuts_noise_after_a_gap_near_the_end():
    speech = tone(1.5)
    # Measured artifacts: a 170-430 ms near-silent gap, then ~0.6 s of noise.
    audio = np.concatenate([speech, silence(0.17, 0.0005), tone(0.5, 0.3)])
    trimmed = xtts.trim_trailing_artifact(audio)
    # Cut at the start of the gap, rounded up to a 10 ms frame boundary.
    assert len(speech) <= len(trimmed) < len(speech) + SR // 100
    assert np.array_equal(trimmed[:len(speech)], speech)


def test_trim_keeps_the_latest_gap():
    audio = np.concatenate([tone(1.0), silence(0.2), tone(0.1), silence(0.2), tone(0.3)])
    assert len(xtts.trim_trailing_artifact(audio)) - int(SR * 1.3) in range(0, SR // 100)


@pytest.mark.parametrize("audio", [
    # Clean ending: trailing silence is not followed by more sound.
    np.concatenate([tone(1.5), silence(0.3)]),
    # Stop consonants in the last word ("repita", "certo") went silent for up
    # to 100 ms; an 80 ms rule cut those words off.
    np.concatenate([tone(1.0), silence(0.10), tone(0.6)]),
    np.concatenate([tone(1.5), silence(0.14), tone(0.4)]),
    # Quiet but audible pauses inside speech stay (measured at ~0.004 RMS).
    np.concatenate([tone(1.5), tone(0.15, 0.006), tone(0.4)]),
    # A gap before the last second is speech, not the artifact.
    np.concatenate([tone(1.0), silence(0.2), tone(1.2)]),
    np.zeros(0, dtype=np.float32),
    tone(0.005),
])
def test_trim_leaves_clean_endings_untouched(audio):
    assert np.array_equal(xtts.trim_trailing_artifact(audio), audio)


def test_trim_applies_to_generated_audio(xtts_model, monkeypatch):
    backend, model, _, _, _ = xtts_model
    logs = []
    monkeypatch.setattr(server, "log", logs.append)
    model.inference.return_value = {"wav": np.concatenate([tone(1.0), silence(0.2), tone(0.5)])}
    assert abs(backend.generate("Olá.", "pt").shape[1] - SR) < SR // 100
    assert logs == ["Trimmed 0.70s of trailing noise"]


@pytest.mark.parametrize("text,pause", [
    ("Pode ser, sim", True), ("Acho que sim, mas não", True), ("Tá bom; tchau", True),
    ("Eu vi: sim", True), ("Ele disse — sim", True), ("Tudo certo. Sim", True),
    ("Pode ser, sim?", True), ("Pronto, foi!", True),
    ("Isso parece certo", False), ("Pronto, terminei de organizar tudo agora", False),
    ("Você quer que eu repita?", False), ("Olá", False), ("Sim, eu vou lá agora", False),
])
def test_xtts_pause_mark_in_last_three_words(text, pause):
    assert xtts.pauses_before_last_word(text) is pause


def test_xtts_keeps_short_final_word_after_comma_pause(xtts_model, monkeypatch):
    # REV-1: XTTS pauses at the comma; that pause plus "sim" sounds exactly
    # like the artifact (measured), so the text decides.
    backend, model, _, _, _ = xtts_model
    logs = []
    monkeypatch.setattr(server, "log", logs.append)
    audio = np.concatenate([tone(1.2), silence(0.2), tone(0.35)])
    model.inference.return_value = {"wav": audio}
    assert backend.generate("Pode ser, sim.", "pt").shape == (1, len(audio))
    assert model.inference.call_args.args[0] == "Pode ser, sim"
    assert logs == []
    # Without the comma the same audio is the artifact and is cut at the gap.
    assert backend.generate("Pode ser sim.", "pt").shape[1] - int(SR * 1.2) in range(0, SR // 100)


@pytest.mark.parametrize("text", ["Pode ser sim.", "Pode ser, sim."])
@pytest.mark.parametrize("audio", [
    np.concatenate([tone(1.213), silence(0.2), tone(0.35)]),
    np.concatenate([tone(0.2), silence(0.2), tone(0.3)]),
    tone(0.005),
])
def test_xtts_stream_matches_whole_audio(xtts_model, monkeypatch, text, audio):
    backend, model, _, _, _ = xtts_model
    logs = []
    monkeypatch.setattr(server, "log", logs.append)
    model.inference.return_value = {"wav": audio}
    model.inference_stream.return_value = (FakeTensor(audio[i:i + 4321])
                                           for i in range(0, len(audio), 4321))
    whole = backend.generate(text, "pt-BR").numpy().reshape(-1)
    streamed = np.concatenate(list(backend.generate_stream(text, "pt-BR")))
    np.testing.assert_array_equal(streamed, whole)
    assert model.inference_stream.call_args == model.inference.call_args
    assert model.inference_stream.call_args.kwargs == {
        "temperature": xtts.TEMPERATURE, "enable_text_splitting": False}
    assert len(logs) == (2 if len(whole) < len(audio) else 0)


def test_xtts_states_license_at_startup_and_in_help(monkeypatch, tmp_path, fake_torch, capsys):
    logs = []
    monkeypatch.setattr(server, "log", logs.append)
    model = Mock()
    model.modules.return_value = []
    model.get_conditioning_latents.return_value = ("lat", "spk")
    fake_coqui(monkeypatch, model)
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"reference")
    xtts.XttsBackend("cpu", str(voice))
    assert xtts.LICENSE_NOTICE in logs
    assert "non-commercial" in xtts.LICENSE_NOTICE and "COQUI_TOS_AGREED" in xtts.LICENSE_NOTICE
    with pytest.raises(SystemExit):
        server.backend_main(xtts.XttsBackend, "coqui-tts", xtts.MODEL_ID, ["--help"],
                            notice=xtts.LICENSE_NOTICE)
    assert "Coqui Public Model License" in " ".join(capsys.readouterr().out.split())


def test_xtts_download_progress_reaches_health(monkeypatch):
    hub = pytest.importorskip("huggingface_hub.utils")
    from debora_whisper.download_progress import DownloadProgress, track_hf_downloads

    class Manager:
        def __init__(self, progress_bar):
            self.progress_bar = progress_bar

    manage = SimpleNamespace(tqdm=None, ModelManager=Manager)
    monkeypatch.setitem(sys.modules, "TTS", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "TTS.utils", SimpleNamespace(manage=manage))
    monkeypatch.setitem(sys.modules, "TTS.utils.manage", manage)
    progress = DownloadProgress()
    with track_hf_downloads(progress):
        assert xtts.report_coqui_downloads().progress_bar is True
        # Coqui's _download_model_files: tqdm(total=..., unit="iB", unit_scale=True).
        with manage.tqdm(total=1000, unit="iB", unit_scale=True) as bar:
            assert isinstance(bar, hub.tqdm)
            bar.update(250)
            assert progress.snapshot() == {"done": 250, "total": 1000}
        assert progress.snapshot() is None


def test_adapter_http_contract_and_origin_checks(qwen_model, monkeypatch):
    backend, model, _, _ = qwen_model
    monkeypatch.setattr(server, "check_connection", lambda _: None)
    logs = []
    monkeypatch.setattr(server, "log", logs.append)
    handler = server.make_handler(backend, "pt", adapter=True)
    code, body = response(handler)
    assert code == 200 and json.loads(body)["sample_rate"] == 24000
    assert response(handler, "/tts", {"text": "Olá"}, {"Origin": "null"})[0] == 403
    assert response(handler, "/tts", {"text": "Olá"}, {"Content-Type": "text/plain"})[0] == 403
    model.generate_voice_clone.assert_not_called()
    code, wav, content_type = response(handler, "/tts", {"text": "Olá, mundo."})
    assert code == 200 and content_type == "audio/wav"
    with wave.open(io.BytesIO(wav), "rb") as audio:
        assert audio.getnframes() == 2400 and audio.getframerate() == 24000
    assert any("s of audio in " in line for line in logs)


def test_adapter_disconnect_cancels_and_removes_hooks(qwen_model, monkeypatch):
    backend, _, _, _ = qwen_model
    client, connection = socket.socketpair()
    # Advance monotonically beyond the 20 ms poll throttle at every call.
    ticks = iter(range(100))
    monkeypatch.setattr(server.time, "monotonic", lambda: next(ticks, 100))
    try:
        with pytest.raises(server.SynthesisCancelled):
            with server.cancellable_synthesis(backend, connection, backend.cancel_modules):
                client.close()
                backend.cancel_modules[0]()
        assert all(not module.hooks for module in backend.cancel_modules)
    finally:
        client.close()
        connection.close()


@pytest.mark.parametrize("filename", ["debora_whisper/tts_server.py",
    "debora_whisper/tts_server_qwen3.py", "debora_whisper/tts_server_kokoro.py",
    "debora_whisper/tts_server_xtts.py", "docs/benchmarks/tts-bench/bench_tts.py"])
def test_pep723_pins_and_scoped_indexes(filename):
    tomllib = pytest.importorskip("tomllib")  # stdlib from Python 3.11
    from packaging.requirements import Requirement
    text = (ROOT / filename).read_text(encoding="utf-8")
    block = text.split("# /// script\n", 1)[1].split("# ///", 1)[0]
    metadata = tomllib.loads("\n".join(line[2:] if line.startswith("# ") else "" for line in block.splitlines()))
    for spec in metadata["dependencies"]:
        requirement = Requirement(spec)
        pins = list(requirement.specifier)
        assert len(pins) == 1 and pins[0].operator == "==" and "*" not in pins[0].version
    if "tts_server" in filename:
        uv = metadata["tool"]["uv"]
        assert uv["index"] == [{"name": "pytorch-cu124", "url": "https://download.pytorch.org/whl/cu124", "explicit": True}]
        assert uv["sources"] == {name: {"index": "pytorch-cu124"} for name in ("torch", "torchaudio")}
    else:
        assert metadata["dependencies"] == []  # stdlib-only benchmark needs no index


def test_default_transcript_is_exact_and_packaged():
    tomllib = pytest.importorskip("tomllib")  # stdlib from Python 3.11
    expected = "Sistemas iniciados com sucesso. Estou processando o áudio localmente, garantindo a sua privacidade. A captação está limpa. Pode começar a ditar quando quiser. Eu estarei aqui, ouvindo."
    assert (ROOT / "debora_whisper/voices/debora_v2.txt").read_text(encoding="utf-8").splitlines() == [expected]
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    targets = project["tool"]["hatch"]["build"]["targets"]
    # packages = ["debora_whisper"] already ships voices/; a force-include of
    # the same folder adds every file twice and breaks the wheel build.
    assert targets["wheel"]["packages"] == ["debora_whisper"]
    assert "debora_whisper/voices" not in targets["wheel"].get("force-include", {})
    assert "debora_whisper" in targets["sdist"]["include"]


@pytest.mark.parametrize("filename", ["tts_server_qwen3.py", "tts_server_kokoro.py",
                                      "tts_server_xtts.py"])
def test_script_imports_sibling_http_code_without_app_install(monkeypatch, filename):
    monkeypatch.syspath_prepend(str(ROOT / "debora_whisper"))
    namespace = runpy.run_path(str(ROOT / "debora_whisper" / filename), run_name="isolated_backend")
    assert Path(namespace["server"].__file__).resolve() == ROOT / "debora_whisper/tts_server.py"


@pytest.fixture
def bench():
    spec = importlib.util.spec_from_file_location("bench_tts", ROOT / "docs/benchmarks/tts-bench/bench_tts.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    return bench


def test_benchmark_response_metrics_and_repo_output_guard(monkeypatch, bench):
    wav = server.to_wav(np.zeros(24000), 24000)
    opener = Mock()
    opener.open.return_value = io.BytesIO(wav)
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(ticks, 10.25))
    result, metrics = bench.synthesize(opener, "http://127.0.0.1:8765", "Olá.", None, 60)
    assert result == wav
    assert metrics == {"response_seconds": 0.25, "audio_seconds": 1.0, "rtf": 0.25}
    assert bench.percentile([1, 2, 3, 4], 0.5) == 2.5
    assert bench.percentile([1, 2, 3, 4], 0.9) == pytest.approx(3.7)
    with pytest.raises(ValueError, match="outside a repository"):
        bench.output_dir(ROOT / "generated-audio")
    assert len((ROOT / "docs/benchmarks/tts-bench/frases_ptbr.txt").read_text(encoding="utf-8").splitlines()) == 20


@pytest.mark.parametrize("kind", ["audio", "health", "error"])
def test_benchmark_rejects_oversized_responses(bench, monkeypatch, tmp_path, capsys, kind):
    limit = 32
    for name in ("MAX_AUDIO_BYTES", "MAX_HEALTH_BYTES", "MAX_ERROR_BYTES"):
        monkeypatch.setattr(bench, name, limit)
    stream = io.BytesIO(b"x" * (limit + 100))
    read = Mock(wraps=stream.read1)
    stream.read1 = read
    opener = Mock()
    opener.open.return_value = stream
    if kind == "error":
        opener.open.side_effect = urllib.error.HTTPError("http://test/tts", 503, "Failed", {}, stream)
    if kind == "health":
        monkeypatch.setattr(bench, "output_dir", lambda _: tmp_path)
        monkeypatch.setattr(bench.urllib.request, "build_opener", lambda *args: opener)
        with pytest.raises(SystemExit) as exc:
            bench.main([])
        assert exc.value.code == 1
        assert "exceeds 32 bytes" in capsys.readouterr().err
    else:
        with pytest.raises(ValueError, match="exceeds 32 bytes") as exc:
            bench.synthesize(opener, "http://test", "Olá", None, 60)
        if kind == "error":
            assert "HTTP 503" in str(exc.value)
    read.assert_called_once_with(limit + 1)
    assert stream.closed


@pytest.mark.parametrize("http_error", [False, True])
def test_benchmark_deadline_stops_continuously_streaming_response(bench, monkeypatch, http_error):
    clock = [0.0]
    monkeypatch.setattr(bench.time, "monotonic", lambda: clock[0])
    sock = Mock()

    class Trickle(io.BytesIO):
        fp = SimpleNamespace(raw=SimpleNamespace(_sock=sock))

        def read1(self, size):
            assert size > 0
            clock[0] += 0.4
            return b"x"  # Never reaches EOF, never hits the socket idle timeout.

    stream = Trickle()
    opener = Mock()
    opener.open.return_value = stream
    if http_error:
        opener.open.side_effect = urllib.error.HTTPError("http://test", 503, "Failed", {}, stream)
    with pytest.raises(ValueError if http_error else TimeoutError, match="deadline exceeded"):
        bench.fetch(opener, "http://test", 1, 1000)
    assert clock[0] == pytest.approx(1.2)
    assert [call.args[0] for call in sock.settimeout.call_args_list] == pytest.approx([1, 0.6, 0.2])
    assert stream.closed


def test_benchmark_preserves_bounded_http_error_detail(bench):
    stream = io.BytesIO(b"Install espeak-ng")
    opener = Mock()
    opener.open.side_effect = urllib.error.HTTPError("http://test", 503, "Failed", {}, stream)
    with pytest.raises(ValueError, match="HTTP 503: Install espeak-ng"):
        bench.fetch(opener, "http://test", 60, bench.MAX_HEALTH_BYTES)
    assert stream.closed


def test_benchmark_accepts_response_at_byte_limit(bench):
    with io.BytesIO(b"exactly") as response:
        assert bench.read_bounded(response, 7, bench.time.monotonic() + 10) == b"exactly"


@pytest.mark.parametrize("chunked", [False, True])
def test_benchmark_deadline_interrupts_real_socket_reads(bench, chunked):
    client, peer = socket.socketpair()
    stop = threading.Event()

    def send():
        try:
            headers = b"Transfer-Encoding: chunked\r\n" if chunked else b""
            peer.sendall(b"HTTP/1.1 200 OK\r\n" + headers + b"\r\n")
            if chunked:
                peer.sendall(b"1;")  # An unfinished chunk extension stalls read1's framing parser.
            while not stop.is_set():
                peer.sendall(b"x")
                stop.wait(0.005)
        except OSError:
            pass

    worker = threading.Thread(target=send, daemon=True)
    worker.start()
    try:
        client.settimeout(2)
        with http.client.HTTPResponse(client) as response:
            response.begin()
            start = time.monotonic()
            with pytest.raises(TimeoutError, match="deadline exceeded"):
                bench.read_bounded(response, bench.MAX_AUDIO_BYTES, start + 0.1)
            assert time.monotonic() - start < 1
    finally:
        stop.set()
        client.close()
        peer.close()
        worker.join(2)
