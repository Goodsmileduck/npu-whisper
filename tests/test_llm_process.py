"""The voice chat's LLM runs in debora_whisper/llm_server.py's process: its
compile holds the GIL, which froze the app when it ran inside it."""
import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from debora_whisper import llm_server
from debora_whisper import voice_chat as vc
from debora_whisper.dictation_engine import DEFAULT_CONFIG

ROOT = Path(__file__).parent.parent

# openvino_genai stand-in for the real process. FAKE_FAIL_ON: devices that
# fail to load; FAKE_LOAD_SECONDS: load time; FAKE_TOKEN_SECONDS: per token.
FAKE_GENAI = '''
import json, os, time

class StreamingStatus:
    RUNNING, CANCEL = "RUNNING", "CANCEL"

class GenerationConfig:
    pass

class _Tokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, extra_context):
        return json.dumps([messages, extra_context])

class LLMPipeline:
    def __init__(self, path, device, **properties):
        time.sleep(float(os.environ.get("FAKE_LOAD_SECONDS", "0")))
        if device in os.environ.get("FAKE_FAIL_ON", "").split(","):
            raise RuntimeError(device + " is not available")

    def get_tokenizer(self):
        return _Tokenizer()

    def generate(self, prompt, config, streamer):
        print("native noise on stdout", flush=True)
        for token in ["Olá", "!", " Tudo", " bem."]:
            time.sleep(float(os.environ.get("FAKE_TOKEN_SECONDS", "0")))
            if streamer(token) == "CANCEL":
                return
'''


# --- The server, in this process ---------------------------------------------

class FakePipeline:
    fail_on = set()

    def __init__(self, path, device, **properties):
        if device in self.fail_on:
            raise RuntimeError(f"{device} is not available")
        self.path, self.device, self.properties = path, device, properties
        self.prompts = []

    def get_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = \
            lambda messages, add_generation_prompt, extra_context: json.dumps(
                [messages, extra_context])
        return tokenizer

    def generate(self, prompt, config, streamer):
        self.prompts.append((prompt, config))
        for token in ["Olá", "!", " Tudo", " bem."]:
            if streamer(token) == "CANCEL":
                return


@pytest.fixture
def genai(monkeypatch):
    from debora_whisper import device_queries

    # Identity queries must use the same fake backend as model loading.
    # Importing real OpenVINO starts its global telemetry executor.
    core = MagicMock()
    core.get_property.side_effect = lambda device, prop: (
        "0" if prop == "DEVICE_ID" else "Test device")
    monkeypatch.setattr(device_queries, "openvino_core", lambda: core)
    module = MagicMock()
    module.LLMPipeline = FakePipeline
    module.StreamingStatus.CANCEL, module.StreamingStatus.RUNNING = "CANCEL", "RUNNING"
    monkeypatch.setitem(sys.modules, "openvino_genai", module)
    FakePipeline.fail_on = set()
    return module


def _serve(*requests):
    done = threading.Event()

    class Output(io.StringIO):
        def write(self, text):
            result = super().write(text)
            if '"done": true' in text:
                done.set()
            return result

    def lines():
        for request in requests:
            done.clear()
            yield json.dumps(request) + "\n"
            assert done.wait(2), "Reply did not finish before closing input"

    out = Output()
    code = llm_server.serve("model", "GPU", "cache", lines(), out)
    return code, [json.loads(line) for line in out.getvalue().splitlines()]


def test_server_says_where_it_loaded_and_streams_the_reply(genai):
    code, out = _serve({"id": 1, "messages": [{"role": "user", "content": "oi"}],
                        "max_new_tokens": 400})
    assert code == 0
    assert out[0]["ready"] == "GPU"
    assert [m["text"] for m in out if "text" in m] == ["Olá", "!", " Tudo", " bem."]
    assert out[-1] == {"id": 1, "done": True}
    assert out[0]["allocation"]["target"] == "OV:GPU.0 Test device"


@pytest.mark.parametrize("broken_input", [False, True])
def test_server_joins_worker_when_input_stops(genai, monkeypatch, broken_input):
    workers = []
    thread = threading.Thread
    started = threading.Event()
    cancellations = []

    def generate(pipe, request, cancelled, say):
        started.set()
        cancellations.append((request["id"], cancelled.wait(1)))

    def track_worker(*args, **kwargs):
        worker = thread(*args, **kwargs)
        workers.append(worker)
        return worker

    def lines():
        yield json.dumps({"id": 1, "messages": [], "max_new_tokens": 9})
        assert started.wait(2), "Generation did not start"
        yield json.dumps({"id": 2, "messages": [], "max_new_tokens": 9})
        if broken_input:
            raise OSError("input closed")

    monkeypatch.setattr(llm_server.threading, "Thread", track_worker)
    monkeypatch.setattr(llm_server, "generate", generate)
    start = time.monotonic()
    if broken_input:
        with pytest.raises(OSError, match="input closed"):
            llm_server.serve("model", "GPU", "cache", lines(), io.StringIO())
    else:
        assert llm_server.serve("model", "GPU", "cache", lines(), io.StringIO()) == 0
    assert workers
    assert all(not worker.is_alive() for worker in workers)
    assert cancellations == [(1, True), (2, True)]
    assert time.monotonic() - start < 1


def test_server_templates_without_thinking(genai):
    pipes = []
    genai.LLMPipeline = lambda *a, **k: pipes.append(FakePipeline(*a, **k)) or pipes[-1]
    _serve({"id": 1, "messages": [], "max_new_tokens": 400})
    prompt, config = pipes[0].prompts[0]
    assert json.loads(prompt)[1] == {"enable_thinking": False}
    assert config.max_new_tokens == 400
    assert config.apply_chat_template is False  # already templated
    assert config.repetition_penalty == 1.1


def test_each_reply_draws_its_own_seed(genai):
    """The default seed, a fixed 0, made replies repeat themselves."""
    from types import SimpleNamespace
    genai.GenerationConfig = SimpleNamespace
    pipes = []
    genai.LLMPipeline = lambda *a, **k: pipes.append(FakePipeline(*a, **k)) or pipes[-1]
    _serve({"id": 1, "messages": [], "max_new_tokens": 9},
           {"id": 2, "messages": [], "max_new_tokens": 9})
    seeds = [config.rng_seed for _, config in pipes[0].prompts]
    assert len(seeds) == 2 and seeds[0] != seeds[1]
    assert pipes[0].properties == {"CACHE_DIR": "cache"}


def test_server_falls_back_to_the_cpu(genai):
    FakePipeline.fail_on = {"GPU"}
    _, out = _serve()
    assert out[0] == {"log": "cannot load the LLM on GPU: GPU is not available"}
    assert out[1]["ready"] == "CPU"


def test_server_that_loads_nowhere_says_why_and_exits(genai):
    FakePipeline.fail_on = {"GPU", "CPU"}
    code, out = _serve()
    assert code == 1
    assert out[-1] == {"failed": "GPU: GPU is not available; CPU: CPU is not available"}


def test_cancelled_reply_stops(genai):
    cancelled, out = threading.Event(), []
    cancelled.set()
    llm_server.generate(FakePipeline("m", "GPU"), {"id": 3, "messages": [],
                                                   "max_new_tokens": 9},
                        cancelled, out.append)
    assert out == [{"id": 3, "done": True}]


def test_already_cancelled_reply_never_enters_pipeline(genai):
    cancelled, out = threading.Event(), []
    cancelled.set()
    pipe = MagicMock()
    llm_server.generate(pipe, {"id": 3, "messages": [], "max_new_tokens": 9},
                        cancelled, out.append)
    pipe.get_tokenizer.assert_not_called()
    pipe.generate.assert_not_called()
    assert out == [{"id": 3, "done": True}]


# --- The client, with a real process -------------------------------------------

@pytest.fixture
def llm(monkeypatch, tmp_path):
    """A real llm_server process over the fake openvino_genai."""
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "openvino_genai.py").write_text(FAKE_GENAI, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(fake), str(ROOT)]))
    for name in ("FAKE_FAIL_ON", "FAKE_LOAD_SECONDS", "FAKE_TOKEN_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(vc, "_llm", None)
    monkeypatch.setattr(vc.paths, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "model").mkdir()
    logs = []
    yield {**DEFAULT_CONFIG, "llm_model": str(tmp_path / "model")}, logs
    vc.stop_llm()


def _reply(config, logs, stop=None):
    out = []
    vc.generate_reply([{"role": "user", "content": "oi"}], config, out.append,
                      stop or threading.Event(), log=logs.append)
    return out


def test_reply_streams_from_the_llm_process(llm):
    config, logs = llm
    assert "".join(_reply(config, logs)) == "Olá! Tudo bem."
    assert vc._llm.device == "GPU"
    assert vc._llm.process.pid != os.getpid()
    assert any("loaded on GPU" in m for m in logs)
    # Native output on stdout goes to the log file, never into a reply.
    assert "Voice chat: LLM process: native noise on stdout" not in logs


def test_one_process_serves_every_reply(llm):
    config, logs = llm
    _reply(config, logs)
    first = vc._llm
    assert "".join(_reply(config, logs)) == "Olá! Tudo bem."
    assert vc._llm is first
    assert vc.llm_loaded(config)


def test_llm_falls_back_to_the_cpu(llm, monkeypatch):
    config, logs = llm
    from debora_whisper.hardware_allocation import allocation
    allocation.configure({**config, "voice_chat": True}, logs.append)
    monkeypatch.setenv("FAKE_FAIL_ON", "GPU")
    _reply(config, logs)
    assert vc._llm.device == "CPU"
    summary = next(line for line in logs if line.startswith("Hardware allocation"))
    assert f"pid {vc._llm.process.pid}" in summary
    assert "planned GPU, fell back to OV:CPU" in summary
    assert "GPU is not available" in summary


def test_llm_that_loads_nowhere_fails_the_reply(llm, monkeypatch):
    config, logs = llm
    monkeypatch.setenv("FAKE_FAIL_ON", "GPU,CPU")
    with pytest.raises(RuntimeError, match="CPU is not available"):
        _reply(config, logs)
    assert not vc.llm_loaded(config)


def test_stop_cancels_the_reply(llm, monkeypatch):
    config, logs = llm
    monkeypatch.setenv("FAKE_TOKEN_SECONDS", "0.3")
    stop, out = threading.Event(), []
    vc.generate_reply([], config, lambda t: out.append(t) or stop.set(), stop,
                      log=logs.append)
    assert out == ["Olá"]
    # The process is still there for the next reply.
    monkeypatch.setenv("FAKE_TOKEN_SECONDS", "0")
    assert vc._llm.running


def test_stop_while_loading_returns_at_once(llm, monkeypatch):
    config, logs = llm
    monkeypatch.setenv("FAKE_LOAD_SECONDS", "30")
    stop = threading.Event()
    threading.Timer(0.5, stop.set).start()
    start = time.time()
    assert _reply(config, logs, stop) == []
    assert time.time() - start < 5


def test_hung_load_is_killed(llm, monkeypatch):
    config, logs = llm
    monkeypatch.setenv("FAKE_LOAD_SECONDS", "60")
    process = vc.start_llm(config, logs.append)
    with pytest.raises(RuntimeError, match="did not load in 1s; process killed"):
        process.wait_loaded(timeout=1)
    process.process.wait(10)
    assert not process.running


def test_crashed_process_is_started_again(llm):
    config, logs = llm
    _reply(config, logs)
    first = vc._llm
    from debora_whisper.processes import kill_tree
    kill_tree(first.process)
    first.process.wait(10)
    assert "".join(_reply(config, logs)) == "Olá! Tudo bem."
    assert vc._llm is not first


def test_process_exits_when_the_app_closes_its_stdin(llm):
    config, logs = llm
    _reply(config, logs)
    process = vc._llm.process
    vc.stop_llm()
    assert process.poll() is not None
