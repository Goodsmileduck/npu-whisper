"""config["voice_chat"]: the final text goes to a local LLM and its reply is
spoken by a local TTS server instead of being typed."""
import io
import http.client
import json
import re
import sys
import socket
import threading
import time
import urllib.error
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from debora_whisper import dictation_engine as de
from debora_whisper import voice_chat as vc
from debora_whisper.dictation_engine import DictationApp, DEFAULT_CONFIG

AUDIO = np.zeros(16000, dtype=np.float32)


def _wav(seconds=0.1, rate=24000):
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x10" * int(seconds * rate))
    return out.getvalue()


@pytest.fixture
def server():
    """TTS stub (/tts, /health). Set .tts_status or .health_status."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            srv.health_requests += 1
            if srv.health_error:
                return self._send(json.dumps({"ok": False, "error": srv.health_error}).encode(),
                                  "application/json", status=503)
            if self.path != "/health":
                return self.send_error(404)
            self._send(srv.health_body, "application/json", status=srv.health_status)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            srv.requests.append((self.path, body))
            if self.path != "/tts" or srv.tts_status != 200:
                return self.send_error(srv.tts_status if self.path == "/tts" else 404)
            self._send(_wav(), "audio/wav")

        def _send(self, payload, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            try:
                self.wfile.write(payload)
            except OSError:
                pass

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.requests, srv.tts_status, srv.health_status = [], 200, 200
    srv.health_error, srv.health_requests = None, 0
    srv.health_body = b'{"ok": true}'
    srv.tts_url = f"http://127.0.0.1:{srv.server_port}"
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join()


@pytest.fixture
def pcm_server():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            body = b'{"ok": true, "stream": true}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            srv.body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "audio/L16; rate=24000; channels=1")
            self.send_header("X-Sample-Rate", srv.rate)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for chunk in srv.chunks:
                    self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                    if srv.gate:
                        srv.sent.set()
                        if not srv.gate.wait(5):
                            return
                self.wfile.write(srv.ending)
            except OSError:
                pass

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.daemon_threads = False
    srv.tts_url = f"http://127.0.0.1:{srv.server_port}"
    raw = np.arange(7200, dtype="<i2").tobytes()
    srv.chunks = [raw[:1], raw[1:6000], raw[6000:]]
    srv.ending = b"0\r\n\r\n"
    srv.rate = "24000"
    srv.gate, srv.sent = None, threading.Event()
    thread = threading.Thread(target=srv.serve_forever)
    thread.start()
    try:
        yield srv
    finally:
        if srv.gate:
            srv.gate.set()
        srv.shutdown()
        srv.server_close()
        thread.join()


def test_synthesize_stream_pcm(pcm_server):
    chunks = list(vc.synthesize_stream("Olá", _cfg(pcm_server, tts_voice=None, language="pt")))
    assert len(chunks) >= 2
    assert all(rate == 24000 and samples.dtype == np.float32 for samples, rate in chunks)
    assert all(len(samples) >= 2400 for samples, _ in chunks[:-1])
    np.testing.assert_array_equal(np.concatenate([s for s, _ in chunks]),
                                  np.arange(7200, dtype=np.float32) / 32768)
    assert pcm_server.body == {"text": "Olá", "language": "pt", "voice": "", "stream": True}


def test_synthesize_stream_wav_fallback(server):
    chunks = list(vc.synthesize_stream("Olá", _cfg(server)))
    assert len(chunks) == 1 and chunks[0][1] == 24000
    assert len(chunks[0][0]) == 2400
    assert server.requests[0][1]["stream"] is True


def test_synthesize_stream_rejected(server):
    server.tts_status = 503
    with pytest.raises(vc.TTSRejected, match="HTTP 503"):
        list(vc.synthesize_stream("Olá", _cfg(server)))


@pytest.mark.parametrize("fallback", [False, True])
def test_synthesize_stream_bounds_total_bytes(pcm_server, server, monkeypatch, fallback):
    monkeypatch.setattr(vc, "TTS_MAX_RESPONSE_BYTES", 5000 if not fallback else 100)
    with pytest.raises(ValueError, match="larger than"):
        list(vc.synthesize_stream("Olá", _cfg(server if fallback else pcm_server)))


@pytest.mark.parametrize("ending", [b"", b"10\r\nabc"])
def test_synthesize_stream_rejects_truncated_chunked_body(pcm_server, ending):
    pcm_server.ending = ending
    with pytest.raises(http.client.IncompleteRead):
        list(vc.synthesize_stream("Olá", _cfg(pcm_server)))


@pytest.mark.parametrize("rate", ["1", "7999", "192001"])
def test_synthesize_stream_rejects_absurd_sample_rate(pcm_server, rate):
    pcm_server.rate = rate
    with pytest.raises(ValueError, match="sample rate"):
        list(vc.synthesize_stream("Olá", _cfg(pcm_server)))


def test_synthesize_stream_rejects_odd_pcm(pcm_server):
    pcm_server.chunks = [b"abc"]
    with pytest.raises(ValueError, match="odd byte"):
        list(vc.synthesize_stream("Olá", _cfg(pcm_server)))


def test_synthesize_stream_cancellation_closes_connection(pcm_server):
    pcm_server.chunks = [b"\x00\x10" * 2400] * 2
    pcm_server.gate = threading.Event()
    cancellation = vc.TTSCancellation()
    stream = vc.synthesize_stream("Olá", _cfg(pcm_server, _tts_cancellation=cancellation))
    assert len(next(stream)[0]) == 2400
    errors = []

    def read():
        try:
            next(stream)
        except Exception as e:
            errors.append(e)

    worker = threading.Thread(target=read)
    worker.start()
    try:
        cancellation.cancel()
        worker.join(3)
        assert not worker.is_alive() and errors
        assert not cancellation._sockets
    finally:
        pcm_server.gate.set()
        worker.join(3)
        stream.close()


@pytest.mark.parametrize("url", ["file:///tmp/voice", "ftp://localhost/tts"])
def test_synthesize_stream_rejects_non_http_url(url):
    with pytest.raises(ValueError, match="http\\(s\\)"):
        list(vc.synthesize_stream("Olá", {**DEFAULT_CONFIG, "tts_url": url}))


class FakeLLM:
    """Stands in for the OpenVINO model: streams .chunks, or raises .error."""

    def __init__(self, chunks=(), error=None):
        self.chunks, self.error, self.calls = list(chunks), error, []

    def __call__(self, messages, on_text, stop):
        self.calls.append(messages)
        if self.error:
            raise self.error
        for chunk in self.chunks:
            if stop.is_set():
                return
            on_text(chunk)


def _cfg(server, **extra):
    return {**DEFAULT_CONFIG, "voice_chat": True, "language": "pt",
            "tts_url": server.tts_url, **extra}


def _chat(server, llm, **extra):
    """A VoiceChat whose playback is recorded, not played."""
    played = []
    chat = vc.VoiceChat(_cfg(server, **extra), log=lambda m: None, llm=llm,
                        play=lambda samples, rate, stop: played.append((len(samples), rate)))
    return chat, played


def _tts_texts(server):
    return [body["text"] for path, body in server.requests if path == "/tts"]


@pytest.mark.parametrize("buffered, pending, expected", [
    (2, [(1, 2), (1, 2)], 0),  # synthesis faster than playback
    (2, [(3, 2), (3, 2)], 2),  # the last sentence determines the gap
    (2, [(4, 10), (1, 1)], 2),  # the first sentence determines the gap
    (2, [(0, 2), (3, 2)], 0),  # active synthesis's elapsed time was subtracted
    (2, [], 0),  # no speculation about sentences the writer has not written
])
def test_prebuffer_known_sentence_schedule(buffered, pending, expected):
    assert vc.prebuffer_wait(buffered, pending) == expected


def test_prebuffer_cap_counts_time_already_held():
    assert vc.prebuffer_wait(1, [(10, 2)], max_wait=3) == 3
    assert vc.prebuffer_wait(1, [(10, 2)], max_wait=3, held=2.5) == 0.5
    assert vc.prebuffer_wait(1, [(10, 2)], max_wait=3, held=4) == 0


def test_synthesis_model_learns_observed_ratio_and_spoken_speed():
    model = vc.SynthesisModel(k=0)
    spoken = "x" * 40
    for _ in range(150):
        model.update(spoken, synth=4, audio=2, seconds=2)
    synth, audio = model.predict(spoken)
    assert audio == pytest.approx(2)
    assert synth / audio == pytest.approx(2)
    assert model.chars_per_second == pytest.approx(20)
    assert vc.SynthesisModel().predict("x" * 28) == pytest.approx((4.19, 2))


def test_synthesis_model_separates_raw_duration_from_playback_coverage():
    model = vc.SynthesisModel()
    spoken = "x" * 20
    for _ in range(150):
        model.update(spoken, synth=2, audio=4, seconds=1)
    synth, playback = model.predict(spoken)
    assert synth == pytest.approx(2 + model.k * model.sd)
    assert playback == pytest.approx(1)
    assert model.raw_chars_per_second == pytest.approx(5)
    assert model.chars_per_second == pytest.approx(20)


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan")])
def test_synthesis_model_ignores_invalid_durations(bad):
    model = vc.SynthesisModel()
    model.update("texto", synth=bad, audio=1, seconds=1)
    model.update("texto", synth=1, audio=bad, seconds=1)
    model.update("texto", synth=1, audio=1, seconds=bad)
    assert model == vc.SynthesisModel()


def test_synthesis_model_bounds_outliers():
    model = vc.SynthesisModel()
    for synth, audio in [(1e100, 1e-100), (1e-100, 1e100)] * 100:
        model.update("texto", synth, audio, seconds=audio)
        assert 0 <= model.a <= 5
        assert 0.05 <= model.b <= 5
        assert 3 <= model.chars_per_second <= 40
        assert 3 <= model.raw_chars_per_second <= 40


@pytest.mark.parametrize("enabled", [True, False])
def test_slow_synthesis_prebuffers_without_underruns(monkeypatch, enabled):
    monkeypatch.setattr(vc, "ensure_tts_server", lambda *a: None)
    monkeypatch.setattr(vc, "wait_tts_server", lambda *a: True)
    rendered, starts, ends, logs = [], [], [], []
    sentences = ["Primeira frase longa.", "Segunda frase longa.", "Terceira frase longa."]

    def synth(spoken, config):
        time.sleep(0.12)
        rendered.append(spoken)
        return np.full(600, 0.1, np.float32), 10000

    def play(samples, rate, stop):
        starts.append((time.monotonic(), len(rendered)))
        stop.wait(len(samples) / rate)
        ends.append(time.monotonic())

    monkeypatch.setattr(vc, "synthesize", synth)
    chat = vc.VoiceChat({**DEFAULT_CONFIG, "voice_chat_prebuffer": enabled},
                        llm=FakeLLM([" ".join(sentences)]), play=play, log=logs.append)
    chat.respond("oi")
    assert len(starts) == len(ends) == 3
    holds = [line for line in logs if "prebuffer held" in line]
    gaps = [starts[i + 1][0] - ends[i] for i in range(2)]
    if enabled:
        assert starts[0][1] == 3
        assert max(gaps) < 0.05
        assert len(holds) == 1
        assert re.fullmatch(r"Voice chat: prebuffer held \d+\.\d{2}s "
                            r"\(predicted gap \d+\.\d{2}s, buffered 0\.06s, pending 2\)", holds[0])
    else:
        assert starts[0][1] == 1
        assert max(gaps) > 0.03
        assert holds == []


def test_prebuffer_trailing_silence_does_not_count_as_playback(monkeypatch):
    monkeypatch.setattr(vc, "ensure_tts_server", lambda *a: None)
    monkeypatch.setattr(vc, "wait_tts_server", lambda *a: True)
    starts, ends, durations, logs = [], [], [], []
    sentences = ["Primeira frase longa.", "Segunda frase longa.",
                 "Terceira frase longa.", "Uma quarta frase aqui."]
    # trim_silence keeps 50 ms of the trailing silence: 4 s raw -> 1 s heard.
    audio = np.concatenate((np.full(950, 0.1, np.float32), np.zeros(3050, np.float32)))

    def synth(spoken, config):
        time.sleep(2)
        return audio, 1000

    def play(samples, rate, stop):
        starts.append(time.monotonic())
        durations.append(len(samples) / rate)
        stop.wait(durations[-1] + vc.SENTENCE_GAP_SECONDS)
        ends.append(time.monotonic())

    monkeypatch.setattr(vc, "synthesize", synth)
    chat = vc.VoiceChat(DEFAULT_CONFIG, llm=FakeLLM([" ".join(sentences)]),
                        play=play, log=logs.append)
    assert chat.respond("oi") == " ".join(sentences)
    assert durations == [1.0] * 4
    assert max(starts[i + 1] - ends[i] for i in range(3)) < 0.2
    holds = [line for line in logs if "prebuffer held" in line]
    assert len(holds) == 1
    held = float(re.search(r"prebuffer held ([\d.]+)s", holds[0])[1])
    assert 2.5 < held <= DEFAULT_CONFIG["voice_chat_prebuffer_max"] + 0.2


@pytest.fixture
def held_reply(monkeypatch):
    """The second synthesis waits while the first clip is in the prebuffer."""
    monkeypatch.setattr(vc, "ensure_tts_server", lambda *a: None)
    monkeypatch.setattr(vc, "wait_tts_server", lambda *a: True)
    instances = []

    def start(**config):
        holding, release, played = threading.Event(), threading.Event(), threading.Event()
        stop = threading.Event()
        calls, logs, errors, workers = [], [], [], []
        original_wait = vc.prebuffer_wait

        def predict(*args, **kwargs):
            result = original_wait(*args, **kwargs)
            if result:
                holding.set()
            return result

        def synth(spoken, config):
            workers.append(threading.current_thread())
            calls.append(spoken)
            if len(calls) == 2:
                assert release.wait(5)
            return np.full(1000, 0.1, np.float32), 10000

        monkeypatch.setattr(vc, "prebuffer_wait", predict)
        monkeypatch.setattr(vc, "synthesize", synth)
        chat = vc.VoiceChat({**DEFAULT_CONFIG, **config}, log=logs.append,
                            llm=FakeLLM(["Primeira frase longa. Segunda frase longa."]),
                            play=lambda *a: played.set())

        def respond():
            try:
                chat.respond("oi", stop=stop)
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=respond, daemon=True)
        instances.append((chat, worker, release, errors, workers))
        worker.start()
        assert holding.wait(2)
        assert not played.is_set()
        return chat, worker, release, played, stop, logs

    yield start
    for chat, worker, release, errors, workers in instances:
        chat.interrupt()
        release.set()
        worker.join(2)
        for renderer in workers:
            renderer.join(2)
            assert not renderer.is_alive()
        assert not worker.is_alive()
        assert not errors


@pytest.mark.parametrize("external", [False, True])
def test_interrupt_during_prebuffer_returns_promptly(held_reply, external):
    chat, worker, release, played, stop, logs = held_reply()
    before = time.monotonic()
    stop.set() if external else chat.interrupt()
    worker.join(0.5)
    assert not worker.is_alive()
    assert time.monotonic() - before < 0.5
    assert not played.is_set()
    assert len([line for line in logs if "prebuffer held" in line]) == 1


def test_pause_during_prebuffer_resumes_without_another_hold(held_reply):
    chat, worker, release, played, stop, logs = held_reply()
    assert chat.pause()
    assert not played.wait(0.15)
    assert chat.resume()
    # Even though the next synthesis is still blocked, resume is immediate.
    assert played.wait(0.5)
    release.set()
    worker.join(2)
    assert not worker.is_alive()


def test_prebuffer_cap_releases_first_clip(held_reply):
    chat, worker, release, played, stop, logs = held_reply(voice_chat_prebuffer_max=0.15)
    assert played.wait(0.5)
    assert any("prebuffer held" in line for line in logs)


@pytest.mark.parametrize("finished", [False, True])
def test_prebuffer_does_not_wait_for_unknown_sentences(monkeypatch, finished):
    monkeypatch.setattr(vc, "ensure_tts_server", lambda *a: None)
    monkeypatch.setattr(vc, "wait_tts_server", lambda *a: True)
    monkeypatch.setattr(vc, "synthesize", lambda *a: (np.full(1000, 0.1, np.float32), 10000))
    played = threading.Event()
    logs = []

    def llm(messages, on_text, stop):
        on_text("Primeira frase longa. ")
        if not finished:
            assert played.wait(2)  # writer is still running when playback begins

    chat = vc.VoiceChat(DEFAULT_CONFIG, llm=llm, play=lambda *a: played.set(), log=logs.append)
    chat.respond("oi")
    assert played.is_set()
    assert not any("prebuffer held" in line or "LLM failed" in line for line in logs)


def test_prebuffer_restarts_after_an_underrun(monkeypatch):
    monkeypatch.setattr(vc, "ensure_tts_server", lambda *a: None)
    monkeypatch.setattr(vc, "wait_tts_server", lambda *a: True)
    first_played = threading.Event()
    rendered, played, logs = [], [], []

    def synth(spoken, config):
        if rendered:
            time.sleep(0.12)
        rendered.append(spoken)
        return np.full(600, 0.1, np.float32), 10000

    def play(*args):
        played.append(len(rendered))
        first_played.set()

    def llm(messages, on_text, stop):
        on_text("Primeira frase longa. ")
        assert first_played.wait(2)
        on_text("Segunda frase longa. Terceira frase longa.")

    monkeypatch.setattr(vc, "synthesize", synth)
    chat = vc.VoiceChat(DEFAULT_CONFIG, llm=llm, play=play, log=logs.append)
    chat.respond("oi")
    assert played == [1, 3, 3]
    assert len([line for line in logs if "prebuffer held" in line]) == 1


def test_prebuffer_clears_audio_indicator_between_clips(monkeypatch):
    monkeypatch.setattr(vc, "ensure_tts_server", lambda *a: None)
    monkeypatch.setattr(vc, "wait_tts_server", lambda *a: True)
    first_played, idle = threading.Event(), threading.Event()
    calls, talking, logs = [], [], []

    def synth(spoken, config):
        calls.append(spoken)
        if len(calls) == 2:
            assert first_played.wait(2)
        elif len(calls) == 3:
            assert idle.wait(2)
        return np.full(600, 0.1, np.float32), 10000

    def play(*args):
        first_played.set()

    def on_audio(active):
        talking.append(active)
        if not active:
            idle.set()

    def llm(messages, on_text, stop):
        on_text("Primeira frase longa. ")
        assert first_played.wait(2)
        on_text("Segunda frase longa. Terceira frase longa.")

    monkeypatch.setattr(vc, "synthesize", synth)
    chat = vc.VoiceChat(DEFAULT_CONFIG, llm=llm, play=play, log=logs.append)
    chat.on_audio = on_audio
    chat.respond("oi")
    assert talking == [True, False, True, False]
    assert not any("TTS failed" in line for line in logs)


@pytest.mark.parametrize("cached", [False, True])
def test_waiting_feedback_bypasses_prebuffer(monkeypatch, cached):
    monkeypatch.setattr(vc, "ensure_tts_server", lambda *a: None)
    monkeypatch.setattr(vc, "wait_tts_server", lambda *a: True)
    monkeypatch.setattr(vc.threading, "Timer", lambda delay, callback: MagicMock(start=callback))
    feedback_played, written = threading.Event(), threading.Event()
    logs = []
    audio = np.full(600, 0.1, np.float32), 10000

    def synth(spoken, config):
        if spoken != "Um instante.":
            assert feedback_played.wait(1)
        return audio

    def llm(messages, on_text, stop):
        on_text("Primeira frase longa. ")
        written.set()

    monkeypatch.setattr(vc, "synthesize", synth)
    chat = vc.VoiceChat({**DEFAULT_CONFIG, "voice_chat_backend": "claude"},
                        llm=llm, play=lambda *a: feedback_played.set(), log=logs.append)
    if cached:
        chat._waiting_audio(chat.config)
    original_waiting = chat._waiting_audio

    def waiting(*args):
        assert written.wait(1)  # ensure there is pending speech to predict
        return original_waiting(*args)

    monkeypatch.setattr(chat, "_waiting_audio", waiting)
    chat.respond("oi")
    assert feedback_played.is_set()
    assert not any("prebuffer held" in line or "TTS failed" in line for line in logs)


@pytest.mark.parametrize("key, value", [
    ("voice_chat_prebuffer", "true"), ("voice_chat_prebuffer", 1),
    ("voice_chat_prebuffer", None), ("voice_chat_prebuffer_max", True),
    ("voice_chat_prebuffer_max", 0), ("voice_chat_prebuffer_max", -1),
    ("voice_chat_prebuffer_max", "3"), ("voice_chat_prebuffer_max", None),
    ("voice_chat_prebuffer_max", float("inf")), ("voice_chat_prebuffer_max", float("nan")),
])
def test_invalid_prebuffer_config_is_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        de.validate_config({**DEFAULT_CONFIG, key: value})


def test_prebuffer_defaults_and_valid_config():
    assert DEFAULT_CONFIG["voice_chat_prebuffer"] is True
    assert DEFAULT_CONFIG["voice_chat_prebuffer_max"] == 3.0
    de.validate_config({**DEFAULT_CONFIG, "voice_chat_prebuffer": False,
                        "voice_chat_prebuffer_max": 0.1})


def _pt_now(t):
    return f" Agora é {vc._PT_WEEKDAYS[t.tm_wday]}, " + time.strftime("%d/%m/%Y, %H:%M.", t)


def _en_now(t):
    return time.strftime(" It is now %A, %Y-%m-%d %H:%M.", t)


def _respond_timed(chat, text):
    """respond() and the clock read around it: the prompt states the current
    minute, which may tick over while the reply is produced."""
    before = time.localtime()
    reply = chat.respond(text)
    return reply, (before, time.localtime())


def test_reply_is_streamed_and_spoken_sentence_by_sentence(server):
    llm = FakeLLM(["A capital ", "é Canberra. Fica", " no sul! Mais", " algo?"])
    chat, played = _chat(server, llm)
    reply, around = _respond_timed(chat, "qual é a capital da austrália")
    assert reply == "A capital é Canberra. Fica no sul! Mais algo?"
    # "Fica no sul!" is too short to send alone: it waits for the next one.
    assert _tts_texts(server) == ["A capital é Canberra.", "Fica no sul! Mais algo?"]
    assert played == [(2400, 24000)] * 2
    system, user = llm.calls[0]
    assert system["role"] == "system"
    assert system["content"] in {vc.VOICE_CHAT_PROMPTS["pt"] + _pt_now(t) for t in around}
    assert user == {"role": "user", "content": "qual é a capital da austrália"}
    assert [b["language"] for p, b in server.requests if p == "/tts"] == ["pt"] * 2


def test_a_short_reply_is_still_spoken(server):
    chat, played = _chat(server, FakeLLM(["Opa! ", "Sim."]))
    assert chat.respond("oi") == "Opa! Sim."
    assert _tts_texts(server) == ["Opa! Sim."] and len(played) == 1


def test_emoji_is_neither_shown_nor_spoken(server):
    chat, played = _chat(server, FakeLLM(["Tudo certo por aqui, e você?\n😊"]))
    assert chat.respond("oi") == "Tudo certo por aqui, e você?"
    assert chat._history[-1]["content"] == "Tudo certo por aqui, e você?"
    assert _tts_texts(server) == ["Tudo certo por aqui, e você?"]


def test_emoji_inside_a_sentence_is_not_spoken():
    assert vc.without_emoji("Oi 😊, tudo bem? 👍🏽 Família 👨‍👩‍👧 e ❤️ 1️⃣") == \
        "Oi , tudo bem? Família e 1"
    assert vc.without_emoji("ação ‍ 25° ©") == "ação ‍ 25° ©"


def test_other_languages_get_the_english_prompt_and_their_language(server):
    llm = FakeLLM(["Hola, ¿qué tal?"])
    chat, _ = _chat(server, llm, language="es")
    _, around = _respond_timed(chat, "hola")
    assert llm.calls[0][0]["content"] in {
        vc.DEFAULT_VOICE_CHAT_PROMPT + " The user speaks Spanish: reply in Spanish."
        + _en_now(t) for t in around}


def test_configured_prompt_wins_over_her_portuguese_one(server):
    llm = FakeLLM(["ok"])
    chat, _ = _chat(server, llm, llm_prompt="Seja breve.")
    chat.respond("oi")
    assert llm.calls[0][0]["content"].startswith(
        "Seja breve. The user speaks Portuguese: reply in Portuguese.")


def test_configured_prompt_is_used(server):
    llm = FakeLLM(["ok"])
    chat, _ = _chat(server, llm, llm_prompt="Seja breve.", language="auto")
    _, around = _respond_timed(chat, "oi")
    assert llm.calls[0][0]["content"] in {"Seja breve." + _en_now(t) for t in around}


def test_earlier_turns_are_sent_as_history(server):
    llm = FakeLLM(["Canberra."])
    chat, _ = _chat(server, llm)
    chat.respond("capital da austrália")
    llm.chunks = ["Uns 450 mil."]
    chat.respond("e quantos habitantes")
    assert llm.calls[1][1:] == [
        {"role": "user", "content": "capital da austrália"},
        {"role": "assistant", "content": "Canberra."},
        {"role": "user", "content": "e quantos habitantes"}]


def test_a_repeated_reply_is_not_kept_in_the_history(server):
    """A copy in the history made every next reply a copy too."""
    llm = FakeLLM(["Parece que quer brincar."])
    chat, _ = _chat(server, llm)
    chat.respond("ah, mas você")
    assert chat.respond("e...") == "Parece que quer brincar."  # still spoken
    llm.chunks = ["Não sei, não tenho acesso à previsão."]
    chat.respond("vai chover amanhã?")
    assert [m["content"] for m in llm.calls[2][1:]] == [
        "ah, mas você", "Parece que quer brincar.", "e...", "vai chover amanhã?"]


def test_history_resets_after_a_long_pause(server):
    llm = FakeLLM(["Canberra."])
    chat, _ = _chat(server, llm)
    chat.respond("capital da austrália")
    chat._last_turn -= vc.HISTORY_IDLE_RESET_SECONDS + 1
    chat.respond("oi")
    assert len(llm.calls[1]) == 2


def test_markdown_and_inline_reasoning_are_not_spoken(server):
    chat, _ = _chat(server, FakeLLM(["<think>hm</think>**Olá!**\n\n- `um` item"]))
    assert chat.respond("oi") == "Olá! - um item"
    assert _tts_texts(server) == ["Olá! - um item"]


def test_llm_failure_returns_nothing_and_speaks_nothing(server):
    chat, played = _chat(server, FakeLLM(error=RuntimeError("model crashed")))
    assert chat.respond("oi") == ""
    assert played == [] and _tts_texts(server) == []


def test_sentence_the_tts_rejects_is_skipped_and_the_next_still_spoken(server):
    server.tts_status = 500  # what Chatterbox answers for text it cannot speak
    logs = []
    chat, played = _chat(server, FakeLLM(["Primeira frase longa. Segunda frase longa."]))
    chat.log = logs.append
    assert chat.respond("oi") == "Primeira frase longa. Segunda frase longa."
    assert played == []
    assert len(_tts_texts(server)) == 2
    assert any("TTS skipped 'Primeira frase longa.' (HTTP 500" in m for m in logs)


def test_unreachable_tts_still_returns_the_reply(server, monkeypatch):
    monkeypatch.setattr(vc, "synthesize", MagicMock(side_effect=ConnectionResetError()))
    chat, played = _chat(server, FakeLLM(["Primeira frase longa. Segunda frase longa."]))
    assert chat.respond("oi") == "Primeira frase longa. Segunda frase longa."
    assert played == []
    assert vc.synthesize.call_count == 1  # no retry per sentence once the server is gone


@pytest.mark.parametrize("ending", ["running", "interrupt", "shutdown"])
def test_tts_reset_is_only_a_failure_while_running(server, monkeypatch, ending):
    logs, renderers, stopping_at_kill = [], [], []
    chat, played = _chat(server, FakeLLM(["Uma frase para falar."]))
    chat.log = logs.append
    monkeypatch.setattr(vc, "_tts_stopping", threading.Event())
    vc._tts_stopping.set()  # starting a new turn clears a previous shutdown
    monkeypatch.setattr(vc, "_tts_process", None)
    monkeypatch.setattr(vc, "kill_tree", lambda process: stopping_at_kill.append(
        vc._tts_stopping.is_set()))

    def synthesize(text, config):
        renderers.append(threading.current_thread())
        if ending == "interrupt":
            chat.interrupt()
        elif ending == "shutdown":
            monkeypatch.setattr(vc, "_tts_process", MagicMock())
            vc.stop_tts_server()
        raise ConnectionResetError(10054, "connection reset")

    monkeypatch.setattr(vc, "synthesize", synthesize)
    chat.respond("oi")
    assert len(renderers) == 1
    renderers[0].join(3)
    assert not renderers[0].is_alive()
    assert played == []
    failures = [line for line in logs if "TTS failed" in line]
    if ending == "running":
        assert len(failures) == 1
        assert "10054" in failures[0] and "showing the reply only" in failures[0]
    else:
        assert failures == []
    if ending == "shutdown":
        assert stopping_at_kill == [True]


def test_interrupt_stops_the_reply(server):
    played = []
    chat = vc.VoiceChat(_cfg(server), log=lambda m: None, llm=FakeLLM(["Um. ", "Dois. ", "Três."]),
                        play=lambda s, r, stop: played.append(1) or chat.interrupt())
    reply = chat.respond("conta até três")
    assert played == [1]
    assert reply.startswith("Um.")


def test_interrupted_turn_stays_stopped_when_the_next_starts(server):
    """The next turn must not revive a generation still winding down."""
    stops = []
    llm = FakeLLM(["Oi."])
    chat, _ = _chat(server, lambda m, on_text, stop: stops.append(stop) or llm(m, on_text, stop))
    chat.respond("um")
    chat.respond("dois")
    assert stops[0] is not stops[1] and stops[0].is_set()


@pytest.mark.parametrize("url", ["file:///C:/Windows/win.ini", "ftp://host/v1"])
def test_non_http_tts_url_is_never_opened(url):
    with patch("urllib.request.OpenerDirector.open") as opened:
        with pytest.raises(ValueError):
            vc.synthesize("oi", {**DEFAULT_CONFIG, "tts_url": url})
    opened.assert_not_called()


def test_environment_proxy_is_not_used(server, monkeypatch):
    # The proxy points nowhere: the request only succeeds if it bypasses it.
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    chat, played = _chat(server, FakeLLM(["Direto."]))
    assert chat.respond("oi") == "Direto."
    assert len(played) == 1


@pytest.mark.parametrize("key, value", [
    ("tts_url", "127.0.0.1:8765"), ("tts_url", None),
    ("tts_timeout_seconds", -1), ("tts_timeout_seconds", "30"), ("tts_timeout_seconds", True),
    ("llm_model", ""), ("llm_model", None), ("llm_device", 3), ("tts_voice", ""), ("tts_voice", 1),
    ("harness_new_session_on_start", "true"), ("harness_new_session_on_start", 1),
    ("harness_new_session_on_start", None), ("harness_session_name", " "),
    ("harness_session_name", None), ("harness_session_name", 42),
])
def test_invalid_settings_are_rejected(key, value):
    with pytest.raises(ValueError):
        de.validate_config({**DEFAULT_CONFIG, key: value})


@pytest.mark.parametrize("buffer, done, rest", [
    ("Oi. Tudo", ["Oi."], "Tudo"),
    ("Oi", [], "Oi"),
    ("Sim! Não? Talvez… ", ["Sim!", "Não?", "Talvez…"], ""),
    ("Linha um\nLinha", ["Linha um"], "Linha"),
    ("Custa 3.50 reais", [], "Custa 3.50 reais"),
])
def test_split_sentences(buffer, done, rest):
    assert vc.split_sentences(buffer) == (done, rest)


# --- The LLM model files ---------------------------------------------------

def test_hub_model_is_downloaded_once(monkeypatch, tmp_path):
    calls = []

    def snapshot_download(repo, local_files_only=False):
        calls.append(local_files_only)
        if local_files_only and len(calls) == 1:
            raise FileNotFoundError(repo)
        (tmp_path / "openvino_model.bin").write_bytes(b"weights")
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        MagicMock(snapshot_download=snapshot_download))
    assert vc._model_path("OpenVINO/Qwen3-8B-int4-cw-ov", log=lambda m: None) == str(tmp_path)
    assert vc._model_path("OpenVINO/Qwen3-8B-int4-cw-ov", log=lambda m: None) == str(tmp_path)
    assert calls == [True, False, True]  # the second load stays offline


def test_interrupted_hub_download_is_resumed(monkeypatch, tmp_path):
    calls = []

    def snapshot_download(repo, local_files_only=False):
        calls.append(local_files_only)
        if not local_files_only:
            (tmp_path / "openvino_model.bin").write_bytes(b"weights")
        return str(tmp_path)  # the offline lookup finds the snapshot without weights

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        MagicMock(snapshot_download=snapshot_download))
    logs = []
    assert vc._model_path("OpenVINO/Qwen3-8B-int4-cw-ov", log=logs.append) == str(tmp_path)
    assert calls == [True, False]
    assert "incomplete" in logs[0]


# --- In the dictation flow ---------------------------------------------------

def _app(**config):
    app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False, **config})
    app.recorder = MagicMock()
    app.whisper = MagicMock()
    app._model_ready.set()
    return app


def _wait_reply(app):
    worker = app._voice_worker
    if worker is not None:
        worker.join(5)
        assert not worker.is_alive()


def _say(app, text, is_final=True, wait=True):
    app.whisper.transcribe.return_value = text
    app._finish_recording(audio=AUDIO, is_final=is_final)
    if wait:
        _wait_reply(app)


@pytest.fixture
def typed():
    out = []
    with patch.object(de, "type_text", side_effect=lambda t, auto_enter=False: out.append(t)), \
            patch.object(de, "type_draft_text", side_effect=lambda t: out.append(("draft", t))), \
            patch.object(de, "delete_text", side_effect=lambda n: out.append(("delete", n))), \
            patch.object(de.platforms.current(), "type_draft_keys",
                         side_effect=lambda t: out.append(("draft_keys", t))), \
            patch.object(de.platforms.current(), "backspace_text",
                         side_effect=lambda n: out.append(("backspace", n))), \
            patch.object(de, "get_input_target", return_value=("window", "field")):
        yield out


def test_final_text_goes_to_the_llm_and_nothing_is_typed(typed):
    app = _app(voice_chat=True)
    states = []
    app.add_callback(lambda s, d: states.append((s, d)))
    with patch.object(app.voice_chat, "respond", return_value="Tudo ótimo!") as respond:
        _say(app, " ola tudo bem")
    assert respond.call_args.args[0] == "ola tudo bem"
    assert typed == []
    assert app._history[-1]["text"] == "ola tudo bem\n→ Tudo ótimo!"
    assert states[-1] == (de.AppState.READY, {"text": "Tudo ótimo!"})


def test_the_reply_shows_as_speaking_never_as_ready(typed):
    """While the reply is said the app is not idle: no green Ready until it
    is done, and in continuous listening not even then."""
    app = _app(voice_chat=True, continuous_listening=True)
    app.is_recording = True
    states = []
    app.add_callback(lambda s, d: states.append((s, d)))

    def respond(text, on_reply, stop):
        on_reply("Oi,", 0.5)
        on_reply("Oi, tudo bem!")
        return "Oi, tudo bem!"
    with patch.object(app.voice_chat, "respond", side_effect=respond):
        _say(app, "oi")
    kinds = [s for s, _ in states]
    assert de.AppState.READY not in kinds
    # The audio's length goes along for the balloon's pace, when there is one.
    assert states[-3:] == [(de.AppState.SPEAKING, {"text": "Oi,", "seconds": 0.5}),
                           (de.AppState.SPEAKING, {"text": "Oi, tudo bem!"}),
                           (de.AppState.RECORDING, {"draft_text": ""})]


def test_microphone_keeps_listening_while_the_reply_plays(typed):
    app = _app(voice_chat=True)
    with patch.object(app.voice_chat, "respond", return_value="Oi."):
        _say(app, "oi")
    app.recorder.set_muted.assert_not_called()


def test_voice_chat_off_types_the_text(typed):
    app = _app(voice_chat=False)
    with patch.object(app.voice_chat, "respond") as respond:
        _say(app, "ola tudo bem")
    respond.assert_not_called()
    assert typed == ["ola tudo bem "]


def test_drafts_are_shown_not_typed_or_sent(typed):
    app = _app(voice_chat=True, continuous_listening=True)
    app.is_recording = True
    states = []
    app.add_callback(lambda s, d: states.append((s, d)))
    with patch.object(app.voice_chat, "respond", return_value="Oi!") as respond:
        _say(app, "ola tudo", is_final=False)
        respond.assert_not_called()
        _say(app, "ola tudo bem")
    respond.assert_called_once()
    assert typed == []
    assert (de.AppState.RECORDING, {"draft_text": "ola tudo"}) in states
    # Still listening: back to the microphone after the reply.
    assert states[-1] == (de.AppState.RECORDING, {"draft_text": ""})


def test_short_thanks_is_preserved_in_voice_chat_but_silence_is_not_sent(typed):
    app = _app(voice_chat=True)
    with patch.object(app.voice_chat, "respond", return_value="De nada.") as respond:
        _say(app, "Obrigado.")
        _say(app, "")
    respond.assert_called_once()
    assert app.history[0]["text"].startswith("Obrigado.")
    assert typed == []


def test_failed_reply_warns(typed):
    app = _app(voice_chat=True, beep_on_start=True)
    app.chimes = MagicMock()
    with patch.object(app.voice_chat, "respond", return_value=""):
        _say(app, "oi")
    app.chimes.play.assert_any_call("warning")
    assert app._history[-1]["text"] == "oi"


def test_hotkey_while_speaking_interrupts_the_reply():
    app = _app(voice_chat=True)
    app.voice_chat.speaking = True
    watched = threading.Event()
    with patch.object(app.voice_chat, "interrupt") as interrupt, \
            patch.object(app, "_watch_key", side_effect=lambda *args: watched.set()), \
            patch.object(app, "_start_recording") as start:
        try:
            app.toggle_recording()
            assert watched.wait(2)
        finally:
            app.stop_monitoring()
    interrupt.assert_called_once()
    start.assert_not_called()


def test_stop_interrupts_the_reply(monkeypatch):
    # keyboard.unhook_all starts global listener threads even with no hooks.
    monkeypatch.setattr("keyboard.unhook_all", MagicMock())
    app = _app(voice_chat=True)
    with patch.object(app.voice_chat, "interrupt") as interrupt:
        app.stop()
    interrupt.assert_called_once()


@pytest.mark.parametrize("barge_in", [True, False])
@pytest.mark.parametrize("backend", ["local", "claude"])
def test_speech_during_reply_is_transcribed_and_sent_in_order(typed, barge_in, backend):
    app = _app(voice_chat=True, voice_chat_barge_in=barge_in, voice_chat_backend=backend)
    started, release = threading.Event(), threading.Event()
    turns, stops = [], []

    def respond(text, on_reply, stop):
        turns.append(text)
        stops.append(stop)
        if len(turns) == 1:
            started.set()  # still thinking; no audio has been played
            assert release.wait(3)
        return "Resposta."

    with patch.object(app.voice_chat, "respond", side_effect=respond), \
            patch.object(app.voice_chat, "interrupt") as interrupt:
        try:
            _say(app, "sobre aquele projeto...", wait=False)
            assert started.wait(2)
            _say(app, "quero mudar a interface", wait=False)
            assert len(app.history) == 2  # saved before either reply finishes
            assert app.whisper.transcribe.call_count == 2
            assert turns == ["sobre aquele projeto..."]
            assert stops[0].is_set() is barge_in
            assert interrupt.call_count == int(barge_in)
            assert app.history[1]["voice_chat_status"] == ("barge-in" if barge_in else "queued")
        finally:
            release.set()
            _wait_reply(app)
    assert turns == ["sobre aquele projeto...", "quero mudar a interface"]
    assert stops[0] is not stops[1]
    if barge_in:
        assert app.history[0]["voice_chat_status"] == "interrupted"
    app.recorder.set_muted.assert_not_called()
    assert typed == []


def test_echo_is_logged_and_kept_in_history_without_interrupting(typed, monkeypatch):
    app = _app(voice_chat=True)
    logs = []
    monkeypatch.setattr(de, "log", logs.append)
    app.voice_chat._remember_spoken("A próxima etapa é conferir o projeto.", 3)
    with patch.object(app.voice_chat, "respond") as respond, \
            patch.object(app.voice_chat, "interrupt") as interrupt:
        _say(app, "A proxima etapa e conferir o projeto!")
    respond.assert_not_called()
    interrupt.assert_not_called()
    assert app.history[0]["voice_chat_status"] == "echo"
    assert app.history[0]["text"] == "A proxima etapa e conferir o projeto!"
    assert any("Voice chat: ignored echo" in line for line in logs)
    assert any("Final transcription" in line for line in logs)


def test_echo_filter_can_be_disabled(typed):
    app = _app(voice_chat=True, voice_chat_echo_filter=False)
    app.voice_chat._remember_spoken("Um instante.", 1)
    with patch.object(app.voice_chat, "respond", return_value="Oi.") as respond:
        _say(app, "Um instante.")
    respond.assert_called_once()


def test_echo_uses_capture_time_instead_of_transcription_time():
    chat = vc.VoiceChat({})
    chat._spoken = [(100, 104, "O projeto tem quatro arquivos.")]
    assert chat.is_echo("O projeto tem quatro arquivos", 101, 103)
    assert not chat.is_echo("O projeto tem quatro arquivos", 90, 99)
    assert not chat.is_echo("O projeto tem quatro arquivos", 110, 113)
    assert not chat.is_echo("Quero mudar a interface", 101, 103)


def test_interrupted_thinking_keeps_the_local_user_message(server):
    started = threading.Event()
    messages = []

    def llm(turns, on_text, stop):
        messages.append(turns)
        if len(messages) == 1:
            started.set()
            assert stop.wait(3)
        else:
            on_text("Certo.")

    chat, played = _chat(server, llm)
    worker = threading.Thread(target=chat.respond, args=("sobre o projeto...",))
    worker.start()
    try:
        assert started.wait(2)
        chat.interrupt()
        worker.join(2)
        assert not worker.is_alive()
        assert played == []
        chat.respond("quero mudar a interface")
    finally:
        chat.interrupt()
        worker.join(3)
    assert [m["content"] for m in messages[1][1:]] == [
        "sobre o projeto...", "quero mudar a interface"]


def test_played_waiting_phrase_is_an_echo_reference(server):
    chat, _ = _chat(server, FakeLLM(["Um instante."]))
    chat.respond("oi")
    now = time.monotonic()
    assert chat.is_echo("Um instante", now - 1, now)


@pytest.mark.parametrize("module", ["debora_whisper.app", "debora_whisper.dictation_engine"])
@pytest.mark.parametrize("session_flags, fresh", [
    ([], False), (["--harness-new-session-on-start"], True),
    (["--no-harness-new-session-on-start"], False),
])
def test_voice_chat_flag_turns_the_mode_on(monkeypatch, module, session_flags, fresh):
    import importlib
    import sys
    mod = importlib.import_module(module)
    started = []
    monkeypatch.setattr(sys, "argv", ["debora", "--voice-chat", "--device", "CPU",
                                    "--harness-session-name", "Debora test", *session_flags])
    monkeypatch.setattr(mod, "load_config", lambda: {
        **DEFAULT_CONFIG, "harness_new_session_on_start": bool(session_flags)})
    app_class = "GUIApp" if module.endswith(".app") else "DictationApp"
    monkeypatch.setattr(mod, app_class, lambda config: started.append(config) or MagicMock())
    # CLI parsing must not leave a real inventory worker running into later tests.
    inventory = MagicMock()
    monkeypatch.setattr(mod, "log_hardware_inventory", inventory)
    if module.endswith(".app"):
        monkeypatch.setattr(mod, "_claim_single_instance", lambda: True)
    mod.main()
    inventory.assert_called_once_with(started[0])
    assert started[0]["voice_chat"] is True
    assert started[0]["harness_new_session_on_start"] is fresh
    assert started[0]["harness_session_name"] == "Debora test"


# --- TTS server started by the app ------------------------------------------

@pytest.fixture
def no_tts_process(monkeypatch):
    monkeypatch.setattr(vc, "_tts_process", None)
    monkeypatch.setattr(vc, "_tts_server_key", None)
    yield
    with vc._tts_health_cache_lock:
        workers = [worker for _, worker in vc._tts_health_polls.values()]
    vc.stop_tts_server()
    for worker in workers:
        worker.join(3)


def _closed_port_url():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


def test_tts_stop_cancels_health_under_lock_and_kills_before_join(monkeypatch):
    process = MagicMock()
    cancelled = threading.Event()
    worker = MagicMock()
    order = []
    monkeypatch.setattr(vc, "_tts_process", process)
    monkeypatch.setattr(vc, "_tts_server_key", "old")
    monkeypatch.setattr(vc, "_tts_health_polls", {"url": (cancelled, worker)})
    monkeypatch.setattr(vc, "_tts_health_cache", {"url": {}})
    monkeypatch.setattr(vc, "_tts_capabilities_cache", {"url": {}})
    cancel_polls = vc._cancel_tts_health_polls

    def cancel():
        assert vc._tts_lock._is_owned()
        return cancel_polls()

    def kill(actual):
        assert actual is process
        assert cancelled.is_set()
        assert not vc._tts_health_polls
        assert not vc._tts_health_cache
        assert not vc._tts_capabilities_cache
        order.append("kill")

    def join(timeout):
        assert order == ["kill"]
        assert 0 <= timeout <= vc.TTS_HEALTH_SECONDS + 1.0
        order.append("join")

    # Teardown also cancels health polls, independently of server shutdown.
    with monkeypatch.context() as shutdown_patch:
        shutdown_patch.setattr(vc, "_cancel_tts_health_polls", cancel)
        shutdown_patch.setattr(vc, "kill_tree", kill)
        worker.join.side_effect = join
        vc.stop_tts_server()
    assert order == ["kill", "join"]


def test_tts_server_command_is_started_when_nothing_answers(no_tts_process, tmp_path):
    config = {**DEFAULT_CONFIG, "tts_url": _closed_port_url(),
              "tts_server_command": [sys.executable, "-c", "print('carregando'); import time; time.sleep(30)"]}
    log_path = tmp_path / "tts_server.log"
    vc.ensure_tts_server(config, log=lambda m: None, log_path=log_path)
    process = vc._tts_process
    assert process is not None and process.poll() is None
    vc.ensure_tts_server(config, log=lambda m: None, log_path=log_path)
    assert vc._tts_process is process  # one server per app process
    deadline = time.time() + 10
    while "carregando" not in log_path.read_text() and time.time() < deadline:
        time.sleep(0.1)
    vc.stop_tts_server()
    assert process.poll() is not None
    assert "carregando" in log_path.read_text()


def test_running_tts_server_is_reused(no_tts_process, server):
    with patch("subprocess.Popen") as popen:
        vc.ensure_tts_server({**DEFAULT_CONFIG, "tts_url": server.tts_url,
                              "tts_server_command": ["never-run"]})
    popen.assert_not_called()


def test_tts_server_up_needs_a_healthy_answer(server):
    config = {**DEFAULT_CONFIG, "tts_url": server.tts_url}
    assert vc.tts_server_up(config)
    server.health_status = 500
    assert not vc.tts_server_up(config)


def test_ready_capabilities_survive_loading_poll_cleanup(no_tts_process, server):
    config = _cfg(server)
    vc._cache_tts_health(config, {"ok": False, "state": "loading", "download": {"done": 1}})
    server.health_body = b'{"ok": true, "stream": true}'
    vc._start_tts_health_poll(config)
    with vc._tts_health_cache_lock:
        worker = vc._tts_health_polls[server.tts_url][1]
    worker.join(3)
    assert not worker.is_alive()
    assert vc.tts_download_progress(config) is None
    assert vc._tts_loading_health(config) == {}
    assert vc._tts_capabilities(config)["stream"] is True
    assert vc._tts_capabilities({"tts_url": server.tts_url + "/"})["stream"] is True
    assert vc._tts_capabilities({"tts_url": server.tts_url + "/other"}) == {}


@pytest.mark.parametrize("state", ["loading", "unavailable", "error", "wav", "stop"])
def test_ready_capabilities_invalidated(no_tts_process, server, state):
    config = _cfg(server)
    server.health_body = b'{"ok": true, "stream": true}'
    assert vc.tts_server_up(config)
    assert vc._tts_capabilities(config)["stream"] is True
    if state == "stop":
        vc.stop_tts_server()
    elif state == "error":
        server.health_error = "failed startup"
        with pytest.raises(vc.TTSStartupError):
            vc.tts_server_up(config)
    else:
        server.health_body = b'{"ok": false, "state": "loading"}' if state == "loading" else b'{"ok": true}'
        server.health_status = {"loading": 503, "unavailable": 500, "wav": 200}[state]
        assert vc._check_tts_health(config) == {"wav": "ready"}.get(state, state)
    assert not vc._tts_capabilities(config).get("stream")


@pytest.mark.parametrize("body", [b"", b"invalid", b"[]", b'{"stream": "true"}',
                                  b"x" * (vc.TTS_HEALTH_MAX_BYTES + 1)],
                         ids=["empty", "invalid", "array", "string-flag", "oversized"])
def test_invalid_ready_capabilities_fall_back_to_wav(server, body):
    config = _cfg(server)
    server.health_body = b'{"ok": true, "stream": true}'
    assert vc.tts_server_up(config)
    server.health_body = body
    assert vc.tts_server_up(config)
    assert not vc._tts_capabilities(config).get("stream")


@pytest.mark.parametrize("status", [200, 503])
def test_trickling_health_answer_cannot_stall_the_check(status):
    # Each byte arrives before the socket timeout, so only an absolute
    # deadline ends the check (it runs under _tts_lock at startup).
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    done = threading.Event()

    def trickle():
        conn, _ = listener.accept()
        with conn:
            conn.recv(4096)
            head = (f"HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n"
                    if status == 503 else "HTTP/1.1 200 OK\r\nX-Slow: ")
            try:
                if status == 503:
                    conn.sendall(f"{head}Content-Length: 100000\r\n\r\n".encode())
                for _ in range(100):
                    if done.is_set():
                        return
                    conn.sendall(b"a")
                    time.sleep(0.2)
            except OSError:
                pass

    thread = threading.Thread(target=trickle, daemon=True)
    thread.start()
    config = {**DEFAULT_CONFIG, "tts_url": f"http://127.0.0.1:{listener.getsockname()[1]}"}
    started = time.monotonic()
    try:
        assert not vc.tts_server_up(config)
    finally:
        done.set()
        listener.close()
    assert time.monotonic() - started < vc.TTS_HEALTH_SECONDS + 1.5


@pytest.mark.parametrize("error", [
    "Cannot read reference transcript voice.txt. Create this UTF-8 file with the exact words.",
    "Install espeak-ng and set PHONEMIZER_ESPEAK_LIBRARY.",
])
def test_live_failed_tts_does_not_wait_again_on_repeated_turns(server, monkeypatch, error):
    server.health_error = error
    process = MagicMock()
    process.poll.return_value = None  # Startup failed, but the HTTP server is still alive.
    monkeypatch.setattr(vc, "_tts_process", process)
    logs, shown = [], []
    chat, played = _chat(server, FakeLLM(["A resposta continua disponível."]),
                         tts_server_command=["never-run"])
    # A managed process retains the settings it was started with.
    monkeypatch.setattr(vc, "_tts_server_key",
                        (chat.config["tts_backend"], tuple(chat.config["tts_server_command"]),
                         urllib.parse.urlparse(chat.config["tts_url"]).port, None))
    chat.log = logs.append
    with patch.object(vc, "kill_tree") as kill_tree, \
            patch.object(vc.time, "sleep", side_effect=AssertionError("must not poll startup")):
        for _ in range(2):
            chat.respond("Olá", on_reply=shown.append)
    assert server.health_requests == 2
    assert shown == ["A resposta continua disponível."] * 2
    assert played == [] and server.requests == []
    assert sum("TTS startup failed" in line and error in line for line in logs) == 2
    assert not any("not reachable" in line for line in logs)
    kill_tree.assert_not_called()
    process.kill.assert_not_called()


def test_external_failed_tts_is_not_started_again(server, monkeypatch):
    server.health_error = "Install espeak-ng"
    monkeypatch.setattr(vc, "_tts_process", None)
    logs = []
    with patch("subprocess.Popen") as popen:
        vc.ensure_tts_server(_cfg(server), log=logs.append)
    popen.assert_not_called()
    assert any(server.health_error in line for line in logs)


def test_wait_gives_up_when_the_started_server_dies(no_tts_process):
    config = {**DEFAULT_CONFIG, "tts_url": _closed_port_url(),
              "tts_server_command": [sys.executable, "-c", "raise SystemExit(1)"]}
    vc.ensure_tts_server(config, log=lambda m: None)
    start = time.time()
    assert not vc.wait_tts_server(config, timeout=30)
    assert time.time() - start < 10


def test_crashed_tts_server_is_not_restarted_every_reply(no_tts_process):
    config = {**DEFAULT_CONFIG, "tts_url": _closed_port_url(),
              "tts_server_command": [sys.executable, "-c", "raise SystemExit(1)"]}
    logged = []
    vc.ensure_tts_server(config, log=logged.append)
    vc._tts_process.wait(10)
    with patch("subprocess.Popen") as popen:
        vc.ensure_tts_server(config, log=logged.append)
        vc.ensure_tts_server(config, log=logged.append)
    popen.assert_not_called()
    assert sum("exited (code 1)" in m for m in logged) == 1


@pytest.mark.parametrize("command", ["python server.py", [], ["python", ""], [1]])
def test_invalid_tts_server_command_is_rejected(command):
    with pytest.raises(ValueError):
        de.validate_config({**DEFAULT_CONFIG, "tts_server_command": command})


# --- Default TTS command (uv) -------------------------------------------------

@pytest.fixture
def voices(monkeypatch, tmp_path):
    monkeypatch.setattr(vc.paths, "VOICES_DIR", tmp_path)
    monkeypatch.setattr(vc.paths, "BUNDLED_VOICES_DIR", tmp_path / "bundled")
    monkeypatch.setattr(vc.shutil, "which", lambda name: "C:/uv/uv.exe")
    (tmp_path / "isabel.wav").write_bytes(_wav())
    return tmp_path


def test_default_command_runs_the_bundled_server_with_uv(voices):
    command = vc.tts_command({**DEFAULT_CONFIG, "tts_voice": "isabel", "language": "pt"})
    assert command == ["C:/uv/uv.exe", "run", "--script", str(vc.TTS_SERVER_SCRIPT),
                       "--port", "8765", "--voice", str(voices / "isabel.wav"),
                       "--language", "pt"]
    assert vc.TTS_SERVER_SCRIPT.is_file()


def test_voice_can_be_a_path(voices):
    path = voices / "outra.wav"
    path.write_bytes(_wav())
    command = vc.tts_command({**DEFAULT_CONFIG, "tts_voice": str(path), "language": "auto"})
    assert command[-2:] == ["--voice", str(path)]


def test_missing_voice_falls_back_to_the_default_voice(voices):
    logged = []
    command = vc.tts_command({**DEFAULT_CONFIG, "tts_voice": "ninguem"}, log=logged.append)
    assert "--voice" not in command
    assert "not found" in logged[0]


def test_voices_are_listed_by_name(voices):
    (voices / "Carol.wav").write_bytes(_wav())
    (voices / "notas.txt").write_text("x")
    (voices / "bundled").mkdir()
    (voices / "bundled" / "isabel.wav").write_bytes(_wav())
    (voices / "bundled" / "mari.wav").write_bytes(_wav())
    assert vc.list_voices() == ["Carol", "isabel", "mari"]


def test_bundled_voice_is_used_unless_the_voices_folder_has_one(voices):
    (voices / "bundled").mkdir()
    (voices / "bundled" / "mari.wav").write_bytes(_wav())
    assert vc.resolve_voice("mari") == voices / "bundled" / "mari.wav"
    (voices / "mari.wav").write_bytes(_wav())
    assert vc.resolve_voice("mari") == voices / "mari.wav"


def test_the_default_voice_ships_with_the_app():
    from debora_whisper import paths
    assert DEFAULT_CONFIG["tts_voice"] == "debora_v2"
    for name in ("carol", "debora", "debora_v2", "isabel", "mari"):
        path = paths.BUNDLED_VOICES_DIR / f"{name}.wav"
        assert "debora_whisper" in path.parts
        with wave.open(str(path)) as w:
            assert w.getnchannels() == 1 and 8 < w.getnframes() / w.getframerate() < 20


@pytest.mark.parametrize("voice, sent", [
    ("isabel", "isabel.wav"),  # a name in the voices folder
    (None, ""),                # Chatterbox's own voice
    ("ninguem", None),         # missing: the server keeps its voice
])
def test_each_request_names_the_voice(voices, server, voice, sent):
    """A voice picked in Settings speaks the next sentence, without
    restarting the TTS server."""
    vc.synthesize("Olá, tudo bem com você?", {**DEFAULT_CONFIG, "tts_url": server.tts_url,
                                               "tts_voice": voice})
    body = server.requests[-1][1]
    if sent is None:
        assert "voice" not in body
    elif sent:
        assert body["voice"] == str((voices / sent).resolve())
    else:
        assert body["voice"] == ""


# --- TTS server: voice per request --------------------------------------------

def test_tts_server_accepts_only_audio_files_as_voice(voices):
    from debora_whisper import tts_server
    assert tts_server.check_voice(None) is None
    assert tts_server.check_voice("") == ""
    assert tts_server.check_voice(str(voices / "isabel.wav")) == str(voices / "isabel.wav")
    (voices / "segredo.txt").write_text("x")
    for bad in (str(voices / "segredo.txt"), str(voices / "falta.wav"), 3,
                r"\\attacker\share\x.wav", "//attacker/share/x.wav"):
        with pytest.raises(ValueError):
            tts_server.check_voice(bad)


@pytest.mark.parametrize("headers, status", [
    ({"Content-Type": "application/json"}, 400),  # gets past the gate: empty text
    ({"Content-Type": "text/plain"}, 403),
    ({"Content-Type": "application/json", "Origin": "https://evil.example"}, 403),
])
def test_tts_server_refuses_requests_from_web_pages(headers, status):
    from debora_whisper import tts_server
    model = type("Model", (), {"sr": 24000, "conds": None})()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), tts_server.make_handler(model, "pt"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{srv.server_port}/tts", data=b'{"text": ""}', headers=headers)
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=5)
        assert error.value.code == status
    finally:
        srv.shutdown()
        srv.server_close()


def test_numbers_too_big_to_spell_are_left_as_digits():
    big = "1" * 40
    assert vc.spoken_numbers(f"São {big} estrelas.", "pt") == f"São {big} estrelas."


def test_tts_server_prepares_each_voice_once(voices):
    from debora_whisper import tts_server
    model = MagicMock(conds="startup", builtin_conds="builtin")
    model.prepare_conditionals.side_effect = lambda path: setattr(model, "conds", f"conds:{path}")
    chosen = tts_server.Voices(model)
    isabel = str(voices / "isabel.wav")
    for voice, conds in [(isabel, f"conds:{isabel}"), ("", "builtin"),
                         (None, "startup"), (isabel, f"conds:{isabel}")]:
        chosen.use(voice)
        assert model.conds == conds
    model.prepare_conditionals.assert_called_once_with(isabel)


def test_configured_command_wins(voices):
    assert vc.tts_command({**DEFAULT_CONFIG, "tts_server_command": ["x"]}) == ["x"]


@pytest.mark.parametrize("url", ["http://192.168.0.9:8765", "not a url"])
def test_remote_tts_url_is_never_started(voices, url):
    assert vc.tts_command({**DEFAULT_CONFIG, "tts_url": url}) is None


def test_without_uv_nothing_is_started(no_tts_process, monkeypatch):
    monkeypatch.setattr(vc.shutil, "which", lambda name: None)
    with patch("subprocess.Popen") as popen:
        vc.ensure_tts_server({**DEFAULT_CONFIG, "tts_url": _closed_port_url()},
                             log=lambda m: None)
    # Only the TTS server counts: telemetry threads left by other tests may
    # run nvidia-smi meanwhile.
    started = [c for c in popen.call_args_list if str(vc.TTS_SERVER_SCRIPT) in map(str, c.args[0])]
    assert started == []
    assert vc._tts_process is None


def test_bundled_server_declares_its_own_environment():
    header = vc.TTS_SERVER_SCRIPT.read_text(encoding="utf-8").split("# ///")[1]
    assert "chatterbox-tts" in header and "download.pytorch.org/whl/cu124" in header


class _Registry:
    """winreg stand-in holding the user's saved environment variables."""
    HKEY_CURRENT_USER = "HKCU"

    def __init__(self, values):
        self.values = values

    def OpenKey(self, root, path):
        from contextlib import nullcontext
        return nullcontext(path)

    def QueryValueEx(self, key, name):
        if name not in self.values:
            raise FileNotFoundError(name)
        return self.values[name], 1


def test_saved_variables_are_used_when_the_terminal_predates_them(monkeypatch, tmp_path):
    """A terminal opened before MODELS_DIR/HF_HOME were set does not pass
    them on; the app must still use the folders they point to."""
    import importlib
    from debora_whisper import paths
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "winreg", _Registry(
        {"MODELS_DIR": str(tmp_path), "HF_HOME": str(tmp_path / "huggingface")}))
    monkeypatch.delenv("MODELS_DIR", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    try:
        importlib.reload(paths)
        assert paths.MODEL_DIR == tmp_path / "debora-whisper" / "models"
        assert paths.VOICES_DIR == tmp_path / "voices"
        import os
        assert os.environ["HF_HOME"] == str(tmp_path / "huggingface")
    finally:
        monkeypatch.undo()
        importlib.reload(paths)


def test_paths_follow_models_dir(monkeypatch, tmp_path):
    import importlib
    from debora_whisper import paths
    monkeypatch.setitem(sys.modules, "winreg", _Registry({}))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    try:
        importlib.reload(paths)
        assert paths.MODEL_DIR == tmp_path / "debora-whisper" / "models"
        assert paths.CACHE_DIR == tmp_path / "debora-whisper" / "ov-cache"
        assert paths.VOICES_DIR == tmp_path / "voices"
        assert paths.CONFIG_FILE == Path.home() / ".debora" / "config.json"
        monkeypatch.delenv("MODELS_DIR")
        importlib.reload(paths)
        assert paths.MODEL_DIR == Path.home() / ".debora" / "models"
    finally:
        monkeypatch.undo()
        importlib.reload(paths)


def test_engine_waits_for_its_speech_model():
    app = DictationApp({**DEFAULT_CONFIG})
    app._model_ready.set()
    assert app._wait_speech_model()
    app._model_ready.clear()
    app._stopping.set()
    assert not app._wait_speech_model()


def test_each_spoken_sentence_is_logged(server):
    logs = []
    chat, _ = _chat(server, FakeLLM(["A capital é Canberra. Fica no sul do país."]))
    chat.log = logs.append
    chat.respond("capital da austrália")
    assert [m for m in logs if m.startswith("Voice chat: reply ")] == [
        "Voice chat: reply 'A capital é Canberra.'",
        "Voice chat: reply 'Fica no sul do país.'"]


def test_tts_timings_follow_each_chunk_through_playback(server):
    logs = []
    chat, played = _chat(server, FakeLLM(["A capital é Canberra. Fica no sul do país."]))
    chat.log = logs.append
    chat.respond("capital")
    chunks = [line for line in logs if " chunk=" in line]
    assert len(chunks) == len(played) == 2
    for line in chunks:
        times = {key: float(value) for key, value in re.findall(r"(\w+)=(\d+\.\d+)s", line)}
        assert times["enqueued"] <= times["synth_start"] <= times["synth_end"] <= times["play"]
        assert times["audio"] == pytest.approx(0.1)
    assert len([line for line in logs if " first_audio=" in line]) == 1
    assert not any("first_audio=-" in line for line in logs)


@pytest.mark.parametrize("boundary", [",", ";", ":", " —"])
def test_only_the_first_reply_chunk_splits_at_a_clause(server, boundary):
    head = "Eu posso explicar isso" + boundary
    tail = " porque temos bastante tempo para conversar. "
    later = "Agora temos outra frase, que deve continuar inteira."
    chat, _ = _chat(server, FakeLLM([head + " ", tail, later]))
    vc._cache_tts_health(chat.config, {"ok": True, "stream": False})
    chat.respond("explique")
    assert _tts_texts(server) == [vc.speakable(head), tail.strip(), later]


@pytest.mark.parametrize("enabled,advertised", [
    (True, True), (False, True), (True, False), (True, None), (True, "true")])
@pytest.mark.parametrize("fragmented", [False, True])
def test_first_reply_keeps_commas_only_when_tts_streams(monkeypatch, enabled, advertised, fragmented):
    monkeypatch.setattr(vc, "_tts_capabilities_cache", {})
    monkeypatch.setattr(vc, "ensure_tts_server", lambda *a: None)
    monkeypatch.setattr(vc, "wait_tts_server", lambda *a: True)
    config = {**DEFAULT_CONFIG, "language": "pt", "tts_stream": enabled}
    health = {"ok": True}
    if advertised is not None:
        health["stream"] = advertised
    vc._tts_capabilities_cache[config["tts_url"].rstrip("/")] = health
    chunks = ["Claro, ", "posso ajudar com isso agora mesmo, ",
              "é só me dizer o que precisa."]
    sentence = "".join(chunks)
    audio = (np.full(2400, 0.2, np.float32), 24000)
    whole = MagicMock(return_value=audio)
    streaming = MagicMock(side_effect=lambda *a: iter([audio]))
    monkeypatch.setattr(vc, "synthesize", whole)
    monkeypatch.setattr(vc, "synthesize_stream", streaming)
    chat = vc.VoiceChat(config, play=lambda *a: None, log=lambda *a: None,
                        llm=FakeLLM(chunks if fragmented else [sentence]))
    chat.respond("explique")
    if enabled and advertised is True:
        assert [call.args[0] for call in streaming.call_args_list] == [sentence]
        whole.assert_not_called()
    else:
        assert [call.args[0] for call in whole.call_args_list] == [
            "Claro, posso ajudar com isso agora mesmo,", "é só me dizer o que precisa."]
        streaming.assert_not_called()


def test_first_chunk_uses_twelve_complete_words_without_punctuation():
    head = "uma duas três quatro cinco seis sete oito nove dez onze doze"
    assert vc.split_sentences(head + " tre", first=True) == ([head], "tre")
    assert vc.split_sentences(head[:-1], first=True) == ([], head[:-1])
    # A clause later in the same delta must not move the cut past word 12.
    assert vc.split_sentences(head + " treze quatorze, depois", first=True) == (
        [head], "treze quatorze, depois")
    assert vc.split_sentences("Eu vi, mas ainda não terminei", first=True)[0] == []


@pytest.mark.parametrize("enabled", [True, False])
def test_first_sentence_with_empty_capabilities_cache(monkeypatch, enabled):
    assert vc._tts_capabilities_cache == {}
    monkeypatch.setattr(vc, "ensure_tts_server", lambda *a: None)

    def ready(config):
        vc._cache_tts_health(config, {"ok": True, "stream": True})
        return True

    monkeypatch.setattr(vc, "wait_tts_server", ready)
    chunks = ["Claro, posso ajudar com isso agora mesmo, ", "é só me dizer o que precisa."]
    requests = []
    audio = np.full(2400, 0.2, np.float32), 24000
    monkeypatch.setattr(vc, "synthesize_stream", lambda text, cfg: requests.append(text) or iter([audio]))
    monkeypatch.setattr(vc, "synthesize", lambda text, cfg: requests.append(text) or audio)
    chat = vc.VoiceChat({**DEFAULT_CONFIG, "tts_stream": enabled}, play=lambda *a: None,
                        log=lambda *a: None, llm=FakeLLM(chunks))
    chat.respond("oi")
    assert requests == (["".join(chunks)] if enabled else [c.strip() for c in chunks])
    assert vc.split_sentences("Eu posso explicar — depois", first=True)[0] == []
    assert vc.split_sentences("a b c d, resto", first=True)[0] == []


def test_warmup_discards_audio_and_caches_waiting_phrase(server):
    chat, played = _chat(server, FakeLLM())
    for _ in range(2):
        worker = chat.warm_tts()
        worker.join(5)
        assert not worker.is_alive()
    assert _tts_texts(server) == ["Olá, estou pronta.", "Um instante."]
    assert not played
    assert chat._waiting_audio(chat.config) is chat._waiting_audio(chat.config)
    assert len(server.requests) == 2


@pytest.mark.parametrize("key,value", [("tts_voice", "another"), ("language", "en"),
                                      ("tts_backend", "kokoro"),
                                      ("tts_url", "http://127.0.0.1:9999")])
def test_waiting_cache_is_invalidated_by_settings(key, value, monkeypatch):
    synth = MagicMock(side_effect=[(AUDIO, 16000), (AUDIO.copy(), 16000)])
    monkeypatch.setattr(vc, "synthesize", synth)
    chat = vc.VoiceChat({**DEFAULT_CONFIG, "language": "pt"})
    old = chat._waiting_audio(chat.config)
    chat._tts_warmed = True
    assert chat._waiting_audio(chat.config) is old
    chat.config[key] = value
    assert chat._waiting_audio(chat.config) is not old
    assert not chat._tts_warmed
    assert synth.call_count == 2


def test_warmup_returns_while_server_loads_and_failure_does_not_poison_cache(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def wait(config):
        entered.set()
        assert release.wait(5)
        return True

    monkeypatch.setattr(vc, "wait_tts_server", wait)
    synth = MagicMock(side_effect=[vc.TTSRejected("cold"), (AUDIO, 16000)])
    monkeypatch.setattr(vc, "synthesize", synth)
    chat = vc.VoiceChat(dict(DEFAULT_CONFIG), log=lambda m: None)
    worker = chat.warm_tts()
    try:
        assert entered.wait(5) and worker.is_alive()
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert chat._waiting_audio(chat.config)[1] == 16000
    assert synth.call_count == 2


def test_waiting_phrase_is_reused_across_turns(server, monkeypatch):
    # Trigger feedback deterministically, before the writer's first delta.
    def timer(delay, callback):
        return MagicMock(start=callback)

    monkeypatch.setattr(vc.threading, "Timer", timer)
    chat, played = _chat(server, FakeLLM(["A capital é Canberra."]), voice_chat_backend="claude")
    logs = []
    chat.log = logs.append
    chat.respond("capital")
    chat.respond("capital")
    assert _tts_texts(server) == ["Um instante.", "A capital é Canberra.", "A capital é Canberra."]
    assert len(played) == 4
    assert len([line for line in logs if " cached" in line]) == 1


# --- Switching voice chat while running ---------------------------------------

@pytest.fixture
def loader(monkeypatch):
    """The engine's LLM loading, recorded instead of run."""
    calls = []
    monkeypatch.setattr(de, "ensure_tts_server", lambda *a: calls.append("tts"))
    # Never warm the user's real server from the engine loading tests.
    monkeypatch.setattr(vc.VoiceChat, "warm_tts", MagicMock())
    monkeypatch.setattr(de, "download_llm", lambda *a: calls.append("download"))
    monkeypatch.setattr(de, "llm_loaded", lambda config: False)

    def load_llm(config, log, log_path, stop):
        calls.append(("llm", de._inference_gate.locked()))
        return object()

    monkeypatch.setattr(de, "load_llm", load_llm)
    return calls


def _notices(app):
    notices = []
    app.add_callback(lambda s, d: d.get("notice") and notices.append(d["notice"]))
    return notices


def _wait_for(condition, timeout=5):
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline
        time.sleep(0.02)


def test_switching_on_loads_the_llm_in_the_background(loader):
    app = _app()
    notices = _notices(app)
    app.set_voice_chat(True)
    _wait_for(lambda: "Voice chat ready" in notices)
    assert app.config["voice_chat"] is True
    # Downloaded first, then compiled with every transcription held back.
    assert loader == ["tts", "download", ("llm", True)]
    vc.VoiceChat.warm_tts.assert_called_once()
    assert notices == ["Voice chat: loading the LLM...", "Voice chat ready"]


def test_the_llm_waits_for_the_speech_model(loader):
    """Compiling the LLM on the iGPU during the NPU's warmup lost the NPU."""
    app = _app()
    app._model_ready.clear()
    app.set_voice_chat(True)
    time.sleep(0.3)
    assert loader == ["tts"]
    app._model_ready.set()
    _wait_for(lambda: len(loader) == 3)


def test_a_loaded_llm_is_ready_at_once(loader, monkeypatch):
    monkeypatch.setattr(de, "llm_loaded", lambda config: True)
    app = _app()
    notices = _notices(app)
    app.set_voice_chat(True)
    _wait_for(lambda: notices)
    assert notices == ["Voice chat ready"]
    assert loader == ["tts"]


def test_failed_llm_load_is_told(loader, monkeypatch):
    monkeypatch.setattr(de, "load_llm", MagicMock(side_effect=RuntimeError("no GPU")))
    app = _app()
    notices = _notices(app)
    app.set_voice_chat(True)
    _wait_for(lambda: len(notices) == 2)
    assert notices[-1] == "Voice chat: the LLM did not load (see app.log)"


def test_transcription_waits_while_the_llm_compiles(typed):
    app = _app()
    app.whisper.transcribe.return_value = "ola"
    with de._inference_gate:
        worker = threading.Thread(target=app._finish_recording,
                                  kwargs={"audio": AUDIO, "is_final": True})
        worker.start()
        time.sleep(0.3)
        app.whisper.transcribe.assert_not_called()
    worker.join(5)
    app.whisper.transcribe.assert_called_once()
    assert typed == ["ola "]


def test_switching_off_stops_the_reply_and_forgets_the_conversation():
    app = _app(voice_chat=True)
    app.voice_chat._history = [{"role": "user", "content": "oi"}]
    with patch.object(app.voice_chat, "interrupt") as interrupt:
        app.set_voice_chat(False)
    interrupt.assert_called_once()
    assert app.voice_chat._history == []
    assert app.config["voice_chat"] is False


@pytest.mark.parametrize("readable", [True, False], ids=["verified", "backspace"])
@pytest.mark.usefixtures("windows_text_provider")
def test_sentence_typed_as_dictation_goes_to_the_llm_after_the_switch(typed, monkeypatch, readable):
    app = _app(voice_chat=False, continuous_listening=True, inline_drafts=True)
    monkeypatch.setattr(de.platforms.current(), "text_before_caret",
                        lambda count: app._draft_typed_text[-count:] if readable else None)
    app.is_recording = True
    _say(app, "ola tudo", is_final=False)  # typed as a dictation draft
    draft_method, delete_method = ("draft", "delete") if readable else ("draft", "backspace")
    assert typed == [(draft_method, "ola tudo... ")]
    app.config["voice_chat"] = True
    with patch.object(app.voice_chat, "respond", return_value="Oi!") as respond:
        _say(app, "ola tudo bem")
    respond.assert_called_once()
    assert respond.call_args.args[0] == "ola tudo bem"
    assert typed == [(draft_method, "ola tudo... "), (delete_method, len("ola tudo... "))]
    assert app._draft_typed_text == ""
    assert app._draft_target is None
    assert app._draft_strategy is None


def _gui(**config):
    from debora_whisper.app import GUIApp
    gui = GUIApp.__new__(GUIApp)
    gui._config = {**DEFAULT_CONFIG, **config}
    gui._engine = MagicMock()
    gui._engine.stop_if_idle.return_value = None
    gui._overlay, gui._tray = MagicMock(), MagicMock()
    gui._settings_win = MagicMock(is_open=True)
    gui._settings_status, gui._settings_set_apply = MagicMock(), MagicMock()
    return gui


def test_tray_item_switches_without_a_new_engine():
    gui = _gui()
    engine = gui._engine
    with patch.dict(gui._set_voice_chat.__globals__, {"save_config": MagicMock()}) as g:
        gui._set_voice_chat(True)
        g["save_config"].assert_called_once()
    assert gui._config["voice_chat"] is True
    engine.set_voice_chat.assert_called_once_with(True, listen=True)
    gui._settings_win.set_voice_chat.assert_called_once_with(True)
    assert gui._engine is engine


def test_settings_switch_voice_chat_without_a_new_engine():
    gui = _gui()
    engine = gui._engine
    factory = MagicMock()
    with patch.dict(gui._on_settings_apply.__globals__,
                    {"save_config": MagicMock(), "DictationApp": factory}):
        gui._on_settings_apply({**gui._config, "voice_chat": True})
    engine.set_voice_chat.assert_called_once_with(True)
    engine.stop_if_idle.assert_not_called()
    factory.assert_not_called()


@pytest.mark.parametrize("name", ["", "   ", "\t\r\n", "Debora & whoami", "Debora\n", "x" * 81])
def test_settings_apply_rejects_invalid_session_name(name, monkeypatch):
    from debora_whisper.ui.settings import SettingsWindow

    window = SettingsWindow.__new__(SettingsWindow)
    original = dict(DEFAULT_CONFIG)
    window._config = original.copy()
    for attr, value in {
        "_model_radio_var": original["model_size"], "_hotkey_var": original["hotkey"],
        "_beep_var": True, "_enter_var": False, "_inline_drafts_var": False,
        "_voice_chat_var": True, "_backend_var": "Claude Code",
        "_harness_new_session_var": True, "_harness_session_name_var": name,
        "_voice_var": "test", "_balloon_var": True, "_font_size_var": "16",
        "_tts_engine_var": "Chatterbox (clones the voice)",
    }.items():
        setattr(window, attr, MagicMock(get=MagicMock(return_value=value)))
    window._harness_cwd = None
    window._get_selected_lang_code = lambda: "pt"
    window._status_label, window._win = MagicMock(), MagicMock()
    window._on_apply = MagicMock()
    monkeypatch.setattr(de, "detect_devices", lambda: ["CPU"])
    monkeypatch.setattr(de, "select_device", lambda *args: "CPU")

    window._apply()

    assert window._config == original
    window._on_apply.assert_not_called()
    assert "harness_session_name" in window._status_label.configure.call_args.kwargs["text"]


@pytest.mark.parametrize("name", ["", "   ", "Debora | whoami"])
def test_settings_callback_rejects_invalid_name_before_save(name):
    gui = _gui()
    original = gui._config.copy()
    with patch.dict(gui._on_settings_apply.__globals__, {"save_config": MagicMock()}) as g:
        gui._on_settings_apply({**original, "harness_session_name": name})
        g["save_config"].assert_not_called()
    assert gui._config == original
    gui._engine.stop_if_idle.assert_not_called()
    assert "harness_session_name" in gui._settings_status.call_args.args[0]


def test_rebuilt_engine_starts_voice_chat_itself():
    gui = _gui()
    old = gui._engine
    with patch.dict(gui._on_settings_apply.__globals__,
                    {"save_config": MagicMock(), "DictationApp": MagicMock(),
                     "is_model_downloaded": lambda size: True}):
        gui._on_settings_apply({**gui._config, "voice_chat": True, "hotkey": "ctrl+alt+v"})
    old.set_voice_chat.assert_not_called()


def test_notice_shows_only_in_the_tray_without_changing_state():
    gui = _gui()
    gui._update_ui(de.AppState.RECORDING, {"notice": "Voice chat ready"})
    gui._overlay.show_notice.assert_not_called()
    gui._overlay.show_recording.assert_not_called()
    gui._tray.update_state.assert_called_once_with("recording", "Débora Whisper — Voice chat ready")


@pytest.mark.parametrize("data, seconds", [
    ({"text": "Oi.", "seconds": 1.5}, 1.5), ({"text": "Oi."}, None)])
def test_speaking_passes_the_audio_length_to_the_balloon(data, seconds):
    gui = _gui()
    gui._audio_poll_id = None
    gui._update_ui(de.AppState.SPEAKING, data)
    gui._overlay.show_speaking.assert_called_once_with("Oi.", seconds)


def test_talking_only_animates_the_mascot():
    gui = _gui()
    gui._update_ui(de.AppState.SPEAKING, {"talking": "debora", "active": True})
    gui._overlay.set_talking.assert_called_once_with("debora", True)
    gui._overlay.show_speaking.assert_not_called()
    gui._tray.update_state.assert_not_called()


def test_engine_wires_both_voices_to_the_ui():
    events = []
    app = de.DictationApp.__new__(de.DictationApp)
    app._stopping = threading.Event()
    app._state = de.AppState.RECORDING
    app._callbacks = [lambda state, data: events.append(data)]
    app._talking("user", True)
    assert events == [{"talking": "user", "active": True}]


# --- Latency and playback -------------------------------------------------------

def test_voice_chat_ends_a_sentence_after_a_shorter_silence():
    config = {"voice_chat": False}
    recorder = de.AudioRecorder(sample_rate=16000, config=config)
    assert recorder.end_silence_frames == int(16000 * 1.5)
    config["voice_chat"] = True  # switched while running
    assert recorder.end_silence_frames == int(16000 * 0.8)
    config["voice_chat_end_silence_seconds"] = 0.6
    assert recorder.end_silence_frames == int(16000 * 0.6)


def test_silence_around_a_clip_is_trimmed():
    rate = 1000
    speech = np.full(500, 0.2, dtype=np.float32)
    clip = np.concatenate([np.zeros(300, np.float32), speech, np.zeros(700, np.float32)])
    trimmed = vc.trim_silence(clip, rate)
    assert len(trimmed) == 500 + 2 * 50  # 50 ms kept on each side
    assert len(vc.trim_silence(np.zeros(100, np.float32), rate)) == 0


def test_hiss_after_the_speech_is_cut():
    rate = 1000
    speech = np.full(500, 0.2, dtype=np.float32)
    hiss = np.full(400, 0.008, dtype=np.float32)  # -42 dB, like Chatterbox's
    clip = np.concatenate([speech, np.zeros(1000, np.float32), hiss])
    trimmed = vc.trim_silence(clip, rate)
    assert len(trimmed) == 500 + 50  # the speech and 50 ms of its pause
    assert trimmed[-1] == 0 and trimmed[0] == np.float32(0.2)


def test_a_pause_before_more_speech_is_kept():
    rate = 1000
    speech = np.full(500, 0.2, dtype=np.float32)
    clip = np.concatenate([speech, np.zeros(1000, np.float32), speech])
    assert len(vc.trim_silence(clip, rate)) == 2000


@pytest.mark.parametrize("language, text, spoken", [
    ("pt", "Hoje é sexta-feira, dia 09 outubro de 2026, às 10h18.",
     "Hoje é sexta-feira, dia nove outubro de dois mil e vinte e seis, às dez e dezoito."),
    ("pt", "Em 09/10/2026, às 21:05, 1h ou 2h.",
     "Em nove de outubro de dois mil e vinte e seis, às vinte e uma e cinco, "
     "uma hora ou duas horas."),
    ("pt", "O 1º lugar, 3,5% e 1.500 reais.",
     "O primeiro lugar, três vírgula cinco por cento e mil e quinhentos reais."),
    ("en", "It is 10/09/2026 at 10:05, 3.5%.",
     "It is October ninth, twenty twenty-six at ten oh five, three point five percent."),
    ("auto", "Oi 2", "Oi 2"),
    ("xx", "Oi 2", "Oi 2"),
])
def test_numbers_are_spoken_as_words(language, text, spoken):
    assert vc.spoken_numbers(text, language) == spoken


@pytest.mark.parametrize("language", ["pt", "en"])
@pytest.mark.parametrize("token", [
    "config.json.bak-20261010", "v2.1.296", "RTX4060", "abc123def456",
    "20261010.json", "backup_20261010", "2026-10-10", "/2026/10/log.txt",
    "backup-09/10/2026", "log-14:07", "v10h18", "build1h", "build3.5%",
    "item1º", "item2ª", "3-files", "12_34", "2.1.296", "3/4",
    r"C:\logs\20261010.txt", "file_2.txt", "a/b/3",
    "app.py:10:20", "app.py:10", r"C:\123", "src/app.py:42",
])
def test_numbers_inside_identifiers_are_unchanged(language, token):
    text = f"Arquivo ({token})."
    assert vc.spoken_numbers(text, language) == text


@pytest.mark.parametrize("language", ["pt", "en"])
@pytest.mark.parametrize("text, pt, en", [
    ("-10", "-dez", "-ten"),
    ("Valor -10", "Valor -dez", "Valor -ten"),
    ("(-10)", "(-dez)", "(-ten)"),
    ("10-20", "dez-vinte", "ten-twenty"),
    ("-5%", "-cinco por cento", "-five percent"),
    ("Total:10", "Total:dez", "Total:ten"),
    ("14:07", "catorze e sete", "fourteen oh seven"),
    ("09/10/2026", "nove de outubro de dois mil e vinte e seis",
     "September tenth, twenty twenty-six"),
])
def test_numeric_expressions_are_spoken_as_words(language, text, pt, en):
    assert vc.spoken_numbers(text, language) == (pt if language == "pt" else en)


@pytest.mark.parametrize("text, spoken", [
    ("3 arquivos", "três arquivos"),
    ("10,5", "dez vírgula cinco"),
    ("2026", "dois mil e vinte e seis"),
    ("foram 3.", "foram três."),
    ("3. Depois 4!", "três. Depois quatro!"),
    ("foram 3...", "foram três..."),
    ("3: três arquivos.", "três: três arquivos."),
    ("às 14:07", "às catorze e sete"),
    ("(3), 4; 5?", "(três), quatro; cinco?"),
    ("10 % e 2ª feira", "dez por cento e segunda feira"),
    ("config.json.bak-20261010 e 3 arquivos.", "config.json.bak-20261010 e três arquivos."),
])
def test_standalone_numbers_keep_sentence_punctuation(text, spoken):
    assert vc.spoken_numbers(text, "pt") == spoken


@pytest.mark.parametrize("sentence, offer", [
    ("Como posso te ajudar hoje?", True),
    ("Pode dizer o que quer que eu faça?", True),
    ("Vai me dizer o que precisa?", True),
    ("O que deseja fazer agora?", True),
    ("How can I help you today?", True),
    ("O que você quer dizer com isso?", False),
    ("Posso ajudar com o ditado.", False),
    ("Pode repetir?", False),
])
def test_offers_of_help_are_recognized(sentence, offer):
    assert vc.is_help_offer(sentence) is offer


def test_a_closing_offer_of_help_is_neither_said_nor_remembered(server):
    llm = FakeLLM(["Que legal, Alexandre! Pode dizer o que quer que eu faça?"])
    chat, _ = _chat(server, llm)
    assert chat.respond("eu sou o Alexandre") == "Que legal, Alexandre!"
    assert _tts_texts(server) == ["Que legal, Alexandre!"]
    assert chat._history[-1] == {"role": "assistant", "content": "Que legal, Alexandre!"}


def test_a_reply_that_is_only_an_offer_of_help_is_kept(server):
    chat, _ = _chat(server, FakeLLM(["Como posso ajudar você hoje?"]))
    assert chat.respond("oi Débora") == "Como posso ajudar você hoje?"


def test_the_overlay_keeps_digits_but_the_tts_gets_words(server):
    chat, _ = _chat(server, FakeLLM(["Hoje é dia 9 de outubro, às 10h18."]))
    assert chat.respond("que dia é hoje") == "Hoje é dia 9 de outubro, às 10h18."
    assert _tts_texts(server) == ["Hoje é dia nove de outubro, às dez e dezoito."]


class FakeStream:
    instances = []

    def __init__(self, samplerate, channels, dtype):
        self.rate, self.written, self.ended = samplerate, [], None
        self.latency = 0.0
        FakeStream.instances.append(self)

    def start(self):
        pass

    def write(self, data):
        self.written.append(len(data))

    def stop(self):
        self.ended = "drained"

    def abort(self):
        self.ended = "dropped"

    def close(self):
        pass


@pytest.fixture
def stream(monkeypatch):
    FakeStream.instances = []
    monkeypatch.setitem(sys.modules, "sounddevice", MagicMock(OutputStream=FakeStream))
    return FakeStream.instances


def test_clips_of_a_reply_share_one_stream_with_a_pause_between(stream):
    player, stop = vc.StreamPlayer(), threading.Event()
    player(np.ones(24000, np.float32), 24000, stop)
    player(np.ones(24000, np.float32), 24000, stop)
    player.close(interrupted=False)
    assert len(stream) == 1
    gap = int(24000 * vc.SENTENCE_GAP_SECONDS)
    assert sum(stream[0].written) == 2 * 24000 + gap
    assert stream[0].ended == "drained"  # the last clip plays to its end


def test_stream_continuations_have_no_sentence_gap(stream):
    player, stop = vc.StreamPlayer(), threading.Event()
    try:
        for continuation in [False, True, True, False, True]:
            player(np.ones(2400, np.float32), 24000, stop, continuation=continuation)
    finally:
        player.close(interrupted=False)
    assert sum(stream[0].written) == 5 * 2400 + int(24000 * vc.SENTENCE_GAP_SECONDS)


@pytest.fixture
def streaming_chat(server, monkeypatch):
    server.health_body = b'{"ok": true, "stream": true}'
    monkeypatch.setitem(DEFAULT_CONFIG, "tts_url", server.tts_url)
    monkeypatch.setattr(vc, "synthesize", MagicMock(side_effect=AssertionError("whole WAV requested")))


def test_stream_playback_starts_before_generation_ends(streaming_chat, stream, monkeypatch):
    began = threading.Event()
    ended, requests, shown, logs = [], [], [], []
    original_write = FakeStream.write

    def write(output, data):
        original_write(output, data)
        began.set()

    def synth(text, config):
        requests.append(text)
        began.clear()
        yield np.full(2400, 0.2, np.float32), 24000
        assert began.wait(3), "first audio waited for the stream to end"
        yield np.full(4800, 0.2, np.float32), 24000
        ended.append(text)

    monkeypatch.setattr(FakeStream, "write", write)
    monkeypatch.setattr(vc, "synthesize_stream", synth)
    chat = vc.VoiceChat(dict(DEFAULT_CONFIG), log=logs.append,
                        llm=FakeLLM(["Primeira frase longa. Segunda frase longa."]))
    chat._remember_spoken = MagicMock(wraps=chat._remember_spoken)
    chat._synthesis_model.update = MagicMock(wraps=chat._synthesis_model.update)
    chat.respond("oi", on_reply=lambda *args: shown.append(args))
    assert requests == ended == ["Primeira frase longa.", "Segunda frase longa."]
    assert len(shown) == 2
    assert chat._remember_spoken.call_count == 4  # every speech clip, including continuations
    assert sum(call.args[1] for call in chat._remember_spoken.call_args_list) == pytest.approx(0.6)
    assert chat._synthesis_model.update.call_count == 2
    for call in chat._synthesis_model.update.call_args_list:
        assert call.args[2:] == pytest.approx((0.3, 0.3))
    assert sum(stream[0].written) == 14400 + int(24000 * vc.SENTENCE_GAP_SECONDS)
    assert not any("prebuffer held" in line for line in logs)
    timings = [line for line in logs if " chunk=" in line]
    assert len(timings) == 2
    for line in timings:
        times = {key: float(value) for key, value in re.findall(r"(\w+)=(\d+\.\d+)s", line)}
        assert times["synth_start"] <= times["first_chunk"] <= times["play"] <= times["synth_end"]
        assert times["audio"] == pytest.approx(0.3)


def test_stream_audio_indicator_bridges_fifty_ms_gaps(streaming_chat, monkeypatch):
    from types import SimpleNamespace
    clock = [100.0]
    monkeypatch.setattr(vc, "time", SimpleNamespace(**{**vars(time), "monotonic": lambda: clock[0]}))
    gaps = [threading.Event() for _ in range(3)]
    quiet = threading.Event()
    played, events, off_under_lock = [], [], []
    player_thread = threading.current_thread()

    def synth(text, config):
        for index in range(3):
            yield np.full(100, 0.2, np.float32), 1000
            assert gaps[index].wait(3), "player never reached the empty queue"
            clock[0] += 0.05 if index < 2 else 0.15
        # Keep the turn open to observe the debounced off, before final cleanup.
        with chat._playback:
            chat._playback.notify_all()
        assert quiet.wait(3), "indicator did not turn off after sustained idle"

    def play(samples, rate, stop):
        played.append(clock[0])

    chat = vc.VoiceChat(dict(DEFAULT_CONFIG), play=play,
                        llm=FakeLLM(["Primeira frase longa."]))
    original_wait = chat._playback.wait

    def wait(timeout=None):
        if threading.current_thread() is player_thread and played:
            gaps[len(played) - 1].set()
        return original_wait(timeout)

    def on_audio(active):
        events.append(active)
        if not active:
            off_under_lock.append(chat._playback._is_owned())
            quiet.set()

    monkeypatch.setattr(vc, "synthesize_stream", synth)
    monkeypatch.setattr(chat._playback, "wait", wait)
    chat.on_audio = on_audio
    chat.respond("oi")
    assert played == pytest.approx([100.0, 100.05, 100.1])
    assert events == [True, False]
    assert off_under_lock == [False]


@pytest.mark.parametrize("interrupt", [False, True])
def test_stream_failure_or_stop_does_not_replay(streaming_chat, monkeypatch, interrupt):
    began, closed = threading.Event(), threading.Event()
    played, requests, logs, shown = [], [], [], []

    def synth(text, config):
        requests.append(text)
        try:
            yield np.full(2400, 0.2, np.float32), 24000
            assert began.wait(3)
            if interrupt:
                yield np.full(2400, 0.3, np.float32), 24000
            else:
                raise OSError("stream broke")
        finally:
            closed.set()

    def play(samples, rate, stop):
        played.append(samples.copy())
        if interrupt:
            chat.interrupt()
        began.set()

    monkeypatch.setattr(vc, "synthesize_stream", synth)
    chat = vc.VoiceChat(dict(DEFAULT_CONFIG), play=play, log=logs.append,
                        llm=FakeLLM(["Primeira frase longa."]))
    chat.respond("oi", on_reply=lambda *args: shown.append(args))
    assert closed.is_set() and len(requests) == len(played) == len(shown) == 1
    if interrupt:
        assert chat._tts_cancellation.cancelled
    else:
        assert any("Voice chat: TTS failed" in line for line in logs)


@pytest.mark.parametrize("enabled,advertised", [
    (True, True), (False, True), (True, False), (False, False), (True, None), (True, "true")])
def test_stream_requires_config_and_health(server, monkeypatch, enabled, advertised):
    server.health_body = json.dumps({"ok": True, "stream": advertised}).encode()
    whole = MagicMock(return_value=(np.full(2400, 0.2, np.float32), 24000))
    streaming = MagicMock(return_value=iter([(np.full(2400, 0.2, np.float32), 24000)]))
    monkeypatch.setattr(vc, "synthesize", whole)
    monkeypatch.setattr(vc, "synthesize_stream", streaming)
    chat = vc.VoiceChat(_cfg(server, tts_stream=enabled), play=lambda *a: None,
                        llm=FakeLLM(["Primeira frase longa."]), log=lambda *a: None)
    chat.respond("oi")
    assert server.health_requests >= 1
    assert vc._tts_loading_health(chat.config) == {}
    assert streaming.call_count == int(enabled and advertised is True)
    assert whole.call_count == int(not (enabled and advertised is True))


@pytest.mark.parametrize("paused_clip", [0, 1])
def test_stream_pause_resume_restores_echo_intervals(streaming_chat, monkeypatch, paused_clip):
    clock = [100.0]
    resumed_trims = []
    original_trim = vc.trim_silence

    def trim(samples, rate, **kwargs):
        result = original_trim(samples, rate, **kwargs)
        if not kwargs:  # default leading/trailing (including hiss) trim on replay
            resumed_trims.append(result)
        return result

    monkeypatch.setattr(vc, "trim_silence", trim)
    # Replace only voice_chat's clock so thread/event deadlines still use real time.
    from types import SimpleNamespace
    monkeypatch.setattr(vc, "time", SimpleNamespace(**{**vars(time), "monotonic": lambda: clock[0]}))
    monkeypatch.setattr(vc, "synthesize_stream", lambda *a: iter([
        (np.full(1000, 0.2, np.float32), 1000),
        (np.full(1000, 0.3, np.float32), 1000)]))
    spoken = "Primeira frase longa."
    calls, shown, logs = [], [], []
    paused = False
    resumed_at = []

    def play(samples, rate, stop):
        nonlocal paused
        calls.append(samples.copy())
        begin = clock[0]
        if paused:
            assert chat._spoken[-1][0] >= resumed_at[0]
            assert len(chat._spoken) == 2  # a fresh interval, not one spanning the pause
        # A one-word echo requires coverage throughout the capture, avoiding
        # a false pass from the two-second room tail of an earlier interval.
        assert chat.is_echo("Primeira", begin + 0.01, begin + len(samples) / rate)
        if len(calls) - 1 == paused_clip and not paused:
            clock[0] += 0.01
            assert chat.pause()
            assert stop.is_set()
            assert all(end <= clock[0] for _, end, _ in chat._spoken)
            clock[0] += 5  # beyond any echo tail from before the pause
            assert not chat.is_echo(spoken, clock[0], clock[0] + 0.01)
            assert chat.resume()
            resumed_at.append(clock[0])
            paused = True
        else:
            clock[0] += len(samples) / rate

    chat = vc.VoiceChat(dict(DEFAULT_CONFIG), play=play, log=logs.append, llm=FakeLLM([spoken]))
    chat.respond("oi", on_reply=lambda *args: shown.append(args))
    assert paused and len(calls) == 3  # two speech clips plus the replay
    np.testing.assert_array_equal(calls[paused_clip], calls[paused_clip + 1])
    assert len(resumed_trims) == 1
    np.testing.assert_array_equal(resumed_trims[0], calls[paused_clip + 1])
    assert len(shown) == 1
    assert sum(" chunk=" in line for line in logs) == 1


def test_stream_trims_only_sentence_edges(streaming_chat, monkeypatch):
    # Silence at chunk boundaries is an intentional pause, including an
    # entirely silent middle chunk; only the sentence edges are trimmed.
    chunks = [np.r_[np.zeros(100), np.full(100, 0.2), np.zeros(100)].astype(np.float32),
              np.zeros(100, np.float32),
              np.r_[np.zeros(100), np.full(100, 0.2), np.zeros(100)].astype(np.float32)]
    monkeypatch.setattr(vc, "synthesize_stream", lambda *a: iter((c, 1000) for c in chunks))
    played = []
    chat = vc.VoiceChat(dict(DEFAULT_CONFIG), play=lambda samples, *a: played.append(samples),
                        llm=FakeLLM(["Primeira frase longa."]), log=lambda *a: None)
    chat.respond("oi")
    expected = np.concatenate(chunks)[50:-50]
    np.testing.assert_array_equal(np.concatenate(played), expected)


def test_stream_single_chunk_trims_long_final_silence(streaming_chat, monkeypatch):
    samples = np.r_[np.zeros(100), np.full(200, 0.2), np.zeros(300)].astype(np.float32)
    monkeypatch.setattr(vc, "synthesize_stream", lambda *a: iter([(samples, 1000)]))
    played = []
    chat = vc.VoiceChat(dict(DEFAULT_CONFIG), play=lambda samples, *a: played.append(samples),
                        llm=FakeLLM(["Primeira frase longa."]), log=lambda *a: None)
    chat.respond("oi")
    np.testing.assert_array_equal(np.concatenate(played), vc.trim_silence(samples, 1000))


def test_stream_releases_speech_before_waiting_for_tail(streaming_chat, monkeypatch):
    heard = threading.Event()
    speech = np.full(200, 0.2, np.float32)
    played = []

    def synth(*args):
        yield np.r_[speech, np.zeros(400)].astype(np.float32), 1000
        assert heard.wait(3), "speech was held with its silent tail"
        np.testing.assert_array_equal(np.concatenate(played), speech)
        yield np.zeros(200, np.float32), 1000

    def play(samples, *args):
        played.append(samples.copy())
        heard.set()

    monkeypatch.setattr(vc, "synthesize_stream", synth)
    chat = vc.VoiceChat(dict(DEFAULT_CONFIG), play=play, llm=FakeLLM(["Uma frase longa."]), log=lambda *a: None)
    chat.respond("oi")
    assert len(np.concatenate(played)) == 250


@pytest.mark.parametrize("chunk_size", [100, 2000])
def test_stream_trims_hiss_after_acoustic_pause(streaming_chat, monkeypatch, chunk_size):
    samples = np.r_[np.full(200, 0.2), np.zeros(1000), np.full(400, 0.008)].astype(np.float32)
    monkeypatch.setattr(vc, "synthesize_stream", lambda *a: iter(
        (samples[i:i + chunk_size], 1000) for i in range(0, len(samples), chunk_size)))
    played = []
    chat = vc.VoiceChat(dict(DEFAULT_CONFIG), play=lambda s, *a: played.append(s),
                        llm=FakeLLM(["Uma frase longa."]), log=lambda *a: None)
    chat.respond("oi")
    np.testing.assert_array_equal(np.concatenate(played), vc.trim_silence(samples, 1000))


def test_stream_pause_replays_previous_device_buffer(streaming_chat, stream, monkeypatch):
    chunks = [np.full(2400, value, np.float32) for value in (0.2, 0.3)]
    monkeypatch.setattr(vc, "synthesize_stream", lambda *a: iter((s, 24000) for s in chunks))
    original_init, original_write = FakeStream.__init__, FakeStream.write
    captured = []

    def init(output, *args, **kwargs):
        original_init(output, *args, **kwargs)
        output.latency = 1.0 if len(stream) == 1 else 0
        captured.append([])

    def write(output, data):
        original_write(output, data)
        captured[-1].append(data.ravel().copy())
        if len(stream) == 1 and np.any(data == np.float32(0.3)):
            assert chat.pause()
            assert chat.resume()

    monkeypatch.setattr(FakeStream, "__init__", init)
    monkeypatch.setattr(FakeStream, "write", write)
    chat = vc.VoiceChat(dict(DEFAULT_CONFIG), llm=FakeLLM(["Uma frase longa."]), log=lambda *a: None)
    chat.respond("oi")
    assert len(stream) == 2 and stream[0].ended == "dropped"
    np.testing.assert_array_equal(np.concatenate(captured[1]), np.concatenate(chunks))


def test_stream_reply_deadline_spans_sentences(streaming_chat, monkeypatch):
    requests, logs = [], []
    expired = threading.Event()
    clock = [100.0]
    from types import SimpleNamespace
    monkeypatch.setattr(vc, "time", SimpleNamespace(**{**vars(time), "monotonic": lambda: clock[0]}))

    def log(line):
        logs.append(line)
        if "reply deadline exceeded" in line:
            expired.set()

    def synth(text, config):
        requests.append(text)
        clock[0] += 6
        if len(requests) == 2:
            with chat._playback:
                chat._playback.notify_all()
            assert expired.wait(3)
        yield np.full(100, 0.2, np.float32), 1000

    monkeypatch.setattr(vc, "synthesize_stream", synth)
    chat = vc.VoiceChat({**DEFAULT_CONFIG, "tts_reply_timeout_seconds": 10},
                        play=lambda *a: None, log=log,
                        llm=FakeLLM(["Primeira frase longa. Segunda frase longa. Terceira frase longa. Quarta frase longa."]))
    # Keep startup time out of this short regression deadline.
    monkeypatch.setattr(vc, "wait_tts_server", lambda cfg: True)
    vc._cache_tts_health(chat.config, {"ok": True, "stream": True})
    chat.respond("oi")
    assert len(requests) == 2
    assert any("reply deadline exceeded" in line for line in logs)
    assert chat._tts_cancellation.cancelled


@pytest.mark.parametrize("text_only_clips", [False, True])
def test_stream_deadline_cancels_speech_but_preserves_llm_turn(streaming_chat, monkeypatch, text_only_clips):
    expired = threading.Event()
    displayed, quiet = threading.Event(), threading.Event()
    quiet_before_turn_end = []
    turn_stop = threading.Event()
    observed, played, shown = [], [], []
    clock = [100.0]
    from types import SimpleNamespace
    monkeypatch.setattr(vc, "time", SimpleNamespace(**{**vars(time), "monotonic": lambda: clock[0]}))

    def log(line):
        if "only speech cancelled" in line:
            expired.set()

    def llm(messages, on_text, stop):
        # Time spent in Claude before speech must not consume the TTS budget.
        clock[0] += 100
        on_text("Primeira frase longa. ")
        assert expired.wait(3)
        observed.append((stop is turn_stop, stop.is_set(), chat._tts_cancellation.cancelled))
        if text_only_clips:
            for index in range(8):
                displayed.clear()
                on_text(f"Continuação depois do prazo número {index}. ")
                assert displayed.wait(3)
            # Keep the LLM turn open: final cleanup must not mask a stuck indicator.
            quiet_before_turn_end.append(quiet.wait(1))
        on_text("Resposta depois do prazo.")

    def synth(text, config):
        yield np.full(100, 0.2, np.float32), 1000
        assert expired.wait(3)

    def play(samples, rate, stop):
        played.append(samples)
        clock[0] += 11
        with chat._playback:
            chat._playback.notify_all()
        assert expired.wait(3)
        observed.append(stop.is_set())

    monkeypatch.setattr(vc, "synthesize_stream", synth)
    chat = vc.VoiceChat({**DEFAULT_CONFIG, "tts_reply_timeout_seconds": 10},
                        llm=llm, play=play, log=log)
    chat.on_audio = lambda active: quiet.set() if not active else None

    def on_reply(text, *args):
        shown.append(text)
        if expired.is_set():
            clock[0] += 0.05
            displayed.set()

    reply = chat.respond("oi", stop=turn_stop, on_reply=on_reply)
    assert (True, False, True) in observed and True in observed
    assert len(played) == 1
    assert "Resposta depois do prazo." in reply
    assert "Resposta depois do prazo." in shown[-1]
    if text_only_clips:
        assert quiet_before_turn_end == [True]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("waiting", [False, True])
def test_silent_gap_resets_speech_deadline(server, monkeypatch, streaming, waiting):
    from types import SimpleNamespace
    clock = [100.0]
    began = time.monotonic()
    # Let the indicator's idle debounce elapse between the simulated jumps.
    monkeypatch.setattr(vc, "time", SimpleNamespace(
        **{**vars(time), "monotonic": lambda: clock[0] + time.monotonic() - began}))
    heard, idle = threading.Event(), threading.Event()
    requested, played, logs = [], [], []
    audio = np.full(100, 0.2, np.float32), 1000

    def synth(text, config):
        requested.append(text)
        clock[0] += 6
        return audio

    def llm(messages, on_text, stop):
        if not waiting:
            on_text("Vou rodar os testes. ")
        assert idle.wait(4), "first speech burst did not finish"
        clock[0] += 100
        # Let the watchdog observe the silent interval before the next burst.
        time.sleep(0.2)
        on_text("Os testes passaram. ")

    def play(samples, rate, stop):
        played.append(samples)
        heard.set()

    config = {**DEFAULT_CONFIG, "tts_url": server.tts_url,
              "tts_reply_timeout_seconds": 10, "voice_chat_prebuffer": False,
              "voice_chat_backend": "claude" if waiting else "local"}
    monkeypatch.setattr(vc, "wait_tts_server", lambda cfg: True)
    monkeypatch.setattr(vc, "synthesize", synth)
    monkeypatch.setattr(vc, "synthesize_stream", lambda text, cfg: iter([synth(text, cfg)]))
    vc._cache_tts_health(config, {"ok": True, "stream": streaming})
    chat = vc.VoiceChat(config, llm=llm, play=play, log=logs.append)
    monkeypatch.setattr(chat, "_show_audio", lambda active: idle.set() if not active and heard.is_set() else None)
    reply = chat.respond("oi")
    assert "Os testes passaram." in reply
    assert requested == ["Um instante." if waiting else "Vou rodar os testes.", "Os testes passaram."]
    assert len(played) == 2
    assert not any("deadline exceeded" in line for line in logs)


def test_wav_reply_deadline_cancels_synthesis(server, monkeypatch):
    from types import SimpleNamespace
    clock = [100.0]
    monkeypatch.setattr(vc, "time", SimpleNamespace(**{**vars(time), "monotonic": lambda: clock[0]}))
    expired = threading.Event()
    logs, played = [], []

    def log(line):
        logs.append(line)
        if "reply deadline exceeded" in line:
            expired.set()

    def synth(text, config):
        clock[0] += 11
        with chat._playback:
            chat._playback.notify_all()
        assert expired.wait(3)
        assert config["_tts_cancellation"].cancelled
        raise OSError("cancelled socket")

    config = {**DEFAULT_CONFIG, "tts_url": server.tts_url, "tts_reply_timeout_seconds": 10}
    monkeypatch.setattr(vc, "wait_tts_server", lambda cfg: True)
    monkeypatch.setattr(vc, "synthesize", synth)
    vc._cache_tts_health(config, {"ok": True})
    chat = vc.VoiceChat(config, llm=FakeLLM(["Uma frase longa."]),
                        play=lambda *args: played.append(args), log=log)
    assert chat.respond("oi") == "Uma frase longa."
    assert expired.is_set() and not played
    assert not any("TTS failed" in line for line in logs)


def test_stream_all_silent_sentence_shows_text(streaming_chat, monkeypatch):
    monkeypatch.setattr(vc, "synthesize_stream", lambda *a: iter([
        (np.zeros(2400, np.float32), 24000)]))
    played, shown = [], []
    chat = vc.VoiceChat(dict(DEFAULT_CONFIG), llm=FakeLLM(["Uma frase silenciosa."]),
                        play=lambda *a: played.append(a), log=lambda *a: None)
    assert chat.respond("oi", on_reply=lambda *a: shown.append(a[0])) == "Uma frase silenciosa."
    assert shown == ["Uma frase silenciosa."]
    assert played == []


def test_stream_cold_cache_non_streaming_backend_keeps_first_sentence(server, monkeypatch):
    server.health_body = json.dumps({"ok": True, "stream": False}).encode()
    # Defer discovery until the renderer's readiness check so splitting really
    # sees a cold cache, rather than health fetched by startup reconciliation.
    monkeypatch.setattr(vc, "ensure_tts_server", lambda *a: None)
    whole = MagicMock(return_value=(np.full(2400, 0.2, np.float32), 24000))
    streaming = MagicMock()
    monkeypatch.setattr(vc, "synthesize", whole)
    monkeypatch.setattr(vc, "synthesize_stream", streaming)
    sentence = "Esta primeira parte tem palavras suficientes, e esta segunda termina a frase."
    chat = vc.VoiceChat(_cfg(server), llm=FakeLLM([sentence]),
                        play=lambda *a: None, log=lambda *a: None)
    assert vc._tts_capabilities(chat.config) == {}
    # Unknown capabilities preserve the full first sentence: a cold non-streaming
    # backend trades early clause audio for avoiding fragmented streaming input.
    chat.respond("oi")
    assert whole.call_count == 1
    assert whole.call_args.args[0] == sentence
    streaming.assert_not_called()


def test_stream_renderer_join_is_bounded(streaming_chat, monkeypatch):
    release = threading.Event()
    logs = []
    monkeypatch.setattr(vc, "TTS_RENDER_JOIN_SECONDS", 0.02)

    def synth(*args):
        yield np.full(100, 0.2, np.float32), 1000
        release.wait(3)

    monkeypatch.setattr(vc, "synthesize_stream", synth)
    chat = vc.VoiceChat(dict(DEFAULT_CONFIG), play=lambda *a: chat.interrupt(),
                        log=logs.append, llm=FakeLLM(["Uma frase longa."]))
    try:
        began = time.monotonic()
        chat.respond("oi")
        assert time.monotonic() - began < 2
        assert any("renderer did not finish" in line for line in logs)
    finally:
        release.set()


def test_reply_deadline_interrupts_stalled_stream_read(pcm_server, monkeypatch):
    pcm_server.chunks = [b"\x00\x10" * 2400] * 2
    pcm_server.gate = threading.Event()
    logs, played = [], []
    config = _cfg(pcm_server, tts_reply_timeout_seconds=0.25)
    vc._cache_tts_health(config, {"ok": True, "stream": True})
    monkeypatch.setattr(vc, "wait_tts_server", lambda *a: True)
    chat = vc.VoiceChat(config, play=lambda s, *a: played.append(s),
                        log=logs.append, llm=FakeLLM(["Uma frase longa."]))
    try:
        began = time.monotonic()
        chat.respond("oi")
        assert time.monotonic() - began < 2
        assert played and any("reply deadline exceeded" in line for line in logs)
        assert chat._tts_cancellation.cancelled and not chat._tts_cancellation._sockets
    finally:
        pcm_server.gate.set()


def test_cancellation_serializes_shutdown_before_close(monkeypatch):
    cancellation = vc.TTSCancellation()
    calls = []

    class Socket:
        def shutdown(self, how):
            assert cancellation._lock.locked()
            calls.append("shutdown")

        def detach(self):
            assert cancellation._lock.locked()
            calls.append("detach")
            return 123 if calls.count("detach") == 1 else -1

    sock = Socket()
    cancellation.attach(sock)
    monkeypatch.setattr(vc.socket, "close", lambda handle: calls.append("close"))
    workers = [threading.Thread(target=cancellation.cancel) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(1)
        assert not worker.is_alive()
    assert calls[:3] == ["shutdown", "detach", "close"]
    assert calls.count("close") == 1
    with pytest.raises(vc.TTSRejected, match="cancelled"):
        cancellation.attach(Socket())


def test_stream_trims_final_silence_spanning_chunks_like_whole_wav(streaming_chat, monkeypatch):
    # XTTS sends its held tail last, so the final silence spans many chunks;
    # a pause inside the sentence spanning chunks is still played.
    whole = np.r_[np.full(250, 0.2), np.zeros(300), np.full(250, 0.2),
                  np.zeros(400)].astype(np.float32)
    chunks = [whole[i:i + 100] for i in range(0, len(whole), 100)]
    monkeypatch.setattr(vc, "synthesize_stream", lambda *a: iter((c, 1000) for c in chunks))
    played = []
    chat = vc.VoiceChat(dict(DEFAULT_CONFIG), play=lambda samples, *a: played.append(samples),
                        llm=FakeLLM(["Primeira frase longa."]), log=lambda *a: None)
    chat._synthesis_model.update = MagicMock(wraps=chat._synthesis_model.update)
    chat.respond("oi")
    expected = vc.trim_silence(whole, 1000)
    np.testing.assert_array_equal(np.concatenate(played), expected)
    assert chat._synthesis_model.update.call_args.args[3] == pytest.approx(len(expected) / 1000)


def test_streamed_sentence_counts_once_for_echo(streaming_chat, monkeypatch):
    # Ten clips of one sentence must not multiply her "não" past the user's.
    monkeypatch.setattr(vc, "synthesize_stream", lambda *a: iter(
        [(np.full(100, 0.2, np.float32), 1000)] * 10))
    heard = []

    def play(samples, rate, stop):
        now = time.monotonic()
        heard.append((chat.is_echo("não não não", now, now),
                      chat.is_echo("não sei a resposta", now, now)))

    chat = vc.VoiceChat(dict(DEFAULT_CONFIG), play=play, log=lambda *a: None,
                        llm=FakeLLM(["Não sei a resposta agora."]))
    chat.respond("oi")
    assert len(heard) >= 10
    assert heard[-1] == (False, True)
    assert len(chat._spoken) == 1


def test_stream_external_stop_closes_inflight_http(streaming_chat, pcm_server):
    pcm_server.chunks = [b"\x00\x10" * 2400] * 2
    pcm_server.gate = threading.Event()
    stop = threading.Event()
    played = []

    def play(samples, rate, clip_stop):
        played.append(samples)
        stop.set()

    chat = vc.VoiceChat(_cfg(pcm_server, tts_timeout_seconds=2), play=play,
                        llm=FakeLLM(["Primeira frase longa."]), log=lambda *a: None)
    began = time.monotonic()
    chat.respond("oi", stop=stop)
    assert time.monotonic() - began < 1.5
    assert len(played) == 1 and chat._tts_cancellation.cancelled
    assert not chat._tts_cancellation._sockets


def test_interrupt_drops_the_queued_audio(stream):
    player, stop = vc.StreamPlayer(), threading.Event()
    stop.set()
    player(np.ones(24000, np.float32), 24000, stop)
    player.close(interrupted=True)
    assert stream[0].written == [] and stream[0].ended == "dropped"


def test_reply_plays_through_one_stream_player(server, stream, monkeypatch):
    chat = vc.VoiceChat(_cfg(server), log=lambda m: None,
                        llm=FakeLLM(["A capital é Canberra. Fica no sul do país."]))
    assert chat.respond("capital") == "A capital é Canberra. Fica no sul do país."
    assert len(stream) == 1 and stream[0].ended == "drained"


@pytest.mark.shipped_defaults
def test_voice_chat_with_turbo_is_what_users_get():
    assert DEFAULT_CONFIG["voice_chat"] is True
    assert DEFAULT_CONFIG["model_size"] == "turbo"
    de.validate_config(DEFAULT_CONFIG)


def test_debora_prompt_asks_for_speakable_text_only():
    prompt = vc.DEFAULT_VOICE_CHAT_PROMPT
    assert prompt.startswith("You are Débora")
    for rule in ("emoji", "markdown", "parentheses", "URLs", "percent"):
        assert rule in prompt


def test_debora_prompt_fixes_what_the_last_chat_got_wrong():
    prompt = vc.DEFAULT_VOICE_CHAT_PROMPT
    for rule in ("feminine", "ask them to repeat", "do not know the user's name",
                 "offer of help"):
        assert rule in prompt
    assert "most likely said" not in prompt
    pt = vc.VOICE_CHAT_PROMPTS["pt"]
    assert pt.startswith("Você é a Débora")
    for rule in ("feminino", "emoji", "peça para repetir", "não sabe o nome",
                 "qual é meu nome",
                 "oferecendo ajuda", "por cento"):
        assert rule in pt


@pytest.mark.parametrize("text", ["sobre o projeto...", "sobre o projeto…", "Eu queria, mas."])
def test_voice_chat_draft_updates_endpoint_through_asr(typed, text):
    app = _app(voice_chat=True)
    app.recorder = de.AudioRecorder(config=app.config)
    segment = app.recorder.endpoint.start()
    app.whisper.transcribe.return_value = text
    app._finish_recording(audio=AUDIO, is_final=False, segment_id=segment, audio_end=len(AUDIO))
    assert app.recorder.end_silence_frames == 2 * app.config["sample_rate"]


def test_claude_keeps_an_interrupted_reserved_message_and_drains_before_next(tmp_path):
    import queue
    from debora_whisper.harness import HarnessSession

    session = HarnessSession.__new__(HarnessSession)
    session.process = MagicMock()
    session.process.poll.return_value = None
    session.error = None
    session._turn_lock = threading.Lock()
    session._send = MagicMock()
    session._events = queue.Queue()
    session.log = MagicMock()
    session.session_file = tmp_path / "session.json"
    session.session_id = "test-session"
    session.key = (str(tmp_path),)
    first = threading.Event()
    first.set()  # interrupted before send starts, still preserve the user
    session._events.put({"type": "result"})
    session.send("sobre o projeto...", MagicMock(), first)
    assert session._events.empty()
    session._events.put({"type": "result"})
    session.send("quero mudar a interface", MagicMock(), threading.Event())
    sent = [call.args[0] for call in session._send.call_args_list]
    assert [m["type"] for m in sent] == ["user", "control_request", "user"]
    assert [m["message"]["content"] for m in sent if m["type"] == "user"] == [
        "sobre o projeto...", "quero mudar a interface"]


def test_on_audio_marks_when_her_voice_is_heard(server):
    """The mascot zooms from her first clip to the end of the reply."""
    events = []
    chat, played = _chat(server, FakeLLM(["A capital é Canberra. ", "Fica no sul, perto do mar."]))
    chat.on_audio = lambda active: events.append((active, len(played)))
    chat.respond("qual é a capital da austrália")
    assert events[0] == (True, 0)  # before the first clip plays
    assert events[-1] == (False, 2)  # after the last one
    assert all(a != b for (a, _), (b, _) in zip(events, events[1:]))  # once per change


def _harness_session(tmp_path, saved, events, closed=None):
    """closed: set it to close Claude's output after events (None: at once)."""
    import io
    import json as _json
    from debora_whisper.harness import HarnessSession, harness_key
    config = {**DEFAULT_CONFIG, "harness_cwd": str(tmp_path),
              "harness_memory_file": str(tmp_path / "voice_memory.md")}
    key = harness_key(config)[0]
    session_file = tmp_path / "harness_session.json"
    session_file.write_text(_json.dumps({key: saved}), encoding="utf-8")

    def output():
        for event in events:
            yield _json.dumps(event) + "\n"
        if closed is not None:
            closed.wait(5)

    process = MagicMock(pid=1, stdin=io.StringIO(), stdout=output(), stderr=iter([]))
    process.poll.return_value = None
    commands = []
    session = HarnessSession(
        config, log=lambda m: None,
        command_factory=lambda config, sid, resume: commands.append((sid, resume)) or ["claude"],
        process_factory=lambda *a, **kw: process, session_file=session_file)
    return session, commands, lambda: _json.loads(
        session_file.read_text(encoding="utf-8")).get(key)


@pytest.mark.parametrize("boundary", ["tool_use", "message_start", "text", "assistant"])
def test_claude_text_blocks_are_separated_in_the_reply(server, tmp_path, boundary):
    def stream(event):
        return {"type": "stream_event", "event": event}

    def delta(text):
        return stream({"type": "content_block_delta", "delta": {
            "type": "text_delta", "text": text}})

    if boundary == "assistant":
        between = {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Faço uma cópia de segurança."},
            {"type": "tool_use", "id": "tool-1", "name": "Bash"}]}}
    elif boundary == "message_start":
        between = stream({"type": "message_start"})
    else:
        between = stream({"type": "content_block_start", "content_block": {
            "type": boundary, "id": "tool-1", "name": "Bash"}})
    events = [delta("Faço uma cópia de segu"), delta("rança."), between,
              delta(""), delta("Pron"), delta("to."), {"type": "result"}]
    closed = threading.Event()
    session, _, _ = _harness_session(tmp_path, None, events, closed)
    chunks, shown = [], []

    def llm(messages, on_text, stop):
        def emit(chunk):
            chunks.append(chunk)
            on_text(chunk)
        session.send(messages[-1]["content"], emit, stop)

    chat, _ = _chat(server, llm, voice_chat_backend="claude")
    try:
        reply = chat.respond("Faça a cópia.", on_reply=lambda text, seconds=None: shown.append(text))
    finally:
        closed.set()
    expected = "Faço uma cópia de segurança. Pronto."
    assert chunks == ["Faço uma cópia de segu", "rança.", " Pron", "to."]
    assert reply == shown[-1] == expected
    assert " ".join(_tts_texts(server)) == expected


@pytest.mark.parametrize("before, after", [
    ("segurança. ", "Pronto."), ("segurança.", " Pronto."),
    ("segurança.\n", "Pronto."), ("segurança.", "\nPronto."),
    ("Valor confirmado", ", correto."), ("Concluído", "."),
    *[("Concluído", punctuation) for punctuation in ";:!?)]}…”’»›"],
])
@pytest.mark.parametrize("boundary", ["message_start", "tool_use"])
def test_claude_block_separator_preserves_whitespace_and_punctuation(
        tmp_path, before, after, boundary):
    assert _blocks_across(tmp_path, before, after, boundary) == [before, after]


@pytest.mark.parametrize("quote", ['"Sim."', "'Sim.'"])
@pytest.mark.parametrize("boundary", ["message_start", "tool_use"])
def test_claude_block_separator_keeps_the_space_before_an_opening_quote(
        tmp_path, quote, boundary):
    assert _blocks_across(tmp_path, "Ele respondeu:", quote, boundary) == [
        "Ele respondeu:", " " + quote]


def _blocks_across(tmp_path, before, after, boundary):
    between = ({"type": "message_start"} if boundary == "message_start" else
               {"type": "content_block_start", "content_block": {
                   "type": "tool_use", "id": "tool-1", "name": "Bash"}})
    events = [
        {"type": "stream_event", "event": {"delta": {"type": "text_delta", "text": before}}},
        {"type": "stream_event", "event": between},
        {"type": "stream_event", "event": {"delta": {"type": "text_delta", "text": after}}},
        {"type": "result"},
    ]
    closed = threading.Event()
    session, _, _ = _harness_session(tmp_path, None, events, closed)
    chunks = []
    try:
        session.send("oi", chunks.append, threading.Event())
    finally:
        closed.set()
    return chunks


def test_claude_session_that_does_not_resume_is_forgotten(tmp_path):
    saved = "6f1c2d4e-0000-4000-8000-000000000001"
    session, commands, stored = _harness_session(tmp_path, saved, [])  # exits at once
    assert commands == [(saved, True)]
    with pytest.raises(RuntimeError):
        session.send("oi", MagicMock(), threading.Event())
    assert stored() is None


def test_claude_session_that_resumed_is_kept_after_a_later_failure(tmp_path):
    saved = "6f1c2d4e-0000-4000-8000-000000000002"
    closed = threading.Event()
    session, _, stored = _harness_session(
        tmp_path, saved, [{"type": "result", "session_id": saved}], closed)
    session.send("oi", MagicMock(), threading.Event())
    closed.set()
    with pytest.raises(RuntimeError):
        session.send("de novo", MagicMock(), threading.Event())
    assert stored() == saved


def test_invalid_saved_claude_session_is_never_passed_as_a_flag(tmp_path):
    _, commands, _ = _harness_session(tmp_path, "--dangerously-skip-permissions", [])
    sid, resume = commands[0]
    assert resume is False and not sid.startswith("-")


@pytest.fixture
def claude_runtime(tmp_path, monkeypatch):
    """Exercise the actual start/stop lifecycle without launching Claude."""
    from functools import partial
    from debora_whisper import harness

    monkeypatch.setattr(harness.paths, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(harness, "_run_sessions", {})
    monkeypatch.setattr(harness, "_harness", None)
    monkeypatch.setattr(harness.HarnessSession, "_read", lambda self: None)
    monkeypatch.setattr(harness.HarnessSession, "_read_stderr", lambda self: None)
    processes = MagicMock(side_effect=lambda *a, **kw: MagicMock(pid=1, poll=lambda: None))
    monkeypatch.setattr(harness, "HarnessSession",
                        partial(harness.HarnessSession, process_factory=processes))
    config = {**DEFAULT_CONFIG, "harness_cwd": str(tmp_path)}
    yield harness, config, processes
    harness.stop_harness()


@pytest.mark.parametrize("fresh", [False, True])
def test_claude_first_start_chooses_saved_or_fresh_session(claude_runtime, fresh):
    import uuid
    harness, config, processes = claude_runtime
    saved = str(uuid.uuid4())
    store = harness.paths.CONFIG_DIR / "harness_session.json"
    # A previous app run: write disk only, without populating run memory.
    store.write_text(json.dumps({harness.harness_key(config)[0]: saved}), encoding="utf-8")
    config["harness_new_session_on_start"] = fresh
    session = harness.start_harness(config)
    command = processes.call_args.args[0]
    assert str(uuid.UUID(session.session_id)) == session.session_id
    assert (session.session_id != saved) is fresh
    assert command[command.index("--session-id" if fresh else "--resume") + 1] == session.session_id
    assert ("--resume" in command) is not fresh
    assert command[command.index("--name") + 1] == "Débora Whisper"
    assert json.loads(store.read_text(encoding="utf-8"))[session.key[0]] == saved


def test_claude_restarts_keep_this_runs_session_before_and_after_result(claude_runtime):
    harness, config, processes = claude_runtime
    config["harness_new_session_on_start"] = True
    first = harness.start_harness(config)
    assert harness.start_harness(config) is first
    assert processes.call_count == 1
    changed = {**config, "harness_model": "sonnet", "harness_session_name": "Debora test"}
    second = harness.start_harness(changed)
    assert second is not first
    assert second.session_id == first.session_id
    command = processes.call_args.args[0]
    assert command[command.index("--session-id") + 1] == first.session_id
    assert command[command.index("--name") + 1] == "Debora test"
    assert first._prompt_file is None
    second._events.put({"type": "result", "session_id": second.session_id})
    second.send("oi", MagicMock(), threading.Event())
    stored = json.loads(second.session_file.read_text(encoding="utf-8"))
    assert stored[second.key[0]] == second.session_id
    harness.stop_harness()
    assert harness.start_harness(changed).session_id == second.session_id
    assert "--resume" in processes.call_args.args[0]
    # A new Débora process has no run memory, even though disk has a UUID.
    harness.stop_harness()
    harness._run_sessions.clear()
    assert harness.start_harness(changed).session_id != second.session_id
    assert "--session-id" in processes.call_args.args[0]


def test_claude_run_sessions_are_per_folder_and_resettable(claude_runtime, tmp_path):
    harness, config, processes = claude_runtime
    config["harness_new_session_on_start"] = True
    first = harness.start_harness(config)
    other = tmp_path / "other"
    other.mkdir()
    second = harness.start_harness({**config, "harness_cwd": str(other)})
    assert first.session_id != second.session_id
    returned = harness.start_harness(config)
    assert returned.session_id == first.session_id
    returned._events.put({"type": "result"})
    returned.send("oi", MagicMock(), threading.Event())
    harness.reset_harness(config)
    assert first.key[0] not in json.loads(first.session_file.read_text(encoding="utf-8"))
    assert harness.start_harness(config).session_id != first.session_id
    assert "--session-id" in processes.call_args.args[0]
    assert harness.start_harness({**config, "harness_cwd": str(other)}).session_id == second.session_id


def test_claude_enabling_fresh_starts_preserves_an_active_session(claude_runtime):
    harness, config, processes = claude_runtime
    first = harness.start_harness(config)
    second = harness.start_harness({**config, "harness_new_session_on_start": True})
    assert second.session_id == first.session_id
    assert "--session-id" in processes.call_args.args[0]


@pytest.mark.parametrize("completed", [False, True])
def test_claude_disabling_fresh_starts_preserves_this_runs_session(claude_runtime, completed):
    import uuid
    harness, config, processes = claude_runtime
    saved = str(uuid.uuid4())
    store = harness.paths.CONFIG_DIR / "harness_session.json"
    store.write_text(json.dumps({harness.harness_key(config)[0]: saved}), encoding="utf-8")
    first = harness.start_harness({**config, "harness_new_session_on_start": True})
    assert first.session_id != saved
    if completed:
        first._events.put({"type": "result"})
        first.send("oi", MagicMock(), threading.Event())
    second = harness.start_harness(config)
    assert second.session_id == first.session_id
    assert ("--resume" in processes.call_args.args[0]) is completed
    assert json.loads(store.read_text(encoding="utf-8"))[first.key[0]] == (
        first.session_id if completed else saved)


@pytest.mark.parametrize("fresh", [False, True])
def test_claude_exit_before_initialization_retries_selected_uuid_without_resume(claude_runtime, fresh):
    harness, config, processes = claude_runtime
    config["harness_new_session_on_start"] = fresh
    first = harness.start_harness(config)
    first.process.poll = lambda: 1  # Popen succeeded, but initialization failed.
    second = harness.start_harness(config)
    assert second.session_id == first.session_id
    command = processes.call_args.args[0]
    assert "--resume" not in command
    assert command[command.index("--session-id") + 1] == first.session_id
    second._events.put({"type": "result"})
    second.send("retry this voice turn", MagicMock(), threading.Event())
    assert second._confirmed


@pytest.mark.parametrize("launcher", ["claude.cmd", "claude.bat"])
@pytest.mark.parametrize("char", list('&|<>^%"\r\n'))
def test_claude_batch_launcher_rejects_unsafe_session_name(claude_runtime, monkeypatch, launcher, char):
    harness, config, processes = claude_runtime
    monkeypatch.setattr(harness.shutil, "which", lambda executable: launcher)
    with pytest.raises(ValueError, match="harness_session_name"):
        harness.start_harness({**config, "harness_session_name": f"Debora{char}test"})
    processes.assert_not_called()


def test_claude_failed_launch_does_not_reserve_a_run_session(claude_runtime):
    harness, config, processes = claude_runtime
    config["harness_new_session_on_start"] = True
    factory = processes.side_effect
    processes.side_effect = OSError("cannot launch Claude")
    with pytest.raises(OSError, match="cannot launch Claude"):
        harness.start_harness(config)
    assert harness._run_sessions == {}
    processes.side_effect = factory
    harness.start_harness(config)
    assert "--session-id" in processes.call_args.args[0]


def test_claude_unavailable_run_session_is_forgotten(claude_runtime):
    harness, config, processes = claude_runtime
    config["harness_new_session_on_start"] = True
    first = harness.start_harness(config)
    first._events.put({"type": "result"})
    first.send("oi", MagicMock(), threading.Event())
    harness.stop_harness()
    second = harness.start_harness(config)
    second.error = "Session not found"
    with pytest.raises(RuntimeError, match="Session not found"):
        second.send("oi", MagicMock(), threading.Event())
    assert harness.start_harness(config).session_id != first.session_id
    assert "--session-id" in processes.call_args.args[0]


@pytest.mark.parametrize("text, started, echo", [
    ("sim", 10.5, True),     # heard while she was saying it
    ("sim", 12.5, False),    # her sentence ended: the user answered
    ("quê", 13.0, False),
    ("ela disse sim mesmo", 12.5, True),  # longer text keeps the 2 s room tail
])
def test_short_answers_after_her_sentence_are_not_echo(text, started, echo, monkeypatch):
    chat = vc.VoiceChat({"language": "pt"})
    chat._spoken = [(10.0, 12.0, "Sim, ela disse que quê mesmo")]
    assert chat.is_echo(text, started, started + 0.4) is echo


def test_her_text_enters_the_balloon_with_her_voice(server):
    shown = []
    chat, played = _chat(server, FakeLLM(["A capital é Canberra. ", "Fica no sul, perto do mar."]))
    seconds = []
    chat.respond("qual é a capital da austrália",
                 on_reply=lambda text, audio: (shown.append((text, len(played))),
                                               seconds.append(audio)))
    # Each sentence appears as its clip starts, never while it is synthesized.
    assert shown == [("A capital é Canberra.", 0),
                     ("A capital é Canberra. Fica no sul, perto do mar.", 1)]
    # With how long its trimmed clip plays, for the balloon's pace.
    assert seconds == [samples / rate for samples, rate in played] and all(seconds)


def test_her_text_still_shows_without_a_voice(server, monkeypatch):
    monkeypatch.setattr(vc, "wait_tts_server", lambda config, *a, **kw: False)
    shown = []
    chat, played = _chat(server, FakeLLM(["A capital é Canberra."]))
    chat.respond("capital", on_reply=shown.append)
    assert shown == ["A capital é Canberra."] and played == []


@pytest.fixture
def playing_reply(monkeypatch):
    """Hold the first clip until playback is paused or interrupted."""
    monkeypatch.setattr(vc, "ensure_tts_server", lambda *a: None)
    monkeypatch.setattr(vc, "wait_tts_server", lambda *a: True)
    instances = []

    def start(**config):
        app = _app(voice_chat=True, **config)
        started, cut, rendered = threading.Event(), threading.Event(), threading.Event()
        played, shown, talking, errors, logs = [], [], [], [], []
        sentences = ["A capital é Canberra.", "Fica no sul, perto do mar."]

        def synth(text, config):
            index = sentences.index(text)
            if index == 1:
                rendered.set()
            return np.full(2400, 0.1 * (index + 1), np.float32), 24000

        def play(samples, rate, stop):
            played.append(samples.copy())
            if len(played) == 1:
                started.set()
                if not stop.wait(5):
                    raise AssertionError("first clip was never stopped")
                cut.set()

        monkeypatch.setattr(vc, "synthesize", synth)
        chat = app.voice_chat = vc.VoiceChat(app.config, log=logs.append, play=play,
                                             llm=FakeLLM([sentences[0] + " ", sentences[1]]))
        chat.on_audio = talking.append
        app._voice_active = {"text": "capital", "voice_chat_status": "sent"}
        app._voice_worker = object()  # inspect queued turns without starting another worker
        app._voice_stop = threading.Event()

        def respond():
            try:
                chat.respond("capital", on_reply=lambda text, seconds: shown.append((text, seconds)),
                             stop=app._voice_stop)
            except Exception as e:
                errors.append(e)

        worker = threading.Thread(target=respond, daemon=True)
        instances.append((chat, worker, errors))
        worker.start()
        assert started.wait(3)
        assert rendered.wait(3)
        return app, worker, cut, played, shown, talking, logs

    yield start
    for chat, worker, errors in instances:
        chat.interrupt()
        worker.join(3)
        assert not worker.is_alive()
        assert errors == []


@pytest.mark.parametrize("text", ["", "hum", "ah", "é", "uh", "hmm", "...", "x",
                                  "A capital é Canberra."])
def test_invalid_final_replays_the_cut_sentence(playing_reply, text):
    app, worker, cut, played, shown, talking, logs = playing_reply(voice_chat_pause_timeout=10)
    app._voice_activity(True)
    assert cut.wait(2)
    assert talking == [True, False]
    assert not app._voice_stop.is_set()
    assert not app.voice_chat._tts_cancellation.cancelled
    app._voice_chat_turn(text, AUDIO, True)
    worker.join(3)
    assert not worker.is_alive()
    assert len(played) == 3
    np.testing.assert_array_equal(played[0], played[1])
    assert not np.array_equal(played[1], played[2])
    assert shown[0] == shown[1]  # same paced slide, without duplicated sentence text
    assert shown[2][0] == "A capital é Canberra. Fica no sul, perto do mar."
    assert talking == [True, False, True, False]
    assert app._voice_pending == []
    assert any("paused reply" in line for line in logs)
    assert any("resumed reply" in line for line in logs)


def test_valid_final_interrupts_a_paused_reply(playing_reply):
    app, worker, cut, played, shown, talking, logs = playing_reply()
    app._voice_activity(True)
    assert cut.wait(2)
    app._voice_chat_turn("quero mudar de assunto", AUDIO, True)
    worker.join(3)
    assert not worker.is_alive()
    assert len(played) == 1
    assert app._voice_stop.is_set()
    assert app.voice_chat._tts_cancellation.cancelled
    assert app._voice_active["voice_chat_status"] == "interrupted"
    assert app._voice_pending[0][0] == "quero mudar de assunto"
    assert app._voice_pending[0][1]["voice_chat_status"] == "barge-in"


def test_pause_timeout_replays_without_a_final(playing_reply):
    app, worker, cut, played, shown, talking, logs = playing_reply(voice_chat_pause_timeout=0.05)
    app._voice_activity(True)
    assert cut.wait(3)
    time.sleep(0.2)  # still talking: no timeout yet
    assert app.voice_chat._paused and len(played) == 1
    app._voice_activity(False)
    worker.join(3)
    assert not worker.is_alive()
    assert len(played) == 3
    np.testing.assert_array_equal(played[0], played[1])
    assert any("resumed reply (timeout)" in line for line in logs)


@pytest.mark.parametrize("config", [{"voice_chat_pause_on_speech": False},
                                    {"voice_chat_barge_in": False}])
def test_disabled_pause_keeps_playing_until_final(playing_reply, config):
    app, worker, cut, played, shown, talking, logs = playing_reply(**config)
    app._voice_activity(True)
    assert not app.voice_chat._paused
    assert not app.voice_chat._clip_stop.is_set()
    assert talking == [True]
    app._voice_chat_turn("hum", AUDIO, True)
    assert len(app._voice_pending) == 1  # legacy filler behavior
    assert app._voice_stop.is_set() is config.get("voice_chat_barge_in", True)


def test_draft_does_not_resolve_pause(playing_reply):
    app, worker, cut, played, shown, talking, logs = playing_reply()
    app._voice_activity(True)
    app._voice_chat_turn("mude de assunto", AUDIO, False)
    assert app.voice_chat._paused
    assert not app._voice_stop.is_set()


def test_empty_asr_result_resumes_playback(playing_reply):
    app, worker, cut, played, shown, talking, logs = playing_reply()
    app._voice_activity(True)
    app.whisper.transcribe.return_value = ""
    app._finish_recording(audio=AUDIO, is_final=True)
    worker.join(3)
    assert not worker.is_alive()
    assert len(played) == 3


@pytest.mark.parametrize("text", ["sim", "não", "oi", "ok", "quê?", "ah, mude de assunto", "2"])
def test_short_meaningful_answers_are_not_fillers(text):
    assert not vc.is_voice_filler(text)


@pytest.mark.parametrize("key, value", [
    ("voice_chat_pause_on_speech", "true"), ("voice_chat_pause_on_speech", 1),
    ("voice_chat_pause_on_speech", None), ("voice_chat_pause_timeout", True),
    ("voice_chat_pause_timeout", 0), ("voice_chat_pause_timeout", -1),
    ("voice_chat_pause_timeout", "3"), ("voice_chat_pause_timeout", None),
    ("voice_chat_pause_timeout", float("inf")), ("voice_chat_pause_timeout", float("nan")),
])
def test_invalid_pause_config_is_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        de.validate_config({**DEFAULT_CONFIG, key: value})


def test_recorder_speech_start_pauses_in_the_first_vad_block(monkeypatch):
    app = DictationApp({**DEFAULT_CONFIG, "continuous_listening": True, "voice_chat": True})
    recorder = app.recorder
    calls = []

    class Feed:
        def wait(self, timeout):
            if recorder._write_pos:
                recorder._stop_vad = True
                return
            recorder._buffer[:512] = 0.1
            recorder._write_pos = 512

    recorder._data_cv = Feed()
    monkeypatch.setattr(app.voice_chat, "pause", lambda: calls.append(recorder._write_pos))
    recorder._vad_loop()
    assert calls == [512]


@pytest.fixture
def cancellable_tts_server(monkeypatch, request):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from debora_whisper import tts_server

    class Module:
        def __init__(self):
            self.hooks = []

        def register_forward_pre_hook(self, hook):
            self.hooks.append(hook)
            return SimpleNamespace(remove=lambda: self.hooks.remove(hook))

        def step(self):
            for hook in self.hooks:
                hook(self, ())

    token, flow = Module(), Module()
    started, release, cancelled = threading.Event(), threading.Event(), threading.Event()
    steps = []
    phase = getattr(request, "param", "tokens")
    wav = MagicMock()
    wav.squeeze.return_value.float.return_value.cpu.return_value.numpy.return_value = np.ones(2400)

    def generate(text, language_id):
        for index in range(5):
            (token if phase == "tokens" else flow).step()
            steps.append(index)
            if not started.is_set():
                started.set()
                if not release.wait(5):
                    raise RuntimeError("test did not release synthesis")
        return wav

    model = SimpleNamespace(
        sr=24000, conds=object(), generate=generate,
        t3=SimpleNamespace(tfmr=SimpleNamespace(layers=[token])),
        s3gen=SimpleNamespace(flow=SimpleNamespace(decoder=SimpleNamespace(
            estimator=SimpleNamespace(time_mlp=flow)))))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setattr(tts_server, "drop_attention_spies", lambda model: None)
    monkeypatch.setattr(tts_server, "log", lambda line: cancelled.set() if "Cancelled" in line else None)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), tts_server.make_handler(model, "pt"))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    config = {**DEFAULT_CONFIG, "tts_url": f"http://127.0.0.1:{srv.server_port}"}
    # Omit voice so the stub keeps its startup conditionals.
    monkeypatch.setattr(vc, "resolve_voice", lambda voice: Path("missing-reference.wav"))
    try:
        yield config, started, release, cancelled, steps, (token, flow)
    finally:
        release.set()
        srv.shutdown()
        srv.server_close()
        thread.join(3)


@pytest.mark.parametrize("cancellable_tts_server", ["tokens", "flow"], indirect=True)
def test_interrupt_disconnects_inflight_tts_and_stops_server_work(cancellable_tts_server):
    config, started, release, cancelled, steps, modules = cancellable_tts_server
    chat = vc.VoiceChat(config, log=lambda line: None)
    errors = []

    def synthesize():
        try:
            vc.synthesize("Primeira frase.", {**config, "_tts_cancellation": chat._tts_cancellation})
        except Exception as e:
            errors.append(e)

    worker = threading.Thread(target=synthesize, daemon=True)
    worker.start()
    try:
        assert started.wait(3)
        chat.interrupt()
        release.set()
        assert cancelled.wait(3)
        worker.join(3)
        assert not worker.is_alive() and errors
        assert steps == [0]  # no subsequent token/flow step ran
        assert all(not module.hooks for module in modules)
        samples, rate = vc.synthesize("Segunda frase.", config)
        assert len(samples) == 2400 and rate == 24000
        assert steps == [0, 0, 1, 2, 3, 4]
        assert all(not module.hooks for module in modules)
    finally:
        chat.interrupt()
        release.set()
        worker.join(3)


def test_pause_keeps_inflight_server_synthesis_running(cancellable_tts_server):
    config, started, release, cancelled, steps, modules = cancellable_tts_server
    chat = vc.VoiceChat(config, log=lambda line: None)
    result, errors = [], []
    chat._clip_stop = vc.PlaybackStop(chat._interrupt)
    chat._show_audio(True)
    chat.speaking = True

    def synthesize():
        try:
            result.append(vc.synthesize("Pode continuar.", {
                **config, "_tts_cancellation": chat._tts_cancellation}))
        except Exception as e:
            errors.append(e)

    worker = threading.Thread(target=synthesize, daemon=True)
    worker.start()
    try:
        assert started.wait(3)
        assert chat.pause()
        assert not chat._tts_cancellation.cancelled
        release.set()
        worker.join(3)
        assert not worker.is_alive() and not errors
        assert len(result) == 1 and steps == [0, 1, 2, 3, 4]
        assert not cancelled.is_set()
    finally:
        release.set()
        chat.interrupt()
        worker.join(3)


def test_paused_stream_is_aborted_and_replayed_on_a_new_stream(server, stream, monkeypatch):
    original = FakeStream.write
    chat = vc.VoiceChat(_cfg(server), log=lambda line: None,
                        llm=FakeLLM(["A capital é Canberra. Fica no sul do país."]))
    shown = []

    def write(output, data):
        original(output, data)
        if len(stream) == 1 and len(output.written) == 1:
            assert chat.pause()
            assert chat.resume("invalid transcription")  # may arrive before play() returns

    monkeypatch.setattr(FakeStream, "write", write)
    chat.respond("capital", on_reply=lambda text, seconds: shown.append((text, seconds)))
    assert len(stream) == 2
    assert stream[0].ended == "dropped" and stream[1].ended == "drained"
    assert shown[0] == shown[1]
    assert len(shown) == 3
    assert sum(stream[1].written) == 4800 + int(24000 * vc.SENTENCE_GAP_SECONDS)


def test_pause_can_cut_a_short_clips_buffered_tail(server, stream, monkeypatch):
    waiting = threading.Event()
    original = vc.PlaybackStop.wait

    def wait(stop, timeout=None):
        waiting.set()
        return original(stop, timeout)

    monkeypatch.setattr(vc.PlaybackStop, "wait", wait)
    original_init = FakeStream.__init__

    def init(output, *args, **kwargs):
        original_init(output, *args, **kwargs)
        output.latency = 1.0 if len(stream) == 1 else 0.0

    monkeypatch.setattr(FakeStream, "__init__", init)
    chat = vc.VoiceChat(_cfg(server), log=lambda line: None, llm=FakeLLM(["A capital é Canberra."]))
    worker = threading.Thread(target=chat.respond, args=("capital",), daemon=True)
    worker.start()
    try:
        assert waiting.wait(3)
        assert chat.pause()
        chat.resume("invalid transcription")
        worker.join(3)
        assert not worker.is_alive()
        assert len(stream) == 2 and stream[0].ended == "dropped"
        assert stream[0].written == stream[1].written
    finally:
        chat.interrupt()
        worker.join(3)


def test_tts_server_synthesizes_without_hooks_on_another_model_layout():
    import socket
    from types import SimpleNamespace
    from debora_whisper import tts_server

    client, server = socket.socketpair()
    try:
        with tts_server.cancellable_synthesis(SimpleNamespace(), server):
            pass
    finally:
        client.close()
        server.close()


def test_pause_between_sentences_holds_the_next_one():
    chat = vc.VoiceChat({**DEFAULT_CONFIG, "voice_chat": True}, log=lambda *a: None)
    chat.speaking = True
    assert chat._clip_stop is None  # no clip playing: the next one is still rendering
    assert chat.pause()
    assert chat._paused and chat._pause_deadline is None
    chat.speech_ended()
    assert chat._pause_deadline is not None


def test_breath_while_paused_does_not_start_the_timeout(playing_reply):
    app, worker, cut, played, shown, talking, logs = playing_reply(voice_chat_pause_timeout=0.1)
    app._voice_activity(True)
    assert cut.wait(3)
    app._voice_activity(False)  # a breath
    app._voice_activity(True)   # still talking
    time.sleep(0.3)
    assert app.voice_chat._paused and len(played) == 1
    app._voice_activity(False)
    worker.join(3)
    assert not worker.is_alive() and len(played) == 3


def test_pause_has_a_ceiling_without_end_of_speech(playing_reply, monkeypatch):
    monkeypatch.setattr(vc, "PAUSE_MAX_SECONDS", 0.1)
    app, worker, cut, played, shown, talking, logs = playing_reply()
    app._voice_activity(True)
    worker.join(3)
    assert not worker.is_alive() and len(played) == 3


@pytest.mark.parametrize("text, echo", [
    ("Isso do tempo.", False),           # the user's own word in a short reply
    ("Previsão do tempo.", True),        # a piece of her sentence
    ("Você quer a previsão do tempo?", True),
])
def test_short_reply_sharing_her_words_is_not_echo(text, echo):
    chat = vc.VoiceChat({"language": "pt"})
    chat._spoken = [(10.0, 12.6, "Você quer a previsão do tempo?")]
    assert chat.is_echo(text, 13.0, 14.0) is echo


@pytest.mark.parametrize("name, expected", [
    ("Headphones (LE-HD 450BT)", True),
    ("Headset (LE-HD 450BT Hands-Free)", True),
    ("Speakers (Realtek(R) Audio)", False),
    ("Headset (LE-HD 450BT Hands-Free", True),  # MME cuts at 31 characters
    ("Headphones (Other)", None),
    (None, None),
])
def test_windows_output_form_factor(name, expected):
    from debora_whisper.platforms.windows import is_headphones
    endpoints = [("Headphones (LE-HD 450BT)", 3), ("Headset (LE-HD 450BT Hands-Free)", 5),
                 ("Speakers (Realtek(R) Audio)", 1)]
    assert is_headphones(name, endpoints) is expected


def test_generic_platform_does_not_know_the_output():
    from debora_whisper.platforms.base import Platform
    assert Platform().output_is_headphones("Headphones") is None


@pytest.mark.parametrize("setting, headphones, on", [
    ("auto", True, False), (True, True, False), ("auto", False, True),
    ("auto", None, True), ("on", True, True), ("off", False, False), (False, None, False),
])
def test_echo_filter_follows_the_output_device(monkeypatch, setting, headphones, on):
    from debora_whisper import platforms
    from debora_whisper.platforms.base import Platform

    class Fake(Platform):
        def default_output_name(self):
            return "Fone"

        def output_is_headphones(self, name=None):
            return headphones

    monkeypatch.setattr(platforms, "_current", Fake())
    app = _app(voice_chat=True, voice_chat_echo_filter=setting)
    assert app._echo_filter_on() is on


@pytest.mark.parametrize("value", ["sometimes", 1, 0])
def test_echo_filter_setting_is_validated(value):
    from debora_whisper.dictation_engine import validate_config
    with pytest.raises(ValueError, match="voice_chat_echo_filter"):
        validate_config({**DEFAULT_CONFIG, "voice_chat_echo_filter": value})
