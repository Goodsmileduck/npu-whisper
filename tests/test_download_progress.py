"""Download telemetry without fetching weights or importing model backends."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import http.client
import io
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from debora_whisper.download_progress import DownloadProgress, format_download, track_hf_downloads
from debora_whisper import tts_server, voice_chat as vc


def test_aggregate_concurrent_files_and_resume_even_with_disabled_bars():
    from huggingface_hub.utils import tqdm
    tracker = DownloadProgress()
    callback = Mock()
    with track_hf_downloads(tracker, callback):
        # The snapshot's file-count bar must not be counted as bytes.
        with tqdm(total=2, disable=True) as files:
            with tqdm(total=400_000_000, initial=100_000_000, unit="B", disable=True) as first:
                with tqdm(total=340_000_000, unit="B", disable=True) as second:
                    with ThreadPoolExecutor(2) as pool:
                        list(pool.map(lambda bar: bar.update(105_000_000), [first, second]))
                    files.update(1)
                    assert tracker.snapshot() == {"done": 310_000_000, "total": 740_000_000}
                    assert format_download("STT", tracker.snapshot()) == " DL STT 42% (310/740 MB)"
                    assert callback.call_args.args == (310_000_000, 740_000_000)
                # A completed/closed file's bytes remain in the aggregate.
                assert tracker.snapshot()["total"] == 740_000_000
            assert tracker.snapshot() is None
    assert tracker.snapshot() is None


@pytest.mark.parametrize("disable", [True, False])
def test_hub_snapshot_aggregate_bars_with_growing_totals(disable):
    from huggingface_hub.utils import tqdm

    tracker, callback = DownloadProgress(), Mock()
    with track_hf_downloads(tracker, callback):
        # Hub 2.2.0 snapshot_download creates these two zero-total byte bars.
        # Its per-file _AggregatedTqdm adjusts both totals, refreshes the parent,
        # and forwards initial/resumed bytes and updates to reconstruction.
        with (tqdm(total=0, unit="B", disable=disable, file=io.StringIO(),
                   name="huggingface_hub.snapshot_download.transfer") as transfer,
              tqdm(total=0, unit="B", disable=disable, file=io.StringIO(),
                   name="huggingface_hub.snapshot_download") as reconstruction,
              tqdm(total=2, disable=True) as files):
            assert tracker.snapshot() is None

            def discover_file(total, initial=0):
                reconstruction.total = (reconstruction.total or 0) + total
                transfer.total = (transfer.total or 0) + total
                reconstruction.refresh()
                if initial:
                    reconstruction.update(initial)

            discover_file(100_000_000, initial=20_000_000)
            reconstruction.update(22_000_000)
            transfer.update(22_000_000)
            files.update(1)
            assert tracker.snapshot() == {"done": 42_000_000, "total": 100_000_000}
            assert callback.call_args.args == (42_000_000, 100_000_000)
            assert format_download("STT", tracker.snapshot()) == " DL STT 42% (42/100 MB)"

            discover_file(50_000_000)
            reconstruction.update(8_000_000)
            assert tracker.snapshot() == {"done": 50_000_000, "total": 150_000_000}
            assert callback.call_args.args == (50_000_000, 150_000_000)
            reconstruction.update(100_000_000)
            assert callback.call_args.args == (150_000_000, 150_000_000)
            assert tracker.snapshot() is None
    assert tracker.snapshot() is None


@pytest.mark.parametrize("shared_tracker", [True, False])
def test_overlapping_tracking_contexts_wait_and_restore_hooks(shared_tracker):
    from huggingface_hub import file_download as hf
    from huggingface_hub.utils import tqdm

    def hooks():
        return tqdm.__init__, tqdm.update, tqdm.close, hf.http_get

    original = hooks()
    first = DownloadProgress()
    second = first if shared_tracker else DownloadProgress()
    first_callback, second_callback = Mock(), Mock()
    attempted, entered, release = threading.Event(), threading.Event(), threading.Event()

    def second_download():
        attempted.set()
        with track_hf_downloads(second, second_callback):
            with tqdm(total=200, unit="B", disable=True) as bar:
                bar.update(50)
                entered.set()
                assert release.wait(5)
                bar.update(25)
                assert second.snapshot() == {"done": 75, "total": 200}
                assert second_callback.call_args.args == (75, 200)

    # The first context exits while the second request is still outstanding.
    # Even exceptional exits must restore hooks before the waiter installs its own.
    with ThreadPoolExecutor(1) as pool:
        future = None
        try:
            with pytest.raises(RuntimeError, match="first download interrupted"):
                with track_hf_downloads(first, first_callback):
                    with tqdm(total=100, unit="B", disable=True) as bar:
                        bar.update(17)
                        first_hooks = hooks()
                        future = pool.submit(second_download)
                        assert attempted.wait(5)
                        assert not entered.wait(0.2)
                        assert hooks() == first_hooks
                        assert first.snapshot() == {"done": 17, "total": 100}
                        second_callback.assert_not_called()
                        raise RuntimeError("first download interrupted")
            first_callback.reset_mock()
            assert entered.wait(5)
            assert second.snapshot() == {"done": 50, "total": 200}
            assert second_callback.call_args.args == (50, 200)
        finally:
            release.set()
            if future is not None:
                future.result(timeout=5)

    assert hooks() == original
    assert first.snapshot() is None
    assert second.snapshot() is None
    first_callback.assert_not_called()
    second_callback.reset_mock()
    with tqdm(total=100, unit="B", disable=True) as bar:
        bar.update(10)
    first_callback.assert_not_called()
    second_callback.assert_not_called()


def test_progress_cleared_and_hf_hooks_restored_after_failure():
    from huggingface_hub import file_download as hf
    from huggingface_hub.utils import tqdm
    tracker = DownloadProgress()
    original = (tqdm.__init__, tqdm.update, tqdm.close)
    original_http_get = hf.http_get
    with pytest.raises(RuntimeError):
        with track_hf_downloads(tracker):
            bar = tqdm(total=100, unit="B", disable=True)
            bar.update(17)
            assert format_download("TTS", tracker.snapshot()) == " DL TTS 17% (0/0 MB)"
            raise RuntimeError("download interrupted")
    assert tracker.snapshot() is None
    assert (tqdm.__init__, tqdm.update, tqdm.close) == original
    assert hf.http_get is original_http_get
    bar.close()


@pytest.fixture(params=["requests", "httpx"], ids=["legacy-requests", "current-httpx"])
def hub_http(monkeypatch, request):
    """Exercise the installed Hub's real retry loop with its native responses."""
    from huggingface_hub import file_download as hf
    legacy = hasattr(hf, "_request_wrapper")
    if (request.param == "requests") != legacy:
        pytest.skip("Covered by the other Hub version in the CI matrix")

    transport = Mock()
    if legacy:
        import requests as http
        monkeypatch.setattr(hf, "_request_wrapper", transport)
        monkeypatch.setattr(hf, "reset_sessions", Mock())
        monkeypatch.setattr(hf.constants, "HF_HUB_ENABLE_HF_TRANSFER", False)
        connection_error = http.ConnectionError
    else:
        # Hub 1.x uses httpx; Hub 2.x uses httpx2.
        http = getattr(hf, "httpx2", None) or hf.httpx
        monkeypatch.setattr(hf, "http_stream_backoff", transport)
        connection_error = http.ConnectError
    monkeypatch.setattr(hf.time, "sleep", lambda seconds: None)

    def response(size, *, resumed=False, interrupted=False):
        status = 206 if resumed else 200
        headers = {"Content-Length": str(size)}
        if resumed:
            headers["Content-Range"] = "bytes 40-99/100"
        if legacy:
            result = http.Response()
            result.status_code = status
            result.headers.update(headers)
        else:
            result = http.Response(status, headers=headers,
                                   request=http.Request("GET", "https://example.invalid/weights"))

        def chunks(chunk_size):
            yield b"a" * (40 if interrupted else size)
            if interrupted:
                raise connection_error("interrupted after 40 bytes")

        setattr(result, "iter_content" if legacy else "iter_bytes", chunks)
        return result if legacy else nullcontext(result)

    return transport, response


@pytest.mark.parametrize("resume", [True, False])
def test_http_retry_replaces_attempt_even_when_server_ignores_range(hub_http, resume):
    from huggingface_hub import file_download as hf
    from huggingface_hub.utils import tqdm

    tracker, callback = DownloadProgress(), Mock()
    original_http_get = hf.http_get
    request, response = hub_http
    request.side_effect = [response(100, interrupted=True),
                           response(60 if resume else 100, resumed=resume)]
    with track_hf_downloads(tracker, callback):
        with tqdm(total=20, initial=10, unit="B", disable=True) as other:
            callback.reset_mock()
            target = io.BytesIO()
            hf.http_get("https://example.invalid/weights", target, expected_size=100,
                        displayed_filename="same-name.safetensors")
            assert len(target.getvalue()) == 100
            # The interrupted bar closes after the resumed download returns.
            assert tracker.snapshot() == {"done": 110, "total": 120}
            assert callback.call_args.args == (110, 120)
            assert all(call.args[1] == 120 for call in callback.call_args_list)
            assert request.call_args.kwargs["headers"]["Range"] == "bytes=40-"
            assert request.call_count == 2
            # A separate file with the same display name must still be counted.
            request.side_effect = None
            request.return_value = response(100)
            hf.http_get("https://example.invalid/other", io.BytesIO(),
                        displayed_filename="same-name.safetensors")
            assert tracker.snapshot()["total"] == 220
            other.update(10)
            assert tracker.snapshot() is None
    assert hf.http_get is original_http_get


def test_stt_setup_preserves_callback_and_clears_progress(monkeypatch, tmp_path):
    import huggingface_hub
    from huggingface_hub.utils import tqdm
    from debora_whisper import dictation_engine as de
    monkeypatch.setattr(de, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(de, "model_files_complete", Mock(side_effect=[False, True]))
    callback = Mock()

    def download(*args, **kwargs):
        with tqdm(total=0, unit="B", disable=True) as bar:
            bar.total += 100_000_000
            bar.refresh()
            bar.update(42_000_000)
            assert format_download("STT", de.STT_DOWNLOAD.snapshot()) == " DL STT 42% (42/100 MB)"
            bar.update(58_000_000)
            assert de.STT_DOWNLOAD.snapshot() is None

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert de.setup_model({**de.DEFAULT_CONFIG, "device": "CPU"}, callback).parent == tmp_path
    assert callback.call_args.args == (100_000_000, 100_000_000)
    assert de.STT_DOWNLOAD.snapshot() is None


@pytest.mark.parametrize("progress", [None, {}, {"done": 1, "total": 0},
                                      {"done": 10, "total": 10}, {"done": "bad", "total": 10},
                                      {"done": 1e307, "total": 1e308},
                                      {"done": 10**400, "total": 10**401},
                                      {"done": 1, "total": float("inf")},
                                      {"done": float("nan"), "total": 10},
                                      {"done": True, "total": 10}])
def test_no_segment_when_idle_or_invalid(progress):
    assert format_download("STT", progress) == ""


@pytest.fixture
def loading_tts(monkeypatch):
    from huggingface_hub.utils import tqdm
    started, release = threading.Event(), threading.Event()
    model = SimpleNamespace(package="missing-test-package", model_id="test", sr=24000,
                            _debora_device="CPU")
    failure = []

    def loader():
        with tqdm(total=100_000_000, unit="B", disable=True) as bar:
            bar.update(17_000_000)
            started.set()
            assert release.wait(10)
        if failure:
            raise RuntimeError(failure[0])
        return model, None

    server = tts_server.loading_server(("127.0.0.1", 0), loader, "pt")
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    assert started.wait(5)
    config = {"tts_url": f"http://127.0.0.1:{server.server_port}"}
    monkeypatch.setattr(vc, "_tts_process", None)
    monkeypatch.setattr(vc, "_tts_health_cache", {})
    try:
        yield config, release, failure
    finally:
        with vc._tts_health_cache_lock:
            workers = [worker for _, worker in vc._tts_health_polls.values()]
        vc.stop_tts_server()
        for poller in workers:
            poller.join(3)
        release.set()
        server.shutdown()
        server.server_close()
        worker.join(2)


def test_loading_health_segment_no_extra_network_and_wait_until_ready(loading_tts, monkeypatch):
    config, release, _ = loading_tts
    assert not vc.tts_server_up(config)
    assert format_download("TTS", vc.tts_download_progress(config)) == " DL TTS 17% (17/100 MB)"
    connection = http.client.HTTPConnection(config["tts_url"].removeprefix("http://"), timeout=1)
    try:
        connection.request("POST", "/tts", b'{}')
        response = connection.getresponse()
        assert response.status == 503
        assert json.loads(response.read()) == {"error": "loading"}
    finally:
        connection.close()
    with monkeypatch.context() as patch:
        patch.setattr(vc.http.client, "HTTPConnection", Mock(side_effect=AssertionError("extra poll")))
        for _ in range(10):
            assert vc.tts_download_progress(config)["done"] == 17_000_000
        assert vc.tts_download_progress({"tts_url": "http://unused"}) is None
    # Existing startup checks must wait through loading even for external servers.
    timer = threading.Timer(0.6, release.set)
    timer.start()
    try:
        assert vc.wait_tts_server(config, timeout=5)
    finally:
        timer.cancel()
    assert vc.tts_download_progress(config) is None


def test_loading_failure_remains_terminal_and_clears_segment(loading_tts):
    config, release, failure = loading_tts
    assert not vc.tts_server_up(config)
    failure.append("missing dependency")
    release.set()
    with pytest.raises(vc.TTSStartupError, match="missing dependency"):
        vc.wait_tts_server(config, timeout=5)
    assert vc.tts_download_progress(config) is None


def test_external_loading_server_is_reused(loading_tts, monkeypatch):
    config, _, _ = loading_tts
    monkeypatch.setattr(vc, "_tts_retiring_url", None)
    monkeypatch.setattr(vc, "_tts_server_key", None)
    spawn = Mock(side_effect=AssertionError("duplicate server"))
    monkeypatch.setattr(vc.subprocess, "Popen", spawn)
    vc.ensure_tts_server({**config, "tts_server_command": ["never-run"]})
    spawn.assert_not_called()


def test_concurrent_loading_response_publishes_without_cache_gap(monkeypatch):
    config = {"tts_url": "http://loading-test"}
    health = {"ok": False, "state": "loading", "download": {"done": 1, "total": 10}}
    monkeypatch.setattr(vc, "_tts_health_cache", {})
    connection = Mock()
    connection.getresponse.side_effect = lambda: nullcontext(SimpleNamespace(
        status=503, read=lambda limit: json.dumps(health).encode()))
    monkeypatch.setattr(vc.http.client, "HTTPConnection", Mock(return_value=connection))
    monkeypatch.setattr(vc, "_DeadlineTimer", Mock())
    assert not vc.tts_server_up(config)

    publish = vc._cache_tts_health
    published, release = threading.Event(), threading.Event()

    def paused_publish(config, health=None):
        publish(config, health)
        published.set()
        assert release.wait(5)

    monkeypatch.setattr(vc, "_cache_tts_health", paused_publish)
    with ThreadPoolExecutor(1) as pool:
        poll = pool.submit(vc.tts_server_up, config)
        try:
            assert published.wait(5)
            # A second loading response must replace the snapshot in one write.
            assert vc.tts_download_progress(config) == health["download"]
        finally:
            release.set()
        assert not poll.result(timeout=5)


@pytest.mark.parametrize("poll_state", ["loading", "unavailable", "ready"])
def test_wait_uses_own_loading_response_during_concurrent_poll(monkeypatch, poll_state):
    config = {"tts_url": "http://loading-test"}
    monkeypatch.setattr(vc, "_tts_process", None)
    monkeypatch.setattr(vc, "_tts_health_cache", {})
    monkeypatch.setattr(vc, "_start_tts_health_poll", Mock())
    monkeypatch.setattr(vc, "_DeadlineTimer", Mock())
    waiter_thread = threading.get_ident()
    published, polled, ready = threading.Event(), threading.Event(), threading.Event()

    def response():
        state = ("ready" if ready.is_set() else "loading") if (
            threading.get_ident() == waiter_thread) else poll_state
        return nullcontext(SimpleNamespace(
            status={"ready": 200, "loading": 503, "unavailable": 500}[state],
            read=lambda limit: b'{"ok": false, "state": "loading"}'))

    connection = Mock(getresponse=Mock(side_effect=response))
    monkeypatch.setattr(vc.http.client, "HTTPConnection", Mock(return_value=connection))
    publish = vc._cache_tts_health

    def interleaved_publish(config, health=None):
        publish(config, health)
        if threading.get_ident() == waiter_thread and health is not None:
            published.set()
            assert polled.wait(5)

    monkeypatch.setattr(vc, "_cache_tts_health", interleaved_publish)
    # Advance readiness only once the waiter decides to keep waiting.
    monkeypatch.setattr(vc, "time", SimpleNamespace(
        time=vc.time.time, monotonic=vc.time.monotonic, sleep=lambda seconds: ready.set()))

    def poll():
        assert published.wait(5)
        try:
            return vc.tts_server_up(config)
        finally:
            polled.set()

    with ThreadPoolExecutor(1) as pool:
        concurrent_poll = pool.submit(poll)
        try:
            assert vc.wait_tts_server(config, timeout=5)
            assert ready.is_set()
        finally:
            published.set()
        assert concurrent_poll.result(timeout=5) == (poll_state == "ready")


def test_cached_progress_expires_without_polling(monkeypatch):
    monkeypatch.setattr(vc, "_tts_health_cache", {})
    config = {"tts_url": "http://unused"}
    vc._cache_tts_health(config, {"state": "loading", "download": {"done": 1, "total": 2}})
    assert vc.tts_download_progress(config)
    now = vc.time.monotonic()
    monkeypatch.setattr(vc.time, "monotonic", lambda: now + 7)
    assert vc.tts_download_progress(config) is None


@pytest.mark.parametrize("managed,ending", [
    (True, "ready"), (True, "exited"), (True, "failed"),
    (False, "ready"), (False, "failed"), (False, "disconnected"),
])
def test_tts_poll_outlives_startup_deadline(monkeypatch, managed, ending):
    config = {"tts_url": "http://loading-test"}
    process = Mock(poll=Mock(return_value=None)) if managed else None
    monkeypatch.setattr(vc, "_tts_process", process)
    monkeypatch.setattr(vc, "_tts_health_cache", {})
    monkeypatch.setattr(vc, "_tts_health_polls", {})
    now = [0]
    monkeypatch.setattr(vc, "time", SimpleNamespace(
        time=lambda: now[0], monotonic=lambda: now[0], sleep=vc.time.sleep,
        perf_counter=vc.time.perf_counter))
    refreshed, finish = threading.Event(), threading.Event()
    calls = []

    def health(config):
        vc._cache_tts_health(config)
        if finish.is_set():
            if ending == "failed":
                raise vc.TTSStartupError("failed")
            return "ready" if ending == "ready" else "unavailable"
        calls.append(None)
        # Each response consumes more than the startup budget and cache TTL.
        now[0] += vc.TTS_STARTUP_SECONDS + 1
        vc._cache_tts_health(config, {"state": "loading", "download": {
            "done": len(calls), "total": 100}})
        if len(calls) > 1:
            refreshed.set()
        return "loading"

    monkeypatch.setattr(vc, "_check_tts_health", health)
    worker = None
    try:
        assert not vc.wait_tts_server(config)
        worker = vc._tts_health_polls[config["tts_url"]][1]
        assert refreshed.wait(3)
        assert vc.tts_download_progress(config)["done"] >= 2
        vc._start_tts_health_poll(config, process)
        assert vc._tts_health_polls[config["tts_url"]][1] is worker
        finish.set()
        if ending == "exited":
            process.poll.return_value = 1
        worker.join(3)
        assert not worker.is_alive()
        assert vc.tts_download_progress(config) is None
        assert not vc._tts_health_polls
    finally:
        finish.set()
        # Avoid killing the fake process while still exercising poll cancellation.
        monkeypatch.setattr(vc, "_tts_process", None)
        vc.stop_tts_server()
        if worker is not None:
            worker.join(3)


def test_stopping_tts_polls_with_replaced_time(monkeypatch):
    cancelled = threading.Event()
    worker = threading.Thread(target=cancelled.wait)
    monkeypatch.setattr(vc, "_tts_health_polls", {"test": (cancelled, worker)})
    monkeypatch.setattr(vc, "time", SimpleNamespace())
    worker.start()
    try:
        vc.stop_tts_health_polls()
        assert cancelled.is_set()
        assert not worker.is_alive()
        assert not vc._tts_health_polls
    finally:
        cancelled.set()
        worker.join(3)


def test_stopping_tts_discards_inflight_health_response(monkeypatch):
    config = {"tts_url": "http://loading-test"}
    monkeypatch.setattr(vc, "_tts_process", None)
    monkeypatch.setattr(vc, "_tts_health_cache", {})
    monkeypatch.setattr(vc, "_tts_health_polls", {})
    started, release = threading.Event(), threading.Event()

    def health(config):
        started.set()
        assert release.wait(3)
        vc._cache_tts_health(config, {"download": {"done": 1, "total": 2}})
        return "loading"

    monkeypatch.setattr(vc, "_check_tts_health", health)
    vc._start_tts_health_poll(config)
    worker = vc._tts_health_polls[config["tts_url"]][1]
    try:
        assert started.wait(3)
        vc.stop_tts_server()
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert vc.tts_download_progress(config) is None
    assert not vc._tts_health_polls


@pytest.mark.parametrize("invalid", [{"done": 1e307, "total": 1e308},
                                    {"done": 10**400, "total": 10**401}])
def test_invalid_tts_download_does_not_stop_resource_monitor(monkeypatch, invalid):
    from debora_whisper import dictation_engine as de
    app = de.DictationApp.__new__(de.DictationApp)
    app.config = {"tts_url": "http://untrusted-test"}
    app._accelerator_memory = lambda: ""
    app._stopping = Mock()
    app._stopping.wait.side_effect = [False, False, True]
    monkeypatch.setattr(de.STT_DOWNLOAD, "snapshot", lambda: None)
    monkeypatch.setattr(vc, "_tts_health_cache", {})
    payloads = iter([invalid, {"done": 1, "total": 2}])
    connection = Mock()
    connection.getresponse.side_effect = lambda: nullcontext(SimpleNamespace(
        status=503, read=lambda limit: json.dumps({
            "ok": False, "state": "loading", "download": next(payloads)
        }).encode()[:limit]))
    monkeypatch.setattr(vc.http.client, "HTTPConnection", Mock(return_value=connection))
    monkeypatch.setattr(vc, "_DeadlineTimer", Mock())

    def progress(config):
        assert not vc.tts_server_up(config)
        return vc.tts_download_progress(config)

    monkeypatch.setattr(de, "tts_download_progress", progress)
    logs = []
    monkeypatch.setattr(de, "log", logs.append)
    app._monitor_resources()
    assert len(logs) == 2
    assert " DL " not in logs[0]
    assert " DL TTS 50% " in logs[1]


def test_telemetry_appends_downloads_and_drops_them_when_idle(monkeypatch):
    from debora_whisper import dictation_engine as de
    app = de.DictationApp.__new__(de.DictationApp)
    app.config = {}
    app._accelerator_memory = lambda: ""
    app._stopping = Mock()
    app._stopping.wait.side_effect = [False, False, True]
    monkeypatch.setattr(de.STT_DOWNLOAD, "snapshot", Mock(side_effect=[{"done": 42_000_000, "total": 100_000_000}, None]))
    monkeypatch.setattr(de, "tts_download_progress", Mock(side_effect=[{"done": 17_000_000, "total": 100_000_000}, None]))
    logs = []
    monkeypatch.setattr(de, "log", logs.append)
    app._monitor_resources()
    assert len(logs) == 2
    assert logs[0].endswith(" DL STT 42% (42/100 MB) DL TTS 17% (17/100 MB)")
    assert " DL " not in logs[1]
    assert all(call.args == (5.0,) for call in app._stopping.wait.call_args_list)
