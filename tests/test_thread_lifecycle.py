"""Regression tests for test isolation and cancellable diagnostic workers."""
import io
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests import conftest
from debora_whisper import device_inventory as inventory, dictation_engine as de, voice_chat as vc


@pytest.mark.parametrize("daemon,allowed", [(False, False), (True, False), (False, True)])
def test_guard_identifies_survivor_and_honors_marker(monkeypatch, daemon, allowed):
    monkeypatch.setattr(conftest, "_THREAD_JOIN_SECONDS", 0.02)
    node = SimpleNamespace(nodeid="synthetic::thread-owner",
                           get_closest_marker=lambda name: object() if allowed else None)
    release = threading.Event()
    worker = threading.Thread(target=release.wait, name="intentional-survivor", daemon=daemon)
    with pytest.MonkeyPatch.context() as patches:
        guard = conftest._thread_guard.__wrapped__(SimpleNamespace(node=node), patches)
        next(guard)
        worker.start()
        try:
            match = "synthetic::thread-owner.*intentional-survivor"
            if daemon or allowed:
                with pytest.warns(pytest.PytestWarning, match=match), pytest.raises(StopIteration):
                    next(guard)
            else:
                with pytest.raises(pytest.fail.Exception, match=match):
                    next(guard)
        finally:
            release.set()
            worker.join(2)


def test_guard_ignores_threads_that_predate_the_test():
    release = threading.Event()
    worker = threading.Thread(target=release.wait)
    worker.start()
    node = SimpleNamespace(nodeid="synthetic::baseline", get_closest_marker=lambda name: None)
    try:
        with pytest.MonkeyPatch.context() as patches:
            guard = conftest._thread_guard.__wrapped__(SimpleNamespace(node=node), patches)
            next(guard)
            with pytest.raises(StopIteration):
                next(guard)
        assert worker.is_alive()
    finally:
        release.set()
        worker.join(2)


def test_monitor_shutdown_joins_before_returning():
    app = de.DictationApp({**de.DEFAULT_CONFIG, "beep_on_start": False})
    assert app._resource_thread.is_alive()
    app.stop_monitoring()
    assert not app._resource_thread.is_alive()
    app.stop_monitoring()  # Repeated cleanup is harmless.


def test_health_shutdown_joins_an_inflight_poll(monkeypatch):
    entered = threading.Event()

    def health(config):
        entered.set()
        assert vc._tts_health_poll_context.cancelled.wait(2)
        return "loading"

    monkeypatch.setattr(vc, "_check_tts_health", health)
    config = {"tts_url": "http://test.invalid"}
    vc._start_tts_health_poll(config)
    with vc._tts_health_cache_lock:
        worker = vc._tts_health_polls[config["tts_url"]][1]
    try:
        assert entered.wait(2)
    finally:
        vc.stop_tts_health_polls()
    assert not worker.is_alive()
    assert not vc._tts_health_polls


def test_cancelled_inventory_never_calls_the_planner(monkeypatch):
    entered = threading.Event()

    def probe(*args, cancelled):
        entered.set()
        assert cancelled.wait(2)
        return {}

    planner = Mock()
    monkeypatch.setattr(inventory, "probe_inventory", probe)
    monkeypatch.setattr("debora_whisper.device_planner.plan", planner)
    worker = de.log_hardware_inventory(dict(de.DEFAULT_CONFIG))
    try:
        assert entered.wait(2)
    finally:
        inventory.stop_inventory_workers()
    assert not worker.is_alive()
    planner.assert_not_called()


def test_inventory_shutdown_cancels_queries_and_joins_children(monkeypatch, tmp_path):
    from debora_whisper import processes
    entered = threading.Event()
    cancelled = threading.Event()

    class Process:
        stdin = io.BytesIO()
        stdout = io.BytesIO(b'{}\n')

        def wait(self, timeout):
            entered.set()
            cancelled.wait(timeout)
            raise subprocess.TimeoutExpired("inventory", timeout)

    process = Process()
    kill = Mock()
    monkeypatch.setattr(inventory, "SOURCES", ("tts",))
    monkeypatch.setattr(inventory.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(processes, "kill_tree", kill)
    worker = inventory.start_inventory_worker(
        lambda: inventory.probe_inventory({}, tmp_path / "inventory.json", cancelled=cancelled),
        cancelled=cancelled)
    try:
        assert entered.wait(2)
    finally:
        inventory.stop_inventory_workers()
    assert not worker.is_alive()
    assert not inventory._workers
    assert process.stdin.closed and process.stdout.closed
    kill.assert_called_once_with(process)
