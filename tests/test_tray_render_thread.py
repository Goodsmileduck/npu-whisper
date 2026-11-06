"""The tray must never call pystray (Shell_NotifyIcon) from its caller.

app.py calls update_state/update_audio_level from Tk callbacks. When those
blocked in Shell_NotifyIcon, Ctrl+C landed inside the callback and Tk
swallowed the KeyboardInterrupt ("Exception in Tkinter callback"), so the
app did not close.
"""
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


def _tray_class():
    try:
        import pystray  # noqa: F401
    except ImportError:
        with patch.dict(sys.modules, {"pystray": MagicMock()}):
            sys.modules.pop("debora_whisper.ui.tray", None)
            from debora_whisper.ui.tray import TrayManager
        sys.modules.pop("debora_whisper.ui.tray", None)
        return TrayManager
    from debora_whisper.ui.tray import TrayManager
    return TrayManager


class _FakeIcon:
    """Records which thread touched each pystray attribute."""
    visible = True

    def __init__(self):
        object.__setattr__(self, "calls", [])

    def __setattr__(self, name, value):
        self.calls.append((name, threading.current_thread().name))
        object.__setattr__(self, name, value)

    def update_menu(self):
        self.calls.append(("update_menu", threading.current_thread().name))

    def stop(self):
        pass


def _tray():
    tray = _tray_class()(on_toggle=lambda: None, on_quit=lambda: None)
    tray._icon = _FakeIcon()
    return tray


def _run_renderer(tray):
    tray._running = True
    t = threading.Thread(target=tray._render_loop, name="tray-render", daemon=True)
    t.start()
    return t


def _settle(icon, timeout=2.0):
    """Wait until the renderer stops producing calls."""
    deadline = time.time() + timeout
    last = -1
    while time.time() < deadline:
        n = len(icon.calls)
        if n == last:
            return
        last = n
        time.sleep(0.25)


def test_updates_do_not_touch_pystray_on_caller_thread():
    tray = _tray()
    tray.update_state("recording", "Débora Whisper — Recording...")
    tray.update_audio_level(0.8)
    tray.update_state("ready", "Débora Whisper — Ready")
    tray.refresh()
    tray.update_version("Débora Whisper 0.1.0 · ⬆ atualização disponível", True)
    assert tray._icon.calls == []


def test_first_menu_item_is_read_only_version():
    from types import SimpleNamespace

    tray = _tray()
    text = "Débora Whisper 0.1.0 · a6d8864 (2026-10-26) · ✓ atualizada"
    tray.update_version(text)
    pystray = MagicMock()
    pystray.Menu.side_effect = lambda *items: items
    pystray.MenuItem.side_effect = lambda text, action, **kwargs: SimpleNamespace(
        text=text, action=action, **kwargs)
    with patch.dict(tray._build_menu.__globals__, {"pystray": pystray}):
        first = tray._build_menu()[0]
    assert first.text == text
    assert first.action is None
    assert first.enabled is False


@pytest.mark.parametrize("notifications", [True, False])
def test_version_changes_refresh_menu_and_notify_once_on_renderer(notifications):
    tray = _tray()
    tray._icon.HAS_NOTIFICATION = notifications
    tray._icon.notify = lambda *args: tray._icon.calls.append(
        ("notify", threading.current_thread().name))
    tray._icon.calls.clear()
    tray.update_state("ready")
    t = _run_renderer(tray)
    try:
        _settle(tray._icon)
        tray._icon.calls.clear()
        for text, available in (("new version", True), ("unknown", False), ("new version", True)):
            tray.update_version(text, available)
            _settle(tray._icon)
        assert tray._icon.calls.count(("update_menu", "tray-render")) == 3
        assert tray._icon.calls.count(("notify", "tray-render")) == int(notifications)
        assert {thread for _, thread in tray._icon.calls} == {"tray-render"}
        tray._icon.calls.clear()
        tray.update_version("new version", True)
        _settle(tray._icon)
        assert tray._icon.calls == []
    finally:
        tray.stop()
        t.join(1)


def test_renderer_applies_state_on_its_own_thread():
    tray = _tray()
    t = _run_renderer(tray)
    try:
        tray.update_state("ready", "Débora Whisper — Ready")
        _settle(tray._icon)
        names = {name for name, _ in tray._icon.calls}
        assert {"icon", "title", "menu", "update_menu"} <= names
        assert {thread for _, thread in tray._icon.calls} == {"tray-render"}
    finally:
        tray.stop()
        t.join(1)
    assert not t.is_alive()


def test_repeated_state_makes_no_windows_calls():
    # Continuous mode re-sends RECORDING with every draft.
    tray = _tray()
    t = _run_renderer(tray)
    try:
        tray.update_state("ready", "Débora Whisper — Ready")
        _settle(tray._icon)
        tray._icon.calls.clear()
        for _ in range(5):
            tray.update_state("ready", "Débora Whisper — Ready")
        _settle(tray._icon)
        assert tray._icon.calls == []
    finally:
        tray.stop()
        t.join(1)


def test_silent_recording_does_not_redraw():
    tray = _tray()
    t = _run_renderer(tray)
    try:
        tray.update_state("recording", "Débora Whisper — Recording...")
        tray.update_audio_level(0.0)
        _settle(tray._icon)
        tray._icon.calls.clear()
        time.sleep(0.5)  # five 100ms frames of silence
        assert tray._icon.calls == []
    finally:
        tray.stop()
        t.join(1)


@pytest.mark.parametrize("state", ["ready", "recording"])
def test_mode_switch_refreshes_icon_and_menu_on_renderer(state):
    from debora_whisper.ui.icons import get_icon
    tray = _tray()
    enabled = False
    tray._voice_chat_on = lambda: enabled
    tray.update_state(state)
    t = _run_renderer(tray)
    try:
        _settle(tray._icon)
        for enabled in (True, False):
            tray._icon.calls.clear()
            tray.refresh()
            _settle(tray._icon)
            assert tray._icon.icon is get_icon(state, voice_chat=enabled)
            assert ("update_menu", "tray-render") in tray._icon.calls
            assert {thread for _, thread in tray._icon.calls} == {"tray-render"}
    finally:
        tray.stop()
        t.join(1)
