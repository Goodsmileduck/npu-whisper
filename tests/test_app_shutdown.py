"""Shutdown-path tests for the GUI orchestrator.

Regression guard for Ctrl+C escaping Tk's mainloop as an uncaught traceback:
when that happened, `_quit` never ran, so the global keyboard hook and the
tray icon were left running until the process died.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

_GUI_STACK = (
    "tkinter", "tkinter.constants", "tkinter.font", "tkinter.ttk",
    "tkinter.messagebox", "tkinter.filedialog",
    "customtkinter", "pystray", "keyboard", "sounddevice", "pyperclip",
)

# app.py imports the whole GUI stack at module level, but the shutdown path
# under test is pure Python and touches none of it. Prefer the real modules —
# only when the import genuinely fails (a headless box typically has no system
# tkinter) fall back to stubs, so a working install is never shadowed.
#
# The stubs are then removed again: leaving them in sys.modules would let a
# later test module import a mock instead of the real dependency and pass
# against it, which is how a suite starts lying about what it covers.
#
# The fallback is scoped to ImportError specifically: a broader catch would let
# a genuine bug in app.py on a working GUI box get silently retried against full
# mocks and pass.
def _drop(names):
    for _n in list(names):
        sys.modules.pop(_n, None)


def _under_test():
    """Import app.py, falling back to stubs only for a real missing dependency."""
    try:
        from debora_whisper.app import GUIApp
        return GUIApp
    except ImportError:
        pass

    _drop([n for n in sys.modules if n == "debora_whisper.app" or n.startswith("debora_whisper.ui.")])
    injected = [n for n in _GUI_STACK if n not in sys.modules]
    for _n in injected:
        sys.modules[_n] = MagicMock()
    try:
        from debora_whisper.app import GUIApp
        return GUIApp
    finally:
        # Drop the stubs AND the mock-tainted modules they were imported into,
        # so nothing downstream can resolve to a mock.
        _drop(injected)
        _drop([n for n in sys.modules if n == "debora_whisper.app" or n.startswith("debora_whisper.ui.")])


GUIApp = _under_test()


def test_update_result_reaches_tray_and_open_settings_once():
    from debora_whisper.version import BuildInfo, UPDATE_AVAILABLE, display_version

    app = GUIApp.__new__(GUIApp)
    app._version_checker = MagicMock()
    result = (BuildInfo("0.1.0", commit="a6d8864", date="2026-10-26"), UPDATE_AVAILABLE)
    app._version_checker.result = result
    app._shown_version = None
    app._tray = MagicMock()
    app._settings_win = MagicMock()
    # Created but not shown yet: the text must still reach it.
    app._settings_win.is_open = False
    app._refresh_version()
    app._refresh_version()
    text = display_version(*result)
    app._tray.update_version.assert_called_once_with(text, True)
    app._settings_win.update_version.assert_called_once_with(text)


def test_new_settings_window_receives_current_version():
    from debora_whisper.version import BuildInfo, UNKNOWN, display_version

    app = GUIApp.__new__(GUIApp)
    app._version_checker = MagicMock(result=(BuildInfo("0.1.0"), UNKNOWN))
    app._config = {}
    app._root = MagicMock()
    window = MagicMock()
    with patch.dict(GUIApp._show_settings.__globals__, {
        "SettingsWindow": window, "DictationApp": MagicMock(),
    }):
        app._show_settings()
    assert window.call_args.kwargs["version_text"] == display_version(*app._version_checker.result)


@pytest.mark.parametrize('key,value', [
    ('beep_on_start', True),
    ('beep_on_start', False),
    ('sample_rate', 48000),
    ('max_record_seconds', 30),
])
def test_audio_settings_restart_engine(key, value):
    from debora_whisper.dictation_engine import DEFAULT_CONFIG
    app = GUIApp.__new__(GUIApp)
    app._config = dict(DEFAULT_CONFIG)
    if key == 'beep_on_start':
        app._config[key] = not value
    old_engine = MagicMock()
    old_engine.stop_if_idle.return_value = None  # idle: it stops itself
    app._engine = old_engine
    app._overlay = MagicMock()
    app._tray = MagicMock()
    app._settings_status = MagicMock()
    app._settings_set_apply = MagicMock()
    # Use the function's globals, including on headless hosts where the
    # import helper removes app from sys.modules after installing GUI stubs.
    new_engine = MagicMock()
    engine_factory = MagicMock(return_value=new_engine)
    with patch.dict(GUIApp._on_settings_apply.__globals__, {
        'DictationApp': engine_factory, 'save_config': MagicMock(),
    }):
        app._on_settings_apply({**app._config, key: value})
    old_engine.stop_if_idle.assert_called_once()
    engine_factory.assert_called_once_with(app._config)
    new_engine.start_background.assert_called_once()


@pytest.mark.parametrize("voice_chat", [False, True])
def test_overlay_and_tray_toggle_follow_replaced_engine(voice_chat):
    """Settings replaces the engine; the overlay dot and the tray menu item
    must drive the new engine, not the stopped one they were built with."""
    from types import SimpleNamespace
    from debora_whisper.dictation_engine import DEFAULT_CONFIG

    g = GUIApp.__init__.__globals__
    RealTray, RealOverlay = g["TrayManager"], g["OverlayWindow"]
    captured = {}

    class CapturingOverlay:
        def __init__(self, root, on_toggle=None, **kwargs):
            captured["on_toggle"] = on_toggle

        def set_show_balloon(self, value):
            pass

        def set_balloon_font_size(self, value):
            pass

        def set_voice_mode(self, value):
            captured["voice_mode"] = value

    first, second = MagicMock(), MagicMock()
    first.stop_if_idle.return_value = None
    factory = MagicMock(side_effect=[first, second])
    config = dict(DEFAULT_CONFIG)
    config["voice_chat"] = voice_chat
    with patch.dict(g, {"ctk": MagicMock(), "DictationApp": factory,
                        "OverlayWindow": CapturingOverlay,
                        "save_config": MagicMock()}), \
            patch("debora_whisper.ui.icons.render_app_icon"), patch("PIL.ImageTk.PhotoImage"):
        app = GUIApp(config)  # real __init__ wiring
        assert captured["voice_mode"] is voice_chat
        assert app._tray._voice_chat_on() is voice_chat
        app._settings_win = None
        app._on_settings_apply({**config, "beep_on_start": not config["beep_on_start"]})

    assert app._engine is second
    # Real tray menu handler, built by the real __init__.
    assert isinstance(app._tray, RealTray)
    app._tray._on_toggle_click()
    # Real mascot-click handler, given the callback __init__ passed in.
    overlay = SimpleNamespace(_CLICK_STATES=RealOverlay._CLICK_STATES, _state="ready",
                              _on_toggle=captured["on_toggle"], _scale=1.0, _cur_w=64,
                              _hit_region=lambda x: "mascot",
                              _mascot_geometry=lambda: (2, 34, 62, 64))
    RealOverlay._on_drag_start(overlay, SimpleNamespace(x=0, y=0))
    RealOverlay._on_drag_end(overlay, SimpleNamespace(x=0, y=0))

    assert second.toggle_recording.call_count == 2
    first.toggle_recording.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_voice_mode_switch_updates_overlay_and_tray(enabled):
    app = GUIApp.__new__(GUIApp)
    app._config = {"voice_chat": not enabled}
    app._overlay = MagicMock()
    app._tray = MagicMock()
    app._engine = MagicMock()
    app._settings_win = MagicMock()
    with patch.dict(GUIApp._set_voice_chat.__globals__, {"save_config": MagicMock()}):
        app._set_voice_chat(enabled)
    assert app._config["voice_chat"] is enabled
    app._overlay.set_voice_mode.assert_called_once_with(enabled)
    app._tray.refresh.assert_called_once_with()
    app._engine.set_voice_chat.assert_called_once_with(enabled, listen=True)
    app._settings_win.set_voice_chat.assert_called_once_with(enabled)


class _Stop:
    """Stand-in for engine.stop / tray.stop that records how often it ran."""

    def __init__(self, raises: Exception | None = None):
        self.calls = 0
        self._raises = raises

    def __call__(self):
        self.calls += 1
        if self._raises is not None:
            raise self._raises


class _Root:
    """Minimal Tk root: mainloop can be told to raise, destroy is recorded."""

    def __init__(self, mainloop_raises: Exception | None = None,
                 after_raises: Exception | None = None):
        self.destroyed = 0
        self.scheduled = []
        self._raises = mainloop_raises
        self._after_raises = after_raises

    def mainloop(self):
        if self._raises is not None:
            raise self._raises

    def after(self, delay, fn):
        if self._after_raises is not None:
            raise self._after_raises
        self.scheduled.append(fn)

    def destroy(self):
        self.destroyed += 1


def _bare_app(engine_stop, tray_stop, root=None):
    """A GUIApp carrying only the attributes the shutdown path touches.

    Built with __new__ so the test needs no display, no model files and no
    tray thread — the real __init__ constructs all three.
    """
    app = GUIApp.__new__(GUIApp)
    app._torn_down = False
    app._engine = type("_Engine", (), {"stop": engine_stop})()
    app._tray = type("_Tray", (), {"stop": tray_stop})()
    app._root = root if root is not None else _Root()
    return app


class TestTeardown:
    def test_stops_update_worker(self):
        app = _bare_app(_Stop(), _Stop())
        app._version_checker = MagicMock()
        app._teardown()
        app._teardown()
        app._version_checker.stop.assert_called_once_with()

    def test_teardown_runs_once_even_if_engine_stop_fails(self):
        engine, tray = _Stop(raises=RuntimeError("stop failed")), _Stop()
        app = _bare_app(engine, tray)
        app._teardown()
        app._teardown()
        assert (engine.calls, tray.calls) == (1, 1)

    def test_stops_engine_and_tray(self):
        engine, tray = _Stop(), _Stop()
        _bare_app(engine, tray)._teardown()
        assert (engine.calls, tray.calls) == (1, 1)

    def test_is_idempotent(self):
        """The tray Quit and the Ctrl+C path can both fire; the second is a no-op."""
        engine, tray = _Stop(), _Stop()
        app = _bare_app(engine, tray)
        app._teardown()
        app._teardown()
        assert (engine.calls, tray.calls) == (1, 1)

    def test_tray_still_stops_when_engine_stop_raises(self):
        """A failing engine.stop must not strand the tray icon."""
        engine, tray = _Stop(raises=RuntimeError("hook already gone")), _Stop()
        _bare_app(engine, tray)._teardown()
        assert tray.calls == 1


class TestQuit:
    """The tray's Quit path. Untested at first review, which meant a regression
    dropping `_teardown()` out of `_quit` passed the whole suite."""

    def test_stops_engine_and_tray(self):
        engine, tray = _Stop(), _Stop()
        _bare_app(engine, tray)._quit()
        assert (engine.calls, tray.calls) == (1, 1)

    def test_schedules_destroy_on_the_main_thread(self):
        """_quit runs on the tray thread, so destroy must be queued, not called."""
        root = _Root()
        app = _bare_app(_Stop(), _Stop(), root)

        app._quit()

        assert root.destroyed == 0, "destroy must not run inline on the tray thread"
        assert len(root.scheduled) == 1
        root.scheduled[0]()  # what the mainloop would run
        assert root.destroyed == 1

    def test_teardown_survives_a_dead_root(self):
        """Ctrl+C may have already torn the root down when Quit arrives."""
        engine, tray = _Stop(), _Stop()
        root = _Root(after_raises=RuntimeError("application has been destroyed"))
        app = _bare_app(engine, tray, root)

        app._quit()  # must not propagate

        assert (engine.calls, tray.calls) == (1, 1)

    def test_quit_then_interrupt_does_not_double_stop(self):
        """Both shutdown paths can fire in one exit; the second is a no-op."""
        engine, tray = _Stop(), _Stop()
        root = _Root(mainloop_raises=KeyboardInterrupt())
        app = _bare_app(engine, tray, root)

        app._quit()
        app._mainloop()

        assert (engine.calls, tray.calls) == (1, 1)


class TestMainloopInterrupt:
    def test_keyboard_interrupt_is_swallowed_and_tears_down(self):
        engine, tray = _Stop(), _Stop()
        root = _Root(mainloop_raises=KeyboardInterrupt())
        app = _bare_app(engine, tray, root)

        app._mainloop()  # must not propagate

        assert (engine.calls, tray.calls) == (1, 1)
        assert root.destroyed == 1

    def test_normal_exit_also_tears_down(self):
        engine, tray = _Stop(), _Stop()
        root = _Root()
        app = _bare_app(engine, tray, root)

        app._mainloop()

        assert (engine.calls, tray.calls) == (1, 1)
        assert root.destroyed == 1

    def test_real_errors_still_propagate(self):
        """Only KeyboardInterrupt is absorbed — a genuine crash must surface."""
        engine, tray = _Stop(), _Stop()
        root = _Root(mainloop_raises=RuntimeError("Tk exploded"))
        app = _bare_app(engine, tray, root)

        with pytest.raises(RuntimeError, match="Tk exploded"):
            app._mainloop()

        assert (engine.calls, tray.calls) == (1, 1)  # teardown still ran
