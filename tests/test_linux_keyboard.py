"""The evdev keyboard module: hotkey names, suppression, hooks and the
virtual keyboard. No real device is read or written: they are fakes."""
from types import SimpleNamespace
from unittest.mock import Mock

import sys

import pytest

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux provider")

evdev = pytest.importorskip("evdev")
from evdev import ecodes as e  # noqa: E402

from debora_whisper.platforms import linux_keyboard as lk  # noqa: E402


# -- hotkey names ------------------------------------------------------------

def test_hotkey_names_match_the_keyboard_package():
    # Steps of keys of codes, like keyboard.parse_hotkey.
    assert lk.parse_hotkey("ctrl+space") == (((e.KEY_LEFTCTRL, e.KEY_RIGHTCTRL), (e.KEY_SPACE,)),)
    assert lk.key_to_scan_codes("right alt") == (e.KEY_RIGHTALT,)
    assert lk.key_to_scan_codes("alt gr") == (e.KEY_RIGHTALT,)
    assert lk.key_to_scan_codes("Enter") == (e.KEY_ENTER,)
    assert lk.key_to_scan_codes("f9") == (e.KEY_F9,)
    assert lk.key_to_scan_codes("page down") == (e.KEY_PAGEDOWN,)


@pytest.mark.parametrize("bad", ["", "nokey", "ctrl+"])
def test_unknown_keys_raise_value_error(bad):
    with pytest.raises(ValueError):
        lk.parse_hotkey(bad)


def test_event_names_are_the_keyboard_packages():
    assert lk.event_name(e.KEY_RIGHTALT) == "right alt"
    assert lk.event_name(e.KEY_LEFTCTRL) == "ctrl"
    assert lk.event_name(e.KEY_A) == "a"


# -- the listener, fed fake events ---------------------------------------------

class Mirror:
    def __init__(self):
        self.keys = []

    def write_event(self, event):
        if event.type == e.EV_KEY:
            self.keys.append((event.code, event.value))


def key(code, value):
    return evdev.InputEvent(0, 0, e.EV_KEY, code, value)


def press(listener, mirror, *events):
    for code, value in events:
        event = key(code, value)
        if listener._handle(event) and mirror is not None:
            mirror.write_event(event)
    calls = []
    while not listener._calls.empty():
        callback, args = listener._calls.get_nowait()
        calls.append((callback, args))
    return calls


@pytest.fixture
def listener(monkeypatch):
    listener = lk._Listener()
    monkeypatch.setattr(listener, "_ensure_running", lambda: None)
    monkeypatch.setattr(listener, "_regrab", lambda: None)
    return listener


def test_suppressed_hotkey_fires_once_and_never_reaches_the_app(listener):
    fired = Mock()
    listener.add_hotkey("ctrl+space", fired, suppress=True)
    mirror = Mirror()
    calls = press(listener, mirror, (e.KEY_LEFTCTRL, 1), (e.KEY_SPACE, 1), (e.KEY_SPACE, 2))
    assert [c for c, _ in calls] == [fired]  # auto-repeat does not fire again
    assert listener.is_pressed("ctrl+space")
    press(listener, mirror, (e.KEY_SPACE, 0), (e.KEY_LEFTCTRL, 0))
    assert not listener.is_pressed("ctrl+space")
    # Ctrl alone reaches the app; Space's down, repeat and up do not.
    assert mirror.keys == [(e.KEY_LEFTCTRL, 1), (e.KEY_LEFTCTRL, 0)]


def test_extra_modifier_is_another_shortcut(listener):
    fired = Mock()
    listener.add_hotkey("ctrl+space", fired, suppress=True)
    mirror = Mirror()
    calls = press(listener, mirror, (e.KEY_LEFTCTRL, 1), (e.KEY_LEFTSHIFT, 1),
                  (e.KEY_SPACE, 1), (e.KEY_SPACE, 0))
    assert calls == []
    assert (e.KEY_SPACE, 1) in mirror.keys and (e.KEY_SPACE, 0) in mirror.keys


def test_right_ctrl_counts_as_ctrl(listener):
    fired = Mock()
    listener.add_hotkey("ctrl+space", fired)
    assert [c for c, _ in press(listener, None, (e.KEY_RIGHTCTRL, 1), (e.KEY_SPACE, 1))] == [fired]


def test_hooks_see_named_down_and_up_events(listener):
    seen = []
    listener.hook(seen.append)
    calls = press(listener, None, (e.KEY_RIGHTALT, 1), (e.KEY_RIGHTALT, 2), (e.KEY_RIGHTALT, 0))
    for callback, args in calls:
        callback(*args)
    assert [(ev.name, ev.event_type, ev.scan_code) for ev in seen] == [
        ("right alt", "down", e.KEY_RIGHTALT), ("right alt", "up", e.KEY_RIGHTALT)]


def test_solo_key_tap_works_on_evdev_events(listener):
    from debora_whisper.solo_key import SoloKeyTap
    tapped = Mock()
    listener.hook(SoloKeyTap("right alt", tapped).handle)
    for callback, args in press(listener, None, (e.KEY_RIGHTALT, 1), (e.KEY_RIGHTALT, 0)):
        callback(*args)
    tapped.assert_called_once_with()
    tapped.reset_mock()
    # AltGr+Q types "/" and is no tap.
    for callback, args in press(listener, None, (e.KEY_RIGHTALT, 1), (e.KEY_Q, 1),
                                (e.KEY_Q, 0), (e.KEY_RIGHTALT, 0)):
        callback(*args)
    tapped.assert_not_called()


def test_unhook_all_forgets_hotkeys_and_releases_devices(listener, monkeypatch):
    listener.add_hotkey("ctrl+space", Mock(), suppress=True)
    listener.hook(Mock())
    device, mirror = Mock(), Mock()
    listener._devices["/dev/input/event9"] = (device, mirror)
    monkeypatch.setattr(listener, "_selector", Mock())
    listener.unhook_all()
    assert listener._hotkeys == [] and listener._hooks == [] and listener._devices == {}
    device.ungrab.assert_called_once_with()
    device.close.assert_called_once_with()
    mirror.close.assert_called_once_with()


def test_only_real_keyboards_are_read():
    def device(name, keys):
        return SimpleNamespace(name=name, capabilities=lambda: {e.EV_KEY: keys})
    assert lk.is_keyboard(device("AT Translated Set 2 keyboard", [e.KEY_A, e.KEY_SPACE]))
    assert not lk.is_keyboard(device("Power Button", [e.KEY_POWER]))
    assert not lk.is_keyboard(device("debora-whisper AT keyboard", [e.KEY_A, e.KEY_SPACE]))


def test_send_presses_in_order_and_releases_in_reverse(monkeypatch):
    ui = Mock()
    typist = lk._Typist()
    monkeypatch.setattr(typist, "_ui", lambda: ui)
    typist.emit(lk._parse_keys("ctrl+shift+v"), True, True)
    keys = [c.args[1:] for c in ui.write.call_args_list]
    assert keys == [(e.KEY_LEFTCTRL, 1), (e.KEY_LEFTSHIFT, 1), (e.KEY_V, 1),
                    (e.KEY_V, 0), (e.KEY_LEFTSHIFT, 0), (e.KEY_LEFTCTRL, 0)]
    assert ui.syn.call_count == 6


# -- the keyboard package's return shapes, as the engine uses them -------------

@pytest.fixture
def module_listener(listener, monkeypatch):
    monkeypatch.setattr(lk, "_listener", listener)
    monkeypatch.setattr(lk, "_removers", {})
    return listener


def test_hook_and_add_hotkey_return_removers(module_listener):
    callback, observer = Mock(), Mock()
    remove_hotkey = lk.add_hotkey("ctrl+space", callback, suppress=True)
    remove_hook = lk.hook(observer)
    assert len(module_listener._hotkeys) == 1 and module_listener._hooks == [observer]
    remove_hotkey()
    remove_hook()
    remove_hook()  # twice is harmless, as removing twice in a recovery
    assert module_listener._hotkeys == [] and module_listener._hooks == []
    # The callback still works as a handle too.
    lk.add_hotkey("ctrl+space", callback)
    lk.remove_hotkey(callback)
    assert module_listener._hotkeys == [] and lk._removers == {}


def test_engine_installs_and_reinstalls_hotkeys_on_evdev(module_listener, monkeypatch):
    from debora_whisper import platforms
    from debora_whisper.dictation_engine import DEFAULT_CONFIG, DictationApp
    from debora_whisper.platforms.linux import LinuxPlatform
    monkeypatch.setattr(platforms, "current", lambda: LinuxPlatform())
    app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False})
    try:
        app._install_hotkeys(lk)
        assert app._hotkey_codes == {e.KEY_LEFTCTRL, e.KEY_RIGHTCTRL, e.KEY_SPACE}
        # The engine's observer, the hotkey and the Right Alt tap.
        assert len(module_listener._hotkeys) == 1 and len(module_listener._hooks) == 2
        app._install_hotkeys(lk, announce=False)  # Settings Apply / recovery
        assert len(module_listener._hotkeys) == 1 and len(module_listener._hooks) == 2
        mirror = Mirror()
        calls = press(module_listener, mirror, (e.KEY_LEFTCTRL, 1), (e.KEY_SPACE, 1))
        assert app.toggle_recording in [c for c, _ in calls]
        assert (e.KEY_SPACE, 1) not in mirror.keys
    finally:
        app.stop_monitoring()


# -- keyboards are never left grabbed or blocked ------------------------------

class Device:
    def __init__(self, held=(), queued=()):
        self.name, self.held, self.queued = "AT keyboard", list(held), list(queued)
        self.ungrab, self.grab, self.close = Mock(), Mock(), Mock()

    def active_keys(self):
        return self.held

    def read_one(self):
        return self.queued.pop(0) if self.queued else None


def test_a_held_key_defers_the_grab_without_blocking(listener, monkeypatch):
    mirror = Mock()
    monkeypatch.setattr(lk.evdev.UInput, "from_device", Mock(return_value=mirror))
    monkeypatch.setattr(lk.time, "sleep", Mock(side_effect=AssertionError("blocked")))
    device = Device(held=[e.KEY_A])
    listener._devices["/dev/input/event9"] = (device, None)
    listener._grab_device("/dev/input/event9")
    assert listener._pending == {"/dev/input/event9"}
    device.grab.assert_not_called()
    device.held = []
    listener._grab_device("/dev/input/event9")
    assert listener._pending == set()
    device.grab.assert_called_once_with()
    assert listener._devices["/dev/input/event9"] == (device, mirror)


def run_reader(listener, monkeypatch, device, mirror, events):
    path = "/dev/input/event9"
    listener._devices[path] = (device, mirror)
    device.read = Mock(return_value=events)
    selector = Mock()

    def select(timeout):
        if selector.select.call_count > 1:
            listener._stop.set()
            return []
        return [(SimpleNamespace(data=path), 1)]
    selector.select.side_effect = select
    monkeypatch.setattr(listener, "_selector", selector)
    monkeypatch.setattr(listener, "_scan", lambda: None)
    listener._run()
    return path


def test_a_failing_virtual_keyboard_gives_the_device_back(listener, monkeypatch):
    device, mirror = Device(), Mock()
    mirror.write_event.side_effect = OSError("No such device")
    path = run_reader(listener, monkeypatch, device, mirror,
                      [key(e.KEY_A, 1), key(e.KEY_A, 0)])
    device.ungrab.assert_called_once_with()
    mirror.close.assert_called_once_with()
    assert listener._devices[path] == (device, None)  # still read, no longer grabbed
    assert "cannot forward keys" in listener.error


def test_a_crashing_reader_releases_every_keyboard(listener, monkeypatch):
    device, mirror = Device(), Mock()
    monkeypatch.setattr(listener, "_track", Mock(side_effect=RuntimeError("bug")))
    monkeypatch.setattr(listener, "_loop", Mock(side_effect=RuntimeError("bug")))
    listener._devices["/dev/input/event9"] = (device, mirror)
    listener._run()
    device.ungrab.assert_called_once_with()
    mirror.close.assert_called_once_with()


def test_an_event_bug_still_forwards_the_key(listener, monkeypatch):
    monkeypatch.setattr(listener, "_track", Mock(side_effect=RuntimeError("bug")))
    assert listener._handle(key(e.KEY_A, 1)) is True


def test_suspend_gives_keyboards_back_and_resume_grabs_them(listener, monkeypatch):
    device, mirror = Device(), Mock()
    regrab = Mock()
    monkeypatch.setattr(listener, "_regrab", regrab)
    listener._devices["/dev/input/event9"] = (device, mirror)
    listener._grab = True
    listener.suspend()
    device.ungrab.assert_called_once_with()
    mirror.close.assert_called_once_with()
    assert listener._devices["/dev/input/event9"] == (device, None)
    listener.resume()
    regrab.assert_called_once_with()


def test_resume_drops_keys_typed_while_stopped(listener, monkeypatch):
    monkeypatch.setattr(listener, "_regrab", Mock())
    device = Device(held=[e.KEY_LEFTSHIFT],
                    queued=[key(e.KEY_LEFTCTRL, 1), key(e.KEY_SPACE, 1)])
    listener._devices["/dev/input/event9"] = (device, None)
    listener._pressed = {e.KEY_LEFTCTRL}  # its release went to the shell
    listener._suppressed = {e.KEY_SPACE}
    listener.resume()
    assert device.queued == []
    assert listener._pressed == {e.KEY_LEFTSHIFT}
    assert listener._suppressed == set()


def test_reader_woken_before_resume_ignores_keys_typed_while_stopped(listener, monkeypatch):
    fired = []
    listener.add_hotkey("ctrl+space", lambda: fired.append(True), suppress=True)
    listener.suspend()
    run_reader(listener, monkeypatch, Device(), None,
               [key(e.KEY_LEFTCTRL, 1), key(e.KEY_SPACE, 1)])
    assert listener._calls.empty() and not fired
    assert listener._pressed == set()
    listener.resume()
    assert not listener._suspended


def test_keyboard_plugged_in_while_suspended_is_not_grabbed(listener, monkeypatch):
    monkeypatch.setattr(lk.evdev.UInput, "from_device", Mock(return_value=Mock()))
    device = Device()
    listener._devices["/dev/input/event9"] = (device, None)
    listener.suspend()
    listener._grab_device("/dev/input/event9")
    device.grab.assert_not_called()
    assert listener._devices["/dev/input/event9"] == (device, None)


def test_close_waits_for_the_callback_worker(monkeypatch):
    listener = lk._Listener()
    worker = Mock()
    listener._worker = worker
    listener.close()
    worker.join.assert_called_once_with(1)


def test_resume_without_suppressed_hotkeys_grabs_nothing(listener, monkeypatch):
    regrab = Mock()
    monkeypatch.setattr(listener, "_regrab", regrab)
    listener.resume()
    regrab.assert_not_called()


def test_restarted_reader_reuses_the_live_callback_worker(monkeypatch):
    listener = lk._Listener()
    monkeypatch.setattr(listener, "_scan", lambda: listener._devices.setdefault(
        "/dev/input/event9", (Device(), None)))
    started = []

    class Thread:
        def __init__(self, target, name, daemon):
            self.name, self.alive = name, False

        def start(self):
            self.alive = True
            started.append(self.name)

        def is_alive(self):
            return self.alive

    monkeypatch.setattr(lk.threading, "Thread", Thread)
    listener._ensure_running()
    listener._thread.alive = False  # the reader crashed; its worker lives on
    listener._ensure_running()
    assert started == ["evdev-keyboard", "evdev-callbacks", "evdev-keyboard"]
