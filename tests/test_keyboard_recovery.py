"""Native Windows hotkey contracts without real hooks, windows or key input."""
import ctypes
import importlib
import json
import queue
import sys
import threading
from unittest.mock import Mock, call, patch

import pytest

from debora_whisper import dictation_engine as de
from debora_whisper.platforms import windows_hotkeys as wh
from debora_whisper.platforms.windows import WindowsPlatform


@pytest.mark.parametrize("hotkey,modifiers,vk", [
    ("ctrl+space", wh.MOD_CONTROL, 0x20),
    (" CTRL + Shift + D ", wh.MOD_CONTROL | wh.MOD_SHIFT, ord("D")),
    ("alt+f9", wh.MOD_ALT, 0x78),
    ("windows+7", wh.MOD_WIN, ord("7")),
    ("control+page down", wh.MOD_CONTROL, 0x22),
    ("f24", 0, 0x87),
    ("f12", 0, 0x7b),
    ("ctrl+f12", wh.MOD_CONTROL, 0x7b),
    ("shift+escape", wh.MOD_SHIFT, 0x1b),
    ("ctrl+;", wh.MOD_CONTROL, 0xba),
    ("ctrl+.", wh.MOD_CONTROL, 0xbe),
    ("ctrl+`", wh.MOD_CONTROL, 0xc0),
    ("ctrl+alt+-", wh.MOD_CONTROL | wh.MOD_ALT, 0xbd),
    ("ctrl+/", wh.MOD_CONTROL, 0xbf),
    ("alt+num 0", wh.MOD_ALT, 0x60),
    ("ctrl+menu", wh.MOD_CONTROL, 0x5d),
    ("ctrl+apps", wh.MOD_CONTROL, 0x5d),
    ("scroll lock", 0, 0x91),
    ("pause", 0, 0x13),
    ("ctrl+break", wh.MOD_CONTROL, 0x03),
])
def test_hotkey_maps_to_modifiers_and_vk(native_api, hotkey, modifiers, vk):
    assert wh.parse_hotkey(hotkey) == (modifiers | wh.MOD_NOREPEAT, vk)


@pytest.mark.parametrize("name,vk", [
    *[(f"num {n}", 0x60 + n) for n in range(10)],
    ("num multiply", 0x6a), ("num *", 0x6a), ("multiply", 0x6a),
    ("num plus", 0x6b), ("num add", 0x6b), ("num +", 0x6b), ("add", 0x6b),
    ("num separator", 0x6c), ("separator", 0x6c),
    ("num minus", 0x6d), ("num sub", 0x6d), ("num subtract", 0x6d),
    ("num -", 0x6d), ("subtract", 0x6d),
    ("num decimal", 0x6e), ("num del", 0x6e), ("num .", 0x6e), ("decimal", 0x6e),
    ("num divide", 0x6f), ("num /", 0x6f), ("divide", 0x6f),
])
@pytest.mark.parametrize("padded_upper", [False, True])
def test_explicit_numpad_names_preserve_vk(monkeypatch, name, vk, padded_upper):
    # Do not use native_api: it replaces keyboard's name lookup. Exercise the
    # real parser/resolver, mocking only Win32 and forbidding layout lookup.
    api = Mock()
    api.VkKeyScanW.side_effect = AssertionError("explicit numpad name used layout lookup")
    api.MapVirtualKeyW.side_effect = AssertionError("explicit numpad name used scan lookup")
    monkeypatch.setattr(wh, "_user32", api)
    if padded_upper:
        name = f" {name.upper()} "
    assert wh.key_to_vk(name) == vk
    assert wh.parse_hotkey("ctrl+" + name) == (wh.MOD_CONTROL | wh.MOD_NOREPEAT, vk)
    assert wh.parse_hotkey(name + "+ctrl") == (wh.MOD_CONTROL | wh.MOD_NOREPEAT, vk)
    assert wh.parse_hotkey(name) == (wh.MOD_NOREPEAT, vk)
    api.VkKeyScanW.assert_not_called()
    api.MapVirtualKeyW.assert_not_called()


@pytest.mark.parametrize("name", ["num enter", " NUM ENTER "])
def test_numpad_enter_is_rejected_as_indistinguishable(monkeypatch, name):
    api = Mock()
    monkeypatch.setattr(wh, "_user32", api)
    reason = "RegisterHotKey cannot distinguish num enter from enter"
    with pytest.raises(ValueError, match=reason):
        wh.key_to_vk(name)
    with pytest.raises(ValueError, match=reason):
        wh.parse_hotkey("ctrl+" + name)
    api.VkKeyScanW.assert_not_called()
    api.MapVirtualKeyW.assert_not_called()


@pytest.mark.parametrize("hotkey,reason", [
    ("ctrl", "non-modifier"), ("ctrl+shift", "non-modifier"),
    ("ctrl+a+b", "non-modifier"), ("ctrl+a, b", "sequences"),
    ("", "one chord"), (None, "one chord"),
    ("ctrl+", "unsupported key"), ("ctrl+not-a-key", "unsupported key"),
    ("ctrl+num ++", "unsupported key"), ("ctrl+num +a", "unsupported key"),
    ("ctrl+f25", "unsupported key"), ("right ctrl+space", "left/right"),
    ("right alt", "left/right"),
    ("ctrl+left control", "non-modifier"),
])
def test_unmappable_hotkeys_fail_clearly(native_api, hotkey, reason):
    with pytest.raises(ValueError, match=reason):
        wh.parse_hotkey(hotkey)


@pytest.fixture
def native_api(monkeypatch):
    messages = queue.Queue()
    api = Mock()
    api.RegisterClassW.return_value = 1
    api.CreateWindowExW.return_value = 123
    api.RegisterRawInputDevices.return_value = True
    api.RegisterHotKey.return_value = True
    api.DefWindowProcW.return_value = 0
    api.GetAsyncKeyState.return_value = 0
    api.GetMessageTime.return_value = 1000
    api.VkKeyScanW.side_effect = lambda key: {
        ";": 0xba, ".": 0xbe, "`": 0xc0, "-": 0xbd, "/": 0xbf,
    }.get(key, -1)
    scans = {"space": (57,), "escape": (1,), "left control": (29,),
             "-": (74, 12), ";": (39,), ".": (83, 52), "`": (41,), "/": (53,)}
    monkeypatch.setattr("keyboard.key_to_scan_codes",
                        lambda name, error_if_missing=True: scans.get(name, ()))
    scan_vks = {57: 0x20, 1: 0x1b, 29: 0xa2, 74: 0x6d, 12: 0xbd,
                39: 0xba, 83: 0x6e, 52: 0xbe, 41: 0xc0, 53: 0xbf}
    vk_chars = {0x6d: ord("-"), 0xbd: ord("-"), 0xba: ord(";"),
                0x6e: ord("."), 0xbe: ord("."), 0xc0: ord("`"), 0xbf: ord("/")}
    api.MapVirtualKeyW.side_effect = lambda code, mode: {
        wh.MAPVK_VSC_TO_VK_EX: scan_vks,
        wh.MAPVK_VK_TO_CHAR: vk_chars,
    }.get(mode, {}).get(code, 0)

    def get_message(pointer, *args):
        message, identifier = messages.get(timeout=2)
        if message in (0, -1):
            return message
        pointer._obj.message = message
        pointer._obj.wParam = identifier
        return 1

    def post_message(thread_id, message, *args):
        messages.put((0 if message == wh.WM_QUIT else message, 0))
        return True

    api.GetMessageW.side_effect = get_message
    api.PostThreadMessageW.side_effect = post_message
    monkeypatch.setattr(wh, "_user32", api)
    monkeypatch.setattr(wh, "_kernel32", Mock(GetModuleHandleW=Mock(return_value=42),
                                            GetTickCount=Mock(return_value=1000)))
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 1409, raising=False)
    monkeypatch.setattr(ctypes, "FormatError", lambda error: "Hot key already registered", raising=False)
    monkeypatch.setattr(ctypes, "WinError", lambda error: OSError(error, "native failure"), raising=False)
    return api, messages


def service(hotkey="ctrl+space", voice_key="right alt", **callbacks):
    return wh.WindowsHotkeys(hotkey, callbacks.get("on_press", Mock()),
                             callbacks.get("on_release", Mock()), voice_key,
                             callbacks.get("on_voice", Mock()), callbacks.get("log", Mock()))


def key(vk, scan, flags=0):
    return wh.RAWKEYBOARD(MakeCode=scan, VKey=vk, Flags=flags)


@pytest.mark.parametrize("character,vk", [("-", 0xbd), (";", 0xba), (".", 0xbe)])
def test_punctuation_prefers_unmodified_vk_over_scan_codes(native_api, character, vk):
    api, _ = native_api
    assert wh.parse_hotkey("ctrl+" + character) == (wh.MOD_CONTROL | wh.MOD_NOREPEAT, vk)
    api.VkKeyScanW.assert_called_once_with(character)
    api.MapVirtualKeyW.assert_not_called()


@pytest.mark.parametrize("character,vk", [("-", 0xbd), (";", 0xba), (".", 0xbe)])
@pytest.mark.parametrize("shift_state", [None, 0x100, 0x600])
def test_punctuation_fallback_skips_numpad(native_api, character, vk, shift_state):
    api, _ = native_api
    api.VkKeyScanW.side_effect = None
    api.VkKeyScanW.return_value = -1 if shift_state is None else shift_state | vk
    assert wh.parse_hotkey("ctrl+" + character) == (wh.MOD_CONTROL | wh.MOD_NOREPEAT, vk)


@pytest.mark.parametrize("scan", [74, -0x6d])
def test_punctuation_with_only_numpad_route_is_rejected(native_api, scan):
    api, _ = native_api
    api.VkKeyScanW.side_effect = None
    api.VkKeyScanW.return_value = -1
    with patch("keyboard.key_to_scan_codes", return_value=(scan,)), \
            pytest.raises(ValueError, match="unsupported key"):
        wh.parse_hotkey("ctrl+-")
    assert wh.parse_hotkey("ctrl+subtract") == (wh.MOD_CONTROL | wh.MOD_NOREPEAT, 0x6d)


@pytest.mark.parametrize("dedicated", [True, False])
def test_abnt2_slash_never_registers_ctrl_q(native_api, dedicated):
    api, messages = native_api
    api.VkKeyScanW.side_effect = lambda char: 0x600 | ord("Q")
    api.MapVirtualKeyW.side_effect = lambda code, mode: {
        (0x10, wh.MAPVK_VSC_TO_VK_EX): ord("Q"),
        (0x73, wh.MAPVK_VSC_TO_VK_EX): 0xc1,
        (ord("Q"), wh.MAPVK_VK_TO_CHAR): ord("Q"),
        (0xc1, wh.MAPVK_VK_TO_CHAR): ord("/"),
    }.get((code, mode), 0)
    scans = (0x10, 0x73) if dedicated else (0x10,)
    log = Mock()
    with patch("keyboard.key_to_scan_codes", return_value=scans):
        hotkeys = service(hotkey="ctrl+/", voice_key="", log=log)
    api.VkKeyScanW.assert_called_once_with("/")
    messages.put((0, 0))
    hotkeys._run()
    if dedicated:
        api.RegisterHotKey.assert_called_once_with(
            None, wh.HOTKEY_ID, wh.MOD_CONTROL | wh.MOD_NOREPEAT, 0xc1)
    else:
        api.RegisterHotKey.assert_not_called()
        assert "key '/' requires extra modifiers" in log.call_args.args[0]


def test_abnt2_question_mark_rejects_all_modified_routes(native_api):
    api, _ = native_api
    api.VkKeyScanW.side_effect = lambda char: 0x600 | ord("W")
    api.MapVirtualKeyW.side_effect = lambda code, mode: {
        (0x11, wh.MAPVK_VSC_TO_VK_EX): ord("W"),
        (0x73, wh.MAPVK_VSC_TO_VK_EX): 0xc1,
        (ord("W"), wh.MAPVK_VK_TO_CHAR): ord("W"),
        (0xc1, wh.MAPVK_VK_TO_CHAR): ord("/"),
    }.get((code, mode), 0)
    with patch("keyboard.key_to_scan_codes", return_value=(0x11, 0x73)), \
            pytest.raises(ValueError, match=r"key '\?' requires extra modifiers"):
        wh.parse_hotkey("ctrl+?")
    api.RegisterHotKey.assert_not_called()


@pytest.mark.parametrize("character", ["\U0001f600", "\U00010400", "\ud800"])
def test_non_bmp_or_surrogate_key_is_a_value_error(native_api, character):
    api, _ = native_api
    with pytest.raises(ValueError, match="unsupported key.*BMP"):
        wh.parse_hotkey("ctrl+" + character)
    log = Mock()
    hotkeys = service(hotkey="ctrl+" + character, voice_key=character, log=log)
    assert hotkeys._vk is None and hotkeys._solo is None
    assert log.call_count == 2
    api.VkKeyScanW.assert_not_called()


@pytest.mark.parametrize("name,vk", [("scroll lock", 0x91), ("pause", 0x13),
                                   ("num lock", 0x90), ("print screen", 0x2c)])
def test_special_keys_do_not_use_ambiguous_scan_mapping(native_api, name, vk):
    api, _ = native_api
    api.MapVirtualKeyW.side_effect = None
    api.MapVirtualKeyW.return_value = 0x03
    with patch("keyboard.key_to_scan_codes", return_value=(0x46,)) as lookup:
        assert wh.key_to_vk(name) == vk
    lookup.assert_not_called()
    api.MapVirtualKeyW.assert_not_called()


def test_f12_registration_failure_is_logged_without_stopping(native_api):
    api, messages = native_api
    api.RegisterHotKey.return_value = False
    messages.put((0, 0))
    log = Mock()
    hotkeys = service(hotkey="f12", log=log)
    log.assert_not_called()
    hotkeys._run()
    api.RegisterHotKey.assert_called_once_with(None, wh.HOTKEY_ID, wh.MOD_NOREPEAT, 0x7b)
    assert any("RegisterHotKey failed" in entry.args[0] and "F12" in entry.args[0]
               for entry in log.call_args_list)
    assert any("Voice chat hotkey" in entry.args[0] for entry in log.call_args_list)
    assert not hotkeys._registered and hotkeys._ready.is_set()


def test_owner_thread_registers_dispatches_and_unregisters(native_api):
    api, messages = native_api
    delivered = threading.Event()
    pressed = Mock(side_effect=delivered.set)
    owner_threads = []
    api.RegisterHotKey.side_effect = lambda *args: owner_threads.append(threading.get_ident()) or True
    api.UnregisterHotKey.side_effect = lambda *args: owner_threads.append(threading.get_ident()) or True
    hotkeys = service(on_press=pressed)
    try:
        hotkeys.start()
        api.RegisterHotKey.assert_called_once_with(None, wh.HOTKEY_ID,
                                                   wh.MOD_CONTROL | wh.MOD_NOREPEAT, 0x20)
        messages.put((wh.WM_HOTKEY, wh.HOTKEY_ID))
        assert delivered.wait(2)
        pressed.assert_called_once_with()
    finally:
        hotkeys.stop()
    assert owner_threads == [hotkeys._thread.ident, hotkeys._thread.ident]
    assert owner_threads[0] != threading.get_ident()
    assert not hotkeys._thread.is_alive()
    api.UnregisterHotKey.assert_called_once_with(None, wh.HOTKEY_ID)
    api.DestroyWindow.assert_called_once_with(123)
    api.UnregisterClassW.assert_called_once()
    assert api.RegisterRawInputDevices.call_count == 2
    registration, removal = [entry.args[0]._obj for entry in api.RegisterRawInputDevices.call_args_list]
    assert (registration.usUsagePage, registration.usUsage, registration.dwFlags,
            registration.hwndTarget) == (1, 6, wh.RIDEV_INPUTSINK, 123)
    assert removal.dwFlags == wh.RIDEV_REMOVE and removal.hwndTarget is None
    hotkeys.stop()  # idempotent, no second unregister
    api.UnregisterHotKey.assert_called_once()


def test_wm_hotkey_enqueues_toggle_once(native_api):
    app = de.DictationApp({**de.DEFAULT_CONFIG, "beep_on_start": False})
    app.stop_monitoring()  # Inspect the action queue without a racing consumer.
    app._stopping.clear()
    hotkeys = service(on_press=app.toggle_recording)
    hotkeys._registered = True
    try:
        hotkeys._dispatch_hotkey(999)
        assert app._hotkey_actions.empty()
        hotkeys._dispatch_hotkey(wh.HOTKEY_ID)
        hotkeys._dispatch_hotkey(wh.HOTKEY_ID)  # engine also guards a held press
        callback, args = app._hotkey_actions.get_nowait()
        assert callback == app._handle_hotkey_press and len(args) == 2
        assert app._hotkey_actions.empty()
        hotkeys._stopping.set()
        app._hotkey_held = False
        hotkeys._dispatch_hotkey(wh.HOTKEY_ID)
        assert app._hotkey_actions.empty()
    finally:
        app.stop_monitoring()


@pytest.mark.parametrize("vk,scan,flags", [(0x20, 57, 0), (0x11, 29, 0), (0x11, 29, wh.RI_KEY_E0)])
def test_raw_release_ends_push_to_talk(native_api, vk, scan, flags):
    app = de.DictationApp({**de.DEFAULT_CONFIG, "beep_on_start": False})
    hotkeys = service(on_release=app._release_hotkey)
    app._native_hotkeys = hotkeys
    app._hotkey_held = True
    starter = Mock()
    try:
        with patch.object(de.time, "time", return_value=101.0):
            hotkeys._handle_key(key(vk, scan, flags | wh.RI_KEY_BREAK))
        assert app._hotkey_released_at == 101.0
        with patch.object(app, "_finish_recording") as finish, \
                patch.object(app, "_begin_continuous") as begin:
            app._watch_key(100.0, starter)
        starter.join.assert_called_once()
        finish.assert_called_once_with()
        begin.assert_not_called()
        assert not app._hotkey_held
    finally:
        app.stop_monitoring()


@pytest.mark.parametrize("held", [{0x20, 0xa2}, {0x20, 0xa3}, {0x20}, {0xa2}, set()])
def test_release_poll_uses_physical_modifier_sides(native_api, held):
    api, _ = native_api
    api.GetAsyncKeyState.side_effect = lambda vk: 0x8000 if vk in held else 0
    hotkeys = service()
    hotkeys._registered = True
    assert hotkeys.is_pressed() == (0x20 in held and bool(held & {0xa2, 0xa3}))


def test_poll_release_settles_hold_without_keyboard_listener(native_api):
    app = de.DictationApp({**de.DEFAULT_CONFIG, "beep_on_start": False})
    app._native_hotkeys = service()
    app._native_hotkeys._registered = True
    app._hotkey_held = True
    try:
        with patch.object(de.time, "time", return_value=101.0), \
                patch.object(app, "_finish_recording") as finish, \
                patch("keyboard.is_pressed") as old_poll:
            app._watch_key(100.0, Mock())
        finish.assert_called_once_with()
        old_poll.assert_not_called()
    finally:
        app.stop_monitoring()


@pytest.mark.parametrize("released_vk,scan", [(0x20, 57), (0x11, 29)])
def test_device_less_release_is_settled_by_polling(native_api, released_vk, scan):
    api, _ = native_api
    app = de.DictationApp({**de.DEFAULT_CONFIG, "beep_on_start": False})
    app.stop_monitoring()
    app._stopping.clear()
    hotkeys = service(on_release=app._release_hotkey)
    app._native_hotkeys = hotkeys
    hotkeys._registered = True
    app._hotkey_held = True
    held = {0x20, 0xa2}
    api.GetAsyncKeyState.side_effect = lambda vk: 0x8000 if vk in held else 0
    event = key(released_vk, scan, wh.RI_KEY_BREAK)
    header = wh.RAWINPUTHEADER(dwType=1, hDevice=0,
                               dwSize=ctypes.sizeof(wh.RAWINPUTHEADER) + ctypes.sizeof(event))
    data = bytes(header) + bytes(event)

    def read(handle, command, buffer, size, header_size):
        size._obj.value = len(data)
        if buffer is None:
            return 0
        ctypes.memmove(buffer, data, len(data))
        return len(data)

    def release(timeout):
        assert timeout == 0.05
        held.remove(0xa2 if released_vk == 0x11 else released_vk)
        hotkeys._window_proc(123, wh.WM_INPUT, 1, 77)
        assert app._hotkey_released_at is None  # Raw Input was discarded.
        return False

    api.GetRawInputData.side_effect = read
    starter = Mock()
    try:
        with patch.object(app._stopping, "wait", side_effect=release) as wait, \
                patch.object(de.time, "time", return_value=101.0), \
                patch.object(app, "_finish_recording") as finish, \
                patch("keyboard.is_pressed") as old_poll:
            app._watch_key(100.0, starter)
        wait.assert_called_once_with(0.05)
        starter.join.assert_called_once_with()
        finish.assert_called_once_with()
        old_poll.assert_not_called()
        assert not app._hotkey_held
    finally:
        app.stop_monitoring()


def test_unrelated_raw_release_does_not_end_a_hold(native_api):
    released = Mock()
    hotkeys = service(on_release=released)
    hotkeys._handle_key(key(ord("A"), 30, wh.RI_KEY_BREAK))
    hotkeys._handle_key(key(0x20, 57))
    released.assert_not_called()


def test_raw_tap_keeps_tap_action_after_delayed_audio_start(native_api):
    app = de.DictationApp({**de.DEFAULT_CONFIG, "beep_on_start": False})
    app._hotkey_held = True
    app._native_hotkeys = service(on_release=app._release_hotkey)
    try:
        with patch.object(de.time, "time", return_value=100.1):
            app._native_hotkeys._handle_key(key(0x20, 57, wh.RI_KEY_BREAK))
        with patch.object(de.time, "time", return_value=105.0), \
                patch.object(app, "_finish_recording") as finish, \
                patch.object(app, "_begin_continuous") as begin:
            app._watch_key(100.0, Mock())
        begin.assert_called_once_with(100.0)
        finish.assert_not_called()
    finally:
        app.stop_monitoring()


@pytest.mark.parametrize("voice_key", ["right alt", "alt gr"])
@pytest.mark.parametrize("chord", ["solo", "q", "ctrl", "right ctrl"])
@pytest.mark.parametrize("fake_ctrl", [False, True])
@pytest.mark.parametrize("ctrl_break_first", [False, True])
def test_raw_events_feed_solo_tap_without_suppressing_altgr(
        native_api, voice_key, chord, fake_ctrl, ctrl_break_first):
    tapped = Mock()
    hotkeys = service(voice_key=voice_key, on_voice=tapped)
    ctrl_flags = wh.RI_KEY_E0 if chord == "right ctrl" else 0
    if chord in ("ctrl", "right ctrl"):
        hotkeys._handle_key(key(0x11, 29, ctrl_flags), 90, 1)
    if fake_ctrl:
        hotkeys._handle_key(key(0x11, 29), 100, 1)
    hotkeys._handle_key(key(0x12, 56, wh.RI_KEY_E0), 100, 1)
    hotkeys._handle_key(key(0x12, 56, wh.RI_KEY_E0), 110, 1)  # repeat
    if chord == "q":
        hotkeys._handle_key(key(ord("Q"), 16))
        hotkeys._handle_key(key(ord("Q"), 16, wh.RI_KEY_BREAK))
    ctrl_up = key(0x11, 29, wh.RI_KEY_BREAK)
    if fake_ctrl and ctrl_break_first:
        hotkeys._handle_key(ctrl_up, 150, 1)
    hotkeys._handle_key(key(0x12, 56, wh.RI_KEY_E0 | wh.RI_KEY_BREAK), 150, 1)
    if fake_ctrl and not ctrl_break_first:
        hotkeys._handle_key(ctrl_up, 150, 1)
    if chord in ("ctrl", "right ctrl"):
        hotkeys._handle_key(key(0x11, 29, ctrl_flags | wh.RI_KEY_BREAK), 160, 1)
    assert tapped.call_count == (1 if chord == "solo" else 0)
    # A later solo tap must not inherit the synthetic Ctrl's held state.
    hotkeys._handle_key(key(0x12, 56, wh.RI_KEY_E0), 200, 1)
    hotkeys._handle_key(key(0x12, 56, wh.RI_KEY_E0 | wh.RI_KEY_BREAK), 250, 1)
    assert tapped.call_count == (2 if chord == "solo" else 1)


def test_wm_input_decodes_extended_key_and_calls_default_proc(native_api):
    api, _ = native_api
    tapped = Mock()
    hotkeys = service(on_voice=tapped)
    data = b""

    def read(handle, command, buffer, size, header_size):
        assert handle == 77 and command == wh.RID_INPUT
        assert header_size == ctypes.sizeof(wh.RAWINPUTHEADER)
        size._obj.value = len(data)
        if buffer is None:
            return 0
        ctypes.memmove(buffer, data, len(data))
        return len(data)

    api.GetRawInputData.side_effect = read
    for flags in (wh.RI_KEY_E0, wh.RI_KEY_E0 | wh.RI_KEY_BREAK):
        event = key(0x12, 56, flags)
        header = wh.RAWINPUTHEADER(dwType=1, hDevice=1,
                                   dwSize=ctypes.sizeof(wh.RAWINPUTHEADER) + ctypes.sizeof(event))
        data = bytes(header) + bytes(event)
        hotkeys._window_proc(123, wh.WM_INPUT, 1, 77)
    tapped.assert_called_once_with()
    assert api.DefWindowProcW.call_args_list == [call(123, wh.WM_INPUT, 1, 77)] * 2


def test_raw_read_failure_is_rate_limited_without_event_data(native_api):
    api, _ = native_api
    log = Mock()
    hotkeys = service(log=log)
    api.GetRawInputData.side_effect = RuntimeError("private event data")
    with patch.object(wh.time, "monotonic", side_effect=[10, 11, 16]):
        for _ in range(3):
            hotkeys._window_proc(123, wh.WM_INPUT, 1, 77)
    assert log.call_args_list == [
        call("Windows Raw Input read failed; release polling remains active.")] * 2
    assert api.DefWindowProcW.call_count == 3


@pytest.mark.parametrize("voice_key", ["menu", "apps"])
def test_menu_voice_key_taps(native_api, voice_key):
    tapped = Mock()
    hotkeys = service(voice_key=voice_key, on_voice=tapped)
    hotkeys._handle_key(key(0x5d, 0x5d, wh.RI_KEY_E0))
    hotkeys._handle_key(key(0x5d, 0x5d, wh.RI_KEY_E0 | wh.RI_KEY_BREAK))
    tapped.assert_called_once_with()


def test_key_name_lookup_uses_keyboard_table_without_hooks(native_api):
    api, _ = native_api
    with patch("keyboard.key_to_scan_codes", return_value=(0x20,)) as lookup, \
            patch("keyboard.hook") as hook:
        api.MapVirtualKeyW.return_value = ord("D")
        api.MapVirtualKeyW.side_effect = None
        assert wh.parse_hotkey("ctrl+some layout name") == (wh.MOD_CONTROL | wh.MOD_NOREPEAT, ord("D"))
    lookup.assert_called_once_with("some layout name", error_if_missing=False)
    api.MapVirtualKeyW.assert_called_once_with(0x20, 3)
    hook.assert_not_called()


@pytest.mark.parametrize("scan", [0x1d, 0x21d])
def test_altgr_pair_requires_same_device_and_timestamp(native_api, scan):
    for ctrl_time, ctrl_device in ((99, 1), (100, 2)):
        tapped = Mock()
        hotkeys = service(on_voice=tapped)
        hotkeys._handle_key(key(0x11, scan), ctrl_time, ctrl_device)
        hotkeys._handle_key(key(0x12, 56, wh.RI_KEY_E0), 100, 1)
        hotkeys._handle_key(key(0x12, 56, wh.RI_KEY_E0 | wh.RI_KEY_BREAK), 150, 1)
        tapped.assert_not_called()


@pytest.mark.parametrize("chord", ["solo", "q"])
def test_altgr_synthetic_ctrl_with_altgr_scan_code_is_ignored(native_api, chord):
    # Raw Input as recorded on an ABNT2 keyboard: VK_CONTROL with E0 38.
    tapped, released = Mock(), Mock()
    hotkeys = service(on_voice=tapped, on_release=released)
    hotkeys._handle_key(key(0x11, 0x38, wh.RI_KEY_E0), 100, 1)
    hotkeys._handle_key(key(0x12, 0x38, wh.RI_KEY_E0), 100, 1)
    if chord == "q":
        hotkeys._handle_key(key(ord("Q"), 16), 120, 1)
        hotkeys._handle_key(key(ord("Q"), 16, wh.RI_KEY_BREAK), 130, 1)
    hotkeys._handle_key(key(0x11, 0x38, wh.RI_KEY_E0 | wh.RI_KEY_BREAK), 150, 1)
    hotkeys._handle_key(key(0x12, 0x38, wh.RI_KEY_E0 | wh.RI_KEY_BREAK), 150, 1)
    assert tapped.call_count == (1 if chord == "solo" else 0)
    released.assert_not_called()


def test_intervening_key_prevents_altgr_pair_detection(native_api):
    tapped = Mock()
    hotkeys = service(on_voice=tapped)
    hotkeys._handle_key(key(0x11, 29), 100, 1)
    hotkeys._handle_key(key(ord("Q"), 16), 100, 1)
    hotkeys._handle_key(key(ord("Q"), 16, wh.RI_KEY_BREAK), 100, 1)
    hotkeys._handle_key(key(0x12, 56, wh.RI_KEY_E0), 100, 1)
    hotkeys._handle_key(key(0x12, 56, wh.RI_KEY_E0 | wh.RI_KEY_BREAK), 150, 1)
    tapped.assert_not_called()


def test_left_ctrl_voice_key_keeps_original_hold_duration(native_api):
    tapped = Mock()
    hotkeys = service(voice_key="left ctrl", on_voice=tapped)
    hotkeys._solo._clock = Mock(side_effect=[10, 11])
    hotkeys._handle_key(key(0x11, 29), 100, 1)
    hotkeys._handle_key(key(0x11, 29, wh.RI_KEY_BREAK), 1100, 1)
    tapped.assert_not_called()


def test_injected_raw_input_does_not_spoil_solo_or_release_hotkey(native_api):
    api, _ = native_api
    tapped, released = Mock(), Mock()
    hotkeys = service(on_voice=tapped, on_release=released)
    hotkeys._handle_key(key(0x12, 56, wh.RI_KEY_E0))
    data = b""

    def read(handle, command, buffer, size, header_size):
        size._obj.value = len(data)
        if buffer is None:
            return 0
        ctypes.memmove(buffer, data, len(data))
        return len(data)

    api.GetRawInputData.side_effect = read
    for event in (key(ord("Q"), 16), key(0x20, 57, wh.RI_KEY_BREAK)):
        header = wh.RAWINPUTHEADER(dwType=1, hDevice=0,
                                   dwSize=ctypes.sizeof(wh.RAWINPUTHEADER) + ctypes.sizeof(event))
        data = bytes(header) + bytes(event)
        hotkeys._window_proc(123, wh.WM_INPUT, 1, 77)
    hotkeys._handle_key(key(0x12, 56, wh.RI_KEY_E0 | wh.RI_KEY_BREAK))
    tapped.assert_called_once_with()
    released.assert_not_called()
    assert api.DefWindowProcW.call_count == 2


def test_raw_callback_errors_are_separate_and_rate_limited(native_api):
    log = Mock()
    released = Mock(side_effect=RuntimeError("private event data"))
    hotkeys = service(on_release=released, log=log)
    event = key(0x20, 57, wh.RI_KEY_BREAK)
    with patch.object(hotkeys, "_read_input", return_value=(event, 1)), \
            patch.object(wh.time, "monotonic", side_effect=[10, 11, 16]):
        for _ in range(3):
            hotkeys._window_proc(123, wh.WM_INPUT, 1, 77)
    assert released.call_count == 3
    assert log.call_args_list == [
        call("Windows release callback failed; hotkeys remain active.")] * 2


def test_voice_callback_error_does_not_disable_later_taps(native_api):
    tapped = Mock(side_effect=[RuntimeError("private event data"), None])
    log = Mock()
    hotkeys = service(on_voice=tapped, log=log)
    for _ in range(2):
        hotkeys._handle_key(key(0x12, 56, wh.RI_KEY_E0))
        hotkeys._handle_key(key(0x12, 56, wh.RI_KEY_E0 | wh.RI_KEY_BREAK))
    assert tapped.call_count == 2
    log.assert_called_once_with("Windows voice callback failed; hotkeys remain active.")


def test_wm_hotkey_callback_failure_keeps_both_hotkeys_alive(native_api):
    api, messages = native_api
    pressed = Mock(side_effect=[RuntimeError("private event data"), None])
    tapped, log = Mock(), Mock()
    hotkeys = service(on_press=pressed, on_voice=tapped, log=log)
    events = iter([key(0x12, 56, wh.RI_KEY_E0),
                   key(0x12, 56, wh.RI_KEY_E0 | wh.RI_KEY_BREAK)])
    api.DispatchMessageW.side_effect = lambda pointer: hotkeys._handle_key(next(events))
    for message in (wh.WM_HOTKEY, wh.WM_HOTKEY, wh.WM_INPUT, wh.WM_INPUT, 0):
        messages.put((message, wh.HOTKEY_ID))
    hotkeys._run()
    assert pressed.call_count == 2
    tapped.assert_called_once_with()
    assert call("Windows hotkey callback failed; hotkeys remain active.") in log.call_args_list
    assert not any("stopped:" in entry.args[0] for entry in log.call_args_list)
    api.UnregisterHotKey.assert_called_once()


def test_native_press_timestamp_is_scoped_to_owner_callback(native_api):
    hotkeys = service()
    hotkeys._registered = True
    seen = []

    def pressed():
        seen.append(hotkeys.press_time)
        other = threading.Thread(target=lambda: seen.append(hotkeys.press_time))
        other.start()
        other.join(2)
        assert not other.is_alive()

    hotkeys._on_press = pressed
    with patch.object(wh.time, "time", return_value=100.0):
        hotkeys._dispatch_hotkey(wh.HOTKEY_ID)
    assert seen == [100.0, None]
    assert hotkeys.press_time is None


@pytest.mark.parametrize("release_time,expected_release", [(100.1, 100.1), (99.0, None)])
def test_release_before_wm_hotkey_survives_worker_delay(native_api, release_time, expected_release):
    app = de.DictationApp({**de.DEFAULT_CONFIG, "beep_on_start": False})
    app.stop_monitoring()
    app._stopping.clear()
    hotkeys = service(on_press=app.toggle_recording, on_release=app._release_hotkey)
    app._native_hotkeys = hotkeys
    hotkeys._registered = True
    try:
        with patch.object(de.time, "time", return_value=release_time):
            hotkeys._handle_key(key(0x20, 57, wh.RI_KEY_BREAK))
        assert app._hotkey_last_release == release_time
        # Message was queued at t=100, but dispatched after the release.
        wh._kernel32.GetTickCount.return_value = 6000
        with patch.object(de.time, "time", return_value=105.0):
            hotkeys._dispatch_hotkey(wh.HOTKEY_ID)
        assert hotkeys.press_time is None
        callback, args = app._hotkey_actions.get_nowait()
        assert callback == app._handle_hotkey_press and args[0] == 100.0
        assert app._hotkey_released_at == expected_release
        if expected_release is not None:
            with patch.object(de.time, "time", return_value=110.0), \
                    patch.object(app, "_begin_continuous") as begin, \
                    patch.object(app, "_finish_recording") as finish:
                app._watch_key(args[0], Mock())
            begin.assert_called_once_with(100.0)
            finish.assert_not_called()
    finally:
        app.stop_monitoring()


def test_stop_finishes_teardown_before_logging_hotkey_timeout(native_api):
    app = de.DictationApp({**de.DEFAULT_CONFIG, "beep_on_start": False})
    app._native_hotkeys = Mock()
    app._native_hotkeys.stop.side_effect = RuntimeError("Windows hotkey thread did not stop")
    calls = Mock()
    app.recorder = calls.recorder
    app.chimes = calls.chimes
    app.is_recording = True
    try:
        with patch.object(app, "_interrupt_voice_reply", calls.interrupt), \
                patch.object(de.platforms, "current", return_value=calls.platform), \
                patch.object(de, "log", calls.log):
            app.stop()
        assert not app.is_recording and app._stopping.is_set()
        assert not app._hotkey_worker.is_alive()
        assert calls.mock_calls == [
            call.interrupt(), call.recorder.close(), call.chimes.close(), call.platform.close(),
            call.log("Error stopping hotkeys: Windows hotkey thread did not stop"),
            call.log("Engine stopped."),
        ]
    finally:
        app.stop_monitoring()


@pytest.fixture(params=["debora_whisper.app", "debora_whisper.dictation_engine"])
def entrypoint(request, native_api, monkeypatch, tmp_path):
    module = importlib.import_module(request.param)
    provider = WindowsPlatform()
    monkeypatch.setattr(provider, "handle_termination", Mock())
    monkeypatch.setattr(de.platforms, "current", lambda: provider)
    monkeypatch.setattr(de, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(sys, "argv", ["debora", "--device", "CPU"])
    for name in ("log_folder_moves", "rotate_logs", "log_hardware_inventory", "avoid_lost_npu"):
        monkeypatch.setattr(module, name, Mock(return_value=False))
    factory = Mock()
    if request.param.endswith(".app"):
        monkeypatch.setattr(module, "_claim_single_instance", lambda: True)
        monkeypatch.setattr(module, "GUIApp", factory)
    else:
        monkeypatch.setattr(module, "DictationApp", factory)
    return module, factory


@pytest.mark.parametrize("hotkey", ["right ctrl", "ctrl+a, b", "ctrl+not-a-key", "f12",
                                   "ctrl+\U0001f600"])
def test_saved_unregistrable_hotkey_does_not_block_startup(entrypoint, hotkey):
    module, factory = entrypoint
    de.CONFIG_FILE.write_text(json.dumps({**de.DEFAULT_CONFIG, "hotkey": hotkey}), encoding="utf-8")
    module.main()
    assert factory.call_args.args[0]["hotkey"] == hotkey
    factory.return_value.run.assert_called_once_with()


@pytest.mark.parametrize("hotkey", ["right ctrl", "ctrl+a, b", "", "ctrl+\U0001f600"])
def test_explicit_cli_hotkey_is_validated(entrypoint, hotkey, monkeypatch, capsys):
    module, factory = entrypoint
    monkeypatch.setattr(sys, "argv", ["debora", "--hotkey", hotkey])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert "Invalid hotkey" in capsys.readouterr().err
    factory.assert_not_called()


def test_cli_f12_is_left_to_registration(entrypoint, monkeypatch):
    module, factory = entrypoint
    monkeypatch.setattr(sys, "argv", ["debora", "--device", "CPU", "--hotkey", "f12"])
    module.main()
    assert factory.call_args.args[0]["hotkey"] == "f12"
    factory.return_value.run.assert_called_once_with()


@pytest.mark.parametrize("hotkey", ["right ctrl", "ctrl+a, b", "ctrl+not-a-key"])
def test_unmappable_hotkey_logs_to_app_log_and_background_start_continues(
        native_api, monkeypatch, tmp_path, hotkey):
    api, _ = native_api
    provider, keyboard = WindowsPlatform(), Mock()
    monkeypatch.setattr(provider, "keyboard", lambda: keyboard)
    monkeypatch.setattr(de.platforms, "current", lambda: provider)
    monkeypatch.setattr(de, "LOG_DIR", tmp_path)
    monkeypatch.setattr(de, "LOG_FILE", tmp_path / "app.log")
    config = {**de.DEFAULT_CONFIG, "hotkey": hotkey, "voice_chat": False, "beep_on_start": False}
    de.validate_config(config)
    app = de.DictationApp(config)
    try:
        with patch.object(app, "_start_loader") as loader, \
                patch.object(app, "_start_segment_consumer") as consumer:
            app.start_background()
        loader.assert_called_once_with()
        consumer.assert_called_once_with()
        assert app._native_hotkeys._thread.is_alive()
        assert app._native_hotkeys._vk is None
        assert f"Hotkey {hotkey!r} disabled:" in de.LOG_FILE.read_text(encoding="utf-8")
        api.RegisterHotKey.assert_not_called()
        keyboard.hook.assert_not_called()
    finally:
        app.stop()


@pytest.mark.parametrize("hotkey", ["ctrl+not-a-key", "ctrl+\U0001f600"])
def test_settings_shows_unmappable_hotkey_before_saving(native_api, monkeypatch, hotkey):
    from debora_whisper.ui.settings import SettingsWindow
    monkeypatch.setattr(de.platforms, "current", lambda: WindowsPlatform())
    window = SettingsWindow.__new__(SettingsWindow)
    original = dict(de.DEFAULT_CONFIG)
    window._config = original
    window._get_new_config = Mock(return_value={**original, "hotkey": hotkey})
    window._on_apply = Mock()
    window.update_status = Mock()
    window._apply()
    assert window._config is original
    window._on_apply.assert_not_called()
    message, color = window.update_status.call_args.args
    assert "Invalid hotkey" in message and "unsupported key" in message
    assert color == "#FF453A"


def test_invalid_voice_key_leaves_main_hotkey_available(native_api):
    api, messages = native_api
    messages.put((0, 0))
    log = Mock()
    hotkeys = service(voice_key="not-a-key", log=log)
    hotkeys._run()
    assert hotkeys._solo is None
    api.RegisterHotKey.assert_called_once()
    assert any("Voice chat hotkey 'not-a-key' disabled" in entry.args[0]
               for entry in log.call_args_list)


@pytest.mark.parametrize("failure", ["parse", "conflict", "raw", "window", "class", "get"])
def test_failures_log_and_release_owned_resources(native_api, failure):
    api, messages = native_api
    log = Mock()
    if failure == "conflict":
        api.RegisterHotKey.return_value = False
    elif failure == "raw":
        api.RegisterRawInputDevices.return_value = False
    elif failure == "window":
        api.CreateWindowExW.return_value = None
    elif failure == "class":
        api.RegisterClassW.return_value = 0
    messages.put((-1 if failure == "get" else 0, 0))
    configured = "ctrl" if failure == "parse" else "ctrl+space"
    hotkeys = service(hotkey=configured, voice_key="", log=log)
    hotkeys._run()
    lines = "\n".join(c.args[0] for c in log.call_args_list)
    assert configured in lines
    assert "disabled" in lines or "stopped" in lines
    if failure == "conflict":
        assert "1409" in lines and "another application" in lines
    assert not hotkeys._registered and hotkeys._ready.is_set()
    assert api.UnregisterHotKey.call_count == (1 if failure == "get" else 0)
    assert api.DestroyWindow.call_count == (0 if failure in ("class", "window") else 1)
    assert api.UnregisterClassW.call_count == (0 if failure == "class" else 1)


def test_settings_rebuild_unregisters_before_registering_new_hotkey(native_api, monkeypatch):
    api, _ = native_api
    provider, keyboard = WindowsPlatform(), Mock()
    monkeypatch.setattr(provider, "keyboard", lambda: keyboard)
    monkeypatch.setattr(de.platforms, "current", lambda: provider)
    config = {**de.DEFAULT_CONFIG, "beep_on_start": False, "voice_chat_hotkey": ""}
    first = de.DictationApp(dict(config))
    second = de.DictationApp({**config, "hotkey": "ctrl+shift+d"})
    try:
        first._start_hotkeys(keyboard)
        assert first.stop_if_idle() is None
        second._start_hotkeys(keyboard)
        second._start_hotkeys(keyboard)
        relevant = [entry for entry in api.mock_calls if entry[0] in ("RegisterHotKey", "UnregisterHotKey")]
        assert relevant == [
            call.RegisterHotKey(None, wh.HOTKEY_ID, wh.MOD_CONTROL | wh.MOD_NOREPEAT, 0x20),
            call.UnregisterHotKey(None, wh.HOTKEY_ID),
            call.RegisterHotKey(None, wh.HOTKEY_ID,
                                wh.MOD_CONTROL | wh.MOD_SHIFT | wh.MOD_NOREPEAT, ord("D")),
        ]
        keyboard.add_hotkey.assert_not_called()
        keyboard.hook.assert_not_called()
        keyboard.is_pressed.assert_not_called()
        keyboard.unhook_all.assert_not_called()
    finally:
        first.stop()
        second.stop()
    assert not first._native_hotkeys._thread.is_alive()
    assert not second._native_hotkeys._thread.is_alive()
