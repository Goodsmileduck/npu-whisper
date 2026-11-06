"""Right Alt tapped alone switches voice chat; AltGr combos keep typing."""
import time
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from debora_whisper.dictation_engine import DEFAULT_CONFIG, DictationApp, validate_config
from debora_whisper.solo_key import SoloKeyTap


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


# Scan codes for the event names used below.
CODES = {"right alt": 56, "alt gr": 541, "ctrl": 29, "q": 16, "/": 16, "windows": 91}


def _tap(key="right alt"):
    clock, taps = Clock(), []
    tap = SoloKeyTap(key, lambda: taps.append(clock.now), clock=clock)

    def send(name, kind, after=0.0):
        clock.now += after
        tap.handle(SimpleNamespace(name=name, event_type=kind, scan_code=CODES[name]))
    return send, taps


@pytest.mark.parametrize("name", ["right alt", "alt gr"])
def test_tap_alone_fires_for_both_layout_names(name):
    send, taps = _tap()
    send(name, "down")
    send(name, "up", after=0.1)
    assert len(taps) == 1


def test_altgr_combo_does_not_fire():
    send, taps = _tap()
    send("alt gr", "down")
    send("q", "down", after=0.05)
    send("q", "up", after=0.05)
    send("alt gr", "up", after=0.05)
    assert taps == []


def test_key_held_before_does_not_fire():
    send, taps = _tap()
    send("ctrl", "down")
    send("right alt", "down", after=0.1)
    send("right alt", "up", after=0.1)
    assert taps == []


def test_released_key_no_longer_blocks():
    send, taps = _tap()
    send("ctrl", "down")
    send("ctrl", "up", after=0.1)
    send("right alt", "down", after=0.1)
    send("right alt", "up", after=0.1)
    assert len(taps) == 1


def test_long_held_modifier_still_blocks():
    send, taps = _tap()
    send("ctrl", "down")
    for _ in range(10):
        send("ctrl", "down", after=0.5)  # auto-repeat while held
    send("right alt", "down")
    send("right alt", "up", after=0.1)
    assert taps == []


def test_key_name_changing_between_down_and_up_does_not_stick():
    send, taps = _tap()
    send("alt gr", "down")
    send("/", "down", after=0.05)  # AltGr+Q
    send("alt gr", "up", after=0.05)
    send("q", "up", after=0.05)    # same key, named without AltGr
    send("alt gr", "down", after=0.1)
    send("alt gr", "up", after=0.1)
    assert len(taps) == 1


def test_missed_key_up_blocks_taps_only_until_stale():
    send, taps = _tap()
    send("windows", "down")  # its up event lost behind the lock screen
    send("right alt", "down", after=0.1)
    send("right alt", "up", after=0.1)
    assert taps == []
    send("right alt", "down", after=2.5)
    send("right alt", "up", after=0.1)
    assert len(taps) == 1


def test_ctrl_held_blocks_repeated_taps():
    send, taps = _tap()
    send("ctrl", "down")
    for _ in range(3):
        send("ctrl", "down", after=0.5)  # auto-repeat while held
        send("right alt", "down", after=0.1)
        send("right alt", "up", after=0.1)
    assert taps == []
    send("ctrl", "up", after=0.1)
    send("right alt", "down", after=0.1)
    send("right alt", "up", after=0.1)
    assert len(taps) == 1


def test_long_hold_does_not_fire_and_autorepeat_keeps_first_press():
    send, taps = _tap()
    send("right alt", "down")
    for _ in range(10):
        send("right alt", "down", after=0.1)  # auto-repeat
    send("right alt", "up", after=0.1)
    assert taps == []


def test_up_without_down_is_ignored():
    send, taps = _tap()
    send("right alt", "up")
    assert taps == []


@pytest.mark.parametrize("key", ["right alt", "alt gr"])
def test_engine_hotkey_calls_app_callback(key):
    app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False,
                        "voice_chat_hotkey": key})
    called = threading.Event()
    app.on_voice_chat_toggle = MagicMock(side_effect=called.set)
    keyboard = MagicMock()
    app._register_voice_chat_hotkey(keyboard)
    keyboard.key_to_scan_codes.assert_called_once_with(key)
    handler = keyboard.hook.call_args.args[0]
    handler(SimpleNamespace(name="alt gr", event_type="down", scan_code=541))
    handler(SimpleNamespace(name="alt gr", event_type="up", scan_code=541))
    assert called.wait(2)
    app.on_voice_chat_toggle.assert_called_once()


def test_solo_hook_does_not_wait_for_ui():
    app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False})
    keyboard = MagicMock()
    entered, release = threading.Event(), threading.Event()
    callback_threads = []

    def slow_ui():
        callback_threads.append(threading.get_ident())
        entered.set()
        assert release.wait(2)

    app.on_voice_chat_toggle = slow_ui
    app._register_voice_chat_hotkey(keyboard)
    handler = keyboard.hook.call_args.args[0]
    try:
        handler(SimpleNamespace(name="alt gr", event_type="down", scan_code=541))
        handler(SimpleNamespace(name="alt gr", event_type="up", scan_code=541))
        assert entered.wait(2)
        assert callback_threads == [app._hotkey_worker.ident]
        assert callback_threads != [threading.get_ident()]
    finally:
        release.set()
        app.stop_monitoring()


def test_engine_unknown_hotkey_is_disabled():
    app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False,
                        "voice_chat_hotkey": "unknown key"})
    keyboard = MagicMock()
    keyboard.key_to_scan_codes.side_effect = ValueError("Unknown key")
    with patch("debora_whisper.dictation_engine.log") as log:
        app._register_voice_chat_hotkey(keyboard)
    keyboard.key_to_scan_codes.assert_called_once_with("unknown key")
    keyboard.hook.assert_not_called()
    log.assert_called_once_with(
        "Voice chat hotkey 'unknown key' is not a key; voice chat hotkey disabled.")


def test_engine_hotkey_disabled_when_empty():
    app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False,
                        "voice_chat_hotkey": ""})
    keyboard = MagicMock()
    app._register_voice_chat_hotkey(keyboard)
    keyboard.hook.assert_not_called()


def test_engine_hotkey_without_app_switches_engine():
    app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False, "voice_chat": False})
    with patch.object(app, "set_voice_chat") as set_voice_chat:
        app._on_voice_chat_hotkey()
        for _ in range(200):
            if set_voice_chat.called:
                break
            time.sleep(0.005)
    set_voice_chat.assert_called_once_with(True, True)


@pytest.mark.parametrize("key", ["ctrl+space", "a,b"])
def test_config_rejects_multi_key_hotkey(key):
    with pytest.raises(ValueError, match='must be a single key or "" to disable'):
        validate_config({**DEFAULT_CONFIG, "voice_chat_hotkey": key})


@pytest.mark.parametrize("key", ["right alt", "alt gr", ""])
def test_config_accepts_single_key_hotkey_or_disabled(key):
    validate_config({**DEFAULT_CONFIG, "voice_chat_hotkey": key})


def test_config_rejects_non_string_hotkey():
    with pytest.raises(ValueError, match="voice_chat_hotkey"):
        validate_config({**DEFAULT_CONFIG, "voice_chat_hotkey": 1})


def _listening_app(**state):
    app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False})
    for key, value in state.items():
        setattr(app, key, value)
    return app


def test_mode_switch_starts_continuous_listening_when_idle():
    app = _listening_app()

    def start(_perf):
        app.is_recording = True

    with patch.object(app, "_start_recording", side_effect=start) as start_recording, \
            patch.object(app, "_begin_continuous") as begin:
        app.start_listening()
    start_recording.assert_called_once()
    begin.assert_called_once()


@pytest.mark.parametrize("state", [{"_continuous": True}, {"is_recording": True},
                                   {"_transcribing": True}, {"_hotkey_held": True}])
def test_mode_switch_leaves_a_busy_microphone_alone(state):
    app = _listening_app(**state)
    with patch.object(app, "_start_recording") as start_recording:
        app.start_listening()
    start_recording.assert_not_called()


def test_failed_start_does_not_begin_continuous():
    app = _listening_app()
    with patch.object(app, "_start_recording"), \
            patch.object(app, "_begin_continuous") as begin:
        app.start_listening()
    begin.assert_not_called()


def test_set_voice_chat_listens_only_when_asked():
    app = _listening_app()
    with patch.object(app, "start_listening") as start_listening, \
            patch.object(app, "_warm_up_voice_chat"):
        app.set_voice_chat(True)
        time.sleep(0.05)
        start_listening.assert_not_called()
        app.set_voice_chat(False, listen=True)
        for _ in range(200):
            if start_listening.called:
                break
            time.sleep(0.005)
    start_listening.assert_called_once_with()
