"""Settings hotkey recorder: tk key events become hotkey text the parsers accept."""
from types import SimpleNamespace

import pytest

from debora_whisper.ui import settings as st
from debora_whisper.ui.settings import hotkey_from_event


def ev(keysym, state=0):
    return SimpleNamespace(keysym=keysym, state=state)


CTRL, SHIFT = st._STATE_CONTROL, st._STATE_SHIFT
ALT = st._STATE_ALT


@pytest.mark.parametrize("event, expected", [
    (ev("space", CTRL), "ctrl+space"),
    (ev("F", CTRL | SHIFT), "ctrl+shift+f"),
    (ev("k", CTRL | ALT), "ctrl+alt+k"),
    (ev("F9"), "f9"),
    (ev("Prior", ALT), "alt+page up"),
    (ev("Return", CTRL), "ctrl+enter"),
])
def test_records_chords(event, expected):
    assert hotkey_from_event(event) == (expected, None)


@pytest.mark.parametrize("keysym", ["Control_L", "Shift_R", "Alt_L", "Caps_Lock"])
def test_modifier_alone_keeps_waiting(keysym):
    assert hotkey_from_event(ev(keysym, CTRL)) == (None, None)


@pytest.mark.parametrize("event", [ev("a"), ev("space"), ev("5", SHIFT)])
def test_bare_typing_keys_are_refused(event):
    hotkey, problem = hotkey_from_event(event)
    assert hotkey is None and "Ctrl or Alt" in problem


def test_unknown_key_is_refused():
    hotkey, problem = hotkey_from_event(ev("dead_acute", CTRL))
    assert hotkey is None and "can't be used" in problem


def test_recorded_chords_parse_on_windows():
    from debora_whisper.platforms.windows_hotkeys import parse_hotkey
    for event in (ev("space", CTRL), ev("F", CTRL | SHIFT), ev("F9"), ev("Prior", ALT)):
        hotkey, _ = hotkey_from_event(event)
        parse_hotkey(hotkey)


def test_apply_error_opens_the_tab_that_holds_the_field():
    window = st.SettingsWindow.__new__(st.SettingsWindow)
    window._tabs = SimpleNamespace(set=lambda name: setattr(window, "shown", name))
    window._show_tab_for("Invalid hotkey: unsupported key 'x'")
    assert window.shown == "Dictation"
    window._show_tab_for("XTTS-v2 requires a reference voice (tts_voice)")
    assert window.shown == "Voice chat"
