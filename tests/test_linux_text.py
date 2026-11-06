"""Typing on Linux: paste keys per window, the clipboard round trip and
the window in focus from KWin. No desktop or D-Bus is touched."""
from unittest.mock import Mock, call

import sys

import pytest

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux provider")

from debora_whisper.platforms import kwin_focus, linux_text  # noqa: E402
from debora_whisper.platforms.linux import LinuxPlatform  # noqa: E402


# -- paste keys and typing -----------------------------------------------------

def window(cls):
    return kwin_focus.FocusedWindow("{id}", cls, "caption", 1)


@pytest.mark.parametrize("cls, expected", [
    ("org.kde.konsole", "ctrl+shift+v"),
    ("com.mitchellh.ghostty", "ctrl+shift+v"),
    ("kitty", "ctrl+shift+v"),
    ("code_code", "ctrl+v"),
    ("google-chrome", "ctrl+v"),
    ("org.kde.kate", "ctrl+v"),
])
def test_terminals_paste_with_ctrl_shift_v(cls, expected):
    assert linux_text.paste_key(window(cls)) == expected


def test_unknown_window_and_overrides():
    assert linux_text.paste_key(None) == linux_text.UNKNOWN_PASTE
    assert linux_text.paste_key(None, {"unknown": "shift+insert"}) == "shift+insert"
    assert linux_text.paste_key(window("org.kde.konsole"), {"org.kde.konsole": "shift+insert"}) == "shift+insert"
    assert linux_text.paste_key(window("emacs"), {"Emacs": "ctrl+y"}) == "ctrl+y"


@pytest.fixture
def text_io(monkeypatch):
    clipboard = ["copied before"]
    keys = Mock()
    monkeypatch.setattr("pyperclip.copy", lambda text: clipboard.__setitem__(0, text))
    monkeypatch.setattr("pyperclip.paste", lambda: clipboard[0])
    monkeypatch.setattr(linux_text.keyboard, "send", keys.send)
    monkeypatch.setattr(linux_text.keyboard, "press", keys.press)
    monkeypatch.setattr(linux_text.keyboard, "release", keys.release)
    monkeypatch.setattr(linux_text.time, "sleep", lambda s: None)
    return clipboard, keys


def test_type_text_pastes_then_restores_the_clipboard(text_io):
    clipboard, keys = text_io
    sent = []
    keys.send.side_effect = lambda k: sent.append((k, clipboard[0]))
    linux_text.type_text("Olá, ação", "ctrl+shift+v", auto_enter=True)
    assert sent == [("ctrl+shift+v", "Olá, ação"), ("enter", "Olá, ação")]
    assert clipboard[0] == "copied before"


def test_empty_old_clipboard_is_not_restored(text_io):
    clipboard, _ = text_io
    clipboard[0] = ""  # an image reads back as empty text
    linux_text.type_text("oi", "ctrl+v")
    assert clipboard[0] == "oi"


def test_delete_text_selects_left_then_deletes(text_io):
    _, keys = text_io
    linux_text.delete_text(2)
    assert keys.mock_calls == [call.press("shift"), call.send("left"), call.send("left"),
                               call.release("shift"), call.send("backspace")]


# -- the provider ----------------------------------------------------------------

def test_provider_types_with_the_focused_windows_key(monkeypatch):
    provider = LinuxPlatform()
    monkeypatch.setattr(provider, "_focused", lambda: window("org.kde.konsole"))
    typed = Mock()
    monkeypatch.setattr(linux_text, "type_text", typed)
    provider.type_text("oi", auto_enter=True)
    typed.assert_called_once_with("oi", "ctrl+shift+v", True)
    provider.apply_config({"paste_hotkeys": {"konsole": "shift+insert"}})
    assert provider.paste_key() == "shift+insert"


def test_input_target_is_the_kwin_window(monkeypatch):
    provider = LinuxPlatform()
    monkeypatch.setattr(provider, "_focused", lambda: window("kitty"))
    target = provider.input_target()
    assert provider.same_input_target(target, "{id}")
    assert not provider.same_input_target(target, "{other}")
    assert not provider.same_input_target(None, None)


# -- KWin focus ------------------------------------------------------------------

def kwin_report(focus, sender, resource_class="org.kde.konsole"):
    from jeepney import DBusAddress, HeaderFields, new_method_call
    address = DBusAddress("/Focus", bus_name=focus._name, interface=focus._name)
    activated = new_method_call(address, "Activated", "sssi",
                                ("{abc}", resource_class, "~ : bash", 42))
    activated.header.fields[HeaderFields.sender] = sender
    return activated


def serve(focus, *messages):
    sent = []

    class Conn:
        def receive(self):
            if not messages_left:
                raise OSError("closed")
            return messages_left.pop(0)

        def send(self, message):
            sent.append(message)

    messages_left = list(messages)
    focus._server, focus._started = Conn(), True
    focus._serve()
    return sent


def test_kwin_reports_update_the_focused_window():
    focus = kwin_focus.KWinFocus()
    focus._kwin_owner = ":1.5"
    sent = serve(focus, kwin_report(focus, ":1.5"))
    assert focus.window().resource_class == "org.kde.konsole"
    assert focus.window().id == "{abc}" and focus.window().pid == 42
    assert len(sent) == 1  # KWin's callDBus gets its reply


def test_reports_from_other_processes_are_ignored():
    focus = kwin_focus.KWinFocus()
    focus._kwin_owner = ":1.5"
    serve(focus, kwin_report(focus, ":1.5"), kwin_report(focus, ":1.99", "evil"))
    assert focus.window().resource_class == "org.kde.konsole"


def test_kwin_unavailable_is_reported_not_raised(monkeypatch):
    import jeepney.io.blocking as blocking
    monkeypatch.setattr(blocking, "open_dbus_connection",
                        Mock(side_effect=OSError("no session bus")))
    focus = kwin_focus.KWinFocus()
    assert focus.start() is False
    assert "no session bus" in focus.error
    assert focus.window() is None


# -- .desktop shortcuts ------------------------------------------------------------
