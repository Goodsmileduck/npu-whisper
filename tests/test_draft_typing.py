"""Live drafts typed into the target window and corrected in place."""
import sys
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from debora_whisper import dictation_engine as de
from debora_whisper.dictation_engine import AppState, DictationApp, DEFAULT_CONFIG
from debora_whisper.platforms.base import CaretReadStatus

AUDIO = np.zeros(16000, dtype=np.float32)
pytestmark = pytest.mark.usefixtures("windows_text_provider")
# SendInput's INPUT layout (and its size) only holds with Windows' C longs.
SEND_INPUT = pytest.mark.skipif(sys.platform != "win32", reason="Windows SendInput layout")


class Screen:
    """The text box keys land in, plus where focus is."""

    def __init__(self):
        self.text = ""
        self.target = ("window", "field")
        self.pasted = []
        self.drafted = []
        self.enters = 0

    def paste(self, text, auto_enter=False):
        self.pasted.append(text)
        self.text += text
        if auto_enter:
            self.enters += 1

    def type_draft(self, text):
        if text:
            self.drafted.append(text)
            self.text += text

    def delete(self, count):
        if count > 0:
            self.text = self.text[:-count]


@pytest.fixture
def screen():
    s = Screen()
    with patch.object(de, "type_text", side_effect=s.paste), \
            patch.object(de, "type_draft_text", side_effect=s.type_draft), \
            patch.object(de, "delete_text", side_effect=s.delete), \
            patch.object(de, "get_input_target", side_effect=lambda: s.target), \
            patch.object(de.platforms.current(), "text_before_caret",
                         side_effect=lambda count: s.text[-count:] if count else ""), \
            patch("keyboard.press_and_release") as key:
        key.side_effect = lambda k: setattr(s, "enters", s.enters + 1)
        yield s


def _app(**config):
    app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False,
                        "continuous_listening": True, "inline_drafts": True,
                        **config})
    app.recorder = MagicMock()
    app.whisper = MagicMock()
    app._model_ready.set()
    return app


def _say(app, text, is_final, ptt=False):
    if isinstance(text, Exception):
        app.whisper.transcribe.side_effect = text
    else:
        app.whisper.transcribe.side_effect = None
        app.whisper.transcribe.return_value = text
    if ptt:
        app.is_recording = True
        app.recorder.stop.return_value = AUDIO
        app._finish_recording(is_final=is_final)
    else:
        app._finish_recording(audio=AUDIO, is_final=is_final)


def test_consecutive_ptt_finals_are_separated(screen):
    app = _app(continuous_listening=False)
    _say(app, "O céu está nublado.", is_final=True, ptt=True)
    _say(app, "Preciso comprar pão.", is_final=True, ptt=True)

    assert screen.text == "O céu está nublado. Preciso comprar pão. "
    assert screen.pasted == ["O céu está nublado. ", "Preciso comprar pão. "]
    assert app.recorder.stop.call_count == 2


@pytest.mark.parametrize("ptt", [True, False])
def test_final_separator_is_only_typed(screen, ptt):
    app = _app(continuous_listening=False)
    received = []
    app.add_callback(lambda state, data: received.append((state, data)))

    with patch.object(de, "log") as log:
        _say(app, "Pronto.", is_final=True, ptt=ptt)

    assert screen.pasted == ["Pronto. "]
    assert screen.enters == 0
    assert app.history[0]["text"] == "Pronto."
    assert received[-1] == (AppState.READY, {"text": "Pronto."})
    log.assert_any_call("Final transcription: Pronto.")


def test_ptt_final_with_auto_enter_sends_no_separator(screen):
    app = _app(continuous_listening=False, auto_enter=True)
    _say(app, "Pronto.", is_final=True, ptt=True)

    assert screen.pasted == ["Pronto."]
    assert screen.enters == 1


def test_blank_continuous_final_types_nothing(screen):
    app = _app(continuous_listening=False)
    _say(app, "  ", is_final=True, ptt=False)

    assert screen.pasted == []
    assert app._history == []


@pytest.mark.parametrize("suffix", [" ", "\t", "\n", "\u00a0"])
def test_ptt_final_preserves_existing_whitespace(screen, suffix):
    app = _app(continuous_listening=False)
    _say(app, "Pronto." + suffix, is_final=True, ptt=True)

    assert screen.pasted == ["Pronto." + suffix]
    assert app.history[0]["text"] == "Pronto." + suffix


@pytest.mark.parametrize("text", ["", " ", "\t\n", "\u00a0"])
def test_empty_ptt_final_types_nothing(screen, text):
    app = _app(continuous_listening=False, auto_enter=True)
    _say(app, text, is_final=True, ptt=True)

    assert screen.text == ""
    assert screen.pasted == []
    assert screen.enters == 0
    assert app._history == []


@pytest.mark.parametrize("draft, final, tail, deleted", [
    ("Pronto.", "Pronto.", "", 0),
    ("Olá tudo", "Olá tudo bem.", " bem. ", 4),
])
def test_ptt_final_reuses_typed_draft(screen, draft, final, tail, deleted):
    app = _app(continuous_listening=False)
    _say(app, draft, is_final=False)
    _say(app, final, is_final=True, ptt=True)

    assert screen.text == final + " "
    assert screen.pasted == ([tail] if tail else [])
    de.delete_text.assert_called_once_with(deleted)
    assert screen.enters == 0
    assert app._draft_typed_text == ""
    assert app.history[-1]["text"] == final


def test_ptt_final_after_focus_change_stays_in_history(screen):
    app = _app(continuous_listening=False)
    _say(app, "Primeira parte", is_final=False)
    screen.target = ("other window", "field")
    screen.text = "Texto do usuário"
    _say(app, "Primeira parte.", is_final=True, ptt=True)

    assert screen.text == "Texto do usuário"
    assert screen.pasted == []
    assert app._history[-1]["text"] == "Primeira parte."


def test_ptt_voice_chat_final_types_nothing(screen):
    app = _app(continuous_listening=False, voice_chat=True)
    with patch.object(app, "_voice_chat_turn") as turn:
        _say(app, "Como está o tempo?", is_final=True, ptt=True)

    assert turn.call_args.args[0] == "Como está o tempo?"
    assert screen.pasted == []
    assert screen.text == ""


def test_draft_is_corrected_in_place(screen):
    app = _app()
    _say(app, "Olá tudo", is_final=False)
    _say(app, "Olá tudo bem", is_final=False)
    _say(app, "Olá, tudo bem?", is_final=True)

    assert screen.text == "Olá, tudo bem? "
    assert screen.drafted == ["Olá tudo... ", " bem... "]
    # Only the differing tail is sent as final text.
    assert screen.pasted == [", tudo bem? "]
    assert app._draft_typed_text == ""


def test_drafts_stay_in_overlay_by_default(screen):
    app = _app(inline_drafts=DEFAULT_CONFIG["inline_drafts"])
    _say(app, "Olá tudo", is_final=False)
    assert app.last_draft_text == "Olá tudo..."
    _say(app, "Olá, tudo bem?", is_final=True)

    assert screen.drafted == []
    assert screen.pasted == ["Olá, tudo bem? "]
    assert screen.text == "Olá, tudo bem? "


@pytest.fixture
def terminal(screen, monkeypatch):
    """A TUI where Shift+Left moves the caret without selecting anything."""
    from debora_whisper.platforms import windows_text
    import keyboard

    screen.caret = 0
    screen.keys = []

    def insert(text, auto_enter=False):
        screen.text = screen.text[:screen.caret] + text + screen.text[screen.caret:]
        screen.caret += len(text)
        screen.enters += bool(auto_enter)

    def key(name):
        screen.keys.append(name)
        if name == "left":
            screen.caret = max(0, screen.caret - 1)
        elif name == "backspace" and screen.caret:
            screen.text = screen.text[:screen.caret - 1] + screen.text[screen.caret:]
            screen.caret -= 1

    def draft(text):
        screen.drafted.append(text)
        insert(text)

    def final(text, auto_enter=False):
        screen.pasted.append(text)
        insert(text, auto_enter)

    monkeypatch.setattr(de.type_text, "side_effect", final)
    monkeypatch.setattr(de.delete_text, "side_effect", AssertionError("blind selection"))
    monkeypatch.setattr(de.type_draft_text, "side_effect", draft)
    monkeypatch.setattr(windows_text, "_type_text_ctypes",
                        MagicMock(side_effect=AssertionError("Unicode draft packets")))
    monkeypatch.setattr(windows_text.time, "sleep", lambda seconds: None)
    for name in ("send", "press", "release"):
        monkeypatch.setattr(keyboard, name, key)
    monkeypatch.setattr(de.platforms.current(), "text_before_caret", MagicMock(return_value=None))
    return screen


def test_unreadable_target_trims_only_tail_with_backspaces(terminal, capsys):
    app = _app()
    _say(app, "Olá tudo", is_final=False)
    _say(app, "Olá tudo bem", is_final=False)
    assert terminal.keys == ["backspace"] * len("... ")
    assert terminal.drafted == ["Olá tudo... ", " bem... "]
    _say(app, "Olá, tudo bem?", is_final=True)
    assert terminal.keys == ["backspace"] * (len("... ") + len(" tudo bem... "))
    assert terminal.pasted == [", tudo bem? "]
    assert terminal.text == "Olá, tudo bem? "
    de.platforms.current().text_before_caret.assert_called_once_with(0)
    assert capsys.readouterr().out.count("Inline draft: target unreadable; trimming with Backspace") == 1


def test_claude_code_growing_drafts_leave_only_final_text(terminal):
    app = _app()
    for draft in ("Inline drafts ligado", "Inline drafts ligado. Este texto aqui",
                  "Inline drafts ligado. Este texto aqui já é usando",
                  "Inline drafts ligado. Este texto aqui já é usando ele"):
        _say(app, draft, is_final=False)
    final = "Inline drafts ligado. Este texto aqui já é usando ele."
    _say(app, final, is_final=True)
    assert terminal.text == final + " "
    assert set(terminal.keys) == {"backspace"}
    assert len(terminal.drafted) == 4
    assert len(terminal.pasted) == 1


@pytest.mark.parametrize("legacy", [
    pytest.param(False, id="clipboard-fix"),
    pytest.param(True, id="pr45-replay", marks=SEND_INPUT),
])
@pytest.mark.parametrize("second, tail, deletes, final_tail", [
    ("Oi Débora, tudo bem", "tudo bem... ", 4, "? "),
    ("Oi Débora, tudo bem?", "tudo bem? ", 0, ""),
])
def test_notepad_report_output_sequence(screen, monkeypatch, legacy, second,
                                       tail, deletes, final_tail):
    """Draft 2 was not logged: cover both punctuation outcomes, down to INPUT.

    The legacy branch reconstructs PR #45's emission, not Notepad's handling.
    The fixed branch exercises real provider paste/restore and deletion code.
    """
    import ctypes
    import keyboard
    import pyperclip
    from debora_whisper.platforms import windows_text

    events = []
    clipboard = ["original clipboard"]
    pending = [False]

    def copy(text):
        events.append(("copy", text))
        clipboard[0] = text

    def paste_key(name):
        assert name == "ctrl+v"
        assert not pending[0]
        events.append(("key", name))
        pending[0] = True

    def wait(seconds):
        # The target reads after Ctrl+V, before the serialized restoration.
        if pending[0]:
            screen.text += clipboard[0]
            events.append(("paste", clipboard[0]))
            pending[0] = False

    def backspace(name):
        assert not pending[0]
        assert name == "backspace"
        events.append(("key", name))
        screen.delete(1)

    def send_input(count, ref, size):
        assert count == 2
        assert size == (40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)
        pair = [(p.type, p._input.ki.wVk, p._input.ki.wScan,
                 p._input.ki.dwFlags, p._input.ki.time,
                 bool(p._input.ki.dwExtraInfo)) for p in ref._obj]
        events.append(("SendInput", pair))
        screen.text += chr(pair[0][2])
        return count

    monkeypatch.setattr(ctypes, "windll", MagicMock(), raising=False)
    ctypes.windll.user32.SendInput.side_effect = send_input
    monkeypatch.setattr(pyperclip, "copy", copy)
    monkeypatch.setattr(pyperclip, "paste", lambda: clipboard[0])
    monkeypatch.setattr(keyboard, "press_and_release", paste_key)
    monkeypatch.setattr(keyboard, "send", backspace)
    monkeypatch.setattr(windows_text.time, "sleep", wait)
    monkeypatch.setattr(de.platforms.current(), "text_before_caret", lambda count: None)
    monkeypatch.setattr(de.type_text, "side_effect", windows_text.type_text)
    monkeypatch.setattr(de.type_draft_text, "side_effect",
                        windows_text._type_text_ctypes if legacy else windows_text.type_draft)

    def insertion(text, unicode=False):
        if not text:
            return []
        if unicode:
            return [("SendInput", [(1, 0, ord(c), 4, 0, False),
                                    (1, 0, ord(c), 6, 0, False)]) for c in text]
        return [("copy", text), ("key", "ctrl+v"), ("paste", text),
                ("copy", "original clipboard")]

    app = _app()
    _say(app, "Oi Débora,", is_final=False)
    _say(app, second, is_final=False)
    _say(app, "Oi Débora, tudo bem?", is_final=True)
    assert events == (insertion("Oi Débora, ", legacy) + insertion(tail, legacy)
                      + [("key", "backspace")] * deletes + insertion(final_tail))
    assert screen.text == "Oi Débora, tudo bem? "
    assert clipboard[0] == "original clipboard"
    assert not pending[0]
    assert app._draft_typed_text == ""


@pytest.mark.parametrize("final", ["", RuntimeError("failed final")])
def test_terminal_empty_or_failed_final_resets_strategy(terminal, final):
    app = _app()
    _say(app, "Olá tudo", is_final=False)
    _say(app, final, is_final=True)
    assert terminal.text == ("Olá tudo... " if isinstance(final, Exception) else "")
    assert app._draft_strategy is None
    _say(app, "Nova frase", is_final=False)
    assert de.platforms.current().text_before_caret.call_args_list == [call(0), call(0)]


@pytest.mark.parametrize("complex_text", ["emoji 👍", "cafe\u0301", "mark \ufe0f", "join\u200der",
                                         "hangul \u1100\u1161", "surrogate \ud83d"])
@pytest.mark.parametrize("previous", [False, True])
def test_terminal_complex_unicode_uses_overlay(terminal, complex_text, previous):
    app = _app()
    if previous:
        _say(app, "Olá tudo", is_final=False)
    _say(app, complex_text, is_final=False)
    assert terminal.text == ""
    assert app._inline_drafts_abandoned
    _say(app, "Olá, tudo bem?", is_final=True)
    assert terminal.text == "Olá, tudo bem? "
    assert terminal.pasted == ["Olá, tudo bem? "]


@pytest.mark.parametrize("complex_text", ["emoji 👍", "cafe\u0301", "mark \ufe0f", "join\u200der",
                                         "hangul \u1100\u1161", "surrogate \ud83d",
                                         "control\x07text", "two\r\nlines"])
@pytest.mark.parametrize("previous", [False, True])
def test_readable_complex_unicode_switches_to_overlay(screen, complex_text, previous):
    app = _app(auto_enter=True)
    prefix = "Existing user text: "
    screen.text = prefix
    if previous:
        _say(app, "Olá tudo", is_final=False)
    _say(app, complex_text, is_final=False)

    assert screen.text == prefix
    assert screen.drafted == (["Olá tudo... "] if previous else [])
    assert app.last_draft_text == complex_text + "..."
    assert app._draft_strategy == "overlay"
    assert app._inline_drafts_abandoned
    assert not app._draft_final_suppressed
    assert app._draft_typed_text == ""
    if previous:
        de.delete_text.assert_called_once_with(len("Olá tudo... "))
    else:
        de.delete_text.assert_not_called()

    _say(app, "Simple update", is_final=False)
    assert screen.text == prefix
    assert app.last_draft_text == "Simple update..."
    final = complex_text + " completo! "
    _say(app, final, is_final=True)
    assert screen.text == prefix + final
    assert screen.pasted == [final]
    assert screen.enters == 1
    assert app._history[-1]["text"] == final
    assert app._draft_strategy is None
    assert not app._inline_drafts_abandoned
    _say(app, "Next segment", is_final=False)
    assert screen.text == prefix + final + "Next segment... "
    assert app._draft_strategy == "verified"


@pytest.mark.parametrize("obstacle", ["focus", "mismatch", "unsafe"])
def test_readable_complex_transition_keeps_unremovable_draft(screen, monkeypatch, obstacle):
    app = _app(auto_enter=True)
    _say(app, "Olá tudo", is_final=False)
    if obstacle == "focus":
        screen.target = ("other window", "field")
        screen.text = "Other user text"
    elif obstacle == "mismatch":
        screen.text += "user edit"
    else:
        monkeypatch.setattr(de.platforms.current(), "text_before_caret",
                            lambda count: CaretReadStatus.UNSAFE)
    kept = screen.text
    _say(app, "cafe\u0301", is_final=False)
    assert app._draft_strategy == "overlay"
    assert app._draft_final_suppressed
    _say(app, "cafe\u0301 completo!", is_final=True)

    assert screen.text == kept
    de.delete_text.assert_not_called()
    assert screen.drafted == ["Olá tudo... "]
    assert screen.pasted == []
    assert screen.enters == 0
    assert app._history[-1]["text"] == "cafe\u0301 completo!"


def test_terminal_focus_change_leaves_draft_and_abandons_segment(terminal):
    app = _app()
    _say(app, "Olá tudo", is_final=False)
    terminal.target = ("other window", "field")
    terminal.text, terminal.caret = "user text: ", len("user text: ")
    _say(app, "Olá tudo bem", is_final=False)
    _say(app, "Olá, tudo bem?", is_final=True)
    assert terminal.text == "user text: "
    assert terminal.keys == []


def test_terminal_focus_change_with_complex_draft_does_not_duplicate(terminal):
    app = _app()
    _say(app, "Olá tudo", is_final=False)
    terminal.target = ("other window", "field")
    terminal.text, terminal.caret = "user text: ", len("user text: ")
    _say(app, "Olá tudo 👍", is_final=False)
    _say(app, "Olá tudo 👍!", is_final=True)
    assert terminal.text == "user text: "
    assert terminal.keys == []


def test_terminal_can_paste_complex_unicode_in_final(terminal):
    app = _app()
    _say(app, "Olá tudo", is_final=False)
    _say(app, "Olá 👍 cafe\u0301", is_final=True)
    assert terminal.text == "Olá 👍 cafe\u0301 "


@SEND_INPUT
def test_native_backspace_fallback_sends_only_backspace(monkeypatch):
    import ctypes
    import sys
    from debora_whisper.platforms import windows_text

    monkeypatch.setitem(sys.modules, "keyboard", None)
    monkeypatch.setattr(ctypes, "windll", MagicMock())
    monkeypatch.setattr(windows_text.time, "sleep", lambda seconds: None)
    de.platforms.current().backspace_text(3)
    assert ctypes.windll.user32.keybd_event.call_args_list == [
        call(0x08, 0, 0, 0), call(0x08, 0, 0x0002, 0)] * 3


@pytest.mark.parametrize("strict", [True, False])
def test_readable_draft_keeps_verified_rewrite(screen, monkeypatch, strict):
    seen = []

    def read(count):
        seen.append(count)
        return screen.text[-count:] if count else ""

    monkeypatch.setattr(de.platforms.current(), "text_before_caret", read)
    app = _app(inline_drafts_require_verification=strict)
    _say(app, "Olá tudo", is_final=False)
    _say(app, "Olá tudo bem", is_final=False)
    _say(app, "Olá, tudo bem?", is_final=True)
    assert seen == [0, len("Olá tudo... "), len("Olá tudo bem... ")]
    assert screen.text == "Olá, tudo bem? "
    assert screen.drafted == ["Olá tudo... ", " bem... "]
    assert screen.pasted == [", tudo bem? "]


@pytest.mark.parametrize("observed,strict", [(None, True), (CaretReadStatus.UNSAFE, True),
                                           (CaretReadStatus.UNSAFE, False)])
def test_unreadable_target_keeps_drafts_in_overlay(screen, monkeypatch, capsys, observed, strict):
    read = MagicMock(return_value=observed)
    monkeypatch.setattr(de.platforms.current(), "text_before_caret", read)
    app = _app(inline_drafts_require_verification=strict)
    app.is_recording = True
    states = []
    app.add_callback(lambda state, data: states.append(data))
    message = "Inline draft: target text is unreadable; drafts stay in the overlay"

    for _ in range(2):
        read.reset_mock()
        _say(app, "Olá tudo", is_final=False)
        _say(app, "Olá tudo bem", is_final=False)
        assert app._inline_drafts_abandoned
        assert app._draft_typed_text == ""
        assert app._draft_target is None
        assert states[-1]["draft_text"] == "Olá tudo bem..."
        assert app.last_draft_text == "Olá tudo bem..."
        assert screen.drafted == []
        _say(app, "Olá, tudo bem?", is_final=True)
        # The final needs no additional caret read, including after a timeout.
        read.assert_called_once_with(0)
        assert not app._inline_drafts_abandoned
        assert capsys.readouterr().out.count(message) == 1

    assert screen.pasted == ["Olá, tudo bem? "] * 2
    assert screen.text == "Olá, tudo bem? " * 2
    assert message in de.LOG_FILE.read_text(encoding="utf-8")


@pytest.mark.parametrize("existing", ["", "Existing text. "])
def test_readable_target_allows_first_draft(screen, existing):
    screen.text = existing
    app = _app()
    _say(app, "Olá tudo", is_final=False)
    assert screen.drafted == ["Olá tudo... "]
    assert screen.text == existing + "Olá tudo... "
    assert not app._inline_drafts_abandoned
    de.platforms.current().text_before_caret.assert_called_once_with(0)


def test_final_without_drafts_does_not_probe(screen):
    app = _app()
    _say(app, "Olá tudo", is_final=True)
    assert screen.pasted == ["Olá tudo "]
    de.platforms.current().text_before_caret.assert_not_called()


@pytest.mark.parametrize("readable", [True, False])
def test_draft_probe_rewrite_and_final_hold_output_lock(screen, monkeypatch, readable):
    app = _app()
    provider = de.platforms.current()
    calls = []

    def guarded(name, action):
        def run(*args, **kwargs):
            assert app._output_lock.locked(), f"{name} ran outside _output_lock"
            calls.append(name)
            return action(*args, **kwargs)
        return run

    monkeypatch.setattr(provider, "text_before_caret", guarded(
        "probe", lambda count: (screen.text[-count:] if count else "") if readable else None))
    monkeypatch.setattr(de.type_text, "side_effect", guarded("final", screen.paste))
    monkeypatch.setattr(de.type_draft_text, "side_effect", guarded("draft", screen.type_draft))
    if readable:
        monkeypatch.setattr(de.delete_text, "side_effect", guarded("delete", screen.delete))
    else:
        monkeypatch.setattr(provider, "backspace_text", guarded("delete", screen.delete))

    _say(app, "hello", is_final=False)
    _say(app, "hello world", is_final=False)
    _say(app, "hello world!", is_final=True)

    assert calls == (["probe", "draft", "probe", "delete", "draft", "probe", "delete", "final"]
                     if readable else ["probe", "draft", "delete", "draft", "delete", "final"])
    assert screen.text == "hello world! "
    assert not app._output_lock.locked()


@pytest.mark.parametrize("observed,strict", [("changed", False), ("", False),
                                           (None, True), (None, False)])
@pytest.mark.parametrize("next_is_final", [False, True])
def test_changed_draft_abandons_segment(screen, monkeypatch, capsys, observed, strict, next_is_final):
    read = MagicMock(side_effect=lambda count: observed if count else "")
    monkeypatch.setattr(de.platforms.current(), "text_before_caret", read)
    app = _app(inline_drafts_require_verification=strict)
    app.is_recording = True
    states = []
    app.add_callback(lambda state, data: states.append(data))
    _say(app, "Olá tudo", is_final=False)
    screen.text += "user edit"
    kept = screen.text
    if not next_is_final:
        _say(app, "Olá tudo bem", is_final=False)
        _say(app, "Olá tudo bem hoje", is_final=False)
        assert screen.text == kept
        assert app._inline_drafts_abandoned
        assert states[-1]["draft_text"] == "Olá tudo bem hoje..."
    _say(app, "Olá, tudo bem?", is_final=True)
    assert screen.text == kept
    assert screen.drafted == ["Olá tudo... "]
    assert screen.pasted == []
    assert read.call_args_list == [call(0), call(len("Olá tudo... ")),
                                   call(len("Olá tudo... ") + 8)]
    assert capsys.readouterr().out.count(
        "Inline draft: cannot safely reconcile; keeping the draft and suppressing final paste") == 1
    assert not app._inline_drafts_abandoned
    _say(app, "Nova frase", is_final=False)
    assert screen.drafted[-1] == "Nova frase... "


@pytest.mark.parametrize("final", ["", RuntimeError("failed final")])
def test_abandoned_draft_resets_after_empty_or_failed_final(screen, monkeypatch, final):
    monkeypatch.setattr(de.platforms.current(), "text_before_caret", lambda count: "changed")
    app = _app()
    _say(app, "Olá tudo", is_final=False)
    _say(app, "Olá tudo bem", is_final=False)
    _say(app, final, is_final=True)
    assert screen.text == "Olá tudo... "
    assert not app._inline_drafts_abandoned
    _say(app, "Nova frase", is_final=False)
    assert screen.drafted[-1] == "Nova frase... "


@pytest.mark.parametrize("final", ["Frase corrigida!", "", RuntimeError("failed final")])
def test_suppression_resets_for_next_final_only_segment(screen, monkeypatch, final):
    app = _app()
    _say(app, "Minha frase", is_final=False)
    kept = screen.text
    monkeypatch.setattr(de.platforms.current(), "text_before_caret", lambda count: None)
    _say(app, "Minha frase completa", is_final=False)
    _say(app, final, is_final=True)
    assert screen.text == kept
    assert not app._draft_final_suppressed
    _say(app, "Outra frase!", is_final=True)
    assert screen.text == kept + "Outra frase! "


def test_empty_final_verifies_before_erasing(screen, monkeypatch):
    monkeypatch.setattr(de.platforms.current(), "text_before_caret", lambda count: "changed")
    app = _app()
    _say(app, "Olá tudo", is_final=False)
    _say(app, "", is_final=True)
    assert screen.text == "Olá tudo... "
    assert app._draft_typed_text == ""


@pytest.mark.parametrize("plain,rich", [(" ", "\u00a0"), (" ", "\u202f"),
                                      (" ", "\u2009"), ("'", "\u2019"),
                                      ('"', '\u201c'), ("\n", "\r"),
                                      ("\n", "\v"), ("\n", "\u2029")])
def test_word_one_for_one_changes_rewrite_only_our_tail(screen, monkeypatch, plain, rich):
    app = _app()
    prefix = "Existing user text: "
    screen.text = prefix
    draft = f"Minha{plain}frase"
    _say(app, draft, is_final=False)
    # Simulate transformations in the editor itself, not just in the UIA read.
    screen.text = prefix + screen.text[len(prefix):].replace(plain, rich)
    _say(app, draft + " completa", is_final=False)
    screen.text = prefix + screen.text[len(prefix):].replace(plain, rich)
    _say(app, draft + " completa!", is_final=True)

    assert screen.text == prefix + (draft + " completa").replace(plain, rich) + "! "
    assert screen.pasted == ["! "]
    assert not app._draft_final_suppressed


@pytest.mark.parametrize("suffix", ["\r", "\v", "\r\n", "\x07", "\r\x07", "  "])
def test_word_extra_trailing_marks_preserve_draft_without_duplicate(screen, monkeypatch, suffix):
    app = _app(auto_enter=True)
    screen.text = "Existing user text: "
    _say(app, "Minha frase", is_final=False)
    kept = screen.text
    monkeypatch.setattr(de.platforms.current(), "text_before_caret",
                        lambda count: (screen.text + suffix)[-count:])
    _say(app, "Minha frase completa", is_final=False)
    _say(app, "Minha frase completa!", is_final=True)

    assert screen.text == kept
    assert screen.pasted == []
    assert screen.enters == 0
    de.delete_text.assert_not_called()
    assert app._history[-1]["text"] == "Minha frase completa!"


@pytest.mark.parametrize("actually_trimmed", [False, True])
def test_word_trimmed_trailing_space_never_overdeletes_or_duplicates(screen, monkeypatch, actually_trimmed):
    app = _app()
    screen.text = "Existing user text: "
    _say(app, "Minha frase", is_final=False)
    if actually_trimmed:
        screen.text = screen.text.rstrip()
    kept = screen.text
    monkeypatch.setattr(de.platforms.current(), "text_before_caret",
                        lambda count: screen.text.rstrip()[-count:])
    _say(app, "Frase corrigida!", is_final=True)
    assert screen.text == kept
    assert screen.pasted == []
    de.delete_text.assert_not_called()


@pytest.mark.parametrize("stale", [None, "", "Minha", "Minha frase...", "Minha frase... \r"])
def test_stale_then_fresh_read_reconciles_without_duplicate(screen, monkeypatch, stale):
    app = _app()
    _say(app, "Minha frase", is_final=False)
    read = MagicMock(side_effect=[stale, screen.text])
    wait = MagicMock()
    monkeypatch.setattr(de.platforms.current(), "text_before_caret", read)
    monkeypatch.setattr(de.time, "sleep", wait)
    _say(app, "Minha frase completa!", is_final=True)
    assert screen.text == "Minha frase completa! "
    assert screen.pasted == [" completa! "]
    assert read.call_args_list == [call(len("Minha frase... ")), call(len("Minha frase... ") + 8)]
    wait.assert_called_once_with(0.040)


def test_retry_with_context_and_smart_quotes_can_replace_entire_draft(screen, monkeypatch):
    app = _app()
    prefix = "Existing user text: "
    screen.text = prefix
    _say(app, "d'água", is_final=False)
    screen.text = screen.text.replace("'", "\u2019")
    read = MagicMock(side_effect=["stale", screen.text])
    monkeypatch.setattr(de.platforms.current(), "text_before_caret", read)
    _say(app, "Água fresca!", is_final=True)
    assert screen.text == prefix + "Água fresca! "
    assert screen.pasted == ["Água fresca! "]
    de.delete_text.assert_called_once_with(len("d'água... "))


@pytest.mark.parametrize("cluster", ["e\u0301", "\u0915\u093e", "\u1100\u1161",
                                     "\u2764\ufe0f", "\U0001f469\u200d\U0001f4bb",
                                     "\U0001f44d\U0001f3fd", "\U0001f1e7\U0001f1f7"])
@pytest.mark.parametrize("update_first", [False, True])
def test_grapheme_movement_never_deletes_preceding_user_text(screen, monkeypatch, cluster, update_first):
    app = _app(auto_enter=True)
    screen.text = "X"
    # Seed a pre-existing guarded draft to test the verifier's defense even
    # though the insertion path now prevents this state.
    app._draft_strategy = "verified"
    app._draft_target = screen.target
    app._draft_typed_text = cluster + "... "
    screen.type_draft(app._draft_typed_text)
    kept = screen.text
    # This editor moves one grapheme per key, including for UIA reads.
    # In particular, six movements read/delete all of "Xe\u0301... ".
    units = ["X", cluster, ".", ".", ".", " "]
    assert "".join(units) == kept

    def read(count):
        return "".join(units[-count:]) if count else ""

    def delete(count):
        if count > 0:
            del units[-count:]
            screen.text = "".join(units)

    monkeypatch.setattr(de.platforms.current(), "text_before_caret", read)
    de.delete_text.side_effect = delete
    assert read(len(app._draft_typed_text)) == kept
    if update_first:
        _say(app, "Replacement draft", is_final=False)
    _say(app, "Replacement final!", is_final=True)

    assert screen.text == kept
    de.delete_text.assert_not_called()
    assert screen.drafted == [cluster + "... "]
    assert screen.pasted == []
    assert screen.enters == 0
    assert app._history[-1]["text"] == "Replacement final!"


def test_unsafe_selection_is_not_retried_or_overwritten(screen, monkeypatch):
    app = _app(auto_enter=True)
    _say(app, "Minha frase", is_final=False)
    kept = screen.text
    read = MagicMock(return_value=CaretReadStatus.UNSAFE)
    wait = MagicMock()
    monkeypatch.setattr(de.platforms.current(), "text_before_caret", read)
    monkeypatch.setattr(de.time, "sleep", wait)
    _say(app, "Minha frase completa!", is_final=True)
    read.assert_called_once_with(len("Minha frase... "))
    wait.assert_not_called()
    de.delete_text.assert_not_called()
    assert screen.text == kept
    assert screen.pasted == []
    assert screen.enters == 0


@pytest.mark.parametrize("when", ["first_read", "wait", "second_read"])
def test_focus_change_during_verification_never_deletes_or_pastes(screen, monkeypatch, when):
    app = _app(auto_enter=True)
    _say(app, "Minha frase", is_final=False)
    previous = screen.text
    reads = []

    def move():
        screen.target = ("window", "another control")
        screen.text = "User's other field"

    def read(count):
        reads.append(count)
        if when == "first_read" or (when == "second_read" and len(reads) == 2):
            move()
            return previous  # A late but matching result must not authorize keys.
        return "stale"

    monkeypatch.setattr(de.platforms.current(), "text_before_caret", read)
    monkeypatch.setattr(de.time, "sleep", lambda seconds: move() if when == "wait" else None)
    _say(app, "Minha frase completa!", is_final=True)
    assert screen.text == "User's other field"
    assert screen.pasted == []
    assert screen.enters == 0
    de.delete_text.assert_not_called()
    assert len(reads) == (2 if when == "second_read" else 1)


@pytest.mark.parametrize("replacement", ["Minha frase... ", "minha\u200b frase... ",
                                         "minha\x13 frase... ", "minha\x07 frase... ",
                                         "minha frase\u2026 ", "unrelated text"])
def test_unproven_autocorrect_or_hidden_text_keeps_draft(screen, monkeypatch, replacement):
    app = _app()
    _say(app, "minha frase", is_final=False)
    screen.text = replacement
    _say(app, "minha frase completa!", is_final=True)
    assert screen.text == replacement
    assert screen.pasted == []
    de.delete_text.assert_not_called()


def test_mismatch_diagnostic_once_per_segment_including_recovered_reads(screen, monkeypatch, capsys):
    app = _app()
    draft = "Uma frase muito longa para verificar que o diagnóstico não contém texto"
    _say(app, draft, is_final=False)
    expected = screen.text
    private = "credential-never-dictated\r"
    read = MagicMock(side_effect=[private, expected, "other", "other"])
    monkeypatch.setattr(de.platforms.current(), "text_before_caret", read)
    _say(app, draft + " maior", is_final=False)
    _say(app, draft + " maior!", is_final=True)
    output = capsys.readouterr().out
    assert output.count("Inline draft: verification mismatch;") == 1
    diagnostic = next(line for line in output.splitlines() if "verification mismatch;" in line)
    assert f"expected_len={len(expected)} read_len={len(private)}" in diagnostic
    assert "read_status=text representation_changed=True equivalent=False safe_count=False" in diagnostic
    assert expected[-40:] not in diagnostic
    for logged in (output, de.LOG_FILE.read_text(encoding="utf-8")):
        assert private.strip() not in logged
        assert "expected_tail=" not in logged
        assert "read_tail=" not in logged
    monkeypatch.setattr(de.platforms.current(), "text_before_caret", lambda count: "")
    _say(app, "Outra frase", is_final=False)
    _say(app, "Outra frase!", is_final=True)
    assert capsys.readouterr().out.count("Inline draft: verification mismatch;") == 1


@pytest.mark.parametrize("change_focus", [False, True])
def test_unrelated_editor_text_never_reaches_logs(screen, monkeypatch, capsys, change_focus):
    app = _app()
    _say(app, "Minha frase", is_final=False)
    private = "confidential-document-password"

    def read(count):
        if change_focus:
            screen.target = ("window", "another control")
        return private

    monkeypatch.setattr(de.platforms.current(), "text_before_caret", read)
    _say(app, "Minha frase completa!", is_final=True)
    output = capsys.readouterr().out
    assert private not in output
    assert private not in de.LOG_FILE.read_text(encoding="utf-8")
    assert ("verification mismatch;" in output) is not change_focus
    de.delete_text.assert_not_called()
    assert screen.pasted == []


@pytest.mark.parametrize("expected,observed,equivalent,safe", [
    ("draft ", "prefix draft\r", True, False),
    ("draft ", "prefix draft", True, False),
    ("draft ", "prefix draft \r\x07", True, False),
    ("two\nlines ", "two\r\nlines ", True, False),
    ("draft ", "other draft\u00a0", True, True),
    ("don't ", "don\u2019t ", True, True),
    ("e\u0301... ", "Xe\u0301... ", True, False),
    ("e\u0301... ", "e\u0301... ", True, False),
    ("é... ", "Xé... ", True, True),
    ("a--b ", "a\u2014b ", False, False),
    ("two spaces ", "two  spaces ", False, False),
    ("draft ", "different ", False, False),
    ("draft ", CaretReadStatus.UNSAFE, False, False),
])
def test_draft_representation_match_does_not_guess_key_counts(expected, observed, equivalent, safe):
    assert de._draft_read_matches(expected, observed) == (equivalent, safe)


def test_final_equal_to_draft_still_presses_auto_enter(screen):
    app = _app(auto_enter=True)
    _say(app, "Pronto.", is_final=False)
    _say(app, "Pronto.", is_final=True)

    assert screen.text == "Pronto. "
    assert screen.pasted == []
    assert screen.enters == 1


@pytest.mark.parametrize("draft", ["Olá, tudo", "Olá, tudo bem?"])
def test_console_logs_complete_final_even_if_only_tail_or_nothing_is_typed(screen, capsys, draft):
    app = _app()
    _say(app, draft, is_final=False)
    assert "Final transcription:" not in capsys.readouterr().out

    _say(app, "Olá, tudo bem?", is_final=True)

    assert screen.text == "Olá, tudo bem? "
    output = capsys.readouterr().out
    assert output.count("Final transcription:") == 1
    assert "Final transcription: Olá, tudo bem?" in output
    assert "Final transcription: Olá, tudo bem?" in de.LOG_FILE.read_text(encoding="utf-8")


def test_final_log_is_not_truncated(screen, capsys):
    text = "Uma frase longa. " * 10
    _say(_app(), text, is_final=True)
    assert f"Final transcription: {text.strip()}" in capsys.readouterr().out


def test_focus_change_leaves_old_draft_and_suppresses_final(screen):
    app = _app()
    _say(app, "Primeira parte", is_final=False)
    typed_in_old_window = screen.text
    screen.target = ("other window", "field")
    screen.text = "código do usuário"

    _say(app, "Primeira parte.", is_final=True)

    assert screen.text == "código do usuário"
    assert screen.pasted == []
    assert app._history[-1]["text"] == "Primeira parte."
    assert typed_in_old_window == "Primeira parte... "


def test_focus_moving_to_another_field_counts_as_change(screen):
    app = _app()
    _say(app, "Rascunho", is_final=False)
    screen.target = ("window", "other field")
    screen.text = "outro campo"

    _say(app, "Rascunho final.", is_final=True)

    assert screen.text == "outro campo"
    assert screen.pasted == []


def test_empty_draft_keeps_typed_draft(screen):
    app = _app()
    _say(app, "Além disso", is_final=False)
    _say(app, "Obrigado.", is_final=False)  # dropped as a hallucination

    assert screen.text == "Além disso... "
    assert app._draft_typed_text == "Além disso... "


def test_endpoint_receives_raw_draft_before_display_ellipses(screen):
    app = _app()
    app.whisper.transcribe.return_value = "Minha ideia é"
    app._finish_recording(audio=AUDIO, is_final=False, segment_id=7, audio_end=16000)
    app.recorder.endpoint.update.assert_called_once_with(7, 16000, "Minha ideia é")
    assert screen.text == "Minha ideia é... "


def test_hallucination_feedback_is_empty(screen):
    app = _app()
    app.whisper.transcribe.return_value = "Obrigado."
    app._finish_recording(audio=AUDIO, is_final=False, segment_id=7, audio_end=16000)
    app.recorder.endpoint.update.assert_called_once_with(7, 16000, "")


def test_final_does_not_change_endpoint(screen):
    app = _app()
    app.whisper.transcribe.return_value = "Minha ideia é"
    app._finish_recording(audio=AUDIO, is_final=True, segment_id=7, audio_end=16000)
    app.recorder.endpoint.update.assert_not_called()


def test_segment_consumer_preserves_draft_identity_and_position(monkeypatch, screen):
    from types import SimpleNamespace
    from debora_whisper.vad_endpoint import VadSegment

    app = _app()
    app.recorder = de.AudioRecorder(config=app.config)
    segment_id = app.recorder.endpoint.start()
    app.recorder.segment_queue.put(VadSegment(AUDIO, False, segment_id, len(AUDIO)))
    app.whisper.transcribe.return_value = "Minha ideia é"
    finish = app._finish_recording

    def consume(**kwargs):
        finish(**kwargs)
        app._stopping.set()

    app._finish_recording = consume
    # Run the actual consumer synchronously, stopping after this draft.
    monkeypatch.setattr(de.threading, "Thread", lambda target, **kw: SimpleNamespace(start=target))
    app._start_segment_consumer()
    assert app.recorder.endpoint.incomplete
    assert screen.text == "Minha ideia é... "


def test_empty_final_erases_draft(screen):
    app = _app()
    _say(app, "Hum", is_final=False)
    _say(app, "", is_final=True)

    assert screen.text == ""
    assert app._draft_typed_text == ""


def test_failed_final_keeps_draft_but_next_segment_does_not_erase_it(screen):
    app = _app()
    _say(app, "Texto do rascunho", is_final=False)
    _say(app, RuntimeError("inference failed"), is_final=True)
    assert app._state == AppState.ERROR

    _say(app, "Outra frase.", is_final=True)

    assert screen.text == "Texto do rascunho... Outra frase. "


def test_failed_draft_keeps_tracking_so_final_corrects_it(screen):
    # Real log: the NPU was lost during a draft; the final of the same
    # segment ran on CUDA and typed the whole sentence after the stale draft.
    app = _app()
    _say(app, "Oi, ae, tudo bem", is_final=False)
    _say(app, RuntimeError("inference failed"), is_final=False)

    _say(app, "Oi, e aí, tudo bem?", is_final=True)

    assert screen.text == "Oi, e aí, tudo bem? "


def test_missing_focus_report_is_not_a_focus_change(screen):
    app = _app()
    _say(app, "Tudo", is_final=False)
    screen.target = ("window", None)  # GetGUIThreadInfo blinked

    _say(app, "Tudo bem.", is_final=True)

    assert screen.text == "Tudo bem. "


def test_same_input_target():
    assert de.same_input_target((1, 2), (1, 2))
    assert de.same_input_target((1, None), (1, 2))
    assert not de.same_input_target((1, 2), (1, 3))
    assert not de.same_input_target((1, 2), (4, 2))
    assert not de.same_input_target(None, (1, 2))


def test_continuous_final_shows_neither_transcribing_nor_done(screen):
    app = _app()
    app.is_recording = True
    states = []
    app.add_callback(lambda state, data: states.append(state))

    _say(app, "Frase completa.", is_final=True)

    assert AppState.PROCESSING not in states
    # No "Done" balloon either: straight back to listening.
    assert states == [AppState.RECORDING]


def test_push_to_talk_final_still_shows_transcribing_and_done(screen):
    app = _app(continuous_listening=False)
    states = []
    app.add_callback(lambda state, data: states.append(state))

    _say(app, "Frase completa.", is_final=True)

    assert states == [AppState.PROCESSING, AppState.READY]


@SEND_INPUT
@pytest.mark.parametrize("text", ["é👍", "Oi Débora, ", "tudo bem... ", "tudo bem? "])
def test_unicode_typing_sends_utf16_units(monkeypatch, text):
    import ctypes
    packets = []

    def send_input(count, inputs_ref, size):
        inputs = inputs_ref._obj
        assert count == 2
        assert size == ctypes.sizeof(inputs[0])
        # Windows INPUT includes the mouse union and pointer-sized extra info.
        assert size == (40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)
        packets.append([(p.type, p._input.ki.wVk, p._input.ki.wScan,
                         p._input.ki.dwFlags, p._input.ki.time,
                         bool(p._input.ki.dwExtraInfo)) for p in inputs])
        return count

    monkeypatch.setattr(ctypes, "windll", MagicMock())
    ctypes.windll.user32.SendInput.side_effect = send_input
    monkeypatch.setattr(de.time, "sleep", lambda s: None)

    de._type_text_ctypes(text)

    data = text.encode("utf-16-le")
    units = [int.from_bytes(data[i:i + 2], "little") for i in range(0, len(data), 2)]
    assert packets == [[(1, 0, unit, 4, 0, False), (1, 0, unit, 6, 0, False)]
                       for unit in units]


@pytest.fixture
def clipboard(monkeypatch):
    import pyperclip
    import keyboard

    state = {"text": "user clipboard", "pasted": [], "pending": False}
    monkeypatch.setattr(pyperclip, "paste", lambda: state["text"])
    monkeypatch.setattr(pyperclip, "copy", lambda text: state.update(text=text))

    def key(name):
        if name == "ctrl+v":
            state["pending"] = True

    def wait(seconds):
        # Model a target that reads the clipboard after Ctrl+V returns.
        if state["pending"]:
            state["pasted"].append(state["text"])
            state["pending"] = False

    monkeypatch.setattr(keyboard, "press_and_release", key)
    monkeypatch.setattr(de.time, "sleep", wait)
    return state


def test_drafts_and_final_paste_before_restoring_clipboard(clipboard):
    with patch.object(de, "_type_text_ctypes") as unicode_keys, \
            patch.object(de.threading, "Thread") as background:
        for text, final in [("Olá, tudo... ", False), (" bem? ", False), ("fim", True)]:
            (de.type_text if final else de.type_draft_text)(text)
            assert clipboard["pasted"][-1] == text
            assert clipboard["text"] == "user clipboard"

    unicode_keys.assert_not_called()
    background.assert_not_called()
    assert clipboard["pasted"] == ["Olá, tudo... ", " bem? ", "fim"]


def test_paste_does_not_overwrite_new_user_copy(clipboard, monkeypatch):
    def wait(seconds):
        if seconds == 0.5:
            clipboard["text"] = "new user copy"

    monkeypatch.setattr(de.time, "sleep", wait)
    de.type_draft_text("Olá")
    assert clipboard["text"] == "new user copy"


def test_failed_paste_restores_clipboard_and_releases_lock(clipboard):
    with patch("keyboard.press_and_release", side_effect=RuntimeError("paste failed")):
        with pytest.raises(RuntimeError, match="paste failed"):
            de.type_draft_text("Olá")

    assert clipboard["text"] == "user clipboard"
    assert de._clipboard_lock.acquire(blocking=False)
    de._clipboard_lock.release()
