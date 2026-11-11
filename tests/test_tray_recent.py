"""Tray menu: the last five dictations, one click to the clipboard."""
from unittest.mock import MagicMock

import pytest

from debora_whisper.ui.tray import TrayManager


def _tray(texts, copy=None):
    return TrayManager(on_toggle=MagicMock(), on_quit=MagicMock(),
                       recent_texts=lambda: texts, on_copy_text=copy or MagicMock())


def test_label_is_one_line_truncated_and_escapes_ampersands():
    label = TrayManager._recent_label("a & b\n" + "x" * 100)
    assert "\n" not in label and "a && b" in label
    assert len(label) <= TrayManager.RECENT_LABEL_CHARS + 1
    assert label.endswith("…")


def test_menu_lists_at_most_five_and_click_copies():
    copy = MagicMock()
    items = list(_tray([f"texto {i}" for i in range(8)], copy)._recent_items())
    assert [i.text for i in items] == [f"texto {n}" for n in range(5)]
    items[2](MagicMock())
    copy.assert_called_once_with("texto 2")


def test_empty_history_shows_a_disabled_placeholder():
    (item,) = _tray([])._recent_items()
    assert item.text == "(nothing yet)" and not item.enabled


def test_failing_source_does_not_break_the_menu():
    tray = TrayManager(on_toggle=MagicMock(), on_quit=MagicMock(),
                       recent_texts=MagicMock(side_effect=RuntimeError),
                       on_copy_text=MagicMock())
    assert [i.text for i in tray._recent_items()] == ["(nothing yet)"]


def test_submenu_is_in_the_tray_menu():
    names = [i.text for i in _tray(["a"])._build_menu().items]
    assert "Recent texts" in names


def test_gui_recent_texts_newest_first_without_echoes():
    pytest.importorskip("customtkinter")
    from types import SimpleNamespace
    from debora_whisper.app import GUIApp

    history = [{"text": f"t{i}"} for i in range(7)]
    history[5]["voice_chat_status"] = "echo"
    history.append({"text": "  "})
    fake = SimpleNamespace(_engine=SimpleNamespace(history=history))
    assert GUIApp._recent_texts(fake) == ["t6", "t4", "t3", "t2", "t1"]
    assert GUIApp._recent_texts(SimpleNamespace(_engine=None)) == []
