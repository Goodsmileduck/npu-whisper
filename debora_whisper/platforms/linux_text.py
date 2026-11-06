"""Text insertion on Linux: the clipboard plus a paste key.

Terminals paste with Ctrl+Shift+V (Ctrl+V types a literal ^V there); other
apps with Ctrl+V. The window in focus picks one (KWin on KDE). Keys go
through linux_keyboard's virtual keyboard, so X11 and Wayland behave alike.
"""
import threading
import time

from debora_whisper.platforms import linux_keyboard as keyboard

DEFAULT_PASTE = "ctrl+v"
TERMINAL_PASTE = "ctrl+shift+v"
# When the window is unknown: most dictation goes to a terminal (Claude
# Code), and Ctrl+Shift+V also pastes in browsers.
UNKNOWN_PASTE = TERMINAL_PASTE
# Matched against the lowercased window class (Wayland app id).
TERMINALS = (
    "konsole", "yakuake", "gnome-terminal", "kgx", "ptyxis", "kitty", "alacritty",
    "foot", "wezterm", "ghostty", "tilix", "terminator", "xfce4-terminal",
    "mate-terminal", "lxterminal", "qterminal", "terminology", "cool-retro-term",
    "io.elementary.terminal", "deepin-terminal", "tabby", "warp", "contour", "rio",
    "st-256color", "urxvt", "termite",
)

_clipboard_lock = threading.Lock()


def paste_key(window, overrides=None) -> str:
    """The paste key for `window` (a kwin_focus.FocusedWindow or None).

    `overrides` maps a window class (or a substring of it) to a key, from
    the config's "paste_hotkeys"; "unknown" applies when no window is known."""
    overrides = {k.lower(): v for k, v in (overrides or {}).items()}
    if window is None or not window.resource_class:
        return overrides.get("unknown", UNKNOWN_PASTE)
    cls = window.resource_class.lower()
    if cls in overrides:
        return overrides[cls]
    for pattern, key in overrides.items():
        if pattern != "unknown" and pattern in cls:
            return key
    if any(name in cls for name in TERMINALS):
        return TERMINAL_PASTE
    return DEFAULT_PASTE


def type_text(text: str, paste: str, auto_enter: bool = False):
    """Paste `text` into the focused app, then give the clipboard back."""
    if not text:
        return
    import pyperclip
    with _clipboard_lock:
        try:
            old_clipboard = pyperclip.paste()
        except Exception:
            old_clipboard = None
        pyperclip.copy(text)
        try:
            time.sleep(0.05)
            keyboard.send(paste)
            time.sleep(0.1)
            if auto_enter:
                keyboard.send("enter")
        finally:
            # The app reads the clipboard asynchronously after the key.
            time.sleep(0.5)
            try:
                # Don't overwrite something the user copied meanwhile, nor
                # replace an image (read back as "") with empty text.
                if old_clipboard and pyperclip.paste() == text:
                    pyperclip.copy(old_clipboard)
            except Exception:
                pass


def backspace_text(count: int):
    for _ in range(max(count, 0)):
        keyboard.send("backspace")
        time.sleep(0.001)


def delete_text(count: int):
    """Select `count` characters left of the caret and delete them."""
    if count <= 0:
        return
    keyboard.press("shift")
    try:
        for _ in range(count):
            keyboard.send("left")
            time.sleep(0.001)
    finally:
        keyboard.release("shift")
    keyboard.send("backspace")


def press_enter():
    keyboard.send("enter")
