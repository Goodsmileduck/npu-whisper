"""Windows text insertion, with the existing portable clipboard/keyboard path.

Native APIs are imported only in the Windows helpers. Platform reuses the
keyboard path with no-op native fallbacks when support is unknown.
"""
import threading
import time
import unicodedata


def input_target():
    """Where typed keys land: (foreground window, focused control), or None.

    Comparing the pair, not just the window, also catches focus moving to
    another field of the same window in native apps. Browsers and Electron
    apps report one handle per window, so a click elsewhere on the same page
    goes unnoticed.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class GUITHREADINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("hwndActive", wintypes.HWND),
                ("hwndFocus", wintypes.HWND),
                ("hwndCapture", wintypes.HWND),
                ("hwndMenuOwner", wintypes.HWND),
                ("hwndMoveSize", wintypes.HWND),
                ("hwndCaret", wintypes.HWND),
                ("rcCaret", wintypes.RECT),
            ]

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        info = GUITHREADINFO(cbSize=ctypes.sizeof(GUITHREADINFO))
        thread_id = user32.GetWindowThreadProcessId(hwnd, None)
        focus = info.hwndFocus if user32.GetGUIThreadInfo(thread_id, ctypes.byref(info)) else None
        return (hwnd, focus)
    except Exception:
        return None


def same_input_target(a, b) -> bool:
    """Whether keys still land where they landed before.

    GetGUIThreadInfo briefly reports no focused control (seen in about one
    poll in 300), so a missing focus on either side only compares windows.
    """
    if a is None or b is None or a[0] != b[0]:
        return False
    return a[1] is None or b[1] is None or a[1] == b[1]


def type_draft(text: str):
    """Paste live drafts through the same serialized path as final text."""
    type_text(text)


def can_backspace_draft(text: str) -> bool:
    """Only allow characters with unambiguous single-Backspace deletion.

    Reject non-BMP/surrogates, marks, controls (including joiners), and Hangul
    Jamo sequences. Terminals disagree on code-unit versus grapheme deletion.
    Precomposed accents such as é remain supported.
    """
    return all(ord(c) <= 0xFFFF and unicodedata.category(c)[0] not in "MC"
               and not (0x1100 <= ord(c) <= 0x11FF
                        or 0xA960 <= ord(c) <= 0xA97F
                        or 0xD7B0 <= ord(c) <= 0xD7FF) for c in text)


def backspace_text(count: int, *, fallback=None):
    """Trim a terminal tail without Shift or cursor movement."""
    if count <= 0:
        return
    try:
        import keyboard
        for _ in range(count):
            keyboard.send("backspace")
            time.sleep(0.001)
    except ImportError:
        (fallback or _backspace_text_ctypes)(count)


def _backspace_text_ctypes(count: int):
    import ctypes
    for _ in range(count):
        ctypes.windll.user32.keybd_event(0x08, 0, 0, 0)
        ctypes.windll.user32.keybd_event(0x08, 0, 0x0002, 0)
        time.sleep(0.001)


def delete_text(count: int, *, fallback=None):
    """Delete the specified number of characters using Shift+Left selection."""
    if count <= 0:
        return
    try:
        import keyboard
        keyboard.press("shift")
        for _ in range(count):
            keyboard.send("left")
            time.sleep(0.001)
        keyboard.release("shift")
        keyboard.send("backspace")
    except ImportError:
        (fallback or _delete_text_ctypes)(count)


def _delete_text_ctypes(count: int):
    import ctypes
    user32 = ctypes.windll.user32
    KEYEVENTF_KEYUP = 0x0002
    VK_BACK = 0x08
    VK_SHIFT = 0x10
    VK_LEFT = 0x25
    
    # Press Shift
    user32.keybd_event(VK_SHIFT, 0, 0, 0)
    # Press Left Arrow `count` times
    for _ in range(count):
        user32.keybd_event(VK_LEFT, 0, 0, 0)
        user32.keybd_event(VK_LEFT, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.001)
    # Release Shift
    user32.keybd_event(VK_SHIFT, 0, KEYEVENTF_KEYUP, 0)
    # Press Backspace
    user32.keybd_event(VK_BACK, 0, 0, 0)
    user32.keybd_event(VK_BACK, 0, KEYEVENTF_KEYUP, 0)


_clipboard_lock = threading.Lock()


def type_text(text: str, auto_enter: bool = False, *, fallback=None):
    """Type text into the currently active window using keyboard simulation."""
    if not text:
        return

    try:
        # Try pyperclip + keyboard for reliable pasting
        import pyperclip
        import keyboard

        # Complete restoration before another draft/final uses the clipboard.
        # A detached restore can otherwise overwrite the next text before
        # its Ctrl+V is processed. Unicode key packets are only a fallback:
        # live text should reach the target as one paste, not individual keys.
        with _clipboard_lock:
            try:
                old_clipboard = pyperclip.paste()
            except Exception:
                old_clipboard = None

            pyperclip.copy(text)
            try:
                time.sleep(0.05)
                keyboard.press_and_release("ctrl+v")
                time.sleep(0.1)
                if auto_enter:
                    keyboard.press_and_release("enter")
            finally:
                time.sleep(0.5)
                try:
                    # Don't overwrite something the user copied meanwhile.
                    if old_clipboard is not None and pyperclip.paste() == text:
                        pyperclip.copy(old_clipboard)
                except Exception:
                    pass

    except ImportError:
        # Fallback: use ctypes SendInput on Windows
        from debora_whisper.dictation_engine import log
        log("pyperclip/keyboard not found, using ctypes fallback")
        (fallback or _type_text_ctypes)(text)


def _type_text_ctypes(text: str):
    """Fallback text typing using Windows ctypes."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    
    # Use SendInput with Unicode characters
    INPUT_KEYBOARD = 1
    KEYEVENTF_UNICODE = 0x0004
    KEYEVENTF_KEYUP = 0x0002

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        ]

    class INPUT(ctypes.Structure):
        class _INPUT(ctypes.Union):
            _fields_ = [
                ("ki", KEYBDINPUT),
                ("mi", MOUSEINPUT),
                ("hi", HARDWAREINPUT),
            ]
        _fields_ = [("type", wintypes.DWORD), ("_input", _INPUT)]

    # KEYEVENTF_UNICODE takes UTF-16 code units; characters outside the
    # BMP (e.g. emoji) are sent as their surrogate pair.
    data = text.encode("utf-16-le")
    for unit in (int.from_bytes(data[i:i + 2], "little") for i in range(0, len(data), 2)):
        inputs = (INPUT * 2)()
        # Key down
        inputs[0].type = INPUT_KEYBOARD
        inputs[0]._input.ki.wScan = unit
        inputs[0]._input.ki.dwFlags = KEYEVENTF_UNICODE
        # Key up
        inputs[1].type = INPUT_KEYBOARD
        inputs[1]._input.ki.wScan = unit
        inputs[1]._input.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP

        user32.SendInput(2, ctypes.byref(inputs), ctypes.sizeof(INPUT))
        time.sleep(0.002)


def press_enter():
    import keyboard
    keyboard.press_and_release("enter")
