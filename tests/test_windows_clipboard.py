"""The Windows paste path must keep dictation out of Win+V history."""
import ctypes
import sys

import pytest

from debora_whisper.platforms import windows_text


class _Fn:
    def __init__(self, impl):
        self.impl = impl
        self.argtypes = self.restype = None

    def __call__(self, *args):
        return self.impl(*args)


class FakeClipboard:
    """Records one clipboard transaction through fake user32/kernel32."""

    def __init__(self, open_ok=True, fail_set=()):
        self.open_ok, self.fail_set = open_ok, set(fail_set)
        self.calls, self.formats, self.memory, self.ids = [], {}, {}, {}
        self.user32 = self._dll(OpenClipboard=self._open, CloseClipboard=self._close,
                                EmptyClipboard=self._empty,
                                RegisterClipboardFormatW=self._register,
                                SetClipboardData=self._set)
        self.kernel32 = self._dll(GlobalAlloc=self._alloc, GlobalLock=self._lock,
                                  GlobalUnlock=lambda h: 1, GlobalFree=self._free)

    @staticmethod
    def _dll(**fns):
        return type("Dll", (), {k: _Fn(v) for k, v in fns.items()})()

    def _open(self, hwnd):
        self.calls.append("open")
        return self.open_ok

    def _close(self):
        self.calls.append("close")
        return 1

    def _empty(self):
        self.calls.append("empty")
        return 1

    def _register(self, name):
        return self.ids.setdefault(name, 0xC000 + len(self.ids))

    def _alloc(self, flags, size):
        handle = len(self.memory) + 1
        self.memory[handle] = ctypes.create_string_buffer(size)
        return handle

    def _lock(self, handle):
        return ctypes.addressof(self.memory[handle])

    def _free(self, handle):
        self.calls.append(("free", handle))

    def _set(self, fmt, handle):
        if fmt in self.fail_set:
            return None
        self.formats[fmt] = bytes(self.memory[handle].raw)
        return handle

    def install(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            ctypes, "WinDLL",
            lambda name, **kw: self.user32 if name == "user32" else self.kernel32,
            raising=False)


def test_text_and_history_exclusions_share_one_transaction(monkeypatch):
    fake = FakeClipboard()
    fake.install(monkeypatch)
    assert windows_text._copy_private("Olá 👍") is True
    assert fake.calls == ["open", "empty", "close"]
    assert fake.formats[13] == "Olá 👍".encode("utf-16-le") + b"\0\0"
    by_name = {name: fake.formats[fake.ids[name]] for name, _ in windows_text._PRIVATE_FORMATS}
    assert by_name["CanIncludeInClipboardHistory"] == (0).to_bytes(4, "little")
    assert by_name["CanUploadToCloudClipboard"] == (0).to_bytes(4, "little")
    assert "ExcludeClipboardContentFromMonitorProcessing" in by_name


def test_busy_clipboard_falls_back(monkeypatch):
    fake = FakeClipboard(open_ok=False)
    fake.install(monkeypatch)
    monkeypatch.setattr(windows_text.time, "sleep", lambda s: None)
    assert windows_text._copy_private("x") is False
    assert "close" not in fake.calls


def test_text_failure_closes_and_falls_back(monkeypatch):
    fake = FakeClipboard(fail_set={13})
    fake.install(monkeypatch)
    assert windows_text._copy_private("x") is False
    assert fake.calls[-1] == "close"
    assert ("free", 1) in fake.calls


def test_non_windows_uses_pyperclip(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert windows_text._copy_private("x") is False


def test_copy_falls_back_to_pyperclip(monkeypatch):
    copied = []
    monkeypatch.setattr(windows_text, "_copy_private", lambda text: False)
    windows_text._copy(type("P", (), {"copy": staticmethod(copied.append)}), "abc")
    assert copied == ["abc"]


def test_copy_skips_pyperclip_when_private_copy_works(monkeypatch):
    monkeypatch.setattr(windows_text, "_copy_private", lambda text: True)
    windows_text._copy(type("P", (), {"copy": staticmethod(pytest.fail)}), "abc")
