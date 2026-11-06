"""Start Menu shortcuts and the single-instance guard (Windows only)."""
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows shortcuts")

from debora_whisper import shortcuts  # noqa: E402


@pytest.fixture
def appdata(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
    monkeypatch.setattr(shortcuts, "ICON_PATH", tmp_path / "npw.ico")
    monkeypatch.setattr(shortcuts, "LEGACY_ICON_PATH", tmp_path / "legacy.ico")
    monkeypatch.setattr(shortcuts, "CONFIG_DIR", tmp_path / "home")
    return tmp_path


def _lnk(path):
    script = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:L);"
              "\"$($s.TargetPath)|$($s.Arguments)|$($s.IconLocation)\"")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                         env={**os.environ, "L": str(path)},
                         capture_output=True, text=True, check=True).stdout.strip()
    return out.split("|")


def test_install_creates_a_console_less_start_menu_shortcut(appdata):
    created = shortcuts.install()
    paths = shortcuts.shortcut_paths()
    assert created == [paths["start_menu"]] and paths["start_menu"].exists()
    assert not paths["startup"].exists()
    target, args, icon = _lnk(paths["start_menu"])
    assert target.lower() == str(shortcuts.gui_python()).lower()
    assert target.lower().endswith("pythonw.exe")
    assert args == "-I -m debora_whisper"
    assert icon.lower().startswith(str(shortcuts.ICON_PATH).lower())


def test_autostart_and_remove(appdata):
    shortcuts.install(autostart=True)
    paths = shortcuts.shortcut_paths()
    assert paths["startup"].exists()
    assert sorted(shortcuts.remove()) == sorted(paths.values())
    assert not any(p.exists() for p in paths.values())
    assert not shortcuts.ICON_PATH.exists()
    assert shortcuts.remove() == []


def test_pre_rename_shortcuts_are_replaced_keeping_autostart(appdata):
    legacy = shortcuts.shortcut_paths(shortcuts.LEGACY_SHORTCUT_NAME)
    for path in legacy.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    shortcuts.LEGACY_ICON_PATH.write_bytes(b"")
    created = shortcuts.install()
    assert sorted(created) == sorted(shortcuts.shortcut_paths().values())
    assert not any(p.exists() for p in legacy.values())
    assert not shortcuts.LEGACY_ICON_PATH.exists()


def test_remove_also_removes_pre_rename_shortcuts(appdata):
    legacy = shortcuts.shortcut_paths(shortcuts.LEGACY_SHORTCUT_NAME)["start_menu"]
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(b"")
    assert shortcuts.remove() == [legacy]
    assert not legacy.exists()


def test_paths_with_quotes_do_not_break_the_script(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "it's $(x) dir"))
    monkeypatch.setattr(shortcuts, "ICON_PATH", tmp_path / "npw.ico")
    monkeypatch.setattr(shortcuts, "LEGACY_ICON_PATH", tmp_path / "legacy.ico")
    monkeypatch.setattr(shortcuts, "CONFIG_DIR", tmp_path / "home")
    shortcuts.install()
    assert shortcuts.shortcut_paths()["start_menu"].exists()


def test_missing_appdata_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("APPDATA", raising=False)
    with pytest.raises(RuntimeError, match="APPDATA"):
        shortcuts.shortcut_paths()


def test_second_tray_app_is_refused():
    import ctypes
    import uuid
    from debora_whisper import app
    # A name of its own, so a running tray app does not fail this test.
    name = "Local\\debora-whisper-test-" + uuid.uuid4().hex
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    handles = []
    try:
        assert app._claim_single_instance(name) is True
        handles.append(app._instance_mutex)
        assert app._claim_single_instance(name) is False
        handles.append(app._instance_mutex)
    finally:
        for h in handles:
            kernel32.CloseHandle(h)


@pytest.mark.parametrize("argv", [
    ["--install-shortcut", "--remove-shortcut"],
    ["--autostart"],
    ["--remove-shortcut", "--autostart"],
])
def test_conflicting_shortcut_flags_are_rejected(argv, monkeypatch):
    from debora_whisper import app
    monkeypatch.setattr(sys, "argv", ["debora", *argv])
    with pytest.raises(SystemExit) as e:
        app.main()
    assert e.value.code == 2
