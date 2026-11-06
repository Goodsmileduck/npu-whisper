"""The Linux desktop integration: .desktop shortcuts and the D-Bus tray
icon. No session bus is touched: connections are fakes."""
import sys
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux provider")

from debora_whisper.platforms.linux import LinuxPlatform  # noqa: E402


def test_desktop_exec_quotes_paths():
    from debora_whisper import shortcuts
    assert shortcuts._exec_arg("/usr/bin/python3") == "/usr/bin/python3"
    assert shortcuts._exec_arg("/home/a b/py") == '"/home/a b/py"'
    assert shortcuts._exec_arg("/x/$HOME/py") == '"/x/\\\\$HOME/py"'
    assert shortcuts._exec_arg("/x/50%/py") == '"/x/50%%/py"'


def test_install_and_remove_desktop_entries(monkeypatch, tmp_path):
    from debora_whisper import platforms, shortcuts
    monkeypatch.setattr(platforms, "_current", LinuxPlatform())
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(shortcuts, "CONFIG_DIR", tmp_path / "home")
    monkeypatch.setattr(shortcuts, "PNG_ICON_PATH", tmp_path / "home" / "icon.png")
    created = shortcuts.install(autostart=True)
    menu, autostart = (tmp_path / "data/applications/debora-whisper.desktop",
                       tmp_path / "config/autostart/debora-whisper.desktop")
    assert created == [menu, autostart]
    entry = menu.read_text()
    assert f"Exec={sys.executable} -I -m debora_whisper" in entry.replace('"', "")
    assert "Icon=" in entry and (tmp_path / "home" / "icon.png").exists()
    assert sorted(shortcuts.remove()) == sorted(created)
    assert not menu.exists() and not autostart.exists()


# -- the D-Bus tray icon -----------------------------------------------------------

class SentMessages:
    def __init__(self):
        self.sent = []

    def send(self, message):
        self.sent.append(message)


def tray_call(icon, path, iface, member, signature=None, body=()):
    from jeepney import DBusAddress, new_method_call
    msg = new_method_call(DBusAddress(path, bus_name=icon._bus_name, interface=iface),
                          member, signature, body)
    icon._conn.sent.clear()
    icon._dispatch(msg)
    (reply,) = icon._conn.sent
    return reply


@pytest.fixture
def tray():
    import pystray
    from PIL import Image
    from debora_whisper.platforms.linux_tray import SniIcon
    clicked = Mock()
    menu = pystray.Menu(
        pystray.MenuItem("Start Recording", clicked.toggle, default=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Voice chat", clicked.voice, checked=lambda _: True),
        pystray.MenuItem("Device: CUDA", None, enabled=False),
        pystray.MenuItem("Quit", clicked.quit),
    )
    icon = SniIcon("debora-whisper", Image.new("RGBA", (16, 16), (255, 0, 0, 128)),
                   "Débora Whisper — Ready", menu)
    icon._conn = SentMessages()
    return icon, clicked


def test_tray_menu_layout_follows_the_pystray_menu(tray):
    icon, _ = tray
    reply = tray_call(icon, "/MenuBar", "com.canonical.dbusmenu", "GetLayout", "iias", (0, -1, []))
    _revision, (root, _props, children) = reply.body
    items = [child[1] for child in children]
    assert root == 0
    assert items[0] == (1, {"label": ("s", "Start Recording"), "enabled": ("b", True)}, [])
    assert items[1][1] == {"type": ("s", "separator")}
    assert items[2][1]["toggle-type"] == ("s", "checkmark") and items[2][1]["toggle-state"] == ("i", 1)
    assert items[3][1]["enabled"] == ("b", False)


def test_tray_click_runs_the_item(tray):
    icon, clicked = tray
    tray_call(icon, "/MenuBar", "com.canonical.dbusmenu", "GetLayout", "iias", (0, -1, []))
    tray_call(icon, "/MenuBar", "com.canonical.dbusmenu", "Event", "isvu", (5, "clicked", ("s", ""), 0))
    import time
    deadline = time.time() + 2
    while not clicked.quit.called and time.time() < deadline:
        time.sleep(0.01)
    clicked.quit.assert_called_once()
    clicked.toggle.assert_not_called()


def test_tray_activate_runs_the_default_item(tray):
    icon, clicked = tray
    tray_call(icon, "/StatusNotifierItem", "org.kde.StatusNotifierItem", "Activate", "ii", (0, 0))
    import time
    deadline = time.time() + 2
    while not clicked.toggle.called and time.time() < deadline:
        time.sleep(0.01)
    clicked.toggle.assert_called_once()


def test_tray_properties_carry_title_and_argb_pixmaps(tray):
    icon, _ = tray
    reply = tray_call(icon, "/StatusNotifierItem", "org.freedesktop.DBus.Properties", "GetAll",
                      "s", ("org.kde.StatusNotifierItem",))
    props = reply.body[0]
    assert props["Title"] == ("s", "Débora Whisper — Ready")
    assert props["Menu"] == ("o", "/MenuBar")
    width, height, data = props["IconPixmap"][1][0]
    assert (width, height) == (16, 16) and data[:4] == bytes([128, 255, 0, 0])  # A, R, G, B


def test_tray_title_change_signals_the_desktop(tray):
    icon, _ = tray
    icon._conn.sent.clear()
    icon.title = "Débora Whisper — Recording"
    from jeepney import HeaderFields
    assert [m.header.fields[HeaderFields.member] for m in icon._conn.sent] == ["NewTitle", "NewToolTip"]
