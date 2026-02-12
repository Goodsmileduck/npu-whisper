"""System tray icon manager using pystray."""

import threading
import pystray
from ui.icons import STATE_ICONS


class TrayManager:
    """Manages the system tray icon and context menu."""

    def __init__(self, on_toggle, on_quit, on_settings=None, on_history=None,
                 device="NPU", model="base", hotkey="ctrl+alt+d"):
        self._on_toggle = on_toggle
        self._on_quit = on_quit
        self._on_settings = on_settings
        self._on_history = on_history
        self._device = device
        self._model = model
        self._hotkey = hotkey
        self._state = "loading"
        self._tooltip = "NPU Dictation — Loading..."
        self._icon: pystray.Icon | None = None
        self._thread: threading.Thread | None = None

    def _build_menu(self):
        """Build the right-click context menu with dynamic state text."""
        items = [
            pystray.MenuItem(
                lambda _: "Stop Recording" if self._state == "recording" else "Start Recording",
                self._on_toggle_click,
                default=True,
                enabled=lambda _: self._state in ("ready", "recording"),
            ),
            pystray.Menu.SEPARATOR,
        ]

        if self._on_history:
            items.append(pystray.MenuItem("History", lambda: self._on_history()))

        if self._on_settings:
            items.append(pystray.MenuItem("Settings", lambda: self._on_settings()))

        items.extend([
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                lambda _: f"Device: {self._device}",
                None,
                enabled=False,
            ),
            pystray.MenuItem(
                lambda _: f"Model: whisper-{self._model}",
                None,
                enabled=False,
            ),
            pystray.MenuItem(
                lambda _: f"Hotkey: {self._hotkey}",
                None,
                enabled=False,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._on_quit_click),
        ])

        return pystray.Menu(*items)

    def _on_toggle_click(self, icon=None, item=None):
        self._on_toggle()

    def _on_quit_click(self, icon=None, item=None):
        self._on_quit()

    def start(self):
        """Start the tray icon in a daemon thread."""
        initial_icon = STATE_ICONS[self._state]()
        self._icon = pystray.Icon(
            name="npu-dictation",
            icon=initial_icon,
            title=self._tooltip,
            menu=self._build_menu(),
        )
        self._thread = threading.Thread(target=self._icon.run, daemon=True)
        self._thread.start()

    def update_state(self, state_name: str, tooltip: str | None = None):
        """Update tray icon and tooltip for a new state."""
        self._state = state_name
        if tooltip:
            self._tooltip = tooltip

        if self._icon and self._icon.visible:
            icon_fn = STATE_ICONS.get(state_name)
            if icon_fn:
                self._icon.icon = icon_fn()
            self._icon.title = self._tooltip
            # Force menu rebuild so dynamic text updates
            self._icon.menu = self._build_menu()
            self._icon.update_menu()

    def update_info(self, device: str, model: str, hotkey: str):
        """Update the device/model/hotkey shown in the menu."""
        self._device = device
        self._model = model
        self._hotkey = hotkey

    def stop(self):
        """Stop the tray icon."""
        if self._icon:
            try:
                self._icon.stop()
            except Exception:
                pass
