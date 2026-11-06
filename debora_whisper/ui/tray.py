"""System tray icon manager using pystray."""

import threading
import pystray
from debora_whisper.ui.icons import get_icon, get_volume_icon
from debora_whisper.version import BuildInfo, display_version, package_version


def backend_title(title: str, icon_class=None) -> str:
    """The tooltip as the tray backend can show it. pystray's X11 one (Linux
    without a StatusNotifierWatcher) sets a Latin-1 window name and raised on
    the em dash; accented Latin letters still fit."""
    if not (icon_class or pystray.Icon).__module__.endswith("_xorg"):
        return title
    return title.replace("\u2014", "-").replace("\u2013", "-").encode(
        "latin-1", "replace").decode("latin-1")


class TrayManager:
    """Manages the system tray icon and context menu."""

    def __init__(self, on_toggle, on_quit, on_settings=None, on_history=None, on_hardware_event=None,
                 device="NPU", model="base", hotkey="ctrl+alt+d",
                 on_voice_chat=None, voice_chat_on=None):
        self._on_toggle = on_toggle
        # on_voice_chat() switches voice chat; voice_chat_on() ticks the item.
        self._on_voice_chat = on_voice_chat
        self._voice_chat_on = voice_chat_on or (lambda: False)
        self._on_quit = on_quit
        self._on_settings = on_settings
        self._on_history = on_history
        self._on_hardware_event = on_hardware_event
        self._device = device
        self._model = model
        self._hotkey = hotkey
        self._state = "loading"
        self._tooltip = "Débora Whisper — Loading..."
        self._version = (display_version(BuildInfo(package_version())), False)
        self._update_notified = False
        self._icon: pystray.Icon | None = None
        self._thread: threading.Thread | None = None
        
        # All pystray mutations happen on _render_thread. Setting the icon,
        # title or menu calls Shell_NotifyIcon, which can block on Explorer;
        # done from Tk callbacks it kept the main thread inside a callback,
        # where Tk swallows Ctrl+C's KeyboardInterrupt. Public methods only
        # record what to show and wake the renderer.
        self._level = 0.0
        self._anim_frame = 0
        self._wake = threading.Event()
        self._running = False
        self._render_thread: threading.Thread | None = None

    def _build_menu(self):
        """Build the right-click context menu with dynamic state text."""
        items = [
            pystray.MenuItem(self._version[0], None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                lambda _: "Stop Recording" if self._state == "recording" else "Start Recording",
                self._on_toggle_click,
                default=True,
                enabled=lambda _: self._state in ("ready", "recording"),
            ),
            pystray.Menu.SEPARATOR,
        ]

        if self._on_voice_chat:
            items.append(pystray.MenuItem(
                "Voice chat", lambda: self._on_voice_chat(),
                checked=lambda _: self._voice_chat_on()))

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
        from debora_whisper import platforms
        initial_icon = get_icon(self._state, voice_chat=self._voice_chat_on())
        icon_class = platforms.current().tray_icon_class() or pystray.Icon
        self._icon = icon_class(
            name="debora-whisper",
            icon=initial_icon,
            title=backend_title(self._tooltip, icon_class),
            menu=self._build_menu(),
        )

        import sys
        if sys.platform == "win32" and hasattr(self._icon, "_message_handlers"):
            WM_POWERBROADCAST = 0x021B
            PBT_APMRESUMEAUTOMATIC = 0x0012
            WM_DEVICECHANGE = 0x0219

            def _on_power_broadcast(wparam, lparam):
                if wparam == PBT_APMRESUMEAUTOMATIC and self._on_hardware_event:
                    self._on_hardware_event()
                return 1

            def _on_device_change(wparam, lparam):
                # 0x8000: DEVICEARRIVAL, 0x8004: DEVICEREMOVECOMPLETE, 0x0007: DEVNODES_CHANGED
                if wparam in (0x8000, 0x8004, 0x0007) and self._on_hardware_event:
                    self._on_hardware_event()
                return 1

            self._icon._message_handlers[WM_POWERBROADCAST] = _on_power_broadcast
            self._icon._message_handlers[WM_DEVICECHANGE] = _on_device_change

        self._thread = threading.Thread(target=self._icon.run, daemon=True)
        self._thread.start()

        self._running = True
        self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self._render_thread.start()

    def _render_loop(self):
        """Apply state, tooltip and icon frames to pystray. Only Windows calls
        whose result changed are made: a silent mic or a repeated state costs
        nothing."""
        shown_image = shown_title = shown_state = None
        shown_voice_chat = None
        shown_version = None
        while self._running:
            self._wake.clear()
            state, title = self._state, self._tooltip
            voice_chat = self._voice_chat_on()
            version = self._version
            if state == "recording":
                image = get_volume_icon(self._level, voice_chat=voice_chat)
                interval = 0.1
            elif state in ("loading", "processing", "speaking"):
                self._anim_frame += 1
                image = get_icon(state, self._anim_frame, voice_chat=voice_chat)
                interval = 0.15
            else:
                image = get_icon(state, voice_chat=voice_chat)
                interval = None  # static: sleep until the next update

            icon = self._icon
            if icon is not None and getattr(icon, "visible", True):
                try:
                    if image is not shown_image:
                        icon.icon = image
                        shown_image = image
                    if title != shown_title:
                        icon.title = backend_title(title, type(icon))
                        shown_title = title
                    if (state != shown_state or voice_chat != shown_voice_chat
                            or version != shown_version):
                        # Rebuild so the dynamic menu text follows the state.
                        icon.menu = self._build_menu()
                        icon.update_menu()
                        shown_state = state
                        shown_voice_chat = voice_chat
                        shown_version = version
                    if version[1] and not self._update_notified:
                        self._update_notified = True
                        if getattr(icon, "HAS_NOTIFICATION", False):
                            icon.notify("Atualização disponível", "Débora Whisper")
                except Exception:
                    pass  # icon torn down underneath us during shutdown
            else:
                interval = 0.1  # not shown yet: retry shortly

            self._wake.wait(interval)

    def refresh(self):
        """Wake the renderer after a mode change."""
        self._wake.set()

    def update_version(self, text: str, update_available: bool = False):
        """Schedule version text and an optional one-time notification."""
        self._version = (text, update_available)
        self._wake.set()

    def update_state(self, state_name: str, tooltip: str | None = None):
        """Record a new state/tooltip; the render thread applies it."""
        if state_name != self._state:
            self._anim_frame = 0
        self._state = state_name
        if tooltip:
            self._tooltip = tooltip
        self._wake.set()

    def update_audio_level(self, level: float):
        """Record the mic level; the render thread turns it into icon frames
        while recording."""
        self._level = level

    def update_info(self, device: str, model: str, hotkey: str):
        """Update the device/model/hotkey shown in the menu."""
        self._device = device
        self._model = model
        self._hotkey = hotkey

    def stop(self):
        """Stop the tray icon."""
        self._running = False
        self._wake.set()
        if self._icon:
            try:
                self._icon.stop()
            except Exception:
                pass
