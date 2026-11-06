"""The window in focus on KDE Plasma (X11 or Wayland), as KWin reports it.

Wayland gives clients no way to ask which window has focus. KWin can tell
a script, so this loads one into KWin that calls back over the session
D-Bus on every activation; a lookup is then a read of the last report.
The script is unloaded on close() and, being per process, never outlives
two runs.
"""
import os
import tempfile
import threading

_SCRIPT = """
function report(w) {
    callDBus("%(name)s", "/Focus", "%(name)s", "Activated",
             w ? String(w.internalId) : "", w ? String(w.resourceClass) : "",
             w ? String(w.caption) : "", w ? (w.pid || 0) : 0);
}
workspace.windowActivated.connect(report);
report(workspace.activeWindow);
"""


class FocusedWindow:
    __slots__ = ("id", "resource_class", "caption", "pid")

    def __init__(self, id, resource_class, caption, pid):
        self.id, self.resource_class, self.caption, self.pid = id, resource_class, caption, pid

    def __repr__(self):
        return f"FocusedWindow({self.resource_class!r}, pid={self.pid})"


class KWinFocus:
    def __init__(self):
        self._lock = threading.Lock()
        self._window = None
        self._started = False
        self._ready = threading.Event()
        self._server = None
        self._script_path = None
        self._plugin = f"debora-whisper-focus-{os.getpid()}"
        self._name = f"org.deborawhisper.Focus.p{os.getpid()}"
        self._kwin_owner = None  # KWin's unique bus name: the only caller we trust
        self.error = None

    def start(self, timeout: float = 1.0) -> bool:
        """Load the KWin script; False (with .error) outside KDE."""
        with self._lock:
            if self._started:
                return self.error is None
            self._started = True
        try:
            from jeepney.bus_messages import message_bus
            from jeepney.io.blocking import open_dbus_connection
            self._server = open_dbus_connection(bus="SESSION")
            if self._server.send_and_get_reply(message_bus.RequestName(self._name)).body[0] != 1:
                raise RuntimeError(f"D-Bus name {self._name} is taken")
            self._kwin_owner = self._server.send_and_get_reply(
                message_bus.GetNameOwner("org.kde.KWin")).body[0]
            threading.Thread(target=self._serve, name="kwin-focus", daemon=True).start()
            self._load_script()
        except Exception as exc:
            self.error = f"KWin focus tracking unavailable: {exc}"
            return False
        self._ready.wait(timeout)
        return True

    def _kwin(self, method, signature, body, path="/Scripting",
              interface="org.kde.kwin.Scripting"):
        from jeepney import DBusAddress, MessageType, new_method_call
        from jeepney.io.blocking import open_dbus_connection
        address = DBusAddress(path, bus_name="org.kde.KWin", interface=interface)
        with open_dbus_connection(bus="SESSION") as conn:
            reply = conn.send_and_get_reply(new_method_call(address, method, signature, body),
                                            timeout=2)
        if reply.header.message_type == MessageType.error:
            raise RuntimeError(f"KWin {method}: {reply.body}")
        return reply.body

    def _load_script(self):
        # KWin reads the file after run() returns: it must outlive the call,
        # so it goes only in close().
        fd, self._script_path = tempfile.mkstemp(prefix="debora-focus-", suffix=".js",
                                                 dir=os.environ.get("XDG_RUNTIME_DIR"))
        os.write(fd, (_SCRIPT % {"name": self._name}).encode())
        os.close(fd)
        # A stale script with our name (pid reuse) would report to us twice.
        self._kwin("unloadScript", "s", (self._plugin,))
        script_id = self._kwin("loadScript", "ss", (self._script_path, self._plugin))[0]
        if script_id < 0:
            raise RuntimeError("KWin refused the script")
        # Not /Scripting/Script<id>.run(): after an unload KWin can hand out
        # an id another loaded script still has, and run() starts that one.
        # start() runs every loaded script that is not running yet.
        self._kwin("start", "", ())

    def _serve(self):
        from jeepney import HeaderFields, MessageType, new_method_return
        conn = self._server
        while True:
            try:
                msg = conn.receive()
            except Exception:
                return  # connection closed
            header = msg.header
            if header.message_type != MessageType.method_call:
                continue
            # Any session process may call us; only KWin's reports count.
            if (header.fields.get(HeaderFields.member) == "Activated" and len(msg.body) == 4
                    and header.fields.get(HeaderFields.sender) == self._kwin_owner):
                uid, resource_class, caption, pid = msg.body
                self._window = FocusedWindow(uid, resource_class, caption, pid) if uid else None
                self._ready.set()
            try:
                conn.send(new_method_return(msg))
            except Exception:
                return

    def window(self):
        """The window in focus, or None when unknown."""
        if not self._started:
            self.start()
        return self._window

    def close(self):
        if self._started and self.error is None:
            try:
                self._kwin("unloadScript", "s", (self._plugin,))
            except Exception:
                pass
        if self._server is not None:
            try:
                self._server.close()
            except Exception:
                pass
        if self._script_path is not None:
            try:
                os.unlink(self._script_path)
            except OSError:
                pass
        self._window, self._server, self._script_path = None, None, None
