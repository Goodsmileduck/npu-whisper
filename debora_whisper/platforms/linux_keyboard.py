"""Global hotkeys and key injection on Linux, without root.

The keyboard package needs root on Linux. This module offers the part of
its API the app uses (add_hotkey, hook, is_pressed, send...) on top of
evdev: it reads the keyboards in /dev/input and types through a virtual
keyboard in /dev/uinput, so it works the same on X11 and Wayland. The user
needs read access to /dev/input/event* (the "input" group) and write access
to /dev/uinput.

A hotkey registered with suppress=True must not reach the focused app:
Ctrl+Space held for push-to-talk would otherwise auto-repeat into it. That
takes an exclusive grab of each physical keyboard, whose events are then
re-emitted, unchanged except for the hotkey, through a virtual copy of the
device. The kernel releases a grab when its file closes, so a crashed or
killed app never leaves the keyboard dead.
"""
import queue
import selectors
import threading
import time

try:
    import evdev
    from evdev import ecodes
except ImportError:  # not on Linux, or the dependency is missing
    evdev = ecodes = None

VIRTUAL_PREFIX = "debora-whisper"
RESCAN_SECONDS = 2.0

_MODIFIER_NAMES = {
    "ctrl": ("KEY_LEFTCTRL", "KEY_RIGHTCTRL"),
    "control": ("KEY_LEFTCTRL", "KEY_RIGHTCTRL"),
    "left ctrl": ("KEY_LEFTCTRL",),
    "right ctrl": ("KEY_RIGHTCTRL",),
    "shift": ("KEY_LEFTSHIFT", "KEY_RIGHTSHIFT"),
    "left shift": ("KEY_LEFTSHIFT",),
    "right shift": ("KEY_RIGHTSHIFT",),
    "alt": ("KEY_LEFTALT", "KEY_RIGHTALT"),
    "left alt": ("KEY_LEFTALT",),
    "right alt": ("KEY_RIGHTALT",),
    "alt gr": ("KEY_RIGHTALT",),
    "windows": ("KEY_LEFTMETA", "KEY_RIGHTMETA"),
    "left windows": ("KEY_LEFTMETA",),
    "right windows": ("KEY_RIGHTMETA",),
    "super": ("KEY_LEFTMETA", "KEY_RIGHTMETA"),
    "meta": ("KEY_LEFTMETA", "KEY_RIGHTMETA"),
}
# keyboard-package names that are not just KEY_<NAME>.
_KEY_NAMES = {
    "enter": "KEY_ENTER", "return": "KEY_ENTER", "esc": "KEY_ESC",
    "escape": "KEY_ESC", "page up": "KEY_PAGEUP", "page down": "KEY_PAGEDOWN",
    "caps lock": "KEY_CAPSLOCK", "num lock": "KEY_NUMLOCK",
    "scroll lock": "KEY_SCROLLLOCK", "print screen": "KEY_SYSRQ",
    "del": "KEY_DELETE", "ins": "KEY_INSERT", "menu": "KEY_COMPOSE",
    "-": "KEY_MINUS", "=": "KEY_EQUAL", "[": "KEY_LEFTBRACE", "]": "KEY_RIGHTBRACE",
    ";": "KEY_SEMICOLON", "'": "KEY_APOSTROPHE", "`": "KEY_GRAVE",
    "\\": "KEY_BACKSLASH", ",": "KEY_COMMA", ".": "KEY_DOT", "/": "KEY_SLASH",
}
# Event names, as the keyboard package reports them (SoloKeyTap matches
# "right alt").
_EVENT_NAMES = {
    "KEY_LEFTCTRL": "ctrl", "KEY_RIGHTCTRL": "right ctrl",
    "KEY_LEFTSHIFT": "shift", "KEY_RIGHTSHIFT": "right shift",
    "KEY_LEFTALT": "alt", "KEY_RIGHTALT": "right alt",
    "KEY_LEFTMETA": "windows", "KEY_RIGHTMETA": "right windows",
    "KEY_ENTER": "enter", "KEY_ESC": "esc", "KEY_PAGEUP": "page up",
    "KEY_PAGEDOWN": "page down", "KEY_CAPSLOCK": "caps lock",
}
_MODIFIER_CODES = None


def _log(message):
    try:
        from debora_whisper.dictation_engine import log
        log(message)
    except Exception:
        pass


def _code(name: str) -> int:
    value = getattr(ecodes, name, None) if ecodes else None
    if not isinstance(value, int):
        raise ValueError(f"Key {name!r} is not mapped")
    return value


def key_to_scan_codes(key: str) -> tuple:
    """The evdev codes a key name stands for, e.g. "ctrl" -> both Ctrl keys.

    ValueError when the name is no key, like the keyboard package."""
    name = key.strip().lower()
    if not name or ecodes is None:
        raise ValueError(f"Key {key!r} is not mapped to any known key.")
    if name in _MODIFIER_NAMES:
        return tuple(_code(n) for n in _MODIFIER_NAMES[name])
    if name in _KEY_NAMES:
        return (_code(_KEY_NAMES[name]),)
    try:
        return (_code("KEY_" + name.replace(" ", "").upper()),)
    except ValueError:
        raise ValueError(f"Key {key!r} is not mapped to any known key.") from None


def _parse_keys(hotkey: str) -> list:
    """"ctrl+space" -> [(ctrl codes), (space codes)]. A lone "+" is the key."""
    parts = [p for p in hotkey.replace(" + ", "+").split("+")] if hotkey.strip() != "+" else ["+"]
    if any(not p.strip() for p in parts):
        raise ValueError(f"Hotkey {hotkey!r} has an empty key.")
    return [key_to_scan_codes(p) for p in parts]


def parse_hotkey(hotkey: str) -> tuple:
    """Steps of keys of codes, as the keyboard package nests them:
    "ctrl+space" -> (((ctrl codes), (space codes)),). One step only."""
    return (tuple(_parse_keys(hotkey)),)


def _modifier_codes() -> frozenset:
    global _MODIFIER_CODES
    if _MODIFIER_CODES is None:
        _MODIFIER_CODES = frozenset(_code(n) for names in _MODIFIER_NAMES.values() for n in names)
    return _MODIFIER_CODES


def event_name(code: int) -> str:
    names = ecodes.KEY.get(code) if ecodes else None
    if isinstance(names, (list, tuple)):
        names = names[0]
    if not names:
        return str(code)
    return _EVENT_NAMES.get(names, names.removeprefix("KEY_").lower())


class KeyboardEvent:
    """What hook() callbacks get: event_type "down"/"up", scan_code, name."""

    __slots__ = ("event_type", "scan_code", "name", "time")

    def __init__(self, event_type, scan_code, name, time=None):
        self.event_type, self.scan_code, self.name, self.time = event_type, scan_code, name, time

    def __repr__(self):
        return f"KeyboardEvent({self.name} {self.event_type})"


class _Hotkey:
    def __init__(self, steps, callback, suppress):
        self.steps, self.callback, self.suppress = steps, callback, suppress
        self.trigger = frozenset(steps[-1])
        self.modifiers = steps[:-1]

    def matches(self, code, pressed) -> bool:
        """`code` went down completing the combination, and no other
        modifier is held (Ctrl+Shift+Space is not Ctrl+Space)."""
        if code not in self.trigger:
            return False
        if not all(pressed & set(step) for step in self.modifiers):
            return False
        allowed = {c for step in self.modifiers for c in step} | self.trigger
        return not ((pressed & _modifier_codes()) - allowed)


def is_keyboard(device) -> bool:
    """A real keyboard: letter keys and space. Excludes the power button,
    lid switch and our own virtual devices."""
    if device.name.startswith(VIRTUAL_PREFIX):
        return False
    keys = device.capabilities().get(ecodes.EV_KEY, [])
    return ecodes.KEY_SPACE in keys and ecodes.KEY_A in keys


class _Listener:
    """Reads every keyboard on its own thread; callbacks run on another one,
    so a slow callback never delays the keys it forwards."""

    def __init__(self):
        self._lock = threading.RLock()
        self._hotkeys = []
        self._hooks = []
        self._pressed = set()  # physical keys held, any keyboard
        self._suppressed = set()  # codes whose down was eaten; eat the rest
        self._devices = {}  # path -> (InputDevice, mirror UInput or None)
        self._pending = set()  # paths to grab once no key is held on them
        self._skipped = set()  # paths that are no keyboard
        self._calls = queue.Queue()
        self._selector = selectors.DefaultSelector()
        self._stop = threading.Event()
        self._thread = None
        self._worker = None
        self._grab = False
        self._suspended = False  # between SIGTSTP and resume()
        self.error = None

    # -- registration --------------------------------------------------

    @property
    def listening_thread(self):
        """The reader thread, named like the keyboard package's."""
        return self._thread

    def add_hotkey(self, hotkey, callback, suppress=False):
        entry = _Hotkey(_parse_keys(hotkey), callback, suppress)
        with self._lock:
            self._hotkeys.append(entry)
            if suppress and not self._grab:
                self._grab = True
                self._regrab()
        self._ensure_running()
        return entry

    def remove_hotkey(self, entry):
        with self._lock:
            self._hotkeys = [h for h in self._hotkeys if h is not entry and h.callback is not entry]

    def hook(self, callback):
        with self._lock:
            self._hooks.append(callback)
        self._ensure_running()
        return callback

    def unhook(self, callback):
        with self._lock:
            self._hooks = [h for h in self._hooks if h is not callback]

    def unhook_all(self):
        """Remove every hotkey and hook and give the keyboards back."""
        with self._lock:
            self._hotkeys, self._hooks = [], []
        self.close()

    def is_pressed(self, hotkey) -> bool:
        steps = _parse_keys(hotkey) if isinstance(hotkey, str) else [(hotkey,)]
        with self._lock:
            return all(self._pressed & set(step) for step in steps)

    # -- devices -------------------------------------------------------

    def _ensure_running(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if evdev is None:
                raise ImportError("evdev is not installed; global hotkeys are unavailable.")
            self._stop.clear()
            self._scan()
            if self._grab:  # a reader that died gave its keyboards back
                self._regrab()
            if not self._devices:
                raise PermissionError(
                    "No keyboard is readable in /dev/input. Add your user to the "
                    "'input' group (sudo usermod -aG input $USER) and log in again.")
            self._thread = threading.Thread(target=self._run, name="evdev-keyboard", daemon=True)
            self._thread.start()
            # A reader that crashed leaves its worker running: a second one on
            # the same queue would run callbacks out of order.
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._dispatch, name="evdev-callbacks",
                                                daemon=True)
                self._worker.start()

    def _scan(self):
        """Open keyboards plugged in since the last scan."""
        try:
            paths = evdev.list_devices()
        except OSError:
            return
        for path in paths:
            if path in self._devices or path in self._skipped:
                continue
            try:
                device = evdev.InputDevice(path)
            except OSError:
                continue  # unreadable: no permission, or gone already
            if not is_keyboard(device):
                device.close()
                self._skipped.add(path)
                continue
            self._devices[path] = (device, None)
            self._selector.register(device, selectors.EVENT_READ, path)
            if self._grab:
                self._grab_device(path)

    def _grab_device(self, path):
        """Exclusive access to the device, forwarding through a virtual copy.

        Keys held at grab time would stay pressed forever for the desktop
        (it never sees their release), so a device with keys down is left
        pending; the reader retries it without blocking the other keyboards."""
        device, mirror = self._devices[path]
        if mirror is not None or self._suspended:
            return  # while suspended nothing would forward it; resume() grabs it
        try:
            held = device.active_keys()
        except OSError:
            held = []
        if held:
            self._pending.add(path)
            return
        self._pending.discard(path)
        try:
            mirror = evdev.UInput.from_device(device, name=f"{VIRTUAL_PREFIX} {device.name}")
        except (OSError, evdev.UInputError) as exc:
            self.error = f"cannot mirror {device.name}: {exc}; the hotkey is not suppressed"
            return
        try:
            device.grab()
        except OSError as exc:
            mirror.close()
            self.error = f"cannot grab {device.name}: {exc}; the hotkey is not suppressed"
            return
        self._devices[path] = (device, mirror)

    def _regrab(self):
        for path in list(self._devices):
            self._grab_device(path)

    def suspend(self):
        """Give every keyboard back before the process stops (Ctrl+Z, SIGTSTP):
        a stopped reader forwards nothing, VT switching and SysRq included."""
        with self._lock:
            self._suspended = True
            for path in list(self._devices):
                self._ungrab(path)

    def resume(self):
        """Grab the keyboards again after SIGCONT.

        Keys typed while stopped went to the shell; replaying them would
        fire a hotkey late. Drop them and take the held keys from the
        devices themselves, in case their buffers overflowed."""
        with self._lock:
            self._pressed.clear()
            self._suppressed.clear()
            for device, _mirror in self._devices.values():
                try:
                    while device.read_one() is not None:
                        pass
                    self._pressed.update(device.active_keys())
                except OSError:
                    pass  # unplugged: the reader drops it
            self._suspended = False
            if self._grab:
                self._regrab()

    def _ungrab(self, path, reason=None):
        """Give the device back to the desktop and keep reading it."""
        device, mirror = self._devices[path]
        self._pending.discard(path)
        if mirror is None:
            return
        self._devices[path] = (device, None)
        for close in (device.ungrab, mirror.close):
            try:
                close()
            except OSError:
                pass
        if reason:
            self.error = f"{reason}; the hotkey is not suppressed on {device.name}"
            _log(f"Keyboard: {self.error}")

    def _drop(self, path):
        self._pending.discard(path)
        device, mirror = self._devices.pop(path, (None, None))
        if device is None:
            return
        try:
            self._selector.unregister(device)
        except (KeyError, ValueError):
            pass
        for close in (lambda: device.ungrab() if mirror is not None else None,
                      device.close, mirror.close if mirror is not None else lambda: None):
            try:
                close()
            except OSError:
                pass

    def close(self):
        """Stop reading and release every keyboard."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(1)
        with self._lock:
            for path in list(self._devices):
                self._drop(path)
            self._skipped.clear()
            self._pending.clear()
            self._pressed.clear()
            self._suppressed.clear()
            self._grab = False
            self._thread = None
        self._calls.put(None)
        # A restart right after this must not reuse a worker about to exit.
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(1)

    # -- event loop ----------------------------------------------------

    def _run(self):
        """Whatever ends this thread, no keyboard may stay grabbed: nothing
        would forward its keys, and the user could not even switch VTs."""
        try:
            self._loop()
        except Exception as exc:
            _log(f"Keyboard reader stopped: {exc}")
        finally:
            with self._lock:
                for path in list(self._devices):
                    self._ungrab(path)

    def _loop(self):
        next_scan = time.monotonic() + RESCAN_SECONDS
        while not self._stop.is_set():
            try:
                ready = self._selector.select(timeout=0.25)
            except (OSError, ValueError):
                ready = []
            for key, _mask in ready:
                path = key.data
                with self._lock:
                    if path not in self._devices:
                        continue
                    device, mirror = self._devices[path]
                    try:
                        events = list(device.read())
                    except BlockingIOError:
                        continue
                    except OSError:  # unplugged
                        self._drop(path)
                        continue
                    if self._suspended:
                        # SIGCONT woke this thread before resume() ran: the
                        # keys were typed in the shell, not for the app.
                        continue
                    for event in events:
                        forward = self._handle(event)
                        if mirror is not None and forward:
                            try:
                                mirror.write_event(event)
                            except OSError as exc:
                                self._ungrab(path, f"cannot forward keys: {exc}")
                                mirror = None
            if self._pending:
                with self._lock:
                    for path in list(self._pending):
                        if path in self._devices:
                            self._grab_device(path)
                        else:
                            self._pending.discard(path)
            if time.monotonic() >= next_scan:
                with self._lock:
                    self._scan()
                next_scan = time.monotonic() + RESCAN_SECONDS

    def _handle(self, event) -> bool:
        """Track one event and queue what it triggers. False: eat it."""
        try:
            return self._track(event)
        except Exception as exc:  # a bug here must not swallow keys
            _log(f"Keyboard event failed: {exc}")
            return True

    def _track(self, event) -> bool:
        forward = True
        if event.type == ecodes.EV_KEY:
            code, value = event.code, event.value
            if value == 1:
                self._pressed.add(code)
                for hotkey in self._hotkeys:
                    if hotkey.matches(code, self._pressed):
                        self._calls.put((hotkey.callback, ()))
                        if hotkey.suppress:
                            self._suppressed.add(code)
            elif value == 0:
                self._pressed.discard(code)
            if code in self._suppressed:
                forward = False
                if value == 0:
                    self._suppressed.discard(code)
            if value in (0, 1) and self._hooks:
                kind = "down" if value == 1 else "up"
                key_event = KeyboardEvent(kind, code, event_name(code), event.timestamp())
                for hook in self._hooks:
                    self._calls.put((hook, (key_event,)))
        return forward

    def _dispatch(self):
        while True:
            item = self._calls.get()
            if item is None:
                if self._stop.is_set():
                    return
                continue
            callback, args = item
            try:
                callback(*args)
            except Exception as exc:  # a callback must not stop the others
                _log(f"Keyboard callback failed: {exc}")


class _Typist:
    """A virtual keyboard to press keys in the focused app."""

    def __init__(self):
        self._lock = threading.Lock()
        self._device = None

    def _ui(self):
        if self._device is None:
            if evdev is None:
                raise ImportError("evdev is not installed; cannot press keys.")
            keys = [c for c in ecodes.KEY if isinstance(c, int) and c < ecodes.BTN_MISC]
            self._device = evdev.UInput({ecodes.EV_KEY: keys},
                                        name=f"{VIRTUAL_PREFIX} virtual keyboard")
            # The desktop needs a moment to add a new keyboard before its
            # first keys count.
            time.sleep(0.2)
        return self._device

    def emit(self, steps, press: bool, release: bool):
        codes = [step[0] for step in steps]
        with self._lock:
            ui = self._ui()
            if press:
                for code in codes:
                    ui.write(ecodes.EV_KEY, code, 1)
                    ui.syn()
            if release:
                for code in reversed(codes):
                    ui.write(ecodes.EV_KEY, code, 0)
                    ui.syn()

    def close(self):
        with self._lock:
            if self._device is not None:
                self._device.close()
                self._device = None


_listener = _Listener()
_typist = _Typist()
# Like the keyboard package, add_hotkey and hook return a function that
# removes what they registered; the callback or that function then also
# works with remove_hotkey/unhook.
_removers = {}
_removers_lock = threading.Lock()


def _remover(remove, *handles):
    def remove_():
        with _removers_lock:
            for handle in handles + (remove_,):
                if _removers.get(handle) is remove_:
                    del _removers[handle]
        remove()
    with _removers_lock:
        for handle in handles + (remove_,):
            _removers[handle] = remove_
    return remove_


def _remove(handle, fallback):
    with _removers_lock:
        remove = _removers.get(handle)
    (remove or (lambda: fallback(handle)))()


def add_hotkey(hotkey, callback, args=(), suppress=False, **_ignored):
    target = (lambda: callback(*args)) if args else callback
    entry = _listener.add_hotkey(hotkey, target, suppress)
    return _remover(lambda: _listener.remove_hotkey(entry), callback, entry)


def remove_hotkey(handle):
    _remove(handle, _listener.remove_hotkey)


def hook(callback, suppress=False, **_ignored):
    """Callbacks only observe: suppression is add_hotkey's (suppress=True)."""
    _listener.hook(callback)
    return _remover(lambda: _listener.unhook(callback), callback)


def unhook(callback):
    _remove(callback, _listener.unhook)


def unhook_all():
    with _removers_lock:
        _removers.clear()
    _listener.unhook_all()


def unhook_all_hotkeys():
    with _listener._lock:
        _listener._hotkeys = []


def is_pressed(hotkey) -> bool:
    return _listener.is_pressed(hotkey)


def wait(hotkey=None):
    """Block forever (or until `hotkey` is pressed), like keyboard.wait()."""
    if hotkey is None:
        threading.Event().wait()
        return
    done = threading.Event()
    handle = add_hotkey(hotkey, done.set)
    done.wait()
    remove_hotkey(handle)


def send(hotkey, do_press=True, do_release=True):
    _typist.emit(_parse_keys(hotkey), do_press, do_release)


def press(hotkey):
    send(hotkey, True, False)


def release(hotkey):
    send(hotkey, False, True)


press_and_release = send


def suspend():
    _listener.suspend()


def resume():
    _listener.resume()


def last_error():
    """Why the hotkey could not be suppressed, or None."""
    return _listener.error
