"""Linux provider.

Tested on Ubuntu with KDE Plasma (Wayland) and PipeWire. Each answer falls
back to the portable one from Platform when its tool is missing.
"""
import ctypes
import glob
import json
import os
import shutil
import subprocess
from pathlib import Path

from debora_whisper.platforms.base import Platform
from debora_whisper.platforms import linux_text

# cuDNN 9 opens its sub-libraries (ops, cnn, adv...) by name at run time:
# preloading only libcudnn.so.9 is not enough.
_LIBRARY_PATTERNS = ("lib*.so*",)


def avoid_level_zero_crash():
    """Without an NPU, keep the Level Zero loader from looking for drivers.

    OpenVINO's NPU plugin initializes Level Zero on every ov.Core(). With no
    driver installed, libze_loader 1.28 (Ubuntu 26.04) returns "uninitialized"
    the first time and segfaults the second, so the second Core of a process
    (a device query, a pipeline) took the app down. An empty driver list makes
    every lookup fail cleanly; the NPU is the only OpenVINO device that needs
    Level Zero. Set before any Core so that child processes inherit it.
    """
    if "ZE_ENABLE_ALT_DRIVERS" not in os.environ and not glob.glob("/dev/accel/accel*"):
        os.environ["ZE_ENABLE_ALT_DRIVERS"] = ""


def _headphone_port(sink: dict):
    """True/False from the active port's type, None when it has none.

    PipeWire reports the port type ("Headphones", "Headset", "Speaker") in
    the port's "type" field; older PulseAudio only in its name."""
    active = sink.get("active_port")
    if not active:
        return None
    for port in sink.get("ports", []):
        if port.get("name") != active:
            continue
        kind = (port.get("type") or "").lower()
        name = active.lower()
        if kind in ("headphones", "headset") or "headphone" in name or "headset" in name:
            return True
        if kind == "speaker" or "speaker" in name:
            return False
        return None
    return None


def default_sink_headphones():
    """Whether the default PipeWire/PulseAudio sink plays on headphones."""
    pactl = shutil.which("pactl")
    if not pactl:
        return None
    try:
        default = subprocess.run([pactl, "get-default-sink"], capture_output=True,
                                 text=True, timeout=2).stdout.strip()
        sinks = json.loads(subprocess.run([pactl, "-f", "json", "list", "sinks"],
                                          capture_output=True, text=True, timeout=2).stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    for sink in sinks:
        if sink.get("name") == default:
            return _headphone_port(sink)
    return None


_STOP_SIGNALS = ("SIGTSTP", "SIGTTIN", "SIGTTOU")


def _stop_without_keyboards(signum, frame):
    """SIGTSTP/SIGTTIN/SIGTTOU: give the keyboards back, then stop as the
    default action would."""
    import signal
    try:
        from debora_whisper.platforms import linux_keyboard
        linux_keyboard.suspend()
    except Exception:
        pass  # stopping must not depend on the keyboard module
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)
    # Running again: continued, or the kernel discarded the stop (orphaned
    # process group) and no SIGCONT will come. Either way, take them back.
    _grab_keyboards_again(signal.SIGCONT, frame)


def _grab_keyboards_again(signum, frame):
    """SIGCONT: rearm the stop handlers and grab the keyboards again."""
    import signal
    for name in _STOP_SIGNALS:
        signal.signal(getattr(signal, name), _stop_without_keyboards)
    try:
        from debora_whisper.platforms import linux_keyboard
        linux_keyboard.resume()
    except Exception:
        pass


class LinuxPlatform(Platform):
    name = "linux"

    def __init__(self):
        avoid_level_zero_crash()
        self._preloaded = set()
        self._paste_overrides = {}
        self._focus = None
        self._instance_locks = []

    def apply_config(self, config):
        self._paste_overrides = dict(config.get("paste_hotkeys") or {})

    def keyboard(self):
        from debora_whisper.platforms import linux_keyboard
        return linux_keyboard

    def _focused(self):
        if self._focus is None:
            from debora_whisper.platforms.kwin_focus import KWinFocus
            self._focus = KWinFocus()
            from debora_whisper.dictation_engine import log
            if self._focus.start():
                log("Window in focus tracked through KWin; terminals paste with Ctrl+Shift+V.")
            else:
                log(f"{self._focus.error}; pasting with {linux_text.UNKNOWN_PASTE}.")
        return self._focus.window()

    def paste_key(self):
        return linux_text.paste_key(self._focused(), self._paste_overrides)

    def type_text(self, text: str, auto_enter: bool = False):
        return linux_text.type_text(text, self.paste_key(), auto_enter)

    def type_draft(self, text: str):
        return self.type_text(text)

    def type_draft_keys(self, text: str):
        """Typing arbitrary characters needs the layout; paste instead."""
        return self.type_text(text)

    def backspace_text(self, count: int):
        return linux_text.backspace_text(count)

    def delete_text(self, count: int):
        return linux_text.delete_text(count)

    def press_enter(self):
        return linux_text.press_enter()

    def input_target(self):
        """The KWin window id; None outside KDE. Fields inside one window
        cannot be told apart."""
        window = self._focused()
        return window.id if window is not None else None

    def same_input_target(self, a, b) -> bool:
        return a is not None and a == b

    def tray_icon_class(self):
        """The D-Bus tray icon when the desktop has a StatusNotifierWatcher;
        pystray's (X11, no menu) otherwise."""
        from debora_whisper.platforms import linux_tray
        return linux_tray.SniIcon if linux_tray.watcher_available() else None

    def handle_termination(self):
        """SIGTERM (logout, kill, systemd) and SIGHUP (closed terminal)
        would end the process without cleanup, leaving the KWin script and
        the TTS server running: raise KeyboardInterrupt instead."""
        import signal
        for name in ("SIGTERM", "SIGHUP"):
            signal.signal(getattr(signal, name), signal.default_int_handler)
        # A stopped process (Ctrl+Z on debora-cli) would keep the keyboards
        # grabbed with nothing forwarding them. SIGSTOP can't be caught.
        for name in _STOP_SIGNALS:
            signal.signal(getattr(signal, name), _stop_without_keyboards)
        signal.signal(signal.SIGCONT, _grab_keyboards_again)

    def close(self):
        if self._focus is not None:
            self._focus.close()
            self._focus = None

    library_dir_names = ("lib",)

    def add_library_dir(self, path):
        """Preload the pip-installed CUDA libraries in `path` (globally visible).

        LD_LIBRARY_PATH is read once, at process start, so setting it here
        would be too late for CTranslate2; dlopen'ing each library with
        RTLD_GLOBAL makes the later dlopen by soname find the loaded one."""
        path = Path(path)
        if not path.is_dir():
            return
        # libnvblas would take over every BLAS call made in the process.
        pending = sorted({p for pattern in _LIBRARY_PATTERNS for p in path.glob(pattern)
                          if p.is_file() and p not in self._preloaded
                          and not p.name.startswith("libnvblas")})
        # Dependencies first: retry what failed until no progress is made.
        while pending:
            failed = []
            for lib in pending:
                try:
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                    self._preloaded.add(lib)
                except OSError:
                    failed.append(lib)
            if len(failed) == len(pending):
                break
            pending = failed

    def claim_single_instance(self, name: str) -> bool:
        """An exclusive lock on a file in the session's runtime directory;
        the kernel drops it when the process dies, however it dies. Without
        one, the config folder: a shared /tmp would let another user take
        the name first. A lock file that cannot be opened is unknown: True."""
        import fcntl
        import re
        from debora_whisper.paths import CONFIG_DIR
        folder = os.environ.get("XDG_RUNTIME_DIR") or CONFIG_DIR
        path = Path(folder) / (re.sub(r"[^\w.-]", "_", name) + ".lock")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(path, "a+")
        except OSError:
            return True
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self._instance_locks.append(handle)  # held until exit
        return True

    def boot_time(self):
        try:
            import psutil
            return psutil.boot_time()
        except Exception:
            return None

    def kill_tree(self, process):
        """Kill process and its children, without waiting for them.

        uv runs the real interpreter as a child, like on Windows."""
        if process.poll() is not None:
            return
        try:
            import psutil
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.Error:
                    pass
        except Exception:
            pass
        if process.poll() is None:
            process.kill()

    def output_is_headphones(self, name=None):
        # pactl answers for the default sink, which is the one sounddevice's
        # "default"/"pipewire" output plays on.
        if name is not None and name not in ("default", "pipewire", "pulse",
                                             self.default_output_name()):
            return None
        return default_sink_headphones()

    def open_folder(self, path):
        opener = shutil.which("xdg-open")
        if not opener:
            raise OSError("xdg-open not found")
        subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
