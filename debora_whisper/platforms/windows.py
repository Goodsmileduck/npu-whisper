"""Windows provider.

Each playback endpoint's form factor is in the registry, so telling
headphones from speakers needs no extra package.
"""
import os
import subprocess
from pathlib import Path

from debora_whisper.platforms.base import CaretReadStatus, Platform
from debora_whisper.platforms import windows_text, windows_uia

_RENDER = r"SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Render"
_DESCRIPTION = "{a45c254e-df1c-4efd-8020-67d146a850e0},2"  # "Headphones"
_DEVICE_NAMES = ("{b3f8fa53-0004-438e-9003-51a46e139bfc},26",  # "LE-HD 450BT"
                 "{b3f8fa53-0004-438e-9003-51a46e139bfc},6")   # "Realtek(R) Audio"
_FORM_FACTOR = "{1da5d803-d492-4edd-8c23-e0c0ffee7f0e},0"
_ACTIVE = 1
_HEADPHONES, _HEADSET = 3, 5
# MME, sounddevice's default host API, cuts device names at 31 characters.
_MME_NAME_CHARS = 31


def render_endpoints():
    """[(name as audio APIs show it, form factor)] of active playback devices."""
    import winreg
    endpoints = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _RENDER)
    except OSError:
        return []
    with root:
        index = 0
        while True:
            try:
                guid = winreg.EnumKey(root, index)
            except OSError:
                break
            index += 1
            try:
                with winreg.OpenKey(root, guid) as key:
                    if winreg.QueryValueEx(key, "DeviceState")[0] != _ACTIVE:
                        continue
                with winreg.OpenKey(root, guid + r"\Properties") as props:
                    values = {}
                    for name in (_DESCRIPTION, _FORM_FACTOR, *_DEVICE_NAMES):
                        try:
                            values[name] = winreg.QueryValueEx(props, name)[0]
                        except OSError:
                            pass
            except OSError:
                continue
            device = next((values[n] for n in _DEVICE_NAMES if values.get(n)), None)
            if values.get(_DESCRIPTION) and device:
                endpoints.append((f"{values[_DESCRIPTION]} ({device})",
                                  values.get(_FORM_FACTOR)))
    return endpoints


def is_headphones(name, endpoints):
    """True/False by the matching endpoint's form factor, None if not found."""
    if not name:
        return None
    matches = {factor for full, factor in endpoints
               if full == name or (len(name) >= _MME_NAME_CHARS and full.startswith(name))}
    if len(matches) != 1:
        return None
    return matches.pop() in (_HEADPHONES, _HEADSET)


class WindowsPlatform(Platform):
    name = "windows"

    def native_hotkeys(self, hotkey, on_press, on_release, voice_key, on_voice, log):
        from debora_whisper.platforms.windows_hotkeys import WindowsHotkeys
        return WindowsHotkeys(hotkey, on_press, on_release, voice_key, on_voice, log)

    def type_text(self, text: str, auto_enter: bool = False):
        return windows_text.type_text(text, auto_enter)

    def _type_text_ctypes(self, text: str):
        return windows_text._type_text_ctypes(text)

    def type_draft(self, text: str):
        return windows_text.type_draft(text)

    def delete_text(self, count: int):
        return windows_text.delete_text(count)

    def _delete_text_ctypes(self, count: int):
        return windows_text._delete_text_ctypes(count)

    def _backspace_text_ctypes(self, count: int):
        return windows_text._backspace_text_ctypes(count)

    def input_target(self):
        return windows_text.input_target()

    def text_before_caret(self, count: int) -> str | CaretReadStatus | None:
        return windows_uia.text_before_caret(count)

    def same_input_target(self, a, b) -> bool:
        return windows_text.same_input_target(a, b)

    def nvidia_smi(self):
        system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        if system.is_absolute() and system.is_file():
            return str(system)
        return super().nvidia_smi()

    def add_library_dir(self, path):
        os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(path))

    def boot_time(self):
        import ctypes
        import time
        ctypes.windll.kernel32.GetTickCount64.restype = ctypes.c_uint64
        return time.time() - ctypes.windll.kernel32.GetTickCount64() / 1000

    @property
    def no_window(self):
        return subprocess.CREATE_NO_WINDOW

    def kill_tree(self, process):
        """Kill process and its children, without waiting for them.

        On Windows a venv's python.exe and uv both run the real interpreter as a
        child: killing only the parent leaves that child running, holding its
        GPU or NPU."""
        if process.poll() is not None:
            return
        result = subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                                capture_output=True, creationflags=self.no_window)
        if result.returncode != 0 and process.poll() is None:
            # taskkill may be denied process enumeration even though our
            # existing Popen handle can terminate the child. Otherwise its
            # pipe readers survive the shutdown as well.
            process.kill()

    def output_is_headphones(self, name=None):
        name = self.default_output_name() if name is None else name
        return is_headphones(name, render_endpoints())
