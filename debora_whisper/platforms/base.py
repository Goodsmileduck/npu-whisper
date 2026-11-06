"""What Débora asks of the operating system.

Each method has a portable answer, or "unknown" (None) when only the OS can
tell. A provider per OS (windows.py today) overrides what it knows, so the
rest of the app never checks sys.platform itself.
"""
import shutil
from enum import Enum
from pathlib import Path


class CaretReadStatus(Enum):
    UNSAFE = "unsafe"


class Platform:
    name = "generic"
    no_window = 0

    def native_hotkeys(self, hotkey, on_press, on_release, voice_key, on_voice, log):
        """An owned native hotkey service, or None for the keyboard backend."""
        return None

    def apply_config(self, config):
        """Settings for this provider, from the app's config."""
        pass

    def keyboard(self):
        """The keyboard-package API for hotkeys and keys (the package itself)."""
        import keyboard
        return keyboard

    def close(self):
        """Release what the provider holds, at shutdown."""
        pass

    def tray_icon_class(self):
        """A pystray.Icon replacement for this desktop, or None for pystray's."""
        return None

    def handle_termination(self):
        """Make the OS's "please quit" a Ctrl+C, so shutdown code runs."""
        pass

    def type_text(self, text: str, auto_enter: bool = False):
        """Keep the portable clipboard/keyboard path; native support is unknown."""
        from debora_whisper.platforms import windows_text
        return windows_text.type_text(text, auto_enter, fallback=self._type_text_ctypes)

    def _type_text_ctypes(self, text: str):
        """Native insertion is unknown: no-op."""
        pass

    def type_draft(self, text: str):
        """Drafts share the final text's serialized clipboard transaction."""
        return self.type_text(text)

    def type_draft_keys(self, text: str):
        """Insert an unreadable target's draft without touching the clipboard."""
        return self._type_text_ctypes(text)

    def backspace_text(self, count: int):
        """Erase a simple-character tail with individual Backspace keys."""
        from debora_whisper.platforms import windows_text
        return windows_text.backspace_text(count, fallback=self._backspace_text_ctypes)

    def _backspace_text_ctypes(self, count: int):
        """Native Backspace is unknown: no-op."""
        pass

    def can_backspace_draft(self, text: str) -> bool:
        from debora_whisper.platforms import windows_text
        return windows_text.can_backspace_draft(text)

    def delete_text(self, count: int):
        """Keep keyboard selection/deletion; native support is unknown."""
        from debora_whisper.platforms import windows_text
        return windows_text.delete_text(count, fallback=self._delete_text_ctypes)

    def _delete_text_ctypes(self, count: int):
        """Native deletion is unknown: no-op."""
        pass

    def press_enter(self):
        from debora_whisper.platforms import windows_text
        return windows_text.press_enter()

    def input_target(self):
        """Where keys land is unknown on unsupported systems."""
        return None

    def text_before_caret(self, count: int) -> str | CaretReadStatus | None:
        """Text before the caret, UNSAFE for selections, or None if unavailable.

        Count zero probes readability: return "" for a readable caret, even
        in an empty field; never treat unavailable text as an empty string.
        """
        return None

    def same_input_target(self, a, b) -> bool:
        """Unknown targets cannot be verified."""
        return False

    def nvidia_smi(self):
        """Absolute executable from PATH, never from the current folder."""
        found = shutil.which("nvidia-smi")
        if found is None or not Path(found).is_absolute():
            return None
        path = Path(found).resolve()
        return None if path.parent == Path.cwd().resolve() else str(path)

    # Where pip's nvidia-* wheels keep the libraries add_library_dir loads.
    library_dir_names = ("bin",)

    def add_library_dir(self, path):
        """Extra library directories are only needed on Windows today."""
        pass

    def claim_single_instance(self, name: str) -> bool:
        """False when another process already holds `name`; unknown: True."""
        return True

    def boot_time(self):
        """When the system last started (seconds since the epoch), or None."""
        return None

    def open_folder(self, path):
        """Show `path` in the file manager; OSError when that is unknown."""
        import os
        if not hasattr(os, "startfile"):
            raise OSError("no file manager known on this system")
        os.startfile(path)

    def kill_tree(self, process):
        """Kill a running process, without waiting for it."""
        if process.poll() is not None:
            return
        process.kill()

    def default_output_name(self):
        """Name of the device her voice plays on, as the audio API shows it."""
        try:
            import sounddevice as sd
            return sd.query_devices(kind="output")["name"]
        except Exception:
            return None

    def output_is_headphones(self, name=None):
        """True for headphones or a headset, False for speakers, None when
        unknown. With headphones her voice cannot reach the microphone."""
        return None
