"""The provider for the operating system Débora runs on."""
import ctypes
import locale
import os
import sys

from debora_whisper.platforms.base import Platform

_current = None


def default_language(supported_languages) -> str:
    """Map the user's OS UI locale to a supported language, defaulting to English."""
    try:
        if sys.platform == "win32":
            language_id = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            # es-419 (0x580A) and other regional IDs are missing from the
            # table; the primary language with the default sublanguage is not.
            name = (locale.windows_locale.get(language_id)
                    or locale.windows_locale.get((language_id & 0x3FF) | 0x400, ""))
        else:
            name = locale.getlocale()[0] or os.environ.get("LC_ALL") or os.environ.get("LANG", "")
        code = name.lower().replace("-", "_").split("_")[0].split(".")[0].split("@")[0]
        if code in supported_languages:
            return code
    except Exception:
        pass
    return "en"


def current() -> Platform:
    global _current
    if _current is None:
        if sys.platform == "win32":
            from debora_whisper.platforms.windows import WindowsPlatform
            _current = WindowsPlatform()
        elif sys.platform.startswith("linux"):
            from debora_whisper.platforms.linux import LinuxPlatform
            _current = LinuxPlatform()
        else:
            _current = Platform()
    return _current
