"""Child processes the app starts and must be able to end."""
import subprocess
import sys
from pathlib import Path

from debora_whisper import platforms

# No console window for children of the windowless (pythonw) app.
NO_WINDOW = platforms.current().no_window


def python_executable() -> str:
    """This interpreter for a child talking over stdin/stdout: python.exe
    rather than the Start Menu's pythonw.exe, which may have no standard
    streams. NO_WINDOW keeps it windowless."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").is_file():
        return str(exe.with_name("python.exe"))
    return sys.executable


def kill_tree(process: subprocess.Popen):
    """Kill process and its children as supported by the operating system."""
    platforms.current().kill_tree(process)
