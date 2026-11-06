"""Build identity and read-only update checks."""

import json
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from urllib.request import urlopen

from debora_whisper import __version__, platforms


TIMEOUT = 5
CHECK_INTERVAL = 6 * 60 * 60
MAX_PYPI_BYTES = 4 * 1024 * 1024
PYPI_URL = "https://pypi.org/pypi/debora-whisper/json"
UP_TO_DATE = "up to date"
UPDATE_AVAILABLE = "update available"
AHEAD = "ahead/local changes"
UNKNOWN = "unknown"

_STATUS_TEXT = {
    UP_TO_DATE: "✓ atualizada",
    UPDATE_AVAILABLE: "⬆ atualização disponível",
    AHEAD: "✓ adiantada / alterações locais",
    UNKNOWN: "? não verificada",
}


def package_version() -> str:
    try:
        return metadata.version("debora-whisper")
    except Exception:
        return __version__


@dataclass(frozen=True)
class BuildInfo:
    version: str
    root: Path | None = None
    head: str = ""
    commit: str = ""
    date: str = ""
    dirty: bool = False

    @property
    def label(self) -> str:
        label = f"Débora Whisper {self.version}"
        if self.commit:
            label += f" · {self.commit}{'+dirty' if self.dirty else ''} ({self.date})"
        return label


def display_version(build: BuildInfo, status: str = UNKNOWN) -> str:
    return f"{build.label} · {_STATUS_TEXT[status]}"


def _git(directory: Path, *args: str, allowed=(0,)):
    result = subprocess.run(
        ["git", *args], cwd=directory, capture_output=True, text=True,
        stdin=subprocess.DEVNULL,
        encoding="utf-8", errors="replace", timeout=TIMEOUT,
        creationflags=platforms.current().no_window,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never",
             "GIT_OPTIONAL_LOCKS": "0", "GIT_NO_LAZY_FETCH": "1",
             "SSH_ASKPASS_REQUIRE": "never"},
    )
    if result.returncode not in allowed:
        raise RuntimeError(f"git {args[0]} failed (exit {result.returncode})")
    return result


def get_build_info() -> BuildInfo:
    """Use the package's checkout, never the launch directory."""
    version = package_version()
    directory = Path(__file__).resolve().parent
    try:
        root = Path(_git(directory, "rev-parse", "--show-toplevel").stdout.strip())
        # A wheel's site-packages may happen to sit inside another checkout.
        _git(directory, "ls-files", "--error-unmatch", "__init__.py")
        commit, timestamp = _git(root, "log", "-1", "--format=%h %cI").stdout.split()
        head = _git(root, "rev-parse", "HEAD").stdout.strip()
        dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=normal").stdout)
        return BuildInfo(version, root, head, commit, timestamp.split("T")[0], dirty)
    except Exception:
        return BuildInfo(version)


def _git_status(build: BuildInfo) -> str:
    root = build.root
    branch = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD", allowed=(0, 1))
    remote = "origin"
    if branch.returncode == 0:
        configured = _git(root, "config", "--get", f"branch.{branch.stdout.strip()}.remote",
                          allowed=(0, 1)).stdout.strip()
        if configured and configured != "." and not configured.startswith("-"):
            remote = configured
    output = _git(root, "ls-remote", "--exit-code", remote, "refs/heads/main").stdout.split()
    if (len(output) != 2 or output[1] != "refs/heads/main"
            or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", output[0])):
        raise ValueError("remote main has no valid commit")
    remote_head = output[0]
    if remote_head == build.head:
        return AHEAD if build.dirty else UP_TO_DATE
    present = _git(root, "rev-parse", "--verify", "--quiet", f"{remote_head}^{{commit}}",
                   allowed=(0, 1))
    if present.returncode == 0:
        ancestor = _git(root, "merge-base", "--is-ancestor", remote_head, build.head,
                        allowed=(0, 1))
        if ancestor.returncode == 0:
            return AHEAD
    return UPDATE_AVAILABLE


def _newer(latest: str, installed: str) -> bool:
    try:
        from packaging.version import Version
    except ImportError:
        def parts(value):
            values = tuple(int(part) for part in value.split("."))
            while values and values[-1] == 0:
                values = values[:-1]
            return values
        return parts(latest) > parts(installed)
    return Version(latest) > Version(installed)


def check_update(build: BuildInfo, log) -> str:
    """Return a status even when Git or the network is unavailable."""
    try:
        if build.root is not None:
            status = _git_status(build)
        else:
            with urlopen(PYPI_URL, timeout=TIMEOUT) as response:
                latest = json.loads(response.read(MAX_PYPI_BYTES))["info"]["version"]
            status = UPDATE_AVAILABLE if _newer(latest, build.version) else UP_TO_DATE
    except Exception as exc:
        # The type only: git and URL errors can carry a remote URL with a token.
        log(f"Update check: {UNKNOWN} ({type(exc).__name__})")
        return UNKNOWN
    log(f"Update check: {status}")
    return status


class UpdateChecker:
    """One worker; the UI reads an immutable result without waiting."""

    def __init__(self, enabled, log):
        self.result = (BuildInfo(package_version()), UNKNOWN)
        self._enabled = enabled
        self._log = log
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="update-check", daemon=True)
            self._thread.start()

    def _run(self):
        build = get_build_info()
        self._log(f"Version: {build.label}")
        self.result = (build, UNKNOWN)
        while not self._stop.is_set():
            status = check_update(build, self._log) if self._enabled() else UNKNOWN
            if not self._stop.is_set():
                self.result = (build, status)
            if self._stop.wait(CHECK_INTERVAL):
                break

    def stop(self):
        self._stop.set()
