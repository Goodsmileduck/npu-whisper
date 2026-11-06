"""Where debora-whisper keeps its files.

Config and logs live in ~/.debora. Models, compiled model caches and
voices go to a shared models folder when MODELS_DIR is set (models under
$MODELS_DIR/debora-whisper, voices under $MODELS_DIR/voices), else also under
~/.debora. Models taken from the Hugging Face cache (the voice chat
LLM, faster-whisper, Chatterbox) follow HF_HOME instead.

Before the rename to Débora Whisper these were ~/.npu-dictation and
$MODELS_DIR/npu-whisper. The first run moves them to the new names; if a
folder cannot be moved (an older copy of the app still has files open), this
run keeps using it and the next one tries again.
"""
import os
import shutil
import sys
from pathlib import Path


def user_env(name: str) -> str | None:
    """The variable from this process, or else (Windows) as saved for the
    user: a terminal opened before it was set does not pass it on, and the
    app would quietly use, and download into, other folders."""
    value = os.environ.get(name)
    if value or sys.platform != "win32":
        return value or None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
    except OSError:
        return None
    return os.path.expandvars(str(value)) or None


# huggingface_hub reads HF_HOME when imported, and the TTS server inherits it.
for _name in ("MODELS_DIR", "HF_HOME"):
    if not os.environ.get(_name) and (_value := user_env(_name)):
        os.environ[_name] = _value

# Set by tests: importing the package must never move the user's folders.
NO_MIGRATION_ENV = "DEBORA_WHISPER_NO_MIGRATION"

# One line per moved (or unmovable) legacy folder, for the app log.
MIGRATIONS: list[str] = []


def _holds_files(folder: Path) -> bool:
    return any(p.is_file() for p in folder.rglob("*"))


def _migrated(new: Path, legacy: Path) -> Path:
    """new, after moving legacy there if new is missing or holds only empty
    folders; legacy itself when it cannot be moved."""
    if os.environ.get(NO_MIGRATION_ENV) or not legacy.is_dir():
        return new
    if new.exists() and _holds_files(new):
        return new
    try:
        new.parent.mkdir(parents=True, exist_ok=True)
        if new.exists():
            shutil.rmtree(new)  # no files: made by something that ran before the move
        legacy.rename(new)
    except OSError as e:
        MIGRATIONS.append(f"Could not move {legacy} to {new} ({e}); using {legacy} for now.")
        return legacy
    MIGRATIONS.append(f"Moved {legacy} to {new} (renamed to Débora Whisper).")
    return new


CONFIG_DIR = _migrated(Path.home() / ".debora", Path.home() / ".npu-dictation")
CONFIG_FILE = CONFIG_DIR / "config.json"
LOG_DIR = CONFIG_DIR / "logs"

_shared = user_env("MODELS_DIR")
DATA_DIR = (_migrated(Path(_shared) / "debora-whisper", Path(_shared) / "npu-whisper")
            if _shared else CONFIG_DIR)
MODEL_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "ov-cache"
VOICES_DIR = Path(_shared) / "voices" if _shared else CONFIG_DIR / "voices"
# Voices shipped inside the package, so every install has them; one of the
# same name in VOICES_DIR wins.
BUNDLED_VOICES_DIR = Path(__file__).parent / "voices"
