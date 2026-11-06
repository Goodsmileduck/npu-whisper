"""Folders from before the rename to Débora Whisper move to the new names."""
import importlib
import sys
import types
from pathlib import Path

import pytest

from debora_whisper import paths


def _no_saved_variables():
    def open_key(*_):
        raise OSError("no saved variables")
    return types.SimpleNamespace(HKEY_CURRENT_USER=None, OpenKey=open_key)


@pytest.fixture
def reload_paths(monkeypatch, tmp_path):
    """Reload paths with the home folder in tmp_path and migration allowed."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setitem(sys.modules, "winreg", _no_saved_variables())
    monkeypatch.delenv(paths.NO_MIGRATION_ENV, raising=False)
    monkeypatch.delenv("MODELS_DIR", raising=False)

    def reload():
        return importlib.reload(paths)
    yield reload
    monkeypatch.undo()
    importlib.reload(paths)


def test_legacy_config_folder_is_moved(reload_paths, tmp_path):
    legacy = tmp_path / ".npu-dictation"
    (legacy / "logs").mkdir(parents=True)
    (legacy / "config.json").write_text("{}")
    p = reload_paths()
    assert p.CONFIG_DIR == tmp_path / ".debora"
    assert (tmp_path / ".debora" / "config.json").read_text() == "{}"
    assert not legacy.exists()
    assert len(p.MIGRATIONS) == 1 and "Moved" in p.MIGRATIONS[0]


def test_legacy_models_folder_is_moved(reload_paths, tmp_path, monkeypatch):
    shared = tmp_path / "models"
    (shared / "npu-whisper" / "models").mkdir(parents=True)
    monkeypatch.setenv("MODELS_DIR", str(shared))
    p = reload_paths()
    assert p.MODEL_DIR == shared / "debora-whisper" / "models"
    assert p.MODEL_DIR.is_dir() and not (shared / "npu-whisper").exists()


def test_empty_new_folder_does_not_block_the_move(reload_paths, tmp_path):
    (tmp_path / ".debora").mkdir()
    (tmp_path / ".npu-dictation").mkdir()
    (tmp_path / ".npu-dictation" / "config.json").write_text("{}")
    p = reload_paths()
    assert (p.CONFIG_DIR / "config.json").exists()
    assert not (tmp_path / ".npu-dictation").exists()


def test_new_folder_with_only_empty_folders_does_not_block_the_move(
        reload_paths, tmp_path, monkeypatch):
    shared = tmp_path / "models"
    (shared / "debora-whisper" / "ov-cache").mkdir(parents=True)
    (shared / "npu-whisper" / "models").mkdir(parents=True)
    (shared / "npu-whisper" / "models" / "model.bin").write_bytes(b"x")
    monkeypatch.setenv("MODELS_DIR", str(shared))
    p = reload_paths()
    assert (p.MODEL_DIR / "model.bin").exists()
    assert not (shared / "npu-whisper").exists()


def test_new_folder_wins_and_legacy_is_left_alone(reload_paths, tmp_path):
    (tmp_path / ".debora").mkdir()
    (tmp_path / ".debora" / "config.json").write_text("{}")
    (tmp_path / ".npu-dictation").mkdir()
    p = reload_paths()
    assert p.CONFIG_DIR == tmp_path / ".debora"
    assert (tmp_path / ".npu-dictation").exists() and p.MIGRATIONS == []


def test_unmovable_legacy_folder_keeps_being_used(reload_paths, tmp_path, monkeypatch):
    (tmp_path / ".npu-dictation").mkdir()

    def locked(self, target):
        raise PermissionError("in use")
    monkeypatch.setattr(Path, "rename", locked)
    p = reload_paths()
    assert p.CONFIG_DIR == tmp_path / ".npu-dictation"
    assert "Could not move" in p.MIGRATIONS[0]


def test_tests_never_move_folders(reload_paths, tmp_path, monkeypatch):
    (tmp_path / ".npu-dictation").mkdir()
    monkeypatch.setenv(paths.NO_MIGRATION_ENV, "1")
    p = reload_paths()
    assert p.CONFIG_DIR == tmp_path / ".debora"
    assert (tmp_path / ".npu-dictation").exists()
