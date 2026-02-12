"""Tests for Parakeet TDT integration."""

import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
import numpy as np
import pytest

from dictation_engine import (
    MODEL_REGISTRY, DictationApp, DEFAULT_CONFIG,
    create_model, ParakeetNPU, WhisperNPU,
    LANGUAGES, get_models_for_language, is_model_downloaded, MODEL_DIR,
)


class TestParakeetRegistry:
    """Verify parakeet is correctly registered."""

    def test_parakeet_in_registry(self):
        assert "parakeet" in MODEL_REGISTRY

    def test_parakeet_has_required_fields(self):
        info = MODEL_REGISTRY["parakeet"]
        assert info["backend"] == "parakeet"
        assert info["preferred_device"] == "NPU"
        assert "ov_repo" in info
        assert "local_dir" in info
        assert "description" in info

    def test_all_models_have_backend_field(self):
        for name, info in MODEL_REGISTRY.items():
            assert "backend" in info, f"{name} missing backend field"
            assert info["backend"] in ("whisper", "parakeet"), f"{name} has invalid backend"

    def test_all_models_have_local_dir_field(self):
        for name, info in MODEL_REGISTRY.items():
            assert "local_dir" in info, f"{name} missing local_dir field"


class TestCreateModel:
    """Verify factory function dispatches correctly."""

    @patch("dictation_engine.WhisperNPU")
    def test_creates_whisper_for_whisper_backend(self, mock_cls):
        mock_cls.return_value = MagicMock()
        result = create_model(Path("/fake"), device="NPU", backend="whisper")
        mock_cls.assert_called_once_with(Path("/fake"), device="NPU")

    @patch("dictation_engine.ParakeetNPU")
    def test_creates_parakeet_for_parakeet_backend(self, mock_cls):
        mock_cls.return_value = MagicMock()
        result = create_model(Path("/fake"), device="NPU", backend="parakeet")
        mock_cls.assert_called_once_with(Path("/fake"), device="NPU")


class TestEnsureModelDispatch:
    """Verify ensure_model uses factory pattern."""

    @patch("dictation_engine.create_model")
    @patch("dictation_engine.setup_model")
    def test_ensure_model_parakeet(self, mock_setup, mock_create):
        mock_setup.return_value = Path("/fake/parakeet")
        mock_create.return_value = MagicMock()

        config = {**DEFAULT_CONFIG, "model_size": "parakeet", "device": "NPU"}
        app = DictationApp(config)
        app.ensure_model()

        mock_create.assert_called_once_with(
            Path("/fake/parakeet"), device="NPU", backend="parakeet",
        )

    @patch("dictation_engine.create_model")
    @patch("dictation_engine.setup_model")
    def test_ensure_model_whisper(self, mock_setup, mock_create):
        mock_setup.return_value = Path("/fake/whisper")
        mock_create.return_value = MagicMock()

        config = {**DEFAULT_CONFIG, "model_size": "base", "device": "NPU"}
        app = DictationApp(config)
        app.ensure_model()

        mock_create.assert_called_once_with(
            Path("/fake/whisper"), device="NPU", backend="whisper",
        )


class TestParakeetConfig:
    """Verify parakeet works with config validation."""

    def test_parakeet_config_validates(self):
        from dictation_engine import validate_config
        config = {**DEFAULT_CONFIG, "model_size": "parakeet", "device": "NPU"}
        # Should not raise
        validate_config(config)

    def test_invalid_model_still_rejected(self):
        from dictation_engine import validate_config
        config = {**DEFAULT_CONFIG, "model_size": "nonexistent"}
        with pytest.raises(ValueError, match="model_size"):
            validate_config(config)


class TestLanguageFiltering:
    """Verify language-based model filtering."""

    def test_all_models_have_languages_field(self):
        for name, info in MODEL_REGISTRY.items():
            assert "languages" in info, f"{name} missing languages field"

    def test_english_returns_all_models(self):
        models = get_models_for_language("en")
        assert len(models) == len(MODEL_REGISTRY)
        assert "parakeet" in models

    def test_non_english_excludes_parakeet(self):
        for lang in ("ru", "es", "fr", "de", "ja", "zh"):
            models = get_models_for_language(lang)
            assert "parakeet" not in models, f"parakeet should not be in {lang} models"
            assert "base" in models
            assert "small" in models

    def test_whisper_models_support_all_languages(self):
        for name, info in MODEL_REGISTRY.items():
            if info["backend"] == "whisper":
                assert info["languages"] == "all", f"{name} should support all languages"

    def test_parakeet_english_only(self):
        info = MODEL_REGISTRY["parakeet"]
        assert info["languages"] == ["en"]

    def test_languages_dict_has_entries(self):
        assert len(LANGUAGES) >= 10
        assert "en" in LANGUAGES
        assert LANGUAGES["en"] == "English"


class TestModelDownloadStatus:
    """Verify download status detection."""

    def test_nonexistent_model_not_downloaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("dictation_engine.MODEL_DIR", tmp_path)
        assert is_model_downloaded("base") is False

    def test_empty_dir_not_downloaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("dictation_engine.MODEL_DIR", tmp_path)
        (tmp_path / "whisper-base-openvino").mkdir()
        assert is_model_downloaded("base") is False

    def test_dir_with_xml_is_downloaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("dictation_engine.MODEL_DIR", tmp_path)
        model_dir = tmp_path / "whisper-base-openvino"
        model_dir.mkdir()
        (model_dir / "model.xml").write_text("")
        assert is_model_downloaded("base") is True

    def test_dir_with_onnx_is_downloaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("dictation_engine.MODEL_DIR", tmp_path)
        model_dir = tmp_path / "parakeet-tdt-openvino"
        model_dir.mkdir()
        (model_dir / "encoder-model.onnx").write_text("")
        assert is_model_downloaded("parakeet") is True
