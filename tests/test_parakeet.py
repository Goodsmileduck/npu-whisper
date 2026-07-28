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
    LANGUAGES, PARAKEET_UPSTREAM_LANGUAGES,
    get_models_for_language, is_model_downloaded, MODEL_DIR,
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

    def test_non_european_excludes_parakeet(self):
        # Parakeet's upstream checkpoint covers 25 European languages; the
        # app also exposes ja/zh/ko/tr/ar which the checkpoint does not
        # support, so those must stay excluded.
        for lang in ("ja", "zh", "ko", "tr", "ar"):
            models = get_models_for_language(lang)
            assert "parakeet" not in models, f"parakeet should not be in {lang} models"
            assert "base" in models
            assert "small" in models

    def test_non_english_includes_parakeet_for_supported_languages(self):
        # Regression test for issue #2: Parakeet must be offered for at
        # least one non-English language it actually supports.
        for lang in ("ru", "es", "fr", "de", "pt", "it", "nl", "pl", "uk"):
            models = get_models_for_language(lang)
            assert "parakeet" in models, f"parakeet should be offered for {lang}"

    def test_whisper_models_support_all_languages(self):
        for name, info in MODEL_REGISTRY.items():
            if info["backend"] == "whisper":
                assert info["languages"] == "all", f"{name} should support all languages"

    def test_parakeet_is_multilingual(self):
        info = MODEL_REGISTRY["parakeet"]
        assert isinstance(info["languages"], list)
        assert "en" in info["languages"]
        assert len(info["languages"]) > 1, "parakeet should no longer be English-only"

    def test_parakeet_languages_subset_of_upstream_checkpoint(self):
        info = MODEL_REGISTRY["parakeet"]
        for lang in info["languages"]:
            assert lang in PARAKEET_UPSTREAM_LANGUAGES, (
                f"{lang} is declared for parakeet but the upstream checkpoint "
                f"does not support it"
            )

    def test_parakeet_languages_selectable_in_ui(self):
        # Regression test for issue #2: the registry must never declare a
        # language the UI's LANGUAGES map (and therefore the model picker)
        # cannot select.
        info = MODEL_REGISTRY["parakeet"]
        for lang in info["languages"]:
            assert lang in LANGUAGES, (
                f"{lang} is declared for parakeet but is not in LANGUAGES "
                f"(the UI cannot select it)"
            )

    def test_languages_dict_has_entries(self):
        assert len(LANGUAGES) >= 10
        assert "en" in LANGUAGES
        assert LANGUAGES["en"] == "English"


class TestParakeetTdtConstants:
    """Verify BLANK_IDX / VOCAB_SIZE are derived from the loaded vocab
    rather than trusted blindly (issue #2)."""

    def _bare_parakeet(self, tmp_path):
        # Bypass __init__ (which loads onnxruntime/OpenVINO models) to unit
        # test _load_vocab() in isolation.
        instance = ParakeetNPU.__new__(ParakeetNPU)
        instance.model_path = tmp_path
        instance.vocab = {}
        return instance

    def test_derives_constants_matching_current_checkpoint_vocab(self, tmp_path):
        # 8192 tokens (indices 0-8191) matches the shipped
        # goodsmileduck/parakeet-tdt-0.6b-v3-onnx vocab.txt.
        lines = [f"tok{i} {i}" for i in range(8192)]
        (tmp_path / "vocab.txt").write_text("\n".join(lines), encoding="utf-8")
        instance = self._bare_parakeet(tmp_path)
        instance._load_vocab()
        assert instance.BLANK_IDX == 8192
        assert instance.VOCAB_SIZE == 8193

    def test_derives_constants_for_a_different_sized_vocab(self, tmp_path):
        # A hypothetical re-export with a smaller vocab must not silently
        # keep using the old checkpoint's hardcoded constants.
        lines = [f"tok{i} {i}" for i in range(100)]
        (tmp_path / "vocab.txt").write_text("\n".join(lines), encoding="utf-8")
        instance = self._bare_parakeet(tmp_path)
        instance._load_vocab()
        assert instance.BLANK_IDX == 100
        assert instance.VOCAB_SIZE == 101


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
