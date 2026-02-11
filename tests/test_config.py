"""Tests for config validation."""
import pytest

from dictation_engine import validate_config, DEFAULT_CONFIG


class TestConfigValidation:
    def test_valid_config_passes(self):
        config = DEFAULT_CONFIG.copy()
        validate_config(config)  # should not raise

    def test_invalid_device_raises(self):
        config = {**DEFAULT_CONFIG, "device": "TPU"}
        with pytest.raises(ValueError, match="device"):
            validate_config(config)

    def test_invalid_model_size_raises(self):
        config = {**DEFAULT_CONFIG, "model_size": "huge"}
        with pytest.raises(ValueError, match="model_size"):
            validate_config(config)

    def test_invalid_sample_rate_raises(self):
        config = {**DEFAULT_CONFIG, "sample_rate": "banana"}
        with pytest.raises(ValueError, match="sample_rate"):
            validate_config(config)

    def test_invalid_max_record_seconds_raises(self):
        config = {**DEFAULT_CONFIG, "max_record_seconds": -5}
        with pytest.raises(ValueError, match="max_record_seconds"):
            validate_config(config)
