"""The Settings microphone choice is saved, validated and resolved to a device."""
import sys
from types import SimpleNamespace

import pytest

from debora_whisper.dictation_engine import (
    DEFAULT_CONFIG, resolve_input_device, validate_config,
)


def _fake_sd(devices, default_api=0):
    return SimpleNamespace(
        query_devices=lambda: devices,
        default=SimpleNamespace(hostapi=default_api),
    )


DEVICES = [
    {"name": "Headset", "max_input_channels": 1, "hostapi": 1},
    {"name": "Headset", "max_input_channels": 1, "hostapi": 0},
    {"name": "Speakers", "max_input_channels": 0, "hostapi": 0},
]


class TestInputDeviceConfig:
    def test_defaults_to_system_default(self):
        assert DEFAULT_CONFIG["input_device"] is None
        validate_config(DEFAULT_CONFIG)

    @pytest.mark.parametrize("value", ["", "  ", 3, ["Headset"]])
    def test_rejects_non_names(self, value):
        with pytest.raises(ValueError, match="input_device"):
            validate_config({**DEFAULT_CONFIG, "input_device": value})


class TestResolveInputDevice:
    def test_none_means_default(self):
        assert resolve_input_device(None) is None

    def test_prefers_the_default_host_api(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd(DEVICES))
        assert resolve_input_device("Headset") == 1

    def test_ignores_output_only_devices(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd(DEVICES))
        assert resolve_input_device("Speakers") is None

    def test_unplugged_falls_back_to_default(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd(DEVICES))
        assert resolve_input_device("USB Mic") is None


def test_settings_saves_the_microphone():
    from unittest.mock import MagicMock
    from debora_whisper.ui.settings import SettingsWindow, _SYSTEM_DEFAULT_MIC
    import debora_whisper.dictation_engine as de

    def saved(choice):
        w = SettingsWindow.__new__(SettingsWindow)
        w._config = dict(DEFAULT_CONFIG)
        for attr, value in {
            "_model_radio_var": "turbo", "_hotkey_var": "ctrl+space", "_beep_var": True,
            "_enter_var": False, "_inline_drafts_var": False, "_voice_chat_var": False,
            "_backend_var": "Local (Qwen)", "_harness_new_session_var": False,
            "_harness_session_name_var": "D", "_voice_var": "Chatterbox default",
            "_balloon_var": True, "_font_size_var": "16",
            "_tts_engine_var": "Chatterbox (clones the voice)", "_mic_var": choice,
        }.items():
            setattr(w, attr, MagicMock(get=MagicMock(return_value=value)))
        w._harness_cwd = None
        w._get_selected_lang_code = lambda: "en"
        de.detect_devices = lambda: ["CPU"]
        de.select_device = lambda *a: "CPU"
        return w._get_new_config()["input_device"]

    orig = (de.detect_devices, de.select_device)
    try:
        assert saved(_SYSTEM_DEFAULT_MIC) is None
        assert saved("Headset") == "Headset"
    finally:
        de.detect_devices, de.select_device = orig
