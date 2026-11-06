"""Tests for config validation."""
import json
from unittest.mock import MagicMock

import pytest

from debora_whisper.dictation_engine import validate_config, DEFAULT_CONFIG


class TestConfigValidation:
    @pytest.mark.parametrize("value", [None, "false", 0, 1])
    def test_invalid_update_check_flag(self, value):
        with pytest.raises(ValueError, match="update_check"):
            validate_config({**DEFAULT_CONFIG, "update_check": value})

    def test_update_check_defaults_on_and_accepts_opt_out(self):
        assert DEFAULT_CONFIG["update_check"] is True
        validate_config({**DEFAULT_CONFIG, "update_check": False})
        validate_config({key: value for key, value in DEFAULT_CONFIG.items() if key != "update_check"})

    @pytest.mark.parametrize("key", ["inline_drafts", "inline_drafts_require_verification"])
    @pytest.mark.parametrize("value", [None, "true", 0, 1])
    def test_invalid_inline_drafts_flag(self, key, value):
        with pytest.raises(ValueError, match=key):
            validate_config({**DEFAULT_CONFIG, key: value})

    def test_inline_verification_defaults_to_backspace(self):
        assert DEFAULT_CONFIG["inline_drafts_require_verification"] is False
        config = {key: value for key, value in DEFAULT_CONFIG.items() if not key.startswith("inline_drafts")}
        validate_config(config)
        validate_config({**DEFAULT_CONFIG, "inline_drafts_require_verification": True})

    @pytest.mark.parametrize("name", ["", "   ", None, 42, "a" * 81,
                                     *[f"Debora{c}test" for c in '&|<>^%"\r\n\t!'],
                                     "Debora\n", "\nDebora"])
    def test_invalid_harness_session_name(self, name):
        with pytest.raises(ValueError, match="harness_session_name"):
            validate_config({**DEFAULT_CONFIG, "harness_session_name": name})

    @pytest.mark.parametrize("name", ["Débora Whisper", "Projeto 123 ._-·:()", "a" * 80])
    def test_valid_harness_session_name(self, name):
        validate_config({**DEFAULT_CONFIG, "harness_session_name": name})

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

    @pytest.mark.parametrize("key", ["vad_end_silence_seconds", "vad_incomplete_silence_seconds"])
    @pytest.mark.parametrize("value", [None, True, "3", 0, -1, float("nan"), float("inf")])
    def test_invalid_vad_silence_raises(self, key, value):
        with pytest.raises(ValueError, match=key):
            validate_config({**DEFAULT_CONFIG, key: value})

    def test_existing_config_without_vad_keys_uses_defaults(self):
        config = {key: value for key, value in DEFAULT_CONFIG.items() if not key.startswith("vad_")}
        validate_config(config)


def test_continuous_startup_begins_in_dictation():
    from debora_whisper.dictation_engine import start_in_dictation
    config = {"continuous_listening": True, "voice_chat": True}
    assert start_in_dictation(config) is True  # still warm voice chat up
    assert config["voice_chat"] is False


def test_warm_voice_chat_loads_it_while_dictating():
    from unittest.mock import MagicMock, patch
    from debora_whisper.dictation_engine import DEFAULT_CONFIG, DictationApp
    app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False, "voice_chat": False})
    app.warm_voice_chat = True
    with patch.dict("sys.modules", {"keyboard": MagicMock()}), \
            patch.object(app, "_start_hotkeys"), patch.object(app, "_start_loader"), \
            patch.object(app, "_start_segment_consumer"), \
            patch.object(app, "_warm_up_voice_chat") as warm:
        app.start_background()
    warm.assert_called_once()


def test_startup_keeps_voice_chat_without_continuous_or_when_requested():
    from debora_whisper.dictation_engine import start_in_dictation
    config = {"continuous_listening": False, "voice_chat": True}
    assert start_in_dictation(config) is False
    assert config["voice_chat"] is True
    config = {"continuous_listening": True, "voice_chat": True}
    start_in_dictation(config, voice_chat_requested=True)
    assert config["voice_chat"] is True


@pytest.fixture
def saved_config(tmp_path, monkeypatch):
    from debora_whisper import dictation_engine
    config_file = tmp_path / "config.json"
    monkeypatch.setattr(dictation_engine, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(dictation_engine, "CONFIG_FILE", config_file)
    return config_file


@pytest.mark.parametrize("saved", [None, {}])
@pytest.mark.parametrize("locale_name,expected", [
    ("pt-BR", "pt"), ("pt_PT", "pt"), ("es-MX", "es"), ("zh-CN", "zh"),
    ("ja_JP", "ja"), ("fi_FI", "en"),
])
def test_missing_language_uses_system_ui_language(saved_config, monkeypatch, saved,
                                                locale_name, expected):
    import ctypes
    import locale
    from types import SimpleNamespace
    from debora_whisper import dictation_engine as de, platforms

    detect = MagicMock(return_value=0x0416)
    monkeypatch.setattr(platforms, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(
        kernel32=SimpleNamespace(GetUserDefaultUILanguage=detect)), raising=False)
    monkeypatch.setitem(locale.windows_locale, 0x0416, locale_name)
    if saved is not None:
        saved_config.write_text(json.dumps(saved))

    config = de.load_config()

    assert config["language"] == expected
    assert config["model_size"] in de.get_models_for_language(expected)
    detect.assert_called_once_with()
    assert DEFAULT_CONFIG["language"] == "en"
    if saved is None:
        assert not saved_config.exists()
    else:
        assert json.loads(saved_config.read_text()) == saved
    de.save_config(config)
    assert json.loads(saved_config.read_text())["language"] == expected


def test_regional_ui_language_missing_from_locale_table(saved_config, monkeypatch):
    import ctypes
    import locale
    from types import SimpleNamespace
    from debora_whisper import dictation_engine as de, platforms

    monkeypatch.setattr(platforms, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=SimpleNamespace(
        GetUserDefaultUILanguage=lambda: 0x580A)), raising=False)  # es-419
    monkeypatch.delitem(locale.windows_locale, 0x580A, raising=False)
    monkeypatch.setitem(locale.windows_locale, 0x040A, "es_ES")

    assert de.load_config()["language"] == "es"


def test_detected_language_unsupported_by_saved_model_uses_english(saved_config,
                                                                  monkeypatch):
    from debora_whisper import dictation_engine as de, platforms

    monkeypatch.setattr(platforms, "default_language", lambda languages: "ja")
    saved_config.write_text(json.dumps({"model_size": "parakeet"}))

    assert "parakeet" not in de.get_models_for_language("ja")
    assert de.load_config()["language"] == "en"


@pytest.mark.parametrize("language", ["en", "auto"])
def test_explicit_language_skips_detection(saved_config, monkeypatch, language):
    from debora_whisper import dictation_engine as de, platforms

    detect = MagicMock(side_effect=AssertionError("explicit language must win"))
    monkeypatch.setattr(platforms, "default_language", detect)
    saved_config.write_text(json.dumps({"language": language}))

    assert de.load_config()["language"] == language
    detect.assert_not_called()


def test_missing_language_detection_failure_uses_english(saved_config, monkeypatch):
    import ctypes
    from types import SimpleNamespace
    from debora_whisper import dictation_engine as de, platforms

    detect = MagicMock(side_effect=OSError("UI language unavailable"))
    monkeypatch.setattr(platforms, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(
        kernel32=SimpleNamespace(GetUserDefaultUILanguage=detect)), raising=False)

    assert de.load_config()["language"] == "en"
    detect.assert_called_once_with()


@pytest.mark.parametrize("locale_name,lc_all,lang,expected", [
    ("pt_BR", "de_DE", "en_US", "pt"),
    (None, "es_MX.UTF-8", "en_US", "es"),
    (None, "", "zh_CN.UTF-8", "zh"),
    (None, "", "", "en"),
])
def test_missing_language_uses_non_windows_locale(saved_config, monkeypatch,
                                                 locale_name, lc_all, lang, expected):
    import locale
    from types import SimpleNamespace
    from debora_whisper import dictation_engine as de, platforms

    monkeypatch.setattr(platforms, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(locale, "getlocale", lambda: (locale_name, "UTF-8"))
    monkeypatch.setenv("LC_ALL", lc_all)
    monkeypatch.setenv("LANG", lang)

    assert de.load_config()["language"] == expected


def test_startup_override_survives_unrelated_save(saved_config):
    from debora_whisper.dictation_engine import save_config, start_in_dictation
    config = {**DEFAULT_CONFIG, "continuous_listening": True, "voice_chat": True}
    assert start_in_dictation(config) is True
    validate_config(config)
    config["balloon_width"] = 400
    save_config(config)
    saved = json.loads(saved_config.read_text())
    assert saved["voice_chat"] is True
    assert saved["balloon_width"] == 400
    assert "_saved_voice_chat" not in saved
    assert config["voice_chat"] is False
    assert config["_saved_voice_chat"] is True


def _config_gui(config):
    from debora_whisper.app import GUIApp
    gui = GUIApp.__new__(GUIApp)
    gui._config = config
    gui._engine = MagicMock()
    gui._engine.stop_if_idle.return_value = None
    gui._overlay, gui._tray = MagicMock(), MagicMock()
    gui._settings_win = MagicMock(is_open=True)
    gui._new_engine = MagicMock()
    return gui


@pytest.mark.parametrize("path", ["cli", "gui"])
def test_explicit_voice_chat_off_is_saved(saved_config, monkeypatch, path):
    from debora_whisper.dictation_engine import DictationApp, save_config, start_in_dictation
    config = {**DEFAULT_CONFIG, "continuous_listening": True, "voice_chat": True,
              "beep_on_start": False}
    start_in_dictation(config)
    if path == "cli":
        app = DictationApp(config)
        monkeypatch.setattr(app, "_interrupt_voice_reply", MagicMock())
        monkeypatch.setattr(app, "voice_chat", MagicMock())
        app.set_voice_chat(False)
        save_config(config)
    else:
        _config_gui(config)._set_voice_chat(False)
    saved = json.loads(saved_config.read_text())
    assert saved["voice_chat"] is False
    assert "_saved_voice_chat" not in saved
    assert "_saved_voice_chat" not in config


@pytest.mark.parametrize("rebuild", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_settings_preserve_preference_until_mode_changes(saved_config, rebuild, enabled):
    from debora_whisper.dictation_engine import start_in_dictation
    config = {**DEFAULT_CONFIG, "continuous_listening": True, "voice_chat": True}
    start_in_dictation(config)
    gui = _config_gui(config)
    new_config = {**config, "voice_chat": enabled, "balloon_width": 400}
    if rebuild:
        new_config["hotkey"] = "ctrl+alt+v"
    gui._on_settings_apply(new_config)
    saved = json.loads(saved_config.read_text())
    assert saved["voice_chat"] is True
    assert "_saved_voice_chat" not in saved
    assert config["voice_chat"] is enabled
    assert ("_saved_voice_chat" in config) is (not enabled)
    if enabled:
        gui._on_settings_apply({**new_config, "voice_chat": False})
        assert json.loads(saved_config.read_text())["voice_chat"] is False
        # An open settings window still has its copy of the startup marker.
        gui._on_settings_apply({**new_config, "voice_chat": False})
        saved = json.loads(saved_config.read_text())
        assert saved["voice_chat"] is False
        assert "_saved_voice_chat" not in saved
        assert "_saved_voice_chat" not in config
