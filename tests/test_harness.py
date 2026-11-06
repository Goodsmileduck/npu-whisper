"""Claude's file protections and configured diagnostic permissions."""
from pathlib import PurePosixPath, PureWindowsPath
from unittest.mock import MagicMock

import pytest

from debora_whisper import harness
from debora_whisper.dictation_engine import DEFAULT_CONFIG, validate_config


def _rules(command, flag):
    if flag not in command:
        return []
    rules = []
    for arg in command[command.index(flag) + 1:]:
        if arg.startswith("--"):
            break
        rules.append(arg)
    return rules


@pytest.mark.parametrize("response", ["deny", "allow"])
@pytest.mark.parametrize("mode", ["acceptEdits", "bypassPermissions"])
@pytest.mark.parametrize("config_file, pattern", [
    (PureWindowsPath("C:/Users/Test User/.debora/config.json"),
     "//c/Users/Test User/.debora/config.json*"),
    (PurePosixPath("/home/test/.npu-dictation/config.json"),
     "//home/test/.npu-dictation/config.json*"),
    (PureWindowsPath("C:/Users/Alex[work] {team}/.debora/config.json"),
     r"//c/Users/Alex\[work\] \{team\}/.debora/config.json*"),
    (PurePosixPath(r"/home/Alex[work] *?{team}\name/.debora/config.json"),
     r"//home/Alex\[work\] \*\?\{team\}\\name/.debora/config.json*"),
])
def test_config_edit_rules_always_apply(tmp_path, monkeypatch, response, mode,
                                       config_file, pattern):
    monkeypatch.setattr(harness.paths, "CONFIG_FILE", MagicMock(resolve=lambda: config_file))
    config_dir = tmp_path / "config"
    monkeypatch.setattr(harness.paths, "CONFIG_DIR", config_dir)
    config = {"harness_cwd": str(tmp_path / "project"),
              "harness_permission_response": response, "harness_permission_mode": mode,
              "harness_allowed_tools": ["Edit", "Write", "Read"]}
    command = harness.harness_command(config, "test-session", False)
    # Claude applies Edit(path) to every file editor; Write(path) only warns.
    assert _rules(command, "--disallowedTools") == [f"Edit({pattern})"]
    assert _rules(command, "--allowedTools") == ["Edit", "Write", "Read"]
    assert str(config_dir.resolve()) in [
        command[i + 1] for i, arg in enumerate(command) if arg == "--add-dir"]


@pytest.mark.parametrize("base, extra, expected", [
    (None, [], harness.DEFAULT_ALLOWED_TOOLS),
    (None, ["WebSearch", harness.DEFAULT_ALLOWED_TOOLS[0], "WebSearch"],
     (*harness.DEFAULT_ALLOWED_TOOLS, "WebSearch")),
    (["Read", "Read"], ["WebSearch", "Read", "WebSearch"], ("Read", "WebSearch")),
    ([], ["WebSearch"], ("WebSearch",)),
    ([], [], ()),
])
def test_extra_tools_extend_the_effective_list(base, extra, expected):
    config = {"harness_allowed_tools": base, "harness_extra_allowed_tools": extra}
    validate_config({**DEFAULT_CONFIG, **config})
    assert harness.harness_allowed_tools(config) == expected
    assert _rules(harness.harness_command(config, "test-session", False),
                  "--allowedTools") == list(expected)


def test_extra_tools_default_to_empty_and_change_the_session_key():
    assert DEFAULT_CONFIG["harness_extra_allowed_tools"] == []
    assert harness.harness_allowed_tools({}) == harness.DEFAULT_ALLOWED_TOOLS
    config = {"harness_extra_allowed_tools": ["WebSearch"]}
    assert harness.harness_key(config) != harness.harness_key({})
    assert harness.harness_key(config) == harness.harness_key(
        {"harness_extra_allowed_tools": ["WebSearch", "WebSearch"]})


@pytest.mark.parametrize("key", ["harness_allowed_tools", "harness_extra_allowed_tools"])
@pytest.mark.parametrize("value", ["WebSearch", {}, ("Read",), [1], [""], ["  "], False])
def test_invalid_tool_lists_are_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        validate_config({**DEFAULT_CONFIG, key: value})


def test_extra_tools_must_be_a_list_even_when_the_base_is_null():
    with pytest.raises(ValueError, match="harness_extra_allowed_tools"):
        validate_config({**DEFAULT_CONFIG, "harness_extra_allowed_tools": None})
