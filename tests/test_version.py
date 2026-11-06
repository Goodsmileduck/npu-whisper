"""Build and update checks without Git, network access or a display."""

import io
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from debora_whisper import version as v


HEAD = "a" * 40
REMOTE = "b" * 40


@pytest.fixture
def git(monkeypatch, tmp_path):
    replies = {
        ("rev-parse", "--show-toplevel"): (0, str(tmp_path)),
        ("ls-files", "--error-unmatch", "__init__.py"): (0, "__init__.py\n"),
        ("log", "-1", "--format=%h %cI"): (0, "aaaaaaa 2026-10-26T12:00:00-03:00\n"),
        ("rev-parse", "HEAD"): (0, HEAD),
        ("status", "--porcelain", "--untracked-files=normal"): (0, ""),
        ("symbolic-ref", "--quiet", "--short", "HEAD"): (0, "feature\n"),
        ("config", "--get", "branch.feature.remote"): (1, ""),
        ("ls-remote", "--exit-code", "origin", "refs/heads/main"): (0, f"{HEAD}\trefs/heads/main\n"),
        ("rev-parse", "--verify", "--quiet", f"{REMOTE}^{{commit}}"): (0, REMOTE),
        ("merge-base", "--is-ancestor", REMOTE, HEAD): (1, ""),
    }

    def run(args, **kwargs):
        assert args[0] == "git"
        reply = replies[tuple(args[1:])]
        if isinstance(reply, Exception):
            raise reply
        return subprocess.CompletedProcess(args, reply[0], reply[1], "")

    mock = MagicMock(side_effect=run)
    monkeypatch.setattr(v.subprocess, "run", mock)
    monkeypatch.setattr(v.platforms, "current", lambda: SimpleNamespace(no_window=123))
    monkeypatch.setattr(v.metadata, "version", lambda name: "0.1.0")
    return replies, mock


@pytest.fixture(autouse=True)
def http(monkeypatch):
    mock = MagicMock(side_effect=OSError("offline"))
    monkeypatch.setattr(v, "urlopen", mock)
    return mock


@pytest.mark.parametrize("dirty", ["", " M debora_whisper/app.py\n", "?? local.txt\n"])
def test_checkout_identity_and_hidden_bounded_subprocess(git, dirty):
    replies, run = git
    replies[("status", "--porcelain", "--untracked-files=normal")] = (0, dirty)
    build = v.get_build_info()
    suffix = "+dirty" if dirty else ""
    assert build.label == f"Débora Whisper 0.1.0 · aaaaaaa{suffix} (2026-10-26)"
    assert build.head == HEAD
    assert run.call_args_list[0].kwargs["cwd"] == Path(v.__file__).resolve().parent
    for call in run.call_args_list:
        assert call.kwargs["timeout"] == 5
        assert call.kwargs["creationflags"] == 123
        assert call.kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
        assert call.kwargs["env"]["GIT_OPTIONAL_LOCKS"] == "0"


@pytest.mark.parametrize("failure", [
    FileNotFoundError("git"), subprocess.TimeoutExpired("git", 5), (128, ""),
])
def test_missing_git_or_checkout_falls_back_to_metadata(git, failure):
    replies, _ = git
    replies[("rev-parse", "--show-toplevel")] = failure
    assert v.get_build_info() == v.BuildInfo("0.1.0")


def test_wheel_inside_unrelated_checkout_is_not_a_git_build(git):
    replies, _ = git
    replies[("ls-files", "--error-unmatch", "__init__.py")] = (1, "")
    assert v.get_build_info().root is None


def test_metadata_missing_uses_packaged_version(monkeypatch):
    metadata = MagicMock(side_effect=v.metadata.PackageNotFoundError("debora-whisper"))
    monkeypatch.setattr(v.metadata, "version", metadata)
    assert v.package_version() == v.__version__
    metadata.assert_called_once_with("debora-whisper")


def test_regular_install_uses_installed_distribution_version(git, monkeypatch):
    replies, _ = git
    replies[("rev-parse", "--show-toplevel")] = (128, "")
    metadata = MagicMock(return_value="2.4.1")
    monkeypatch.setattr(v.metadata, "version", metadata)
    assert v.get_build_info().label == "Débora Whisper 2.4.1"
    metadata.assert_called_once_with("debora-whisper")


@pytest.mark.parametrize("remote,dirty,present,ancestor,expected", [
    (HEAD, False, 0, 1, v.UP_TO_DATE),
    (HEAD, True, 0, 1, v.AHEAD),
    (REMOTE, False, 0, 0, v.AHEAD),
    (REMOTE, True, 0, 0, v.AHEAD),
    (REMOTE, False, 0, 1, v.UPDATE_AVAILABLE),
    (REMOTE, True, 0, 1, v.UPDATE_AVAILABLE),
    (REMOTE, False, 1, 1, v.UPDATE_AVAILABLE),
])
def test_git_update_status(git, http, tmp_path, remote, dirty, present, ancestor, expected):
    replies, run = git
    replies[("ls-remote", "--exit-code", "origin", "refs/heads/main")] = (
        0, f"{remote}\trefs/heads/main\n")
    replies[("rev-parse", "--verify", "--quiet", f"{REMOTE}^{{commit}}")] = (present, "")
    replies[("merge-base", "--is-ancestor", REMOTE, HEAD)] = (ancestor, "")
    build = v.BuildInfo("0.1.0", tmp_path, HEAD, dirty=dirty)
    log = MagicMock()
    assert v.check_update(build, log) == expected
    log.assert_called_once_with(f"Update check: {expected}")
    commands = [call.args[0][1] for call in run.call_args_list]
    assert not {"fetch", "pull", "checkout", "reset"}.intersection(commands)
    assert ("merge-base" in commands) == (remote != HEAD and present == 0)
    http.assert_not_called()


@pytest.mark.parametrize("configured", ["upstream", ".", "", "--upload-pack=x"])
def test_upstream_remote_or_origin(git, tmp_path, configured):
    replies, run = git
    replies[("config", "--get", "branch.feature.remote")] = (0, configured)
    remote = "upstream" if configured == "upstream" else "origin"
    replies[("ls-remote", "--exit-code", remote, "refs/heads/main")] = (
        0, f"{HEAD}\trefs/heads/main\n")
    assert v.check_update(v.BuildInfo("0.1.0", tmp_path, HEAD), MagicMock()) == v.UP_TO_DATE
    assert any(call.args[0] == ["git", "ls-remote", "--exit-code", remote, "refs/heads/main"]
               for call in run.call_args_list)


def test_detached_head_uses_origin(git, tmp_path):
    replies, _ = git
    replies[("symbolic-ref", "--quiet", "--short", "HEAD")] = (1, "")
    assert v.check_update(v.BuildInfo("0.1.0", tmp_path, HEAD), MagicMock()) == v.UP_TO_DATE


@pytest.mark.parametrize("failure", [
    FileNotFoundError("git"), subprocess.TimeoutExpired("git", 5),
    (128, ""), (2, ""), (0, "invalid\trefs/heads/main"),
])
def test_remote_errors_are_unknown_with_one_log_line(git, tmp_path, failure):
    replies, _ = git
    replies[("ls-remote", "--exit-code", "origin", "refs/heads/main")] = failure
    log = MagicMock()
    assert v.check_update(v.BuildInfo("0.1.0", tmp_path, HEAD), log) == v.UNKNOWN
    assert log.call_count == 1
    assert log.call_args.args[0].startswith("Update check: unknown (")


def test_remote_error_log_omits_a_token_in_the_message(git, tmp_path):
    replies, _ = git
    replies[("ls-remote", "--exit-code", "origin", "refs/heads/main")] = (
        subprocess.TimeoutExpired(["git", "ls-remote", "https://user:tok3n@host/r"], 5))
    log = MagicMock()
    v.check_update(v.BuildInfo("0.1.0", tmp_path, HEAD), log)
    assert "tok3n" not in log.call_args.args[0]


def test_git_runs_without_stdin_or_ssh_prompts(git, tmp_path):
    _, run = git
    v.check_update(v.BuildInfo("0.1.0", tmp_path, HEAD), MagicMock())
    kwargs = run.call_args.kwargs
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["env"]["SSH_ASKPASS_REQUIRE"] == "never"


def test_ancestry_error_is_not_an_available_update(git, tmp_path):
    replies, _ = git
    replies[("ls-remote", "--exit-code", "origin", "refs/heads/main")] = (
        0, f"{REMOTE}\trefs/heads/main\n")
    replies[("merge-base", "--is-ancestor", REMOTE, HEAD)] = (128, "")
    assert v.check_update(v.BuildInfo("0.1.0", tmp_path, HEAD), MagicMock()) == v.UNKNOWN


@pytest.mark.parametrize("installed,latest,expected", [
    ("0.1.0", "0.1.0", v.UP_TO_DATE),
    ("0.9.0", "0.10.0", v.UPDATE_AVAILABLE),
    ("0.2.0", "0.1.0", v.UP_TO_DATE),
    ("0.1", "0.1.0", v.UP_TO_DATE),
])
@pytest.mark.parametrize("without_packaging", [False, True])
def test_pypi_version_comparison(git, http, monkeypatch, installed, latest, expected,
                                 without_packaging):
    if without_packaging:
        monkeypatch.setitem(sys.modules, "packaging.version", None)
    http.side_effect = None
    http.return_value.__enter__.return_value = io.StringIO(json.dumps({"info": {"version": latest}}))
    log = MagicMock()
    assert v.check_update(v.BuildInfo(installed), log) == expected
    log.assert_called_once_with(f"Update check: {expected}")
    http.assert_called_once_with(v.PYPI_URL, timeout=5)
    git[1].assert_not_called()


def test_packaging_understands_prereleases():
    pytest.importorskip("packaging.version")
    assert v._newer("1.0", "1.0rc1")
    assert not v._newer("1.0rc1", "1.0")


@pytest.mark.parametrize("body", [None, "not json", "{}", '{"info":{"version":"invalid"}}'])
def test_pypi_failures_are_unknown(http, body):
    if body is not None:
        http.side_effect = None
        http.return_value.__enter__.return_value = io.StringIO(body)
    log = MagicMock()
    assert v.check_update(v.BuildInfo("0.1.0"), log) == v.UNKNOWN
    assert log.call_count == 1


def test_disabled_checker_only_reads_local_identity(git, http):
    log = MagicMock()
    checker = v.UpdateChecker(lambda: False, log)
    checker._stop = MagicMock()
    checker._stop.is_set.return_value = False
    checker._stop.wait.return_value = True
    checker._run()
    assert checker.result[0].head == HEAD
    assert checker.result[1] == v.UNKNOWN
    assert log.call_count == 1
    assert log.call_args.args[0].startswith("Version: Débora Whisper 0.1.0 · aaaaaaa")
    assert all(call.args[0][1] != "ls-remote" for call in git[1].call_args_list)
    http.assert_not_called()


def test_failures_wait_six_hours_between_checks(git):
    replies, run = git
    replies[("ls-remote", "--exit-code", "origin", "refs/heads/main")] = OSError("offline")
    log = MagicMock()
    checker = v.UpdateChecker(lambda: True, log)
    checker._stop = MagicMock()
    checker._stop.is_set.return_value = False
    checker._stop.wait.side_effect = [False, True]
    checker._run()
    assert [call.args for call in checker._stop.wait.call_args_list] == [(21600,), (21600,)]
    assert sum(call.args[0][1] == "ls-remote" for call in run.call_args_list) == 2
    assert log.call_count == 3  # identity, then one line per failed check


def test_start_does_not_wait_for_identity_and_stop_cancels_check(git, http, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def identity():
        entered.set()
        assert release.wait(2)
        return v.BuildInfo("0.1.0")

    monkeypatch.setattr(v, "get_build_info", identity)
    checker = v.UpdateChecker(lambda: True, MagicMock())
    checker.start()
    worker = checker._thread
    try:
        assert entered.wait(2)
        checker.start()
        assert checker._thread is worker
        assert worker.daemon
        assert worker is not threading.current_thread()
        checker.stop()
    finally:
        checker.stop()
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    http.assert_not_called()


@pytest.mark.parametrize("status,text", [
    (v.UP_TO_DATE, "✓ atualizada"),
    (v.UPDATE_AVAILABLE, "⬆ atualização disponível"),
    (v.AHEAD, "✓ adiantada / alterações locais"),
    (v.UNKNOWN, "? não verificada"),
])
def test_shared_display_text(status, text):
    build = v.BuildInfo("0.1.0", commit="a6d8864", date="2026-10-26")
    assert v.display_version(build, status) == f"Débora Whisper 0.1.0 · a6d8864 (2026-10-26) · {text}"
