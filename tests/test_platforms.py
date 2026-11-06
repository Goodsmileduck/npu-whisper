"""OS resources follow the operating system's provider."""
import ctypes
import os
import shutil
import subprocess
import sys
from unittest.mock import Mock, call

import pytest

from debora_whisper import device_queries as queries, dictation_engine as de, platforms, processes
from debora_whisper.platforms.base import Platform
from debora_whisper.platforms.linux import LinuxPlatform
from debora_whisper.platforms.windows import WindowsPlatform
from debora_whisper.platforms import windows_text


def test_generic_platform_uses_keyboard_backend():
    assert Platform().native_hotkeys(None, None, None, "", None, None) is None


@pytest.mark.parametrize("method", ["type_text", "_type_text_ctypes", "type_draft", "delete_text"])
@pytest.mark.parametrize("fails", [False, True])
def test_windows_typed_output_delegates_without_listener(monkeypatch, method, fails):
    provider = WindowsPlatform()
    inject = Mock(side_effect=RuntimeError("injection failed") if fails else None,
                  return_value="typed")
    monkeypatch.setattr(windows_text, method, inject)
    value = 3 if method == "delete_text" else "text"
    if fails:
        with pytest.raises(RuntimeError, match="injection failed"):
            getattr(provider, method)(value)
    else:
        assert getattr(provider, method)(value) == "typed"
    inject.assert_called_once_with(*((value, False) if method == "type_text" else (value,)))


@pytest.mark.parametrize("returncode", [None, 0, 1])
def test_generic_kill_tree(returncode):
    process = Mock()
    process.poll.return_value = returncode
    Platform().kill_tree(process)
    if returncode is None:
        process.kill.assert_called_once_with()
    else:
        process.kill.assert_not_called()


@pytest.mark.parametrize("returncode", [None, 0, 1])
@pytest.mark.parametrize("taskkill_status", [0, 1])
def test_windows_kill_tree(monkeypatch, returncode, taskkill_status):
    # The provider module also imports on systems without Windows constants.
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    run = Mock(return_value=Mock(returncode=taskkill_status))
    monkeypatch.setattr(subprocess, "run", run)
    process = Mock(pid=1234)
    process.poll.return_value = returncode
    WindowsPlatform().kill_tree(process)
    if returncode is None:
        run.assert_called_once_with(
            ["taskkill", "/T", "/F", "/PID", "1234"],
            capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        run.assert_not_called()
    if returncode is None and taskkill_status:
        process.kill.assert_called_once_with()
    else:
        process.kill.assert_not_called()


def test_no_window_comes_from_provider():
    assert Platform().no_window == 0
    assert processes.NO_WINDOW == platforms.current().no_window


def test_kill_tree_delegates_to_current_provider(monkeypatch):
    provider = Mock()
    current = Mock(return_value=provider)
    monkeypatch.setattr(platforms, "current", current)
    process = Mock()
    processes.kill_tree(process)
    current.assert_called_once_with()
    provider.kill_tree.assert_called_once_with(process)
    process.poll.assert_not_called()
    process.kill.assert_not_called()


def test_audio_fixture_keeps_real_provider():
    provider = platforms.current()
    expected = {"win32": WindowsPlatform, "linux": LinuxPlatform}.get(sys.platform, Platform)
    assert type(provider) is expected
    assert provider.default_output_name() is None
    assert provider.output_is_headphones("Headphones") is None


def test_windows_nvidia_smi_prefers_system32(monkeypatch, tmp_path):
    system = tmp_path / "Windows" / "System32" / "nvidia-smi.exe"
    system.parent.mkdir(parents=True)
    system.touch()
    monkeypatch.setenv("SystemRoot", str(system.parent.parent))
    which = Mock(return_value=str(tmp_path / "bin" / "nvidia-smi.exe"))
    monkeypatch.setattr(shutil, "which", which)
    assert WindowsPlatform().nvidia_smi() == str(system)
    which.assert_not_called()


@pytest.mark.parametrize("provider", [Platform, WindowsPlatform])
@pytest.mark.parametrize("found", [None, "nvidia-smi.exe", "bin/nvidia-smi.exe", "cwd", "absolute"])
def test_nvidia_smi_path(monkeypatch, tmp_path, provider, found):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SystemRoot", str(tmp_path / "no-windows"))
    expected = str(tmp_path / "bin" / "nvidia-smi.exe") if found == "absolute" else None
    if found == "cwd":
        found = str(tmp_path / "nvidia-smi.exe")
    elif found == "absolute":
        found = str(tmp_path / "bin" / "nvidia-smi.exe")
    which = Mock(return_value=found)
    monkeypatch.setattr(shutil, "which", which)
    assert provider().nvidia_smi() == expected
    which.assert_called_once_with("nvidia-smi")


@pytest.mark.parametrize("found", [None, "absolute-path"])
def test_device_queries_delegates(monkeypatch, found):
    provider = Mock()
    provider.nvidia_smi.return_value = found
    monkeypatch.setattr(platforms, "current", Mock(return_value=provider))
    assert queries.nvidia_smi() == found
    provider.nvidia_smi.assert_called_once_with()


@pytest.mark.parametrize("provider", [Platform, WindowsPlatform])
@pytest.mark.parametrize("path_env", [None, "existing-path"])
@pytest.mark.parametrize("has_add_dll_directory", [False, True])
def test_add_library_dir(monkeypatch, tmp_path, provider, path_env, has_add_dll_directory):
    if path_env is None:
        monkeypatch.delenv("PATH", raising=False)
    else:
        monkeypatch.setenv("PATH", path_env)
    add = Mock()
    if has_add_dll_directory:
        monkeypatch.setattr(os, "add_dll_directory", add, raising=False)
    else:
        monkeypatch.delattr(os, "add_dll_directory", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "existing-libraries")
    provider().add_library_dir(tmp_path)
    expected = str(tmp_path) + os.pathsep + (path_env or "") if provider is WindowsPlatform else path_env
    assert os.environ.get("PATH") == expected
    assert os.environ["LD_LIBRARY_PATH"] == "existing-libraries"
    if provider is WindowsPlatform and has_add_dll_directory:
        add.assert_called_once_with(str(tmp_path))
        add.return_value.close.assert_not_called()
    else:
        add.assert_not_called()


@pytest.mark.parametrize("return_name", [False, True])
@pytest.mark.parametrize("result", ["success", "missing", "failure"])
def test_has_nvidia_gpu_absolute_path(monkeypatch, tmp_path, return_name, result):
    executable = str(tmp_path / "bin" / "nvidia-smi.exe")
    monkeypatch.setattr(queries, "nvidia_smi", lambda: None if result == "missing" else executable)
    check = Mock(return_value=b"NVIDIA RTX\n")
    if result == "failure":
        check.side_effect = OSError("driver unavailable")
    monkeypatch.setattr(subprocess, "check_output", check)
    expected = "NVIDIA RTX" if return_name else True
    if result != "success":
        expected = None if return_name else False
    assert de.has_nvidia_gpu(return_name) == expected
    if result == "missing":
        check.assert_not_called()
    else:
        args = [executable]
        if return_name:
            args += ["--query-gpu=name", "--format=csv,noheader"]
        check.assert_called_once_with(args, stderr=subprocess.DEVNULL,
                                      creationflags=processes.NO_WINDOW, timeout=None)


@pytest.mark.parametrize("result", ["success", "missing", "failure"])
def test_accelerator_memory_absolute_path(monkeypatch, tmp_path, result):
    executable = str(tmp_path / "bin" / "nvidia-smi.exe")
    monkeypatch.setattr(queries, "nvidia_smi", lambda: None if result == "missing" else executable)
    check = Mock(return_value=b"5324, 8192\n")
    if result == "failure":
        check.side_effect = OSError("driver unavailable")
    monkeypatch.setattr(subprocess, "check_output", check)
    app = de.DictationApp.__new__(de.DictationApp)
    app.config = {"device": "CUDA"}
    assert app._accelerator_memory() == (" VRAM 5.2/8.0G" if result == "success" else "")
    if result == "missing":
        check.assert_not_called()
    else:
        check.assert_called_once_with(
            [executable, "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
            creationflags=processes.NO_WINDOW)


@pytest.mark.parametrize("failure", [False, True])
def test_cuda_pipeline_adds_package_bins_through_provider(monkeypatch, tmp_path, failure):
    import site
    sub = platforms.current().library_dir_names[0]  # bin on Windows, lib on Linux
    bins = [tmp_path / "nvidia" / package / sub for package in ("cublas", "cudnn")]
    for path in bins:
        path.mkdir(parents=True)
    (tmp_path / "nvidia" / "without-bin").mkdir()
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(tmp_path)])
    add = Mock()
    if failure:
        add.side_effect = OSError("DLL directory unavailable")
    monkeypatch.setattr(platforms.current(), "add_library_dir", add)
    model = Mock()
    monkeypatch.setattr(de, "import_faster_whisper", lambda: model)
    monkeypatch.setattr(de, "has_nvidia_gpu", lambda **kwargs: "NVIDIA RTX")
    log = Mock()
    monkeypatch.setattr(de, "log", log)
    pipeline = de.FasterWhisperCUDA("turbo")
    if failure:
        assert add.call_count == 1
        log.assert_any_call("Warning: Failed to inject NVIDIA DLL paths: DLL directory unavailable")
    else:
        assert {call.args[0] for call in add.call_args_list} == set(bins)
        assert add.call_count == 2
    model.assert_called_once_with("turbo", device="cuda", compute_type="int8_float16")
    assert pipeline.pipeline is model.return_value


@pytest.mark.parametrize("wrapper,method,args,kwargs", [
    ("type_text", "type_text", ("Olá",), {"auto_enter": True}),
    ("type_text", "type_text", ("",), {"auto_enter": False}),
    ("_type_text_ctypes", "_type_text_ctypes", ("é👍",), {}),
    ("type_draft_text", "type_draft", ("draft",), {}),
    ("delete_text", "delete_text", (3,), {}),
    ("get_input_target", "input_target", (), {}),
    ("same_input_target", "same_input_target", ((1, 2), (1, None)), {}),
])
def test_text_delegates_to_current_provider(monkeypatch, wrapper, method, args, kwargs):
    provider = Mock(spec=Platform)
    current = Mock(return_value=provider)
    monkeypatch.setattr(platforms, "current", current)
    assert getattr(de, wrapper)(*args, **kwargs) is getattr(provider, method).return_value
    current.assert_called_once_with()
    getattr(provider, method).assert_called_once_with(*args, **kwargs)


@pytest.fixture
def text_io(monkeypatch):
    keyboard, clipboard = Mock(), Mock()
    state = {"text": "original"}
    clipboard.paste.side_effect = lambda: state["text"]
    clipboard.copy.side_effect = lambda text: state.update(text=text)
    monkeypatch.setitem(sys.modules, "keyboard", keyboard)
    monkeypatch.setitem(sys.modules, "pyperclip", clipboard)
    sleep = Mock()
    monkeypatch.setattr(windows_text.time, "sleep", sleep)
    return keyboard, clipboard, state, sleep


@pytest.mark.parametrize("provider", [Platform, WindowsPlatform])
@pytest.mark.parametrize("method,auto_enter", [("type_text", False), ("type_text", True),
                                               ("type_draft", False)])
def test_provider_paste_transaction(text_io, provider, method, auto_enter):
    keyboard, clipboard, state, sleep = text_io
    events = []

    def wait(seconds):
        assert windows_text._clipboard_lock.locked()
        events.append((seconds, state["text"]))

    sleep.side_effect = wait
    kwargs = {"auto_enter": auto_enter} if method == "type_text" else {}
    getattr(provider(), method)("Olá 👍", **kwargs)
    assert events == [(0.05, "Olá 👍"), (0.1, "Olá 👍"), (0.5, "Olá 👍")]
    assert keyboard.press_and_release.call_args_list == (
        [call("ctrl+v"), call("enter")] if auto_enter else [call("ctrl+v")])
    assert clipboard.copy.call_args_list == [call("Olá 👍"), call("original")]
    assert state["text"] == "original"
    assert de._clipboard_lock is windows_text._clipboard_lock
    assert not de._clipboard_lock.locked()


@pytest.mark.parametrize("failure", ["read", "restore", "paste", "new-copy"])
def test_provider_clipboard_failures(text_io, failure):
    keyboard, clipboard, state, sleep = text_io
    if failure == "read":
        clipboard.paste.side_effect = RuntimeError("unavailable")
    elif failure == "restore":
        clipboard.paste.side_effect = ["original", RuntimeError("unavailable")]
    elif failure == "paste":
        keyboard.press_and_release.side_effect = RuntimeError("paste failed")
    else:
        sleep.side_effect = lambda seconds: state.update(text="new copy") if seconds == 0.5 else None
    if failure == "paste":
        with pytest.raises(RuntimeError, match="paste failed"):
            WindowsPlatform().type_draft("draft")
    else:
        WindowsPlatform().type_draft("draft")
    assert state["text"] == {"read": "draft", "restore": "draft",
                             "paste": "original", "new-copy": "new copy"}[failure]
    assert not windows_text._clipboard_lock.locked()


@pytest.mark.parametrize("provider", [Platform, WindowsPlatform])
def test_provider_empty_text_and_deletion(text_io, provider):
    keyboard, clipboard, state, sleep = text_io
    instance = provider()
    instance.type_text("", auto_enter=True)
    instance.type_draft("")
    instance.delete_text(0)
    instance.delete_text(-1)
    assert not keyboard.mock_calls
    assert not clipboard.mock_calls
    sleep.assert_not_called()


@pytest.mark.parametrize("provider", [Platform, WindowsPlatform])
def test_provider_delete_and_enter(text_io, provider):
    keyboard, _, _, sleep = text_io
    instance = provider()
    instance.delete_text(2)
    assert keyboard.mock_calls == [call.press("shift"), call.send("left"), call.send("left"),
                                   call.release("shift"), call.send("backspace")]
    assert sleep.call_args_list == [call(0.001), call(0.001)]
    instance.press_enter()
    keyboard.press_and_release.assert_called_once_with("enter")


@pytest.fixture
def user32(monkeypatch):
    native = Mock()
    monkeypatch.setattr(ctypes, "windll", Mock(user32=native), raising=False)
    return native


@pytest.mark.parametrize("missing", ["keyboard", "pyperclip"])
def test_provider_unicode_fallback(monkeypatch, text_io, user32, missing):
    monkeypatch.setitem(sys.modules, missing, None)
    log = Mock()
    monkeypatch.setattr(de, "log", log)
    packets = []

    def send_input(count, inputs_ref, size):
        inputs = inputs_ref._obj
        assert count == 2
        assert size == ctypes.sizeof(inputs[0])
        packets.append([(item.type, item._input.ki.wVk, item._input.ki.wScan,
                         item._input.ki.dwFlags) for item in inputs])
        return count

    user32.SendInput.side_effect = send_input
    WindowsPlatform().type_text("é👍", auto_enter=True)
    assert packets == [[(1, 0, unit, 4), (1, 0, unit, 6)]
                       for unit in (0xE9, 0xD83D, 0xDC4D)]
    assert text_io[3].call_args_list == [call(0.002)] * 3
    assert not text_io[0].mock_calls  # Native fallback never added Enter.
    log.assert_called_once_with("pyperclip/keyboard not found, using ctypes fallback")


def test_provider_native_deletion(monkeypatch, text_io, user32):
    monkeypatch.setitem(sys.modules, "keyboard", None)
    WindowsPlatform().delete_text(2)
    assert user32.keybd_event.call_args_list == [
        call(0x10, 0, 0, 0),
        call(0x25, 0, 0, 0), call(0x25, 0, 2, 0),
        call(0x25, 0, 0, 0), call(0x25, 0, 2, 0),
        call(0x10, 0, 2, 0), call(0x08, 0, 0, 0), call(0x08, 0, 2, 0),
    ]
    assert text_io[3].call_args_list == [call(0.001)] * 2


def test_generic_native_text_support_is_unknown(monkeypatch, user32):
    monkeypatch.setitem(sys.modules, "keyboard", None)
    monkeypatch.setitem(sys.modules, "pyperclip", None)
    monkeypatch.setattr(de, "log", Mock())
    provider = Platform()
    assert provider.type_text("text") is None
    assert provider.type_draft("draft") is None
    assert provider._type_text_ctypes("text") is None
    assert provider.delete_text(2) is None
    assert provider.input_target() is None
    assert not provider.same_input_target(None, None)
    assert not user32.mock_calls


@pytest.mark.parametrize("result", ["focus", "no-focus", "no-window", "error"])
def test_provider_input_target(user32, result):
    user32.GetForegroundWindow.return_value = 0 if result == "no-window" else 11
    user32.GetWindowThreadProcessId.return_value = 33

    def gui_info(thread_id, info_ref):
        assert thread_id == 33
        assert info_ref._obj.cbSize == ctypes.sizeof(info_ref._obj)
        info_ref._obj.hwndFocus = 22
        return result != "no-focus"

    user32.GetGUIThreadInfo.side_effect = gui_info
    if result == "error":
        user32.GetForegroundWindow.side_effect = OSError("unavailable")
    expected = {"focus": (11, 22), "no-focus": (11, None),
                "no-window": None, "error": None}[result]
    assert WindowsPlatform().input_target() == expected
    if result in ("no-window", "error"):
        user32.GetGUIThreadInfo.assert_not_called()
    else:
        user32.GetWindowThreadProcessId.assert_called_once_with(11, None)


@pytest.mark.parametrize("a,b,expected", [
    ((1, 2), (1, 2), True), ((1, None), (1, 2), True),
    ((1, 2), (1, None), True), ((1, 2), (1, 3), False),
    ((1, 2), (4, 2), False), (None, (1, 2), False), ((1, 2), None, False),
])
def test_provider_same_input_target(a, b, expected):
    assert WindowsPlatform().same_input_target(a, b) is expected
