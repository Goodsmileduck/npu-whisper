"""The Linux provider's process side: CUDA libraries, the single-instance
lock, headphones, boot time and the Level Zero guard. Nothing real is
loaded: libraries, devices and commands are fakes."""
import ctypes
import sys

import pytest

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux provider")

from debora_whisper.platforms import linux as linux_platform  # noqa: E402
from debora_whisper.platforms.linux import LinuxPlatform  # noqa: E402


def test_cuda_libraries_preload_dependencies_first(monkeypatch, tmp_path):
    for name in ("libcudnn.so.9", "libcudnn_ops.so.9"):
        (tmp_path / name).touch()
    loaded = []

    def cdll(path, mode):
        assert mode == ctypes.RTLD_GLOBAL
        if path.endswith("libcudnn.so.9") and not any(p.endswith("ops.so.9") for p in loaded):
            raise OSError("needs ops")
        loaded.append(path)

    monkeypatch.setattr(linux_platform.ctypes, "CDLL", cdll)
    provider = LinuxPlatform()
    provider.add_library_dir(tmp_path)
    provider.add_library_dir(tmp_path)  # already loaded: nothing again
    assert [p.rsplit("/", 1)[1] for p in loaded] == ["libcudnn_ops.so.9", "libcudnn.so.9"]
    assert provider.library_dir_names == ("lib",)


def test_second_instance_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    first, second = LinuxPlatform(), LinuxPlatform()
    assert first.claim_single_instance(r"Local\debora-test") is True
    assert second.claim_single_instance(r"Local\debora-test") is False
    for handle in first._instance_locks:
        handle.close()
    assert second.claim_single_instance(r"Local\debora-test") is True


def test_instance_lock_without_runtime_dir_stays_out_of_tmp(monkeypatch, tmp_path):
    import debora_whisper.paths as paths
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(paths, "CONFIG_DIR", tmp_path / "config")
    provider = LinuxPlatform()
    assert provider.claim_single_instance(r"Local\debora-test") is True
    assert (tmp_path / "config" / "Local_debora-test.lock").exists()
    for handle in provider._instance_locks:
        handle.close()


def test_unopenable_instance_lock_does_not_block_startup(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    (tmp_path / "Local_debora-test.lock").mkdir()  # another user's squatted name
    assert LinuxPlatform().claim_single_instance(r"Local\debora-test") is True


@pytest.mark.parametrize("port, expected", [
    ({"name": "analog-output-headphones", "type": "Headphones"}, True),
    ({"name": "analog-output-speaker", "type": "Speaker"}, False),
    ({"name": "[Out] Headset", "type": "Unknown"}, True),
    ({"name": "hdmi-output-0", "type": "HDMI"}, None),
])
def test_headphones_from_the_active_port(port, expected):
    sink = {"active_port": port["name"], "ports": [port]}
    assert linux_platform._headphone_port(sink) is expected


def test_boot_time_is_in_the_past():
    import time
    assert 0 < LinuxPlatform().boot_time() < time.time()


# --- Level Zero ----------------------------------------------------------------

@pytest.mark.parametrize("accel, preset, expected", [
    ([], None, ""),                                # no NPU: no driver lookup
    (["/dev/accel/accel0"], None, None),           # an NPU keeps Level Zero
    ([], "/opt/ze/libze_intel_vpu.so", "/opt/ze/libze_intel_vpu.so"),  # the user's choice
])
def test_level_zero_is_kept_from_crashing_without_an_npu(monkeypatch, accel, preset, expected):
    import os
    monkeypatch.setattr(linux_platform.glob, "glob", lambda pattern: accel)
    if preset is None:
        monkeypatch.delenv("ZE_ENABLE_ALT_DRIVERS", raising=False)
    else:
        monkeypatch.setenv("ZE_ENABLE_ALT_DRIVERS", preset)
    linux_platform.avoid_level_zero_crash()
    assert os.environ.get("ZE_ENABLE_ALT_DRIVERS") == expected


@pytest.mark.parametrize("name", ["SIGTSTP", "SIGTTIN", "SIGTTOU"])
def test_a_stop_releases_keyboards_and_sigcont_grabs_them_again(monkeypatch, name):
    import signal
    from unittest.mock import Mock
    from debora_whisper.platforms import linux_keyboard
    stop = getattr(signal, name)
    handlers, sent = {}, []
    monkeypatch.setattr(signal, "signal", lambda sig, handler: handlers.__setitem__(sig, handler))
    suspend, resume = Mock(), Mock()
    monkeypatch.setattr(linux_keyboard, "suspend", suspend)
    monkeypatch.setattr(linux_keyboard, "resume", resume)

    def kill(pid, sig):
        assert handlers[sig] is signal.SIG_DFL  # stops as the default would
        sent.append(sig)
    monkeypatch.setattr(linux_platform.os, "kill", kill)

    LinuxPlatform().handle_termination()
    handlers[stop](stop, None)
    suspend.assert_called_once_with()
    assert sent == [stop]
    # Back from the stop (or the kernel discarded it): rearmed and grabbing.
    resume.assert_called_once_with()
    assert handlers[stop] is linux_platform._stop_without_keyboards

    handlers[signal.SIGCONT](signal.SIGCONT, None)  # idempotent
    assert resume.call_count == 2
    assert handlers[stop] is linux_platform._stop_without_keyboards
