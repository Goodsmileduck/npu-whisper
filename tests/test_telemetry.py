"""The telemetry line's accelerator memory: CUDA VRAM or NPU allocation."""
import sys
import types

from debora_whisper.dictation_engine import DictationApp


def _app(device):
    app = DictationApp.__new__(DictationApp)
    app.config = {"device": device}
    return app


def test_npu_memory_comes_from_openvino(monkeypatch):
    class Core:
        def get_property(self, device, name):
            assert (device, name) == ("NPU", "NPU_DEVICE_ALLOC_MEM_SIZE")
            return 3 * 1024 ** 3 // 2

    monkeypatch.setitem(sys.modules, "openvino", types.SimpleNamespace(Core=Core))
    assert _app("NPU")._accelerator_memory() == " NPU 1.5G"


def test_npu_memory_failure_is_silent(monkeypatch):
    class Core:
        def get_property(self, device, name):
            raise RuntimeError("driver")

    monkeypatch.setitem(sys.modules, "openvino", types.SimpleNamespace(Core=Core))
    assert _app("NPU")._accelerator_memory() == ""


def test_cuda_memory_comes_from_nvidia_smi(monkeypatch, tmp_path):
    import subprocess
    from debora_whisper import device_queries
    monkeypatch.setattr(device_queries, "nvidia_smi", lambda: str(tmp_path / "nvidia-smi.exe"))
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: b"5324, 8192\n")
    assert _app("CUDA")._accelerator_memory() == " VRAM 5.2/8.0G"


def test_cpu_has_no_accelerator_memory():
    assert _app("CPU")._accelerator_memory() == ""
