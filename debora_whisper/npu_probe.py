"""Checks, in a process of its own, whether a lost NPU works again.

Loading a model on a lost NPU can hang inside the driver while holding the
GIL: inside the app that froze everything (tray, hotkey, Ctrl+C), and nothing
in the process could cancel it. Here the app waits with a timeout and kills
this process if it hangs.

    python -m debora_whisper.npu_probe <model_path> <backend> <model_size> <language> <sample_rate>

Exit code 0: the model loaded on the NPU and transcribed half a second of
silence. Otherwise the last line of output says why.
"""
import subprocess
import json
import sys
from pathlib import Path

# A first compile after a driver reset took 169 s on whisper-turbo.
NPU_PROBE_TIMEOUT = 300


def run(model_path: str, backend: str, model_size: str, language: str, sample_rate: int) -> int:
    import numpy as np
    from debora_whisper.dictation_engine import create_model
    try:
        model = create_model(Path(model_path), device="NPU", backend=backend, model_size=model_size)
        from debora_whisper.hardware_allocation import model_description
        print("@allocation " + json.dumps(model_description(model, model_size, "NPU")), flush=True)
        # Loaders fall back silently on benign errors, so loading is not
        # proof the NPU works.
        if model.device != "NPU":
            print(f"probe model loaded on {model.device}, not NPU")
            return 1
        model.transcribe(np.zeros(sample_rate // 2, dtype=np.float32),
                         sample_rate=sample_rate, language=language)
    except Exception as e:
        print(f"{type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else e}")
        return 1
    return 0


def probe_command(config: dict, model_path) -> list[str]:
    from debora_whisper.dictation_engine import MODEL_REGISTRY
    from debora_whisper.processes import python_executable
    return [python_executable(), "-m", "debora_whisper.npu_probe", str(model_path),
            MODEL_REGISTRY[config["model_size"]]["backend"], config["model_size"],
            config.get("language", "en"), str(config["sample_rate"])]


def probe_npu(config: dict, model_path, timeout=NPU_PROBE_TIMEOUT, command=None):
    """Return if the NPU runs the model again; raise RuntimeError otherwise."""
    from debora_whisper.processes import NO_WINDOW, kill_tree
    process = subprocess.Popen(
        command or probe_command(config, model_path),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=NO_WINDOW)
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(process)
        # Windows communicate() owns a reader thread. Drain it after killing
        # the child, but bound the wait if native driver teardown is stuck.
        try:
            process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        raise RuntimeError(f"no answer in {timeout}s (the NPU driver is probably "
                           f"hung); probe process {process.pid} killed") from None
    from debora_whisper.dictation_engine import log
    for line in output.decode("utf-8", "replace").splitlines():
        if line.startswith("@allocation "):
            log("Hardware allocation (mode=npu_probe):\n  STT probe  "
                + json.loads(line.removeprefix("@allocation "))
                + "; temporary worker now exited")
    if process.returncode != 0:
        lines = output.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(lines[-1] if lines else f"exit code {process.returncode}")


if __name__ == "__main__":
    path, backend, size, language, rate = sys.argv[1:6]
    sys.exit(run(path, backend, size, language, int(rate)))
