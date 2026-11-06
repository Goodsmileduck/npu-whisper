# Agent Context for Débora Whisper

## Project Overview
Débora Whisper (`debora-whisper`, formerly `npu-whisper`) is a local voice-to-text dictation engine for Windows, originally built to leverage Intel NPU via OpenVINO. It has since been expanded to support NVIDIA GPUs through `faster-whisper`. It features a desktop overlay (Dynamic Island style), a system tray, and types transcribed text directly into the user's active window.

## Key Architectural Details
- **OS**: Windows 11; requires Administrator privileges for global hotkeys.
- **Naming**: The display name is "Débora Whisper" (with the accent); every identifier, path, package and command uses ASCII (`debora-whisper`, `debora_whisper`, `~/.debora`). See `docs/BRANDING.md`. `paths.py` moves the pre-rename `~/.npu-dictation` and `$MODELS_DIR/npu-whisper` folders on first run; tests set `DEBORA_WHISPER_NO_MIGRATION` so they never touch them.
- **Packaging**: Code lives in the `debora_whisper` package (`app.py`, `dictation_engine.py`, `ui/`). `pyproject.toml` is the single source of dependencies (runtime by default; `cuda`, `export`, `test` extras) and defines the `debora` (tray app) and `debora-cli` commands. Users install with `uv tool install`; tagged `v*` releases publish to PyPI via `.github/workflows/release.yml`. The app must never pip-install at runtime.
- **Hardware Fallback**: Models attempt to load on the requested hardware. If OpenCL/CUDA fails or devices are lost, the engine gracefully falls back (e.g., NPU -> GPU -> CPU).
- **Parakeet Bucketing**: The Parakeet model requires static input shapes for OpenVINO NPU compilation, so it uses pre-compiled shape buckets for its encoder graph. The decoder runs on the GPU or CPU.
- **Logging Subsystem**: Logs are split between `~/.debora/logs/app.log` (startup events, transcription timings, and hardware names) and `~/.debora/logs/telemetry.log` (background stats like CPU, RAM, VRAM or NPU memory, and audio buffer health).
  `Hardware allocation (mode=...)` blocks go to the console and `app.log` on startup, actual loads, mode changes and fallbacks, with unchanged snapshots suppressed. They report actual devices/PIDs separately from advisory `Device plan` lines. STT drafts share the main STT; Parakeet lists encoder buckets, decoder and CPU preprocessing. LLM workers report over JSON and TTS reports its own PID/device through `/health` (never infer it from the `uv` launcher or main process). Inactive resident models remain visible. Claude is external; inventory helpers load no models, and NPU recovery probes report temporary allocations. `pythonw` shortcuts have no console; their summaries remain in `app.log`.
- **NVIDIA GPU Integration**: The project uses a CPU-only PyTorch installation to save disk space. To support `faster-whisper` on CUDA, it dynamically loads NVIDIA DLLs installed via PyPI (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`). NVIDIA cards are detected via a subprocess call to `nvidia-smi`, bypassing `torch.cuda.is_available()`.

## Token-Efficient Workflow
- Start with targeted `rg` searches and read only the relevant functions or line ranges. Avoid dumping entire files, workflows, logs, or test collections when a focused query answers the question.
- Keep track of relevant files, symbols, and findings. Re-read only changed sections or information needed to resolve a new uncertainty.
- Batch independent searches and keep tool output concise. Request summaries, failure details, or filtered results instead of large outputs that will be truncated.
- Finish the intended code and test edits before starting the full test suite. Do not edit source or test files while tests are running; this can invalidate results and cause source-inspection failures.
- Run focused tests for the affected behavior, then the full suite when appropriate or required. Repeat checks only after relevant changes, failures, or unresolved concerns; avoid redundant builds and test runs.
- For long-running tests and CI, use completion notifications or reasonably spaced status checks. Keep progress updates useful without repeatedly querying unchanged status.
- Preserve required validation and correctness. Reduce redundant context and operations rather than skipping necessary investigation or checks.
