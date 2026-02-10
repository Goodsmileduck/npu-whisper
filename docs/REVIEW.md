# Repository Review: npu-whisper

**Date:** February 10, 2026

## Overview
This is a local voice-to-text dictation application leveraging Intel NPU (Neural Processing Unit) via OpenVINO and OpenAI's Whisper model. The project provides zero-cloud speech recognition with 100% local processing.

---

## Strengths

1. **Well-documented architecture** — Clean visual diagrams in README explaining the data flow and fallback strategy (NPU → GPU → CPU)

2. **Intelligent device fallback** — Gracefully degrades from NPU to GPU to CPU when unavailable, with multiple model size options

3. **User-friendly interface**
   - Simple hotkey-based activation (Ctrl+Alt+D by default)
   - Configurable settings stored in JSON
   - Both Python CLI and PowerShell launcher for Windows users
   - Audio beep feedback for recording start/stop

4. **Smart text output** — Uses clipboard + keyboard simulation with recovery for reliable pasting, featuring:
   - Fallback to Windows ctypes if pyperclip/keyboard unavailable
   - Optional auto-Enter for Claude Code integration
   - Clipboard restoration after paste

5. **Model export optimization**
   - INT8 quantized models for NPU efficiency
   - Support for base/small/medium Whisper sizes
   - Lazy loading of model on first use (fast startup)

6. **Comprehensive setup automation**
   - Python virtual environment management
   - Automated dependency installation
   - NPU driver detection and diagnostics
   - Interactive setup wizard

---

## Areas for Improvement

### Error Handling Gaps
- No graceful handling for corrupted/incomplete audio files
- Model export fallback retry logic only removes one flag; could be more robust
- No timeout protection for long model loading operations
- Limited validation of audio duration before processing

### Threading Concerns
- Recording and UI responsiveness could block on large audio files
- No explicit thread cleanup or join operations
- Daemon threads could leave orphaned processes

### Input Validation
- Config file has no validation before use
- No bounds checking on `max_record_seconds`
- Hotkey string format not validated before passing to `keyboard` library

### Incomplete PowerShell Script
- The script is truncated (line 231 appears incomplete)
- Missing help text implementation (`if ($Help)` branch)
- Model verification logic cuts off mid-implementation

### Audio Processing Limitations
- Single-channel mono only (no stereo support)
- No noise cancellation or audio preprocessing
- Fixed 16kHz sample rate (though correct for Whisper)
- No detection of silence periods before processing

### Dependency Management
- Requirements don't pin exact versions (uses `>=2025.1`)
- Could cause compatibility issues if major API changes occur
- No Python version upper bound specified in requirements

### Logging & Debugging
- Truncated log entries for long transcriptions (80 char limit)
- No timestamp in console output (only in file logs)
- No debug mode for troubleshooting issues

### Resource Management
- No memory usage tracking or limits
- Audio frames stored in memory completely (no streaming)
- No cleanup of temporary model cache on errors

### Documentation Gaps
- No troubleshooting section beyond README
- No performance benchmarks (RTF metrics shown but no baselines)
- Missing language code examples beyond "en"

### Cross-platform Support
- Windows-specific (uses `winsound`, `ctypes.windll`, Device Manager checks)
- Could limit to documented Windows 10/11 in setup

---

## Code Quality Issues

- **Magic numbers**: Sample rate (16000), frame sizes (1024), timeouts hardcoded with no explanation
- **Bare except clauses**: Some error handling too broad (e.g., `except Exception` in fallback code)
- **Type hints incomplete**: Function parameters lack type annotations in several places
- **Config mutation**: `config` dict modified in place during CLI processing (no immutability)

---

## Recommendations

1. **Add proper type hints** throughout (use `typing.Optional`, etc.)
2. **Implement structured logging** with levels (DEBUG, INFO, WARNING, ERROR)
3. **Add unit tests** for audio processing and config management
4. **Complete the PowerShell script** and add help documentation
5. **Add audio pre-processing** options (silence detection, volume normalization)
6. **Pin dependency versions** in requirements.txt with tested combinations
7. **Implement context managers** for resource cleanup (streams, models)
8. **Add retry logic** with exponential backoff for model operations

---

## Security Considerations

- Clipboard modification could be exploited in sensitive contexts — consider user warnings
- No file system sandbox; config/models stored in user home directory without validation
- Hotkey registration is global — could interfere with system shortcuts

---

## Summary

This is a **well-architected and user-friendly project** with solid core functionality. Minor enhancements to error handling, testing, and documentation would make it production-ready.

| Category | Rating | Notes |
|----------|--------|-------|
| Architecture | ⭐⭐⭐⭐⭐ | Clean design with good separation of concerns |
| User Experience | ⭐⭐⭐⭐☆ | Intuitive but could use better feedback |
| Code Quality | ⭐⭐⭐☆☆ | Good overall, needs type hints and tests |
| Documentation | ⭐⭐⭐⭐☆ | Strong README, could expand troubleshooting |
| Robustness | ⭐⭐⭐☆☆ | Good fallbacks, needs error handling polish |
