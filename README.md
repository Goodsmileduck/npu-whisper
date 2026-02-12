# NPU Whisper

Local voice-to-text dictation for Windows, powered by Intel NPU via OpenVINO. Press a hotkey, speak, and text appears at your cursor. Zero cloud, zero cost, zero data leaving your machine.

![Windows](https://img.shields.io/badge/platform-Windows%2011-0078D4?logo=windows)
![Python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **NPU-accelerated** — runs on Intel AI Boost (Meteor Lake / Lunar Lake / Arrow Lake)
- **5 models** — Whisper base/small/medium/turbo + Parakeet TDT 0.6B (best accuracy)
- **15 languages** — English, Russian, Spanish, French, German, Japanese, Chinese, Korean, and more
- **Dynamic Island overlay** — always-visible floating pill with waveform, timer, and status
- **System tray** — lives in your taskbar, right-click for settings/history
- **One-click settings** — language-first model picker with download status badges
- **Claude Code mode** — auto-paste + Enter for hands-free prompt submission
- **Audio chimes** — pleasant start/stop feedback
- **Auto-fallback** — NPU -> GPU -> CPU, DEVICE_LOST auto-recovery

## Requirements

- **Windows 11**
- **Intel Core Ultra** CPU with NPU (Meteor Lake / Lunar Lake / Arrow Lake), or any Intel CPU with iGPU
- **Python 3.10+**
- **16 GB RAM** recommended (NPU/GPU share system memory)

## Quick Start

```powershell
# 1. Clone
git clone https://github.com/Goodsmileduck/npu-whisper.git
cd npu-whisper

# 2. First-time setup (creates venv, installs deps, downloads model, warms NPU cache)
.\Start-Dictation.ps1 -Setup

# 3. Launch (GUI mode with system tray + Dynamic Island overlay)
.\Start-Dictation.ps1
```

Press **Ctrl+Space** to start recording, press again to stop. Transcribed text is pasted at your cursor.

## Models

| Model | Params | Device | Speed (3s audio) | WER | Languages | Notes |
|-------|--------|--------|-------------------|-----|-----------|-------|
| **base** | 74M | NPU | 0.2s | 5.0% | All 15 | Default, great for short commands |
| **small** | 244M | NPU | 0.6s | 3.4% | All 15 | Balanced speed/accuracy |
| **parakeet** | 600M | NPU+GPU | 0.2s | 3.7% | English only | Best English accuracy on NPU |
| **medium** | 769M | GPU | — | 2.9% | All 15 | High accuracy, slower |
| **turbo** | 809M | GPU | 2.4s | 2.3% | All 15 | Best multilingual quality |

> **WER** = Word Error Rate on LibriSpeech test-clean (lower is better). Whisper WER from [HuggingFace model cards](https://huggingface.co/openai/whisper-base); Parakeet from [NVIDIA](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

The Settings dialog shows which models are already downloaded and filters them by your selected language (e.g., selecting Russian hides the English-only Parakeet model).

## Usage

### GUI Mode (default)

```powershell
.\Start-Dictation.ps1
```

Launches with a Dynamic Island overlay at the top of your screen and a system tray icon. Click the green dot or press Ctrl+Space to record.

### CLI Mode

```powershell
.\Start-Dictation.ps1 -CLI
```

Console-only, no GUI. Press Ctrl+Space to toggle recording.

### Claude Code Mode

```powershell
.\Start-Dictation.ps1 -AutoEnter
```

Pastes text and presses Enter automatically — speak your prompt and it submits to Claude Code.

### Override Settings

```powershell
.\Start-Dictation.ps1 -Device GPU -Model turbo -Language ru
```

## Configuration

Stored at `~/.npu-dictation/config.json`:

```json
{
  "device": "NPU",
  "model_size": "base",
  "language": "en",
  "hotkey": "ctrl+space",
  "auto_enter": false,
  "beep_on_start": true,
  "max_record_seconds": 60,
  "sample_rate": 16000
}
```

Or change settings from the GUI: right-click the system tray icon and select **Settings**.

## File Paths

| Path | Purpose |
|------|---------|
| `~/.npu-dictation/config.json` | User configuration |
| `~/.npu-dictation/models/` | Downloaded model files |
| `~/.npu-dictation/ov-cache/` | OpenVINO compilation cache (do not delete) |
| `~/.npu-dictation/dictation.log` | Runtime log |
| `~/.npu-dictation/venv/` | Python virtual environment |

## How It Works

```
+-------------------+     +------------------+     +------------------+
|  Ctrl+Space       | --> |  Microphone      | --> |  Whisper or      |
|  (global hotkey)  |     |  16kHz mono      |     |  Parakeet model  |
+-------------------+     +------------------+     +--------+---------+
                                                            |
                          +------------------+              |
                          |  Clipboard paste | <------------+
                          |  into any window |
                          +------------------+
```

The model runs **100% locally** on your Intel NPU. No internet required after initial model download.

**Whisper** models use OpenVINO GenAI's `WhisperPipeline` with INT8 quantization on NPU or GPU.

**Parakeet TDT** uses a three-stage hybrid pipeline for best accuracy:

```
+---------------+     +----------------+     +----------------+
|  nemo128.onnx | --> |  Encoder       | --> |  TDT Decoder   |
|  Mel spectro  |     |  OpenVINO NPU  |     |  OpenVINO GPU  |
|  (CPU)        |     |                |     |  (CPU fallback)|
+---------------+     +----------------+     +----------------+
```

## Troubleshooting

### NPU device not found
- Check Device Manager for "Intel(R) AI Boost" or "Intel(R) NPU Accelerator"
- Install/update NPU driver from Windows Update or [Intel's site](https://www.intel.com/content/www/us/en/download/794734/)
- Minimum driver version: 32.0.100.3104

### Slow first run
OpenVINO compiles the model graph for your specific NPU on first launch. This takes 1-15 minutes depending on model size and is cached for subsequent runs.

### DEVICE_LOST error
The NPU driver crashed. Reboot to reset it. The GUI auto-falls back to GPU when this happens.

### Hotkey doesn't work
- PowerShell must run as **Administrator** (the `keyboard` library requires elevated privileges)
- Check no other app is capturing the same hotkey
- Try a different hotkey: `.\Start-Dictation.ps1 -Hotkey "ctrl+shift+space"`

### Audio not recording
- Check Windows Settings -> Privacy -> Microphone permissions
- Ensure your mic is the default recording device
- Select a specific mic in the Settings dialog

## Development

```powershell
# Run tests (82 tests)
~\.npu-dictation\venv\Scripts\python.exe -m pytest tests/ -v

# Auto-reload during development
pip install watchfiles
watchfiles "python app.py" .
```

## License

MIT
