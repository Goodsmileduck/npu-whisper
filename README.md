# npu-whisper
App for do dictation using Intel NPU
# NPU Dictation Engine

Local voice-to-text dictation powered by your Intel NPU (AI Boost) — zero cloud, zero subscription, zero data leaving your machine.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Start-Dictation.ps1  (PowerShell launcher)             │
│  - Manages venv, checks NPU driver, launches engine     │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  dictation_engine.py   (Python app)                     │
│                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────┐  │
│  │  Global       │   │  Audio       │   │  Text      │  │
│  │  Hotkey       │──▶│  Recorder    │──▶│  Output    │  │
│  │  (keyboard)   │   │  (sounddev)  │   │  (paste)   │  │
│  └──────────────┘   └──────┬───────┘   └────────────┘  │
│                            │                            │
│                  ┌─────────▼─────────┐                  │
│                  │  WhisperNPU       │                  │
│                  │  (OpenVINO)       │                  │
│                  │                   │                  │
│                  │  NPU ← preferred  │                  │
│                  │  GPU ← fallback   │                  │
│                  │  CPU ← last resort│                  │
│                  └───────────────────┘                  │
└─────────────────────────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  Intel NPU          │
              │  (AI Boost)         │
              │  via OpenVINO IR    │
              │  INT8 quantized     │
              └─────────────────────┘
```

## How It Works

1. **Hotkey press** (Ctrl+Alt+D) → starts recording from microphone
2. **Hotkey press again** → stops recording
3. **Audio** → sent to Whisper model running on your NPU via OpenVINO
4. **Transcribed text** → pasted into whatever window is active (terminal, browser, etc.)

The model runs **100% locally** on your Intel NPU. No internet required after setup.

## Requirements

- **OS:** Windows 10/11
- **CPU:** Intel Core Ultra (Meteor Lake / Lunar Lake) with NPU, or any Intel CPU (falls back to iGPU/CPU)
- **Python:** 3.10+
- **NPU Driver:** Intel NPU driver via Windows Update or [manual download](https://www.intel.com/content/www/us/en/download/794734/)

## Quick Start

```powershell
# 1. First-time setup (installs deps, detects NPU, exports Whisper model)
.\Start-Dictation.ps1 -Setup

# 2. Start dictating
.\Start-Dictation.ps1

# 3. Press Ctrl+Alt+D to start recording, press again to stop
#    Text appears wherever your cursor is!
```

## Usage with Claude Code

```powershell
# Auto-press Enter after dictation — speaks directly to Claude Code
.\Start-Dictation.ps1 -AutoEnter

# Custom hotkey that won't conflict with terminal shortcuts
.\Start-Dictation.ps1 -AutoEnter -Hotkey "ctrl+shift+space"
```

## Configuration

Config is stored at `~/.npu-dictation/config.json`:

```json
{
  "device": "NPU",
  "model_size": "base",
  "language": "en",
  "hotkey": "ctrl+alt+d",
  "auto_enter": false,
  "auto_punctuate": true,
  "beep_on_start": true,
  "max_record_seconds": 60,
  "sample_rate": 16000
}
```

### CLI Overrides

```powershell
# Use GPU instead of NPU
.\Start-Dictation.ps1 -Device GPU

# Use larger model for better accuracy
.\Start-Dictation.ps1 -Model small

# Russian language
.\Start-Dictation.ps1 -Language ru

# Indonesian
.\Start-Dictation.ps1 -Language id
```

## Model Sizes

| Model  | RAM    | NPU Speed  | Accuracy | Best For                    |
|--------|--------|------------|----------|-----------------------------|
| base   | ~1 GB  | Very fast  | Good     | Short commands, Claude Code |
| small  | ~2 GB  | Fast       | Better   | General dictation           |
| medium | ~4 GB  | Moderate   | Best     | Long-form, technical terms  |

> **Note:** `large` model exceeds current NPU memory limits and is not supported.

## If NPU Isn't Available

The engine automatically falls back:
1. **NPU** → fastest, lowest power (requires Intel Core Ultra)
2. **GPU** → Intel iGPU via OpenVINO, still fast
3. **CPU** → works on any machine, slower but reliable

You can force a device: `.\Start-Dictation.ps1 -Device CPU`

## Troubleshooting

### "NPU device not found"
- Check Device Manager for "Intel(R) AI Boost" or "Intel(R) NPU Accelerator"
- Install/update NPU driver from Windows Update or Intel's site
- Reboot after driver install

### Slow first transcription
- First run compiles the model for NPU (5-15 min). Cached after that.
- Subsequent startups are much faster.

### Model export fails
- Ensure you have enough disk space (~2-5 GB for model files)
- Try: `.\Start-Dictation.ps1 -Setup -Device CPU` to verify basics work
- Then retry with NPU

### Hotkey doesn't work
- Run PowerShell as Administrator
- Check no other app is capturing the same hotkey
- Try a different hotkey: `-Hotkey "ctrl+shift+space"`

### Audio not recording
- Check Windows microphone permissions (Settings → Privacy → Microphone)
- Ensure your mic is the default recording device

## Project Structure

```
npu-whisper/
├── Start-Dictation.ps1     # PowerShell launcher (entry point)
├── dictation_engine.py      # Python engine (core logic)
└── README.md                # This file

~/.npu-dictation/            # Created at runtime
├── config.json              # User configuration
├── dictation.log            # Runtime logs
├── venv/                    # Python virtual environment
└── models/                  # Exported OpenVINO models
    └── whisper-base-openvino/
```

## How This Uses Your NPU

The Intel NPU (AI Boost) is a dedicated neural network accelerator designed for
sustained AI workloads with low power consumption. Unlike the CPU/GPU which handle
general computation, the NPU is optimized for matrix operations used in transformer
models like Whisper.

OpenVINO converts the Whisper model to INT8 precision and compiles it specifically
for your NPU's architecture. The compiled model is cached, so subsequent loads are
near-instant.

Typical performance on Intel Core Ultra NPU:
- **whisper-base**: ~3x real-time (10s audio → ~3s processing)
- **whisper-small**: ~1.5x real-time (10s audio → ~7s processing)

This means for typical Claude Code prompts (5-15 seconds of speech), you get
transcription in 2-5 seconds — fast enough to feel natural.

## License

MIT — do whatever you want with it.