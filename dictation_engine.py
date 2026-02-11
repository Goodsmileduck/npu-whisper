"""
NPU Dictation Engine - Local voice-to-text using OpenVINO Whisper on Intel NPU
Designed for Windows with Intel Core Ultra NPU (AI Boost)

Usage: python dictation_engine.py [--setup] [--device NPU|GPU|CPU] [--model base|small|medium]
"""

import sys
import os
import time
import json
import argparse
import threading
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG_DIR = Path.home() / ".npu-dictation"
CONFIG_FILE = CONFIG_DIR / "config.json"
MODEL_DIR = CONFIG_DIR / "models"
LOG_FILE = CONFIG_DIR / "dictation.log"

DEFAULT_CONFIG = {
    "device": "NPU",           # NPU, GPU, CPU
    "model_size": "base",      # base, small, medium (large not supported on NPU)
    "language": "en",          # Language code or "auto"
    "hotkey": "ctrl+alt+d",    # Global hotkey to toggle recording
    "auto_enter": False,       # Press Enter after pasting (useful for Claude Code)
    "beep_on_start": True,     # Audio feedback when recording starts/stops
    "max_record_seconds": 60,  # Max recording length
    "sample_rate": 16000,      # Whisper expects 16kHz
}

# Model registry: pre-exported OpenVINO IR models from HuggingFace (preferred for NPU)
# Fallback uses optimum-cli export from original HuggingFace models
MODEL_REGISTRY = {
    "base": {
        "repo": "openai/whisper-base",
        "ov_repo": "OpenVINO/whisper-base-int8-ov",
        "description": "Fastest, good for short commands (~1GB RAM)",
    },
    "small": {
        "repo": "openai/whisper-small",
        "ov_repo": "OpenVINO/whisper-small-int8-ov",
        "description": "Balanced speed/accuracy (~2GB RAM)",
    },
    "medium": {
        "repo": "openai/whisper-medium",
        "ov_repo": "OpenVINO/whisper-medium-int8-ov",
        "description": "Best accuracy on NPU (~4GB RAM, slower)",
    },
    "turbo": {
        "repo": "openai/whisper-large-v3-turbo",
        "ov_repo": "FluidInference/whisper-large-v3-turbo-int4-ov-npu",
        "description": "Large-v3 distilled, INT4 NPU-optimized (~566MB)",
    },
}


def log(msg: str):
    """Simple logging to file and console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Config management
# ---------------------------------------------------------------------------
def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            saved = json.load(f)
        config = {**DEFAULT_CONFIG, **saved}
    else:
        config = DEFAULT_CONFIG.copy()
    return config


def save_config(config: dict):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    log(f"Config saved to {CONFIG_FILE}")


def validate_config(config: dict):
    """Validate config values. Raises ValueError on invalid values."""
    valid_devices = {"NPU", "GPU", "CPU"}
    if config.get("device") not in valid_devices:
        raise ValueError(f"device must be one of {valid_devices}, got '{config.get('device')}'")

    valid_models = set(MODEL_REGISTRY.keys())
    if config.get("model_size") not in valid_models:
        raise ValueError(f"model_size must be one of {valid_models}, got '{config.get('model_size')}'")

    sr = config.get("sample_rate")
    if not isinstance(sr, (int, float)) or sr <= 0:
        raise ValueError(f"sample_rate must be a positive number, got '{sr}'")

    max_rec = config.get("max_record_seconds")
    if max_rec is not None and (not isinstance(max_rec, (int, float)) or max_rec <= 0):
        raise ValueError(f"max_record_seconds must be a positive number or null, got '{max_rec}'")


# ---------------------------------------------------------------------------
# Model setup (export Whisper to OpenVINO IR format)
# ---------------------------------------------------------------------------
def setup_model(config: dict):
    """Download or export Whisper model in OpenVINO format."""
    model_size = config["model_size"]
    model_info = MODEL_REGISTRY[model_size]
    model_path = MODEL_DIR / f"whisper-{model_size}-openvino"

    if model_path.exists() and any(model_path.glob("*.xml")):
        log(f"Model already available at {model_path}")
        return model_path

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Strategy 1: Download pre-exported OpenVINO model from HuggingFace (preferred)
    ov_repo = model_info.get("ov_repo")
    if ov_repo:
        log(f"Downloading pre-exported model from {ov_repo}...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(ov_repo, local_dir=str(model_path))
            if any(model_path.glob("*.xml")):
                log(f"Model downloaded to {model_path}")
                return model_path
        except Exception as e:
            log(f"Download failed: {e}, falling back to local export...")

    # Strategy 2: Export locally using optimum-cli
    import subprocess
    log(f"Exporting {model_info['repo']} to OpenVINO format...")
    log(f"This may take several minutes on first run.")

    weight_format = "int8" if config["device"] == "NPU" else "fp16"

    cmd = [
        sys.executable, "-m", "optimum.commands.optimum_cli", "export", "openvino",
        "--model", model_info["repo"],
        "--task", "automatic-speech-recognition",
        "--weight-format", weight_format,
        str(model_path),
    ]

    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log(f"Export failed: {result.stderr}")
        sys.exit(1)

    log(f"Model exported to {model_path}")
    return model_path


# ---------------------------------------------------------------------------
# Whisper pipeline using OpenVINO GenAI
# ---------------------------------------------------------------------------
class WhisperNPU:
    """Whisper speech-to-text using OpenVINO on NPU/GPU/CPU."""

    def __init__(self, model_path: Path, device: str = "NPU"):
        self.model_path = model_path
        self.device = device
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self):
        """Load the OpenVINO Whisper pipeline."""
        log(f"Loading Whisper pipeline on {self.device}...")
        start = time.time()

        try:
            # Try openvino_genai first (recommended for Whisper)
            import openvino_genai as ov_genai
            self.pipeline = ov_genai.WhisperPipeline(
                str(self.model_path), self.device
            )
            self._backend = "genai"
            log(f"Loaded via openvino_genai in {time.time() - start:.1f}s")
            return
        except ImportError:
            log("openvino_genai not found, trying optimum-intel...")
        except Exception as e:
            log(f"openvino_genai failed: {e}, trying optimum-intel...")

        try:
            # Fallback to optimum-intel
            from optimum.intel import OVModelForSpeechSeq2Seq
            from transformers import AutoProcessor
            
            self._processor = AutoProcessor.from_pretrained(str(self.model_path))
            self._model = OVModelForSpeechSeq2Seq.from_pretrained(
                str(self.model_path),
                device=self.device,
            )
            self._backend = "optimum"
            log(f"Loaded via optimum-intel in {time.time() - start:.1f}s")
        except Exception as e:
            log(f"Failed to load model: {e}")
            log("Falling back to CPU...")
            if self.device != "CPU":
                self.device = "CPU"
                self._load_pipeline()
            else:
                raise

    def transcribe(self, audio_data, sample_rate: int = 16000, language: str = "en") -> str:
        """Transcribe audio numpy array to text."""
        import numpy as np
        start = time.time()

        # Ensure float32 normalized to [-1, 1]
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        if self._backend == "genai":
            config = self.pipeline.get_generation_config()
            config.max_new_tokens = 448
            if language and language != "auto":
                config.language = f"<|{language}|>"
            config.task = "transcribe"
            config.return_timestamps = False
            
            result = self.pipeline.generate(audio_data, config)
            text = str(result).strip()

        elif self._backend == "optimum":
            import torch
            inputs = self._processor(
                audio_data, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            )
            forced_decoder_ids = self._processor.get_decoder_prompt_ids(
                language=language, task="transcribe"
            )
            predicted_ids = self._model.generate(
                inputs["input_features"],
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=448,
            )
            text = self._processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()

        elapsed = time.time() - start
        audio_duration = len(audio_data) / sample_rate
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        log(f"Transcribed {audio_duration:.1f}s audio in {elapsed:.1f}s (RTF: {rtf:.2f}) on {self.device}")
        
        return text


# ---------------------------------------------------------------------------
# Audio recording
# ---------------------------------------------------------------------------
class AudioRecorder:
    """Record audio from microphone using sounddevice."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 max_record_seconds: float = None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_record_seconds = max_record_seconds
        self.recording = False
        self._frames = []
        self._stream = None
        self._lock = threading.Lock()
        self._timer = None

    def start(self):
        """Start recording."""
        import sounddevice as sd

        with self._lock:
            self._frames = []
            self.recording = True

        def callback(indata, frames, time_info, status):
            with self._lock:
                if self.recording:
                    self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=1024,
            callback=callback,
        )
        self._stream.start()
        log("Recording started...")

        # Enforce max recording duration
        if self.max_record_seconds:
            self._timer = threading.Timer(self.max_record_seconds, self._timeout_stop)
            self._timer.daemon = True
            self._timer.start()

    def _timeout_stop(self):
        """Called when max_record_seconds is reached."""
        if self.recording:
            log(f"Max recording time ({self.max_record_seconds}s) reached, stopping.")
            self.stop()

    def stop(self):
        """Stop recording and return audio as numpy array."""
        import numpy as np

        with self._lock:
            self.recording = False
            frames = list(self._frames)
            self._frames = []

        if self._timer:
            self._timer.cancel()
            self._timer = None

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not frames:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(frames, axis=0).flatten()
        duration = len(audio) / self.sample_rate
        log(f"Recording stopped. Duration: {duration:.1f}s")
        return audio


# ---------------------------------------------------------------------------
# Text output (type into active window)
# ---------------------------------------------------------------------------
def type_text(text: str, auto_enter: bool = False):
    """Type text into the currently active window using keyboard simulation."""
    if not text:
        return

    try:
        # Try pyperclip + keyboard for reliable pasting
        import pyperclip
        import keyboard

        # Save current clipboard
        try:
            old_clipboard = pyperclip.paste()
        except Exception:
            old_clipboard = ""

        # Copy transcribed text and paste
        pyperclip.copy(text)
        time.sleep(0.05)
        keyboard.press_and_release("ctrl+v")
        time.sleep(0.1)

        if auto_enter:
            keyboard.press_and_release("enter")

        # Restore clipboard after a short delay
        def restore():
            time.sleep(0.5)
            try:
                pyperclip.copy(old_clipboard)
            except Exception:
                pass

        threading.Thread(target=restore, daemon=True).start()
        log(f"Typed: {text[:80]}{'...' if len(text) > 80 else ''}")

    except ImportError:
        # Fallback: use ctypes SendInput on Windows
        log("pyperclip/keyboard not found, using ctypes fallback")
        _type_text_ctypes(text)


def _type_text_ctypes(text: str):
    """Fallback text typing using Windows ctypes."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    
    # Use SendInput with Unicode characters
    INPUT_KEYBOARD = 1
    KEYEVENTF_UNICODE = 0x0004
    KEYEVENTF_KEYUP = 0x0002

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        ]

    class INPUT(ctypes.Structure):
        class _INPUT(ctypes.Union):
            _fields_ = [
                ("ki", KEYBDINPUT),
                ("mi", MOUSEINPUT),
                ("hi", HARDWAREINPUT),
            ]
        _fields_ = [("type", wintypes.DWORD), ("_input", _INPUT)]

    for char in text:
        inputs = (INPUT * 2)()
        # Key down
        inputs[0].type = INPUT_KEYBOARD
        inputs[0]._input.ki.wScan = ord(char)
        inputs[0]._input.ki.dwFlags = KEYEVENTF_UNICODE
        # Key up
        inputs[1].type = INPUT_KEYBOARD
        inputs[1]._input.ki.wScan = ord(char)
        inputs[1]._input.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP

        user32.SendInput(2, ctypes.byref(inputs), ctypes.sizeof(INPUT))
        time.sleep(0.002)


# ---------------------------------------------------------------------------
# Audio feedback (beep)
# ---------------------------------------------------------------------------
def beep(freq: int = 800, duration_ms: int = 150):
    """Play a short beep sound."""
    try:
        import winsound
        winsound.Beep(freq, duration_ms)
    except Exception:
        pass  # Non-Windows or no audio


# ---------------------------------------------------------------------------
# Main application loop
# ---------------------------------------------------------------------------
class DictationApp:
    """Main dictation application with hotkey toggle."""

    def __init__(self, config: dict):
        self.config = config
        self.recorder = AudioRecorder(
            sample_rate=config["sample_rate"],
            max_record_seconds=config.get("max_record_seconds"),
        )
        self.whisper = None  # Lazy-loaded
        self.is_recording = False

    def ensure_model(self):
        """Ensure model is loaded."""
        if self.whisper is None:
            model_path = setup_model(self.config)
            self.whisper = WhisperNPU(model_path, device=self.config["device"])

    def toggle_recording(self):
        """Toggle recording on/off."""
        if self.is_recording:
            # Stop recording and transcribe
            if self.config["beep_on_start"]:
                threading.Thread(target=beep, args=(600, 100), daemon=True).start()

            audio = self.recorder.stop()
            self.is_recording = False

            if len(audio) < self.config["sample_rate"] * 0.3:
                log("Recording too short, ignoring.")
                return

            # Transcribe in background to keep hotkey listener responsive
            def _transcribe_and_type():
                self.ensure_model()
                text = self.whisper.transcribe(
                    audio,
                    sample_rate=self.config["sample_rate"],
                    language=self.config["language"],
                )
                if text:
                    type_text(text, auto_enter=self.config["auto_enter"])
                else:
                    log("No speech detected.")

            threading.Thread(target=_transcribe_and_type, daemon=True).start()
        else:
            # Start recording
            if self.config["beep_on_start"]:
                threading.Thread(target=beep, args=(800, 150), daemon=True).start()

            self.is_recording = True
            self.recorder.start()

    def run(self):
        """Run the main application loop with global hotkey."""
        import keyboard

        hotkey = self.config["hotkey"]
        
        log("=" * 60)
        log("NPU Dictation Engine")
        log(f"  Device:  {self.config['device']}")
        log(f"  Model:   whisper-{self.config['model_size']}")
        log(f"  Hotkey:  {hotkey}")
        log(f"  Lang:    {self.config['language']}")
        log(f"  Auto-Enter: {self.config['auto_enter']}")
        log("=" * 60)
        log(f"Press {hotkey} to start/stop dictation. Ctrl+C to quit.")
        log("")

        # Pre-load model on startup
        log("Pre-loading model (first time may take several minutes)...")
        self.ensure_model()
        log("Ready! Waiting for hotkey...")

        # Register global hotkey
        keyboard.add_hotkey(hotkey, self.toggle_recording, suppress=True)

        try:
            keyboard.wait()  # Block forever, handling hotkeys
        except KeyboardInterrupt:
            log("\nShutting down...")
            if self.is_recording:
                self.recorder.stop()


# ---------------------------------------------------------------------------
# Setup / Install dependencies
# ---------------------------------------------------------------------------
def run_setup():
    """Interactive setup: install dependencies and export model."""
    import subprocess

    log("=" * 60)
    log("NPU Dictation Engine - Setup")
    log("=" * 60)

    # 1. Check Python version
    log(f"Python: {sys.version}")
    
    # 2. Install pip dependencies
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        log("\nInstalling dependencies from requirements.txt...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file), "--quiet"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            log(f"  WARNING: pip install failed: {result.stderr[:500]}")
        else:
            log("  Dependencies installed successfully.")
    else:
        log(f"\nWARNING: requirements.txt not found at {requirements_file}")
        log("  Install manually: pip install -r requirements.txt")

    # 3. Check NPU availability
    log("\nChecking available devices...")
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        log(f"  Available OpenVINO devices: {devices}")
        
        if "NPU" in devices:
            log("  [OK] NPU detected!")
        else:
            log("  [--] NPU not found. Will fall back to GPU or CPU.")
            log("    Make sure Intel NPU driver is installed via Windows Update")
            log("    or from: https://www.intel.com/content/www/us/en/download/794734/")
        
        if "GPU" in devices:
            log("  [OK] GPU detected (Intel iGPU)")
    except ImportError:
        log("  Could not import openvino - installation may have failed")

    # 4. Create default config
    config = load_config()
    
    # Auto-detect best device
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        if "NPU" in devices:
            config["device"] = "NPU"
        elif "GPU" in devices:
            config["device"] = "GPU"
        else:
            config["device"] = "CPU"
    except Exception:
        config["device"] = "CPU"

    save_config(config)

    # 5. Export model
    log(f"\nExporting whisper-{config['model_size']} for {config['device']}...")
    try:
        model_path = setup_model(config)
        log(f"Model ready at: {model_path}")
    except Exception as e:
        log(f"Model export failed: {e}")
        log("You can retry later with: python dictation_engine.py --setup")

    log("\n" + "=" * 60)
    log("Setup complete!")
    log(f"Config file: {CONFIG_FILE}")
    log(f"Start dictating: python dictation_engine.py")
    log(f"Or use the PowerShell launcher: .\\Start-Dictation.ps1")
    log("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NPU Dictation Engine")
    parser.add_argument("--setup", action="store_true", help="Run first-time setup")
    parser.add_argument("--device", choices=["NPU", "GPU", "CPU"], help="Override device")
    parser.add_argument("--model", choices=["base", "small", "medium", "turbo"], help="Model size")
    parser.add_argument("--language", type=str, help="Language code (e.g., en, ru, id)")
    parser.add_argument("--auto-enter", action="store_true", help="Press Enter after typing")
    parser.add_argument("--hotkey", type=str, help="Global hotkey (e.g., ctrl+alt+d)")
    args = parser.parse_args()

    if args.setup:
        run_setup()
        return

    config = load_config()

    # Apply CLI overrides
    if args.device:
        config["device"] = args.device
    if args.model:
        config["model_size"] = args.model
    if args.language:
        config["language"] = args.language
    if args.auto_enter:
        config["auto_enter"] = True
    if args.hotkey:
        config["hotkey"] = args.hotkey

    validate_config(config)

    app = DictationApp(config)
    app.run()


if __name__ == "__main__":
    main()