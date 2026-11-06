# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "qwen-tts==0.1.1", "torch==2.6.0", "torchaudio==2.6.0",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu124"
# url = "https://download.pytorch.org/whl/cu124"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu124" }
# torchaudio = { index = "pytorch-cu124" }
# ///
"""Qwen3-TTS Base voice cloning, in its own uv environment."""
from pathlib import Path

if __package__:
    from . import tts_server as server
else:  # uv run --script: siblings are on sys.path, the app need not be installed.
    import tts_server as server

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
# Bound file I/O separately from the model's reference-text workload.
MAX_TRANSCRIPT_BYTES = 16_000
MAX_TRANSCRIPT_CHARS = 4_000
LANGUAGES = dict(zip(
    ("pt", "en", "zh", "ja", "ko", "de", "fr", "ru", "es", "it", "auto"),
    ("Portuguese", "English", "Chinese", "Japanese", "Korean", "German",
     "French", "Russian", "Spanish", "Italian", "Auto"),
))


def reference(voice):
    if not voice:
        raise ValueError("Qwen3 requires tts_voice / --voice: a WAV with its UTF-8 .txt transcript")
    server.check_voice(voice)
    path = Path(voice).resolve()
    if path.suffix.lower() != ".wav":
        raise ValueError("Qwen3 reference must be a WAV with a matching .txt transcript")
    sidecar = path.with_suffix(".txt")
    try:
        with sidecar.open("rb") as stream:
            raw = stream.read(MAX_TRANSCRIPT_BYTES + 1)
        if len(raw) > MAX_TRANSCRIPT_BYTES:
            raise ValueError(f"Reference transcript {sidecar} exceeds {MAX_TRANSCRIPT_BYTES} bytes")
        text = raw.decode("utf-8-sig").strip()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read reference transcript {sidecar}. Create this UTF-8 file "
                         "with the exact words spoken in the WAV, then restart the TTS server.") from exc
    if len(text) > MAX_TRANSCRIPT_CHARS:
        raise ValueError(f"Reference transcript {sidecar} exceeds {MAX_TRANSCRIPT_CHARS} characters")
    if not text:
        raise ValueError(f"Empty reference transcript {sidecar}; write the exact words spoken "
                         "in the WAV, then restart the TTS server.")
    key = (str(path), path.stat().st_mtime_ns, text)
    return str(path), text, key


class QwenBackend:
    package = "qwen-tts"
    model_id = MODEL_ID
    sr = 24000

    def __init__(self, device, voice):
        self.startup_voice = voice
        self.cache = {}
        self.use(voice)  # Fail before importing/loading any model if the sidecar is missing.
        import torch
        from qwen_tts import Qwen3TTSModel
        if device == "cuda" and not torch.cuda.is_available():
            server.log("CUDA not available; using the CPU (much slower).")
            device = "cpu"
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device == "cuda" else torch.float32
        self.precision = str(dtype).removeprefix("torch.")
        self._debora_device = device.upper()
        self.model = Qwen3TTSModel.from_pretrained(
            MODEL_ID, device_map=device, dtype=dtype, attn_implementation="sdpa")
        # Include the audio tokenizer: it is a wrapper, not a registered child module.
        roots = (self.model.model, self.model.model.speech_tokenizer.model)
        self.cancel_modules = list(dict.fromkeys(
            module for root in roots for module in root.modules() if not list(module.children())))
        with torch.inference_mode():
            self.prepare_voice()

    def use(self, voice):
        self.reference = reference(self.startup_voice if voice is None else voice)

    def prepare_voice(self):
        path, text, key = self.reference
        if key not in self.cache:
            prompt = self.model.create_voice_clone_prompt(
                ref_audio=path, ref_text=text, x_vector_only_mode=False)
            if len(self.cache) >= server.MAX_CACHED_VOICES:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = prompt
        self.prompt = self.cache[key]

    def generate(self, text, language_id):
        import torch
        language = LANGUAGES.get(language_id.lower().split("-")[0], language_id)
        wavs, self.sr = self.model.generate_voice_clone(
            text=text, language=language, voice_clone_prompt=self.prompt,
            non_streaming_mode=True)
        return torch.as_tensor(wavs[0]).reshape(1, -1)


if __name__ == "__main__":
    server.backend_main(QwenBackend, QwenBackend.package, MODEL_ID)
