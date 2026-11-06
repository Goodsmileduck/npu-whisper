# /// script
# requires-python = "==3.11.*"
# dependencies = [
#     "coqui-tts==0.27.5",
#     # 5.x removed isin_mps_friendly; before 4.57 lacks
#     # _extract_generation_mode_kwargs. Both break XTTS generation.
#     "transformers==4.57.6",
#     "torch==2.6.0", "torchaudio==2.6.0",
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
"""Coqui XTTS-v2 voice cloning, in its own uv environment.

XTTS-v2's weights are under the Coqui Public Model License (CPML), which
allows non-commercial use only: https://coqui.ai/cpml
"""
import os
import re
import time

if __package__:
    from . import tts_server as server
else:  # uv run --script: siblings are on sys.path, the app need not be installed.
    import tts_server as server

MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"
SAMPLE_RATE = 24000
# Conditioning latents kept in memory, most recently prepared last.
MAX_CACHED_VOICES = 4
LANGUAGES = ("en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs",
             "ar", "zh-cn", "hu", "ko", "ja", "hi")
# With a final "." (also "!", ";", ":" or "…") XTTS appends ~0.6 s of noise
# after a short silence; without it the sentence ends cleanly. "?" changes the
# intonation, so it stays.
_TRAILING_PUNCTUATION = re.compile(r"[\s.!;:…]+$")
# Trailing-artifact trim: 10 ms frames whose RMS is below SILENCE_RMS count
# as silence (pauses between words mostly stay above it). Measured with
# debora_v2: the gap before the artifact lasts 170-430 ms, but stop consonants
# inside the last words ("repita", "certo") also fall silent for 70-100 ms,
# and an 80 ms minimum cut those words off. 150 ms errs on keeping speech:
# dropping the final punctuation already prevents the artifact.
FRAME_SECONDS = 0.01
SILENCE_RMS = 0.002
MIN_SILENCE_SECONDS = 0.15
TAIL_SECONDS = 1.0
# A pause mark among the last words ("Pode ser, sim") makes XTTS pause there
# for 230-400 ms before the final word(s). That pause and the word after it
# cannot be told from the artifact by sound: over 72 measured clips the gap,
# the loudness against the speech before it, the length and the spectral
# centroid of the sound after it all overlapped. Such texts are never trimmed.
_PAUSE_MARKS = re.compile(r"[,;:.!?…—–-]")
# Sampling temperature. The model's default (0.75) sometimes renders a phrase
# with an echo; a listening test with debora_v2 preferred 0.45 (no echo, still
# natural) over 0.6 and 0.75.
TEMPERATURE = 0.45
LICENSE_NOTICE = ("XTTS-v2 weights are under the Coqui Public Model License (non-commercial "
                  "use only, https://coqui.ai/cpml); choosing this backend accepts it "
                  "(COQUI_TOS_AGREED=1).")


def xtts_language(language_id):
    """XTTS's code for a config/request language, or ValueError."""
    code = str(language_id).strip().lower().replace("_", "-")
    if code in ("zh", "zh-cn", "zh-hans"):
        return "zh-cn"
    if code in LANGUAGES:
        return code
    base = code.split("-")[0]
    if base in LANGUAGES:
        return base
    raise ValueError(f"XTTS-v2 does not speak language {language_id!r}; "
                     f"use one of {', '.join(LANGUAGES)}")


def speakable_text(text):
    """text without the final punctuation that makes XTTS add noise, or
    ValueError if nothing is left to speak."""
    stripped = _TRAILING_PUNCTUATION.sub("", text.strip())
    if not stripped:
        raise ValueError(f"nothing to speak in {text[:80]!r}")
    return stripped


def pauses_before_last_word(text):
    """True if the last three words carry a pause mark (ignoring a final "?"
    or "!"), so a silence before the final words is speech, not the artifact.
    Three: the 1 s window fits two short words after the pause ("sim, mas
    não" paused 280-360 ms before "mas não")."""
    words = text.rstrip(" ?!").split()[-3:]
    return bool(_PAUSE_MARKS.search(" ".join(words)))


def trim_trailing_artifact(samples, sample_rate=SAMPLE_RATE):
    """Cut the noise XTTS can still append after the sentence: in the last
    TAIL_SECONDS, a silence of at least MIN_SILENCE_SECONDS followed by more
    sound ends the audio at that silence. Trailing silence alone is kept."""
    import numpy as np
    samples = np.asarray(samples)
    frame = max(1, int(sample_rate * FRAME_SECONDS))
    start = max(0, len(samples) - int(sample_rate * TAIL_SECONDS))
    count = (len(samples) - start) // frame
    if count == 0:
        return samples
    tail = samples[start:start + count * frame].astype(np.float64).reshape(count, frame)
    silent = np.sqrt(np.mean(tail ** 2, axis=1)) < SILENCE_RMS
    min_frames = int(round(MIN_SILENCE_SECONDS / FRAME_SECONDS))
    cut = None
    i = 0
    while i < count:
        if not silent[i]:
            i += 1
            continue
        j = i
        while j < count and silent[j]:
            j += 1
        # Keep the latest qualifying gap: it removes the least audio.
        if j - i >= min_frames and j < count:
            cut = i
        i = j
    return samples if cut is None else samples[:start + cut * frame]


def report_coqui_downloads():
    """Coqui downloads XTTS with requests, not huggingface_hub: give its
    progress bars HF's tqdm class (with byte units) so the shared download
    tracker in tts_server reports them in /health like the other backends."""
    from huggingface_hub.utils import tqdm as hub_tqdm
    from TTS.utils import manage

    def byte_bar(*args, total=None, unit=None, unit_scale=False, **kwargs):
        return hub_tqdm(*args, total=total, unit="B", unit_scale=True, **kwargs)

    manage.tqdm = byte_bar
    return manage.ModelManager(progress_bar=True)


class XttsBackend:
    package = "coqui-tts"
    model_id = MODEL_ID
    sr = SAMPLE_RATE

    def __init__(self, device, voice):
        if not voice:
            raise ValueError("XTTS-v2 clones a reference voice: set tts_voice / --voice "
                             "to a WAV file, then restart the TTS server.")
        self.startup_voice = voice
        self.cache = {}  # (path, mtime_ns) -> (gpt_cond_latent, speaker_embedding)
        self.use(voice)  # Fail before importing/loading the model if the file is missing.
        # The user picked this backend, which accepts XTTS-v2's non-commercial
        # CPML; without this Coqui asks on stdin, which this server never reads.
        server.log(LICENSE_NOTICE)
        os.environ["COQUI_TOS_AGREED"] = "1"
        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        if device == "cuda" and not torch.cuda.is_available():
            server.log("CUDA not available; using the CPU (much slower).")
            device = "cpu"
        path = report_coqui_downloads().download_model(MODEL_ID)[0]
        config = XttsConfig()
        config.load_json(os.path.join(path, "config.json"))
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=str(path), use_deepspeed=False)
        if device == "cuda":
            self.model.cuda()
        self._debora_device = device.upper()
        self.precision = "float32"
        self.cancel_modules = [module for module in self.model.modules()
                               if not list(module.children())]
        with torch.inference_mode():
            self.prepare_voice()

    def use(self, voice):
        voice = self.startup_voice if voice is None else voice
        if not voice:
            raise ValueError("XTTS-v2 has no voice of its own; pick a reference WAV in tts_voice")
        server.check_voice(voice)
        path = os.path.abspath(voice)
        self.voice = (path, os.stat(path).st_mtime_ns)

    def prepare_voice(self):
        key = self.voice
        latents = self.cache.pop(key, None)
        if latents is None:
            start = time.time()
            latents = self.model.get_conditioning_latents(audio_path=[key[0]])
            server.log(f"Voice {key[0]} prepared in {time.time() - start:.1f}s")
            if len(self.cache) >= MAX_CACHED_VOICES:
                self.cache.pop(next(iter(self.cache)))
        self.cache[key] = latents  # Reinsert: the least recently used goes first.
        self.latents = latents

    def _inference_args(self, text, language_id):
        gpt_cond_latent, speaker_embedding = self.latents
        spoken = speakable_text(text)
        return (spoken, xtts_language(language_id), gpt_cond_latent, speaker_embedding)

    def _trim_tail(self, raw, spoken):
        samples = raw if pauses_before_last_word(spoken) else trim_trailing_artifact(raw)
        if len(samples) < len(raw):
            server.log(f"Trimmed {(len(raw) - len(samples)) / self.sr:.2f}s of trailing noise")
        return samples

    def _infer(self, method, args):
        return method(*args, temperature=TEMPERATURE, enable_text_splitting=False)

    def generate(self, text, language_id):
        import torch
        args = self._inference_args(text, language_id)
        out = self._infer(self.model.inference, args)
        samples = self._trim_tail(out["wav"].reshape(-1), args[0])
        return torch.as_tensor(samples).reshape(1, -1)

    def generate_stream(self, text, language_id):
        import numpy as np
        args = self._inference_args(text, language_id)
        tail = np.empty(0, dtype=np.float32)
        retain = int(self.sr * TAIL_SECONDS)
        # coqui-tts 0.27.5 defaults: stream_chunk_size=20, overlap_wav_len=1024.
        chunks = self._infer(self.model.inference_stream, args)
        try:
            for chunk in chunks:
                if hasattr(chunk, "cpu"):
                    chunk = chunk.float().cpu().numpy()
                tail = np.concatenate((tail, np.asarray(chunk).reshape(-1)))
                if len(tail) > retain:
                    yield tail[:-retain]
                    tail = tail[-retain:]
            tail = self._trim_tail(tail, args[0])
            if len(tail):
                yield tail
        finally:
            if hasattr(chunks, "close"):
                chunks.close()


if __name__ == "__main__":
    server.backend_main(XttsBackend, XttsBackend.package, MODEL_ID, notice=LICENSE_NOTICE)
