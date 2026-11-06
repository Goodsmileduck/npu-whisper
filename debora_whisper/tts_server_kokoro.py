# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "kokoro==0.9.4", "misaki[en]==0.9.4", "phonemizer-fork==3.3.2",
#     "transformers==4.57.3", "torch==2.6.0", "torchaudio==2.6.0",
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
"""Kokoro-82M, fixed Brazilian Portuguese pf_dora voice (no cloning)."""
if __package__:
    from . import tts_server as server
else:
    import tts_server as server

MODEL_ID = "hexgrad/Kokoro-82M"


def check_espeak():
    """Exercise the actual pt-BR phonemizer, including DLL/data discovery."""
    try:
        from misaki.espeak import EspeakG2P
        EspeakG2P(language="pt-br")("Olá.")
    except Exception as exc:
        raise RuntimeError(
            "Kokoro's pt-BR phonemizer (espeak-ng, bundled through espeakng-loader) "
            "failed to load. Delete the uv cache entry for this script to reinstall it, "
            "and check that Windows did not block the DLL (Smart App Control), "
            f"then restart the TTS server. Details: {exc}") from exc


class KokoroBackend:
    package = "kokoro"
    model_id = MODEL_ID
    sr = 24000
    precision = "float32"

    def __init__(self, device, voice):
        check_espeak()  # Check before loading/downloading weights.
        import torch
        from kokoro import KPipeline
        if device == "cuda" and not torch.cuda.is_available():
            server.log("CUDA not available; using the CPU.")
            device = "cpu"
        self._debora_device = device.upper()
        self.pipeline = KPipeline(lang_code="p", repo_id=MODEL_ID, device=device)
        self.pipeline.load_voice("pf_dora")
        self.cancel_modules = [module for module in self.pipeline.model.modules()
                               if not list(module.children())]

    def use(self, voice):
        pass  # HTTP compatibility: reference WAVs do not change pf_dora.

    def prepare_voice(self):
        pass

    def generate(self, text, language_id):
        import torch
        if language_id.lower() not in ("p", "pt", "pt-br", "portuguese", "auto"):
            raise ValueError("Kokoro pf_dora supports pt-BR; use language pt")
        chunks = [audio.reshape(-1) for _, _, audio in self.pipeline(text, voice="pf_dora")
                  if audio is not None]
        if not chunks:
            raise ValueError("Kokoro produced no audio")
        return torch.cat(chunks).reshape(1, -1)


if __name__ == "__main__":
    server.backend_main(KokoroBackend, KokoroBackend.package, MODEL_ID)
