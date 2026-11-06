# Benchmark TTS (pt-BR)

Stage 2 is implemented, **benchmark pending**. No latency improvement is claimed yet.
Run from the repository root in PowerShell, with the app stopped. Start **one**
server in terminal A; wait for `Listening` and check `/health`. First startup
creates an isolated uv environment and downloads model weights. These commands
have not been run as part of implementation.

```powershell
# Chatterbox (baseline)
uv run --script debora_whisper/tts_server.py --device cuda --voice debora_whisper/voices/debora_v2.wav
# Or Qwen3-TTS 0.6B Base (reads debora_v2.txt beside the WAV)
uv run --script debora_whisper/tts_server_qwen3.py --device cuda --voice debora_whisper/voices/debora_v2.wav
# Or Kokoro, fixed pf_dora / lang_code p, no cloning
uv run --script debora_whisper/tts_server_kokoro.py --device cpu
# Repeat Kokoro on the GPU in a separate run
uv run --script debora_whisper/tts_server_kokoro.py --device cuda
```

Kokoro's pt-BR phonemizer, espeak-ng, comes bundled with the script's
environment through `espeakng-loader`, so no system install is needed. If
`/health` reports that it failed to load, check whether Windows blocked the DLL
(Smart App Control) and rebuild the script environment.

Qwen requires the exact reference transcript in UTF-8 `<voice>.txt`, next to
`<voice>.wav`. `debora_whisper/voices/debora_v2.txt` ships with the app;
other voices (including user overrides of `debora_v2.wav`) need their own sidecar.
Missing/empty transcripts and an espeak-ng load failure return an actionable HTTP 503
from `/health`; correct the problem and restart the server. No transcript is guessed.

In terminal B, for each selected backend:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
uv run --script docs/benchmarks/tts-bench/bench_tts.py --out "$env:TEMP\debora-tts-bench"
# Optional GPU monitoring in another terminal; record idle and peak VRAM separately
nvidia-smi --query-gpu=memory.used,clocks.sm,temperature.gpu --format=csv -l 1
```

The client uses only the standard library (PEP 723, no dependencies or indexes).
It sends the 20 shipped sentences sequentially, after one unmeasured warmup;
`--warmup 0` includes the first cold request. `--sentences`, `--url`, `--voice`
(reference WAV) and `--timeout` can be overridden. The server uses the startup
voice when `--voice` is omitted. Keep sentences/voice/device/warmup identical
when comparing runs and record competing STT/LLM activity.

Each sentence reports full HTTP **time to response**, WAV duration and RTF
(response seconds / audio seconds); p50/p90 use linear interpolation. This is
one complete WAV per request, **not time to a [streaming chunk](../../TTS-PROTOCOL.md)**, nor voice-to-voice
latency. Startup/model loading is excluded. The tool writes numbered WAVs,
`sentences.csv` and `summary.json` (including server inventory) into a unique run
directory under `--out`, defaulting to the system temporary directory. Output
inside repositories is rejected. Never commit generated audio; publish only
measurements. Listen to at least ten matched phrases per backend for pt-BR quality.

Pins/APIs checked against [qwen-tts 0.1.1](https://pypi.org/project/qwen-tts/0.1.1/),
[Qwen source](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py),
[Kokoro 0.9.4](https://pypi.org/project/kokoro/0.9.4/),
[Kokoro pipeline](https://github.com/hexgrad/kokoro/blob/main/kokoro/pipeline.py),
[Misaki 0.9.4](https://pypi.org/project/misaki/0.9.4/) and
[phonemizer-fork 3.3.2](https://pypi.org/project/phonemizer-fork/3.3.2/).
Server headers pin direct dependencies exactly and scope the CUDA 12.4 index
to torch/torchaudio. Qwen uses SDPA rather than requiring FlashAttention on Windows.
Dependency resolution and real GPU execution remain to be validated on this machine.
