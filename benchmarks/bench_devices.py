"""Turbo latency on one device, through the app's own create_model.

Usage (app venv, app closed, one process per device):
    python benchmarks/bench_devices.py <speech.wav> <dense.wav> <NPU|GPU|CPU|CUDA> [beam]

speech.wav: ~26s of 16 kHz mono speech; dense.wav: >=30s of fast, dense
speech. beam (CUDA only): "app" = what the app sends, or an int to force
faster-whisper's beam_size. Prints one JSON line per clip; latency is the
median of BENCH_REPS (default 5) calls after one warm-up call. cpu_s is the
process CPU time per call.
"""
import json, os, statistics, sys, time, wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from debora_whisper import dictation_engine as de

speech_wav, dense_wav, device = sys.argv[1:4]
beam = sys.argv[4] if len(sys.argv) > 4 else "app"
REPS = int(os.environ.get("BENCH_REPS", "5"))


def load(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


speech, dense = load(speech_wav), load(dense_wav)
clips = [("2s", speech[:2 * 16000]), ("5s", speech[:5 * 16000]),
         ("10s", speech[:10 * 16000]), ("20s", speech[:20 * 16000]),
         ("26s", speech[:26 * 16000]), ("30s-dense", dense[:30 * 16000])]

info = de.MODEL_REGISTRY["turbo"]
t0 = time.perf_counter()
m = de.create_model(de.MODEL_DIR / info["local_dir"], device=device,
                    backend=info["backend"], model_size="turbo")
load_s = time.perf_counter() - t0

if device == "CUDA" and beam != "app":
    orig = m.pipeline.transcribe
    m.pipeline.transcribe = lambda *a, **k: orig(*a, beam_size=int(beam), **k)

label = device + (f"-beam{beam}" if device == "CUDA" else "")
print(json.dumps({"label": label, "active": str(m.device), "load_s": round(load_s, 2)}), flush=True)

m.transcribe(speech[:5 * 16000].copy(), 16000, "en")  # warm-up

for name, audio in clips:
    wall, cpu, text = [], [], ""
    for _ in range(REPS):
        c0, w0 = time.process_time(), time.perf_counter()
        text = m.transcribe(audio.copy(), 16000, "en")
        wall.append(time.perf_counter() - w0)
        cpu.append(time.process_time() - c0)
    med = statistics.median(wall)
    print(json.dumps({"label": label, "clip": name,
                      "median_s": round(med, 3), "max_s": round(max(wall), 3),
                      "rtf": round(med / (len(audio) / 16000), 3),
                      "cpu_s": round(statistics.median(cpu), 2),
                      "words": len(text.split())}), flush=True)
