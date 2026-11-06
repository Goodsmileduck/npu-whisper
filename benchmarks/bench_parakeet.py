#!/usr/bin/env python3
"""Parakeet TDT (ParakeetNPU) benchmark harness — issue #3.

Measures, per utterance: mel-preprocessing time, encoder time, TDT decoder
time, total end-to-end latency, and which shape bucket was selected. Also
measures one-time pipeline load/compile time.

IMPORTANT: this script produces numbers only when run on the target hardware
(Windows 11 / Intel Core Ultra with the NPU driver installed). It cannot be
run in this repo's development sandbox (WSL/Linux, no NPU) — any number in
the PR that introduced this file is either produced by you running this
script, or explicitly labelled "not measured".

Usage:
    python benchmarks/bench_parakeet.py --device NPU
    python benchmarks/bench_parakeet.py --device NPU --audio a.wav b.wav
    python benchmarks/bench_parakeet.py --device NPU --latency-override
    python benchmarks/bench_parakeet.py --device NPU --csv out.csv

See benchmarks/README.md for the full before/after workflow.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from debora_whisper.dictation_engine import ParakeetNPU, MODEL_DIR, MODEL_REGISTRY, is_model_downloaded


def synthetic_utterance(duration_s: float, sample_rate: int = 16000, seed: int = 0) -> np.ndarray:
    """Generate deterministic pseudo-speech-like noise for latency testing.

    Not real speech — fine for measuring shape/compile/inference latency,
    NOT fine for measuring transcription accuracy. Use --audio with real
    WAV files if you want to sanity-check output text alongside latency.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate)
    # Band-limited-ish noise so the mel preprocessor sees non-trivial energy
    # across frames, rather than pure silence (which some VADs/pipelines
    # special-case).
    return (rng.standard_normal(n).astype(np.float32) * 0.05)


def load_wav(path: Path, sample_rate: int = 16000) -> np.ndarray:
    import wave
    with wave.open(str(path), "rb") as wf:
        if wf.getframerate() != sample_rate:
            print(f"WARNING: {path} is {wf.getframerate()}Hz, expected {sample_rate}Hz "
                  f"— results may not reflect real usage.")
        raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if wf.getnchannels() > 1:
            audio = audio.reshape(-1, wf.getnchannels()).mean(axis=1)
        return audio


def _csv_safe(value: str) -> str:
    """Prefix values that would be interpreted as formulas if this CSV is
    opened in Excel/Sheets (transcript text starting with =, +, -, or @).

    benchmarks/README.md tells the user to paste this CSV into a
    spreadsheet, and the `text` field is untrusted ASR output (from
    synthetic audio or a user-supplied --audio WAV), so this guards against
    CSV/formula injection there.
    """
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@"):
        return "'" + value
    return value


def bench_one(model: ParakeetNPU, audio: np.ndarray, sample_rate: int, label: str) -> dict:
    t0 = time.time()
    mel, _ = model._preprocess(audio)
    t1 = time.time()
    actual_frames = mel.shape[2]

    bucket, was_truncated = model.select_bucket(actual_frames)

    mel = model.pad_to_bucket(mel, bucket)
    if was_truncated:
        actual_frames = bucket

    t2 = time.time()
    enc_result = model.enc_compiled[bucket]({
        "audio_signal": mel,
        "length": np.array([actual_frames], dtype=np.int64),
    })
    enc_out = enc_result["outputs"]
    enc_len = int(enc_result["encoded_lengths"][0])
    t3 = time.time()

    text = model._tdt_greedy_decode(enc_out, enc_len)
    t4 = time.time()

    return {
        "label": label,
        "audio_duration_s": round(len(audio) / sample_rate, 2),
        "mel_frames": actual_frames,
        "bucket_frames": bucket,
        "was_truncated": was_truncated,
        "mel_preprocess_s": round(t1 - t0, 4),
        "encoder_s": round(t3 - t2, 4),
        "decoder_s": round(t4 - t3, 4),
        "latency_total_s": round(t4 - t0, 4),
        "rtf": round((t4 - t0) / (len(audio) / sample_rate), 4) if len(audio) else 0,
        "text": text,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="NPU", choices=["NPU", "GPU", "CPU"])
    ap.add_argument("--model-dir", type=Path, default=None,
                     help="Path to parakeet model dir (default: MODEL_DIR/parakeet-tdt-openvino)")
    ap.add_argument("--audio", nargs="*", type=Path, default=None,
                     help="WAV files (16kHz mono) to benchmark. Default: synthetic utterances.")
    ap.add_argument("--durations", type=str, default="2,5,9,16,20",
                     help="Synthetic utterance durations in seconds (used when --audio is not given). "
                          "Includes an over-length case (>largest bucket) by default.")
    ap.add_argument("--runs", type=int, default=1,
                     help="Extra warm repeats per utterance after the first, to average out jitter.")
    ap.add_argument("--latency-override", action="store_true",
                     help="Set PARAKEET_LATENCY_OVERRIDE=1 (ov::intel_npu::compilation_mode_params "
                          "performance-hint-override=latency). UNMEASURED lever — this flag lets you "
                          "measure it, it does not claim it helps.")
    ap.add_argument("--csv", type=Path, default=None, help="Write results to this CSV path.")
    args = ap.parse_args()

    if args.latency_override:
        import os
        os.environ["PARAKEET_LATENCY_OVERRIDE"] = "1"

    model_dir = args.model_dir
    if model_dir is None:
        model_dir = MODEL_DIR / MODEL_REGISTRY["parakeet"]["local_dir"]
    if not model_dir.exists():
        print(f"ERROR: model dir not found: {model_dir}")
        print("Run: debora-cli --model parakeet --setup")
        sys.exit(1)

    print(f"Loading ParakeetNPU from {model_dir} on {args.device} "
          f"(latency_override={args.latency_override})...")
    load_start = time.time()
    model = ParakeetNPU(model_dir, device=args.device)
    load_time_s = time.time() - load_start
    print(f"Loaded in {load_time_s:.1f}s (this is the first-run compile cost: "
          f"{len(model.MEL_BUCKETS)} bucket graph(s) unless CACHE_DIR was already warm)")

    sample_rate = 16000
    utterances = []
    if args.audio:
        for p in args.audio:
            utterances.append((p.stem, load_wav(p, sample_rate)))
    else:
        for d in (float(x) for x in args.durations.split(",")):
            utterances.append((f"synthetic_{d:g}s", synthetic_utterance(d, sample_rate)))

    rows = []
    for label, audio in utterances:
        for run_idx in range(1 + args.runs):
            tag = label if run_idx == 0 else f"{label}_run{run_idx}"
            result = bench_one(model, audio, sample_rate, tag)
            rows.append(result)
            trunc = " TRUNCATED" if result["was_truncated"] else ""
            print(f"{tag:>20s}: {result['audio_duration_s']:>5.1f}s audio -> "
                  f"bucket={result['bucket_frames']:>4d}f{trunc} | "
                  f"mel={result['mel_preprocess_s']*1000:>6.1f}ms "
                  f"enc={result['encoder_s']*1000:>6.1f}ms "
                  f"dec={result['decoder_s']*1000:>6.1f}ms "
                  f"total={result['latency_total_s']*1000:>7.1f}ms "
                  f"rtf={result['rtf']:.2f}")

    print(f"\nload_time_s={load_time_s:.2f} device={model.device} "
          f"buckets={model.MEL_BUCKETS} latency_override={model.latency_override}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            fieldnames = ["label", "audio_duration_s", "mel_frames", "bucket_frames",
                          "was_truncated", "mel_preprocess_s", "encoder_s", "decoder_s",
                          "latency_total_s", "rtf", "text"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({**r, "label": _csv_safe(r["label"]), "text": _csv_safe(r["text"])})
            w.writerow({})
            w.writerow({"label": "load_time_s", "audio_duration_s": round(load_time_s, 2)})
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
