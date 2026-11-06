# Benchmarks

## bench_parakeet.py

Measures the Parakeet TDT (`ParakeetNPU`) pipeline: model load/compile time, and
per-utterance latency broken down by stage (mel preprocessing / encoder / TDT
decoder), plus which shape bucket was selected. This is the harness required by
[issue #3](https://github.com/alexandre-machado/debora-whisper/issues/3).

**No numbers in this repo, this README, or the PR that introduced this script are
measured.** This environment (the one the change was authored in) has no Intel
NPU — it's WSL/Linux. Every latency and compile-time figure must come from you
running this script on the target machine (Windows 11 / Intel Core Ultra) and is
only real once you've run it and reported the output.

### Usage (Windows, from the repo root, with the project venv active)

```powershell
# Default: NPU device, synthetic test utterances at 2s/5s/9s/16s/20s
# (the last two exercise the largest bucket and the over-length/truncation path)
python benchmarks\bench_parakeet.py --device NPU

# Against your own recordings instead of synthetic audio (16kHz mono WAV):
python benchmarks\bench_parakeet.py --device NPU --audio path\to\clip1.wav path\to\clip2.wav

# Repeat each utterance N times after the first (warm) run, to average out jitter:
python benchmarks\bench_parakeet.py --device NPU --runs 5

# Turn on the untried ov::intel_npu::compilation_mode_params
# performance-hint-override="latency" lever (NPU's own default for that
# sub-property is "efficiency" — this measures whether overriding it helps):
python benchmarks\bench_parakeet.py --device NPU --latency-override

# Write results to CSV for before/after diffing:
python benchmarks\bench_parakeet.py --device NPU --csv results_after.csv
```

### Producing a before/after pair for this PR

Shape bucketing only exists on this branch; `main` still has the single
1600-frame static shape. To get a real before/after:

```powershell
# 1. On main (pre-bucketing baseline):
git checkout main
python benchmarks\bench_parakeet.py --device NPU --csv bench_before.csv
# (benchmarks/bench_parakeet.py doesn't exist on main yet — copy it over first,
#  e.g. `git show feat/parakeet-shape-bucketing:benchmarks/bench_parakeet.py > benchmarks\bench_parakeet.py`
#  since the harness itself has no bucketing dependency)

# 2. On this branch (post-bucketing):
git checkout feat/parakeet-shape-bucketing
python benchmarks\bench_parakeet.py --device NPU --csv bench_after.csv

# 3. Compare bench_before.csv vs bench_after.csv per-utterance latency_total_s,
#    and separately note load_time_s (first-run compile cost: 1 graph on main,
#    len(MEL_BUCKETS) graphs on this branch).
```

Also worth one extra run to see the `compilation_mode_params` effect in isolation:

```powershell
python benchmarks\bench_parakeet.py --device NPU --csv bench_after_latency_override.csv --latency-override
```

### What to report back

Paste the CSV (or the printed table) plus:
- the machine (CPU model, OpenVINO version: `python -c "import openvino; print(openvino.__version__)"`)
- `load_time_s` for main vs this branch (first-run, cold `CACHE_DIR`) — this is the
  "N buckets means N compiled graphs" cost the issue asks to measure
- per-utterance `latency_total_s` for main vs this branch, at minimum for a ~3s
  and a ~5s clip (the workflow this change targets)
- whether `--latency-override` changed anything, with numbers — it is not assumed
  to help
