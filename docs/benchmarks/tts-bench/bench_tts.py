# /// script
# requires-python = "==3.12.*"
# dependencies = []
# ///
"""Benchmark a running /tts server using only the Python standard library."""
import argparse
import csv
import http.client
import io
import json
import math
from pathlib import Path
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave

MAX_AUDIO_BYTES = 50_000_000
MAX_HEALTH_BYTES = 64_000
MAX_ERROR_BYTES = 8_000
READ_CHUNK_BYTES = 64 * 1024


def read_bounded(response, limit, deadline):
    """Read at most limit + 1 bytes, including from a continuously sending peer."""
    sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
    timer = None
    aborted = threading.Event()
    if sock is not None:
        # Chunked HTTP framing may read multiple times inside read1(). Abort
        # the socket at the deadline even if those reads keep receiving data.
        def abort():
            aborted.set()
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        timer = threading.Timer(max(0, deadline - time.monotonic()), abort)
        timer.daemon = True
        timer.start()
    try:
        return _read_chunks(response, limit, deadline)
    except (OSError, http.client.HTTPException) as exc:
        # Either way the deadline ended the read: the clock can read just short
        # of it when the socket timeout or the abort timer fires (coarse on Windows).
        if aborted.is_set() or isinstance(exc, TimeoutError) or time.monotonic() >= deadline:
            raise TimeoutError("Response read deadline exceeded") from exc
        raise
    finally:
        if timer is not None:
            timer.cancel()


def _read_chunks(response, limit, deadline):
    data = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Response read deadline exceeded")
        # urllib's HTTPResponse wraps a buffered socket. Tighten each blocking
        # read to the remaining budget; read1 returns after one raw read, so a
        # trickling peer cannot keep a large read() alive indefinitely.
        sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
        if sock is not None:
            sock.settimeout(remaining)
        chunk = response.read1(min(READ_CHUNK_BYTES, limit + 1 - len(data)))
        if time.monotonic() >= deadline:
            raise TimeoutError("Response read deadline exceeded")
        data.extend(chunk)
        if len(data) > limit:
            raise ValueError(f"Response exceeds {limit} bytes")
        if not chunk:
            return bytes(data)


def fetch(opener, request, timeout, limit):
    deadline = time.monotonic() + timeout
    try:
        with opener.open(request, timeout=timeout) as response:
            return read_bounded(response, limit, deadline)
    except urllib.error.HTTPError as exc:
        with exc:
            try:
                detail = read_bounded(exc.fp, MAX_ERROR_BYTES, deadline).decode("utf-8", errors="replace")
            except (OSError, ValueError) as error:
                detail = str(error)
        raise ValueError(f"HTTP {exc.code}: {detail}") from exc


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def output_dir(path):
    target = Path(path).resolve() if path else Path(tempfile.gettempdir()) / "debora-tts-bench"
    target = target.resolve()
    repo = Path(__file__).resolve().parents[3]
    if target.is_relative_to(repo) or any((p / ".git").exists() for p in (target, *target.parents)):
        raise ValueError("--out must be outside a repository; use $env:TEMP\\debora-tts-bench")
    target.mkdir(parents=True, exist_ok=True)
    # A separate run directory avoids overwriting another backend's listening samples.
    return Path(tempfile.mkdtemp(prefix="run-", dir=target))


def synthesize(opener, url, text, voice, timeout):
    body = {"text": text, "language": "pt"}
    if voice:
        body["voice"] = str(Path(voice).resolve())
    request = urllib.request.Request(url.rstrip("/") + "/tts",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    wav = fetch(opener, request, timeout, MAX_AUDIO_BYTES)
    elapsed = time.perf_counter() - start
    with wave.open(io.BytesIO(wav), "rb") as audio:
        seconds = audio.getnframes() / audio.getframerate()
    if seconds <= 0:
        raise ValueError("Server returned an empty WAV")
    return wav, {"response_seconds": elapsed, "audio_seconds": seconds, "rtf": elapsed / seconds}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--sentences", type=Path, default=Path(__file__).with_name("frases_ptbr.txt"))
    parser.add_argument("--voice", help="Optional reference WAV; omitted uses the server's startup voice")
    parser.add_argument("--out", help="WAV/CSV/JSON output directory, outside any repository (default: TEMP)")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured requests before the list (default: 1)")
    args = parser.parse_args(argv)
    sentences = [line.strip() for line in args.sentences.read_text(encoding="utf-8-sig").splitlines()
                 if line.strip() and not line.startswith("#")]
    if not sentences or any(len(text) > 1000 for text in sentences):
        parser.error("Provide non-empty sentences of at most 1000 characters each")
    if args.warmup < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--warmup must be non-negative and --timeout finite and positive")
    try:
        out = output_dir(args.out)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        health = json.loads(fetch(opener, args.url.rstrip("/") + "/health",
                                  args.timeout, MAX_HEALTH_BYTES))
        if not health.get("ok"):
            raise ValueError(f"Server not ready: {health}")
        for _ in range(args.warmup):
            synthesize(opener, args.url, sentences[0], args.voice, args.timeout)
        rows = []
        for index, text in enumerate(sentences, 1):
            wav, metrics = synthesize(opener, args.url, text, args.voice, args.timeout)
            filename = f"{index:02d}.wav"
            (out / filename).write_bytes(wav)
            rows.append({"file": filename, "text": text, **metrics})
            print(f"{index:02d}: response={metrics['response_seconds']:.3f}s "
                  f"audio={metrics['audio_seconds']:.3f}s RTF={metrics['rtf']:.3f}")
        summary = {"health": health, "count": len(rows), "warmup": args.warmup,
                   "url": args.url, "voice": args.voice, "metrics": {}}
        for metric in ("response_seconds", "audio_seconds", "rtf"):
            values = [row[metric] for row in rows]
            stats = {"p50": percentile(values, 0.5), "p90": percentile(values, 0.9)}
            summary["metrics"][metric] = stats
            print(f"{metric}: p50={stats['p50']:.3f} p90={stats['p90']:.3f}")
        with (out / "sentences.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Output: {out}")
    except (OSError, ValueError, wave.Error, http.client.HTTPException) as exc:
        parser.exit(1, f"Benchmark failed: {exc}\n")


if __name__ == "__main__":
    main()
