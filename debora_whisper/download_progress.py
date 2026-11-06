"""Byte progress shared by telemetry and the standalone TTS scripts."""
from contextlib import contextmanager
from functools import wraps
import math
import threading


class DownloadProgress:
    """Aggregate discovered files, retaining completed bytes until reset."""

    def __init__(self):
        self._lock = threading.RLock()
        self._files = {}

    def update(self, key, done, total, *, closed=False):
        with self._lock:
            self._files[key] = (min(max(done, 0), total), total, closed)
            return self._totals()

    def _totals(self):
        return {"done": sum(f[0] for f in self._files.values()),
                "total": sum(f[1] for f in self._files.values())}

    def _snapshot(self):
        if not any(not closed and done < total for done, total, closed in self._files.values()):
            return None
        return self._totals()

    def snapshot(self):
        with self._lock:
            return self._snapshot()

    def clear(self):
        with self._lock:
            self._files.clear()


def format_download(label, progress):
    if not isinstance(progress, dict):
        return ""
    done, total = progress.get("done"), progress.get("total")
    # Bound before converting integers to floats (isfinite can itself overflow).
    if (type(done) not in (int, float) or type(total) not in (int, float)
            or not 0 <= done < total <= 2**63 - 1
            or not math.isfinite(done) or not math.isfinite(total)):
        return ""
    return f" DL {label} {min(99, round((done / total) * 100))}% ({done / 1_000_000:.0f}/{total / 1_000_000:.0f} MB)"


STT_DOWNLOAD = DownloadProgress()
_HF_DOWNLOAD_LOCK = threading.Lock()


@contextmanager
def track_hf_downloads(tracker, callback=None):
    """Observe HF byte bars, including HTTP/Xet and resumed downloads.

    Patch methods on HF's shared tqdm class so aliases already imported by
    download helpers are covered too. File-count and model-loading bars are
    ignored. Counters remain valid when tqdm display is disabled.

    Serialize contexts for their entire lifetime: these hooks are process-wide,
    and another download must not capture or restore an active context's hooks.
    """
    with _HF_DOWNLOAD_LOCK:
        with _observe_hf_downloads(tracker, callback):
            yield


@contextmanager
def _observe_hf_downloads(tracker, callback):
    from huggingface_hub.utils import tqdm
    from huggingface_hub import file_download

    original_init, original_update, original_close = tqdm.__init__, tqdm.update, tqdm.close
    original_http_get = file_download.http_get
    bars = {}
    current_bars = {}
    transfer = threading.local()
    lock = threading.RLock()
    tracker.clear()

    def publish(bar, amount=0, closed=False):
        with lock:
            entry = bars.get(id(bar))
            if entry is None or current_bars.get(entry[0]) != id(bar):
                return
            entry[1] += amount
            # Snapshot aggregate bars start at zero and grow as files are found.
            entry[2] = bar.total
            progress = tracker.update(*entry, closed=closed)
            if callback and not closed:
                callback(progress["done"], progress["total"])

    def init(bar, *args, **kwargs):
        original_init(bar, *args, **kwargs)
        total = kwargs.get("total")
        # Hub 2.x reports transfer and reconstruction for the same files. Count
        # reconstruction once (including resumed bytes), not both stages.
        if (kwargs.get("unit") == "B" and total is not None and total >= 0
                and kwargs.get("name") != "huggingface_hub.snapshot_download.transfer"):
            with lock:
                # Retain the bar to prevent object-id reuse between files.
                key = getattr(transfer, "key", None) or ("bar", id(bar))
                bars[id(bar)] = [key, kwargs.get("initial", 0), total]
                current_bars[key] = id(bar)
                retained.append(bar)
                publish(bar)

    @wraps(original_http_get)
    def http_get(url, temp_file, *args, **kwargs):
        # The destination survives recursive retries, unlike the bar or its
        # truncated display name. Separate files with identical names stay separate.
        previous = getattr(transfer, "key", None)
        transfer.key = ("http", temp_file)
        try:
            return original_http_get(url, temp_file, *args, **kwargs)
        finally:
            transfer.key = previous

    def update(bar, n=1):
        result = original_update(bar, n)
        publish(bar, n)
        return result

    def close(bar):
        publish(bar, closed=True)
        return original_close(bar)

    retained = []
    tqdm.__init__, tqdm.update, tqdm.close = init, update, close
    file_download.http_get = http_get
    try:
        yield
    finally:
        tqdm.__init__, tqdm.update, tqdm.close = original_init, original_update, original_close
        file_download.http_get = original_http_get
        tracker.clear()
