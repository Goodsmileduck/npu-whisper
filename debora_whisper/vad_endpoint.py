"""Punctuation feedback for continuous dictation's silence timeout."""
import threading
import re
from typing import Any, NamedTuple


class VadSegment(NamedTuple):
    audio: Any
    is_final: bool
    segment_id: int
    audio_end: int  # Frames from the beginning of this segment, including silence.
    captured_at: float | None = None  # End of capture, before ASR/queue delays.


def incomplete_sentence(text: str) -> bool | None:
    """Classify raw ASR text, before display ellipses are appended."""
    text = text.rstrip().rstrip('\"\'”’»)]}').rstrip()
    if not text or not any(char.isalnum() for char in text):
        return None
    if text.endswith(("...", "…")):
        return True
    words = re.findall(r"\w+", text.casefold())
    if words and words[-1] in {"e", "mas", "porque", "que", "se", "ou", "então",
                              "and", "but", "because", "or", "if", "so"}:
        return True
    return not text.endswith((".", "?", "!"))


class AdaptiveEndpoint:
    """Share draft feedback between the inference worker and VAD thread.

    Only a result covering the latest speech of the active segment can
    change its timeout. Audio positions are relative, never ring indices.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._generation = 0
        self._active = None
        self._speech_end = 0
        self._result_end = -1
        self._incomplete = None

    def start(self) -> int:
        with self._lock:
            self._generation += 1
            self._active = self._generation
            self._speech_end = 0
            self._result_end = -1
            self._incomplete = None
            return self._active

    def finish(self):
        with self._lock:
            self._active = None
            self._incomplete = None

    def speech(self, audio_end: int):
        with self._lock:
            self._speech_end = audio_end
            if audio_end > self._result_end:
                self._incomplete = None

    def update(self, segment_id: int, audio_end: int, text: str):
        incomplete = incomplete_sentence(text)
        with self._lock:
            if (segment_id != self._active or self._active is None
                    or audio_end < self._speech_end
                    or audio_end <= self._result_end):
                return
            self._result_end = audio_end
            self._incomplete = incomplete

    @property
    def incomplete(self) -> bool:
        with self._lock:
            return self._incomplete is True
