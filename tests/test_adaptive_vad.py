"""Adaptive endpoint behavior without a microphone, model or wall-clock sleeps."""
import queue

import numpy as np
import pytest

from debora_whisper import dictation_engine as de
from debora_whisper.vad_endpoint import AdaptiveEndpoint, VadSegment, incomplete_sentence


@pytest.mark.parametrize("text, expected", [
    ("Minha ideia é", True), ("Então,", True), ("Exemplo:", True),
    ("Primeiro;", True), ("Ainda...", True), ("Ainda…", True),
    ("Pronto.", False), ("Tudo bem?", False), ("Sim!", False),
    ('Ele disse “pronto.”  ', False), ("(Ainda...)", True),
    ("", None), ("  ", None), ("...", None),
])
def test_raw_punctuation(text, expected):
    assert incomplete_sentence(text) is expected


def test_late_results_cannot_extend_a_closed_or_new_segment():
    endpoint = AdaptiveEndpoint()
    old = endpoint.start()
    endpoint.finish()
    endpoint.update(old, 100, "Ainda")
    assert not endpoint.incomplete
    current = endpoint.start()
    endpoint.update(old, 200, "Ainda")
    assert not endpoint.incomplete
    endpoint.update(current, 100, "Agora")
    assert endpoint.incomplete


def test_resumed_speech_invalidates_feedback_until_draft_covers_it():
    endpoint = AdaptiveEndpoint()
    segment = endpoint.start()
    endpoint.speech(100)
    endpoint.update(segment, 110, "Minha ideia")
    assert endpoint.incomplete
    endpoint.speech(200)
    assert not endpoint.incomplete
    endpoint.update(segment, 150, "Minha ideia é")
    assert not endpoint.incomplete
    endpoint.update(segment, 210, "Minha ideia é")
    assert endpoint.incomplete
    endpoint.update(segment, 205, "Pronto.")
    assert endpoint.incomplete
    endpoint.update(segment, 220, "Minha ideia é essa.")
    assert not endpoint.incomplete


def test_empty_latest_draft_returns_to_normal_wait():
    endpoint = AdaptiveEndpoint()
    segment = endpoint.start()
    endpoint.update(segment, 100, "Ainda")
    endpoint.update(segment, 200, "")
    assert not endpoint.incomplete


def run_vad(monkeypatch, phases, draft_text="Minha ideia é", stop_at=None, **config):
    """Feed 32 ms blocks through the actual VAD loop with immediate ASR feedback."""
    rec = de.AudioRecorder(config={"continuous_listening": True, **config})
    blocks = [speech for speech, count in phases for _ in range(count)]
    tick = [0]
    finals = []

    class Feed:
        def wait(self, timeout):
            if tick[0] == len(blocks):
                rec._stop_vad = True
                return
            speech = blocks[tick[0]]
            start = rec._write_pos
            end = start + 512
            rec._buffer[start:end] = 0.1 if speech else 0.0
            rec._write_pos = end % rec.capacity
            tick[0] += 1
            if tick[0] == stop_at:
                # Same state change as end_continuous, already inside _lock.
                rec.paused = True
                rec.continuous = False
                rec.endpoint.finish()

    class Consumer(queue.Queue):
        def put(self, segment):
            assert isinstance(segment, VadSegment)
            if segment.is_final:
                finals.append((tick[0], segment))
            elif draft_text is not None:
                rec.endpoint.update(segment.segment_id, segment.audio_end, draft_text)

    rec._data_cv = Feed()
    rec.segment_queue = Consumer()
    monkeypatch.setattr(de.time, "time", lambda: 1000 + tick[0] * 0.032)
    rec._vad_loop()
    assert not rec.endpoint.incomplete
    return finals


@pytest.mark.parametrize("text, silence_blocks", [
    ("Pronto.", 47), (None, 47), ("Minha ideia é", 94),
    ("Continuando...", 94),
])
def test_actual_silence_deadline(monkeypatch, text, silence_blocks):
    # 38 speech blocks = 1.216s. Repeated drafts during silence must not
    # restart the timeout; finalization stays tied to the last speech block.
    finals = run_vad(monkeypatch, [(True, 38), (False, 110)], text)
    assert len(finals) == 1
    assert finals[0][0] == 38 + silence_blocks


def test_two_second_pause_keeps_continuation_in_same_segment(monkeypatch):
    finals = run_vad(monkeypatch, [
        (True, 38), (False, 63), (True, 38), (False, 110),
    ])
    assert len(finals) == 1
    assert finals[0][0] == 38 + 63 + 38 + 94
    audio = finals[0][1].audio
    assert np.count_nonzero(audio) == 76 * 512


def test_manual_stop_cuts_without_waiting_for_extended_silence(monkeypatch):
    finals = run_vad(monkeypatch, [(True, 38), (False, 110)], stop_at=100)
    assert len(finals) == 1
    assert finals[0][0] == 100


def test_duration_limit_still_cuts_incomplete_segment(monkeypatch):
    finals = run_vad(monkeypatch, [(True, 38), (False, 110)], segment_max_seconds=3)
    assert len(finals) == 1
    # Existing segment limit includes the 0.5s lookback.
    assert finals[0][0] == 79


def test_voice_chat_keeps_its_own_timeout(monkeypatch):
    finals = run_vad(monkeypatch, [(True, 38), (False, 110)], "Pronto.", voice_chat=True)
    assert finals[0][0] == 38 + 26


def test_timeouts_are_configurable_and_extension_never_shortens_base(monkeypatch):
    finals = run_vad(monkeypatch, [(True, 38), (False, 110)],
                     vad_end_silence_seconds=1.1, vad_incomplete_silence_seconds=2)
    assert finals[0][0] == 38 + 63
    finals = run_vad(monkeypatch, [(True, 38), (False, 110)],
                     vad_end_silence_seconds=2, vad_incomplete_silence_seconds=1)
    assert finals[0][0] == 38 + 63


def test_short_timeout_does_not_wait_for_missing_fresh_draft(monkeypatch):
    finals = run_vad(monkeypatch, [(True, 38), (False, 110)],
                     vad_end_silence_seconds=0.7, vad_incomplete_silence_seconds=2)
    assert finals[0][0] == 38 + 22


@pytest.mark.parametrize("draft", ["sobre o projeto...", "sobre o projeto…", "Eu queria, mas."])
def test_incomplete_voice_chat_draft_extends_silence(monkeypatch, draft):
    finals = run_vad(monkeypatch, [(True, 38), (False, 110)], draft, voice_chat=True)
    assert finals[0][0] == 38 + 63  # 2 s, shorter than dictation's 3 s


def test_voice_chat_requires_actual_minimum_speech(monkeypatch):
    finals = run_vad(monkeypatch, [(True, 3), (False, 110)], voice_chat=True)
    assert finals == []  # lookback and silence must not count as voiced frames
