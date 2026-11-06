"""Continuous mode must skip drafts that newer queued audio already covers."""
from debora_whisper.dictation_engine import drop_superseded_drafts


def _d(tag):
    return (tag, False)


def _f(tag):
    return (tag, True)


def test_single_draft_is_kept():
    assert drop_superseded_drafts([_d("a")]) == [_d("a")]


def test_only_newest_of_queued_drafts_is_kept():
    assert drop_superseded_drafts([_d("a"), _d("b"), _d("c")]) == [_d("c")]


def test_final_supersedes_queued_draft():
    # The sentence ended while a draft waited: go straight to the final.
    assert drop_superseded_drafts([_d("a"), _f("a-final")]) == [_f("a-final")]


def test_finals_are_never_dropped():
    items = [_f("one"), _d("two-draft"), _f("two")]
    assert drop_superseded_drafts(items) == [_f("one"), _f("two")]


def test_draft_of_next_segment_after_final_is_kept():
    items = [_f("one"), _d("two-draft")]
    assert drop_superseded_drafts(items) == items


def test_empty():
    assert drop_superseded_drafts([]) == []
