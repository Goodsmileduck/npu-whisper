"""Tests for icon generation."""

import pytest
from PIL import Image

from debora_whisper.ui.icons import (
    icon_loading, icon_ready, icon_recording, icon_processing, icon_error,
    icon_speaking, STATE_ICONS, ICON_SIZE, get_icon, get_volume_icon,
)


class TestIconGeneration:
    """Verify each icon returns a 64x64 RGBA image."""

    @pytest.mark.parametrize("fn", [
        icon_loading, icon_ready, icon_recording, icon_processing, icon_error,
        icon_speaking,
    ])
    def test_icon_size_and_mode(self, fn):
        img = fn()
        assert isinstance(img, Image.Image)
        assert img.size == (ICON_SIZE, ICON_SIZE)
        assert img.mode == "RGBA"

    def test_state_icons_dict_has_all_states(self):
        expected = {"loading", "ready", "recording", "processing", "error", "speaking"}
        assert set(STATE_ICONS.keys()) == expected

    def test_state_icons_callable(self):
        for name, fn in STATE_ICONS.items():
            img = fn()
            assert isinstance(img, Image.Image), f"STATE_ICONS['{name}'] did not return an Image"

    def test_icons_are_not_fully_transparent(self):
        """Each icon should have some non-transparent pixels."""
        for name, fn in STATE_ICONS.items():
            img = fn()
            alpha = img.split()[3]  # Alpha channel
            assert alpha.getextrema()[1] > 0, f"Icon '{name}' is fully transparent"

    def test_error_icon_differs_from_recording(self):
        """Error icon (X overlay) should differ from plain recording icon."""
        rec = icon_recording()
        err = icon_error()
        assert rec.tobytes() != err.tobytes()


def test_each_bars_glow_leaves_the_previous_bar_whole():
    """Each bar is composited on its own layer: drawn straight onto the icon,
    the next bar's glow overwrote the right edge of the one before it."""
    from debora_whisper.ui.icons import render_bars
    size = 256
    img = render_bars("#06B6D4", [1.0] * 5, size=size)
    bar_w, spacing = size * 0.12, size * 0.06
    start_x = (size - (5 * bar_w + 4 * spacing)) / 2
    for i in range(5):
        right_edge = start_x + i * (bar_w + spacing) + bar_w
        r, g, b, a = img.getpixel((int(right_edge - bar_w * 0.25), size // 2))
        assert a > 200 and g > 150, f"bar {i} cut at its right edge: {(r, g, b, a)}"


@pytest.mark.parametrize("level", [0.0, 0.02, 0.03, 0.5, 1.0])
def test_recording_colors_follow_voice_mode(monkeypatch, level):
    from debora_whisper.ui import icons
    monkeypatch.setattr(icons, "_vol_frame_counter", 0)
    green = get_volume_icon(level)
    monkeypatch.setattr(icons, "_vol_frame_counter", 0)
    purple = get_volume_icon(level, voice_chat=True)
    assert green.tobytes() != purple.tobytes()
    for image, voice_chat in [(green, False), (purple, True)]:
        pixels = [p for p in image.getdata() if p[3] > 100]
        r, g, b = [sum(p[c] for p in pixels) / len(pixels) for c in range(3)]
        assert b > r > g if voice_chat else g > b > r
    if level < 0.03:
        assert green is get_icon("recording")
        assert purple is get_icon("recording", voice_chat=True)


def test_recording_palette_and_cached_frames(monkeypatch):
    from debora_whisper.ui import icons
    assert (icons.C_RECORDING_IDLE, icons.C_RECORDING_ACTIVE) == ("#047857", "#10B981")
    assert (icons.C_VOICE_CHAT_IDLE, icons.C_VOICE_CHAT_ACTIVE) == ("#7E22CE", "#A855F7")

    def unexpected_render(*args, **kwargs):
        pytest.fail("Recording frames must be pre-rendered")

    monkeypatch.setattr(icons, "render_bars", unexpected_render)
    for voice_chat in (False, True):
        idle = get_icon("recording", voice_chat=voice_chat)
        active = [get_volume_icon(0.5, voice_chat=voice_chat) for _ in range(8)]
        assert active[0] is active[4]
        assert active[0].tobytes() != idle.tobytes()


@pytest.mark.parametrize("state", ["ready", "processing", "speaking", "loading", "error"])
def test_other_states_keep_the_same_icon_in_both_modes(state):
    for frame in range(12):
        assert get_icon(state, frame) is get_icon(state, frame, voice_chat=True)
