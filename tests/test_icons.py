"""Tests for icon generation."""

import pytest
from PIL import Image

from ui.icons import (
    icon_loading, icon_ready, icon_recording, icon_processing, icon_error,
    STATE_ICONS, ICON_SIZE,
)


class TestIconGeneration:
    """Verify each icon returns a 64x64 RGBA image."""

    @pytest.mark.parametrize("fn", [
        icon_loading, icon_ready, icon_recording, icon_processing, icon_error,
    ])
    def test_icon_size_and_mode(self, fn):
        img = fn()
        assert isinstance(img, Image.Image)
        assert img.size == (ICON_SIZE, ICON_SIZE)
        assert img.mode == "RGBA"

    def test_state_icons_dict_has_all_states(self):
        expected = {"loading", "ready", "recording", "processing", "error"}
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
