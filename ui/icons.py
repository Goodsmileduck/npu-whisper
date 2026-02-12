"""Dynamic icon generation for system tray states using Pillow."""

import math
from PIL import Image, ImageDraw

ICON_SIZE = 64
APP_ICON_COLOR = "#1976D2"


def render_app_icon(size, color=APP_ICON_COLOR):
    """Render the app icon: filled circle with two sound wave arcs.

    Uses 3x supersample + LANCZOS downscale for smooth edges.
    Returns an RGBA PIL Image at (size, size).
    """
    s = 3
    ss = size * s
    img = Image.new("RGBA", (ss, ss), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Filled circle — positioned left of center
    cx = int(ss * 0.30)
    cy = ss // 2
    cr = int(ss * 0.20)
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=color)

    # Arc center — between circle center and right edge
    ax = cx + cr // 2
    ay = cy

    # Arc line width and angle span
    lw = max(int(ss * 0.065), 3)
    cap_r = lw / 2
    start_deg, end_deg = 308, 52

    # Two concentric sound-wave arcs with rounded caps
    for r in [int(ss * 0.19), int(ss * 0.34)]:
        draw.arc([ax - r, ay - r, ax + r, ay + r],
                 start=start_deg, end=end_deg, fill=color, width=lw)
        for deg in [start_deg, end_deg]:
            rad = math.radians(deg)
            px = ax + r * math.cos(rad)
            py = ay + r * math.sin(rad)
            draw.ellipse([px - cap_r, py - cap_r,
                          px + cap_r, py + cap_r], fill=color)

    return img.resize((size, size), Image.LANCZOS)


def _app_icon_with_status(dot_color, show_x=False):
    """App icon with a colored status dot in the bottom-right corner."""
    base = render_app_icon(ICON_SIZE)
    draw = ImageDraw.Draw(base)

    # Status dot: 20px diameter with 2px white outline
    dot_d = 20
    outline_w = 2
    x2 = ICON_SIZE - 2
    y2 = ICON_SIZE - 2
    x1 = x2 - dot_d
    y1 = y2 - dot_d

    # White outline ring
    draw.ellipse([x1 - outline_w, y1 - outline_w,
                  x2 + outline_w, y2 + outline_w], fill="white")
    # Colored fill
    draw.ellipse([x1, y1, x2, y2], fill=dot_color)

    if show_x:
        m = 5
        draw.line([x1 + m, y1 + m, x2 - m, y2 - m], fill="white", width=2)
        draw.line([x2 - m, y1 + m, x1 + m, y2 - m], fill="white", width=2)

    return base


def icon_loading() -> Image.Image:
    """App icon with gray status dot — model loading."""
    return _app_icon_with_status("#808080")


def icon_ready() -> Image.Image:
    """App icon with green status dot — ready for recording."""
    return _app_icon_with_status("#22C55E")


def icon_recording() -> Image.Image:
    """App icon with red status dot — microphone active."""
    return _app_icon_with_status("#EF4444")


def icon_processing() -> Image.Image:
    """App icon with yellow status dot — transcribing."""
    return _app_icon_with_status("#EAB308")


def icon_error() -> Image.Image:
    """App icon with red status dot + white X — error state."""
    return _app_icon_with_status("#EF4444", show_x=True)


STATE_ICONS = {
    "loading": icon_loading,
    "ready": icon_ready,
    "recording": icon_recording,
    "processing": icon_processing,
    "error": icon_error,
}
