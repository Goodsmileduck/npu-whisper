"""Glass UI rendering helpers — 2x supersampled PIL shapes + Windows 11 Acrylic."""

from PIL import Image, ImageDraw

# --- Constants ---------------------------------------------------------------

SUPERSAMPLE = 3
TRANSPARENT_COLOR = "#F0F0F0"
TRANSPARENT_RGB = (240, 240, 240)

# Glass palette (RGBA)
GLASS_BG_TOP = (17, 17, 22)
GLASS_BG_BOTTOM = (10, 10, 14)
GLASS_BORDER = (255, 255, 255, 30)
GLASS_HIGHLIGHT = (255, 255, 255, 15)


# --- DWM Title Bar Color -----------------------------------------------------

def set_title_bar_color(window, bg_hex, text_hex="#FFFFFF"):
    """Set Windows 11 title bar background and text colors via DWM.

    Also enables immersive dark mode so window controls (min/max/close)
    render as white icons — visible on colored backgrounds.

    Silent no-op on non-Windows or pre-Win11 systems.
    """
    try:
        import ctypes
        from ctypes import c_int, byref, windll
        hwnd = windll.user32.GetParent(window.winfo_id())

        def _to_colorref(h):
            h = h.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return b << 16 | g << 8 | r

        # DWMWA_USE_IMMERSIVE_DARK_MODE = 20 (white window controls)
        dm = c_int(1)
        windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 20, byref(dm), ctypes.sizeof(dm))
        # DWMWA_CAPTION_COLOR = 35
        bg = c_int(_to_colorref(bg_hex))
        windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 35, byref(bg), ctypes.sizeof(bg))
        # DWMWA_TEXT_COLOR = 36
        text = c_int(_to_colorref(text_hex))
        windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 36, byref(text), ctypes.sizeof(text))
    except Exception:
        pass


# --- DWM Acrylic -------------------------------------------------------------

def apply_acrylic(window, extend_to_client=False):
    """Apply Windows 11 Acrylic backdrop to a CTkToplevel window.

    Args:
        window: CTkToplevel or CTk window.
        extend_to_client: If True, extend the DWM frame into the entire
            client area so the acrylic blur fills the window background.
            The window bg must be set to "black" for DWM to treat those
            pixels as transparent to the blur.

    Silent no-op on non-Windows or pre-Win11 systems.
    """
    try:
        import ctypes
        from ctypes import Structure, c_int, byref, windll
        hwnd = windll.user32.GetParent(window.winfo_id())
        # DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        val = c_int(1)
        windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 20, byref(val), ctypes.sizeof(val))
        # DWMWA_SYSTEMBACKDROP_TYPE = 38, value 3 = Acrylic
        val2 = c_int(3)
        windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 38, byref(val2), ctypes.sizeof(val2))

        if extend_to_client:
            class MARGINS(Structure):
                _fields_ = [("l", c_int), ("r", c_int),
                            ("t", c_int), ("b", c_int)]
            margins = MARGINS(-1, -1, -1, -1)
            windll.dwmapi.DwmExtendFrameIntoClientArea(hwnd, byref(margins))
    except Exception:
        pass


# --- Pill rendering ----------------------------------------------------------

def render_pill(w, h, radius, border_color_rgba=GLASS_BORDER,
                bg_top=GLASS_BG_TOP, bg_bottom=GLASS_BG_BOTTOM,
                border_width=2):
    """Render a glass pill shape at Nx and downsample for smooth edges.

    Uses a single rounded_rectangle for the outer shape so the border
    outline and fill share the exact same edge curve — no misalignment.

    Returns an RGBA PIL Image at (w, h).
    """
    s = SUPERSAMPLE
    sw, sh, sr, sb = w * s, h * s, radius * s, border_width * s
    rect = [0, 0, sw - 1, sh - 1]

    # 1. Fill with top gradient color (full shape)
    img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(rect, radius=sr, fill=(*bg_top, 255))

    # 2. Gradient: blend bottom color from top to bottom inside the shape
    bottom_img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    ImageDraw.Draw(bottom_img).rounded_rectangle(
        rect, radius=sr, fill=(*bg_bottom, 255))
    mask = Image.new("L", (sw, sh), 0)
    for y in range(sh):
        row_val = int(y / max(sh - 1, 1) * 255)
        row = Image.new("L", (sw, 1), row_val)
        mask.paste(row, (0, y))
    img = Image.composite(bottom_img, img, mask)

    # 3. Border outline — drawn on top, uses the same rect/radius
    #    so it's perfectly aligned with the fill edge
    draw2 = ImageDraw.Draw(img)
    br = border_color_rgba
    draw2.rounded_rectangle(rect, radius=sr, fill=None,
                            outline=br, width=sb)

    # 4. Subtle top highlight
    hl_y = sb + 1
    draw2.line([sr, hl_y, sw - 1 - sr, hl_y],
               fill=GLASS_HIGHLIGHT, width=1)

    # Downsample
    return img.resize((w, h), Image.LANCZOS)


# --- Dot rendering -----------------------------------------------------------

def render_dot(diameter, color_rgba):
    """Render a filled circle at 2x and downsample.

    Args:
        diameter: Output diameter in pixels.
        color_rgba: (r, g, b, a) tuple.

    Returns an RGBA PIL Image at (diameter, diameter).
    """
    s = SUPERSAMPLE
    sd = diameter * s
    img = Image.new("RGBA", (sd, sd), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([0, 0, sd - 1, sd - 1], fill=color_rgba)
    return img.resize((diameter, diameter), Image.LANCZOS)


# --- Icon rendering ----------------------------------------------------------

def _hex_to_rgba(hex_color, alpha=255):
    """Convert '#RRGGBB' to (r, g, b, a)."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


def render_icon_mic(size, color):
    """Render a microphone icon at 2x and downsample.

    Args:
        size: Output size (square).
        color: '#RRGGBB' hex color string.
    """
    s = SUPERSAMPLE
    ss = size * s
    img = Image.new("RGBA", (ss, ss), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    c = _hex_to_rgba(color)
    cx = ss // 2

    # Mic capsule body (rounded rectangle)
    cap_w = ss * 5 // 16
    cap_h = ss * 7 // 16
    cap_top = ss * 2 // 16
    cap_r = cap_w // 2
    draw.rounded_rectangle(
        [cx - cap_w, cap_top, cx + cap_w, cap_top + cap_h],
        radius=cap_r, fill=c)

    # U-shaped cradle (arc)
    arc_w = ss * 7 // 16
    arc_top = cap_top + cap_h // 3
    arc_bottom = cap_top + cap_h + ss * 2 // 16
    draw.arc([cx - arc_w, arc_top, cx + arc_w, arc_bottom],
             start=0, end=180, fill=c, width=max(ss // 12, 2))

    # Stem
    stem_top = arc_bottom - ss // 16
    stem_bottom = ss * 13 // 16
    lw = max(ss // 12, 2)
    draw.line([cx, stem_top, cx, stem_bottom], fill=c, width=lw)

    # Base
    base_w = ss * 4 // 16
    draw.line([cx - base_w, stem_bottom, cx + base_w, stem_bottom],
              fill=c, width=lw)

    return img.resize((size, size), Image.LANCZOS)


def render_icon_stop(size, color):
    """Render a rounded-square stop icon at 2x and downsample.

    Args:
        size: Output size (square).
        color: '#RRGGBB' hex color string.
    """
    s = SUPERSAMPLE
    ss = size * s
    img = Image.new("RGBA", (ss, ss), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    c = _hex_to_rgba(color)

    margin = ss * 3 // 16
    r = max(ss // 8, 2)
    draw.rounded_rectangle([margin, margin, ss - margin, ss - margin],
                           radius=r, fill=c)

    return img.resize((size, size), Image.LANCZOS)


def render_button(diameter, ring_color, bg_color, icon_img):
    """Render a circular button with ring, dark background, and icon overlay.

    Args:
        diameter: Output diameter.
        ring_color: '#RRGGBB' for the ring.
        bg_color: '#RRGGBB' for the dark inner fill.
        icon_img: RGBA PIL Image to composite in the center.

    Returns an RGBA PIL Image at (diameter, diameter).
    """
    s = SUPERSAMPLE
    sd = diameter * s
    img = Image.new("RGBA", (sd, sd), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    ring_w = max(sd // 12, 2)
    # Outer ring
    draw.ellipse([0, 0, sd - 1, sd - 1], fill=_hex_to_rgba(ring_color))
    # Inner fill
    draw.ellipse([ring_w, ring_w, sd - 1 - ring_w, sd - 1 - ring_w],
                 fill=_hex_to_rgba(bg_color))

    # Composite icon centered
    icon_s = icon_img.resize((sd // 2, sd // 2), Image.LANCZOS)
    offset = (sd - icon_s.width) // 2
    img.paste(icon_s, (offset, offset), icon_s)

    return img.resize((diameter, diameter), Image.LANCZOS)


# --- Waveform rendering ------------------------------------------------------

def render_waveform(width, height, levels, bar_width=3, gap=3, color="#FF6961"):
    """Render audio waveform bars at 2x and downsample.

    Args:
        width: Output width.
        height: Output height.
        levels: List of float values 0.0-1.0.
        bar_width: Width of each bar in output pixels.
        gap: Gap between bars in output pixels.
        color: '#RRGGBB' hex color.

    Returns an RGBA PIL Image at (width, height).
    """
    s = SUPERSAMPLE
    sw, sh = width * s, height * s
    sbw, sg = bar_width * s, gap * s

    img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    c = _hex_to_rgba(color)

    max_h = sh - 4 * s  # 4px margin top+bottom at output scale
    mid_y = sh // 2
    x = 0
    for lv in levels:
        if x + sbw > sw:
            break
        bh = max(4 * s, int(lv * max_h))
        y1 = mid_y - bh // 2
        y2 = mid_y + bh // 2
        r = sbw // 2
        draw.rounded_rectangle([x, y1, x + sbw, y2], radius=r, fill=c)
        x += sbw + sg

    return img.resize((width, height), Image.LANCZOS)


# --- Compositing helpers -----------------------------------------------------

def composite_on_transparent(rgba_image):
    """Paste RGBA image onto the transparent-color background.

    Windows `-transparentcolor` only matches EXACT pixel values.
    Semi-transparent edge pixels from LANCZOS would blend with the bg
    color producing light-colored fringe that is NOT exactly the
    transparent color — visible as artifacts.  Fix: threshold alpha
    to binary (opaque or transparent) before compositing.  The 2x
    supersampled geometry still provides smoother corner placement.

    Returns an RGB PIL Image.
    """
    # Binary alpha: eliminate semi-transparent fringe pixels
    r, g, b, a = rgba_image.split()
    a = a.point(lambda v: 255 if v >= 128 else 0)
    clean = Image.merge("RGBA", (r, g, b, a))

    bg = Image.new("RGB", clean.size, TRANSPARENT_RGB)
    bg.paste(clean, (0, 0), clean)
    return bg


def pil_to_photo(image):
    """Convert a PIL Image to a tkinter-compatible PhotoImage.

    Caller must keep a reference to prevent garbage collection.
    """
    from PIL import ImageTk
    return ImageTk.PhotoImage(image)


# --- Pill cache --------------------------------------------------------------

class PillCache:
    """Simple cache for rendered pill images, keyed by (w, h)."""

    def __init__(self, max_size=8):
        self._cache: dict[tuple[int, int], Image.Image] = {}
        self._max_size = max_size

    def get(self, w, h, **kwargs) -> Image.Image:
        """Get or render a pill. kwargs are passed to render_pill."""
        key = (w, h)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        img = render_pill(w, h, **kwargs)
        if len(self._cache) >= self._max_size:
            # Evict oldest
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = img
        return img

    def invalidate(self):
        """Clear all cached pills."""
        self._cache.clear()
