"""One translucent window for the mascot and the current conversation.

Uses PIL supersampled rendering via ui.glass for anti-aliased shapes.
"""

import tkinter as tk
from pathlib import Path
from time import monotonic

from debora_whisper.ui.glass import (
    TRANSPARENT_COLOR, PillCache, composite_on_transparent, pil_to_photo,
    _hex_to_rgba, render_pill,
)

# Débora's face, shown at the panel's left edge.
MASCOT_PATH = Path(__file__).parent / "assets" / "debora.jpg"
# Her animated bust, 12 fps, 228x128, cut from a generated video: a calm
# loop (played forward then back, so it has no seam) and a zoom into her face
# and back, repeated while someone talks (the user or her voice).
MASCOT_LOOP_PATH = Path(__file__).parent / "assets" / "mascot_loop.webp"
MASCOT_ZOOM_PATH = Path(__file__).parent / "assets" / "mascot_zoom.webp"

# Color used for window transparency (never appears in UI)
_TRANSPARENT = TRANSPARENT_COLOR


class OverlayWindow:
    """A fixed mascot anchor with an optional, clipped conversation to its right."""

    # --- Dimensions ---
    COMPACT_W = 64
    COMPACT_H = 38
    RADIUS = 6   # a slightly rounded rectangle, not a capsule
    # Whole-window opacity: Tk has no per-pixel alpha, so text fades too.
    OPACITY = 0.85

    # --- iOS-inspired dark palette ---
    BG = "#0A0A0A"
    TEXT = "#D8D8DC"      # a slightly gray white, softer than pure white
    TEXT_DIM = "#9A9AA0"
    # Windows 11's text-optimized Segoe; Segoe UI where it is missing.
    FONTS = (("Segoe UI Variable Text", "Segoe UI Variable Text Semibold"),
             ("Segoe UI", "Segoe UI Semibold"))
    GREEN = "#30D158"
    RED = "#FF453A"
    VIOLET = "#8B5CF6"
    VOICE_MODE_COLOR = "#A855F7"
    AMBER = "#FF9F0A"
    BLUE = "#0A84FF"
    GRAY = "#48484A"

    # The mascot flaps its wings in silence and zooms while someone talks;
    # it stands still only while loading or after an error.
    _MASCOT_STILL_STATES = ("loading", "error")
    MASCOT_FPS = 12
    MASCOT_FEATHER = 2  # logical px of haze where the mascot meets the panel
    MASCOT_ASPECT = 16 / 9  # the whole video frame, uncropped

    # States in which a click toggles recording (speaking: cuts the reply).
    _CLICK_STATES = ("ready", "recording", "speaking")
    # A press that moves less than this (logical px) is a click, not a drag.
    _CLICK_SLOP = 4

    # --- Balloon dimensions ---
    BALLOON_DEFAULT_W = 360
    BALLOON_MIN_W = 200
    BALLOON_PAD = 6
    BALLOON_GAP = 8
    BALLOON_FONT_SIZE = 16
    BALLOON_DURATION = 2500  # the former result -> ready auto-dismiss delay
    SLIDE_SECONDS = 0.2
    # A sentence whose audio stops with more left to slide than this was cut
    # short (interrupted); less is just the output device's buffer latency.
    SLIDE_CUT_SECONDS = 0.3
    # While she works on the input, the text breathes between these
    # opacities over the panel, once per period.
    PULSE_SECONDS = 1.4
    PULSE_MIN = 0.45

    # --- Resize grip: a small glass capsule at the right edge ---
    GRIP_W = 4
    GRIP_H = 24
    GRIP_INSET = 4      # from the window's right edge
    GRIP_HIT_W = 16     # invisible hit area, centered on the capsule
    GRIP_IDLE = 0.2     # white opacity over the panel
    GRIP_ACTIVE = 0.5   # hover and drag
    GRIP_FADE_MS = 150
    GRIP_STEPS = 5

    def __init__(self, root, on_toggle=None, pos_x=None, pos_y=10, on_pos_changed=None,
                 balloon_width=None, on_width_changed=None):
        """
        Args:
            root: Parent Tk/CTk window.
            on_toggle: Callback() to toggle recording on mascot click.
            pos_x: Initial mascot center X in desktop pixels.
            pos_y: Initial top Y position.
            on_pos_changed: Callback(x, y) when dragged to a new position.
            balloon_width: Preferred text area width in logical pixels, or None.
            on_width_changed: Callback(width) after resizing the text area.
        """
        self._root = root
        self._on_toggle = on_toggle
        self._on_pos_changed = on_pos_changed
        self._on_width_changed = on_width_changed
        self._balloon_width = balloon_width
        self._win: tk.Toplevel | None = None
        self._canvas: tk.Canvas | None = None

        # Position is in desktop pixels; dimensions below are logical (96 DPI).
        self._pos_x: int | None = pos_x
        self._pos_y: int = pos_y

        s = self._scale = self._get_scale()
        self._cur_w = self.COMPACT_W * s
        self._cur_h = self.COMPACT_H * s
        self._window_x = None

        # Drag state
        self._drag_offset_x = 0
        self._drag_offset_y = 0

        # State
        self._state = "loading"
        self._hidden = False
        self._continuous_listening = False
        self._voice_mode = False
        self._mascot_clip = "loop"
        self._mascot_index = 0
        self._mascot_anim_id = None
        self._talking = set()  # "user" and/or "debora" while their voice is heard
        self._user_text = ""
        self._reply_text = ""
        self._voice_turn = False
        self._turn_finished = False
        self._text_dismissed = False
        self._balloon_id = None
        self._show_balloon = True
        self._slide_id = None
        self._slides = {}
        self._pace = None  # audio seconds for the next slide (show_speaking)
        self._slide_held = False
        self._grip_level = 0  # 0 idle .. GRIP_STEPS active
        self._grip_hover = False
        self._grip_fade_id = None

        # PIL rendering state
        self._pill_cache = PillCache()
        self._photo_refs: list = []  # prevent GC of PhotoImages
        self._build()

    def _get_scale(self) -> float:
        """Read the effective DPI of this window, including Windows display scaling."""
        try:
            import ctypes
            from ctypes import wintypes

            get_dpi = ctypes.windll.user32.GetDpiForWindow
            get_dpi.argtypes = [wintypes.HWND]
            get_dpi.restype = wintypes.UINT
            window = self._win if self._win is not None else self._root
            dpi = get_dpi(window.winfo_id())
            if dpi > 0:
                return dpi / 96.0
        except (AttributeError, OSError, tk.TclError):
            pass
        try:
            dpi = float(self._root.winfo_fpixels("1i"))
            if dpi > 0:
                return dpi / 96.0
        except (AttributeError, ValueError, tk.TclError):
            pass
        return 1.0

    @staticmethod
    def _font_pixels(points, scale):
        # Negative Tk sizes are pixels. Positive sizes would apply Tk's global
        # points-to-pixels scaling AGAIN, which may belong to another monitor.
        return -max(1, round(points * 96 / 72 * scale))

    def _refresh_scale(self, event=None):
        """Resize all artwork together when moving displays or changing DPI."""
        if event is not None and event.widget is not self._win:
            return
        scale = self._get_scale()
        if scale == self._scale:
            return
        self._scale = scale
        self._reset_slides()
        self._position()
        self._redraw()

    # --- Window setup -----------------------------------------------------

    def _build(self):
        self._win = tk.Toplevel(self._root)
        self._win.overrideredirect(True)
        self._win.attributes("-topmost", True)
        self._win.configure(bg=_TRANSPARENT)

        try:
            self._win.attributes("-transparentcolor", _TRANSPARENT)
        except Exception:
            pass  # Non-Windows fallback: square corners
        try:
            self._win.attributes("-alpha", self.OPACITY)
        except Exception:
            pass

        # Re-assert topmost periodically so the Windows 11 taskbar doesn't cover it
        def _force_topmost():
            if self._win and self._win.winfo_exists():
                if not self._hidden:
                    self._win.attributes("-topmost", True)
                    self._win.lift()
                    self._refresh_scale()
                self._root.after(500, _force_topmost)

        self._canvas = tk.Canvas(
            self._win, bg=_TRANSPARENT, highlightthickness=0,
            width=self.COMPACT_W, height=self.COMPACT_H,
        )
        self._canvas.pack(fill="both", expand=True)
        # A child canvas is a real clipping viewport, not a second Toplevel.
        # It has the same flat glass color and inherits the window's opacity.
        self._text_canvas = tk.Canvas(
            self._canvas, bg=self.BG, highlightthickness=0, borderwidth=0)
        self._text_canvas.bind("<ButtonPress-1>", lambda e: self._dismiss_balloon())
        self._text_canvas.configure(cursor="hand2")

        self._canvas.bind("<Leave>", self._on_leave)
        self._canvas.bind("<Motion>", self._on_mouse_move)
        self._canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self._canvas.bind("<B1-Motion>", self._on_drag_move)
        self._canvas.bind("<ButtonRelease-1>", self._on_drag_end)

        self._win.update_idletasks()
        self._restore_position()
        self._win.bind("<Configure>", self._refresh_scale)
        _force_topmost()
        self._apply_window_flags()
        self._redraw()

    def _apply_window_flags(self):
        """Prevent focus stealing and hide from taskbar (Windows)."""
        try:
            import ctypes
            GWL_EXSTYLE = -20
            WS_EX_NOACTIVATE = 0x08000000
            WS_EX_TOOLWINDOW = 0x00000080
            WS_EX_APPWINDOW = 0x00040000
            WS_EX_TOPMOST = 0x00000008
            hwnd = ctypes.windll.user32.GetParent(self._win.winfo_id())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style = (style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST) & ~WS_EX_APPWINDOW
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        except Exception:
            pass

    def _get_screen_width(self) -> int:
        """Get the primary screen width reliably."""
        try:
            import ctypes
            return ctypes.windll.user32.GetSystemMetrics(0)
        except Exception:
            return self._win.winfo_screenwidth()

    def _monitor_work_area(self, x, y, full=False):
        """Find the monitor containing the mascot's anchor.
        full: the whole monitor, taskbar included, not just the work area."""
        try:
            import ctypes
            from ctypes import wintypes
            class POINT(ctypes.Structure):
                _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]
            class RECT(ctypes.Structure):
                _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                            ("right", wintypes.LONG), ("bottom", wintypes.LONG)]
            class MONITORINFO(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", RECT),
                            ("rcWork", RECT), ("dwFlags", wintypes.DWORD)]
            
            user32 = ctypes.windll.user32
            user32.MonitorFromPoint.argtypes = [POINT, wintypes.DWORD]
            user32.MonitorFromPoint.restype = wintypes.HANDLE
            user32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MONITORINFO)]
            user32.GetMonitorInfoW.restype = wintypes.BOOL
            pt = POINT(x, y)
            hMonitor = user32.MonitorFromPoint(pt, 2) # MONITOR_DEFAULTTONEAREST
            
            mi = MONITORINFO()
            mi.cbSize = ctypes.sizeof(MONITORINFO)
            if user32.GetMonitorInfoW(hMonitor, ctypes.byref(mi)):
                rect = mi.rcMonitor if full else mi.rcWork
                return rect.left, rect.top, rect.right, rect.bottom
        except Exception:
            pass
        return 0, 0, self._win.winfo_screenwidth(), self._win.winfo_screenheight()

    def _mascot_geometry(self):
        """Use exactly the same rounded pixel dimensions as the drawn clip."""
        pad = round(2 * self._scale)
        height = round(self.COMPACT_H * self._scale) - 2 * pad
        right = pad + round(height * self.MASCOT_ASPECT)
        return pad, height, right, right + pad

    def _text_geometry(self, width):
        mascot_right = self._mascot_geometry()[2]
        gap = round(self.BALLOON_GAP * self._scale)
        bar_width = max(1, round(self.GRIP_W * self._scale))
        bar_left = width - round(self.GRIP_INSET * self._scale) - bar_width
        text_left = mascot_right + gap
        text_right = max(text_left, bar_left - gap)
        return text_left, text_right, bar_left, bar_width

    def _grip_image(self, level, height):
        """The resize capsule at fade level 0 (idle) .. GRIP_STEPS (hover,
        drag), pre-blended over the flat panel: Tk images have no alpha."""
        from PIL import Image
        s = self._scale
        width = max(1, round(self.GRIP_W * s))
        height = max(1, min(round(self.GRIP_H * s), height))
        cache = self.__dict__.setdefault("_grip_cache", {})
        key = (width, height, level)
        if key not in cache:
            opacity = self.GRIP_IDLE + (self.GRIP_ACTIVE - self.GRIP_IDLE) * level / self.GRIP_STEPS
            capsule = render_pill(width, height, radius=max(1, round(2 * s)),
                                  bg_top=(255, 255, 255), bg_bottom=(216, 216, 222),
                                  border_width=0)
            capsule.putalpha(capsule.getchannel("A").point(lambda a: round(a * opacity)))
            image = Image.new("RGBA", (width, height), _hex_to_rgba(self.BG))
            image.alpha_composite(capsule)
            cache[key] = image
        return cache[key]

    def _set_grip_active(self, active):
        """Fade the capsule toward its hover/drag look, or back to idle."""
        active = active or getattr(self, "_dragging", False) and self._press_region == "resize"
        if active == self._grip_hover:
            return
        self._grip_hover = active
        if self._grip_fade_id is None:
            self._grip_fade_id = self._root.after(
                self.GRIP_FADE_MS // self.GRIP_STEPS, self._grip_fade_tick)

    def _grip_fade_tick(self):
        self._grip_fade_id = None
        target = self.GRIP_STEPS if self._grip_hover else 0
        if self._grip_level == target:
            return
        self._grip_level += 1 if target > self._grip_level else -1
        self._draw_grip()
        self._grip_fade_id = self._root.after(
            self.GRIP_FADE_MS // self.GRIP_STEPS, self._grip_fade_tick)

    def _draw_grip(self):
        c = self._canvas
        c.delete("grip")
        w, h = int(self._cur_w), int(self._cur_h)
        left, right, bar, _ = self._text_geometry(w)
        if not self._conversation_lines() or right <= left:
            return
        image = self._grip_image(self._grip_level, h - 2 * round(self.BALLOON_PAD * self._scale))
        photo = pil_to_photo(image)
        self.__dict__["_grip_photo"] = photo
        c.create_image(bar, (h - image.height) // 2, image=photo, anchor="nw", tags="grip")

    def _conversation_lines(self):
        if not self._show_balloon or self._text_dismissed:
            return []
        if self._voice_turn:
            lines = [("user", f"Você: {self._user_text}", self.TEXT_DIM)]
            if self._reply_text:
                lines.append(("reply", f"Débora: {self._reply_text}", self.TEXT))
            return lines if self._user_text or self._reply_text else []
        return [("user", self._user_text, self.TEXT)] if self._user_text else []

    def _text_font(self):
        import tkinter.font as tkfont
        return tkfont.Font(root=self._root, font=self._font(
            self._font_pixels(self.BALLOON_FONT_SIZE, self._scale)))

    def _restore_position(self):
        # The root may be on another monitor. First move this window, then
        # restore the saved mascot center using the destination monitor's DPI.
        center, top = self._pos_x, self._pos_y
        self._position()
        self._win.update_idletasks()
        scale = self._get_scale()
        if scale != self._scale:
            self._scale = scale
            self._window_x = None
            self._pos_x, self._pos_y = center, top
            self._position()

    def _position(self):
        """Only the mascot is the position anchor; text never recenters it."""
        mascot_w = self._mascot_geometry()[3]
        mascot_h = round(self.COMPACT_H * self._scale)
        if self._window_x is None:
            center = self._get_screen_width() // 2 if self._pos_x is None else self._pos_x
            self._window_x = center - mascot_w // 2
        x, y = self._window_x, self._pos_y
        wl, wt, wr, wb = self._monitor_work_area(
            x + mascot_w // 2, y + mascot_h // 2, full=True)
        x = max(wl, min(x, wr - mascot_w))
        y = max(wt, min(y, wb - mascot_h))
        self._window_x, self._pos_x, self._pos_y = x, x + mascot_w // 2, y
        w, h = mascot_w, mascot_h
        lines = self._conversation_lines()
        if lines:
            requested = self._balloon_width or self.BALLOON_DEFAULT_W
            w += min(round(requested * self._scale), max(0, wr - x - mascot_w))
            # Both sides share one row; her reply pushes the user's text left.
            # The row is centered on the mascot; only a font taller than her
            # makes the window grow.
            h = max(h, self._text_font().metrics("linespace"))
        self._cur_w, self._cur_h = w, h
        # +negative coordinates are absolute desktop positions in Tk geometry.
        self._win.geometry(f"{w}x{h}+{x}+{y}")
        self._canvas.configure(width=w, height=h)

    def _flat(self) -> dict:
        """render_pill arguments for a flat panel: one color, no border."""
        bg = _hex_to_rgba(self.BG)[:3]
        return {"bg_top": bg, "bg_bottom": bg, "border_width": 0}

    def _font(self, size: int, semibold: bool = False) -> tuple:
        """(family, size) in the most legible installed family."""
        families = getattr(self, "_font_families", None)
        if families is None:
            try:
                import tkinter.font as tkfont
                installed = set(tkfont.families(self._root))
            except Exception:
                installed = set()
            families = next((pair for pair in self.FONTS if pair[0] in installed),
                            self.FONTS[-1])
            self._font_families = families
        return (families[1] if semibold else families[0], size)

    def _redraw(self):
        if self._hidden:
            return
        c = self._canvas
        c.delete("all")
        self._photo_refs.clear()
        s = self._scale
        w, h = int(self._cur_w), int(self._cur_h)
        frame = self._pill_cache.get(w, h, radius=round(self.RADIUS * s),
                                     **self._flat()).copy()
        self._sync_mascot_animation()
        pad, mascot_h, _, mascot_w = self._mascot_geometry()
        frames = self._mascot_frames(mascot_h, max(1, round(self.MASCOT_FEATHER * s)),
                                     max(1, round(self.RADIUS * s)), clip=self._mascot_clip)
        if frames:
            frame.alpha_composite(frames[self._mascot_index % len(frames)], (pad, pad))
        if self._voice_mode:
            frame.alpha_composite(self._voice_border(mascot_w, h, s), (0, 0))
        photo = pil_to_photo(composite_on_transparent(frame))
        self._photo_refs.append(photo)
        c.create_image(0, 0, image=photo, anchor="nw")
        left, right, bar, bar_w = self._text_geometry(w)
        if self._conversation_lines() and right > left:
            self._draw_grip()
            line_h = min(h, self._text_font().metrics("linespace"))
            self._text_canvas.place(x=left, y=(h - line_h) // 2,
                                    width=right - left, height=line_h)
            self._draw_text(right - left)
        else:
            self._text_canvas.place_forget()

    @staticmethod
    def _line_target(line_width, visible_width):
        return min(0, visible_width - line_width)

    def _slide_position(self, slide, now):
        start, end, began, pace = slide
        if pace:
            # Her sentence: a steady teleprompter pace, so its last word
            # comes into view as she finishes saying it.
            return start + (end - start) * min(1.0, max(0.0, (now - began) / pace))
        progress = min(1.0, max(0.0, (now - began) / self.SLIDE_SECONDS))
        return start + (end - start) * (1 - (1 - progress) ** 3)

    def _slide_offset(self, key, target, now):
        slide = self._slides.get(key)
        if slide is None:
            slide = (0.0, 0.0, now, None)
        current = self._slide_position(slide, now)
        if target != slide[1] and not self._slide_held:
            # Retarget from the current position, even if another sentence
            # arrived before the last animation finished.
            slide = (current, target, now, self._pace)
        self._slides[key] = slide
        return current

    def _hold_cut_slide(self):
        """Her voice stopped: an interrupted sentence stops sliding where
        it is, instead of rushing to its end."""
        slide = self._slides.get("row")
        if slide is None or not slide[3]:
            return
        now = monotonic()
        if slide[2] + slide[3] - now > self.SLIDE_CUT_SECONDS:
            current = self._slide_position(slide, now)
            self._slides["row"] = (current, current, now, None)
            self._slide_held = True

    def _draw_text(self, visible_width):
        canvas = self._text_canvas
        canvas.delete("all")
        font = self._text_font()
        now = monotonic()
        # From the input until her voice is ready: her text may arrive
        # (speaking) well before the audio does.
        pulsing = (self._state == "processing" or self._state == "speaking"
                   and not self.__dict__.get("_voice_heard"))
        # One row: "Você: …" then "Débora: …", slid left together, so her
        # reply pushes the user's words under the mascot like new words do.
        items, x = [], 0
        space = font.measure("   ")
        for key, text, fill in self._conversation_lines():
            # Newlines in a transcript/reply never create additional rows.
            text = " ".join(text.split())
            if pulsing:
                fill = self._pulse(fill, now)
            item = canvas.create_text(0, 0, text=text, fill=fill, font=font, anchor="nw")
            bbox = canvas.bbox(item)
            # Tk text bboxes include font bearings: normalize their left edge,
            # so fitting text starts at text_left.
            canvas.move(item, x - bbox[0], 0)
            items.append(item)
            x += bbox[2] - bbox[0] + space
        width = max(0, x - space)
        target = self._line_target(width, visible_width)
        offset = self._slide_offset("row", target, now)
        self._pace = None  # only the sentence it came with
        for item in items:
            canvas.move(item, offset, 0)
        moving = not self._slide_held and abs(offset - target) > 0.01
        if (moving or pulsing) and self._slide_id is None:
            self._slide_id = self._root.after(16 if moving else 50, self._slide_tick)

    def _pulse(self, fill, now):
        """fill faded toward the panel by the breathing pulse at time now."""
        import math
        wave = 0.5 + 0.5 * math.cos(2 * math.pi * now / self.PULSE_SECONDS)
        opacity = self.PULSE_MIN + (1 - self.PULSE_MIN) * wave
        bg, fg = _hex_to_rgba(self.BG), _hex_to_rgba(fill)
        return "#" + "".join(f"{round(b + (f - b) * opacity):02X}" for f, b in zip(fg[:3], bg[:3]))

    def _slide_tick(self):
        self._slide_id = None
        if self._hidden:
            return
        left, right, _, _ = self._text_geometry(int(self._cur_w))
        if self._conversation_lines() and right > left:
            self._draw_text(right - left)

    def _reset_slides(self):
        if self._slide_id is not None:
            self._root.after_cancel(self._slide_id)
            self._slide_id = None
        self._slides.clear()
        self._slide_held = False

    def _update_layout(self, *, force_hidden=False):
        was_hidden = self._hidden
        self._hidden = force_hidden or getattr(self, "_explicitly_hidden", False) or (self._state == "ready" and not self._voice_mode
                        and not self._continuous_listening
                        and not getattr(self, "_dragging", False)
                        and not self._conversation_lines())
        if self._hidden and not was_hidden:
            self._win.withdraw()
            self._sync_mascot_animation()
            self._reset_slides()
            if self._grip_fade_id is not None:
                self._root.after_cancel(self._grip_fade_id)
                self._grip_fade_id = None
            self._grip_hover = False
            self._grip_level = 0
        if was_hidden and not self._hidden:
            self._scale = self._get_scale()
        self._position()
        self._redraw()
        if was_hidden and not self._hidden:
            # Paint and position before mapping. Keep NOACTIVATE on both
            # sides of deiconify, which may recreate the Windows wrapper.
            self._apply_window_flags()
            self._win.deiconify()
            self._win.attributes("-topmost", True)
            self._apply_window_flags()

    # --- Drag, resize and click -------------------------------------------

    def _hit_region(self, x):
        _, _, bar, bar_w = self._text_geometry(int(self._cur_w))
        if self._conversation_lines() and self._cur_w > self._mascot_geometry()[3]:
            reach = (self.GRIP_HIT_W * self._scale - bar_w) / 2
            if bar - reach <= x <= bar + bar_w + reach:
                return "resize"
            if x >= self._mascot_geometry()[2]:
                return "text"
        return "mascot"

    def _on_drag_start(self, event):
        self._press_region = self._hit_region(event.x)
        self._press_x, self._press_y = event.x, event.y
        self._dragging = False
        self._drag_offset_x, self._drag_offset_y = event.x, event.y
        self._resize_start = (self._cur_w - self._mascot_geometry()[3]) / self._scale
        if self._press_region == "resize":
            self._set_grip_active(True)

    def _on_drag_move(self, event):
        if self._press_region == "text":
            return
        if not self._dragging:
            slop = self._CLICK_SLOP * self._scale
            if (abs(event.x - self._press_x) <= slop
                    and abs(event.y - self._press_y) <= slop):
                return
            self._dragging = True
        if self._press_region == "resize":
            requested = self._resize_start + (event.x - self._press_x) / self._scale
            mascot_w = self._mascot_geometry()[3]
            _, _, right, _ = self._monitor_work_area(self._pos_x, self._pos_y, full=True)
            maximum = max(self.BALLOON_MIN_W,
                          (right - self._window_x - mascot_w) / self._scale)
            self._balloon_width = round(max(self.BALLOON_MIN_W, min(requested, maximum)))
        else:
            self._window_x = self._win.winfo_x() + event.x - self._drag_offset_x
            self._pos_y = self._win.winfo_y() + event.y - self._drag_offset_y
        self._update_layout()

    def _on_drag_end(self, event):
        if self._dragging:
            self._dragging = False
            if self._press_region == "resize":
                self._set_grip_active(self._hit_region(event.x) == "resize")
                if self._on_width_changed:
                    self._on_width_changed(self._balloon_width)
            elif self._on_pos_changed:
                self._on_pos_changed(self._pos_x, self._pos_y)
            self._update_layout()
        elif self._press_region == "text":
            self._dismiss_balloon()
        elif (self._press_region == "mascot" and self._state in self._CLICK_STATES
              and self._on_toggle):
            self._on_toggle()

    def _on_leave(self, event):
        self._canvas.configure(cursor="")
        self._set_grip_active(False)

    def _on_mouse_move(self, event):
        region = self._hit_region(event.x)
        cursor = "sb_h_double_arrow" if region == "resize" else (
            "hand2" if region == "text" or self._state in self._CLICK_STATES else "")
        self._canvas.configure(cursor=cursor)
        self._set_grip_active(region == "resize")

    # --- Mascot -----------------------------------------------------------

    VOICE_GLOW = 4  # logical px the voice mode's purple fades over, inward
    VOICE_RIM = 1   # logical px of dark panel kept outside the purple

    def _voice_border(self, width: int, height: int, s: float):
        """The voice mode's purple glow around the mascot, fading inward.
        The window's corners are a color key with no partial alpha, so they
        are jagged; a dark rim stays outside the purple, whose curve is
        supersampled and blurred to stay smooth against the panel. Cached
        per size because _redraw runs on every mascot frame."""
        cache = self.__dict__.setdefault("_border_cache", {})
        key = (width, height, s)
        if key not in cache:
            from PIL import Image, ImageChops, ImageDraw, ImageFilter
            ss = 4
            radius = max(1, round(self.RADIUS * s)) * ss
            rim = max(1, round(self.VOICE_RIM * s)) * ss
            glow = max(2, round(self.VOICE_GLOW * s)) * ss
            size = (width * ss, height * ss)

            def rect(inset):
                mask = Image.new("L", size)
                ImageDraw.Draw(mask).rounded_rectangle(
                    (inset, inset, size[0] - 1 - inset, size[1] - 1 - inset),
                    radius=max(ss, radius - inset), fill=255)
                return mask

            outer = rect(rim).filter(ImageFilter.GaussianBlur(ss * 0.4))
            inner = rect(rim + glow // 2).filter(ImageFilter.GaussianBlur(glow / 2))
            alpha = ImageChops.subtract(outer, inner).resize(
                (width, height), Image.LANCZOS).point(
                lambda v: 0 if v < 8 else round(v * 0.9))
            border = Image.new("RGBA", (width, height), self.VOICE_MODE_COLOR)
            border.putalpha(alpha)
            cache[key] = border
        return cache[key]

    def _mascot_frames(self, height: int, feather: int = 0, radius: int = 0,
                       clip: str = "loop") -> list:
        """A mascot clip's frames as rounded rectangles height px tall in the
        video's 16:9, whose edge fades out over about feather px (a haze into
        the panel), cached per size. The loop falls back to the still image
        (center-cropped to 16:9); a clip that does not load has no frames."""
        cache = self.__dict__.setdefault("_mascot_cache", {})
        key = (clip, height, feather, radius)
        if key in cache:
            return cache[key]
        from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageSequence
        all_sources = self.__dict__.setdefault("_mascot_sources", {})
        sources = all_sources.get(clip)
        if sources is None:
            sources = []
            paths = (MASCOT_LOOP_PATH, MASCOT_PATH) if clip == "loop" else (MASCOT_ZOOM_PATH,)
            for path in paths:
                try:
                    with Image.open(path) as img:
                        sources = [f.convert("RGBA") for f in ImageSequence.Iterator(img)]
                    break
                except Exception:
                    continue
            all_sources[clip] = sources
        frames = []
        if sources and height > 0:
            width = round(height * self.MASCOT_ASPECT)
            ss = 4  # supersampled, for a smooth edge
            inset = feather * ss
            mask = Image.new("L", (width * ss, height * ss), 0)
            ImageDraw.Draw(mask).rounded_rectangle(
                (inset, inset, width * ss - 1 - inset, height * ss - 1 - inset),
                radius=radius * ss, fill=255)
            if feather:
                mask = mask.filter(ImageFilter.GaussianBlur(inset / 2))
            mask = mask.resize((width, height), Image.LANCZOS)
            for src in sources:
                thumb = ImageOps.fit(src, (width, height), Image.LANCZOS)
                thumb.putalpha(mask)
                frames.append(thumb)
        cache[key] = frames
        return frames

    def _sync_mascot_animation(self):
        """Pause the mascot while hidden, loading, or showing an error."""
        animate = not self._hidden and self._state not in self._MASCOT_STILL_STATES
        running = self.__dict__.get("_mascot_anim_id")
        if animate and not running:
            self._mascot_anim_id = self._root.after(
                1000 // self.MASCOT_FPS, self._mascot_tick)
        elif not animate:
            if running:
                self._root.after_cancel(running)
                self._mascot_anim_id = None
            self._mascot_clip = "loop"
            self._mascot_index = 0

    def _mascot_tick(self):
        self._mascot_anim_id = None
        if self._hidden:
            return
        count = len(self._mascot_frames(1, clip=self._mascot_clip))
        talking = bool(self.__dict__.get("_talking"))
        step = 1
        if self._mascot_clip == "zoom" and not talking and self._mascot_index < count // 2:
            # Silence early in the zoom: play it back out, never jump mid-zoom.
            step = -1
        self._mascot_index += step
        if self._mascot_index < 0 or self._mascot_index >= count:
            # A clip boundary: the zoom repeats while someone talks.
            self._mascot_clip = "zoom" if talking else "loop"
            self._mascot_index = 0
        self._redraw()  # reschedules through _sync_mascot_animation

    def set_talking(self, who: str, active: bool):
        """who ("user" or "debora") started or stopped talking. The zoom
        starts at once and ends at a clip boundary (see _mascot_tick)."""
        talking = self.__dict__.setdefault("_talking", set())
        was = bool(talking)
        (talking.add if active else talking.discard)(who)
        if who == "debora" and active:
            self._voice_heard = True
        elif who == "debora":
            self._hold_cut_slide()
        if talking and not was and self._mascot_clip == "loop":
            self._mascot_clip, self._mascot_index = "zoom", 0
            if self._state not in self._MASCOT_STILL_STATES:
                self._redraw()

    # --- Public state API -------------------------------------------------

    def _cancel_dismiss(self):
        if self._balloon_id is not None:
            self._root.after_cancel(self._balloon_id)
            self._balloon_id = None

    def _finish_turn(self):
        self._turn_finished = True
        if self._balloon_id is None and not self._text_dismissed:
            self._balloon_id = self._root.after(self.BALLOON_DURATION, self._dismiss_balloon)

    def _new_turn(self, voice_chat=False):
        self._cancel_dismiss()
        self._reset_slides()
        self._user_text = self._reply_text = ""
        self._voice_turn = voice_chat
        self._turn_finished = False
        self._text_dismissed = False

    def _set_state(self, state):
        self._explicitly_hidden = False
        if state in self._MASCOT_STILL_STATES:
            # A replaced or failed engine never reports that the voices
            # stopped; nobody is talking until a new one says so.
            self.__dict__.setdefault("_talking", set()).clear()
        if state == "processing":
            self._voice_heard = False  # a new input: pulse until she speaks
        self._state = state

    def show_loading(self):
        self._set_state("loading")
        self._update_layout()

    def show_ready(self, *, continuous_listening=False):
        self._continuous_listening = continuous_listening
        if self._user_text or self._reply_text:
            self._finish_turn()
        self._set_state("ready")
        self._update_layout()

    def show_recording(self, draft_text="", voice_chat=False):
        # Continuous listening resumes with an empty draft after TTS. Keep
        # that turn through its normal delay, until actual new speech arrives.
        finishing = self._voice_turn and self._state in ("speaking", "processing")
        if finishing and not draft_text:
            self._finish_turn()
        elif draft_text and (self._turn_finished or self._state != "recording"):
            self._new_turn(voice_chat)
        elif self._state != "recording" and (not self._turn_finished or not self._voice_turn):
            self._new_turn(voice_chat)
        self._set_state("recording")
        if draft_text:
            self._user_text = draft_text
        self._update_layout()

    def show_processing(self, text="", voice_chat=False):
        self._set_state("processing")
        if text.strip():
            if self._turn_finished:
                self._new_turn(voice_chat)
            self._voice_turn = voice_chat
            self._user_text = text
            self._cancel_dismiss()
        self._update_layout()

    def show_result(self, text, *, continuous_listening=False):
        self._continuous_listening = continuous_listening
        if self._voice_turn and not self._turn_finished:
            self._reply_text = text
        else:
            if self._turn_finished:
                self._new_turn()
            self._user_text = text
        self._set_state("ready")
        self._finish_turn()
        self._update_layout()

    def show_speaking(self, text, seconds=None):
        """seconds: how long the audio of its newest sentence plays; the
        text then slides at the pace of her voice (None: no audio)."""
        self._cancel_dismiss()
        self._voice_turn = True
        self._reply_text = text
        self._pace = seconds if seconds and seconds > 0 else None
        self._slide_held = False
        self._set_state("speaking")
        self._update_layout()

    def show_notice(self, text):
        """Notices belong to the tray; never overwrite conversation content."""

    def show_error(self):
        if self._state in ("speaking", "processing") and (self._user_text or self._reply_text):
            self._finish_turn()
        self._set_state("error")
        self._update_layout()

    def set_voice_mode(self, enabled: bool):
        """Keep the mode visible even between turns."""
        if self._voice_mode != enabled:
            self._voice_mode = enabled
            self._update_layout()

    def set_show_balloon(self, enabled):
        """The Settings checkbox controls the in-window text area."""
        self._show_balloon = enabled
        self._update_layout()

    def set_balloon_font_size(self, size):
        self.BALLOON_FONT_SIZE = max(10, min(size, 32))
        self._reset_slides()
        self._update_layout()

    def _dismiss_balloon(self):
        # Retain the turn internally so another streamed sentence cannot
        # reopen text that the user just dismissed.
        self._cancel_dismiss()
        self._text_dismissed = True
        self._reset_slides()
        self._update_layout()

    def hide(self):
        """Hide until the next state notification, unless the microphone is open."""
        if self._state == "recording" or self._continuous_listening:
            return
        self._cancel_dismiss()
        self._text_dismissed = True
        self._reset_slides()
        self._set_state("ready")
        self._explicitly_hidden = True
        self._update_layout(force_hidden=True)
