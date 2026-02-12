"""iPhone Dynamic Island-style floating overlay for dictation state.

Uses PIL 2x supersampled rendering via ui.glass for anti-aliased shapes.
"""

import time
import tkinter as tk
from collections import deque

from ui.glass import (
    TRANSPARENT_COLOR, PillCache, composite_on_transparent, pil_to_photo,
    render_button, render_dot, render_icon_mic, render_icon_stop,
    render_waveform, render_pill, _hex_to_rgba,
)

# Color used for window transparency (never appears in UI)
_TRANSPARENT = TRANSPARENT_COLOR


class OverlayWindow:
    """Always-visible floating pill that shows dictation state.

    Compact capsule when idle, expands on hover and during recording.
    Shows live audio waveform bars while recording.
    """

    # --- Dimensions ---
    COMPACT_W = 150
    COMPACT_H = 38
    HOVER_H = 54
    EXPANDED_W = 300
    EXPANDED_H = 70
    RADIUS = 19  # half compact height -> perfect capsule ends
    MARGIN = 14  # left/right padding for button and waveform
    BTN_CX = 26  # button center x (MARGIN + button radius approx)
    BORDER = 2   # border thickness

    # --- iOS-inspired dark palette ---
    BG = "#0A0A0A"
    BG_HOVER = "#0A0A0A"
    BORDER_COLOR = "#333333"
    TEXT = "#FFFFFF"
    TEXT_DIM = "#8E8E93"
    GREEN = "#30D158"
    RED = "#FF453A"
    AMBER = "#FF9F0A"
    GRAY = "#48484A"
    WAVE_COLOR = "#FF6961"

    # The dot button hit area (left side of the pill)
    _DOT_HIT_X = 38

    # --- Balloon dimensions ---
    BALLOON_MAX_W = 360
    BALLOON_PAD = 12
    BALLOON_GAP = 20      # gap between pill and balloon
    BALLOON_RADIUS = 12
    BALLOON_FONT_SIZE = 16
    BALLOON_DURATION = 6000  # ms before auto-dismiss

    def __init__(self, root, on_toggle=None):
        """
        Args:
            root: Parent Tk/CTk window.
            on_toggle: Callback() to toggle recording on dot click.
        """
        self._root = root
        self._on_toggle = on_toggle
        self._win: tk.Toplevel | None = None
        self._canvas: tk.Canvas | None = None

        # Animated size
        self._cur_w = float(self.COMPACT_W)
        self._cur_h = float(self.COMPACT_H)
        self._tgt_w = float(self.COMPACT_W)
        self._tgt_h = float(self.COMPACT_H)

        # Position (None = auto-center, set on first drag)
        self._pos_x: int | None = None
        self._pos_y: int = 10

        # Drag state
        self._drag_offset_x = 0
        self._drag_offset_y = 0

        # State
        self._state = "loading"
        self._hover = False
        self._rec_start = 0.0
        self._result_text = ""
        self._wave: deque[float] = deque(maxlen=26)

        # Timer IDs
        self._anim_id = None
        self._rec_id = None
        self._auto_hide_id = None

        # Balloon
        self._balloon_win: tk.Toplevel | None = None
        self._balloon_id = None
        self._show_balloon = True

        # PIL rendering state
        self._pill_cache = PillCache()
        self._photo_refs: list = []  # prevent GC of PhotoImages
        self._mic_button_photo = None
        self._stop_button_photo = None
        self._pre_render_buttons()

        self._build()

    def _pre_render_buttons(self):
        """Pre-render mic and stop button images."""
        # Green mic button: ring + dark bg + mic icon
        mic_icon = render_icon_mic(12, self.GREEN)
        mic_btn = render_button(24, self.GREEN, "#1B3A20", mic_icon)
        self._mic_button_img = mic_btn

        # Red stop button: ring + dark bg + stop icon
        stop_icon = render_icon_stop(12, self.RED)
        stop_btn = render_button(26, self.RED, "#3A1B1B", stop_icon)
        self._stop_button_img = stop_btn

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

        self._canvas = tk.Canvas(
            self._win, bg=_TRANSPARENT, highlightthickness=0,
            width=self.COMPACT_W, height=self.COMPACT_H,
        )
        self._canvas.pack(fill="both", expand=True)

        self._canvas.bind("<Enter>", lambda e: self._set_hover(True))
        self._canvas.bind("<Leave>", self._on_leave)
        self._canvas.bind("<Motion>", self._on_mouse_move)
        self._canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self._canvas.bind("<B1-Motion>", self._on_drag_move)

        self._win.update_idletasks()
        self._position()
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
            hwnd = ctypes.windll.user32.GetParent(self._win.winfo_id())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style = (style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW
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

    def _position(self):
        """Position the window, keeping it centered on its anchor point."""
        w = int(self._cur_w)
        h = int(self._cur_h)
        if self._pos_x is None:
            # Default: center horizontally at top of screen
            x = (self._get_screen_width() - w) // 2
        else:
            # Keep centered on the user's chosen position
            x = self._pos_x - w // 2
        y = self._pos_y
        self._win.geometry(f"{w}x{h}+{x}+{y}")
        self._canvas.configure(width=w, height=h)

    # --- PIL-based drawing ------------------------------------------------
    # All elements are composited onto the pill RGBA image first, then
    # the final result is flattened onto TRANSPARENT_COLOR and placed as
    # one single canvas image.  This avoids the transparent-color
    # punch-through that happens when layering separate images.

    @staticmethod
    def _paste_centered(base, overlay, cx, cy):
        """Alpha-composite overlay onto base, centered at (cx, cy)."""
        ow, oh = overlay.size
        x = cx - ow // 2
        y = cy - oh // 2
        base.alpha_composite(overlay, (x, y))

    def _redraw(self):
        from PIL import Image
        c = self._canvas
        c.delete("all")
        self._photo_refs.clear()
        w, h = int(self._cur_w), int(self._cur_h)
        r = min(self.RADIUS, h // 2)
        mid = h // 2

        # Start with glass pill as the base image
        border_rgba = _hex_to_rgba(self.BORDER_COLOR)
        pill = self._pill_cache.get(
            w, h, radius=r, border_color_rgba=border_rgba,
            border_width=self.BORDER)
        # Work on a copy so the cache stays clean
        frame = pill.copy()

        bx = self.BTN_CX
        margin = self.MARGIN
        text_items = []  # (x, y, text, fill, font, anchor) — drawn after image

        if self._state == "loading":
            dot = render_dot(10, _hex_to_rgba(self.GRAY))
            self._paste_centered(frame, dot, bx, mid)
            text_items.append((w // 2 + 6, mid, "Loading...",
                               self.TEXT_DIM, ("Segoe UI", 10), "center"))

        elif self._state == "ready":
            if self._hover and w > self.COMPACT_W + 20:
                self._paste_centered(frame, self._mic_button_img, bx, mid)
                text_items.append((w // 2 + 10, mid, "Start recording",
                                   self.TEXT_DIM, ("Segoe UI", 10), "center"))
            else:
                dot = render_dot(10, _hex_to_rgba(self.GREEN))
                self._paste_centered(frame, dot, bx, mid)
                text_items.append((w // 2 + 6, mid, "Ready",
                                   self.TEXT, ("Segoe UI", 10, "bold"), "center"))

        elif self._state == "recording":
            self._paste_centered(frame, self._stop_button_img, bx, mid)
            # Waveform
            wave_start = 114
            wave_end = w - margin
            levels = list(self._wave)
            if levels:
                wave_w = wave_end - wave_start
                wave_h = h - 8
                if wave_w > 0 and wave_h > 0:
                    wave_img = render_waveform(wave_w, wave_h, levels,
                                               bar_width=3, gap=3,
                                               color=self.WAVE_COLOR)
                    frame.alpha_composite(wave_img,
                                          (wave_start, mid - wave_h // 2))
            # Timer text
            elapsed = time.time() - self._rec_start
            text_items.append((50, mid, f"{elapsed:.1f}s",
                               self.TEXT, ("Segoe UI", 14, "bold"), "w"))

        elif self._state == "processing":
            dot = render_dot(10, _hex_to_rgba(self.AMBER))
            self._paste_centered(frame, dot, bx, mid)
            text_items.append((w // 2 + 6, mid, "Transcribing...",
                               self.TEXT, ("Segoe UI", 10, "bold"), "center"))

        elif self._state == "result":
            dot = render_dot(10, _hex_to_rgba(self.GREEN))
            self._paste_centered(frame, dot, bx, mid)
            text_items.append((w // 2 + 6, mid, "Done",
                               self.TEXT, ("Segoe UI", 10, "bold"), "center"))

        elif self._state == "error":
            dot = render_dot(10, _hex_to_rgba(self.RED))
            self._paste_centered(frame, dot, bx, mid)
            text_items.append((w // 2 + 6, mid, "Error",
                               self.RED, ("Segoe UI", 10, "bold"), "center"))

        # Flatten to RGB on transparent background and place as one image
        composited = composite_on_transparent(frame)
        photo = pil_to_photo(composited)
        self._photo_refs.append(photo)
        c.create_image(0, 0, image=photo, anchor="nw")

        # Draw text on top (ClearType AA handled by tkinter)
        for tx, ty, text, fill, font, anchor in text_items:
            c.create_text(tx, ty, text=text, fill=fill, font=font,
                          anchor=anchor)

    # --- Drag to reposition -----------------------------------------------

    def _on_drag_start(self, event):
        """Click on dot -> toggle recording. Click elsewhere -> start drag."""
        if event.x <= self._DOT_HIT_X and self._state in ("ready", "recording"):
            # Clicked the dot button — toggle recording
            self._drag_is_click = True
            if self._on_toggle:
                self._on_toggle()
            return
        self._drag_is_click = False
        self._drag_offset_x = event.x
        self._drag_offset_y = event.y

    def _on_drag_move(self, event):
        """Move window to follow the mouse."""
        if getattr(self, "_drag_is_click", False):
            return  # Was a button click, not a drag
        x = self._win.winfo_x() + event.x - self._drag_offset_x
        y = self._win.winfo_y() + event.y - self._drag_offset_y
        w = int(self._cur_w)
        # Store the center x so resizing stays anchored to the drag position
        self._pos_x = x + w // 2
        self._pos_y = y
        self._win.geometry(f"+{x}+{y}")

    # --- Hover & cursor ---------------------------------------------------

    def _on_leave(self, event):
        self._canvas.configure(cursor="")
        self._set_hover(False)

    def _on_mouse_move(self, event):
        """Show hand cursor when over the dot button area."""
        if event.x <= self._DOT_HIT_X and self._state in ("ready", "recording"):
            self._canvas.configure(cursor="hand2")
        else:
            self._canvas.configure(cursor="")

    def _set_hover(self, hovered):
        if self._state != "ready":
            return
        self._hover = hovered
        if hovered:
            self._animate(self.EXPANDED_W, self.HOVER_H)
        else:
            self._animate(self.COMPACT_W, self.COMPACT_H)

    # --- Animation --------------------------------------------------------

    def _animate(self, tw, th):
        self._tgt_w = float(tw)
        self._tgt_h = float(th)
        if self._anim_id:
            self._root.after_cancel(self._anim_id)
        self._anim_tick()

    def _anim_tick(self):
        dw = self._tgt_w - self._cur_w
        dh = self._tgt_h - self._cur_h
        if abs(dw) < 1.5 and abs(dh) < 1.5:
            self._cur_w = self._tgt_w
            self._cur_h = self._tgt_h
            self._position()
            self._redraw()
            self._anim_id = None
            return
        self._cur_w += dw * 0.25
        self._cur_h += dh * 0.25
        self._position()
        self._redraw()
        self._anim_id = self._root.after(16, self._anim_tick)

    # --- Public state API -------------------------------------------------

    def _cancel_timers(self):
        for attr in ("_anim_id", "_rec_id", "_auto_hide_id"):
            tid = getattr(self, attr, None)
            if tid:
                self._root.after_cancel(tid)
                setattr(self, attr, None)
        self._dismiss_balloon()

    def show_loading(self):
        """Gray dot — model loading."""
        self._cancel_timers()
        self._state = "loading"
        self._hover = False
        self._animate(self.COMPACT_W, self.COMPACT_H)

    def show_ready(self):
        """Green dot — idle, waiting for hotkey."""
        self._cancel_timers()
        self._state = "ready"
        self._hover = False
        self._animate(self.COMPACT_W, self.COMPACT_H)

    def show_recording(self):
        """Expand with pulsing red dot, timer, and waveform."""
        self._cancel_timers()
        self._state = "recording"
        self._rec_start = time.time()
        self._wave.clear()
        self._animate(self.EXPANDED_W, self.EXPANDED_H)
        self._tick_recording()

    def _tick_recording(self):
        if self._state != "recording":
            return
        self._redraw()
        self._rec_id = self._root.after(80, self._tick_recording)

    def show_processing(self):
        """Amber dot — transcribing."""
        self._cancel_timers()
        self._state = "processing"
        self._animate(self.EXPANDED_W, self.COMPACT_H)

    def show_result(self, text: str):
        """Green dot — transcription done. Text shown in balloon only."""
        self._cancel_timers()
        self._state = "result"
        self._result_text = text
        self._animate(self.COMPACT_W, self.COMPACT_H)
        self._auto_hide_id = self._root.after(2500, self.show_ready)
        if self._show_balloon and text.strip():
            self._show_balloon_popup(text)

    def show_error(self):
        """Red dot — error state."""
        self._cancel_timers()
        self._state = "error"
        self._animate(self.COMPACT_W, self.COMPACT_H)

    def update_audio_level(self, level: float):
        """Feed audio level (0.0-1.0) for waveform visualization."""
        # Light smoothing — responsive to speech but not jittery
        if self._wave:
            prev = self._wave[-1]
            level = prev * 0.2 + level * 0.8
        self._wave.append(level)

    def set_show_balloon(self, enabled: bool):
        """Enable or disable the text balloon under the notch."""
        self._show_balloon = enabled

    def set_balloon_font_size(self, size: int):
        """Set the font size for balloon text."""
        self.BALLOON_FONT_SIZE = max(10, min(size, 32))

    # --- Balloon popup ----------------------------------------------------

    def _show_balloon_popup(self, text: str):
        """Show a dark tooltip-style balloon below the pill with full text."""
        self._dismiss_balloon()

        bw = self._balloon_win = tk.Toplevel(self._root)
        bw.overrideredirect(True)
        bw.attributes("-topmost", True)
        bw.configure(bg=_TRANSPARENT)
        try:
            bw.attributes("-transparentcolor", _TRANSPARENT)
        except Exception:
            pass

        pad = self.BALLOON_PAD
        r = self.BALLOON_RADIUS
        max_w = self.BALLOON_MAX_W

        canvas = tk.Canvas(bw, bg=_TRANSPARENT, highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        # Measure text to determine balloon size
        tmp_id = canvas.create_text(
            0, 0, text=text, font=("Segoe UI", self.BALLOON_FONT_SIZE),
            width=max_w - 2 * pad, anchor="nw",
        )
        bbox = canvas.bbox(tmp_id)
        canvas.delete(tmp_id)

        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        bw_w = text_w + 2 * pad
        bw_h = text_h + 2 * pad
        # Clamp minimum width
        bw_w = max(bw_w, 120)

        canvas.configure(width=bw_w, height=bw_h)

        # Glass pill background for balloon (PIL-rendered)
        border_rgba = _hex_to_rgba(self.BORDER_COLOR)
        bg_rgba = _hex_to_rgba(self.BG)
        balloon_pill = render_pill(
            bw_w, bw_h, radius=r,
            border_color_rgba=border_rgba,
            bg_top=bg_rgba[:3], bg_bottom=bg_rgba[:3],
            border_width=self.BORDER)
        composited = composite_on_transparent(balloon_pill)
        photo = pil_to_photo(composited)
        # Store reference to prevent GC
        canvas._photo_ref = photo
        canvas.create_image(0, 0, image=photo, anchor="nw")

        # Draw text
        canvas.create_text(
            pad, pad, text=text, font=("Segoe UI", self.BALLOON_FONT_SIZE),
            fill=self.TEXT, width=max_w - 2 * pad, anchor="nw",
        )

        # Click anywhere on balloon to dismiss
        canvas.bind("<ButtonPress-1>", lambda e: self._dismiss_balloon())

        # Position below the pill
        pill_x = self._win.winfo_x()
        pill_y = self._win.winfo_y()
        pill_w = int(self._cur_w)
        pill_h = int(self._cur_h)
        bx = pill_x + (pill_w - bw_w) // 2
        by = pill_y + pill_h + self.BALLOON_GAP
        bw.geometry(f"{bw_w}x{bw_h}+{bx}+{by}")

        # Apply no-focus flags
        try:
            import ctypes
            GWL_EXSTYLE = -20
            WS_EX_NOACTIVATE = 0x08000000
            WS_EX_TOOLWINDOW = 0x00000080
            WS_EX_APPWINDOW = 0x00040000
            hwnd = ctypes.windll.user32.GetParent(bw.winfo_id())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style = (style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        except Exception:
            pass

        # Auto-dismiss
        self._balloon_id = self._root.after(self.BALLOON_DURATION, self._dismiss_balloon)

    def _dismiss_balloon(self):
        """Destroy the balloon popup if it exists."""
        if self._balloon_id:
            self._root.after_cancel(self._balloon_id)
            self._balloon_id = None
        if self._balloon_win:
            try:
                self._balloon_win.destroy()
            except Exception:
                pass
            self._balloon_win = None

    def hide(self):
        """Dynamic Island is always visible — hide means go to ready."""
        self.show_ready()
