"""Settings dialog: neutral light/dark surfaces (following the OS) and one gold accent."""

import customtkinter as ctk
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog


# Palette: neutral surfaces, one gold accent (the mascot's hair and the app icon).
# Resolved to plain colors for the OS theme when the window opens, so it follows
# the system setting whatever the overlay's appearance mode is. Contrast checked:
# text >= 6:1, controls >= 3:1, dark text on gold >= 8:1.
_PALETTES = {
    "light": dict(
        CONTENT_BG="#FBFAF7", CARD_BG="#F4F1E9", CARD_BORDER="#E2DDD0",
        INPUT_BG="#FFFFFF", INPUT_BORDER="#8F8878",
        SECTION_TEXT="#1C1A14", DESC_TEXT="#5C5748",
        ACCENT="#E8A317", ACCENT_HOVER="#D18F0C", ACCENT_TEXT="#1A1305",
        ACCENT_CTRL="#9A6400", ACCENT_CTRL_HOVER="#7F5200", ACCENT_CTRL_TEXT="#FFFFFF",
        BTN_SEC_BG="#EDE9DD", BTN_SEC_HOVER="#E2DDD0", BTN_SEC_TEXT="#2B281F",
        POPUP_BG="#FFFFFF", POPUP_HOVER="#F4E7C8",
        TAB_TRACK="#E2DDD0", TAB_SELECTED="#FFFFFF",
        BADGE_OK="#0F6B4A", BADGE_DL="#5C5748",
        ERROR_TEXT="#B3261E", WARN_TEXT="#8A4B00", OK_TEXT="#0F6B4A",
    ),
    "dark": dict(
        CONTENT_BG="#141A26", CARD_BG="#1B2233", CARD_BORDER="#2C3650",
        INPUT_BG="#10151F", INPUT_BORDER="#6B7794",
        SECTION_TEXT="#EEF1F7", DESC_TEXT="#A3ACC2",
        ACCENT="#F2B53A", ACCENT_HOVER="#FFC85C", ACCENT_TEXT="#1A1305",
        ACCENT_CTRL="#F2B53A", ACCENT_CTRL_HOVER="#FFC85C", ACCENT_CTRL_TEXT="#1A1305",
        BTN_SEC_BG="#232C42", BTN_SEC_HOVER="#2C3650", BTN_SEC_TEXT="#E4E8F2",
        POPUP_BG="#1B2233", POPUP_HOVER="#2C3650",
        TAB_TRACK="#0E131C", TAB_SELECTED="#2C3650",
        BADGE_OK="#5BD6A0", BADGE_DL="#A3ACC2",
        ERROR_TEXT="#FF9B94", WARN_TEXT="#FBBF24", OK_TEXT="#5BD6A0",
    ),
}
_IS_DARK = False

# The names the rest of the module reads. _apply_palette() rebinds them for the
# OS theme; they start as the light palette so the module works unopened.
_CONTENT_BG = _PALETTES["light"]["CONTENT_BG"]
_CARD_BG = _PALETTES["light"]["CARD_BG"]
_CARD_BORDER = _PALETTES["light"]["CARD_BORDER"]
_INPUT_BG = _PALETTES["light"]["INPUT_BG"]
_INPUT_BORDER = _PALETTES["light"]["INPUT_BORDER"]
_SECTION_TEXT = _PALETTES["light"]["SECTION_TEXT"]
_DESC_TEXT = _PALETTES["light"]["DESC_TEXT"]
_ACCENT = _PALETTES["light"]["ACCENT"]
_ACCENT_HOVER = _PALETTES["light"]["ACCENT_HOVER"]
_ACCENT_TEXT = _PALETTES["light"]["ACCENT_TEXT"]
_ACCENT_CTRL = _PALETTES["light"]["ACCENT_CTRL"]
_ACCENT_CTRL_HOVER = _PALETTES["light"]["ACCENT_CTRL_HOVER"]
_ACCENT_CTRL_TEXT = _PALETTES["light"]["ACCENT_CTRL_TEXT"]
_BTN_SEC_BG = _PALETTES["light"]["BTN_SEC_BG"]
_BTN_SEC_HOVER = _PALETTES["light"]["BTN_SEC_HOVER"]
_BTN_SEC_TEXT = _PALETTES["light"]["BTN_SEC_TEXT"]
_POPUP_BG = _PALETTES["light"]["POPUP_BG"]
_POPUP_HOVER = _PALETTES["light"]["POPUP_HOVER"]
_TAB_TRACK = _PALETTES["light"]["TAB_TRACK"]
_TAB_SELECTED = _PALETTES["light"]["TAB_SELECTED"]
_BADGE_OK = _PALETTES["light"]["BADGE_OK"]
_BADGE_DL = _PALETTES["light"]["BADGE_DL"]
_ERROR_TEXT = _PALETTES["light"]["ERROR_TEXT"]
_WARN_TEXT = _PALETTES["light"]["WARN_TEXT"]
_OK_TEXT = _PALETTES["light"]["OK_TEXT"]


def _apply_palette(dark: bool):
    """Point the module's color names at the light or the dark palette."""
    global _IS_DARK
    _IS_DARK = dark
    for name, value in _PALETTES["dark" if dark else "light"].items():
        globals()["_" + name] = value


def _os_is_dark() -> bool:
    try:
        import darkdetect
        return bool(darkdetect.isDark())
    except Exception:
        return False


# app.py reports status in the overlay's bright colors, which fail contrast
# on this window: show the palette's readable equivalent instead.
def _status_color(color):
    return {"#FF453A": _ERROR_TEXT, "#FF9F0A": _WARN_TEXT, "#30D158": _OK_TEXT,
            "gray60": _DESC_TEXT}.get(color, color)

_SYSTEM_DEFAULT_MIC = "System default"
_HOTKEY_HINT = "Hold it to talk. A quick tap starts or stops listening."

# Close to square, opened in the middle of the screen.
_WIN_W, _WIN_H = 660, 700

# tts_voice null: Chatterbox's own voice.
_DEFAULT_VOICE = "Chatterbox default"
_TTS_ENGINES = {
    "chatterbox": "Chatterbox (clones the voice)",
    "qwen3": "Qwen3-TTS (experimental, clones the voice)",
    "kokoro": "Kokoro (fastest, pt-BR voice pf_dora)",
    # Choosing it accepts the Coqui Public Model License (non-commercial use).
    "xtts": "XTTS-v2 (fast, clones the voice, non-commercial)",
}
# Engines that only clone a reference WAV: "Chatterbox default" is no voice.
_CLONE_ONLY_ENGINES = {
    "qwen3": "Qwen3 requires a reference voice (tts_voice); select a WAV voice with a "
             "matching UTF-8 .txt transcript instead of Chatterbox default.",
    "xtts": "XTTS-v2 requires a reference voice (tts_voice); select a WAV voice "
            "instead of Chatterbox default.",
}
_TTS_ENGINE_KEYS = {label: key for key, label in _TTS_ENGINES.items()}


_TK_MODIFIER_KEYS = {"Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R",
                     "Meta_L", "Meta_R", "Super_L", "Super_R", "Win_L", "Win_R",
                     "ISO_Level3_Shift", "Caps_Lock", "Num_Lock"}
_TK_KEY_NAMES = {
    "space": "space", "Return": "enter", "Tab": "tab", "BackSpace": "backspace",
    "Delete": "delete", "Insert": "insert", "Home": "home", "End": "end",
    "Prior": "page up", "Next": "page down", "Up": "up", "Down": "down",
    "Left": "left", "Right": "right", "Pause": "pause",
}
# Alt is 0x20000 on Windows (where 0x8 is Num Lock) and Mod1 (0x8) elsewhere;
# Control and Shift are portable.
_STATE_SHIFT, _STATE_CONTROL = 0x1, 0x4
_STATE_ALT = 0x20000 if sys.platform == "win32" else 0x8


def hotkey_from_event(event):
    """Hotkey text for a key press ("ctrl+shift+f5"), or an error message.

    Returns (hotkey, None) when complete, (None, None) while only modifiers
    are down, and (None, message) when the key cannot be a hotkey."""
    keysym = event.keysym
    if keysym in _TK_MODIFIER_KEYS:
        return None, None
    if len(keysym) == 1 and keysym.isalnum():
        key = keysym.lower()
    elif keysym.startswith("F") and keysym[1:].isdigit() and 1 <= int(keysym[1:]) <= 24:
        key = keysym.lower()
    elif keysym in _TK_KEY_NAMES:
        key = _TK_KEY_NAMES[keysym]
    else:
        return None, "That key can't be used. Try a letter, number, F-key or arrow."
    mods = []
    if event.state & _STATE_CONTROL:
        mods.append("ctrl")
    if event.state & _STATE_ALT:
        mods.append("alt")
    if event.state & _STATE_SHIFT:
        mods.append("shift")
    # Shift alone still types a character; F-keys are safe on their own.
    if not ({"ctrl", "alt"} & set(mods)) and not (key.startswith("f") and key[1:].isdigit()):
        return None, "Hold Ctrl or Alt with it, so typing is not interrupted."
    return "+".join(mods + [key]), None


def _make_wide_dropdown(om, values, on_select):
    """Patch CTkOptionMenu to show a full-width popup aligned with the field.

    Keyboard: Space/Enter/Down on the focused field opens it; Up/Down move,
    Enter picks, Esc closes (and leaves the Settings window open)."""
    _state = {"win": None, "buttons": [], "index": 0}

    def _open():
        if om.cget("state") == "disabled":
            return
        if _state["win"] and _state["win"].winfo_exists():
            _close()
            return

        x = om.winfo_rootx()
        y = om.winfo_rooty() + om.winfo_height() + 2
        w = om.winfo_width()

        win = tk.Toplevel(om)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        _state["win"] = win

        frame = ctk.CTkFrame(win, fg_color=_POPUP_BG,
                             border_color=_CARD_BORDER,
                             border_width=1, corner_radius=6)
        frame.pack(fill="both", expand=True)

        _state["buttons"] = []
        for val in values:
            btn = ctk.CTkButton(
                frame, text=val, anchor="w", corner_radius=4,
                fg_color="transparent", hover_color=_POPUP_HOVER,
                text_color=_SECTION_TEXT, height=28,
                font=ctk.CTkFont(size=13),
                command=lambda v=val: _select(v),
            )
            btn.pack(fill="x", padx=3, pady=1)
            _state["buttons"].append(btn)
        current = om.get()
        _state["index"] = values.index(current) if current in values else 0
        _highlight()

        h = min(len(values) * 30 + 10, 500)
        win.geometry(f"{w}x{h}+{x}+{y}")

        _grab(win)
        win.bind("<Button-1>", _on_click)
        win.bind("<Escape>", lambda e: (_close(), "break")[1])
        win.bind("<Up>", lambda e: _move(-1))
        win.bind("<Down>", lambda e: _move(1))
        win.bind("<Return>", lambda e: _pick())
        win.bind("<space>", lambda e: _pick())
        try:
            win.focus_force()
        except tk.TclError:
            pass

    def _highlight():
        for i, btn in enumerate(_state["buttons"]):
            btn.configure(fg_color=_POPUP_HOVER if i == _state["index"] else "transparent")

    def _move(step):
        count = len(_state["buttons"])
        if count:
            _state["index"] = (_state["index"] + step) % count
            _highlight()
        return "break"

    def _pick():
        if _state["buttons"]:
            _select(values[_state["index"]])
        return "break"

    def _grab(win, tries=25):
        # On X11 the popup maps after this returns, and a grab on an unmapped
        # window fails: retry until it shows (Windows maps it at once).
        if _state["win"] is not win:
            return
        try:
            win.grab_set()
        except tk.TclError:
            if tries > 1:
                win.after(20, _grab, win, tries - 1)

    def _select(value):
        om.set(value)
        _close()
        if on_select:
            on_select(value)

    def _on_click(event):
        if not _state["win"]:
            return
        win = _state["win"]
        wx, wy = win.winfo_rootx(), win.winfo_rooty()
        ww, wh = win.winfo_width(), win.winfo_height()
        if not (wx <= event.x_root <= wx + ww and wy <= event.y_root <= wy + wh):
            _close()

    def _close():
        if _state["win"] and _state["win"].winfo_exists():
            try:
                _state["win"].grab_release()
            except Exception:
                pass
            _state["win"].destroy()
        _state["win"] = None

    om._open_dropdown_menu = _open

    # Reachable with Tab, and visible while focused.
    canvas = getattr(om, "_canvas", None)
    if canvas is not None:
        canvas.configure(takefocus=True, highlightthickness=2,
                         highlightbackground=_CONTENT_BG, highlightcolor=_ACCENT_CTRL)
        for seq in ("<space>", "<Return>", "<Down>"):
            canvas.bind(seq, lambda e: (_open(), "break")[1])


class SettingsWindow:
    """Modal-ish settings dialog with language-first model selection."""

    def __init__(self, root: ctk.CTk, config: dict, input_devices: list[dict],
                 on_apply=None, version_text=""):
        """
        Args:
            root: Parent CTk window.
            config: Current config dict (will be copied, not mutated directly).
            input_devices: List of dicts with 'index' and 'name' keys.
            on_apply: Callback(new_config) called when user clicks Apply.
        """
        self._root = root
        self._config = dict(config)
        self._input_devices = input_devices
        self._on_apply = on_apply
        self._version_text = version_text
        self._version_label = None
        self._win: ctk.CTkToplevel | None = None
        self._model_radio_var = ctk.StringVar(value=config.get("model_size", "turbo"))
        self._model_rows_frame: ctk.CTkFrame | None = None
        self._recording_hotkey = False
        self._status_label: ctk.CTkLabel | None = None
        self._apply_btn: ctk.CTkButton | None = None

    @property
    def is_open(self) -> bool:
        return self._win is not None and self._win.winfo_exists()

    def _centered_geometry(self, width: int, height: int) -> str:
        """Geometry string that centers the window, shrunk to fit small screens."""
        screen_w = self._win.winfo_screenwidth()
        screen_h = self._win.winfo_screenheight()
        width = min(width, screen_w - 40)
        height = min(height, screen_h - 80)  # taskbar and title bar
        x = (screen_w - width) // 2
        y = max(0, (screen_h - height) // 2 - 20)
        return f"{width}x{height}+{x}+{y}"

    def show(self):
        if self._win is not None and self._win.winfo_exists():
            self._win.focus_force()
            return

        _apply_palette(_os_is_dark())
        self._win = ctk.CTkToplevel(self._root)
        self._win.title("Débora Whisper — Settings")
        self._win.geometry(self._centered_geometry(_WIN_W, _WIN_H))
        self._win.resizable(False, False)
        self._win.configure(fg_color=_CONTENT_BG)

        self._win.update_idletasks()

        # Title bar color via DWM
        from debora_whisper.ui.glass import set_title_bar_color
        set_title_bar_color(self._win, _CONTENT_BG, _SECTION_TEXT, dark=_IS_DARK)

        # Window icon — set _iconbitmap_method_called to prevent CTkToplevel
        # from overriding with its default icon at 200ms
        from debora_whisper.ui.icons import render_app_icon
        from PIL import ImageTk
        self._icon_photo = ImageTk.PhotoImage(render_app_icon(32))
        self._win._iconbitmap_method_called = True
        self._win.iconphoto(False, self._icon_photo)
        # Windows resets the title-bar icon to customtkinter's blue default just
        # after the window opens; set ours again once that has happened.
        for delay in (250, 600):
            self._win.after(delay, self._reapply_window_icon)

        # Raise it once; staying topmost would float it over the window the
        # user is dictating into.
        self._win.attributes("-topmost", True)
        self._win.after(300, lambda: self._win.winfo_exists()
                        and self._win.attributes("-topmost", False))
        self._win.bind("<Escape>", lambda e: self._win.destroy())
        self._win.bind("<Control-Return>", lambda e: self._apply())

        # --- Title block: the mascot's gold glyph and the name, on the surface ---
        header = ctk.CTkFrame(self._win, fg_color=_CONTENT_BG, corner_radius=0, height=56)
        header.pack(fill="x")
        header.pack_propagate(False)

        icon_img = render_app_icon(26, color=_ACCENT_CTRL)
        self._logo_photo = ctk.CTkImage(light_image=icon_img, dark_image=icon_img,
                                        size=(26, 26))
        ctk.CTkLabel(header, image=self._logo_photo, text="",
                     fg_color="transparent").pack(side="left", padx=(20, 0))
        ctk.CTkLabel(
            header, text="Débora Whisper", fg_color="transparent",
            font=ctk.CTkFont(size=17, weight="bold"), text_color=_SECTION_TEXT,
        ).pack(side="left", padx=(10, 0))
        ctk.CTkFrame(self._win, fg_color=_CARD_BORDER, height=1,
                     corner_radius=0).pack(fill="x")

        # Shared dropdown styling
        dd_opts = dict(
            fg_color=_INPUT_BG, button_color=_BTN_SEC_BG,
            button_hover_color=_BTN_SEC_HOVER, text_color=_SECTION_TEXT,
            text_color_disabled=_DESC_TEXT,
            dropdown_fg_color=_POPUP_BG, dropdown_hover_color=_POPUP_HOVER,
            dropdown_text_color=_SECTION_TEXT,
        )

        # --- Bottom bar (packed first to reserve space) ---
        bottom = ctk.CTkFrame(self._win, fg_color=_CONTENT_BG, corner_radius=0)
        bottom.pack(side="bottom", fill="x")

        self._version_label = ctk.CTkLabel(
            bottom, text=self._version_text, font=ctk.CTkFont(size=11),
            text_color=_DESC_TEXT, fg_color="transparent", wraplength=620,
        )
        self._version_label.pack(padx=20, pady=(4, 0), anchor="w")

        self._status_label = ctk.CTkLabel(
            bottom, text="", font=ctk.CTkFont(size=12),
            text_color=_DESC_TEXT, fg_color="transparent",
        )
        self._status_label.pack(padx=20, pady=(4, 0), anchor="w")

        btn_frame = ctk.CTkFrame(bottom, fg_color="transparent")
        btn_frame.pack(padx=20, pady=(4, 12), fill="x")
        self._apply_btn = ctk.CTkButton(
            btn_frame, text="Apply", command=self._apply, width=100,
            fg_color=_ACCENT, hover_color=_ACCENT_HOVER, text_color=_ACCENT_TEXT,
            text_color_disabled=_DESC_TEXT, font=ctk.CTkFont(weight="bold"),
        )
        self._apply_btn.pack(side="right", padx=(8, 0))
        ctk.CTkButton(
            btn_frame, text="Close", command=self._win.destroy, width=100,
            fg_color=_BTN_SEC_BG, hover_color=_BTN_SEC_HOVER,
            text_color=_BTN_SEC_TEXT, text_color_disabled=_DESC_TEXT, border_color=_CARD_BORDER, border_width=1,
        ).pack(side="right")

        # --- Tabs: one job per tab, so each shows a handful of controls ---
        tabs = ctk.CTkTabview(
            self._win, fg_color=_CONTENT_BG, corner_radius=8,
            segmented_button_fg_color=_TAB_TRACK,
            segmented_button_selected_color=_TAB_SELECTED,
            segmented_button_selected_hover_color=_TAB_SELECTED,
            segmented_button_unselected_color=_TAB_TRACK,
            segmented_button_unselected_hover_color=_BTN_SEC_HOVER,
            text_color=_SECTION_TEXT,
        )
        # Full-width tab bar with room to click.
        tabs._segmented_button.configure(
            height=36, width=_WIN_W - 60, font=ctk.CTkFont(size=13, weight="bold"))
        tabs.pack(fill="both", expand=True, padx=12, pady=(8, 0))
        self._tabs = tabs
        tab_names = ("Dictation", "Voice chat", "Claude Code")
        pages = []
        for name in tab_names:
            tabs.add(name)
            page = ctk.CTkScrollableFrame(
                tabs.tab(name), fg_color=_CONTENT_BG,
                scrollbar_button_color=_CARD_BORDER,
                scrollbar_button_hover_color=_INPUT_BORDER,
            )
            page.pack(fill="both", expand=True)
            pages.append(page)
        dictation, voice_chat, claude = pages
        self._bind_tab_keys(tab_names)

        def heading(parent, text, first=False):
            ctk.CTkLabel(
                parent, text=text, fg_color="transparent",
                font=ctk.CTkFont(size=15, weight="bold"), text_color=_SECTION_TEXT,
            ).pack(padx=8, pady=(4 if first else 22, 0), anchor="w")

        def label(parent, text):
            ctk.CTkLabel(
                parent, text=text, fg_color="transparent",
                font=ctk.CTkFont(size=13, weight="bold"), text_color=_SECTION_TEXT,
            ).pack(padx=8, pady=(12, 0), anchor="w")

        def hint(parent, text):
            ctk.CTkLabel(
                parent, text=text, font=ctk.CTkFont(size=11), text_color=_DESC_TEXT,
                fg_color="transparent", justify="left", wraplength=560,
            ).pack(padx=8, pady=(3, 0), anchor="w")

        field = {"padx": 8, "pady": (4, 0), "fill": "x"}

        chk_opts = dict(
            fg_color=_ACCENT_CTRL, border_color=_INPUT_BORDER,
            hover_color=_ACCENT_CTRL_HOVER, text_color=_SECTION_TEXT,
            text_color_disabled=_DESC_TEXT, checkmark_color=_ACCENT_CTRL_TEXT,
        )

        def checkbox(parent, text, var):
            ctk.CTkCheckBox(parent, text=text, variable=var, **chk_opts
                            ).pack(padx=8, pady=(8, 0), anchor="w")

        # ===== Dictation: Speech =====
        heading(dictation, "Speech", first=True)
        label(dictation, "Language")

        from debora_whisper.dictation_engine import LANGUAGES
        self._lang_codes = list(LANGUAGES.keys())
        lang_display = list(LANGUAGES.values())
        current_lang = self._config.get("language", "en")
        current_display = LANGUAGES.get(current_lang, current_lang)

        self._lang_var = ctk.StringVar(value=current_display)
        self._lang_dropdown = ctk.CTkOptionMenu(
            dictation, values=lang_display, variable=self._lang_var,
            command=self._on_language_change, **dd_opts,
        )
        self._lang_dropdown.pack(**field)

        label(dictation, "Model")
        hint(dictation, "The tag shows where it runs. Débora switches to another "
                        "device if that one is unavailable.")
        self._model_rows_frame = ctk.CTkFrame(dictation, fg_color="transparent")
        self._model_rows_frame.pack(padx=8, pady=(4, 0), fill="x")
        self._build_model_list()
        self._model_radio_var.trace_add("write", self._refresh_model_cards)

        # ===== Dictation: Input and feedback =====
        heading(dictation, "Input and feedback")
        label(dictation, "Dictation hotkey")
        hotkey_row = ctk.CTkFrame(dictation, fg_color="transparent")
        hotkey_row.pack(padx=8, pady=(4, 0), fill="x")
        self._hotkey_var = ctk.StringVar(value=self._config.get("hotkey", "ctrl+space"))
        self._hotkey_entry = ctk.CTkEntry(
            hotkey_row, textvariable=self._hotkey_var,
            fg_color=_INPUT_BG, border_color=_INPUT_BORDER,
            text_color=_SECTION_TEXT,
        )
        self._hotkey_entry.pack(side="left", fill="x", expand=True)
        self._record_btn = ctk.CTkButton(
            hotkey_row, text="Record", command=self._toggle_hotkey_recording, width=75,
            fg_color=_BTN_SEC_BG, hover_color=_BTN_SEC_HOVER, text_color=_BTN_SEC_TEXT,
            text_color_disabled=_DESC_TEXT,
        )
        self._record_btn.pack(side="left", padx=(6, 0))
        ctk.CTkButton(
            hotkey_row, text="Reset", command=self._reset_hotkey, width=75,
            fg_color=_BTN_SEC_BG, hover_color=_BTN_SEC_HOVER, text_color=_BTN_SEC_TEXT,
            text_color_disabled=_DESC_TEXT,
        ).pack(side="left", padx=(6, 0))
        self._hotkey_hint = ctk.CTkLabel(
            dictation, text=_HOTKEY_HINT, font=ctk.CTkFont(size=11),
            text_color=_DESC_TEXT, fg_color="transparent", justify="left", wraplength=560)
        self._hotkey_hint.pack(padx=8, pady=(3, 0), anchor="w")

        label(dictation, "Microphone")
        mic_names = [_SYSTEM_DEFAULT_MIC]
        for d in self._input_devices:
            # Windows lists a device once per host API: show each name once.
            if d["name"] not in mic_names:
                mic_names.append(d["name"])
        saved_mic = self._config.get("input_device")
        if saved_mic and saved_mic not in mic_names:
            mic_names.append(saved_mic)  # unplugged: keep the choice visible
        self._mic_var = ctk.StringVar(value=saved_mic or _SYSTEM_DEFAULT_MIC)
        self._mic_dropdown = ctk.CTkOptionMenu(
            dictation, values=mic_names, variable=self._mic_var, **dd_opts,
        )
        self._mic_dropdown.pack(**field)

        self._beep_var = ctk.BooleanVar(value=self._config.get("beep_on_start", True))
        checkbox(dictation, "Play a sound when listening starts and stops", self._beep_var)
        self._inline_drafts_var = ctk.BooleanVar(value=self._config.get("inline_drafts", False))
        checkbox(dictation, "Type live drafts as you speak (can garble some editors)",
                 self._inline_drafts_var)
        self._balloon_var = ctk.BooleanVar(value=self._config.get("show_balloon", True))
        checkbox(dictation, "Show the text balloon beside the mascot", self._balloon_var)

        label(dictation, "Balloon text size")
        font_sizes = ["12", "14", "16", "18", "20", "24"]
        self._font_size_var = ctk.StringVar(
            value=str(self._config.get("balloon_font_size", 16)))
        self._font_size_dropdown = ctk.CTkOptionMenu(
            dictation, values=font_sizes, variable=self._font_size_var, **dd_opts,
        )
        self._font_size_dropdown.pack(**field)
        ctk.CTkFrame(dictation, fg_color="transparent", height=8).pack()

        # ===== Voice chat =====
        heading(voice_chat, "Conversation", first=True)
        self._voice_chat_var = ctk.BooleanVar(value=self._config.get("voice_chat", False))
        checkbox(voice_chat, "Voice chat: talk and hear the reply", self._voice_chat_var)

        label(voice_chat, "Replies come from")
        backends = {"local": "This computer (Qwen)", "claude": "Claude Code"}
        self._backend_var = ctk.StringVar(
            value=backends.get(self._config.get("voice_chat_backend"), "This computer (Qwen)"))
        self._backend_dropdown = ctk.CTkOptionMenu(
            voice_chat, values=list(backends.values()), variable=self._backend_var,
            command=self._on_backend_change, **dd_opts,
        )
        self._backend_dropdown.pack(**field)
        hint(voice_chat, "Claude Code has its own options on the Claude Code tab.")

        heading(voice_chat, "Spoken reply")
        label(voice_chat, "Voice engine")
        self._tts_engine_var = ctk.StringVar(
            value=_TTS_ENGINES[self._config.get("tts_backend", "chatterbox")])
        self._tts_engine_dropdown = ctk.CTkOptionMenu(
            voice_chat, values=list(_TTS_ENGINES.values()), variable=self._tts_engine_var,
            command=self._on_tts_engine_change, **dd_opts,
        )
        self._tts_engine_dropdown.pack(**field)
        hint(voice_chat, "Kokoro has its own voice, so the voice list is off. "
                         "The first use downloads the model.\n"
                         "Choosing XTTS-v2 accepts the Coqui Public Model License "
                         "(non-commercial use only).")

        label(voice_chat, "Voice")
        voice_row = ctk.CTkFrame(voice_chat, fg_color="transparent")
        voice_row.pack(padx=8, pady=(4, 0), fill="x")
        voice_names = self._voice_choices()
        self._voice_var = ctk.StringVar(
            value=self._config.get("tts_voice") or _DEFAULT_VOICE)
        self._voice_dropdown = ctk.CTkOptionMenu(
            voice_row, values=voice_names, variable=self._voice_var, **dd_opts,
        )
        self._voice_dropdown.pack(side="left", fill="x", expand=True)
        self._on_tts_engine_change(self._tts_engine_var.get())
        ctk.CTkButton(
            voice_row, text="Open voices folder", command=self._open_voices_folder, width=130,
            fg_color=_BTN_SEC_BG, hover_color=_BTN_SEC_HOVER,
            text_color=_BTN_SEC_TEXT, text_color_disabled=_DESC_TEXT, border_color=_CARD_BORDER, border_width=1,
        ).pack(side="left", padx=(8, 0))
        hint(voice_chat, "To add a voice, save a ~10 s recording of one speaker as "
                     "<name>.wav in the voices folder.")

        # ===== Claude Code =====
        heading(claude, "Typing into Claude Code", first=True)
        self._enter_var = ctk.BooleanVar(value=self._config.get("auto_enter", False))
        checkbox(claude, "Press Enter after typing, to send the prompt", self._enter_var)

        heading(claude, "Voice chat with Claude")
        self._claude_note = ctk.CTkLabel(
            claude, text="", font=ctk.CTkFont(size=12), text_color=_DESC_TEXT,
            fg_color="transparent", justify="left", wraplength=560,
        )
        self._claude_note.pack(padx=8, pady=(4, 0), anchor="w")

        label(claude, "Working folder")
        folder_row = ctk.CTkFrame(claude, fg_color="transparent")
        folder_row.pack(padx=8, pady=(4, 0), fill="x")
        self._harness_cwd = self._config.get("harness_cwd")
        self._harness_folder_var = ctk.StringVar(
            value=str(Path(self._harness_cwd).expanduser()) if self._harness_cwd else str(Path.home()))
        ctk.CTkEntry(
            folder_row, textvariable=self._harness_folder_var, state="readonly",
            fg_color=_INPUT_BG, border_color=_INPUT_BORDER, text_color=_DESC_TEXT,
        ).pack(side="left", fill="x", expand=True)
        self._browse_btn = ctk.CTkButton(
            folder_row, text="Browse…", command=self._browse_harness_folder, width=75,
            fg_color=_BTN_SEC_BG, hover_color=_BTN_SEC_HOVER, text_color=_BTN_SEC_TEXT, text_color_disabled=_DESC_TEXT,
        )
        self._browse_btn.pack(side="left", padx=(6, 0))
        self._home_btn = ctk.CTkButton(
            folder_row, text="Use home", command=self._use_harness_home, width=75,
            fg_color=_BTN_SEC_BG, hover_color=_BTN_SEC_HOVER, text_color=_BTN_SEC_TEXT, text_color_disabled=_DESC_TEXT,
        )
        self._home_btn.pack(side="left", padx=(6, 0))

        hint(claude, "Claude Code runs in this folder. Default: your home folder.")

        label(claude, "Conversation name")
        self._harness_session_name_var = ctk.StringVar(
            value=self._config.get("harness_session_name", "Débora Whisper"))
        self._session_entry = ctk.CTkEntry(
            claude, textvariable=self._harness_session_name_var,
            fg_color=_INPUT_BG, border_color=_INPUT_BORDER, text_color=_SECTION_TEXT,
        )
        self._session_entry.pack(**field)
        hint(claude, "How the conversation is named in Claude.")

        self._harness_new_session_var = ctk.BooleanVar(
            value=self._config.get("harness_new_session_on_start", False))
        self._new_session_chk = ctk.CTkCheckBox(
            claude, text="New Claude conversation on each Débora start",
            variable=self._harness_new_session_var, **chk_opts)
        self._new_session_chk.pack(padx=8, pady=(12, 8), anchor="w")
        self._on_backend_change(self._backend_var.get())

        # Patch dropdowns to show full-width popups
        _make_wide_dropdown(self._lang_dropdown, lang_display,
                            self._on_language_change)
        _make_wide_dropdown(self._mic_dropdown, mic_names, None)
        _make_wide_dropdown(self._font_size_dropdown, font_sizes, None)
        _make_wide_dropdown(self._voice_dropdown, voice_names, None)
        _make_wide_dropdown(self._tts_engine_dropdown, list(_TTS_ENGINES.values()),
                            self._on_tts_engine_change)
        _make_wide_dropdown(self._backend_dropdown, list(backends.values()),
                            self._on_backend_change)

    def _reapply_window_icon(self):
        if self.is_open:
            self._win.iconphoto(False, self._icon_photo)

    def _bind_tab_keys(self, names):
        """Ctrl+1..3 jump to a tab; Ctrl+Tab / Ctrl+Shift+Tab cycle."""
        def go(i):
            self._tabs.set(names[i % len(names)])
            return "break"

        def step(delta):
            return go(names.index(self._tabs.get()) + delta)

        for i in range(len(names)):
            self._win.bind(f"<Control-Key-{i + 1}>", lambda e, i=i: go(i))
        self._win.bind("<Control-Tab>", lambda e: step(1))
        self._win.bind("<Control-Shift-Tab>", lambda e: step(-1))
        self._win.bind("<Control-ISO_Left_Tab>", lambda e: step(-1))

    def _toggle_hotkey_recording(self):
        if self._recording_hotkey:
            self._stop_hotkey_recording()
            return
        self._recording_hotkey = True
        self._record_btn.configure(text="Cancel")
        self._hotkey_entry.configure(state="readonly", border_color=_ACCENT_CTRL)
        self._hotkey_hint.configure(text="Press the keys you want, or Esc to cancel.",
                                    text_color=_SECTION_TEXT)
        inner = self._hotkey_entry._entry
        inner.bind("<KeyPress>", self._on_hotkey_key)
        inner.focus_set()

    def _stop_hotkey_recording(self, hint_text=None, error=False):
        self._recording_hotkey = False
        self._record_btn.configure(text="Record")
        self._hotkey_entry.configure(state="normal", border_color=_INPUT_BORDER)
        self._hotkey_entry._entry.unbind("<KeyPress>")
        self._hotkey_hint.configure(text=hint_text or _HOTKEY_HINT,
                                    text_color=_ERROR_TEXT if error else _DESC_TEXT)

    def _on_hotkey_key(self, event):
        if event.keysym == "Escape":
            self._stop_hotkey_recording()
            return "break"
        hotkey, problem = hotkey_from_event(event)
        if problem:
            self._hotkey_hint.configure(text=problem, text_color=_ERROR_TEXT)
        elif hotkey:
            self._hotkey_var.set(hotkey)
            self._stop_hotkey_recording()
        return "break"

    def _reset_hotkey(self):
        if self._recording_hotkey:
            self._stop_hotkey_recording()
        from debora_whisper.dictation_engine import DEFAULT_CONFIG
        self._hotkey_var.set(DEFAULT_CONFIG["hotkey"])

    def _show_tab_for(self, message: str):
        """Bring the tab that holds the field an error is about to the front."""
        tabs = getattr(self, "_tabs", None)
        if tabs is None:
            return
        text = message.lower()
        if "hotkey" in text:
            tabs.set("Dictation")
        elif "reference voice" in text:
            tabs.set("Voice chat")

    def _on_backend_change(self, label):
        """Claude Code harness options only matter when Claude answers voice chat."""
        claude = label == "Claude Code"
        state = "normal" if claude else "disabled"
        for widget in (self._browse_btn, self._home_btn, self._session_entry,
                       self._new_session_chk):
            widget.configure(state=state)
        self._claude_note.configure(
            text="Claude Code answers voice chat." if claude else
                 "Off for now: this computer answers voice chat. "
                 "Switch it on the Voice chat tab to use these.")

    def _on_tts_engine_change(self, label):
        self._voice_dropdown.configure(
            state="disabled" if _TTS_ENGINE_KEYS[label] == "kokoro" else "normal")

    def _browse_harness_folder(self):
        folder = filedialog.askdirectory(parent=self._win, title="Working folder",
                                         initialdir=self._harness_folder_var.get())
        if folder:
            self._harness_cwd = folder
            self._harness_folder_var.set(folder)

    def _use_harness_home(self):
        self._harness_cwd = None
        self._harness_folder_var.set(str(Path.home()))

    def _voice_choices(self) -> list[str]:
        """Chatterbox's own voice, then each <name>.wav in the voices folder,
        and the configured voice even when it is a path elsewhere."""
        from debora_whisper.voice_chat import list_voices
        names = [_DEFAULT_VOICE] + list_voices()
        current = self._config.get("tts_voice")
        if current and current not in names:
            names.append(current)
        return names

    def _open_voices_folder(self):
        from debora_whisper import paths, platforms
        try:
            paths.VOICES_DIR.mkdir(parents=True, exist_ok=True)
            platforms.current().open_folder(paths.VOICES_DIR)
        except OSError as e:
            self.update_status(f"Cannot open {paths.VOICES_DIR}: {e}", "#FF453A")

    def update_status(self, text: str, color: str = "gray60"):
        """Update the status label (called from app.py on state changes)."""
        if self._status_label and self._win and self._win.winfo_exists():
            self._status_label.configure(
                text=text, text_color=_status_color(color))

    def update_version(self, text: str):
        """Follow update checks while the window is open."""
        self._version_text = text
        if self._version_label and self.is_open:
            self._version_label.configure(text=text)

    def set_voice_chat(self, enabled: bool):
        """Follow a switch made from the tray, so Apply does not undo it."""
        self._voice_chat_var.set(enabled)

    def set_apply_enabled(self, enabled: bool):
        """Enable/disable the Apply button during model loading."""
        if self._apply_btn and self._win and self._win.winfo_exists():
            self._apply_btn.configure(state="normal" if enabled else "disabled")

    def _get_selected_lang_code(self) -> str:
        """Convert display name back to language code."""
        from debora_whisper.dictation_engine import LANGUAGES
        display = self._lang_var.get()
        for code, name in LANGUAGES.items():
            if name == display:
                return code
        return "en"

    def _on_language_change(self, value: str):
        """Rebuild model list when language changes."""
        current_model = self._model_radio_var.get()
        self._build_model_list()
        from debora_whisper.dictation_engine import get_models_for_language
        lang_code = self._get_selected_lang_code()
        available = get_models_for_language(lang_code)
        if current_model not in available:
            first = next(iter(available))
            self._model_radio_var.set(first)

    def _build_model_list(self):
        """Build/rebuild the model radio button list based on selected language."""
        from debora_whisper.dictation_engine import (
            get_models_for_language, is_model_downloaded,
        )

        for child in self._model_rows_frame.winfo_children():
            child.destroy()
        self._model_cards = {}

        lang_code = self._get_selected_lang_code()
        models = get_models_for_language(lang_code)

        device_labels = {"NPU": "NPU", "GPU": "GPU", "CPU": "CPU"}

        for key, info in models.items():
            card = ctk.CTkFrame(
                self._model_rows_frame,
                fg_color=_CARD_BG, border_color=_CARD_BORDER,
                border_width=1, corner_radius=8,
            )
            card.pack(fill="x", pady=2)
            self._model_cards[key] = card

            # Top row: radio button + badge
            top_row = ctk.CTkFrame(card, fg_color="transparent")
            top_row.pack(fill="x", padx=10, pady=(6, 0))

            device = info["preferred_device"]
            device_tag = device_labels.get(device, device)
            radio = ctk.CTkRadioButton(
                top_row, text=f"{key}  [{device_tag}]",
                variable=self._model_radio_var, value=key,
                font=ctk.CTkFont(size=13, weight="bold"),
                fg_color=_ACCENT_CTRL, border_color=_INPUT_BORDER,
                hover_color=_ACCENT_CTRL_HOVER, text_color=_SECTION_TEXT,
            )
            radio.pack(side="left")

            downloaded = is_model_downloaded(key)
            badge_text = "Downloaded" if downloaded else "Not downloaded"
            badge_color = _BADGE_OK if downloaded else _BADGE_DL
            ctk.CTkLabel(
                top_row, text=badge_text,
                font=ctk.CTkFont(size=11), text_color=badge_color,
                fg_color="transparent",
            ).pack(side="right")

            # Bottom row: description, wrapped so long ones are never clipped
            ctk.CTkLabel(
                card, text=info["description"],
                font=ctk.CTkFont(size=11), text_color=_DESC_TEXT,
                fg_color="transparent", anchor="w", justify="left", wraplength=470,
            ).pack(fill="x", padx=(40, 10), pady=(0, 6), anchor="w")
        self._refresh_model_cards()

    def _refresh_model_cards(self, *_):
        """Outline the chosen model in gold."""
        chosen = self._model_radio_var.get()
        for key, card in getattr(self, "_model_cards", {}).items():
            try:
                card.configure(border_color=_ACCENT_CTRL if key == chosen else _CARD_BORDER)
            except tk.TclError:
                pass  # card destroyed by a language change

    def _get_new_config(self) -> dict:
        """Build new config dict from current UI state."""
        new_config = dict(self._config)
        new_config["model_size"] = self._model_radio_var.get()
        new_config["hotkey"] = self._hotkey_var.get().strip()
        new_config["language"] = self._get_selected_lang_code()
        new_config["beep_on_start"] = self._beep_var.get()
        new_config["auto_enter"] = self._enter_var.get()
        new_config["inline_drafts"] = self._inline_drafts_var.get()
        new_config["voice_chat"] = self._voice_chat_var.get()
        new_config["voice_chat_backend"] = "claude" if self._backend_var.get() == "Claude Code" else "local"
        new_config["harness_cwd"] = self._harness_cwd
        new_config["harness_new_session_on_start"] = self._harness_new_session_var.get()
        new_config["harness_session_name"] = self._harness_session_name_var.get()
        voice = self._voice_var.get()
        new_config["tts_backend"] = _TTS_ENGINE_KEYS[self._tts_engine_var.get()]
        new_config["tts_voice"] = None if voice == _DEFAULT_VOICE else voice
        new_config["show_balloon"] = self._balloon_var.get()
        mic = self._mic_var.get()
        new_config["input_device"] = None if mic == _SYSTEM_DEFAULT_MIC else mic
        try:
            new_config["balloon_font_size"] = int(self._font_size_var.get())
        except ValueError:
            new_config["balloon_font_size"] = 16
        from debora_whisper.dictation_engine import detect_devices, select_device
        new_config["device"] = select_device(new_config, detect_devices()) or "CPU"
        return new_config

    def _apply(self):
        """Apply settings without closing the window."""
        new_config = self._get_new_config()
        from debora_whisper.dictation_engine import validate_config, validate_hotkey
        try:
            validate_config(new_config)
            validate_hotkey(new_config)
            if (new_config.get("tts_backend") in _CLONE_ONLY_ENGINES
                    and new_config.get("tts_voice") is None
                    and not new_config.get("tts_server_command")):
                raise ValueError(_CLONE_ONLY_ENGINES[new_config["tts_backend"]])
        except ValueError as e:
            self._show_tab_for(str(e))
            self.update_status(str(e), "#FF453A")
            return
        self._config = dict(new_config)
        if self._on_apply:
            self._on_apply(new_config)
