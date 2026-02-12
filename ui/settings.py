"""Settings dialog — Docker Desktop inspired clean theme with Intel blue."""

import customtkinter as ctk
import tkinter as tk


# Intel blue palette
_HEADER_BG = "#0071C5"         # Intel blue header bar
_CONTENT_BG = "#FFFFFF"        # White content area
_SECTION_TEXT = "#111827"      # Near-black for section labels
_DESC_TEXT = "#6B7280"         # Gray-500 for descriptions
_CARD_BG = "#F9FAFB"          # Gray-50 for model cards
_CARD_BORDER = "#E5E7EB"      # Gray-200 border
_INPUT_BG = "#F9FAFB"         # Gray-50 for inputs
_INPUT_BORDER = "#D1D5DB"     # Gray-300
_ACCENT = "#0071C5"           # Intel blue accent
_ACCENT_HOVER = "#005A9E"     # Darker Intel blue on hover
_BTN_SEC_BG = "#F3F4F6"       # Gray-100 secondary button
_BTN_SEC_HOVER = "#E5E7EB"    # Gray-200
_BTN_SEC_TEXT = "#374151"      # Gray-700
_BADGE_OK = "#10B981"          # Emerald-500 for "Downloaded"
_BADGE_DL = "#9CA3AF"          # Gray-400 for "Download"


def _make_wide_dropdown(om, values, on_select):
    """Patch CTkOptionMenu to show a full-width popup aligned with the field."""
    _state = {"win": None}

    def _open():
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

        frame = ctk.CTkFrame(win, fg_color="#FFFFFF",
                             border_color=_CARD_BORDER,
                             border_width=1, corner_radius=6)
        frame.pack(fill="both", expand=True)

        for val in values:
            ctk.CTkButton(
                frame, text=val, anchor="w", corner_radius=4,
                fg_color="transparent", hover_color="#E8F0FE",
                text_color=_SECTION_TEXT, height=28,
                font=ctk.CTkFont(size=13),
                command=lambda v=val: _select(v),
            ).pack(fill="x", padx=3, pady=1)

        h = min(len(values) * 30 + 10, 500)
        win.geometry(f"{w}x{h}+{x}+{y}")

        win.grab_set()
        win.bind("<Button-1>", _on_click)
        win.bind("<Escape>", lambda e: _close())

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


class SettingsWindow:
    """Modal-ish settings dialog with language-first model selection."""

    def __init__(self, root: ctk.CTk, config: dict, input_devices: list[dict],
                 on_apply=None):
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
        self._win: ctk.CTkToplevel | None = None
        self._model_radio_var = ctk.StringVar(value=config.get("model_size", "base"))
        self._model_rows_frame: ctk.CTkFrame | None = None
        self._status_label: ctk.CTkLabel | None = None
        self._apply_btn: ctk.CTkButton | None = None

    @property
    def is_open(self) -> bool:
        return self._win is not None and self._win.winfo_exists()

    def show(self):
        if self._win is not None and self._win.winfo_exists():
            self._win.focus_force()
            return

        self._win = ctk.CTkToplevel(self._root)
        self._win.title("NPU Dictation — Settings")
        self._win.geometry("540x780")
        self._win.resizable(False, False)
        self._win.configure(fg_color=_CONTENT_BG)

        self._win.update_idletasks()

        # Title bar color via DWM
        from ui.glass import set_title_bar_color
        set_title_bar_color(self._win, _HEADER_BG, "#FFFFFF")

        # Window icon — set _iconbitmap_method_called to prevent CTkToplevel
        # from overriding with its default icon at 200ms
        from ui.icons import render_app_icon
        from PIL import ImageTk
        self._icon_photo = ImageTk.PhotoImage(render_app_icon(32))
        self._win._iconbitmap_method_called = True
        self._win.iconphoto(False, self._icon_photo)

        self._win.attributes("-topmost", True)

        # --- Blue header banner ---
        header = ctk.CTkFrame(self._win, fg_color=_HEADER_BG, corner_radius=0,
                              height=48)
        header.pack(fill="x")
        header.pack_propagate(False)

        icon_img = render_app_icon(22, color="#FFFFFF")
        self._logo_photo = ctk.CTkImage(light_image=icon_img, size=(22, 22))
        ctk.CTkLabel(header, image=self._logo_photo, text="",
                     fg_color="transparent").pack(side="left", padx=(16, 0))
        ctk.CTkLabel(
            header, text="NPU Dictation", fg_color="transparent",
            font=ctk.CTkFont(size=14, weight="bold"), text_color="#FFFFFF",
        ).pack(side="left", padx=(8, 0))
        ctk.CTkLabel(
            header, text="Settings", fg_color="transparent",
            font=ctk.CTkFont(size=12), text_color="#8ECAE6",
        ).pack(side="left", padx=(8, 0))

        pad = {"padx": 20, "pady": (10, 0)}

        # Shared dropdown styling
        dd_opts = dict(
            fg_color=_INPUT_BG, button_color=_ACCENT,
            button_hover_color=_ACCENT_HOVER, text_color=_SECTION_TEXT,
            dropdown_fg_color="#FFFFFF", dropdown_hover_color=_CARD_BG,
            dropdown_text_color=_SECTION_TEXT,
        )

        # --- Bottom bar (packed first to reserve space) ---
        bottom = ctk.CTkFrame(self._win, fg_color=_CONTENT_BG)
        bottom.pack(side="bottom", fill="x")

        self._status_label = ctk.CTkLabel(
            bottom, text="", font=ctk.CTkFont(size=12),
            text_color=_DESC_TEXT, fg_color="transparent",
        )
        self._status_label.pack(padx=20, pady=(4, 0), anchor="w")

        btn_frame = ctk.CTkFrame(bottom, fg_color="transparent")
        btn_frame.pack(padx=20, pady=(4, 12), fill="x")
        self._apply_btn = ctk.CTkButton(
            btn_frame, text="Apply", command=self._apply, width=100,
            fg_color=_ACCENT, hover_color=_ACCENT_HOVER, text_color="#FFFFFF",
        )
        self._apply_btn.pack(side="right", padx=(8, 0))
        ctk.CTkButton(
            btn_frame, text="Close", command=self._win.destroy, width=100,
            fg_color=_BTN_SEC_BG, hover_color=_BTN_SEC_HOVER,
            text_color=_BTN_SEC_TEXT, border_color=_CARD_BORDER, border_width=1,
        ).pack(side="right")

        # --- Scrollable content area ---
        scroll = ctk.CTkScrollableFrame(
            self._win, fg_color=_CONTENT_BG,
            scrollbar_button_color=_CARD_BORDER,
            scrollbar_button_hover_color=_INPUT_BORDER,
        )
        scroll.pack(fill="both", expand=True)

        # --- Language dropdown ---
        ctk.CTkLabel(
            scroll, text="Language", fg_color="transparent",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=_SECTION_TEXT,
        ).pack(**pad, anchor="w")

        from dictation_engine import LANGUAGES
        self._lang_codes = list(LANGUAGES.keys())
        lang_display = list(LANGUAGES.values())
        current_lang = self._config.get("language", "en")
        current_display = LANGUAGES.get(current_lang, current_lang)

        self._lang_var = ctk.StringVar(value=current_display)
        self._lang_dropdown = ctk.CTkOptionMenu(
            scroll, values=lang_display, variable=self._lang_var,
            command=self._on_language_change, **dd_opts,
        )
        self._lang_dropdown.pack(padx=20, pady=(4, 0), fill="x")

        # --- Model radio list ---
        ctk.CTkLabel(
            scroll, text="Model", fg_color="transparent",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=_SECTION_TEXT,
        ).pack(**pad, anchor="w")

        self._model_rows_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        self._model_rows_frame.pack(padx=20, pady=(4, 0), fill="x")

        self._build_model_list()

        # --- Hotkey ---
        ctk.CTkLabel(
            scroll, text="Hotkey", fg_color="transparent",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=_SECTION_TEXT,
        ).pack(**pad, anchor="w")
        self._hotkey_var = ctk.StringVar(value=self._config.get("hotkey", "ctrl+space"))
        ctk.CTkEntry(
            scroll, textvariable=self._hotkey_var,
            fg_color=_INPUT_BG, border_color=_INPUT_BORDER,
            text_color=_SECTION_TEXT,
        ).pack(padx=20, pady=(4, 0), fill="x")

        # --- Microphone ---
        ctk.CTkLabel(
            scroll, text="Microphone", fg_color="transparent",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=_SECTION_TEXT,
        ).pack(**pad, anchor="w")
        mic_names = ([d["name"] for d in self._input_devices]
                     if self._input_devices else ["(default)"])
        self._mic_var = ctk.StringVar(value=mic_names[0])
        self._mic_dropdown = ctk.CTkOptionMenu(
            scroll, values=mic_names, variable=self._mic_var, **dd_opts,
        )
        self._mic_dropdown.pack(padx=20, pady=(4, 0), fill="x")

        # --- Balloon font size ---
        ctk.CTkLabel(
            scroll, text="Balloon font size", fg_color="transparent",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=_SECTION_TEXT,
        ).pack(**pad, anchor="w")
        font_sizes = ["12", "14", "16", "18", "20", "24"]
        self._font_size_var = ctk.StringVar(
            value=str(self._config.get("balloon_font_size", 16)))
        self._font_size_dropdown = ctk.CTkOptionMenu(
            scroll, values=font_sizes, variable=self._font_size_var, **dd_opts,
        )
        self._font_size_dropdown.pack(padx=20, pady=(4, 0), fill="x")

        # --- Toggles ---
        toggles_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        toggles_frame.pack(padx=20, pady=(10, 10), fill="x")

        chk_opts = dict(
            fg_color=_ACCENT, border_color=_INPUT_BORDER,
            hover_color=_ACCENT_HOVER, text_color=_SECTION_TEXT,
            checkmark_color="#FFFFFF",
        )
        self._beep_var = ctk.BooleanVar(value=self._config.get("beep_on_start", True))
        ctk.CTkCheckBox(
            toggles_frame, text="Beep on start/stop",
            variable=self._beep_var, **chk_opts,
        ).pack(anchor="w", pady=2)

        self._enter_var = ctk.BooleanVar(value=self._config.get("auto_enter", False))
        ctk.CTkCheckBox(
            toggles_frame, text="Auto-Enter after paste (Claude Code mode)",
            variable=self._enter_var, **chk_opts,
        ).pack(anchor="w", pady=2)

        self._balloon_var = ctk.BooleanVar(value=self._config.get("show_balloon", True))
        ctk.CTkCheckBox(
            toggles_frame, text="Show text balloon after transcription",
            variable=self._balloon_var, **chk_opts,
        ).pack(anchor="w", pady=2)

        # Patch dropdowns to show full-width popups
        _make_wide_dropdown(self._lang_dropdown, lang_display,
                            self._on_language_change)
        _make_wide_dropdown(self._mic_dropdown, mic_names, None)
        _make_wide_dropdown(self._font_size_dropdown, font_sizes, None)

    def update_status(self, text: str, color: str = "gray60"):
        """Update the status label (called from app.py on state changes)."""
        if self._status_label and self._win and self._win.winfo_exists():
            self._status_label.configure(text=text, text_color=color)

    def set_apply_enabled(self, enabled: bool):
        """Enable/disable the Apply button during model loading."""
        if self._apply_btn and self._win and self._win.winfo_exists():
            self._apply_btn.configure(state="normal" if enabled else "disabled")

    def _get_selected_lang_code(self) -> str:
        """Convert display name back to language code."""
        from dictation_engine import LANGUAGES
        display = self._lang_var.get()
        for code, name in LANGUAGES.items():
            if name == display:
                return code
        return "en"

    def _on_language_change(self, value: str):
        """Rebuild model list when language changes."""
        current_model = self._model_radio_var.get()
        self._build_model_list()
        from dictation_engine import get_models_for_language
        lang_code = self._get_selected_lang_code()
        available = get_models_for_language(lang_code)
        if current_model not in available:
            first = next(iter(available))
            self._model_radio_var.set(first)

    def _build_model_list(self):
        """Build/rebuild the model radio button list based on selected language."""
        from dictation_engine import (
            get_models_for_language, is_model_downloaded, MODEL_REGISTRY,
        )

        for child in self._model_rows_frame.winfo_children():
            child.destroy()

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

            # Top row: radio button + badge
            top_row = ctk.CTkFrame(card, fg_color="transparent")
            top_row.pack(fill="x", padx=10, pady=(6, 0))

            device = info["preferred_device"]
            device_tag = device_labels.get(device, device)
            radio = ctk.CTkRadioButton(
                top_row, text=f"{key}  [{device_tag}]",
                variable=self._model_radio_var, value=key,
                font=ctk.CTkFont(size=13, weight="bold"),
                fg_color=_ACCENT, border_color=_INPUT_BORDER,
                hover_color=_ACCENT_HOVER, text_color=_SECTION_TEXT,
            )
            radio.pack(side="left")

            downloaded = is_model_downloaded(key)
            badge_text = "Downloaded" if downloaded else "Download"
            badge_color = _BADGE_OK if downloaded else _BADGE_DL
            ctk.CTkLabel(
                top_row, text=badge_text,
                font=ctk.CTkFont(size=11), text_color=badge_color,
                fg_color="transparent",
            ).pack(side="right")

            # Bottom row: description (full width, no truncation)
            ctk.CTkLabel(
                card, text=info["description"],
                font=ctk.CTkFont(size=11), text_color=_DESC_TEXT,
                fg_color="transparent", anchor="w",
            ).pack(fill="x", padx=(40, 10), pady=(0, 6), anchor="w")

    def _get_new_config(self) -> dict:
        """Build new config dict from current UI state."""
        new_config = dict(self._config)
        new_config["model_size"] = self._model_radio_var.get()
        new_config["hotkey"] = self._hotkey_var.get().strip()
        new_config["language"] = self._get_selected_lang_code()
        new_config["beep_on_start"] = self._beep_var.get()
        new_config["auto_enter"] = self._enter_var.get()
        new_config["show_balloon"] = self._balloon_var.get()
        try:
            new_config["balloon_font_size"] = int(self._font_size_var.get())
        except ValueError:
            new_config["balloon_font_size"] = 16
        from dictation_engine import MODEL_REGISTRY
        model_info = MODEL_REGISTRY.get(new_config["model_size"], {})
        new_config["device"] = model_info.get("preferred_device", "CPU")
        return new_config

    def _apply(self):
        """Apply settings without closing the window."""
        new_config = self._get_new_config()
        self._config = dict(new_config)
        if self._on_apply:
            self._on_apply(new_config)
