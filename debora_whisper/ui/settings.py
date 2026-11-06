"""Settings dialog — Docker Desktop inspired clean theme with Intel blue."""

import customtkinter as ctk
import tkinter as tk
from pathlib import Path
from tkinter import filedialog


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


def _make_wide_dropdown(om, values, on_select):
    """Patch CTkOptionMenu to show a full-width popup aligned with the field."""
    _state = {"win": None}

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

        _grab(win)
        win.bind("<Button-1>", _on_click)
        win.bind("<Escape>", lambda e: _close())

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
        self._win.title("Débora Whisper — Settings")
        self._win.geometry("540x780")
        self._win.resizable(False, False)
        self._win.configure(fg_color=_CONTENT_BG)

        self._win.update_idletasks()

        # Title bar color via DWM
        from debora_whisper.ui.glass import set_title_bar_color
        set_title_bar_color(self._win, _HEADER_BG, "#FFFFFF")

        # Window icon — set _iconbitmap_method_called to prevent CTkToplevel
        # from overriding with its default icon at 200ms
        from debora_whisper.ui.icons import render_app_icon
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
            header, text="Débora Whisper", fg_color="transparent",
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

        self._version_label = ctk.CTkLabel(
            bottom, text=self._version_text, font=ctk.CTkFont(size=11),
            text_color=_DESC_TEXT, fg_color="transparent", wraplength=500,
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

        from debora_whisper.dictation_engine import LANGUAGES
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

        # --- Voice chat voice ---
        ctk.CTkLabel(
            scroll, text="Voice chat voice", fg_color="transparent",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=_SECTION_TEXT,
        ).pack(**pad, anchor="w")
        ctk.CTkLabel(
            scroll, text="Voice engine", fg_color="transparent",
            font=ctk.CTkFont(size=12), text_color=_SECTION_TEXT,
        ).pack(padx=20, pady=(4, 0), anchor="w")
        self._tts_engine_var = ctk.StringVar(
            value=_TTS_ENGINES[self._config.get("tts_backend", "chatterbox")])
        self._tts_engine_dropdown = ctk.CTkOptionMenu(
            scroll, values=list(_TTS_ENGINES.values()), variable=self._tts_engine_var,
            command=self._on_tts_engine_change, **dd_opts,
        )
        self._tts_engine_dropdown.pack(padx=20, pady=(4, 0), fill="x")
        ctk.CTkLabel(
            scroll, text="Kokoro ignores the voice. First use downloads the model.\n"
                         "XTTS-v2 is licensed for non-commercial use only (Coqui CPML).",
            font=ctk.CTkFont(size=11), text_color=_DESC_TEXT, fg_color="transparent",
            justify="left",
        ).pack(padx=20, pady=(2, 0), anchor="w")
        voice_row = ctk.CTkFrame(scroll, fg_color="transparent")
        voice_row.pack(padx=20, pady=(4, 0), fill="x")
        voice_names = self._voice_choices()
        self._voice_var = ctk.StringVar(
            value=self._config.get("tts_voice") or _DEFAULT_VOICE)
        self._voice_dropdown = ctk.CTkOptionMenu(
            voice_row, values=voice_names, variable=self._voice_var, **dd_opts,
        )
        self._voice_dropdown.pack(side="left", fill="x", expand=True)
        self._on_tts_engine_change(self._tts_engine_var.get())
        ctk.CTkButton(
            voice_row, text="Open folder", command=self._open_voices_folder, width=100,
            fg_color=_BTN_SEC_BG, hover_color=_BTN_SEC_HOVER,
            text_color=_BTN_SEC_TEXT, border_color=_CARD_BORDER, border_width=1,
        ).pack(side="left", padx=(8, 0))
        ctk.CTkLabel(
            scroll, text="A ~10 s recording of one voice, saved as <name>.wav in that folder.",
            font=ctk.CTkFont(size=11), text_color=_DESC_TEXT, fg_color="transparent",
        ).pack(padx=20, pady=(2, 0), anchor="w")

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

        self._inline_drafts_var = ctk.BooleanVar(value=self._config.get("inline_drafts", False))
        ctk.CTkCheckBox(
            toggles_frame, text="Type live drafts into the window (may garble some editors)",
            variable=self._inline_drafts_var, **chk_opts,
        ).pack(anchor="w", pady=2)

        self._voice_chat_var = ctk.BooleanVar(value=self._config.get("voice_chat", False))
        ctk.CTkCheckBox(
            toggles_frame, text="Voice chat: talk and hear the reply",
            variable=self._voice_chat_var, **chk_opts,
        ).pack(anchor="w", pady=2)

        backends = {"local": "Local (Qwen)", "claude": "Claude Code"}
        self._backend_var = ctk.StringVar(
            value=backends.get(self._config.get("voice_chat_backend"), "Local (Qwen)"))
        self._backend_dropdown = ctk.CTkOptionMenu(
            toggles_frame, values=list(backends.values()), variable=self._backend_var, **dd_opts,
        )
        self._backend_dropdown.pack(fill="x", pady=(4, 8))
        ctk.CTkLabel(
            toggles_frame, text="Harness folder", text_color=_SECTION_TEXT,
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w")
        folder_row = ctk.CTkFrame(toggles_frame, fg_color="transparent")
        folder_row.pack(fill="x", pady=(4, 8))
        self._harness_cwd = self._config.get("harness_cwd")
        self._harness_folder_var = ctk.StringVar(
            value=str(Path(self._harness_cwd).expanduser()) if self._harness_cwd else str(Path.home()))
        ctk.CTkEntry(
            folder_row, textvariable=self._harness_folder_var, state="readonly",
            fg_color=_INPUT_BG, border_color=_INPUT_BORDER, text_color=_DESC_TEXT,
        ).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            folder_row, text="Browse…", command=self._browse_harness_folder, width=75,
            fg_color=_BTN_SEC_BG, hover_color=_BTN_SEC_HOVER, text_color=_BTN_SEC_TEXT,
        ).pack(side="left", padx=(6, 0))
        ctk.CTkButton(
            folder_row, text="Use home", command=self._use_harness_home, width=75,
            fg_color=_BTN_SEC_BG, hover_color=_BTN_SEC_HOVER, text_color=_BTN_SEC_TEXT,
        ).pack(side="left", padx=(6, 0))

        self._harness_new_session_var = ctk.BooleanVar(
            value=self._config.get("harness_new_session_on_start", False))
        ctk.CTkCheckBox(
            toggles_frame, text="New Claude conversation on each Débora start",
            variable=self._harness_new_session_var, **chk_opts,
        ).pack(anchor="w", pady=2)
        ctk.CTkLabel(
            toggles_frame, text="Claude session name", text_color=_SECTION_TEXT,
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", pady=(4, 0))
        self._harness_session_name_var = ctk.StringVar(
            value=self._config.get("harness_session_name", "Débora Whisper"))
        ctk.CTkEntry(
            toggles_frame, textvariable=self._harness_session_name_var,
            fg_color=_INPUT_BG, border_color=_INPUT_BORDER, text_color=_DESC_TEXT,
        ).pack(fill="x", pady=(4, 8))

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
        _make_wide_dropdown(self._voice_dropdown, voice_names, None)
        _make_wide_dropdown(self._tts_engine_dropdown, list(_TTS_ENGINES.values()),
                            self._on_tts_engine_change)
        _make_wide_dropdown(self._backend_dropdown, list(backends.values()), None)

    def _on_tts_engine_change(self, label):
        self._voice_dropdown.configure(
            state="disabled" if _TTS_ENGINE_KEYS[label] == "kokoro" else "normal")

    def _browse_harness_folder(self):
        folder = filedialog.askdirectory(parent=self._win, title="Harness folder",
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
            self._status_label.configure(text=text, text_color=color)

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
            self.update_status(str(e), "#FF453A")
            return
        self._config = dict(new_config)
        if self._on_apply:
            self._on_apply(new_config)
