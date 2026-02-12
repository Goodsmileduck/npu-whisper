"""Transcription history popup window."""

import customtkinter as ctk


class HistoryWindow:
    """Scrollable list of recent transcriptions with copy buttons."""

    def __init__(self, root: ctk.CTk, history: list[dict]):
        """
        Args:
            root: Parent CTk window.
            history: List of dicts with 'timestamp', 'text', 'duration' keys.
        """
        self._root = root
        self._history = history
        self._win: ctk.CTkToplevel | None = None

    def show(self):
        if self._win is not None and self._win.winfo_exists():
            self._win.focus_force()
            return

        self._win = ctk.CTkToplevel(self._root)
        self._win.title("NPU Dictation — History")
        self._win.geometry("500x400")
        self._win.attributes("-topmost", True)

        self._win.update_idletasks()
        from ui.glass import apply_acrylic
        apply_acrylic(self._win)

        # Set title bar icon — prevent CTkToplevel from overriding at 200ms
        from ui.icons import render_app_icon
        from PIL import ImageTk
        self._icon_photo = ImageTk.PhotoImage(render_app_icon(32))
        self._win._iconbitmap_method_called = True
        self._win.iconphoto(False, self._icon_photo)

        if not self._history:
            ctk.CTkLabel(
                self._win, text="No transcriptions yet.",
                font=ctk.CTkFont(size=14), text_color="gray60",
            ).pack(expand=True)
            return

        scroll = ctk.CTkScrollableFrame(self._win)
        scroll.pack(fill="both", expand=True, padx=8, pady=8)

        for entry in reversed(self._history):
            frame = ctk.CTkFrame(scroll)
            frame.pack(fill="x", pady=4)

            # Timestamp + duration
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            dur = entry.get("duration", 0)
            header = f"{ts}  ({dur:.1f}s)"
            ctk.CTkLabel(
                frame, text=header,
                font=ctk.CTkFont(size=11), text_color="gray60",
            ).pack(anchor="w", padx=8, pady=(4, 0))

            # Text + copy button row
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x", padx=8, pady=(0, 4))

            text = entry.get("text", "")
            ctk.CTkLabel(
                row, text=text, wraplength=380, anchor="w", justify="left",
            ).pack(side="left", fill="x", expand=True)

            # Capture text in closure
            def _make_copy(t):
                def _copy():
                    try:
                        import pyperclip
                        pyperclip.copy(t)
                    except Exception:
                        self._root.clipboard_clear()
                        self._root.clipboard_append(t)
                return _copy

            ctk.CTkButton(
                row, text="Copy", width=50, height=28,
                command=_make_copy(text),
            ).pack(side="right", padx=(8, 0))
