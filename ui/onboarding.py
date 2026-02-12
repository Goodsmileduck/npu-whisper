"""First-run onboarding wizard with download/compilation progress."""

import threading
import customtkinter as ctk


class OnboardingWindow:
    """Modal window shown on first run to download and compile the model."""

    def __init__(self, root: ctk.CTk, config: dict, on_done=None):
        """
        Args:
            root: Parent CTk window.
            config: App config dict.
            on_done: Callback() when setup completes.
        """
        self._root = root
        self._config = config
        self._on_done = on_done
        self._win: ctk.CTkToplevel | None = None
        self._progress: ctk.CTkProgressBar | None = None
        self._phase_label: ctk.CTkLabel | None = None
        self._detail_label: ctk.CTkLabel | None = None

    def show(self):
        self._win = ctk.CTkToplevel(self._root)
        self._win.title("NPU Dictation — First Run Setup")
        self._win.geometry("500x220")
        self._win.resizable(False, False)
        self._win.attributes("-topmost", True)
        self._win.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing

        self._win.update_idletasks()
        from ui.glass import apply_acrylic
        apply_acrylic(self._win)

        # Set title bar icon — prevent CTkToplevel from overriding at 200ms
        from ui.icons import render_app_icon
        from PIL import ImageTk
        self._icon_photo = ImageTk.PhotoImage(render_app_icon(32))
        self._win._iconbitmap_method_called = True
        self._win.iconphoto(False, self._icon_photo)

        ctk.CTkLabel(
            self._win, text="Setting up NPU Dictation",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(pady=(24, 8))

        self._phase_label = ctk.CTkLabel(
            self._win, text="Downloading model...",
            font=ctk.CTkFont(size=14),
        )
        self._phase_label.pack(pady=(0, 8))

        self._progress = ctk.CTkProgressBar(self._win, width=400)
        self._progress.pack(pady=(0, 8))
        self._progress.set(0)

        self._detail_label = ctk.CTkLabel(
            self._win, text="This only happens once. Please wait...",
            font=ctk.CTkFont(size=12), text_color="gray60",
        )
        self._detail_label.pack(pady=(0, 16))

        # Start setup in background
        threading.Thread(target=self._run_setup, daemon=True).start()

    def _update_ui(self, phase: str = None, detail: str = None, progress: float = None):
        """Thread-safe UI update."""
        def _apply():
            if phase and self._phase_label:
                self._phase_label.configure(text=phase)
            if detail and self._detail_label:
                self._detail_label.configure(text=detail)
            if progress is not None and self._progress:
                if progress < 0:
                    self._progress.configure(mode="indeterminate")
                    self._progress.start()
                else:
                    self._progress.configure(mode="determinate")
                    self._progress.stop()
                    self._progress.set(min(progress, 1.0))
        self._root.after(0, _apply)

    def _run_setup(self):
        """Run model download + cache warmup in background thread."""
        from dictation_engine import setup_model, create_model, MODEL_REGISTRY, log
        import numpy as np

        # Phase 1: Download
        def _dl_progress(downloaded, total):
            if total > 0:
                self._update_ui(progress=downloaded / total)

        try:
            model_path = setup_model(self._config, progress_callback=_dl_progress)
        except SystemExit:
            self._update_ui(phase="Download failed!", detail="Check your internet connection and try again.")
            return

        # Phase 2: Compile
        self._update_ui(phase="Compiling for NPU...", detail="This can take 5-15 minutes on first run.", progress=-1)
        try:
            device = self._config["device"]
            model_info = MODEL_REGISTRY[self._config["model_size"]]
            model = create_model(model_path, device=device, backend=model_info["backend"])
            silence = np.zeros(self._config["sample_rate"], dtype=np.float32)
            model.transcribe(silence, sample_rate=self._config["sample_rate"],
                             language=self._config["language"])
            log("Onboarding: cache warm-up complete.")
        except Exception as e:
            log(f"Onboarding: cache warm-up failed (non-fatal): {e}")

        # Done
        self._update_ui(phase="Setup complete!", detail="You're ready to dictate.", progress=1.0)
        self._root.after(1000, self._finish)

    def _finish(self):
        if self._win:
            self._win.destroy()
        if self._on_done:
            self._on_done()
