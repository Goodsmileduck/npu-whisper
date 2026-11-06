"""DPI regressions: monitor changes and Tk's independent font scaling."""

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

tk = pytest.importorskip("tkinter")
from debora_whisper.ui.overlay import OverlayWindow


@pytest.mark.parametrize("dpi", [96, 120, 144, 192, 240])
def test_scale_uses_window_dpi_and_pointer_sized_handle(monkeypatch, dpi):
    get_dpi = Mock(return_value=dpi)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(
        user32=SimpleNamespace(GetDpiForWindow=get_dpi)), raising=False)
    overlay = OverlayWindow.__new__(OverlayWindow)
    # A handle wider than 32 bits must survive the ctypes call.
    overlay._win = Mock()
    overlay._win.winfo_id.return_value = 0x123456789
    overlay._root = Mock()

    assert overlay._get_scale() == dpi / 96
    get_dpi.assert_called_once_with(0x123456789)
    assert ctypes.sizeof(get_dpi.argtypes[0]) == ctypes.sizeof(ctypes.c_void_p)
    overlay._root.winfo_fpixels.assert_not_called()


@pytest.mark.parametrize("failure", [None, OSError("DPI unavailable")])
def test_invalid_dpi_falls_back_to_tk(monkeypatch, failure):
    get_dpi = Mock(return_value=0, side_effect=failure)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(
        user32=SimpleNamespace(GetDpiForWindow=get_dpi)), raising=False)
    overlay = OverlayWindow.__new__(OverlayWindow)
    overlay._win = None
    overlay._root = Mock()
    overlay._root.winfo_fpixels.return_value = 144
    assert overlay._get_scale() == 1.5
    overlay._root.winfo_fpixels.return_value = 0
    assert overlay._get_scale() == 1.0


@pytest.mark.parametrize("scale", [1.0, 1.25, 1.5, 2.0, 2.5])
def test_fonts_use_pixels_with_one_dpi_conversion(scale):
    assert OverlayWindow._font_pixels(16, scale) == -round(16 * 96 / 72 * scale)


def _overlay(scale=1.0, state="ready"):
    with patch.object(OverlayWindow, "_get_scale", return_value=scale), \
            patch.object(OverlayWindow, "_build"):
        overlay = OverlayWindow(Mock(), on_toggle=Mock(), on_pos_changed=Mock(),
                                on_width_changed=Mock(), pos_x=132, pos_y=10)
    overlay._state = state
    overlay._win = Mock(winfo_x=Mock(return_value=100), winfo_y=Mock(return_value=10))
    overlay._canvas = Mock()
    overlay._text_font = Mock(side_effect=lambda: Mock(
        metrics=Mock(return_value=round(26 * overlay._scale))))
    overlay._monitor_work_area = Mock(return_value=(0, 0, 1920, 1080))
    overlay._redraw = Mock()
    overlay._apply_window_flags = Mock()
    overlay._get_scale = Mock(return_value=scale)
    overlay._position()
    return overlay


@pytest.mark.parametrize("scale", [1.0, 1.25, 1.5, 2.0, 2.5])
@pytest.mark.parametrize("state", ["ready", "recording", "processing", "speaking", "loading", "error"])
def test_voice_mode_border_preserves_geometry(monkeypatch, scale, state):
    from PIL import ImageChops
    from debora_whisper.ui import overlay as ov
    overlay = _overlay(scale, state)
    overlay._text_canvas = Mock()
    overlay._redraw = Mock(wraps=lambda: OverlayWindow._redraw(overlay))
    monkeypatch.setattr(ov, "pil_to_photo", lambda image: image)
    geometry = overlay._cur_w, overlay._cur_h, overlay._window_x, overlay._pos_y
    text_geometry = overlay._text_geometry(overlay._cur_w)
    overlay._win.reset_mock()
    overlay._redraw()
    original = overlay._photo_refs[0].copy()

    overlay.set_voice_mode(True)
    bordered = overlay._photo_refs[0].copy()
    pad, height, right, outer = overlay._mascot_geometry()
    changed = ImageChops.difference(original.convert("RGB"), bordered.convert("RGB")).getbbox()
    assert changed is not None
    # Along the balloon's edge around the mascot, never over the text.
    assert 0 <= changed[0] < changed[2] <= outer
    assert 0 <= changed[1] < changed[3] <= overlay._cur_h
    rim = max(1, round(OverlayWindow.VOICE_RIM * scale))
    x = (pad + right) // 2
    assert max(bordered.convert("RGB").getpixel((x, 0))) < 40  # dark rim
    r, g, b = bordered.convert("RGB").getpixel((x, rim + 1))
    assert b > r > g  # purple just inside the rim
    center = ((pad + right) // 2, pad + height // 2)
    assert original.getpixel(center) == bordered.getpixel(center)

    overlay._redraw.reset_mock()
    overlay.set_voice_mode(True)
    overlay._redraw.assert_not_called()
    overlay.set_voice_mode(False)
    overlay._redraw.assert_called_once()
    if state == "ready":
        assert overlay._hidden
    else:
        assert overlay._photo_refs[0].tobytes() == original.tobytes()
    assert (overlay._cur_w, overlay._cur_h, overlay._window_x, overlay._pos_y) == geometry
    assert overlay._text_geometry(overlay._cur_w) == text_geometry


@pytest.mark.parametrize("finish", ["show_ready", "show_result"])
def test_idle_hides_only_after_the_balloon_timer(finish):
    overlay = _overlay()
    overlay.show_speaking("Finished reply")
    args = ("Finished reply",) if finish == "show_result" else ()
    getattr(overlay, finish)(*args)
    overlay._win.withdraw.assert_not_called()
    delay, dismiss = overlay._root.after.call_args.args
    assert delay == overlay.BALLOON_DURATION
    dismiss()
    overlay._win.withdraw.assert_called_once_with()
    assert overlay._hidden and not overlay._conversation_lines()
    overlay.show_ready()
    overlay._win.withdraw.assert_called_once_with()


@pytest.mark.parametrize("finish", ["show_ready", "show_result", "hide"])
def test_idle_hides_immediately_when_balloon_disabled(finish):
    overlay = _overlay(state="speaking")
    overlay.set_show_balloon(False)
    overlay.show_speaking("Finished reply")
    args = ("Finished reply",) if finish == "show_result" else ()
    getattr(overlay, finish)(*args)
    overlay._win.withdraw.assert_called_once_with()
    assert overlay._hidden


def test_startup_and_empty_result_hide_immediately():
    overlay = _overlay(state="loading")
    overlay.show_ready()
    overlay._win.withdraw.assert_called_once_with()
    overlay.show_recording()
    overlay.show_result("")
    assert overlay._win.withdraw.call_count == 2


@pytest.mark.parametrize("region", ["mascot", "resize"])
def test_idle_waits_for_drag_end_before_hiding(region):
    overlay = _overlay(state="recording")
    overlay.show_result("Finished turn")
    overlay._press_region = region
    overlay._dragging = True
    overlay._root.after.call_args.args[1]()
    assert not overlay._hidden
    overlay._win.withdraw.assert_not_called()
    overlay._on_drag_end(SimpleNamespace(x=20, y=10))
    assert overlay._hidden
    overlay._win.withdraw.assert_called_once_with()


def test_unhide_refreshes_dpi_before_positioning():
    overlay = _overlay()
    overlay.show_ready()
    overlay._get_scale.return_value = 2.0
    positioned = []
    position = overlay._position

    def record_position():
        positioned.append(overlay._scale)
        position()

    overlay._position = record_position
    overlay.show_recording()
    assert positioned == [2.0]
    assert overlay._cur_h == overlay.COMPACT_H * 2.0
    overlay._win.deiconify.assert_called_once_with()


@pytest.mark.parametrize("voice_mode", [False, True])
@pytest.mark.parametrize("dragging", [False, True])
def test_hide_dismisses_text_and_forces_hidden_state(voice_mode, dragging):
    overlay = _overlay()
    overlay.set_voice_mode(voice_mode)
    overlay.show_recording("Question", voice_chat=True)
    overlay.show_speaking("Reply")
    overlay._dragging = dragging
    overlay.hide()
    overlay._update_layout()
    assert overlay._hidden
    assert not overlay._conversation_lines()
    assert overlay._balloon_id is None
    overlay._win.withdraw.assert_called_once_with()


@pytest.mark.parametrize("show, kwargs", [
    ("show_recording", {}), ("show_recording", {"voice_chat": True}),
    ("show_ready", {"continuous_listening": True}),
    ("show_result", {"text": "", "continuous_listening": True}),
])
@pytest.mark.parametrize("balloon", [False, True])
def test_open_microphone_states_always_show_overlay(show, kwargs, balloon):
    overlay = _overlay()
    overlay.show_ready()
    assert overlay._hidden
    overlay.set_show_balloon(balloon)
    overlay._win.withdraw.reset_mock()
    getattr(overlay, show)(**kwargs)
    overlay._dismiss_balloon()
    overlay.hide()
    assert not overlay._hidden
    overlay._win.deiconify.assert_called_once_with()
    overlay._win.withdraw.assert_not_called()


@pytest.mark.parametrize("show, args", [
    ("show_recording", ()), ("show_processing", ()),
    ("show_speaking", ("Reply",)), ("show_error", ()),
    ("show_loading", ()), ("set_voice_mode", (True,)),
])
def test_hidden_overlay_reappears_without_focus_or_position_changes(show, args):
    overlay = _overlay()
    overlay.show_ready()
    position = overlay._pos_x, overlay._pos_y
    calls = Mock()
    calls.attach_mock(overlay._apply_window_flags, "flags")
    calls.attach_mock(overlay._win.deiconify, "show")
    calls.attach_mock(overlay._win.attributes, "attributes")
    getattr(overlay, show)(*args)
    assert not overlay._hidden
    assert [call[0] for call in calls.mock_calls] == ["flags", "show", "attributes", "flags"]
    overlay._win.attributes.assert_called_with("-topmost", True)
    overlay._win.focus_set.assert_not_called()
    overlay._win.focus_force.assert_not_called()
    assert (overlay._pos_x, overlay._pos_y) == position


@pytest.mark.parametrize("voice_mode", [False, True])
def test_listening_keeps_overlay_visible_between_turns(voice_mode):
    overlay = _overlay()
    overlay.set_voice_mode(voice_mode)
    listening = {"continuous_listening": not voice_mode}
    overlay.show_ready(**listening)
    overlay.show_result("Finished turn", **listening)
    overlay._root.after.call_args.args[1]()
    overlay._win.withdraw.assert_not_called()
    assert not overlay._hidden
    overlay.set_voice_mode(False)
    if voice_mode:
        assert overlay._hidden
    overlay.show_ready()
    assert overlay._hidden


def test_disabling_balloon_hides_an_idle_result():
    overlay = _overlay()
    overlay.show_result("Finished turn")
    overlay.set_show_balloon(False)
    overlay._win.withdraw.assert_called_once_with()


def test_hidden_overlay_stops_animation_and_restarts_on_recording():
    overlay = _overlay(state="recording")
    overlay._redraw.side_effect = overlay._sync_mascot_animation
    overlay._root.after.return_value = "mascot-timer"
    overlay._sync_mascot_animation()
    overlay.show_ready()
    overlay._root.after_cancel.assert_called_once_with("mascot-timer")
    assert overlay._mascot_anim_id is None
    overlay._canvas.reset_mock()
    overlay._root.after.reset_mock()
    OverlayWindow._redraw(overlay)
    overlay._mascot_tick()
    overlay._slide_tick()
    overlay._canvas.delete.assert_not_called()
    overlay._root.after.assert_not_called()
    overlay.show_recording()
    overlay._root.after.assert_called_once_with(1000 // overlay.MASCOT_FPS, overlay._mascot_tick)


def test_monitor_change_rescales_even_without_animation():
    overlay = _overlay(2.0)
    overlay.show_result("Visible transcription")
    x = overlay._window_x
    overlay._get_scale = Mock(return_value=1.25)
    overlay._redraw.reset_mock()
    overlay._refresh_scale()
    assert overlay._scale == 1.25
    assert overlay._cur_w == overlay._mascot_geometry()[3] + round(360 * 1.25)
    assert overlay._cur_h == round(38 * 1.25)
    assert overlay._window_x == x
    assert overlay._conversation_lines()[0][1] == "Visible transcription"
    overlay._refresh_scale()
    overlay._redraw.assert_called_once()


@pytest.mark.parametrize("root_scale, monitor_scale", [(1.0, 2.0), (2.0, 1.0), (1.0, 1.5)])
def test_saved_position_uses_destination_dpi_at_startup(root_scale, monitor_scale):
    overlay = _overlay(root_scale)
    overlay._window_x = None
    overlay._pos_x, overlay._pos_y = 32, 1042
    overlay._get_scale = Mock(return_value=monitor_scale)
    overlay._restore_position()
    width = overlay._mascot_geometry()[3]
    assert overlay._window_x == max(0, 32 - width // 2)
    assert overlay._pos_y == min(1042, 1080 - round(38 * monitor_scale))


@pytest.mark.parametrize("scale", [1.0, 1.25, 1.5, 2.0])
@pytest.mark.parametrize("x", [2, 30, 59])
def test_a_click_on_mascot_toggles(scale, x):
    overlay = _overlay(scale)
    press = SimpleNamespace(x=x * scale, y=10)
    overlay._on_drag_start(press)
    overlay._on_drag_move(SimpleNamespace(x=x * scale + 3 * scale, y=10))
    overlay._on_drag_end(press)
    overlay._on_toggle.assert_called_once()
    overlay._on_pos_changed.assert_not_called()


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_a_drag_moves_without_toggling(scale):
    overlay = _overlay(scale)
    overlay._on_drag_start(SimpleNamespace(x=20, y=10))
    overlay._on_drag_move(SimpleNamespace(x=20 + 30 * scale, y=10))
    overlay._on_drag_end(SimpleNamespace(x=20 + 30 * scale, y=10))
    overlay._on_toggle.assert_not_called()
    overlay._on_pos_changed.assert_called_once_with(overlay._pos_x, overlay._pos_y)


@pytest.mark.parametrize("state", ["loading", "processing", "error"])
def test_a_click_does_nothing_while_busy(state):
    overlay = _overlay(state=state)
    overlay._on_drag_start(SimpleNamespace(x=20, y=10))
    overlay._on_drag_end(SimpleNamespace(x=20, y=10))
    overlay._on_toggle.assert_not_called()


@pytest.mark.parametrize("method,args", [
    ("show_loading", ()), ("show_ready", ()), ("show_processing", ()),
    ("show_error", ()), ("show_notice", ("Loading a model, please wait",)),
])
def test_status_and_notice_never_change_conversation(method, args):
    overlay = _overlay()
    getattr(overlay, method)(*args)
    assert overlay._conversation_lines() == []
    overlay.show_processing("abre o log", voice_chat=True)
    overlay.show_speaking("O log mostra…")
    before = overlay._conversation_lines()
    getattr(overlay, method)(*args)
    assert overlay._conversation_lines() == before


def test_text_is_a_soft_white_in_a_legible_font(monkeypatch):
    import tkinter.font as tkfont
    overlay = OverlayWindow.__new__(OverlayWindow)
    overlay._root = None
    assert OverlayWindow.TEXT.upper() != "#FFFFFF"
    monkeypatch.setattr(tkfont, "families", lambda root=None: ["Segoe UI", "Segoe UI Semibold",
                                                                "Segoe UI Variable Text",
                                                                "Segoe UI Variable Text Semibold"])
    assert overlay._font(14) == ("Segoe UI Variable Text", 14)
    assert overlay._font(14, semibold=True) == ("Segoe UI Variable Text Semibold", 14)


def test_font_falls_back_to_segoe_ui(monkeypatch):
    import tkinter.font as tkfont
    overlay = OverlayWindow.__new__(OverlayWindow)
    overlay._root = None
    monkeypatch.setattr(tkfont, "families", lambda root=None: ["Segoe UI", "Arial"])
    assert overlay._font(14, semibold=True) == ("Segoe UI Semibold", 14)


def test_panel_is_flat():
    from debora_whisper.ui.glass import render_pill
    overlay = OverlayWindow.__new__(OverlayWindow)
    img = render_pill(150, 38, radius=OverlayWindow.RADIUS, **overlay._flat())
    # One color edge to edge: no border, gradient or highlight.
    inner = img.crop((8, 2, 142, 36)).convert("RGB")
    assert len(set(inner.getdata())) == 1
    assert img.getpixel((75, 0))[:3] == img.getpixel((75, 19))[:3]


def test_panel_is_translucent_and_slightly_rounded():
    assert 0.5 < OverlayWindow.OPACITY < 1.0
    assert OverlayWindow.RADIUS < OverlayWindow.COMPACT_H // 4


@pytest.fixture
def tk_root(monkeypatch):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    # Exercise real Tk measurement/rendering without showing test windows.
    toplevel = tk.Toplevel

    def hidden_toplevel(*args, **kwargs):
        window = toplevel(*args, **kwargs)
        window.withdraw()
        return window

    monkeypatch.setattr(tk, "Toplevel", hidden_toplevel)
    yield root
    for timer in root.tk.call("after", "info"):
        root.after_cancel(timer)
    root.destroy()


@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0])
def test_real_tk_clipped_text_ignores_other_monitors_font_scale(tk_root, monkeypatch, scale):
    monkeypatch.setattr(OverlayWindow, "_get_scale", lambda self: scale)
    monkeypatch.setattr(OverlayWindow, "_monitor_work_area", lambda *a, **kw: (0, 0, 4000, 2000))
    now = [0.0]
    monkeypatch.setattr("debora_whisper.ui.overlay.monotonic", lambda: now[0])
    overlay = OverlayWindow(tk_root, pos_x=200)
    assert overlay._cur_w == overlay._mascot_geometry()[3]
    assert overlay._cur_h == round(38 * scale)
    assert len(tk_root.winfo_children()) == 1  # only one Toplevel
    measurements = []
    for tk_scale in (96 / 72, 192 / 72):
        tk_root.tk.call("tk", "scaling", tk_scale)
        overlay._new_turn(True)
        overlay.show_processing("Oi", voice_chat=True)
        overlay.show_speaking("Uma resposta bastante longa " * 12)
        now[0] += 1
        overlay._slide_tick()
        canvas = overlay._text_canvas
        boxes = [canvas.bbox(item) for item in canvas.find_all()]
        left, right, _, _ = overlay._text_geometry(overlay._cur_w)
        visible = right - left
        # One row: the long reply pushes the user's words behind the divider.
        assert boxes[0][0] < 0
        assert boxes[0][2] < boxes[1][0] < visible
        assert boxes[1][2] == visible
        assert boxes[0][1] == boxes[1][1] >= 0
        assert boxes[1][3] <= int(canvas.place_info()["height"])
        assert int(canvas.place_info()["x"]) == left
        assert int(canvas.place_info()["width"]) == visible
        assert canvas.cget("scrollregion") == ""
        measurements.append((boxes, overlay._cur_w, overlay._cur_h))
    assert measurements[0] == measurements[1]
    assert overlay._cur_h == round(38 * scale)  # never a second line
    overlay._new_turn(True)
    overlay.show_processing("Oi", voice_chat=True)
    overlay.show_speaking("Olá!")
    now[0] += 1
    overlay._slide_tick()
    boxes = [canvas.bbox(item) for item in canvas.find_all()]
    assert boxes[0][0] == 0  # fitting text starts exactly at the divider + gap
    overlay.set_show_balloon(False)
    assert overlay._cur_w == overlay._mascot_geometry()[3]
    assert not overlay._text_canvas.place_info()


def test_voice_turn_keeps_both_sides_through_speech_and_delay():
    overlay = _overlay()
    overlay.show_recording("abre", voice_chat=True)
    overlay.show_processing("abre o log", voice_chat=True)
    overlay.show_speaking("O log mostra.")
    overlay.show_speaking("O log mostra. O TTS demorou.")
    assert [line[1] for line in overlay._conversation_lines()] == [
        "Você: abre o log", "Débora: O log mostra. O TTS demorou."]
    assert overlay._balloon_id is None
    overlay.show_result("O log mostra. O TTS demorou.")
    assert len(overlay._conversation_lines()) == 2
    overlay._root.after.assert_called_with(overlay.BALLOON_DURATION, overlay._dismiss_balloon)
    overlay._dismiss_balloon()
    assert overlay._conversation_lines() == []
    assert overlay._cur_w == overlay._mascot_geometry()[3]


def test_continuous_listening_keeps_finished_turn_until_new_speech():
    overlay = _overlay()
    overlay.show_processing("um", voice_chat=True)
    overlay.show_speaking("resposta")
    overlay.show_recording("", voice_chat=True)
    assert [line[1] for line in overlay._conversation_lines()] == ["Você: um", "Débora: resposta"]
    assert overlay._balloon_id is not None
    overlay.show_recording("dois", voice_chat=True)
    assert [line[1] for line in overlay._conversation_lines()] == ["Você: dois"]
    assert overlay._balloon_id is None


def test_text_click_dismisses_until_next_turn_without_toggling():
    overlay = _overlay()
    overlay.show_processing("pergunta", voice_chat=True)
    overlay.show_speaking("primeira frase")
    press = SimpleNamespace(x=100, y=10)
    overlay._on_drag_start(press)
    overlay._on_drag_end(press)
    overlay.show_speaking("primeira frase. segunda frase.")
    assert overlay._conversation_lines() == []
    overlay._on_toggle.assert_not_called()
    overlay.show_ready()
    overlay.show_recording("outra pergunta", voice_chat=True)
    assert overlay._conversation_lines()[0][1] == "Você: outra pergunta"


def test_plain_dictation_keeps_the_existing_dismiss_delay():
    overlay = _overlay()
    overlay.show_recording("rascunho")
    overlay.show_processing()
    assert overlay._conversation_lines()[0][1] == "rascunho"
    overlay.show_result("transcrição final")
    assert overlay._conversation_lines() == [("user", "transcrição final", overlay.TEXT)]
    overlay._root.after.assert_called_with(2500, overlay._dismiss_balloon)
    overlay.show_recording()
    assert overlay._conversation_lines() == []


def test_failed_reply_finishes_without_showing_error_text():
    overlay = _overlay()
    overlay.show_processing("pergunta", voice_chat=True)
    overlay.show_speaking("resposta parcial")
    lines = overlay._conversation_lines()
    overlay.show_error()
    assert overlay._conversation_lines() == lines
    assert overlay._balloon_id is not None


@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0])
def test_explicit_text_geometry_and_independent_eased_offsets(scale):
    overlay = _overlay(scale)
    overlay.show_processing("oi", voice_chat=True)
    overlay.show_speaking("resposta")
    left, right, bar, _ = overlay._text_geometry(overlay._cur_w)
    assert left == overlay._mascot_geometry()[2] + round(overlay.BALLOON_GAP * scale)
    assert right == bar - round(overlay.BALLOON_GAP * scale)
    visible = right - left
    width = visible + 100 * scale
    target = overlay._line_target(width, visible)
    assert left + target + width == right
    assert overlay._slide_offset("reply", target, 0) == 0
    midway = overlay._slide_offset("reply", target, 0.1)
    assert target < midway < target / 2  # ease-out has covered over half the distance
    assert overlay._slide_offset("user", 0, 0.1) == 0
    assert overlay._slide_offset("reply", target, 0.2) == target
    assert overlay._slide_offset("reply", target - 30, 0.2) == target
    assert overlay._slide_offset("reply", target - 30, 0.5) == target - 30


def test_her_sentence_slides_at_the_pace_of_its_audio():
    overlay = _overlay()
    overlay.show_processing("oi", voice_chat=True)
    overlay.show_speaking("resposta", seconds=4.0)
    overlay._slides.clear()
    overlay._pace = 4.0
    assert overlay._slide_offset("row", -100, 0) == 0
    overlay._pace = None  # what _draw_text does after its first draw
    # Linear: a quarter of the audio, a quarter of the way.
    assert overlay._slide_offset("row", -100, 1.0) == -25
    assert overlay._slide_offset("row", -100, 2.0) == -50
    assert overlay._slide_offset("row", -100, 4.0) == -100
    # The next sentence without audio keeps the quick ease-out.
    assert overlay._slide_offset("row", -140, 4.0) == -100
    assert overlay._slide_offset("row", -140, 4.0 + overlay.SLIDE_SECONDS) == -140


def test_next_sentence_continues_from_where_the_last_one_is():
    overlay = _overlay()
    overlay._pace = 2.0
    overlay._slide_offset("row", -100, 0)
    overlay._pace = 4.0
    assert overlay._slide_offset("row", -300, 1.0) == -50
    assert overlay._slide_offset("row", -300, 3.0) == -175
    assert overlay._slide_offset("row", -300, 5.0) == -300


@pytest.mark.parametrize("left, held", [(3.0, True), (0.1, False)])
def test_an_interrupted_sentence_stops_sliding(monkeypatch, left, held):
    now = [0.0]
    monkeypatch.setattr("debora_whisper.ui.overlay.monotonic", lambda: now[0])
    overlay = _overlay()
    overlay.show_processing("oi", voice_chat=True)
    overlay.show_speaking("resposta", seconds=4.0)
    overlay._slides.clear()
    overlay._pace = 4.0
    overlay._slide_offset("row", -100, 0)
    now[0] = 4.0 - left
    overlay.set_talking("debora", True)
    overlay.set_talking("debora", False)
    assert overlay._slide_held is held
    stopped = overlay._slide_offset("row", -100, 10.0)
    # Cut short: it stays put; only buffer latency left: it finishes.
    assert stopped == (-25 if held else -100)
    overlay.show_speaking("resposta. mais", seconds=2.0)
    assert not overlay._slide_held


@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0])
def test_width_handle_clamps_and_saves_logical_pixels(scale):
    overlay = _overlay(scale)
    overlay.show_result("texto")
    x = overlay._window_x
    _, _, bar, _ = overlay._text_geometry(overlay._cur_w)
    press = SimpleNamespace(x=bar - 2 * scale, y=10)
    overlay._on_mouse_move(press)
    overlay._canvas.configure.assert_called_with(cursor="sb_h_double_arrow")
    overlay._on_drag_start(press)
    overlay._on_drag_move(SimpleNamespace(x=press.x + 60 * scale, y=10))
    overlay._on_drag_end(press)
    overlay._on_width_changed.assert_called_with(420)
    assert overlay._window_x == x
    for delta, expected in [(-5000, 200), (5000, round((1920 - x - overlay._mascot_geometry()[3]) / scale))]:
        _, _, bar, _ = overlay._text_geometry(overlay._cur_w)
        overlay._on_drag_start(SimpleNamespace(x=bar, y=10))
        overlay._on_drag_move(SimpleNamespace(x=bar + delta, y=10))
        overlay._on_drag_end(press)
        assert overlay._balloon_width == expected
        assert x + overlay._cur_w <= 1920
    overlay._on_toggle.assert_not_called()
    overlay._on_pos_changed.assert_not_called()


@pytest.mark.parametrize("monitor", [(0, 0, 1920, 1080), (-1920, -1080, 0, 0)])
def test_expansion_preserves_mascot_monitor_and_taskbar_position(monitor):
    overlay = _overlay()
    left, top, right, bottom = monitor
    overlay._monitor_work_area.return_value = monitor
    overlay._window_x = right - 90
    overlay._pos_y = bottom - 38  # taskbar included
    overlay._position()
    anchor = overlay._window_x, overlay._pos_y
    overlay._balloon_width = 500
    overlay.show_processing("pergunta", voice_chat=True)
    overlay.show_speaking("resposta")
    assert (overlay._window_x, overlay._pos_y) == anchor
    assert overlay._window_x + overlay._cur_w == right
    assert overlay._balloon_width == 500  # shrinking the viewport never loses the preference
    assert overlay._monitor_work_area.call_args.kwargs == {"full": True}
    overlay._dismiss_balloon()
    assert (overlay._window_x, overlay._pos_y) == anchor
    overlay._window_x = None  # restoring the saved center/top on startup
    overlay._position()
    assert (overlay._window_x, overlay._pos_y) == anchor


def test_mascot_ships_inside_the_package():
    from PIL import Image
    from debora_whisper.ui.overlay import MASCOT_PATH
    assert "debora_whisper" in MASCOT_PATH.parts
    with Image.open(MASCOT_PATH) as img:
        assert img.size[0] == img.size[1] >= 64


@pytest.mark.parametrize("clip", ["loop", "zoom"])
def test_mascot_clips_ship_and_load(clip):
    from debora_whisper.ui import overlay as ov
    path = ov.MASCOT_LOOP_PATH if clip == "loop" else ov.MASCOT_ZOOM_PATH
    overlay = OverlayWindow.__new__(OverlayWindow)
    assert "debora_whisper" in path.parts
    frames = overlay._mascot_frames(34, radius=6, clip=clip)
    assert len(frames) > 1
    assert all(f.size == (60, 34) and f.mode == "RGBA" for f in frames)  # 16:9
    assert frames[0].getpixel((0, 0))[3] < 16  # outside the rounded corner


def test_mascot_falls_back_to_the_still_image(monkeypatch, tmp_path):
    import debora_whisper.ui.overlay as ov
    monkeypatch.setattr(ov, "MASCOT_LOOP_PATH", tmp_path / "missing.webp")
    overlay = OverlayWindow.__new__(OverlayWindow)
    frames = overlay._mascot_frames(34)
    assert len(frames) == 1 and frames[0].size == (60, 34)  # square still, cropped


@pytest.mark.parametrize("state, animated", [
    ("recording", True), ("speaking", True), ("ready", True), ("processing", True),
    ("loading", False), ("error", False),
])
def test_mascot_flaps_in_silence_and_stands_still_only_loading_or_failed(state, animated):
    overlay = OverlayWindow.__new__(OverlayWindow)
    overlay._hidden = False
    overlay._root = Mock()
    overlay._root.after.return_value = "timer"
    overlay._state = state
    overlay._mascot_index = 5
    overlay._mascot_clip = "loop"
    overlay._mascot_anim_id = None
    overlay._sync_mascot_animation()
    assert overlay._root.after.called is animated
    if not animated:
        assert overlay._mascot_index == 0


def test_an_error_stops_the_timer():
    overlay = OverlayWindow.__new__(OverlayWindow)
    overlay._hidden = False
    overlay._root = Mock()
    overlay._state = "error"
    overlay._mascot_index = 7
    overlay._mascot_clip = "zoom"
    overlay._mascot_anim_id = "timer"
    overlay._sync_mascot_animation()
    overlay._root.after_cancel.assert_called_once_with("timer")
    assert overlay._mascot_anim_id is None and overlay._mascot_index == 0
    assert overlay._mascot_clip == "loop"


@pytest.mark.parametrize("state", ["ready", "recording", "processing", "speaking"])
def test_silence_loops_whatever_the_state(state):
    overlay = _overlay(state=state)
    overlay.show_speaking("Oi!") if state == "speaking" else None
    loop_len = len(overlay._mascot_frames(1, clip="loop"))
    for _ in range(loop_len + 3):
        overlay._mascot_tick()
    assert (overlay._mascot_clip, overlay._mascot_index) == ("loop", 3)


@pytest.mark.parametrize("who", ["user", "debora"])
def test_talking_zooms_and_repeats_until_it_stops(who):
    overlay = _overlay(state="recording")
    for _ in range(3):
        overlay._mascot_tick()
    overlay.set_talking(who, True)
    assert (overlay._mascot_clip, overlay._mascot_index) == ("zoom", 0)
    zoom_len = len(overlay._mascot_frames(1, clip="zoom"))
    for _ in range(zoom_len + 2):
        overlay._mascot_tick()
    assert (overlay._mascot_clip, overlay._mascot_index) == ("zoom", 2)
    # A new state (more of the reply, a draft) never restarts the zoom.
    overlay.show_speaking("Oi! Tudo bem?")
    overlay.show_recording("e você")
    assert (overlay._mascot_clip, overlay._mascot_index) == ("zoom", 2)


def test_silence_early_in_the_zoom_plays_it_back_out_to_the_loop():
    overlay = _overlay(state="speaking")
    overlay.set_talking("debora", True)
    for _ in range(3):
        overlay._mascot_tick()
    overlay.set_talking("debora", False)
    shown = []
    for _ in range(4):
        overlay._mascot_tick()
        shown.append((overlay._mascot_clip, overlay._mascot_index))
    assert shown == [("zoom", 2), ("zoom", 1), ("zoom", 0), ("loop", 0)]


def test_silence_late_in_the_zoom_finishes_it_then_loops():
    overlay = _overlay(state="recording")
    overlay.set_talking("user", True)
    zoom_len = len(overlay._mascot_frames(1, clip="zoom"))
    for _ in range(zoom_len - 2):
        overlay._mascot_tick()
    overlay.set_talking("user", False)
    overlay._mascot_tick()
    assert (overlay._mascot_clip, overlay._mascot_index) == ("zoom", zoom_len - 1)
    overlay._mascot_tick()
    assert (overlay._mascot_clip, overlay._mascot_index) == ("loop", 0)


def test_zoom_lasts_while_either_one_talks():
    overlay = _overlay(state="speaking")
    overlay.set_talking("debora", True)
    overlay.set_talking("user", True)  # barge-in
    overlay.set_talking("debora", False)
    zoom_len = len(overlay._mascot_frames(1, clip="zoom"))
    for _ in range(zoom_len + 1):
        overlay._mascot_tick()
    assert overlay._mascot_clip == "zoom"


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_resize_grip_is_a_small_glass_capsule(scale):
    overlay = _overlay(scale)
    h = round(60 * scale)
    idle, active = overlay._grip_image(0, h), overlay._grip_image(overlay.GRIP_STEPS, h)
    assert idle.size == active.size == (round(4 * scale), round(24 * scale))
    bg = int(overlay.BG[1:3], 16)
    middle = (idle.width // 2, idle.height // 2)
    assert abs(idle.getpixel(middle)[0] - (bg + 0.2 * (255 - bg))) <= 6  # soft gradient
    assert abs(active.getpixel(middle)[0] - (bg + 0.5 * (255 - bg))) <= 12
    assert idle.getpixel((0, 0))[0] < idle.getpixel(middle)[0]  # rounded caps
    # A short balloon never gets a capsule taller than its text area.
    assert overlay._grip_image(0, 10).height == 10


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_resize_grip_hit_area_and_hover_fade(scale):
    overlay = _overlay(scale)
    overlay._draw_grip = Mock()
    overlay.show_result("texto")
    _, _, bar, bar_w = overlay._text_geometry(overlay._cur_w)
    assert bar == overlay._cur_w - round(4 * scale) - bar_w
    center = bar + bar_w / 2
    reach = 8 * scale
    assert overlay._hit_region(center - reach) == "resize"
    assert overlay._hit_region(center + reach) == "resize"
    assert overlay._hit_region(center - reach - 1) == "text"
    overlay._on_mouse_move(SimpleNamespace(x=center, y=10))
    overlay._canvas.configure.assert_called_with(cursor="sb_h_double_arrow")
    delay, tick = overlay._root.after.call_args.args
    assert delay == overlay.GRIP_FADE_MS // overlay.GRIP_STEPS
    for _ in range(overlay.GRIP_STEPS + 1):
        overlay._grip_fade_tick()
    assert overlay._grip_level == overlay.GRIP_STEPS
    assert overlay._draw_grip.call_count == overlay.GRIP_STEPS  # nothing else is redrawn
    overlay._on_leave(SimpleNamespace(x=0, y=0))
    for _ in range(overlay.GRIP_STEPS + 1):
        overlay._grip_fade_tick()
    assert overlay._grip_level == 0


def test_mascot_edge_fades_into_the_panel():
    overlay = OverlayWindow.__new__(OverlayWindow)
    hard = overlay._mascot_frames(34, radius=6)[0]
    soft = overlay._mascot_frames(34, feather=2, radius=6)[0]
    # The center stays opaque; the rim is fainter than a hard edge's.
    assert soft.getpixel((30, 17))[3] == 255
    rim = [(30, 0), (0, 17), (59, 17), (30, 33)]
    assert sum(soft.getpixel(p)[3] for p in rim) < sum(hard.getpixel(p)[3] for p in rim) / 2


def test_text_pulses_while_she_processes_the_input():
    overlay = _overlay()
    a = overlay._pulse(overlay.TEXT, 0)
    b = overlay._pulse(overlay.TEXT, overlay.PULSE_SECONDS / 2)
    assert a == overlay.TEXT  # full brightness at the top of the pulse
    dim = int(b[1:3], 16)
    bg, fg = int(overlay.BG[1:3], 16), int(overlay.TEXT[1:3], 16)
    assert dim == round(bg + (fg - bg) * overlay.PULSE_MIN)


@pytest.mark.parametrize("show", ["show_loading", "show_error"])
def test_a_reloaded_or_failed_engine_never_leaves_the_mascot_zooming(show):
    overlay = _overlay(state="speaking")
    overlay.set_talking("debora", True)
    getattr(overlay, show)()  # the old engine's "stopped" event never comes
    overlay._state = "ready"
    zoom_len = len(overlay._mascot_frames(1, clip="zoom"))
    for _ in range(zoom_len + 1):
        overlay._mascot_tick()
    assert overlay._mascot_clip == "loop"


def test_text_pulses_from_the_input_until_her_voice_starts():
    overlay = _overlay(state="recording")
    pulsing = []
    overlay._pulse = Mock(side_effect=lambda fill, now: pulsing.append(overlay._state) or fill)
    overlay._text_canvas = Mock(bbox=Mock(return_value=(0, 0, 10, 10)))
    overlay._text_font = Mock(return_value=Mock(metrics=Mock(return_value=26),
                                                measure=Mock(return_value=6)))

    def draws_pulsing():
        pulsing.clear()
        overlay._draw_text(300)
        return bool(pulsing)

    overlay.show_processing("oi", voice_chat=True)
    assert draws_pulsing()
    overlay.show_speaking("Olá!")  # her text, but no audio yet
    assert draws_pulsing()
    overlay.set_talking("debora", True)
    assert not draws_pulsing()
    overlay.set_talking("debora", False)  # a gap between sentences
    assert not draws_pulsing()
    overlay.show_processing("e agora?", voice_chat=True)
    assert draws_pulsing()
