"""Best-effort caret verification; no COM objects leave the UIA thread."""
import ctypes
import ntpath
import queue
import threading

from debora_whisper.platforms.base import CaretReadStatus

try:
    # Import before starting our worker: comtypes initializes its importing
    # thread's apartment. Our process-lifetime worker explicitly uses MTA.
    import comtypes.client
except Exception as exc:
    comtypes = None
    _import_failure = ("comtypes missing" if isinstance(exc, ModuleNotFoundError)
                       and exc.name in ("comtypes", "comtypes.client")
                       else f"comtypes import failed: {type(exc).__name__}")
else:
    _import_failure = "comtypes unavailable"


_TIMEOUT = 0.150
_query_lock = threading.Lock()
_worker = None
_warning_lock = threading.Lock()
_warned_unavailable = False


def _warn_unavailable(reason):
    """Report a broken UIA installation once, separately from target failures."""
    global _warned_unavailable
    with _warning_lock:
        if _warned_unavailable:
            return
        _warned_unavailable = True
    from debora_whisper.dictation_engine import log
    log(f"WARNING: Inline drafts: UI Automation unavailable ({reason}); "
        "reinstall with `uv tool install --editable . --force` "
        "(keep your installed extras, e.g. .[cuda]), then restart")


def text_before_caret(count: int) -> str | CaretReadStatus | None:
    """Wait at most 150 ms, including type-library generation on first use.

    A hung provider holds the single slot; later calls return unknown rather
    than accumulate requests. Late results are discarded, never reused.
    """
    global _worker
    if count < 0:
        return None
    if comtypes is None:
        _warn_unavailable(_import_failure)
        return None
    if not _query_lock.acquire(blocking=False):
        return None
    done = threading.Event()
    result = [None]

    try:
        if _worker is None:
            requests = queue.SimpleQueue()
            thread = threading.Thread(target=_serve, args=(comtypes, requests),
                                      name="uia-caret", daemon=True)
            thread.start()
            _worker = requests, thread
        _worker[0].put((count, result, done))
    except Exception as exc:
        _query_lock.release()
        _warn_unavailable(f"worker start failed: {type(exc).__name__}")
        return None
    return result[0] if done.wait(_TIMEOUT) else None


def _serve(api, requests):
    # Keep this apartment alive until process exit, including when idle or a
    # provider fails. GetModule's internal ctypes/type-library cycles can live
    # beyond a query; CoUninitialize here would invalidate their COM pointers.
    initialized = False
    while (request := requests.get()) is not None:
        count, result, done = request
        try:
            if not initialized:
                api.CoInitializeEx(api.COINIT_MULTITHREADED)
                initialized = True
            result[0] = _read_text_before_caret(count, api)
        except Exception as exc:
            _warn_unavailable(f"UIA initialization failed: {type(exc).__name__}")
        finally:
            _query_lock.release()
            done.set()


def _release(pointer):
    """Release our reference now, and prevent comtypes from releasing it twice."""
    if pointer:
        try:
            pointer.Release()
        finally:
            # comtypes overrides .value with a read-only property. Clear the
            # underlying c_void_p so its __del__ is harmless even if a ctypes
            # cycle is collected later, on a different thread.
            ctypes.c_void_p.value.__set__(pointer, None)


def _system_uia_path() -> str:
    # Never let comtypes search the working directory for a type library.
    kernel32 = ctypes.WinDLL("kernel32.dll", winmode=0x00000800)
    get_system_directory = kernel32.GetSystemDirectoryW
    get_system_directory.argtypes = [ctypes.c_wchar_p, ctypes.c_uint]
    get_system_directory.restype = ctypes.c_uint
    buffer = ctypes.create_unicode_buffer(32768)
    length = get_system_directory(buffer, len(buffer))
    if not 0 < length < len(buffer) or not ntpath.isabs(buffer.value):
        raise OSError("Windows system directory unavailable")
    return ntpath.join(buffer.value, "UIAutomationCore.dll")


def _read_text_before_caret(count: int, api) -> str | CaretReadStatus | None:
    owned = []

    def own(pointer):
        owned.append(pointer)
        return pointer

    try:
        try:
            uia = api.client.GetModule(_system_uia_path())
            automation = own(api.client.CreateObject(uia.CUIAutomation, interface=uia.IUIAutomation))
        except Exception as exc:
            _warn_unavailable(f"UIA initialization failed: {type(exc).__name__}")
            return None
        element = own(automation.GetFocusedElement())
        raw_pattern = own(element.GetCurrentPattern(uia.UIA_TextPattern2Id))
        pattern = own(raw_pattern.QueryInterface(uia.IUIAutomationTextPattern2))
        active, caret = pattern.GetCaretRange()
        own(caret)
        if not active or not caret:
            return None
        selection = own(pattern.GetSelection())
        if selection.Length > 1:
            return CaretReadStatus.UNSAFE
        if selection.Length == 0:
            return None
        selected = own(selection.GetElement(0))
        if selected.CompareEndpoints(uia.TextPatternRangeEndpoint_Start,
                                     selected, uia.TextPatternRangeEndpoint_End) != 0:
            return CaretReadStatus.UNSAFE
        before = own(caret.Clone())
        before.MoveEndpointByUnit(uia.TextPatternRangeEndpoint_Start,
                                  uia.TextUnit_Character, -count)
        text = before.GetText(-1)
        # A readable empty field / count-zero probe returns "", not unknown.
        return str(text) if isinstance(text, str) else None
    except Exception:
        # Exceptions and their tracebacks must also stay on the worker.
        return None
    finally:
        # Include the intermediate IUnknown from GetCurrentPattern. Assigning
        # None alone cannot release pointers retained by ctypes/traceback cycles.
        for pointer in reversed(owned):
            try:
                _release(pointer)
            except Exception:
                # A failed provider must not prevent releasing other pointers.
                pass
        owned.clear()
