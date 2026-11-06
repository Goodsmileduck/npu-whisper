"""Replay the Notepad report without ASR; dry-run unless --send is supplied.

Run from this checkout with its Windows Python environment. See the adjacent
notepad-inline-spaces.md for the limits of this reconstruction.
"""
import argparse
import ctypes
import os
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("unicode", "clipboard"), default="clipboard")
    parser.add_argument("--second", default="Oi Débora, tudo bem",
                        help="unlogged second draft; also try 'Oi Débora, tudo bem?'")
    parser.add_argument("--send", action="store_true", help="send keys to the focused textbox")
    parser.add_argument("--delay", type=float, default=5, help="seconds to focus an empty textbox")
    parser.add_argument("--native-backspace", action="store_true",
                        help="isolate keyboard's Backspace mapping with direct VK_BACK")
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay must be nonnegative")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from debora_whisper.platforms import windows_text as output

    previous = ""
    steps = []
    for raw, final in [("Oi Débora,", False), (args.second, False),
                       ("Oi Débora, tudo bem?", True)]:
        text = raw.strip()
        text += "... " if not final and not re.search(r"[.,!?;:]$", text) else " "
        if not output.can_backspace_draft(text):
            parser.error("this replay requires simple BMP text, like the reported phrase")
        common = os.path.commonprefix([previous, text])
        count, tail = len(previous) - len(common), text[len(common):]
        method = "clipboard" if final else args.mode
        steps.append((count, tail, method))
        print(f"{'final' if final else 'draft'}: Backspace x {count}; {method} {tail!r}")
        previous = text
    print(f"Expected: {previous!r}")
    if not args.send:
        print("Dry run. Add --send to replay in an empty Notepad tab or terminal input.")
        return
    if sys.platform != "win32":
        parser.error("--send requires Windows")

    import keyboard
    print(f"Backspace scan codes: {keyboard.key_to_scan_codes('backspace')}")
    print(f"Focus an EMPTY textbox within {args.delay:g}s; release all modifiers. No Enter is sent.")
    time.sleep(args.delay)
    target = output.input_target()
    if target is None:
        raise SystemExit("Cannot identify the focused target")
    user32 = ctypes.windll.user32
    native_send = user32.SendInput

    def traced_send(count, ref, size):
        packets = [(p.type, p._input.ki.wVk, p._input.ki.wScan,
                    p._input.ki.dwFlags, p._input.ki.time,
                    bool(p._input.ki.dwExtraInfo)) for p in ref._obj]
        sent = native_send(count, ref, size)
        print(f"SendInput(count={count}, size={size}, packets={packets}) -> {sent}")
        if sent != count:
            raise RuntimeError("SendInput did not accept all events; replay stopped")
        return sent

    erase = output._backspace_text_ctypes if args.native_backspace else output.backspace_text
    with patch.object(user32, "SendInput", traced_send):
        for count, tail, method in steps:
            if not output.same_input_target(target, output.input_target()):
                raise SystemExit("Focus changed; replay stopped")
            erase(count)
            if tail:
                (output._type_text_ctypes if method == "unicode" else output.type_draft)(tail)
            time.sleep(1)
    print("Replay complete. Inspect/copy the actual result; no UIA verification was used.")


if __name__ == "__main__":
    main()
