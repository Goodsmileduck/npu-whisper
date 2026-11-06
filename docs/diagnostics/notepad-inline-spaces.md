# Notepad inline drafts: spaces after the common prefix

The report has two draft completions (1.6 s and 2.6 s of audio), followed by
the final `Oi Débora, tudo bem?`. Only the first draft, `Oi Débora,`, was
provided; draft 2 is absent from the app log. Exact reconstruction of the
user's run is therefore impossible. The tests and replay cover two plausible
draft-2 strings, without claiming either was the actual recognition result.

## Reconstructed PR #45 output

For draft 2 `Oi Débora, tudo bem`:

1. Draft 1 sends Unicode packets for `Oi Débora, ` (10 UTF-16 units).
2. Draft 2 retains that entire prefix, deletes nothing, and sends Unicode
   packets for `tudo bem... ` (12 units, including synthetic ellipsis/space).
3. Final retains `Oi Débora, tudo bem`, sends four Backspace down/up pairs to
   erase `... `, then pastes only `? ` with Ctrl+V. Clipboard restoration
   completes before the output lock is released.

For draft 2 `Oi Débora, tudo bem?`, step 2 sends `tudo bem? ` (10 units).
The final equals the formatted draft: **no Backspace and no final paste**.
The complete final-transcription log does not mean that complete text was
pasted. Either sequence can retain corrupted draft letters in its common
prefix because the unreadable strategy cannot verify what Notepad inserted.

For every UTF-16 unit `u`, `_type_text_ctypes` sends one two-element INPUT
array: `(type=1, wVk=0, wScan=u, flags=4, time=0, extra=0)`, followed by
`(type=1, wVk=0, wScan=u, flags=6, time=0, extra=0)`. INPUT includes the mouse
union and is 40 bytes on Windows x64, 28 on x86. `é` is `0x00e9`; spaces
are `0x0020`. No VK_SPACE is assigned. This matches Microsoft's
[KEYBDINPUT contract](https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-keybdinput).

`git show da84190:dictation_engine.py` has the same Unicode packet builder,
per-unit 2 ms delay, Backspace-per-character deletion and common-prefix diff.
Its clipboard restoration was detached; current restoration is serialized.
The locally installed `keyboard/_winkeyboard.py` explicitly bypasses
`VK_PACKET` in its low-level hook and passes it to `CallNextHookEx`. There
is no evidence here of keyboard mapping these packets to VK_SPACE, wrong
key-up flags, missing Unicode scan fields, or an off-by-one deletion count.

## What is fixed, and what remains unconfirmed

Missing `comtypes` silently routed Notepad into an unverified path. That path
sent drafts as individual Unicode packets and then trusted them when keeping
the common prefix. It is the implicated insertion path, **not a confirmed
explanation for why Notepad displayed spaces**. The supplied log cannot
distinguish input rejection, target processing, or external input/hook effects.

The mitigation uses the existing serialized clipboard insertion for both
draft and final tails, while retaining Backspace-only deletion for unreadable
targets. UIA installation/initialization failure now emits one actionable
warning; an unsupported target alone does not emit that warning. Editable
install dependency refresh is documented in README.

Mocked regressions assert the complete legacy SendInput sequence and the new
copy / Ctrl+V / target-read / restore / Backspace sequence, plus the expected
final text. They do not emulate Windows 11 Notepad. Clipboard delivery still
uses a fixed delay; neither method proves that an unreadable target accepted
the intended text. No real desktop reproduction was performed for this patch.

## Manual desktop check

Pause dictation and use a fresh, empty Notepad tab for each run. From this
checkout, use an environment containing the project's dependencies:

```powershell
python docs/diagnostics/replay_inline_drafts.py --mode unicode
python docs/diagnostics/replay_inline_drafts.py --mode unicode --send
python docs/diagnostics/replay_inline_drafts.py --mode clipboard --send
```

The first command only prints the plan. For `--send`, focus the empty target
within five seconds and release all modifiers. No Enter is sent. Repeat in
the terminal's empty input field, then repeat both modes with
`--second 'Oi Débora, tudo bem?'`. Use `--native-backspace` to compare direct
VK_BACK against the keyboard library's scan-code mapping. The script aborts
between steps if focus changes and reports every actual SendInput structure
and return value. A successful return means events entered the input stream,
not that Notepad accepted the text; see Microsoft's
[SendInput contract](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput).

Save the console trace and actual pasted/copied result, including `repr(text)`
or code points to distinguish U+0020 spaces from invisible/rendered characters.
Record Windows/Notepad versions, keyboard layout, and whether the app's hotkey
hooks were running. If only Unicode mode fails, that confirms the transport
mitigation; identifying the spaces' native cause still requires observing
Notepad's input handling on the affected desktop.
