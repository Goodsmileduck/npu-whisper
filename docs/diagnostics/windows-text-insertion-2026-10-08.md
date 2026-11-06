# Windows dictation: inserting and revising text globally

Research date: 2026-10-08. Trigger: spoken `Olá, tudo bem?` became
`Olá,bbbbbbbem?` in Windows 11 Notepad. A later dictated message contained
fragmented punctuation and a repeated clause. No recording of those examples
is available, so neither the original ASR output nor the exact point of
corruption can be reconstructed from the reported text alone.

## Existing approaches

| Approach | What it provides | Limits for this project |
| --- | --- | --- |
| Clipboard plus paste shortcut | Inserts a whole Unicode string at the current selection; Handy uses it by default on Windows. | Delivery and restoration are asynchronous from the target's perspective. This does not establish ownership of an editable draft range. |
| Simulated Unicode keys | Inserts text without changing the clipboard. | Must check delivery, account for target processing, and handle failures. Simulating deletion still assumes the caret and document have not moved. |
| UI Automation | Can inspect text, selection, and supported control patterns. `ValuePattern` can set the value of supporting controls. | `TextPattern` is read-only; setting a whole value is not a universal replacement for an insertion or composition API. Support varies by control. |
| Text Services Framework (TSF) | Provides text services, edit sessions, and compositions that can be updated while speech is in progress and committed when complete. | Requires a Windows text-service integration and validation against supported applications; it is not a drop-in Python call that edits every application. |

Primary sources:

- [Handy paste methods](https://handy.computer/docs/paste-methods).
- [Microsoft: KEYBDINPUT](https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-keybdinput).
- [Microsoft: adding text with UI Automation](https://learn.microsoft.com/en-us/dotnet/framework/ui-automation/add-content-to-a-text-box-using-ui-automation).
- [Microsoft: TextPattern and TSF](https://learn.microsoft.com/en-us/dotnet/framework/ui-automation/ui-automation-textpattern-overview).
- [Microsoft: compositions, including the speech-input lifecycle](https://learn.microsoft.com/en-us/windows/win32/tsf/compositions).

Handy's implementation also has an experimental paste transaction using
Windows delayed clipboard rendering (`WM_RENDERFORMAT`). It observes reads
after the paste shortcut, tracks clipboard ownership, and waits before
restoration. Its implementation still has timing/timeout policies: this is
stronger evidence than a fixed sleep alone, not a guarantee that the intended
document was edited. See [Handy's transaction implementation](https://github.com/cjpais/Handy/blob/main/src-tauri/src/paste_tx/mod.rs).

## Streaming recognition is a separate problem

The paper [Turning Whisper into Real-Time Transcription System](https://aclanthology.org/2023.ijcnlp-demo.3/)
and its [Whisper-Streaming implementation](https://github.com/ufal/whisper_streaming)
use local agreement: confirm the prefix shared by successive hypotheses and
keep the unstable remainder provisional. The implementation uses word times
and overlap handling to avoid recommitting recognized audio.

Our character-prefix comparison only computes edits between two strings; it
does not establish agreement across hypotheses or deduplicate overlapping
audio across finalized segments. Fixing text insertion alone cannot establish
that punctuation and repeated clauses originate outside the recognizer.

## Findings in this branch

- The final-text log lived in `type_text`, so equal draft/final strings emitted
  no text log and differing strings logged only the appended tail.
- Drafts used Unicode key injection; finals used clipboard paste with a
  detached restoration thread. A restore could interfere with a later paste.
- Draft formatting adds `...` directly to the destination document. These are
  synthetic display punctuation, not necessarily recognized speech. If a final
  fails or focus changes, that punctuation may remain in the old draft.
- Checking the foreground window and focused control does not detect moving
  the caret or editing text within that same control. Therefore relative
  Shift+Left deletion cannot establish that it still selects the owned draft.
- Final audio extraction adds trailing frames even when there is no trailing
  silence to trim. Duration cuts can consequently read beyond the processed
  audio; the following segment also includes lookback. This needs an explicit
  sample-boundary regression test before attributing the reported duplicate
  clause to this path.

## Immediate changes versus subsequent design

This patch routes drafts through serialized clipboard pastes, restores the
previous clipboard text before returning, preserves a different clipboard text
copied by the user in the meantime, and logs the complete final transcription
from the engine. Restoration still uses a delay and preserves plain text only.
The patch does not implement TSF, clipboard read receipts, or verified ranges.

Recommended next design, based on the sources above:

1. Keep provisional display punctuation out of the destination document.
2. Treat confirmed and provisional transcription separately. For a broadly
   compatible mode, show provisional text in the overlay and paste confirmed
   text; investigate local agreement for lower-latency continuous output.
3. For editable inline drafts, prototype a TSF composition in Notepad and a
   browser editor. Use UI Automation to verify target context where supported;
   avoid replacing the whole document with `ValuePattern.SetValue`.
4. Add sample-position accounting across forced audio cuts and lookback.
   Preserve intentional spoken repetition; do not blindly remove equal words.
5. Validate with the same recorded audio and separate observations of raw ASR,
   finalized engine text, and actual destination text. Include punctuation,
   pauses, intentional repetition, accents/emoji, caret movement, focus changes,
   clipboard changes, and a slow target application.

The reported Notepad corruption has not been reproduced in a controlled UI
test. Mocked tests verify the changed delivery and logging paths, not actual
Notepad behavior or recognition accuracy for the missing audio recording.

Validation: the full suite reported 403 passed in 300.80 seconds and wrote
`.test-results/pytest-dictation-fix.xml`, but the process returned exit code 1
without a reported test failure. An isolated rerun of `test_draft_typing.py`
reported 23 passed with exit code 0. The full-suite shutdown discrepancy remains
unresolved; the full run should not be described as a clean process exit.
