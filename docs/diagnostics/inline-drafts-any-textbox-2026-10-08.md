# Inline drafts in any text box

Research date: 2026-10-08. Follow-up to
[windows-text-insertion-2026-10-08.md](windows-text-insertion-2026-10-08.md).
Since `ce67c94`, drafts default to the overlay (`inline_drafts: false`). This
study asks how drafts can be typed into the target and rewritten safely in
any editable control.

## Why the current inline mode cannot be made universal

At the time of this study, inline mode typed a draft, then rewrote it with Shift+Left × N and
Backspace. This is correct only if the N characters before the caret still
match the typed draft. Nothing guarantees that:

- the editor transforms input: autocomplete, auto-closing pairs, autocorrect,
  smart quotes, auto-indent (VS Code/Monaco, Word, browser editors);
- the target processes input asynchronously, so the deletion keys can arrive
  before the pasted text has landed;
- the user moves the caret or types in the same control
  (`same_input_target` only compares HWNDs);
- Shift+Left selects by grapheme or by line in some editors and not by
  UTF-16 unit.

Better timing or verification does not solve this problem. Simulated keys only
send intent: the program never holds a reference to the text it inserted.
Dragon has the same limitation. It reads and sets text directly only in
"standard" controls (Edit, RichEdit, HTML fields). Elsewhere it sends
keystrokes, and Nuance documents that selection and correction fail once the
user touches the mouse or keyboard. Its workaround for those controls is a
separate Dictation Box: overlay-only drafts plus one transfer of the final text,
which is our new default ([Nuance][dragon-box]).

## The mechanism editors already support: TSF compositions

Every editor that accepts Chinese, Japanese or Korean input supports
*compositions*. A composition is a range of provisional text owned by a text
service. The service rewrites the range while the user is still entering text,
then commits it. Microsoft's documentation uses speech as its example:

> While the user is speaking, the speech text service creates a composition.
> This composition will remain intact until the entire speech input is
> complete. [...] the application should not perform any spelling or grammar
> checking on any composition text. ([Microsoft: Compositions][ms-comp])

This addresses the problems listed above:

| Current problem | With a composition |
| --- | --- |
| Shift+Left assumes the caret did not move | The range is a TSF object anchored in the document, and edits elsewhere shift its anchors |
| Autocomplete/autocorrect alter the draft | Applications are expected to suspend such processing in compositions |
| `...` and provisional text look final | The text service supplies display attributes, normally rendered as an underline |
| Focus change or click elsewhere | The application terminates the composition and the service receives `ITfCompositionSink::OnCompositionTerminated` |
| Clipboard is overwritten | No clipboard is involved: `ITfRange::SetText` in an edit session |

Key APIs: `ITfContextComposition::StartComposition`, `ITfRange::SetText`
within `ITfContext::RequestEditSession`, `ITfComposition::EndComposition`
([Microsoft: Compositions][ms-comp]).

### Coverage

- **TSF applications**, such as Chromium and therefore Chrome, Edge,
  Electron/VS Code ([Chromium `TSFTextStore`][cr-tsf]), WPF, WinUI/UWP, Office
  and Windows Terminal. Windows Terminal rewrote its IME integration in 1.21
  ([WT 1.21][wt121]).
- **IMM32-only applications** (classic Win32, many Qt/Java/game toolkits) via
  CUAS, which bridges TSF text services to applications that do not use TSF.
  The switch is `HKLM\Software\Microsoft\CTF\SystemShared\CUAS`
  ([Mozc `imm_util.cc`][mozc-imm]). Test before assuming specific behavior.
- **Store/packaged apps**: a TIP running there inherits the AppContainer
  restrictions. It must be TSF-based and digitally signed, or Windows blocks
  it ([Microsoft: IME requirements][ms-ime-req], [Windows 8 cookbook][ms-w8]).
- **Not covered**: applications that read raw keyboard input (games, some
  remote-desktop/VM clients), protected processes, and the secure desktop.
  These continue to use overlay plus paste.

The set of controls that support IMEs is the closest approximation of "any
text box" that Windows offers.

## Precedents

| Project | What it does | Relevance |
| --- | --- | --- |
| [Plover TIP][plover] (GPL-2+) | Plover is a stenography engine that, like us, retracts and rewrites output. It replaced "SendInput backspace storms" with a native x64 TIP doing `ITfRange::SetText` composition rewrites, controlled from Python over a named pipe ([tip README][plover-tip]) | Nearly the same problem. Protocol: JSON lines over `\\.\pipe\plover_tip_v1_<sid>`, ops `set_composition`, `commit`, `finalize`, `cancel`, `undo_committed`, `focus`/`blur`; pipe DACL grants the user and ALL APPLICATION PACKAGES |
| [OpenLess PR #214][openless] (closed, not merged) | A dictation app committing ASR text through its own TIP; x64 + Win32 DLLs over a named pipe; switches the IME profile during an insertion and restores it afterward | Shows practical failures: 32-bit hosts need a Win32 DLL; Word rejects synchronous edit sessions (`TF_E_SYNCHRONOUS`, fixed with asynchronous sessions); the profile-restore code had a GUID round-trip bug |
| [PIME][pime] (LGPL-2.1) | TSF DLL + launcher + Python/Node backends over named pipes | Mature architecture with a C++ DLL and Python backend |
| Mozc, Microsoft SampleIME | Full TIPs | Permissively licensed reference code; this project is MIT, so do not copy Plover's GPL code |

## Proposed design

### Components

1. **`npu_whisper_tip.dll`**, x64 and x86, C++ or Rust (`windows` crate). This
   is a minimal keyboard TIP: it passes keys through, does not change the
   layout, and only rewrites a composition on request.
2. **Pipe server in the engine**, `\\.\pipe\npu_whisper_tip_<sid>`, with
   JSON lines. The engine is the server, and every TIP instance (one per
   process with focus) is a client.
3. **Inline-draft adapter** in `dictation_engine.py`, replacing
   `type_draft_text`/`delete_text` when a TIP client owns the focused
   context.

### Protocol (draft)

```
TIP -> engine  {"op":"focus","pid":1234,"hwnd":...,"ctx":7}
TIP -> engine  {"op":"blur","ctx":7}
engine -> TIP  {"op":"set","seg":42,"seq":3,"text":"Olá tudo"}     # start or rewrite
engine -> TIP  {"op":"commit","seg":42,"seq":4,"text":"Olá, tudo bem? "}
engine -> TIP  {"op":"cancel","seg":42}                             # empty final
TIP -> engine  {"op":"ack","seg":42,"seq":3}
TIP -> engine  {"op":"terminated","seg":42,"text":"Olá tudo"}      # app ended it
```

- `set` rewrites the full composition text. Unlike prefix diffs, it does not
  depend on what was typed before.
- `commit` writes the final text into the composition range and ends it, as one
  edit session.
- `terminated` occurs when the user clicks elsewhere, types, or the app
  cancels. The engine stops tracking the draft. Policy question: either leave
  the committed text and paste the final at the new caret (the current
  focus-change behavior), or keep an `ITfRange` for the terminated text and
  replace it if its content is unchanged when the final arrives. Plover's
  `undo_committed` shows that ranges remain usable after the commit.
- Edit sessions use `TF_ES_READWRITE | TF_ES_ASYNCDONTCARE`. `TF_ES_SYNC` can
  fail with `TF_E_SYNCHRONOUS`, and `DoEditSession` may never run if the
  context is destroyed, so the engine waits for `ack` with a timeout
  ([Microsoft: RequestEditSession][ms-res]).

### Per-segment routing

```
draft arrives
  ├─ TIP client is focused and acked recently  -> set/commit composition
  ├─ inline_drafts == "keys" (legacy, opt-in)   -> current Shift+Left path
  └─ otherwise                                   -> overlay; final pasted once
```

Choosing this per segment, based on the focused context, provides the
fallback: TSF/IMM controls get inline drafts, others get the overlay. No
per-application configuration is needed.

### Activation: the main risk

A keyboard TIP runs only while its profile is the active input method. Options:

- **A. User selects "NPU Whisper" as the input method**, using Win+Space.
  This is the simplest and most reliable option and is what Plover requires.
  Register one profile per language the user has, with that language's
  layout as `hklSubstitute`, so that typing is unchanged. Cost: one entry
  in the input-method list, plus the user must select it.
- **B. The engine activates the profile only while dictating** through
  `ITfInputProcessorProfileMgr::ActivateProfile` with `TF_IPPMF_FORSESSION`
  ("all threads in the current desktop") and restores the previous profile
  afterward ([Microsoft: ActivateProfile][ms-activate]). This is the
  OpenLess approach. It is transparent, but activation from another process,
  timing on the first draft, and interaction with CJK users' IMEs are
  unverified. Plover warns that changing the HKL under a TIP causes keys to
  repeat (`uuuuufffff`).

Recommended: implement A first because it is deterministic, then test B in the
spike.

### Packaging and security

- Native DLLs (x64 + x86) are built in CI, signed (required in Store apps),
  and shipped inside the wheel. Registration runs as a separate, explicit
  administrator command (`npu-whisper-cli tip install|uninstall`:
  `DllRegisterServer`, `RegisterProfile`, `RegisterCategory`), never at
  startup; AGENTS.md forbids runtime installs. Updating a DLL that running
  apps have loaded requires closing those apps or using a versioned path.
- Free signing options (researched 2026-10-08):
  - **SignPath Foundation**: free for OSI-licensed open-source projects with
    an automated build from the public repository (GitHub Actions). The key
    remains in SignPath's HSM, the certificate names the Foundation as
    publisher, and the project must publish a code-signing policy. PIME, a
    TSF IME, is a listed project ([SignPath][signpath], [terms][signpath-terms],
    [PIME on SignPath][signpath-pime]). This is the preferred option.
  - **Certificate generated during installation**: the administrator command
    creates a self-signed certificate, signs the shipped DLLs, trusts the
    certificate on that machine, and deletes the private key, so the key
    cannot sign other files. This is free and needs no external approval, but
    the trust is local only. Smart App Control and Store-app loading
    still need testing.
  - **Unsigned**: Microsoft's Windows 8 guidance requires signing third-party
    IMEs and says Windows blocks noncompliant ones in Store apps
    ([cookbook][ms-w8]); the current requirements page no longer mentions
    signing ([requirements][ms-ime-req]). The spike should test whether
    unsigned DLLs load in desktop and packaged apps.
  - Not free or not eligible: Azure Artifact Signing costs about US$9.99 per
    month and accepts individuals only from the USA/Canada
    ([Microsoft][ms-signing]); Certum's open-source certificate is paid.
- The TIP is loaded into every application that has focus, so a crash there
  crashes the user's application. The DLL must be minimal and never block the
  UI thread: pipe I/O happens on a worker thread, and work is marshaled to
  the TSF thread through a message window, as in Plover.
- The engine currently runs elevated, but the TIP runs at medium IL or in
  AppContainer. The pipe DACL must explicitly allow those clients. Draft text
  is sensitive, so the engine should send it only to the client whose
  `pid` owns the foreground window (`GetWindowThreadProcessId`) and ignore
  others. CTF has a history of cross-process vulnerabilities
  (CVE-2019-1162, [Project Zero via THN][ctf-cve]), so the pipe must not
  accept commands from the TIP beyond the events above.

## Interim option without native code: verified key rewrite

UI Automation can verify state before deleting text.
`IUIAutomationTextPattern2::GetCaretRange` returns the caret, and its
`isActive` output reports whether the control has focus
([Microsoft: GetCaretRange][ms-caret]). Before Shift+Left, read the text
immediately before the caret and confirm that the selection is empty. Delete
only if that text ends with the typed draft; otherwise, consider the draft
lost and paste the final.

This would fix "corrupted text" (it becomes "draft left behind") in controls
that expose TextPattern (Notepad, Word, Chromium with UIA enabled since
Chrome 126, [Chrome UIA][cr-uia]). It does not fix autocomplete/auto-close
(the check fails and inline drafts are abandoned) or editors with partial
accessibility trees (Monaco exposes only nearby lines). A race also remains
between the check and the keys. This option is useful as a guard for the
legacy `keys` mode, not as a path to "any text box".

**Implemented (2026-10-12):** `WindowsPlatform.text_before_caret` uses
`comtypes` and the UIAutomationCore type library to check an active
TextPattern2 caret and a single empty selection. The entire previous draft
must match exactly before rewriting even its differing tail. A mismatch
leaves the draft, abandons inline output for that segment, and pastes the
final in full. The initial implementation preserved blind rewriting for
unknown reads; the terminal fallback below supersedes that behavior.
The caller waits at most 150 ms on a single daemon MTA worker, including
first-use type-library generation. Only one query may run at a time: a hung
provider makes subsequent reads unknown without accumulating workers; late
results are discarded. UIA pointers are explicitly released and cleared on
the worker, including intermediate interfaces and failure paths, so later
cyclic garbage collection cannot release them again on another thread.

**Terminal fallback (2026-10-13):** The reported Claude Code TUI output
(`InDrafts ligado. Este texto aqui já é usando ele. drafts ligado. Este texto
aqui já é usando... ne Drafs liga. Este texto aqui.  este texto aqui. . ...`)
had no "target changed" log. UIA returned `None`, so the old unknown-read path
sent Shift+Left even though the TUI interpreted it as caret movement without
selection. Each replacement spliced another draft into the buffer.

The strategy is now selected once per segment, before its first inline draft,
using the same bounded UIA worker. A string, including `""`, selects the existing
exact-match verification path. If that target later becomes unreadable, inline
output is abandoned; blind Shift+Left is never used. `None` selects the original
`da84190`/`0218339` common-prefix algorithm: erase only the differing tail with
one Backspace per character and type the new draft tail as Unicode key packets,
without a clipboard transaction. The final tail still supports clipboard paste.
This reverses the terminal-incompatible deletion introduced in `2d0afd6` and
avoids the draft paste path introduced in `03d4939` for unreadable targets.
The segment logs once: `Inline draft: target unreadable; trimming with Backspace`.

With `inline_drafts_require_verification: true` (default `false`), an unreadable
target receives no inline drafts; the final is typed once. The segment logs once:
`Inline draft: target text is unreadable; drafts stay in the overlay`. Unsafe
selections also keep drafts in the overlay. Final-only segments never probe;
fallback finals do not repeat the probe or wait for a timed-out UIA worker.

The fallback allows only simple BMP characters: non-BMP characters/surrogates,
combining marks, controls/joiners, and Hangul Jamo disable inline drafts for
the rest of the segment. A preceding simple draft is erased if focus matches,
then the final is typed once. This avoids guessing whether a terminal deletes
code units, code points, or grapheme clusters. Precomposed Portuguese accents
remain supported. The same-input-target check applies before every deletion;
a focus change leaves the old draft and abandons inline output for the segment.
Caret movement and user edits within an unreadable target remain undetectable.

Regression tests simulate a terminal where Shift+Left moves without selecting,
including successive growing Portuguese drafts followed by a final. They check
exact tail Backspace counts, Unicode draft insertion, no leftover text, strict
overlay mode, empty readable fields, and unchanged readable-target verification.
The worker keeps COM initialized for the process lifetime; it never calls
`CoUninitialize`, which could invalidate comtypes' internal type-library
cycles. Tests block real UIA at the module boundary and inject fake APIs.
Missing `comtypes` is also unknown. This follows
[Microsoft's UIA threading guidance](https://learn.microsoft.com/en-us/windows/win32/winauto/uiauto-threading)
and leaves the check-to-keys race described above.

## Word duplication after PR #47 (2026-10-14)

The user reports correct Notepad output but a duplicated full sentence in Word,
with `Inline draft: target changed; leaving the draft and typing the final`.
This establishes that the initial readability probe succeeded. It does **not**
establish what Word returned during verification: the old message covered
different strings, a later timeout/unknown read, and unsafe selections alike.
No captured Word UIA strings are available, so the exact representation that
triggered the report remains unconfirmed; the following regressions use fake
UIA reads, not a live Word reproduction.

The confirmed code defect was exact string equality followed by forgetting the
draft and typing the full final. Even a harmless representation difference
therefore left both versions in the document. The earlier exact-match and
paste-final policies described above are superseded by this fix.

Evidence checked against Microsoft documentation:

| Candidate | Evidence and treatment |
| --- | --- |
| Paragraph/line/cell marks (`\r`, `\v`, `\x07`) or paragraph end in the read | Word's [template example](https://learn.microsoft.com/en-us/office/vba/api/word.template.openasdocument) identifies a paragraph mark as `Chr(13)`. This describes Word's object model, not a guarantee about UIA. [GetCaretRange](https://learn.microsoft.com/en-us/windows/win32/api/uiautomationclient/nf-uiautomationclient-iuiautomationtextpattern2-getcaretrange) supplies the caret range; our code clones it and moves only Start backward. It does not expand to a paragraph or move End forward. Extra terminal marks are a simulated provider difference, not a proven Word UIA behavior. |
| NBSP, narrow/thin and other Unicode spaces | These failed exact equality. Unicode `Zs` characters now compare as spaces, one for one. Interior whitespace runs are not collapsed. |
| Smart quotes/apostrophes | [Microsoft documents Word's automatic straight-to-smart quote replacement](https://support.microsoft.com/en-au/word/smart-quotes-in-word-and-powerpoint). Curly single/double quotes now compare to their straight equivalents. |
| Auto-capitalization, double hyphens, ellipsis, other AutoCorrect | Word documents [sentence capitalization](https://learn.microsoft.com/en-us/office/vba/api/word.autocorrect.correctsentencecaps) and [double-hyphen conversion](https://learn.microsoft.com/en-us/office/vba/api/word.options.autoformatasyoutypereplacesymbols). These are not evidence that either occurred in this report. We do not case-fold, expand symbols, or accept arbitrary spelling changes as proof of ownership. |
| Hidden field/bookmark text | [GetText includes hidden as well as visible text](https://learn.microsoft.com/en-us/dotnet/api/system.windows.automation.text.textpatternrange.gettext). Unknown controls, field delimiters and zero-width characters are not stripped. We have no evidence of a bookmark character in this Word read. |
| Stale UIA read after paste/SendInput | Possible but unconfirmed. Retry once after 40 ms, unless the selection is explicitly unsafe or focus changed. |
| Partial result from the 150 ms budget | The worker returns a completed string or `None`; a timeout cannot return a partly filled string. `GetText(-1)` does not request truncation. [MoveEndpointByUnit](https://learn.microsoft.com/en-us/windows/win32/api/uiautomationclient/nf-uiautomationclient-iuiautomationtextrange-moveendpointbyunit) can move fewer units at a document boundary or substitute a larger supported unit, so returned string length is not necessarily the requested Python length. Late worker results remain discarded. |

Comparison now recognizes normalized suffixes: CR/LF/CRLF, vertical tab and
Unicode line/paragraph separators compare as line breaks; Unicode spaces and
smart quotes compare as above. Terminal whitespace and cell marks can explain
equivalence, but equivalence alone does not authorize deletion. A retry requests
eight additional characters of context so terminal marks do not hide the
beginning of the expected draft. Each read retains the existing 150 ms bound;
two slow reads plus the delay cost at most about 340 ms of waiting. A still-busy
worker immediately returns unknown and does not queue another request.

**Deletion rule:** only a suffix whose normalized text matches with one-for-one
character counts uses the existing common-prefix/tail rewrite. Focus is checked
before and after each read, including after the retry delay. A persistent
trimmed trailing space, extra terminal paragraph/cell mark, or CRLF expansion
does not prove how many key presses select our text. For example, UIA may hide
the final space, or the editor may actually have removed it. Guessing the old
length in the latter case can delete preceding user text. These ambiguous
cases keep the draft, even when their normalized content matches. Thus this fix
prevents duplication for these cases but does not promise a corrected final in
Word when the editable range is still ambiguous.

**Unreconciled segment policy:** retain the existing draft; stop inline updates
and suppress the final paste and automatic Enter for the remainder of the
segment. This also applies after focus changes. Keep the full final in history
and the normal final-transcription log for recovery. Reset the suppression for
the next segment, including after an empty or failed final. There is no downgrade
from verified rewriting to blind Backspace based on HWND/runtime ID: identity
alone cannot establish caret position or draft length. The existing unreadable
terminal strategy remains selected only by the initial probe. A check-to-keys
race still exists; normalization is not an owned TSF composition range.

The first non-exact or unsafe verification per segment logs metadata only:
`expected_len`, `read_len` (`None` when no text was returned), `read_status`
(`text`, `unsafe`, or `unavailable`), `representation_changed` (whether
normalizing the observed text changes it), `equivalent` (normalized suffix
equivalence), and `safe_count` (whether the read authorizes the deletion count).
This includes a stale read that later recovers. Persistent failures additionally
log that the draft was retained and the final paste suppressed. Neither
`expected_tail` nor `read_tail` is logged, and no editor text is included in
these diagnostics. The metadata distinguishes missing reads, representation
changes, and unsafe counts, but cannot identify the exact characters Word
returned or which substitution caused a mismatch. That requires a controlled
reproduction with non-sensitive sample text and inspection of the caret read.

Regression tests cover successful one-for-one rich-editor rewrites, trailing
paragraph/cell marks and space trimming without duplication or over-deletion,
stale-then-fresh reads, changed text/hidden delimiters, focus changes during a
read or retry, once-per-segment diagnostics, and final/empty/error resets.

Validation in this worktree (2026-10-14):

- Focused: `test_draft_typing.py`, `test_windows_uia.py`,
  `test_draft_coalescing.py`, `test_app_shutdown.py`: **198 passed, 1 skipped**.
- Full suite: **1518 passed, 1 skipped, 2 failed**. Both failures are in
  `test_llm_process.py` (`test_hung_load_is_killed` and
  `test_crashed_process_is_started_again`), timing out while waiting for a
  subprocess to exit. Both reproduce with the committed `HEAD` engine loaded
  instead of the modified engine.
- Used the requested virtualenv. The direct pytest command hit Windows sandbox
  ACL errors in temporary directories. Successful runs used a process-local
  runner that preserves inherited ACLs for newly created test directories in
  the worktree; no test or application permissions code was changed.
- No live Word verification was performed. The mismatch diagnostic is still
  needed to confirm which representation Word returned in the reported case.

## Spike plan

1. Minimal TIP (x64) using SampleIME/Mozc as reference: one profile for
   pt-BR/ABNT2, a pipe client, `set`/`commit`/`cancel`. Python test client
   that sends a scripted draft sequence.
2. Matrix, recording `ack` latency and final document text: Notepad, WordPad
   replacement/RichEdit, Word, Chrome `<textarea>` and `contenteditable`,
   Google Docs, VS Code editor and integrated terminal, Windows Terminal
   (Claude Code), Slack/Teams (Electron/WebView2), a Win32 Edit control
   (CUAS), a Qt app, a 32-bit app.
3. Cases: caret moved mid-draft, typing during composition, focus change,
   autocomplete popup open, `auto_enter`, emoji/accents, two drafts per
   second.
4. Activation B: `ActivateProfile(FORSESSION)` from the engine process while
   another app is focused; measure the time until the TIP connects.

Decision gate: if the matrix shows inline compositions working in the
editors that currently fail (VS Code, browser editors, Word), proceed with
x86 build, signing, and packaging. Otherwise, keep overlay-only drafts and
add only the UIA guard.

[dragon-box]: https://www.nuance.com/products/help/dragon15/dragon-for-pc/enx/professionalindividual/Content/Dictation/about_the_dictation_box.htm
[ms-comp]: https://learn.microsoft.com/en-us/windows/win32/tsf/compositions
[cr-tsf]: https://chromium.googlesource.com/chromium/src.git/+/71.0.3555.2/ui/base/ime/win/tsf_text_store.h
[wt121]: https://github.com/microsoft/terminal/releases/tag/v1.21.1272.0
[mozc-imm]: https://code.googlesource.com/mozc/+/HEAD/src/win32/base/imm_util.cc
[ms-ime-req]: https://learn.microsoft.com/en-us/windows/apps/develop/input/input-method-editor-requirements
[ms-w8]: https://learn.microsoft.com/en-us/windows/win32/w8cookbook/third-party-input-method-editors
[plover]: https://github.com/sam0737/plover-windows-input-method
[plover-tip]: https://raw.githubusercontent.com/sam0737/plover-windows-input-method/main/windows/tip/README.md
[openless]: https://github.com/Open-Less/openless/pull/214
[pime]: https://github.com/EasyIME/PIME/
[ms-res]: https://learn.microsoft.com/en-us/windows/win32/api/msctf/nf-msctf-itfcontext-requesteditsession
[ms-activate]: https://learn.microsoft.com/en-us/windows/win32/api/msctf/nf-msctf-itfinputprocessorprofilemgr-activateprofile
[ctf-cve]: https://thehackernews.com/2019/08/ctfmon-windows-vulnerabilities.html
[ms-caret]: https://learn.microsoft.com/en-us/windows/win32/api/uiautomationclient/nf-uiautomationclient-iuiautomationtextpattern2-getcaretrange
[cr-uia]: https://developer.chrome.com/blog/windows-uia-support
[signpath]: https://signpath.org/
[signpath-terms]: https://signpath.org/terms
[signpath-pime]: https://signpath.org/projects/pime
[ms-signing]: https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/code-signing-options
