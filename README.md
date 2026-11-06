# Débora Whisper

![Windows](https://img.shields.io/badge/platform-Windows%2011-0078D4?logo=windows)
![Python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

https://github.com/user-attachments/assets/a9ddebc2-5150-450b-8df5-4471cc705c73

Local voice-to-text dictation for Windows, on Intel NPU or iGPU via OpenVINO, NVIDIA RTX via faster-whisper, or CPU. Press a hotkey, speak, and text appears at your cursor. Zero cloud, zero cost, zero data leaving your machine.

## Features

- **NPU-accelerated** — runs on Intel AI Boost (Meteor Lake / Lunar Lake / Arrow Lake)
- **6 models** — Whisper tiny/base/small/medium/turbo + Parakeet TDT 0.6B (best accuracy)
- **15 languages** — English, Russian, Spanish, French, German, Japanese, Chinese, Korean, and more
- **Dynamic Island overlay** — floating mascot and conversation text; hides when off after the text balloon dismisses
- **System tray** — lives in your taskbar, right-click for settings/history
- **One-click settings** — language-first model picker with download status badges
- **Claude Code mode** — auto-paste + Enter for hands-free prompt submission
- **Audio chimes** — pleasant start/stop feedback
- **Device priority** — `device_priority` in the config (default RTX -> NPU -> iGPU -> CPU) picks the startup device; a DEVICE_LOST moves to the next healthy device on the list

## Requirements

- **Windows 11**, or **Linux** (tested on Ubuntu 26.04 with KDE Plasma on Wayland; see [Linux](#linux))
- **Intel Core Ultra** CPU with NPU (Meteor Lake / Lunar Lake / Arrow Lake), or any Intel CPU with iGPU
- **Python 3.10+**
- **16 GB RAM** recommended (NPU/GPU share system memory)

## Quick Start

### Install as a command (uv)

[uv](https://docs.astral.sh/uv/) installs the app in its own environment and
fetches a matching Python if you have none:

```powershell
winget install astral-sh.uv

# NPU / Intel iGPU / CPU only
uv tool install git+https://github.com/alexandre-machado/debora-whisper

# ...or with the NVIDIA RTX backend (adds ~2 GB of CUDA libraries)
uv tool install "debora-whisper[cuda] @ git+https://github.com/alexandre-machado/debora-whisper"

debora-cli --setup          # detect devices, download the model, warm the cache
debora                      # tray app + overlay
debora-cli                  # console-only mode
```

To launch it from the Start Menu (no console window), optionally also at sign-in:

```powershell
debora --install-shortcut              # Start Menu entry "Débora Whisper"
debora --install-shortcut --autostart  # ...and start with Windows
```

Only one tray app runs at a time; launching a second one exits quietly. The
shortcut points at the Python of the current install: after reinstalling
(`uv tool install --reinstall`, a new Python), run `--install-shortcut` again.

Update with `uv tool upgrade debora-whisper`. To remove, run
`debora --remove-shortcut` first, then `uv tool uninstall debora-whisper`
(models and config stay in `~/.debora/`). Once published to PyPI, the
same works with `uv tool install debora-whisper` or a one-off `uvx debora-whisper`.

### Linux

Tested on Ubuntu 26.04, KDE Plasma 6 (Wayland), PipeWire and an NVIDIA RTX
with the `[cuda]` extra. Other desktops work with less: see the table below.

```bash
sudo apt install wl-clipboard pipx     # the clipboard on Wayland (xclip on X11), and pipx for uv
sudo usermod -aG input "$USER"         # read the keyboards for the hotkey (see the warning below)
echo 'KERNEL=="uinput", SUBSYSTEM=="misc", TAG+="uaccess", OPTIONS+="static_node=uinput"' \
  | sudo tee /etc/udev/rules.d/70-debora-uinput.rules  # type through a virtual keyboard
sudo udevadm control --reload && sudo udevadm trigger /dev/uinput
# log out and back in, so the group and the uinput access apply
pipx install uv && pipx ensurepath  # or uv's own installer, after reading it: https://docs.astral.sh/uv/
uv tool install "debora-whisper[cuda] @ git+https://github.com/alexandre-machado/debora-whisper"
debora-cli --setup
debora --install-shortcut --autostart  # app menu entry + start at login (.desktop files)
```

The global hotkey reads the keyboards in `/dev/input` and types through a
virtual keyboard on `/dev/uinput`. While the app runs, it holds the keyboards
exclusively and mirrors every key except its own hotkey, which never reaches
the app in focus.

> **Security:** the app itself needs no root, but these permissions are broad.
> Membership in the `input` group lets **every program you run** read every
> key you type, passwords and the lock screen included, with no prompt; on
> Wayland that is otherwise reserved to the compositor. The group can also
> write to `/dev/input/event*`, so those programs can inject input too, and
> it applies in every session of that user (SSH or a second seat included),
> not only the one at the local seat. Write access to
> `/dev/uinput` lets a program type and click as you. The rule above grants it
> through `uaccess` only to the user at the local seat, while logged in, which
> is narrower than giving the `input` group write access; do not do the
> latter. Some packages (Steam, for instance) already install the same
> `uaccess` rule. To undo: `sudo gpasswd -d "$USER" input` and remove
> `/etc/udev/rules.d/70-debora-uinput.rules`.

Text is pasted: Ctrl+V, or Ctrl+Shift+V in terminals. On KDE Plasma the app
asks KWin which window has focus and picks the key per window; elsewhere it
uses Ctrl+Shift+V, which most terminals and many other apps accept. Override
per window class with `paste_hotkeys` in the config:

```json
"paste_hotkeys": {"org.kde.konsole": "ctrl+shift+v", "emacs": "ctrl+y", "unknown": "ctrl+v"}
```

| | KDE Plasma (Wayland/X11) | Other desktops |
|---|---|---|
| Hotkey (suppressed) and typing | yes | yes |
| Paste key per window | yes (KWin) | `unknown` key from `paste_hotkeys` |
| Tray icon with menu | yes (StatusNotifierItem) | GNOME needs the AppIndicator extension; otherwise an icon without menu |
| Inline drafts | overlay only (the text before the caret is not readable yet) | same |

`language` defaults to `"en"`, which makes Whisper translate other languages
into English: set your language (`"pt"`, ...) or `"auto"`.

### From a source checkout

```powershell
# 1. Clone
git clone https://github.com/alexandre-machado/debora-whisper.git
cd debora-whisper

# 2. First-time setup (creates venv, installs deps, downloads model, warms NPU cache)
.\Start-Dictation.ps1 -Setup

# 3. Launch (GUI mode with system tray + Dynamic Island overlay)
.\Start-Dictation.ps1
```

**Ctrl+Space** works two ways:

- **Hold** it while you speak (push-to-talk): recording stops when you let go
  and the whole utterance is transcribed and pasted at your cursor.
- **Tap** it to start continuous listening: voice activity detection types each
  sentence as you finish it (a rising two-note chime confirms). Tap again to
  stop; a sentence in progress is still typed.

Set `"tap_action": "toggle"` to make a tap start a recording that the next tap
stops instead.

## Models

| Model | Params | Device | Speed (3s audio) | WER | Languages | Notes |
|-------|--------|--------|-------------------|-----|-----------|-------|
| **tiny** | 39M | NPU | — | — | All 15 | Lowest resource usage; useful for quick commands and CPU/NPU fallback |
| **base** | 74M | NPU | 0.2s | 5.0% | All 15 | Default, great for short commands |
| **small** | 244M | NPU | 0.6s | 3.4% | All 15 | Balanced speed/accuracy |
| **parakeet** | 600M | NPU+GPU | 0.2s | 3.7% | en, es, fr, de, it, nl, pl, pt, ru, uk (10 of 15) | Fast hybrid pipeline; validate Brazilian Portuguese separately |
| **medium** | 769M | GPU | — | 2.9% | All 15 | High accuracy, slower |
| **turbo** | 809M | GPU | 2.4s | 2.3% | All 15 | Best multilingual quality |

> **WER** = Word Error Rate (lower is better), publisher-reported, not independently reproduced here. Whisper WER on LibriSpeech test-clean from [HuggingFace model cards](https://huggingface.co/openai/whisper-base). Parakeet WER (3.7% LibriSpeech test-clean, 17.0% FLEURS multilingual) from the [NVIDIA model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

Parakeet's upstream checkpoint (`nvidia/parakeet-tdt-0.6b-v3`) supports 25 European languages with automatic language identification (no language token needed at inference). This app exposes the intersection of that list with its own 15-language picker: English, Spanish, French, German, Italian, Dutch, Polish, Portuguese, Russian, Ukrainian. It does not cover Japanese, Chinese, Korean, Turkish, or Arabic, which stay Whisper-only.

For long-form audio, prefer Whisper turbo/medium on GPU or Whisper small on
NPU until Parakeet chunking is implemented. The current Parakeet integration
uses static encoder buckets up to approximately 16 seconds; longer inputs are
truncated and logged rather than transcribed in full. The published Parakeet
results also should not be assumed to represent Brazilian Portuguese, because
the upstream evaluation notes distinguish European Portuguese.

The Settings dialog shows which models are already downloaded and filters them by your selected language (e.g., selecting Japanese hides Parakeet, since the checkpoint wasn't trained on it; selecting Russian now shows Parakeet).

## Usage

### GUI Mode (default)

```powershell
.\Start-Dictation.ps1
```

Launches with a Dynamic Island overlay at the top of your screen and a system tray icon. The overlay hides when ready with listening off, after any conversation text dismisses. Press Ctrl+Space to record; Settings and Quit remain available from the tray.

### CLI Mode

```powershell
.\Start-Dictation.ps1 -CLI
```

Console-only, no GUI. Same hotkey gestures as the GUI.

### Claude Code Mode

```powershell
.\Start-Dictation.ps1 -AutoEnter
```

Pastes text and presses Enter automatically — speak your prompt and it submits to Claude Code.

### Override Settings

```powershell
.\Start-Dictation.ps1 -Device GPU -Model turbo -Language ru
```

## Configuration

Stored at `~/.debora/config.json`:

```json
{
  "device_priority": ["CUDA", "NPU", "GPU", "CPU"],
  "model_size": "turbo",
  "language": "en",
  "hotkey": "ctrl+space",
  "voice_chat_hotkey": "right alt",
  "tap_action": "continuous",
  "auto_enter": false,
  "update_check": true,
  "inline_drafts": false,
  "inline_drafts_require_verification": false,
  "beep_on_start": true,
  "max_record_seconds": 60,
  "sample_rate": 16000
}
```

When unset, `language` follows the system UI language (falling back to `"en"`); a fixed language can make Whisper translate speech in another language.

The tray and Settings show the running package version and update status.
Editable Git installs also show the startup commit, date and `+dirty` for
local changes. Update checks run in the background at startup and every six
hours: Git installs compare against `main` on the branch's upstream remote
(or `origin`); regular installs compare against PyPI. Checks never fetch,
pull or install anything. Set `update_check: false` to disable remote checks;
the local build identity remains visible. Offline checks show an unknown
status. `app.log` records the build and each check's result.

`device_priority` is the order devices are tried in. At startup the app uses
the first one present on the machine that can run the model (`CUDA` = NVIDIA
RTX via faster-whisper, Whisper models only; `GPU` = Intel iGPU). When the
active device is lost, it falls back to the next healthy one. `-Device` on the
command line overrides the list for that run. `device` is written by the app
and records the device last chosen.

`voice_chat_hotkey` must be a single key, such as `"right alt"` or `"alt gr"`.
It switches voice chat on and off when tapped alone (pressed and released
with no other key) and starts continuous listening if the microphone is idle;
the main hotkey still only starts and stops listening. It is never blocked, so AltGr combinations
such as AltGr+Q still type. Set it to `""` to turn it off. The tray shows the mode
while listening: green bars for dictation, purple for voice chat; the balloon
gets a purple border in voice chat. When the app starts already listening
(`continuous_listening`), it always starts in dictation unless `--voice-chat`
is given.

`tap_action` is what a short hotkey press does: `continuous` (default) starts
continuous listening, `toggle` starts a recording that the next press stops.
Holding the hotkey is always push-to-talk. `--continuous` starts the app
already listening continuously. A session started by a tap stops by itself
after `continuous_idle_stop_seconds` (default 120) without speech; `null`
keeps it on until the next tap.

Push-to-talk recordings (including voice chat) are checked for speech before
transcription when Silero VAD is loaded or cached locally. Clips without about
200 ms of speech are dropped and logged. Continuous listening downloads Silero
on first use; push-to-talk never downloads it and keeps its previous behavior
if the model is unavailable.

`save_last_recording: true` keeps the audio of the last transcription in
`~/.debora/last_recording.wav`, to reproduce a bad transcription. Each new
transcription overwrites it: in continuous dictation and voice chat that is the
last finished speech segment, voice chat included. A Claude Code voice session
can read it, since it has access to `~/.debora`. Turning the option off deletes
the file. Off by default.

Continuous dictation normally ends a segment after `vad_end_silence_seconds`
(default 1.5) of silence. When the latest transcription draft covers the most
recent speech and looks incomplete, it waits up to
`vad_incomplete_silence_seconds` (default 3.0) instead. These are total seconds
since the last speech, not an extra delay per draft. A final `.`, `?` or `!`
uses the normal wait; no final punctuation, commas, colons, semicolons and
ellipses use the longer wait. This is a punctuation heuristic, not semantic
understanding. Without a usable draft the normal timeout applies. Set both
values equal to disable the extension. Manual stop and the segment duration
limit still apply; push-to-talk and voice chat keep their existing behavior.

In continuous dictation, drafts appear in the overlay and each finished
sentence is pasted once. `inline_drafts: true` also types drafts into the
target window. Before the first draft of each segment, Windows probes UI
Automation with a 150 ms budget (an empty field counts as readable). Readable
targets retain the verified Shift+Left/Backspace path: the entire previous
draft must match before rewriting its differing tail. A mismatch or lost
readability leaves the draft, keeps subsequent drafts in the overlay, and
pastes the final in full.

Unreadable targets, including terminal TUIs, keep the common prefix and erase
only the differing tail with one Backspace per character. Draft and final tails
use serialized clipboard pastes, including when UI Automation is unavailable.
This bypasses the Unicode-packet path implicated in the Notepad report. This fallback
never uses Shift+Left. Drafts with non-BMP characters (such as emoji), combining
marks, joiners, controls, or Hangul Jamo stay in the overlay because terminal
deletion counts differ; any preceding simple draft is erased first if focus
still matches. Precomposed accents such as `é` work inline. Focus changes leave
the old draft and stop inline output for the segment.

Set `inline_drafts_require_verification: true` (default `false`) to keep all
drafts in unreadable targets in the overlay and paste the final once. Strategy
selection never probes for a final-only segment. The Backspace fallback cannot
detect caret moves or edits within the same target; UIA verification also has
a race between reading and sending keys. Overlay-only drafts remain the default.

Missing `comtypes` or failed UI Automation initialization produces one warning
with reinstall instructions, separate from the per-target unreadable message.

### Voice chat with Claude Code

Choose **Claude Code** and a **Harness folder** in Settings, or run
`debora-cli --voice-chat --voice-chat-backend claude --harness-cwd D:\my-project`.
Claude Code must already be installed, authenticated and on PATH. Qwen remains
the default (`"voice_chat_backend": "local"`); speech recognition and TTS stay local.

```json
{
  "voice_chat": true,
  "voice_chat_backend": "claude",
  "harness_cwd": null,
  "harness_model": null,
  "harness_new_session_on_start": false,
  "harness_session_name": "Débora Whisper",
  "harness_permission_mode": "acceptEdits",
  "harness_permission_response": "deny",
  "harness_allowed_tools": null,
  "harness_extra_allowed_tools": [],
  "harness_prompt_file": null,
  "harness_memory_file": null,
  "harness_hotwords": true,
  "language": "pt"
}
```

`harness_cwd: null` uses your home folder; **Browse…** selects a project and
**Use home** restores that default. `harness_model: null` uses Claude's default.
One Claude process stays alive across turns and receives the raw transcript.
Its folder supplies project context and settings. Débora supplies voice rules
from its packaged prompt, plus language and process-start date/time, through a
temporary file; it creates no context files in the project. `harness_prompt_file`
overrides those rules (relative paths are resolved from the harness folder).

Voice recognition corrections have their own memory, separate from Claude's shared
auto-memory: `harness_memory_file: null` creates `~/.debora/harness/voice_memory.md`
with a header on first use. A custom path is resolved from the harness folder;
its parent should be a dedicated memory directory, passed to Claude with `--add-dir`.
Only Débora launches inject this file (last 100 lines, at most 8 KiB; truncation is
logged). Claude records recurring or user-corrected mistakes as
`- "dictêixon engine" → dictation_engine.py (debora-whisper)` and keeps fewer than
100 lines. Content edits do not restart Claude; the next process start reloads them.
Changing the file path restarts it on the next turn. Ordinary terminal sessions
are not configured to load this file. `--add-dir` adds edit permission for the memory
directory; it does not restrict existing permissions or the working directory
(using your home as `harness_cwd` already includes `~/.debora`).

With `harness_hotwords: true` (default), Claude voice chat also feeds the corrected
terms to Whisper as recognition hints: newest entries first, deduplicated ignoring
case, capped at 40 terms / 150 estimated tokens and cached by file modification time.
Project names from `harness_cwd` (the folder name, `[project].name` in `pyproject.toml`,
and `name` in `package.json` without its scope) fill the remaining budget, including
a spaced form of hyphenated/underscored names; the home folder is skipped.
Project hints are cached by folder and both manifest modification times.
Memory edits affect the next transcription without restarting Claude. The mode is
captured when recording starts (including continuous listening); dictation and local
Qwen receive no hints. Set it to `false` to disable. OpenVINO uses a hotwords string
(or `initial_prompt` on older runtimes); faster-whisper uses hotwords when supported.
Parakeet skips hints and logs that once; hint changes log counts, never the terms.

`harness_allowed_tools: null` adds the packaged read-only diagnostic rules via
`--allowedTools`, for both `Bash(...)` and `PowerShell(...)`. These are exact
command forms: git status/log/diff/show/branch, Docker ps/info/images/version
and compose ps, WSL list, zellij list-sessions, nvidia-smi, Get-ChildItem,
Get-Process and Get-Service, plus `Get-Content README.md` and `Test-Path .`
(including their `-Path` forms). Common variants such as `git status --short`,
`git branch --all`, `docker ps -a` and `wsl -l -v` are included; the complete
list is `DEFAULT_ALLOWED_TOOLS` in `debora_whisper/harness.py`. Exact forms
avoid granting arbitrary flags that write files, delete branches, change GPU
settings or follow logs indefinitely.

A list replaces these defaults, for example
`["PowerShell(docker info:*)", "PowerShell(docker ps:*)", "Bash(docker ps:*)"]`.
Use `[]` to add no base rules. `harness_extra_allowed_tools` (default `[]`)
appends rules to the defaults or the explicit list, removing duplicates while
preserving order. For example, `"harness_extra_allowed_tools": ["WebSearch"]`
keeps the default diagnostics and adds web search. Add explicit file paths for
other Get-Content reads. Changing the effective list restarts the harness on the
next turn. In CLI 2.1.295,
`PowerShell(docker info)` and `PowerShell(docker ps)` also allow the compound
`docker info; docker ps`; both `:*` and ` *` prefix forms worked in the manual
[CLI probe](docs/benchmarks/harness-2026-10-09/README.md#regras-de-diagnóstico-sem-pergunta).

Permissions already allowed by Claude's mode/settings proceed normally. Pending
requests are denied and logged by default; `harness_permission_response: "allow"`
automatically approves them instead. Spoken permission questions are a later phase.
File-edit deny rules protect Débora's resolved `config.json` and `config.json*`
siblings even with automatic approval; reads remain available for diagnosis.
Bash can still bypass these rules, so this is defense in depth.
Tools and decisions appear in `~/.debora/logs/app.log`. Replies stream into the
existing TTS; fenced code and list markers are removed. After 1.5 seconds without
text, Débora queues “Um instante.” once. Actual playback depends on TTS readiness.

| Session setting | Default | Behavior |
|---|---|---|
| `harness_new_session_on_start` | `false` | Resume the saved conversation. If `true`, start fresh on the first use of each folder in each Débora process run; later harness restarts resume that run's session. |
| `harness_session_name` | `"Débora Whisper"` | Literal display name passed through Claude's `--name`, on new sessions and resumes. Use `"Debora Whisper"` for an ASCII-only title. |

Both settings are next to **Harness folder** in Settings. `debora` and
`debora-cli` also accept `--harness-new-session-on-start`,
`--no-harness-new-session-on-start`, and `--harness-session-name "My title"`.
The name stays stable across harness restarts; it is not a date format.
Claude Code 2.1.296 exposes `--name` (`-n`) in `claude --help`.

Session IDs are saved by folder in `~/.debora/harness_session.json` and logged.
After stopping Débora, open that folder in a terminal and run
`claude --resume <session-id>` to continue. By default, restarting Débora resumes
the same conversation. With `harness_new_session_on_start: true`, a fresh UUID
is selected at the first harness start for that folder, and retained in memory
even before the first turn finishes. Changing settings or stopping/restarting
the harness in the same app run keeps that UUID. Enabling the option after a
folder's session has started keeps that session until the next app run.
Say “nova conversa” or “new conversation” to discard both the saved and in-memory
session and start a new one. An unavailable resumed session is still forgotten
so the next attempt can start fresh.
Backend/folder changes take effect on the next turn. This backend has no local
eight-turn history limit or idle reset. See the
[protocol findings](docs/benchmarks/harness-2026-10-09/README.md).

### Voice chat with the local OpenVINO LLM + Chatterbox

Voice chat is on by default (`"voice_chat": true`; turn it off in Settings or
the tray to dictate instead). While it is on, nothing is typed: each final transcription goes to a local LLM, and its reply
is spoken by a local TTS server and shown in the overlay. The reply streams and
plays sentence by sentence. The conversation keeps the last 8 turns and starts
over after 10 minutes of silence (Claude keeps its own persistent history).
In continuous listening the microphone stays open during thinking and playback.
New user speech interrupts the reply and becomes the next turn by default
(`voice_chat_barge_in: true`); with `false`, transcriptions wait in order until
the reply finishes. The hotkey also interrupts. This works with Claude and Qwen.
With `voice_chat_pause_on_speech: true` (default), speech detection immediately
pauses her audio while upcoming sentences continue synthesizing. Meaningful final
text interrupts and starts a new turn; empty text, echoes and fillers replay the
cut sentence from its beginning. If no final arrives within
`voice_chat_pause_timeout` seconds (default `3.0`, a finite positive number), she
also resumes from that sentence. Set `voice_chat_pause_on_speech: false` to keep
playing until the final transcription arrives. Pausing requires `voice_chat_barge_in: true`.
With `voice_chat_prebuffer: true` (default), playback briefly buffers audio before
the first sentence and after a gap, using synthesis timings and speech speed
learned during the session to reduce gaps between known sentences. Each hold is
limited to `voice_chat_prebuffer_max` seconds (default `3.0`, a finite positive
number). Sentences still unwritten by the LLM cannot be predicted, and the limit
can allow gaps on slow hardware. Waiting feedback ("Um instante.") and cached
clips play immediately. Set `voice_chat_prebuffer: false` for immediate playback
as each clip becomes ready.
A sentence normally ends after 0.8 s of silence (`voice_chat_end_silence_seconds`).
An incomplete draft, including trailing `...`, `…`, or a dangling connective,
allows 2 s (`voice_chat_incomplete_silence_seconds`; dictation waits 1.5–3 s).

Headphones are recommended for simultaneous listening and playback. With speakers,
the echo filter compares normalized words against audio actually played during
the captured segment, allowing a 2 s echo tail. Up to four words must all match;
longer text needs 60% token overlap. Echo is not sent to the LLM. This is a text
heuristic: it can mistake a user repeating Débora for echo, or miss distorted
speaker audio. `voice_chat_echo_filter: "auto"` (default; `true` means the same)
turns it off when Windows reports the output device as headphones or a headset,
and keeps it on for speakers or an unknown device (wired headphones on a shared
jack may show as speakers). `"on"` and `"off"` (or `false`) force it. Every nonempty final transcription is
logged in `app.log` and added immediately to the usual transcription history,
including queued, interrupted, cancelled, failed and echo-filtered entries with
status labels. History retains its usual last 20 entries; logs retain older text
subject to log rotation. Raw recordings are not archived.

Switch it on or off while the app runs, from **Voice chat** in the tray menu
or in Settings: no restart. Switching on starts the TTS server and loads the
LLM in the background (the overlay says "Voice chat ready"); a sentence spoken
meanwhile waits for it. Switching off cuts the reply in progress; both stay
loaded until the app exits, so switching back is instant.

```json
{
  "voice_chat": true,
  "voice_chat_incomplete_silence_seconds": 2.0,
  "voice_chat_echo_filter": "auto",
  "voice_chat_barge_in": true,
  "voice_chat_pause_on_speech": true,
  "voice_chat_pause_timeout": 3.0,
  "voice_chat_prebuffer": true,
  "voice_chat_prebuffer_max": 3.0,
  "llm_model": "OpenVINO/Qwen3-8B-int4-cw-ov",
  "llm_device": "GPU",
  "llm_prompt": null,
  "tts_backend": "chatterbox",
  "tts_voice": "debora_v2",
  "tts_stream": true,
  "tts_url": "http://127.0.0.1:8765",
  "tts_timeout_seconds": 60,
  "tts_reply_timeout_seconds": 900,
  "tts_server_command": null
}
```

The LLM runs with OpenVINO GenAI, the same runtime as Whisper, in a process
of its own (`debora_whisper/llm_server.py`, started by the app with its own
Python: nothing to install). Loading it holds Python's GIL for the whole
compile, which froze the app when it ran inside it. While it compiles on the
iGPU no transcription runs: both times the two overlapped, the NPU was lost.
A load that takes over 10 minutes is killed. It logs to
`~/.debora/logs/llm_server.log`. `llm_model` is a Hugging Face repo with an OpenVINO
export (downloaded on first use into the Hugging Face cache) or a local
directory. On a Core Ultra 9 185H, the default answers in under a second at
~15 tokens/s on the Arc iGPU (`"llm_device": "GPU"`); if the device fails it
loads on the CPU. Qwen3's thinking is turned off. `llm_prompt: null` uses Débora's
built-in prompt (`DEFAULT_VOICE_CHAT_PROMPT` in `voice_chat.py`): a short,
friendly reply in the user's language, written only as words to be spoken, with
no emoji, markdown, lists, brackets or symbols (it says "percent", not "%").
With `"language": "pt"` she uses her own Brazilian Portuguese prompt
(`VOICE_CHAT_PROMPTS` in `voice_chat.py`); every other language uses the English
one and is told which language to answer in. Emoji that still slip through are
neither shown nor spoken, and dates, times, percentages and numbers
are spelled out (with `num2words`) before they reach Chatterbox, which reads
digits badly; the overlay keeps the digits.

`tts_backend` selects `chatterbox` (the unchanged default), `qwen3`, `kokoro` or
`xtts`. Each has its own server script (`tts_server.py`, `tts_server_qwen3.py`,
`tts_server_kokoro.py` or `tts_server_xtts.py`) and isolated uv environment with pinned dependencies;
none runs in the app's environment. The app
starts it with `uv run --script` (uv builds that environment on the first run,
a few GB, and caches it). It is started only when nothing answers at
`tts_url`, stopped when the app exits, and logs to
`~/.debora/logs/tts_server.log`. `tts_server_command` replaces that
command line (a list of arguments).

Pick the engine in **Settings → Voice chat voice → Voice engine**; Apply switches the app-managed server without restarting the app.
A manually started server at `tts_url` must be restarted manually to switch backends;
`tts_server_command` takes precedence over script selection.
All backends keep the same loopback HTTP API, Origin checks, disconnect
cancellation and complete WAV responses; this stage does not add streaming.

`qwen3` uses **Qwen3-TTS 0.6B Base** (`Qwen/Qwen3-TTS-12Hz-0.6B-Base`, Apache-2.0)
to clone `tts_voice`. It requires a UTF-8 transcript at `<voice>.txt` beside the
WAV. The confirmed transcript `debora_whisper/voices/debora_v2.txt` ships with
the default voice. Other voices need their own exact transcript; a user WAV
that overrides the bundled voice also needs its own sidecar. Missing or empty
transcripts produce an actionable `/health` error; they are never guessed.
`tts_voice: null` is unsupported by Qwen, and Settings rejects it. It uses SDPA
without FlashAttention. On an RTX 4070 Laptop it measured about 2.9 s of compute
per second of audio, slower than Chatterbox, so it is experimental.

`kokoro` uses **Kokoro-82M** (Apache-2.0), voice **pf_dora**, language **p**
(pt-BR). It has **no voice cloning** and ignores `tts_voice`. Its pt-BR
phonemizer, espeak-ng, comes bundled through `espeakng-loader`: no system install
is needed. If it fails to load, `/health` reports how to fix it. Both new backends accept
`--device cpu` or `--device cuda` when launched manually (default CUDA, CPU
fallback when unavailable).

Stage 2 (Qwen3 and Kokoro) is implemented and **pending benchmark**: speed, VRAM
and pt-BR quality have not been measured. See
[exact server and benchmark commands](docs/benchmarks/tts-bench/README.md).

`xtts` uses **Coqui XTTS-v2** (`tts_models/multilingual/multi-dataset/xtts_v2`,
through `coqui-tts`, Python 3.11) to clone `tts_voice` from the WAV alone, with
no transcript. **Its weights are under the Coqui Public Model License (CPML),
which allows non-commercial use only** (https://coqui.ai/cpml). Selecting this
backend is taken as accepting it: the server sets `COQUI_TOS_AGREED=1` instead
of asking on a console nobody sees, and says so in its log and `--help`. Pick
another backend for commercial use. `tts_voice: null` is unsupported (XTTS-v2
has no voice of its own), and Settings rejects it; changing the voice restarts
the managed server. Languages follow XTTS-v2's list (`pt` covers `pt-BR`);
others fail with an error naming the supported codes. A final `.`, `!`, `;`,
`:` or `…` is dropped before synthesis (XTTS-v2 appends noise after it). As a
safety net, a sound after a near-silent gap of at least 150 ms in the last
second is trimmed, unless the last three words carry a pause mark (`Pode ser,
sim`), where that gap is a real pause. On an RTX 4070 Laptop with `debora_v2`
it loaded in about 12 s, used about 2 GB of VRAM and needed about 0.6 s of
compute per second of audio, faster than Chatterbox (1.4 s). The first run
downloads about 1.8 GB to `%LOCALAPPDATA%\tts`. On every start, `coqui-tts`
also fetches the model's `config.json` from Hugging Face and downloads the
whole model again if it changed upstream; without network access it keeps the
local copy.

`tts_stream` (boolean, default `true`) plays XTTS audio as it is generated when
the server's cached `/health` advertises `stream: true`. Set it to `false` for
whole-sentence WAV playback. Kokoro, Qwen3, Chatterbox and older servers retain
WAV playback. Streaming retains the final second for the artifact check above;
first-audio latency depends on hardware and voice, and ~0.5 s is a target, not
a measured guarantee. Warm-up and the waiting announcement still use WAV.
See the [TTS HTTP protocol](docs/TTS-PROTOCOL.md) for framing and cancellation.

`tts_reply_timeout_seconds` (positive number, default `900`) limits each active speech burst, for streaming and WAV synthesis. Queued
sentences, synthesis and playback share the budget; it resets when speech is idle.
Silent LLM/tool work after a waiting announcement does not consume the answer’s
budget. Expiry cancels speech only; reply text continues. Once the budget expires,
the rest of that turn is text only, with no more audio until the next turn. The server also
stops streams with no completed audio chunk for 120 seconds; ongoing synthesis
can exceed that duration as long as chunks keep arriving.

For Chatterbox, Qwen and XTTS-v2, `tts_voice` clones a voice from clean speech: a WAV path, or a name
looked up as `<name>.wav` in the voices folder (see File Paths), then among
the voices that ship with the app: `carol`, `debora`, `debora_v2` (the
default), `isabel` and `mari`. A file of the same name in the voices folder wins. `null` uses
Chatterbox's own voice. Settings lists them all.

Where each part runs best on a Core Ultra laptop with an 8 GB RTX: Whisper on
the NPU (`--device NPU`), the LLM on the Arc iGPU, Chatterbox alone on the RTX.
Whisper on CUDA next to Chatterbox can run out of video memory. If the TTS
server is down, the reply is only shown. `tts_url` receives everything the
LLM says; keep it pointed at a server you trust.

Or change settings from the GUI: right-click the system tray icon and select **Settings**.

Changing the model, device, hotkey, chime, sample rate or maximum recording length rebuilds the engine. That is refused while a recording, transcription or model load is in progress: the Settings window says what is busy, nothing is saved, and you click **Apply** again once it finishes. This keeps two models from running on the same device at once and keeps an in-flight dictation from being dropped. If a load or transcription never finishes (for example a hung driver), quit and restart the app instead.

## File Paths

| Path | Purpose |
|------|---------|
| `~/.debora/config.json` | User configuration |
| `~/.debora/logs/` | `app.log`, `telemetry.log`, `tts_server.log`, `llm_server.log`; the first three with `[YYYY-MM-DD HH:MM:SS] message` lines; at startup a log over 5 MB moves to `<name>.1` |
| Download telemetry | Every 5 s, active model downloads add `DL STT 42% (310/740 MB)` and/or `DL TTS 17%` to the resource line; progress covers files discovered so far and disappears when idle. |
| `~/.debora/npu_lost.json` | An NPU lost to DEVICE_LOST: startup uses the next device and probes the NPU in the background (cleared when a probe passes or Windows restarts) |
| `~/.debora/models/` | Downloaded model files |
| `~/.debora/ov-cache/` | OpenVINO compilation cache (do not delete) |
| `~/.debora/voices/` | Voices for `tts_voice` by name |

With the `MODELS_DIR` environment variable set (a shared models folder), the
models and cache move to `$MODELS_DIR/debora-whisper/` and the voices to
`$MODELS_DIR/voices/`. Models taken from the Hugging Face cache (the voice
chat LLM, faster-whisper on CUDA, Chatterbox) follow `HF_HOME`.

The project was called NPU Whisper before. On its first run Débora Whisper
moves `~/.npu-dictation/` to `~/.debora/` (and `$MODELS_DIR/npu-whisper/` to
`$MODELS_DIR/debora-whisper/`), and `--install-shortcut` replaces the old
"NPU Whisper" shortcuts. If an older copy of the app is still running, the old
folder stays in use until the next start.

## How It Works

```
+-------------------+     +------------------+     +------------------+
|  Ctrl+Space       | --> |  Microphone      | --> |  Whisper or      |
|  (global hotkey)  |     |  16kHz mono      |     |  Parakeet model  |
+-------------------+     +------------------+     +--------+---------+
                                                            |
                          +------------------+              |
                          |  Clipboard paste | <------------+
                          |  into any window |
                          +------------------+
```

The model runs **100% locally** on your Intel NPU. No internet required after initial model download.

**Whisper** models use OpenVINO GenAI's `WhisperPipeline` with INT8 quantization on NPU or GPU.

**Parakeet TDT** uses a three-stage hybrid pipeline for best accuracy:

```
+---------------+     +----------------+     +----------------+
|  nemo128.onnx | --> |  Encoder       | --> |  TDT Decoder   |
|  Mel spectro  |     |  OpenVINO NPU  |     |  OpenVINO GPU  |
|  (CPU)        |     |                |     |  (CPU fallback)|
+---------------+     +----------------+     +----------------+
```

NPU only supports static (fixed) input shapes, so the Parakeet encoder can't
just size itself to each utterance. Instead it's **shape-bucketed**: several
fixed-size graphs (`ParakeetNPU.MEL_BUCKETS` in `debora_whisper/dictation_engine.py`) are
compiled and cached up front, and each utterance runs on the smallest bucket
it fits in, instead of always paying for the longest one. Audio longer than
the largest bucket is truncated (never crashes), and this is logged, not
silent. The current bucket sizes are chosen by reasoning about typical
dictation lengths, not by measurement — see the comment above `MEL_BUCKETS`
and `benchmarks/README.md`.

## Benchmarking

### Benchmark: turbo on NPU vs Intel iGPU vs RTX (2026-10-06)

Whisper turbo on one laptop: Intel Core Ultra (AI Boost NPU, driver
32.0.100.5540; Arc iGPU) and an NVIDIA RTX 4070 Laptop. OpenVINO 2026.4.1 with
`OpenVINO/whisper-large-v3-turbo-int8-ov` on NPU/iGPU; faster-whisper 1.2.1
(`int8_float16`) on the RTX. Same English speech clips on every device, app
closed, one process per device, median of 5 calls after a warm-up.

| Clip | NPU | Intel iGPU | RTX 4070 |
|------|-----|------------|----------|
| 2s | 1.56s | 0.98s | **0.20s** |
| 5s | 1.59s | 1.03s | **0.21s** |
| 10s | 1.80s | 1.16s | **0.27s** |
| 20s | 2.03s | 1.30s | **0.35s** |
| 26s | 2.13s | 1.60s | **0.39s** |
| 30s, dense speech | 2.79s | 1.92s | **0.56s** |
| Model load (cached) | 3.1s | 1.3s | 3.8s |
| CPU time per call | **0.05–0.23s** | ≈ latency | ≈ latency |

- The RTX is 5–7x faster than the NPU and ~4x faster than the iGPU, which is
  why `device_priority` puts `CUDA` first.
- The NPU takes ~1.5–2.8s per call almost regardless of clip length, but
  leaves the CPU idle; the iGPU and RTX keep a CPU core busy waiting on the
  device for the whole call.
- The RTX draws ~54 W while transcribing and ~3.4 W idle with the model
  loaded. Battery impact was not measured (the test machine's battery is dead).
- The dense 30s clip took 3.0s on the RTX before faster-whisper's temperature
  fallback was disabled: its compression ratio (2.53) passed the 2.4
  threshold, so it re-decoded up to 5 times and returned a temperature-1.0
  sample. The app now decodes greedily at temperature 0, like the OpenVINO path.
- All devices produced the same text.

Reproduce with `benchmarks/bench_devices.py` (usage in its docstring).

### Parakeet

`benchmarks/bench_parakeet.py` measures Parakeet pipeline load/compile time
and per-utterance latency by stage (mel / encoder / decoder), and reports
which shape bucket was selected. See `benchmarks/README.md` for usage and
for the exact commands to produce before/after numbers on real NPU hardware.

## Troubleshooting

### NPU device not found
- Check Device Manager for "Intel(R) AI Boost" or "Intel(R) NPU Accelerator"
- Install/update NPU driver from Windows Update or [Intel's site](https://www.intel.com/content/www/us/en/download/794734/)
- Minimum driver version: 32.0.100.3104

### Slow first run
OpenVINO compiles the model graph for your specific NPU on first launch. This takes 1-15 minutes depending on model size and is cached for subsequent runs. For Parakeet specifically, this means 4 sequential encoder-graph compiles (one per shape bucket), not 1 — see the architecture notes above and `benchmarks/README.md` for details — so first launch takes proportionally longer than a single-graph model.

### DEVICE_LOST error
When the error is attributed to one device, the GUI falls back to the next healthy device in `device_priority`. After an NPU loss it probes the NPU in the background (after 30 s, 1 min, 5 min and 15 min) and moves back once it works; a reboot also resets it. Each probe runs in a separate process, killed after 5 minutes: right after a loss, loading a model on the NPU can hang in the driver or recompile for ~3 minutes, and either one inside the app froze it, Ctrl+C included. Later runs also start on the next device and probe the NPU the same way, until a probe passes.

### GPU failed: restart required
OpenVINO GPU errors such as `CL_OUT_OF_RESOURCES`, or a device loss that cannot be pinned on the NPU (Parakeet runs its decoder on the GPU, and a loss during its GPU fallback compile is blamed on the GPU even if the NPU failed first), can leave the OpenCL context in a state where further calls hang. The app does not retry or reload after that, on the GPU or on any other device. Recording and transcription stay disabled, and Settings changes are saved but not applied (the tray menu marks them "after restart"), until you quit and restart the app. The original OpenVINO error is written to `~/.debora/logs/app.log`. If the failure repeats, select NPU or CPU in Settings, then restart. Disabling the retry only prevents a hang; it does not fix the driver or memory problem behind the error.

### Hotkey doesn't work
- PowerShell must run as **Administrator** (the `keyboard` library requires elevated privileges)
- Check no other app is capturing the same hotkey
- Try a different hotkey: `.\Start-Dictation.ps1 -Hotkey "ctrl+shift+space"`

### Audio not recording
- Check Windows Settings -> Privacy -> Microphone permissions
- Ensure your mic is the default recording device
- Select a specific mic in the Settings dialog

## Development

The [Tests workflow](.github/workflows/tests.yml) runs the full unit-test suite
on Windows with Python 3.10, 3.12 and 3.14 for pushes, pull requests and manual
runs. Each job uploads a JUnit report, including when tests fail. Tests use
simulated audio devices and inference backends; hardware latency still requires
the [manual audio checks](docs/AUDIO_LATENCY.md).

```powershell
# Install and run the same test dependencies as CI (no model downloads)
python -m pip install -r requirements-test.txt
python -m pytest tests/ -q

# Auto-reload during development
pip install watchfiles
watchfiles "python -m debora_whisper" debora_whisper
```

Code lives in the `debora_whisper` package: `app.py` (tray app, `debora`),
`dictation_engine.py` (engine and console mode, `debora-cli`) and `ui/`.
Dependencies are declared in `pyproject.toml`: runtime by default, plus the
`cuda` (RTX), `export` (custom model export) and `test` extras.

### Clean install from a branch

To try a branch the way a user would get it, not from an editable install,
quit the app first (tray → Quit, so it does not save its config on exit), then:

```powershell
# 1. Remove the current install
uv tool uninstall debora-whisper

# 2. Start from a fresh config (logs and config only; models stay put)
Rename-Item $HOME\.debora $HOME\.debora.bak

# 3. With MODELS_DIR set, hide your voices there so the bundled ones are used
#    (a voice of the same name in the voices folder wins)
if ($env:MODELS_DIR) { Rename-Item "$env:MODELS_DIR\voices" voices.off }

# 4a. Install the pushed branch (exactly what others would get)...
uv tool install "git+https://github.com/alexandre-machado/debora-whisper@<branch>"
# 4b. ...or the local checkout, without pushing
uv tool install --reinstall .

# 5. Run as Administrator (global hotkeys)
debora
```

What to check: Settings lists the bundled voices (`carol`, `debora`,
`debora_v2`, `isabel`, `mari`) with `debora_v2` selected, and voice chat
speaks with it.

What this does not cover: with `MODELS_DIR` set, models still come from that
folder (unsetting it downloads several GB of models again), and the
uv and Hugging Face caches keep the Chatterbox environment and weights, which a
new machine downloads on first use.

Back to development:

```powershell
uv tool uninstall debora-whisper
Remove-Item -Recurse $HOME\.debora; Rename-Item $HOME\.debora.bak $HOME\.debora
if ($env:MODELS_DIR) { Rename-Item "$env:MODELS_DIR\voices.off" voices }
uv tool install --editable .
```

### Editable install with the NVIDIA backend

The `cuda` extra adds faster-whisper and the CUDA runtime DLLs (~2 GB from
PyPI). An editable install picks up code changes on restart, but dependency
changes in `pyproject.toml` (including newly added dependencies such as `comtypes`)
and changes to extras require a reinstall. After pulling such changes, reinstall
from the checkout, keeping the extras you use, and restart the app:

```powershell
uv tool install --editable ".[cuda]" --reinstall
# back to the lighter install
uv tool install --editable . --reinstall
```

With the extra installed, `device_priority` (`CUDA` first by default) moves
Whisper to the RTX. When voice chat is on, Chatterbox also runs there, and an
8 GB card can run out of video memory. To keep Whisper on the NPU, put `NPU`
first in `device_priority` or start with `--device NPU`. The TTS server is not
affected by the extra: it runs in its own uv environment with its own CUDA
build of torch.

`docs/` is source material (branding, voice recordings) and is not part of the
wheel or the sdist; the voices the app uses ship in `debora_whisper/voices/`.

### Releasing

Bump `__version__` in `debora_whisper/__init__.py`, then push a `vX.Y.Z` tag that
matches it. The [Release workflow](.github/workflows/release.yml) builds the
wheel and sdist, installs the wheel with `uv tool install` as a smoke test and
publishes to PyPI through trusted publishing (the `pypi` environment must be
registered as a trusted publisher on the PyPI project).

## License

MIT
