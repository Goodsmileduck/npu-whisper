"""One persistent Claude Code conversation over newline-delimited JSON."""
import atexit
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from debora_whisper import paths
from debora_whisper.processes import NO_WINDOW, kill_tree

HARNESS_PROMPT = Path(__file__).with_name("harness_prompt.md")
PERMISSION_MODES = ("acceptEdits", "auto", "bypassPermissions", "manual", "dontAsk", "plan")
MEMORY_MAX_LINES = 100
MEMORY_MAX_BYTES = 8192
PROJECT_MANIFEST_MAX_BYTES = 256 * 1024
# Exact forms deliberately avoid writable flags (git branch -D, --output,
# nvidia-smi -pl) and indefinite reads (Get-Content -Wait).
DEFAULT_ALLOWED_TOOLS = tuple(
    f"{tool}({command})"
    for tool in ("Bash", "PowerShell")
    for command in (
        "git status", "git status --short", "git status --porcelain",
        "git log", "git log --oneline", "git log -5 --oneline",
        "git diff", "git diff --stat", "git diff --cached", "git show",
        "git branch", "git branch --all", "git branch --show-current",
        "docker ps", "docker ps --all", "docker ps -a", "docker info",
        "docker images", "docker version", "docker compose ps",
        "docker compose ps --all", "wsl -l -v", "wsl --list",
        "wsl --list --verbose", "zellij list-sessions", "nvidia-smi",
        "Get-ChildItem", "Get-ChildItem -Force", "Get-ChildItem -Path .",
        "Get-Content README.md", "Get-Content -Path README.md",
        "Get-Process", "Get-Service", "Test-Path .", "Test-Path -Path .",
    )
)


def harness_allowed_tools(config: dict) -> tuple[str, ...]:
    rules = config.get("harness_allowed_tools")
    rules = DEFAULT_ALLOWED_TOOLS if rules is None else rules
    return tuple(dict.fromkeys((*rules, *config.get("harness_extra_allowed_tools", []))))


def validate_session_name(name):
    """Keep display names safe even when Windows resolves Claude to a batch file."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("harness_session_name must be a non-empty string")
    if len(name) > 80 or any(not (c.isalpha() or c.isdecimal() or c in " ._-·:()")
                             for c in name):
        raise ValueError("harness_session_name must be at most 80 characters and contain only "
                         "letters, digits, spaces and . _ - · : ( )")


def harness_command(config: dict, session_id: str, resume: bool) -> list[str]:
    """Build argv; executable lookup and the dated prompt happen at launch."""
    command = ["claude", "-p", "--input-format", "stream-json",
               "--output-format", "stream-json", "--verbose", "--include-partial-messages",
               "--resume" if resume else "--session-id", session_id,
               "--name", config.get("harness_session_name", "Débora Whisper"),
               "--permission-mode", config.get("harness_permission_mode", "acceptEdits"),
               "--permission-prompts", "host", "--permission-prompt-tool", "stdio",
               "--add-dir", str(harness_memory_file(config).parent),
               "--append-system-prompt-file", str(config.get("harness_prompt_file") or HARNESS_PROMPT),
               "--system-prompt-snapshot", "off"]
    config_dir = paths.CONFIG_DIR.resolve()
    if not config_dir.is_relative_to(harness_cwd(config)):
        command += ["--add-dir", str(config_dir)]
    config_path = paths.CONFIG_FILE.resolve().as_posix()
    # Claude uses // absolute paths and /c/... for Windows drive paths.
    config_path = re.sub(r"^([A-Za-z]):/", lambda m: f"/{m[1].lower()}/", config_path)
    # Read/Edit use gitignore syntax: backslash escapes literal pattern characters.
    config_path = re.sub(r"[\\*?\[\]{}]", lambda m: "\\" + m[0], config_path)
    config_rule = "//" + config_path.lstrip("/") + "*"
    # An Edit rule covers every file editor (Write rules only warn). Bash can
    # still bypass this: defense in depth, not a permission boundary.
    command += ["--disallowedTools", f"Edit({config_rule})"]
    if config.get("harness_model"):
        command += ["--model", config["harness_model"]]
    rules = harness_allowed_tools(config)
    if rules:
        command += ["--allowedTools", *rules]
    return command


def harness_cwd(config: dict) -> Path:
    return Path(config.get("harness_cwd") or Path.home()).expanduser().resolve()


def harness_memory_file(config: dict) -> Path:
    path = Path(config.get("harness_memory_file") or
                paths.CONFIG_DIR / "harness" / "voice_memory.md").expanduser()
    if not path.is_absolute():
        path = harness_cwd(config) / path
    return path.resolve()


def harness_key(config: dict) -> tuple:
    return (os.path.normcase(str(harness_cwd(config))), config.get("harness_model"),
            config.get("harness_permission_mode", "acceptEdits"),
            config.get("harness_prompt_file"), config.get("language"),
            config.get("harness_permission_response", "deny"),
            os.path.normcase(str(harness_memory_file(config))), harness_allowed_tools(config),
            config.get("harness_new_session_on_start", False),
            config.get("harness_session_name", "Débora Whisper"))


def voice_prompt(prompt: str, language: str | None, now: str) -> str:
    return (prompt.rstrip() + f"\nIdioma configurado: {language or 'auto'} "
            "(auto: acompanhe o idioma do usuário)."
            + f"\nData e hora local ao iniciar este processo: {now}.\n")


def _self_prompt(config: dict) -> str:
    logs = paths.LOG_DIR.resolve()
    return ("\n## Sobre a própria Débora\n"
            "Você pode ler e editar estes arquivos para diagnosticar e corrigir a Débora:\n"
            f"Configuração: {paths.CONFIG_FILE.resolve()}\n"
            f"Inicialização, tempos de transcrição, hardware e turnos de voz: {logs / 'app.log'}\n"
            f"CPU/RAM/VRAM e buffer de áudio: {logs / 'telemetry.log'}\n"
            f"Servidor de voz: {logs / 'tts_server.log'}\n"
            f"Memória de voz: {harness_memory_file(config)}\n")


def memory_terms(text: str) -> list[str]:
    """Extract unique correction terms, in order, without optional context."""
    terms = []
    for line in text.splitlines():
        match = re.fullmatch(r'- "(?:[^"\\]|\\.)+" → (.+?)(?: \([^()]*\))?', line)
        if match:
            term = match[1].strip()
            if term and term not in terms:
                terms.append(term)
    return terms


def bounded_hotwords(terms) -> tuple[str, ...]:
    kept, seen, tokens = [], set(), 0
    for term in terms:
        folded = term.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        # UTF-8 bytes / 3 plus a separator leaves room below Whisper's
        # ~224-token budget, without loading a tokenizer.
        try:
            cost = (len(term.encode("utf-8")) + 2) // 3 + 1
        except UnicodeError:
            continue
        if tokens + cost > 150:
            continue
        kept.append(term)
        tokens += cost
        if len(kept) == 40:
            break
    return tuple(kept)


class ProjectHotwords:
    """Cache names from the harness folder and its two package manifests."""

    def __init__(self):
        self._key = None
        self._terms = ()

    def terms(self, config: dict, log=print) -> tuple[str, ...]:
        try:
            folder = harness_cwd(config)
            if folder == Path.home().resolve():
                return ()
        except (OSError, ValueError, RuntimeError):
            return ()
        files = (folder / "pyproject.toml", folder / "package.json")
        mtimes = []
        for path in files:
            try:
                mtimes.append(path.stat().st_mtime_ns
                              if not path.is_symlink() and path.is_file() else None)
            except OSError:
                mtimes.append(None)
        key = (folder, *mtimes)
        if key == self._key:
            return self._terms
        self._key, self._terms = key, ()
        names, failed = [], False
        if len(folder.name) <= 64 and re.fullmatch(r"[\w .-]+", folder.name):
            names.append(folder.name)
        for path, mtime in zip(files, mtimes):
            if mtime is None:
                continue
            try:
                with path.open("rb") as manifest:
                    content = manifest.read(PROJECT_MANIFEST_MAX_BYTES + 1)
                if len(content) > PROJECT_MANIFEST_MAX_BYTES:
                    failed = True
                    continue
                text = content.decode("utf-8")
                if path.name == "pyproject.toml":
                    data = tomllib.loads(text).get("project", {})
                else:
                    data = json.loads(text)
                name = data.get("name") if isinstance(data, dict) else None
                if isinstance(name, str):
                    if path.name == "package.json" and name.startswith("@"):
                        name = name.partition("/")[2]
                    if re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?", name):
                        names.append(name)
            except (OSError, ValueError, RecursionError):
                failed = True
        if failed:
            log("Voice chat: cannot read project hotword metadata; skipping invalid files")
        terms = []
        for name in names:
            if name:
                terms.append(name)
                spoken = re.sub(r"[-_]+", " ", name).strip()
                if spoken and spoken != name:
                    terms.append(spoken)
        self._terms = tuple(terms)
        return self._terms


class MemoryHotwords:
    """Cache a small, newest-first recognition vocabulary by file mtime."""

    def __init__(self):
        self._key = None
        self._terms = ()

    def terms(self, config: dict, log=print) -> tuple[str, ...]:
        path = harness_memory_file(config)
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            mtime = None
        key = (path, mtime)
        if key == self._key:
            return self._terms
        self._key, self._terms = key, ()
        if mtime is None:
            return self._terms
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as e:
            log(f"Voice chat: cannot read hotword memory ({type(e).__name__})")
            return self._terms
        # Reverse lines before extraction so repeated terms retain their latest entry.
        self._terms = bounded_hotwords(memory_terms("\n".join(reversed(text.splitlines()))))
        return self._terms


def _memory_prompt(path: Path, log) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as file:
            file.write("# Memória de reconhecimento de voz\n")
    except FileExistsError:
        pass
    lines = path.read_text(encoding="utf-8").splitlines()
    kept, size = [], 0
    for line in reversed(lines[-MEMORY_MAX_LINES:]):
        length = len((line + "\n").encode("utf-8"))
        if size + length > MEMORY_MAX_BYTES:
            break
        kept.append(line)
        size += length
    if len(kept) < len(lines):
        log(f"Voice chat: memory truncated for prompt: {path} "
            f"({len(kept)}/{len(lines)} lines, {size} bytes)")
    content = "\n".join(reversed(kept))
    return (f"\n## Erros de reconhecimento já conhecidos\n"
            f"Arquivo de memória de voz: {path}\n\n{content}\n")


def permission_response(request_id: str, request: dict, allow: bool) -> dict:
    response = ({"behavior": "allow", "updatedInput": request["input"]} if allow else
                {"behavior": "deny", "message": "Permissão negada pela Débora."})
    return {"type": "control_response", "response": {
        "subtype": "success", "request_id": request_id, "response": response}}


def _sessions(path: Path, log) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        log(f"Voice chat: cannot read harness sessions ({e})")
        return {}


_sessions_lock = threading.Lock()
# Survives harness restarts, but never a restart of Débora. Include the store
# path so independent session stores do not share conversations.
# Store selection separately from confirmation that Claude saved a conversation.
_run_sessions: dict[tuple[Path, str], tuple[str | None, bool]] = {}


def _save_session(path: Path, cwd: str, session_id: str | None, log):
    with _sessions_lock:
        _run_sessions[(path.resolve(), cwd)] = (session_id, session_id is not None)
        data = _sessions(path, log)
        if session_id is None:
            data.pop(cwd, None)
        else:
            data[cwd] = session_id
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(".tmp")
        pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pending.replace(path)


class HarnessSession:
    """Claude owns history; only the latest transcript goes to send().

    on_event(kind, message) reports tool_use and permission events.
    permission_handler(request) can replace the configured decision later.
    """

    _resumed = _confirmed = False

    def __init__(self, config: dict, log=print, command_factory=harness_command,
                 process_factory=subprocess.Popen, session_file=None, on_event=None,
                 permission_handler=None):
        validate_session_name(config.get("harness_session_name", "Débora Whisper"))
        self.config, self.log = dict(config), log
        self.key = harness_key(config)
        self.cwd = harness_cwd(config)
        if not self.cwd.is_dir():
            log(f"Voice chat: harness folder does not exist: {self.cwd}")
            raise ValueError(f"Harness folder does not exist: {self.cwd}")
        self.on_event, self.permission_handler = on_event, permission_handler
        self.session_file = Path(session_file) if session_file else paths.CONFIG_DIR / "harness_session.json"
        run_key = (self.session_file.resolve(), self.key[0])
        with _sessions_lock:
            if run_key in _run_sessions:
                saved, resumable = _run_sessions[run_key]
            elif config.get("harness_new_session_on_start", False):
                saved, resumable = None, False
            else:
                saved = _sessions(self.session_file, log).get(self.key[0])
                resumable = bool(saved)
        try:
            saved = str(uuid.UUID(saved)) if saved else None
        except (TypeError, ValueError):
            log(f"Voice chat: ignoring an invalid saved Claude session {saved!r}")
            saved = None
        resumable = bool(saved) and resumable
        self.session_id = saved or str(uuid.uuid4())
        # A resumed session that never completes a turn (expired, deleted)
        # is forgotten, so the next start is a new conversation.
        self._resumed, self._confirmed = resumable, False
        self.error: str | None = None
        self._events: queue.Queue = queue.Queue()
        self._send_lock = threading.Lock()
        self._turn_lock = threading.Lock()
        self._prompt_file: Path | None = None
        try:
            source = Path(config.get("harness_prompt_file") or HARNESS_PROMPT).expanduser()
            if not source.is_absolute():
                source = self.cwd / source
            prompt = voice_prompt(source.read_text(encoding="utf-8"), config.get("language"),
                                  time.strftime("%A, %Y-%m-%d %H:%M"))
            prompt += _self_prompt(config)
            prompt += _memory_prompt(harness_memory_file(config), log)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".md",
                                             prefix="debora-voice-", delete=False) as file:
                self._prompt_file = Path(file.name)
                file.write(prompt)
            command = command_factory({**config, "harness_prompt_file": str(self._prompt_file)},
                                      self.session_id, resumable)
            command[0] = shutil.which(command[0]) or command[0]
            self.process = process_factory(
                command, cwd=str(self.cwd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=NO_WINDOW)
        except Exception:
            self._remove_prompt()
            raise
        with _sessions_lock:
            _run_sessions[run_key] = (self.session_id, resumable)
        from debora_whisper.hardware_allocation import allocation
        self._allocation_owner = allocation.claim("CLAUDE")
        allocation.update("CLAUDE", f"external Claude Code; model={config.get('harness_model') or 'CLI default'}; "
                          f"endpoint=Claude Code managed; harness pid {self.process.pid} (CPU orchestration)",
                          owner=self._allocation_owner)
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        log(f"Voice chat: Claude session {self.session_id} in {self.cwd} "
            f"({'resume' if resumable else 'new'}, pid {self.process.pid})")

    @property
    def running(self) -> bool:
        return self.process.poll() is None and self.error is None

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    message = json.loads(line)
                except ValueError:
                    self.log(f"Voice chat: Claude stdout: {line.strip()}")
                    continue
                if isinstance(message, dict):
                    if message.get("type") == "system" and message.get("subtype") == "init":
                        from debora_whisper.hardware_allocation import allocation
                        allocation.update("CLAUDE", f"external Claude Code; model={message.get('model') or 'unreported'}; "
                                          f"endpoint=Claude Code managed; harness pid {self.process.pid} (CPU orchestration)",
                                          owner=self._allocation_owner)
                    self._events.put(message)
        finally:
            self.error = self.error or "Claude closed its output"
            from debora_whisper.hardware_allocation import allocation
            allocation.update("CLAUDE", f"external Claude Code; harness exited pid {self.process.pid}",
                              owner=self._allocation_owner)
            self._events.put(None)

    def _read_stderr(self):
        for line in self.process.stderr:
            if line.strip():
                self.log(f"Voice chat: Claude stderr: {line.strip()}")

    def _send(self, message: dict):
        try:
            with self._send_lock:
                self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
                self.process.stdin.flush()
        except (OSError, ValueError) as e:
            raise RuntimeError(self.error or f"Claude input closed: {e}") from e

    def _notice(self, kind: str, message: str, on_event):
        self.log(f"Voice chat: {message}")
        callback = on_event or self.on_event
        if callback:
            callback(kind, message)

    def _permission_denied(self, denial: dict, denials_seen: set, on_event):
        tool_id = denial.get("tool_use_id")
        if tool_id and tool_id in denials_seen:
            return
        if tool_id:
            denials_seen.add(tool_id)
        tool_input = denial.get("tool_input", denial.get("input", {}))
        detail = tool_input.get("command", tool_input) if isinstance(tool_input, dict) else tool_input
        detail = f"{denial.get('tool_name', 'unknown')} {detail}"
        if len(detail) > 200:
            detail = detail[:197] + "..."
        self._notice("permission", f"Claude permission denied: {detail}", on_event)

    def interrupt(self):
        self._send({"type": "control_request", "request_id": str(uuid.uuid4()),
                    "request": {"subtype": "interrupt"}})

    def send(self, text: str, on_text, stop: threading.Event, on_event=None):
        """Stream text and drain an interrupted turn before accepting another."""
        with self._turn_lock:
            # A reserved voice turn still records its user message when
            # interrupted before the writer starts; then drain its result.
            if not self.running:
                self._forget_unresumed()
                raise RuntimeError(self.error or "Claude is not running")
            self._send({"type": "user", "message": {"role": "user", "content": text}})
            interrupted = None
            last_event = time.monotonic()
            tools_seen = set()
            denials_seen = set()
            last_text = ""
            new_text_block = True
            try:
                while True:
                    now = time.monotonic()
                    if stop.is_set() and interrupted is None:
                        self.interrupt()
                        interrupted = now
                        self.log("Voice chat: interrupting Claude; draining the turn")
                    if ((interrupted is not None and now - interrupted > 15)
                            or now - last_event > 180):
                        raise TimeoutError("Claude did not finish the turn; see app.log")
                    try:
                        message = self._events.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if message is None:
                        raise RuntimeError(self.error)
                    last_event = now
                    kind = message.get("type")
                    if kind == "control_request":
                        request = message["request"]
                        if request.get("subtype") == "can_use_tool":
                            self._notice("permission", f"Claude pede permissão: {request['tool_name']} "
                                         f"{request.get('input', {})}", on_event)
                            allow = self.config.get("harness_permission_response", "deny") == "allow"
                            if self.permission_handler and not stop.is_set():
                                allow = bool(self.permission_handler(request))
                            allow = allow and not stop.is_set()
                            self._send(permission_response(message["request_id"], request, allow))
                            if allow:
                                self.log(f"Voice chat: Claude permission allowed: {request['tool_name']}")
                            else:
                                self._permission_denied(request, denials_seen, on_event)
                        else:
                            self._send({"type": "control_response", "response": {
                                "subtype": "error", "request_id": message["request_id"],
                                "error": "Unsupported control request"}})
                    elif kind == "stream_event" and not message.get("parent_tool_use_id"):
                        event = message.get("event", {})
                        if event.get("type") in ("message_start", "content_block_start"):
                            new_text_block = True
                        block = event.get("content_block", {})
                        if block.get("type") == "tool_use":
                            tools_seen.add(block.get("id"))
                            self._notice("tool_use", f"Claude está usando {block['name']}…", on_event)
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta" and not stop.is_set():
                            chunk = delta["text"]
                            if chunk:
                                if (new_text_block and last_text and not last_text[-1].isspace()
                                        and not chunk[0].isspace()
                                        # Straight quotes may open a quote.
                                        and chunk[0] not in ".,;:!?)]}…”’»›"):
                                    chunk = " " + chunk
                                on_text(chunk)
                                last_text = chunk
                                new_text_block = False
                    elif kind == "assistant":
                        # The full message repeats deltas, but may reveal a tool without partials.
                        new_text_block = True
                        for block in message.get("message", {}).get("content", []):
                            if block.get("type") == "tool_use" and block.get("id") not in tools_seen:
                                tools_seen.add(block.get("id"))
                                self._notice("tool_use", f"Claude está usando {block['name']}…", on_event)
                    elif kind == "result":
                        self._confirmed = True
                        self.session_id = message.get("session_id", self.session_id)
                        try:
                            _save_session(self.session_file, self.key[0], self.session_id, self.log)
                        except OSError as e:
                            self.log(f"Voice chat: cannot save Claude session {self.session_id} ({e})")
                        for denial in message.get("permission_denials", []):
                            self._permission_denied(denial, denials_seen, on_event)
                        if message.get("is_error") and not stop.is_set():
                            raise RuntimeError("; ".join(message.get("errors", [])) or "Claude turn failed")
                        return
            except Exception as e:
                self.error = str(e)
                self.log(f"Voice chat: Claude failed ({e})")
                self.stop()
                self._forget_unresumed()
                raise

    def _forget_unresumed(self):
        if self._resumed and not self._confirmed:
            self._resumed = False
            self.log(f"Voice chat: Claude session {self.session_id} did not resume; "
                     "the next turn starts a new conversation")
            try:
                _save_session(self.session_file, self.key[0], None, self.log)
            except OSError as e:
                self.log(f"Voice chat: cannot forget Claude session {self.session_id} ({e})")

    def _remove_prompt(self):
        if self._prompt_file:
            self._prompt_file.unlink(missing_ok=True)
            self._prompt_file = None

    def stop(self):
        try:
            self.process.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            self.process.wait(5)
        except subprocess.TimeoutExpired:
            kill_tree(self.process)
            self.process.wait(5)
        finally:
            self._remove_prompt()
            from debora_whisper.hardware_allocation import allocation
            allocation.update("CLAUDE", None, owner=self._allocation_owner)


_harness_lock = threading.Lock()
_harness: HarnessSession | None = None


def start_harness(config: dict, log=print) -> HarnessSession:
    global _harness
    key = harness_key(config)
    with _harness_lock:
        if _harness is not None and _harness.key == key and _harness.running:
            return _harness
        if _harness is not None:
            _harness.stop()
            _harness = None
        try:
            _harness = HarnessSession(config, log)
        except Exception as e:
            log(f"Voice chat: cannot start Claude ({e})")
            raise
        return _harness


def stop_harness():
    global _harness
    with _harness_lock:
        harness, _harness = _harness, None
        if harness is not None:
            harness.stop()


def reset_harness(config: dict, log=print):
    """Forget this folder's conversation; the next turn gets a fresh UUID."""
    global _harness
    key = harness_key(config)
    with _harness_lock:
        if _harness is not None and _harness.key[0] == key[0]:
            _harness.stop()
            # An interrupted writer may still be saving its last result.
            with _harness._turn_lock:
                pass
            _harness = None
        _save_session(paths.CONFIG_DIR / "harness_session.json", key[0], None, log)
    log(f"Voice chat: new Claude conversation requested in {key[0]}")


atexit.register(stop_harness)
