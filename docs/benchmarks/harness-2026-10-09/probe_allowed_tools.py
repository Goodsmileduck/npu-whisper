"""Manual CLI permission probe, isolated from user settings and app sessions."""
import json
import shutil
import sys
import tempfile
import threading
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from debora_whisper.harness import HarnessSession, harness_allowed_tools, harness_command


def main():
    syntax = sys.argv[1] if len(sys.argv) > 1 else "packaged"
    tool = sys.argv[2] if len(sys.argv) > 2 else "PowerShell"
    suffix = {"packaged": "", "none": "", "exact": "", "colon": ":*", "glob": " *"}[syntax]
    rules = [f"{tool}({command}{suffix})" for command in ("docker ps", "docker info")]
    if syntax == "none":
        rules = []
    elif syntax == "packaged":
        rules = list(harness_allowed_tools({}))
    with tempfile.TemporaryDirectory(prefix="debora-permissions-") as cwd:
        prompt = Path(cwd) / "prompt.md"
        prompt.write_text("Execute only the exact requested tool and command, once. "
                          "If denied or failed, do not retry or use another tool. Reply briefly.",
                          encoding="utf-8")

        def command_factory(config, session_id, resume):
            if syntax == "packaged":
                return harness_command(config, session_id, resume) + [
                    "--setting-sources", "", "--strict-mcp-config", "--tools", tool]
            return [shutil.which("claude"), "-p", "--input-format", "stream-json",
                    "--output-format", "stream-json", "--verbose", "--include-partial-messages",
                    "--session-id", session_id, "--permission-mode", "acceptEdits",
                    "--permission-prompts", "host", "--permission-prompt-tool", "stdio",
                    "--append-system-prompt-file", config["harness_prompt_file"],
                    "--system-prompt-snapshot", "off", "--setting-sources", "",
                    "--strict-mcp-config", "--tools", tool,
                    *(["--allowedTools", *rules] if rules else [])]

        permissions = []
        calls = []
        results = []

        class ProbeSession(HarnessSession):
            def _read(self):
                try:
                    for line in self.process.stdout:
                        event = json.loads(line)
                        kind = event.get("type")
                        if kind == "system" and event.get("subtype") == "init":
                            print(json.dumps({"model": event.get("model"),
                                              "tools": [t for t in event.get("tools", [])
                                                        if not t.startswith("mcp__")]}), flush=True)
                        if kind in ("assistant", "user"):
                            for block in event.get("message", {}).get("content", []):
                                if isinstance(block, dict) and block.get("type") == "tool_use":
                                    calls.append({"name": block["name"], "input": block["input"]})
                                if isinstance(block, dict) and block.get("type") == "tool_result":
                                    results.append({"is_error": block.get("is_error", False),
                                                    "content": str(block.get("content", ""))[:180]})
                        self._events.put(event)
                finally:
                    self._events.put(None)

        def deny(request):
            permissions.append({"tool": request["tool_name"], "input": request["input"],
                                "reason": request.get("decision_reason"),
                                "suggestions": request.get("permission_suggestions")})
            return False

        config = {"harness_cwd": cwd, "harness_prompt_file": str(prompt),
                  "harness_memory_file": str(Path(cwd) / "memory" / "voice.md")}
        session = ProbeSession(config, log=lambda line: None, command_factory=command_factory,
                               session_file=Path(cwd) / "session.json", permission_handler=deny)
        print(json.dumps({"syntax": syntax, "rules": rules}), flush=True)
        try:
            for command in ("docker info", "docker ps", "docker ps --all", "docker info; docker ps",
                            "docker rm debora-permission-probe-" + uuid.uuid4().hex):
                permissions.clear()
                calls.clear()
                results.clear()
                reply = []
                session.send(f"Use {tool} to execute exactly: {command}\n"
                             "This is a permission probe. Do not change the command or tool.",
                             reply.append, threading.Event())
                print(json.dumps({"command": command, "calls": calls, "permissions": permissions,
                                  "results": results, "reply": "".join(reply)},
                                 ensure_ascii=False), flush=True)
        finally:
            session.stop()


if __name__ == "__main__":
    main()
