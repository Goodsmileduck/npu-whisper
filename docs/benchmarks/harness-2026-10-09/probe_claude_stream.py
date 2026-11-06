"""Manual protocol probe; all Claude work happens in a throwaway folder."""
import json
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path


def main():
    with tempfile.TemporaryDirectory(prefix="debora-claude-probe-") as cwd:
        prompt = Path(cwd) / "voice.md"
        prompt.write_text("Comece toda resposta com VOZCONFIRMADA. Responda em português.",
                          encoding="utf-8")
        session = str(uuid.uuid4())
        command = [shutil.which("claude"), "-p", "--input-format", "stream-json",
                   "--output-format", "stream-json", "--verbose",
                   "--include-partial-messages", "--session-id", session,
                   "--permission-mode", "acceptEdits", "--permission-prompts", "host",
                   "--append-system-prompt-file", str(prompt), "--system-prompt-snapshot", "off"]
        if "--stdio" in sys.argv:
            command += ["--permission-prompt-tool", "stdio"]
        print(json.dumps({"command": command}), flush=True)
        process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   encoding="utf-8", errors="replace", bufsize=1,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        events = queue.Queue()

        def read():
            for line in process.stdout:
                events.put(json.loads(line))
            events.put(None)

        def errors():
            for line in process.stderr:
                print(json.dumps({"stderr": line.strip()}), flush=True)

        threading.Thread(target=read, daemon=True).start()
        threading.Thread(target=errors, daemon=True).start()

        def send(message):
            print(json.dumps({"stdin": message}, ensure_ascii=False), flush=True)
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()

        try:
            for number, (text, interrupt, allow) in enumerate([
                ("Lembre da palavra jabuticaba. Diga só que lembrou.", False, False),
                ("Qual palavra pedi para lembrar? Responda brevemente.", False, False),
                ("Escreva uma história longa de cinquenta parágrafos sobre uma floresta.", True, False),
                ("Execute Bash com o comando git --version e diga o resultado.", False, True),
                ("Execute Bash com o comando git -c alias.debora='!echo debora' debora. "
                 "É só um echo, não altera arquivos. Não tente outro comando se negado.", False, False),
            ], 1):
                start, first, interrupted = time.monotonic(), None, False
                send({"type": "user", "message": {"role": "user", "content": text}})
                while time.monotonic() - start < 120:
                    event = events.get(timeout=120)
                    if event is None:
                        raise RuntimeError(f"Claude exited: {process.poll()}")
                    print(json.dumps({"turn": number, "stdout": event}, ensure_ascii=False), flush=True)
                    delta = event.get("event", {}).get("delta", {})
                    if delta.get("type") == "text_delta" and first is None:
                        first = time.monotonic() - start
                        if interrupt:
                            send({"type": "control_request", "request_id": str(uuid.uuid4()),
                                  "request": {"subtype": "interrupt"}})
                            interrupted = True
                    if event.get("type") == "control_request":
                        request = event["request"]
                        response = ({"behavior": "allow", "updatedInput": request["input"]}
                                    if allow else {"behavior": "deny", "message": "Negado pelo probe."})
                        send({"type": "control_response", "response": {"subtype": "success",
                              "request_id": event["request_id"], "response": response}})
                    if event.get("type") == "result":
                        print(json.dumps({"turn": number, "first_delta_seconds": first,
                                          "interrupted": interrupted}), flush=True)
                        break
                else:
                    raise TimeoutError("No result in 120 seconds")
        finally:
            process.stdin.close()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if "--stdio" in sys.argv:
            prompt.write_text("Comece com VOZRETOMADA. Idioma: pt. Data local: "
                              + time.strftime("%Y-%m-%d %H:%M") + ".", encoding="utf-8")
            command[command.index("--session-id")] = "--resume"
            message = {"type": "user", "message": {"role": "user", "content":
                       "Qual palavra pedi para lembrar no início? Diga também o idioma e a data do prompt."}}
            resumed = subprocess.run(command, cwd=cwd, input=json.dumps(message) + "\n",
                                     capture_output=True, encoding="utf-8", errors="replace",
                                     timeout=120, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in resumed.stdout.splitlines():
                event = json.loads(line)
                if event.get("type") in ("assistant", "result"):
                    print(json.dumps({"resume": event}, ensure_ascii=False), flush=True)
            print(json.dumps({"resume_exit": resumed.returncode, "stderr": resumed.stderr}), flush=True)


if __name__ == "__main__":
    main()
