"""The voice chat's LLM (OpenVINO GenAI), in a process of its own.

Building an LLMPipeline holds the GIL for the whole compile: inside the app
that froze the tray, the overlay and the hotkey (4 s with a warm cache, much
longer without). Here only this process waits.

    python -m debora_whisper.llm_server <model_dir> <device> <cache_dir>

One JSON object per line. On stdout:
    {"ready": "GPU", "seconds": 4.2}    loaded (on the CPU if the device failed)
    {"failed": "..."}                    loaded nowhere; the process exits
    {"log": "..."}                       for the app's log
    {"id": 1, "text": "..."}             a piece of reply 1
    {"id": 1, "done": true}              reply 1 ended (or was cancelled)
    {"id": 1, "error": "..."}            reply 1 failed
On stdin:
    {"id": 1, "messages": [...], "max_new_tokens": 400}
    {"cancel": 1}
The process exits when stdin closes, so it never outlives the app.
"""
import json
import os
import queue
import random
import sys
import threading
import time


def _first_line(e: Exception) -> str:
    return str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__


def load(model_dir: str, device: str, cache_dir: str, say):
    """The LLMPipeline on device, else on the CPU; None if neither loads."""
    import openvino_genai as ov_genai
    errors = []
    for candidate in dict.fromkeys([device, "CPU"]):
        start = time.time()
        try:
            pipe = ov_genai.LLMPipeline(model_dir, candidate, CACHE_DIR=cache_dir)
        except Exception as e:
            errors.append(f"{candidate}: {_first_line(e)}")
            say({"log": f"cannot load the LLM on {errors[-1]}"})
            continue
        from debora_whisper.hardware_allocation import ov_target
        try:
            from debora_whisper.device_queries import openvino_core
            target = ov_target(openvino_core(), candidate)
        except Exception:
            target = f"OV:{candidate} (hardware identity unreported)"
        say({"ready": candidate, "seconds": round(time.time() - start, 1),
             "allocation": {"target": target, "pid": os.getpid(),
                            "planned": device, "fallbacks": errors}})
        return pipe
    say({"failed": "; ".join(errors)})
    return None


def generate(pipe, request: dict, cancelled: threading.Event, say):
    import openvino_genai as ov_genai
    rid = request["id"]
    if cancelled.is_set():
        say({"id": rid, "done": True})
        return
    try:
        # Qwen3's template turns thinking off with enable_thinking; other
        # templates ignore the variable.
        prompt = pipe.get_tokenizer().apply_chat_template(
            request["messages"], add_generation_prompt=True,
            extra_context={"enable_thinking": False})
        generation = ov_genai.GenerationConfig()
        # The prompt is already templated: a second pass would wrap it as a
        # user message, and Qwen3 would think aloud again.
        generation.apply_chat_template = False
        generation.max_new_tokens = request["max_new_tokens"]
        generation.do_sample = True
        generation.temperature = 0.7
        generation.top_p = 0.8
        generation.top_k = 20
        # The default seed is a fixed 0: every reply drew the same numbers.
        generation.rng_seed = random.randrange(1 << 31)
        # With a reply repeated in the history, it answered every question
        # with that reply again: 5 of 6 seeds without this, 0 with it.
        generation.repetition_penalty = 1.1

        def streamer(chunk):
            if cancelled.is_set():
                return ov_genai.StreamingStatus.CANCEL
            say({"id": rid, "text": chunk})
            return ov_genai.StreamingStatus.RUNNING

        pipe.generate(prompt, generation, streamer)
    except Exception as e:
        say({"id": rid, "error": _first_line(e)})
    else:
        say({"id": rid, "done": True})


def serve(model_dir: str, device: str, cache_dir: str, stdin, stdout) -> int:
    lock = threading.Lock()

    def say(message):
        with lock:
            stdout.write(json.dumps(message) + "\n")
            stdout.flush()

    pipe = load(model_dir, device, cache_dir, say)
    if pipe is None:
        return 1
    # Replies run one at a time on a worker, so a cancel can arrive on stdin
    # while one is being generated.
    requests: queue.Queue = queue.Queue()
    cancelled: dict[int, threading.Event] = {}

    def work():
        while (request := requests.get()) is not None:
            generate(pipe, request, cancelled[request["id"]], say)
            cancelled.pop(request["id"], None)

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    try:
        for line in stdin:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if "cancel" in message:
                if (event := cancelled.get(message["cancel"])) is not None:
                    event.set()
            elif "messages" in message:
                cancelled[message["id"]] = threading.Event()
                requests.put(message)
    finally:
        for event in list(cancelled.values()):
            event.set()
        # Cancel accepted requests on EOF or input failure, then release the waiter.
        requests.put(None)
        worker.join()
    return 0


if __name__ == "__main__":
    import os
    # The replies keep stdout to themselves: anything else written there,
    # by OpenVINO's native code too, goes to stderr (the log file).
    replies = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    model_dir, device, cache_dir = sys.argv[1:4]
    sys.exit(serve(model_dir, device, cache_dir, sys.stdin, replies))
