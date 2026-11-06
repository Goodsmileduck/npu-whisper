# TTS HTTP protocol

`GET /health` returns `ok`, `sample_rate`, `inventory`, and a boolean `stream`
when ready. Only XTTS currently advertises streaming. Loading/error responses
keep their existing status and payload.

`POST /tts` accepts local JSON clients only (no `Origin` header), with `text`,
optional `language`, optional `voice`, and optional boolean `stream`. Text is
bounded by `MAX_TEXT_CHARS`. Without streaming, or when the backend lacks it,
the response is the existing 16-bit PCM `audio/wav`.

With `stream: true` on XTTS, the response is HTTP 200 with
`Content-Type: audio/L16; rate=24000; channels=1`, `X-Sample-Rate: 24000`, and
`Transfer-Encoding: chunked`. This local protocol uses **little-endian** int16
mono PCM despite the L16 media type's usual network byte order. Each generated
audio chunk is an HTTP chunk; a zero-length HTTP chunk marks success. Replies
send `Connection: close`, including health, errors and WAV responses.

Validation and generation failures before the first audio chunk return JSON
errors. Failures after headers close the response without the terminating
chunk; clients must reject incomplete bodies. Disconnecting cancels generation.
Voice preparation and the entire generator share the synthesis lock and
inference/cancellation contexts. Logs include total audio, synthesis time and
time to first chunk.

Stream writes have a bounded socket timeout. The server also cancels after
120 seconds without a completed audio chunk, resetting that inactivity budget
after each chunk. The client limits each active speech burst with
`tts_reply_timeout_seconds` (default 900 seconds), for both streaming and WAV.
The budget runs while sentences are queued, synthesizing or playing, and resets
when speech is idle. Silent LLM/tool work, including gaps after the waiting
announcement, does not consume the next burst’s budget. Expiry interrupts pending
HTTP reads and playback only; LLM work and reply text continue.
Once the speech budget expires, the rest of that turn is text only, with no more audio until the next turn.

The client opts in only when `tts_stream` is true (the default) and cached health
advertises streaming. It also accepts a WAV fallback from a streaming request,
bounds total response bytes, and rejects odd or truncated PCM bodies. Audio
chunks of one sentence play without sentence gaps; captions and echo tracking
start once per sentence. While its duration is unknown, caption pacing uses
the synthesis model's prediction. An interrupted or failed stream is never
requested again automatically.

The [benchmark client](benchmarks/tts-bench/README.md) still measures complete
WAV requests; its response latency does not measure streaming first audio.
