# Audio capture latency

The microphone stays open during the session. Each recording starts with up to
1.5 seconds of preceding audio held in memory. The buffer is limited by sample
count, so variable PortAudio callback sizes do not change its duration. It
protects speech preceding the hotkey; it cannot recover samples lost by a device
after the hotkey.

When audio feedback is enabled, one output stream opens during loading and
continues rendering silence between precomputed chimes. Hotkeys submit a tone
without opening devices, generating waveforms, or starting playback threads.
The output uses its default sample rate. Both streams request `blocksize=0` and
`latency="low"`; actual latency and the selected HostAPI are logged. Device
selection remains the system/library default; WASAPI is not assumed or forced.

Readiness requires recent microphone callbacks, not a nonzero signal level.
The end-to-end warmup must also capture new frames; old lookback samples alone
cannot mark the application READY. Output initialization failure disables chimes
and logs the error while allowing microphone initialization to continue.

## Validation on the affected machine

1. Restart the app with audio feedback enabled. Wait for READY, then repeat a
   short phrase starting immediately at the hotkey, including after idle time.
2. Repeat with `beep_on_start` disabled. Applying this setting reloads the
   engine and closes the persistent output stream, giving a capture-only
   baseline. When editing the configuration file directly, restart the app.
3. Compare the actual device/HostAPI and capture telemetry in
   `~/.npu-dictation/dictation.log`:
   - `live_frames`: samples captured after recording started, excluding lookback.
   - `overflows`: callbacks reporting dropped input through PortAudio.
   - `max_callback_gap`: largest interval between Python callback entries,
     including the interval crossing the recording start.
   - `max_adc_gap`: largest positive discontinuity between expected and reported
     ADC timestamps. This is a diagnostic, not an exact lost-sample count.
   - `max_delivery_delay`: largest difference between PortAudio's callback time
     and the first sample's ADC timestamp. These timestamps share a clock;
     they are not subtracted from Python's `perf_counter()`.
4. Check output underflows logged at shutdown. If capture callbacks remain
   regular but speech is missing, inspect actual captured audio and device
   processing separately; callback timing alone does not prove AEC suppression.

ADC metrics remain zero when the backend reports unavailable/nonpositive ADC
timestamps. Silence is valid audio. No configuration here guarantees immunity
to driver stalls or device removal. A stalled microphone produces an error;
automatic device reconnection is not implemented.

Inference failures are separate from audio capture. After an OpenVINO GPU
error such as `CL_OUT_OF_RESOURCES`, the app disables recording and inference
for the rest of the process and asks for a restart instead of reloading the
model (see README, "GPU failed: restart required"). A recording that hits
`max_record_seconds` is kept and transcribed once. Text is never pasted after
the engine has stopped (Quit or a Settings rebuild), and a Settings change
that rebuilds the engine is refused, with a message, while a recording,
transcription or model load is in flight. That includes the GPU reload that
follows an NPU device loss: loads are counted, so an earlier load finishing
cannot mark a later one as idle.

Known limitations: device failures are recognised from OpenVINO error text, so
a fatal error worded differently from the known markers is treated as an
ordinary error. A driver call that hangs keeps the engine busy (and Settings
rebuilds refused) until you quit and restart the app. The latched failure keeps
the original exception in memory for the rest of the process.

Automated tests use simulated streams to check sample continuity, variable block
sizes, chime reuse, readiness failures, telemetry, and cleanup. They do not
establish the latency of a physical Windows audio device.
