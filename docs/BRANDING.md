# Branding: Débora Whisper

Decided 2026-10-08. The project is being renamed from `npu-whisper` to
**Débora Whisper**, dropping the NPU reference now that it also runs on NVIDIA
GPUs (faster-whisper), iGPU and CPU.

## Name rationale

- **Débora** (Portuguese spelling, with the accent) comes from the Hebrew
  *Devorah*, meaning **"bee"**. A bee buzzing softly at your ear fits a
  dictation app called "Whisper".
- The biblical Deborah was a prophetess and judge remembered for the *Song of
  Deborah*: a figure known for her **voice**, which is what this project is about.
- A personal name gives the app a personality (in the line of Alexa and Siri).
  The Brazilian spelling sets it apart from existing English-named tools.
- **Whisper** works as a verb ("sussurrar"), not only as a reference to the
  OpenAI model. The app also runs Parakeet, so the name doesn't promise a
  single model.

## Where the accent goes (and where it doesn't)

| Surface | Form |
|---|---|
| Display name (README title, tray tooltip, overlay, About window, artwork) | **Débora Whisper** |
| PyPI distribution | `debora-whisper` (PEP 508 names must be ASCII) |
| Python package | `debora_whisper` |
| CLI commands | `debora` (tray app) and `debora-cli` (console) |
| Data directory | `~/.debora/` (migrated from `~/.npu-dictation/`) |
| GitHub repo | `debora-whisper` |

Never use the accented form in identifiers, file paths, package names or
commands: it causes encoding issues on Windows, terminals and `uv tool install`.

PyPI availability checked 2026-10-08: `debora-whisper`, `debora`,
`deborah-whisper` and `debora-dictation` were all free.

## Visual direction (for the artwork)

- **Mascot / icon: a bee.** It ties directly to the meaning of the name.
- The icon has to read at tray size (16–32 px) and at the overlay's
  "Dynamic Island" scale. Keep the silhouette simple and recognizable in
  monochrome.
- Tone: friendly and quiet. The app *whispers*, so avoid loud or aggressive
  imagery.
- Wordmark: "Débora Whisper" with the accent kept visible, since it is part of
  the identity.

## Rename checklist

- [x] Package `npu_whisper` → `debora_whisper`, `pyproject.toml` name and entry points (`debora`, `debora-cli`)
- [x] Data dir migration `~/.npu-dictation` → `~/.debora` and `$MODELS_DIR/npu-whisper` → `$MODELS_DIR/debora-whisper` (`paths.py`, `Start-Dictation.ps1`)
- [x] Pre-rename "NPU Whisper" Start Menu/Startup shortcuts replaced by `--install-shortcut`
- [x] Display strings: tray, overlay, settings, history, onboarding, README, AGENTS.md
- [x] Workflows smoke-test `debora` / `debora-cli`
- [x] GitHub repo renamed to `alexandre-machado/debora-whisper`
- [ ] Add a PyPI trusted publisher for the new `debora-whisper` project before the first `v*` tag
- `npu-whisper` was never published on PyPI, so no redirect release is needed.

## Official Mascot Prompt (Concept 02)

This is the definitive prompt to generate the chosen version of the Débora mascot:

> **Subject:** A digital holographic tech fairy/bee hybrid mascot.
> **Pose & Body:** Tinkerbell's graceful, delicate flight silhouette and ethereal hovering posture. Delicate fairy/bee wings.
> **Aesthetic:** Cortana from Halo. Glowing digital AI construct, scanlines, digital glitch artifacts, subtle hexagonal honeycomb mesh patterns across the construct, translucent wireframe elements.
> **Palette:** Heavy emphasis on deep violet/purple bioluminescence, combined with liquid gold circuit traces, and electric cyan/blue neon data streams.


## Bust Prompt (Concept 02, overlay thumbnail)

The overlay shows the mascot as a circle about 32 px wide, so it needs a
head-and-shoulders bust, not the full flying figure. Colors follow Variant 02
of `mascot_concept_matrix.jpg` (gold and cyan), not the violet emphasis above.

> **Subject:** Head-and-shoulders bust portrait of the Débora mascot, a digital holographic tech fairy/bee hybrid, centered, facing slightly to the side with a calm, knowing half-smile.
> **Features:** Golden hair in a high bun with a soft side fringe, pointed elfin ears, translucent cyan-blue holographic skin, the tips of delicate crystalline fairy/bee wings visible behind the shoulders.
> **Aesthetic:** Cortana from Halo. Glowing digital AI construct, subtle scanlines, faint hexagonal honeycomb mesh, liquid gold circuit traces on the neck and collar of a gold bodysuit.
> **Palette:** Liquid gold and electric cyan neon as in Variant 02, with a hint of violet rim light.
> **Composition:** Square 1:1, dark navy background, no text, generous margin around the head so it still reads when cropped to a circle at 32 px.

Save the result as `docs/assets/branding/mascot_v2_bust.png` (1024×1024). The
app ships a square 256 px crop of the current portrait
(`docs/assets/branding/debora_bust.jpg`) as `debora_whisper/ui/assets/debora.jpg`.

## Entrance Animation Prompt (Match to Loop)

This prompt extends the Official Mascot Prompt to create an entrance animation that seamlessly transitions into the idle hovering loop.

> **Subject:** A digital holographic tech fairy/bee hybrid mascot named Débora.
> **Action & Motion:** The animation MUST start with an empty, dark frame (the character is completely absent). Then, the mascot dynamically materializes/spawns into the center of the scene from thin air, constructing herself from swirling glowing data particles and a burst of bioluminescent light (like a hologram turning on). Once fully materialized, she settles smoothly into her signature ethereal hovering posture. The final hovering pose must hold perfectly stable with her wings fluttering rapidly, serving as a seamless match cut into her standard idle loop.
> **Pose & Body:** Tinkerbell's graceful, delicate flight silhouette. Delicate fairy/bee wings.
> **Aesthetic:** Cortana from Halo. Glowing digital AI construct, scanlines, digital glitch artifacts, subtle hexagonal honeycomb mesh patterns across the construct, translucent wireframe elements.
> **Palette:** Heavy emphasis on deep violet/purple bioluminescence, combined with liquid gold circuit traces, and electric cyan/blue neon data streams.
> **Lighting & Environment:** Clean dark navy/black background to isolate the character. High contrast, emphasizing her internal glow, neon rim lighting, and the holographic particle effects of her entrance.

## Voice Synthesis (ElevenLabs)

Débora's voice must reflect her identity: a quiet, ethereal, and helpful AI fairy. She "whispers", so her voice should be soft, close-mic, and non-intrusive, avoiding any loud, energetic or harsh tones.

### Voice Design / Remix Prompt

Use this prompt in ElevenLabs (Voice Design) to generate or remix the base voice:

> A soft, calm, and ethereal young adult female voice. She speaks gently with a breathy, almost whispering texture (ASMR-adjacent), yet perfectly clear, precise, and articulate. Her tone is friendly, warm, and highly intelligent, sounding like a benevolent holographic AI assistant. Subtle Brazilian accent.

### Test Script (Roteiro de Validação)

Use this script to test the generation. It covers her standard interactions: greetings, technical status updates, and a slightly longer conversational piece to evaluate her cadence and warmth.

> "Olá. Sistemas iniciados com sucesso. 
> Estou processando o áudio localmente, garantindo a sua privacidade.
> A captação está limpa, pode começar a ditar quando quiser... Eu estarei aqui, ouvindo.
> Parece que o volume do microfone está um pouco baixo. Não se preocupe, estou ajustando o ganho automaticamente."
