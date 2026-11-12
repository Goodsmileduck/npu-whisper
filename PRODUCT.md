# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

<!-- Note: Débora is a native desktop app (Python, customtkinter + pystray) on Windows, Linux and macOS, not a web product. The schema has no desktop value; `web` is recorded as the closest non-mobile platform. No iOS/Android guidance applies. -->

## Users

General desktop users on Windows, Linux and macOS who want private voice-to-text in any application, plus people who use the chat mode to talk to Claude Code hands-free. They press a hotkey, speak, and text lands at the cursor of whatever window is active. Windows 11 is the primary tested platform; Linux (KDE/Wayland) is supported.

## Product Purpose

A local dictation engine with a floating overlay ("Dynamic Island" style), a system tray, settings, and history. It types transcribed text into the active window, and in chat mode acts as the speech-in/speech-out front end for Claude Code. Success is dictation that feels instant and unobtrusive: the user's attention stays in their own work, not in Débora.

## Positioning

Fully local and hardware-aware: runs on Intel NPU/iGPU (OpenVINO), NVIDIA GPU (faster-whisper) or CPU, with automatic device fallback. Zero cloud, zero cost, nothing leaves the machine. A cloud dictation product cannot truthfully claim this.

## Operating Context

- UI lives on top of other applications: the overlay and tray share the screen with the user's active window, and text is typed into that window.
- Surfaces: overlay (mascot, conversation text, state), tray menu (settings, recent texts, history), settings window (language-first model picker with download-status badges), onboarding, history, telemetry/hardware allocation views.
- Hotkey-driven; the mouse is optional during dictation.

## Capabilities and Constraints

- 6 models (Whisper tiny to turbo, Parakeet TDT 0.6B), 15 dictation languages, device priority with fallback.
- Claude Code mode: auto-paste plus Enter; voice chat mode with TTS.
- The overlay must never steal focus or block clicks, since text goes to the active window.
- Must remain keyboard-operable in settings.
- UI is bilingual (pt-BR and English).
- Icon and mascot must read at tray size (16-32 px) and at the small overlay scale, including in monochrome.
- Toolkit: customtkinter and pystray; the app never pip-installs at runtime.

## Brand Commitments

- Display name is "Débora Whisper" with the accent visible; every identifier, path and command is ASCII (`debora-whisper`, `~/.debora`). See `docs/BRANDING.md`.
- Mascot: a holographic tech fairy (gold hair, cyan-blue glow, violet and gold accents; see `debora_whisper/ui/assets`); friendly and quiet in tone, since the app "whispers". No loud or aggressive imagery.

## Evidence on Hand

- Demo video linked in README; artwork under `docs/assets` and `debora_whisper/ui/assets`.
- No testimonials, customer logos, or benchmark claims to reuse beyond what `docs/benchmarks` records.

## Product Principles

1. Stay out of the way: the user's active window is the product; the UI yields to it.
2. Local and private by default, and say so plainly.
3. Hardware complexity is Débora's problem: show what is running, never demand that the user understand it.
4. Quiet personality: the fairy adds warmth without noise.
5. Bilingual and keyboard-reachable everywhere.

## Accessibility & Inclusion

Keyboard-operable settings; focus never stolen by the overlay; pt-BR and English copy; legibility at 16-32 px icon size.
