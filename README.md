# Clicky

A fully local, open-source macOS menu bar AI assistant. No cloud APIs. No API keys. Everything runs on your machine.

![macOS](https://img.shields.io/badge/macOS-14%2B-black) ![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

- **Push-to-talk** — hold `Ctrl + Option`, speak, release
- **Whisper STT** — transcribes your voice locally via whisper.cpp
- **Qwen 2.5 LLM** — answers using a 3B model running on Apple Silicon via MLX
- **Screen context** — captures your screen at key-down so the AI sees what you see
- **Conversation history** — remembers last 10 turns
- **Floating overlay** — response appears near your cursor, auto-hides after 12s

No Anthropic. No AssemblyAI. No ElevenLabs. No internet required after setup.

---

## Requirements

- macOS 14+ on Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) installed via Homebrew
- Whisper medium model (~1.5 GB)
- MLX + mlx-lm (`pip install mlx-lm`)

---

## Setup

### 1. Install dependencies

```bash
brew install whisper-cpp
pip3 install rumps pyaudio pynput requests pillow
```

### 2. Download Whisper model

```bash
mkdir -p ~/.whisper
curl -L -o ~/.whisper/ggml-medium.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin
```

### 3. Install MLX LM

```bash
pip3 install mlx-lm
```

The Qwen model (~1.7 GB) downloads automatically on first run.

---

## Run

```bash
# Terminal 1 — start both servers
./start-servers.sh

# Terminal 2 — start the app
python3 clicky.py
```

Or just:

```bash
./start-servers.sh & python3 clicky.py
```

---

## Usage

| Action | What happens |
|--------|-------------|
| Hold `Ctrl + Option` | Start recording (icon turns 🔴) |
| Speak | Whisper transcribes in background |
| Release | AI responds, overlay appears near cursor |
| Click "Clear History" in menu | Reset conversation |
| Click "Check Servers" in menu | Verify Whisper + Qwen are running |

---

## Architecture

```
Voice input (pyaudio)
    │
    ▼
Whisper STT (localhost:8081)  ←─── screenshot captured in parallel
    │
    ▼
Qwen 2.5 3B via MLX (localhost:8080)  ←─── streamed tokens
    │
    ▼
Tkinter overlay near cursor
```

- `clicky.py` — main app (menu bar, hotkey, pipeline)
- `start-servers.sh` — launches Whisper + MLX servers
- `leanring-buddy/` — original Swift app (Xcode)

---

## Swift app (optional)

The original SwiftUI macOS app is in `leanring-buddy/`. Open `leanring-buddy.xcodeproj` in Xcode, set your signing team, and run. It uses the same local servers.

---

## License

MIT
