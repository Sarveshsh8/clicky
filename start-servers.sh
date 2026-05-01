#!/bin/bash

# Start Whisper STT server (port 8081) and MLX Qwen LLM server (port 8080)
# Run this before launching the app: ./start-servers.sh

WHISPER_MODEL="$HOME/.whisper/ggml-medium.bin"
QWEN_MODEL="mlx-community/Qwen2.5-3B-Instruct-4bit"

echo "🎙️ Starting Whisper server on port 8081..."
whisper-server --model "$WHISPER_MODEL" --port 8081 &
WHISPER_PID=$!

echo "🤖 Starting MLX Qwen server on port 8080..."
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m mlx_lm.server --model "$QWEN_MODEL" --port 8080 &
QWEN_PID=$!

echo "✅ Both servers running. Press Ctrl+C to stop both."

# Stop both servers when script exits
trap "echo '🛑 Stopping servers...'; kill $WHISPER_PID $QWEN_PID 2>/dev/null" EXIT

wait
