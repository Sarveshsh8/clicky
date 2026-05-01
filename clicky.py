#!/usr/bin/env python3
"""
Clicky - macOS menu bar AI assistant
Requires: pip install rumps pyaudio pynput pillow faster-whisper mlx-lm
"""

import io
import os
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, Future

import pyaudio
import rumps
from faster_whisper import WhisperModel
from mlx_lm import load, stream_generate
from pynput import keyboard

# ── Config ────────────────────────────────────────────────────────────────────

QWEN_MODEL         = "mlx-community/Qwen2.5-3B-Instruct-4bit"
WHISPER_MODEL_SIZE = "base"   # tiny/base/small — base is fast enough and accurate

SYSTEM_PROMPT = (
    "You are Clicky, a concise macOS desktop assistant. "
    "Give short, direct answers. No markdown. No bullet lists unless asked."
)

MAX_HISTORY_TURNS         = 10
OVERLAY_AUTO_HIDE_SECONDS = 12
OVERLAY_UPDATE_MIN_GAP    = 0.05  # ~20fps overlay refresh

# Audio
SAMPLE_RATE = 16_000
CHANNELS    = 1
CHUNK       = 1024
FORMAT      = pyaudio.paInt16

# ── Load models at startup (once) ─────────────────────────────────────────────

print(f"Loading Whisper {WHISPER_MODEL_SIZE}…")
_whisper = WhisperModel(WHISPER_MODEL_SIZE, device="auto", compute_type="int8")
print("Whisper ready.")

print(f"Loading {QWEN_MODEL}…")
_qwen_model, _qwen_tokenizer = load(QWEN_MODEL)
print("Qwen ready. Starting app…")

# ── Audio recorder ────────────────────────────────────────────────────────────

class AudioRecorder:
    def __init__(self):
        self._pa     = pyaudio.PyAudio()
        self._stream = None
        self._frames = []
        self._active = False
        self._lock   = threading.Lock()

    def start(self):
        with self._lock:
            self._frames = []
            self._active = True
        self._stream = self._pa.open(
            format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
            input=True, frames_per_buffer=CHUNK,
            stream_callback=self._on_audio,
        )
        self._stream.start_stream()

    def _on_audio(self, in_data, frame_count, time_info, status):
        with self._lock:
            if self._active:
                self._frames.append(in_data)
        return (None, pyaudio.paContinue)

    def stop(self) -> bytes:
        with self._lock:
            self._active = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        return self._to_wav()

    def _to_wav(self) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self._pa.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(self._frames))
        return buf.getvalue()

    def close(self):
        try:
            self._pa.terminate()
        except Exception:
            pass

# ── Whisper STT ───────────────────────────────────────────────────────────────

def transcribe_wav(wav_data: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_data)
        tmp_path = f.name
    try:
        segments, _ = _whisper.transcribe(tmp_path, language="en", beam_size=1)
        return " ".join(s.text for s in segments).strip()
    finally:
        os.unlink(tmp_path)

# ── Qwen LLM (in-process, no HTTP) ───────────────────────────────────────────

def stream_qwen(history: list[dict], transcript: str, on_chunk) -> str:
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": transcript}]
    )
    prompt = _qwen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    accumulated = ""
    last_update  = 0.0
    for response in stream_generate(_qwen_model, _qwen_tokenizer, prompt, max_tokens=256):
        accumulated += response.text
        now = time.monotonic()
        if now - last_update >= OVERLAY_UPDATE_MIN_GAP:
            on_chunk(accumulated)
            last_update = now
        if response.finish_reason is not None:
            break
    on_chunk(accumulated)
    return accumulated

# ── Tkinter overlay ───────────────────────────────────────────────────────────

class OverlayWindow:
    def __init__(self, tk_root):
        self._root       = tk_root
        self._win        = None
        self._label      = None
        self._hide_after = None

    def _ensure_window(self):
        import tkinter as tk
        if self._win is not None:
            try:
                if self._win.winfo_exists():
                    return
            except Exception:
                pass
        self._win = tk.Toplevel(self._root)
        self._win.withdraw()
        self._win.overrideredirect(True)
        self._win.attributes("-topmost", True)
        self._win.attributes("-alpha", 0.93)
        self._win.configure(bg="#1e1e2e")
        self._label = tk.Label(
            self._win, text="", bg="#1e1e2e", fg="#cdd6f4",
            font=("SF Pro Text", 13), wraplength=420,
            justify="left", padx=16, pady=12,
        )
        self._label.pack()

    def _cursor_pos(self):
        try:
            from AppKit import NSEvent
            loc      = NSEvent.mouseLocation()
            screen_h = self._root.winfo_screenheight()
            return int(loc.x) + 18, int(screen_h - loc.y) + 18
        except Exception:
            return 200, 200

    def show(self, text: str):
        self._ensure_window()
        x, y = self._cursor_pos()
        self._label.config(text=text)
        self._win.geometry(f"+{x}+{y}")
        self._win.deiconify()
        self._win.lift()
        self._reschedule_hide()

    def update_text(self, text: str):
        if self._win and self._label:
            try:
                if self._win.winfo_viewable():
                    self._label.config(text=text)
                    self._win.update_idletasks()
                    return
            except Exception:
                pass
        self.show(text)

    def hide(self):
        if self._win:
            try:
                self._win.withdraw()
            except Exception:
                pass

    def _reschedule_hide(self):
        if self._hide_after is not None:
            try:
                self._root.after_cancel(self._hide_after)
            except Exception:
                pass
        self._hide_after = self._root.after(
            int(OVERLAY_AUTO_HIDE_SECONDS * 1000), self.hide
        )

# ── Main app ──────────────────────────────────────────────────────────────────

class ClickyApp(rumps.App):
    def __init__(self):
        super().__init__("🎙", quit_button="Quit")
        self.menu = ["Status", None, "Clear History"]

        self._recorder    = AudioRecorder()
        self._history: list[dict] = []
        self._hotkey_held = False
        self._proc_lock   = threading.Lock()
        self._processing  = False
        self._overlay     = None
        self._tk_root     = None
        self._executor    = ThreadPoolExecutor(max_workers=2)

        self._init_tk()
        self._setup_hotkey()

    def _init_tk(self):
        import tkinter as tk
        self._tk_root = tk.Tk()
        self._tk_root.withdraw()
        self._overlay = OverlayWindow(self._tk_root)

    # ── Global hotkey (Ctrl + Option) ─────────────────────────────────────────

    def _setup_hotkey(self):
        ctrl_down = [False]
        alt_down  = [False]

        def on_press(key):
            if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                ctrl_down[0] = True
            if key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
                alt_down[0] = True
            if ctrl_down[0] and alt_down[0] and not self._hotkey_held:
                self._hotkey_held = True
                self._on_push_start()

        def on_release(key):
            if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                ctrl_down[0] = False
            if key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
                alt_down[0] = False
            if self._hotkey_held and not (ctrl_down[0] and alt_down[0]):
                self._hotkey_held = False
                self._on_push_stop()

        def _run_listener():
            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()
            listener.join()

        threading.Thread(target=_run_listener, daemon=True).start()

    # ── Push-to-talk ──────────────────────────────────────────────────────────

    def _on_push_start(self):
        with self._proc_lock:
            if self._processing:
                return
        self.title = "🔴"
        self._recorder.start()

    def _on_push_stop(self):
        with self._proc_lock:
            if self._processing:
                return
            self._processing = True
        self.title = "⏳"
        wav_data = self._recorder.stop()
        self._executor.submit(self._pipeline, wav_data)

    # ── AI pipeline ───────────────────────────────────────────────────────────

    def _pipeline(self, wav_data: bytes):
        try:
            transcript = transcribe_wav(wav_data)

            if not transcript:
                self._set_status("No speech detected")
                return

            self._set_status(f"You: {transcript}")
            self._show_overlay(f"You: {transcript}\n\n…")
            self._set_status("Thinking…")

            response_text = stream_qwen(
                self._history,
                transcript,
                on_chunk=lambda text: self._update_overlay(
                    f"You: {transcript}\n\n{text}"
                ),
            )

            self._history.append({"role": "user",      "content": transcript})
            self._history.append({"role": "assistant",  "content": response_text})
            if len(self._history) > MAX_HISTORY_TURNS * 2:
                self._history = self._history[-(MAX_HISTORY_TURNS * 2):]

            self._show_overlay(f"You: {transcript}\n\n{response_text}")
            self._set_status("Ready")

        except Exception as exc:
            self._set_status(f"Error: {exc}")
            self._show_overlay(f"Error: {exc}")
        finally:
            with self._proc_lock:
                self._processing = False
            self.title = "🎙"

    # ── Menu actions ──────────────────────────────────────────────────────────

    @rumps.clicked("Status")
    def _status_noop(self, _):
        pass

    @rumps.clicked("Clear History")
    def _clear_history(self, _):
        self._history.clear()
        self._set_status("History cleared")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, text: str):
        if "Status" in self.menu:
            self.menu["Status"].title = text

    def _show_overlay(self, text: str):
        if self._tk_root and self._overlay:
            self._tk_root.after(0, lambda: self._overlay.show(text))

    def _update_overlay(self, text: str):
        if self._tk_root and self._overlay:
            self._tk_root.after(0, lambda: self._overlay.update_text(text))

    def quit_app(self, _):
        self._recorder.close()
        self._executor.shutdown(wait=False)
        rumps.quit_application()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ClickyApp().run()
