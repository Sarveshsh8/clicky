#!/usr/bin/env python3
"""
Clicky - macOS menu bar AI assistant
Requires: pip install rumps pyaudio pynput requests pillow
Run servers first: ./start-servers.sh
"""

import base64
import io
import json
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, Future

import pyaudio
import requests
import rumps
from pynput import keyboard

# ── Config ────────────────────────────────────────────────────────────────────

WHISPER_URL = "http://localhost:8081/inference"
QWEN_URL    = "http://localhost:8080/v1/chat/completions"
QWEN_MODEL  = "mlx-community/Qwen2.5-3B-Instruct-4bit"

SYSTEM_PROMPT = (
    "You are Clicky, a concise macOS desktop assistant. "
    "You can see the user's screen and hear their voice. "
    "Give short, direct answers. No markdown. No bullet lists unless asked. "
    "If asked about something on screen, reference it specifically."
)

MAX_HISTORY_TURNS         = 10   # keep last N user+assistant pairs
OVERLAY_AUTO_HIDE_SECONDS = 12
OVERLAY_UPDATE_MIN_GAP    = 0.08  # throttle overlay redraws to ~12fps

# Audio
SAMPLE_RATE = 16_000
CHANNELS    = 1
CHUNK       = 1024
FORMAT      = pyaudio.paInt16

# ── Shared HTTP session (reuses TCP connections) ───────────────────────────────

_http = requests.Session()
_http.headers.update({"Connection": "keep-alive"})

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
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
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

# ── Screen capture ────────────────────────────────────────────────────────────

def capture_screenshot_b64() -> str | None:
    """Capture primary screen → base64 JPEG. Returns None on failure."""
    try:
        from PIL import ImageGrab
        img = ImageGrab.grab()
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=55)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None

# ── Whisper STT ───────────────────────────────────────────────────────────────

def transcribe_wav(wav_data: bytes) -> str:
    resp = _http.post(
        WHISPER_URL,
        files={"file": ("audio.wav", wav_data, "audio/wav")},
        data={"response_format": "json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("text", "").strip()

# ── Qwen LLM ──────────────────────────────────────────────────────────────────

def build_user_message(transcript: str, screenshot_b64: str | None) -> dict:
    if screenshot_b64:
        return {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{screenshot_b64}",
                        "detail": "low",
                    },
                },
                {"type": "text", "text": transcript},
            ],
        }
    return {"role": "user", "content": transcript}


def stream_qwen(messages: list[dict], on_chunk) -> str:
    """Stream Qwen response. on_chunk(accumulated_text) called per token."""
    payload = {
        "model": QWEN_MODEL,
        "messages": messages,
        "stream": True,
        "max_tokens": 512,
        "temperature": 0.7,
    }
    accumulated = ""
    last_update  = 0.0
    with _http.post(QWEN_URL, json=payload, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                if delta:
                    accumulated += delta
                    now = time.monotonic()
                    if now - last_update >= OVERLAY_UPDATE_MIN_GAP:
                        on_chunk(accumulated)
                        last_update = now
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    # Always emit final accumulated text
    on_chunk(accumulated)
    return accumulated

# ── Tkinter overlay ───────────────────────────────────────────────────────────

class OverlayWindow:
    """Floating dark window near the cursor. All methods called on main thread via after()."""

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
            self._win,
            text="",
            bg="#1e1e2e",
            fg="#cdd6f4",
            font=("SF Pro Text", 13),
            wraplength=420,
            justify="left",
            padx=16,
            pady=12,
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

# ── Server health check ───────────────────────────────────────────────────────

def check_servers() -> tuple[bool, bool]:
    """Returns (whisper_ok, qwen_ok). Non-blocking, 1s timeout each."""
    def ping(url):
        try:
            _http.get(url.rsplit("/", 1)[0], timeout=1)
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        whisper_f = pool.submit(ping, WHISPER_URL)
        qwen_f    = pool.submit(ping, "http://localhost:8080/v1/models")
        return whisper_f.result(), qwen_f.result()

# ── Main app ──────────────────────────────────────────────────────────────────

class ClickyApp(rumps.App):
    def __init__(self):
        super().__init__("🎙", quit_button="Quit")
        self.menu = ["Status", None, "Clear History", "Check Servers"]

        self._recorder    = AudioRecorder()
        self._history: list[dict] = []  # text-only OpenAI message history
        self._hotkey_held = False
        self._proc_lock   = threading.Lock()
        self._processing  = False
        self._overlay     = None
        self._tk_root     = None
        self._executor    = ThreadPoolExecutor(max_workers=3)

        # Screenshot captured at key-down (while user speaks) in background
        self._screenshot_future: Future | None = None

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

        threading.Thread(
            target=lambda: keyboard.Listener(
                on_press=on_press, on_release=on_release
            ).join(),
            daemon=True,
        ).start()

    # ── Push-to-talk ──────────────────────────────────────────────────────────

    def _on_push_start(self):
        with self._proc_lock:
            if self._processing:
                return
        self.title = "🔴"
        # Capture screenshot NOW while user speaks — hides latency
        self._screenshot_future = self._executor.submit(capture_screenshot_b64)
        self._recorder.start()

    def _on_push_stop(self):
        with self._proc_lock:
            if self._processing:
                return
            self._processing = True
        self.title = "⏳"
        wav_data = self._recorder.stop()
        screenshot_future = self._screenshot_future
        self._screenshot_future = None
        self._executor.submit(self._pipeline, wav_data, screenshot_future)

    # ── AI pipeline ───────────────────────────────────────────────────────────

    def _pipeline(self, wav_data: bytes, screenshot_future: Future | None):
        try:
            # Transcribe + wait for screenshot in parallel
            transcribe_future = self._executor.submit(transcribe_wav, wav_data)
            screenshot_b64    = screenshot_future.result() if screenshot_future else None
            transcript        = transcribe_future.result()

            if not transcript:
                self._set_status("No speech detected")
                return

            self._set_status(f"You: {transcript}")
            self._show_overlay(f"You: {transcript}\n\n…")

            # Build messages — history stores text only (no base64 bloat)
            user_msg = build_user_message(transcript, screenshot_b64)
            messages = (
                [{"role": "system", "content": SYSTEM_PROMPT}]
                + self._history
                + [user_msg]
            )

            self._set_status("Thinking…")
            response_text = stream_qwen(
                messages,
                on_chunk=lambda text: self._update_overlay(
                    f"You: {transcript}\n\n{text}"
                ),
            )

            # Store text-only in history (never store image base64)
            self._history.append({"role": "user",      "content": transcript})
            self._history.append({"role": "assistant", "content": response_text})
            if len(self._history) > MAX_HISTORY_TURNS * 2:
                self._history = self._history[-(MAX_HISTORY_TURNS * 2):]

            self._show_overlay(f"You: {transcript}\n\n{response_text}")
            self._set_status("Ready")

        except requests.exceptions.ConnectionError:
            self._set_status("Server offline")
            self._show_overlay("Servers not running.\nRun: ./start-servers.sh")
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

    @rumps.clicked("Check Servers")
    def _check_servers(self, _):
        self._set_status("Checking servers…")
        def _do():
            whisper_ok, qwen_ok = check_servers()
            w = "✓ Whisper" if whisper_ok else "✗ Whisper"
            q = "✓ Qwen"   if qwen_ok    else "✗ Qwen"
            self._set_status(f"{w}  {q}")
        self._executor.submit(_do)

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
