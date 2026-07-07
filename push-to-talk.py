#!/usr/bin/env python3
"""
Push-to-talk voice dictation for GNOME/Wayland.

Hold Right Ctrl to record, release to transcribe and type.
Uses evdev for key detection, wl-copy + ydotool/wtype for typing.

Usage:
    python3 push-to-talk.py [--key KEY_RIGHTCTRL] [--port 5000]
"""

import argparse
import asyncio
import io
import json
import logging
import math
import os
import signal
import sqlite3
import struct
import subprocess
import sys
import threading
import time as _time
import wave

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ptt")

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit
HOLD_THRESHOLD = 1.5
CORRECTION_WINDOW = 3.0
MODEL_SIZE_ORDER = ["tiny", "base", "small", "medium", "large"]

VOICE_COMMANDS = {
    "select all":       [("key", "29:1 30:1 30:0 29:0")],
    "undo that":        [("key", "29:1 44:1 44:0 29:0")],
    "redo that":        [("key", "29:1 21:1 21:0 29:0")],
    "copy that":        [("key", "29:1 46:1 46:0 29:0")],
    "paste that":       [("key", "29:1 47:1 47:0 29:0")],
    "cut that":         [("key", "29:1 45:1 45:0 29:0")],
    "new line":         [("key", "28:1 28:0")],
    "new paragraph":    [("key", "28:1 28:0"), ("key", "28:1 28:0")],
    "tab key":          [("key", "15:1 15:0")],
    "delete that":      [("key", "14:1 14:0")],
    "delete word":      [("key", "29:1 14:1 14:0 29:0")],
    "period":           [("type", ".")],
    "comma":            [("type", ",")],
    "question mark":    [("type", "?")],
    "exclamation mark": [("type", "!")],
    "exclamation point":[("type", "!")],
    "colon":            [("type", ":")],
    "semicolon":        [("type", ";")],
    "open quote":       [("type", "\"")],
    "close quote":      [("type", "\"")],
}

DEFAULT_APP_CONTEXTS = {
    "org.mozilla.Thunderbird.desktop": {"tones": ["professional"]},
    "org.mozilla.thunderbird.desktop": {"tones": ["professional"]},
    "com.slack.Slack.desktop": {"tones": ["diplomatic"]},
    "org.gnome.Evolution.desktop": {"tones": ["professional"]},
    "org.gnome.Terminal.desktop": {"tones": []},
    "org.gnome.Console.desktop": {"tones": []},
    "com.raggesilver.BlackBox.desktop": {"tones": []},
    "code.desktop": {"tones": []},
}


def find_keyboard():
    """Find the first keyboard device in /dev/input/."""
    import evdev
    for path in evdev.list_devices():
        dev = evdev.InputDevice(path)
        if "virtual" in dev.name.lower():
            continue
        caps = dev.capabilities(verbose=True)
        for (etype, _), codes in caps.items():
            if etype == "EV_KEY":
                key_names = [c[0] if isinstance(c[0], str) else c[0][0] for c in codes]
                if "KEY_A" in key_names and "KEY_ENTER" in key_names:
                    return dev
    return None


class AudioBuffer:
    """Continuously reads from parec into a growing buffer so the pipe never blocks."""

    def __init__(self, proc):
        self.proc = proc
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while True:
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                break
            with self._lock:
                self._buf.extend(chunk)

    def snapshot(self):
        """Return a copy of all audio accumulated so far."""
        with self._lock:
            return bytes(self._buf)

    def stop(self):
        """Stop recording and return the final raw audio bytes."""
        self.proc.send_signal(signal.SIGTERM)
        self._thread.join(timeout=2)
        self.proc.wait()
        with self._lock:
            return bytes(self._buf)


def record_audio():
    """Record audio from default PipeWire/PulseAudio source using parec."""
    proc = subprocess.Popen(
        [
            "parec",
            "--format=s16le",
            "--rate=16000",
            "--channels=1",
            "--latency-msec=10",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return AudioBuffer(proc)


def stop_recording(audio_buf, vad_threshold=-40):
    """Stop recording and return WAV bytes, or None if no speech detected."""
    raw_audio = audio_buf.stop()
    raw_audio = trim_silence(raw_audio, threshold_db=vad_threshold)
    if not raw_audio:
        return None

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_audio)
    return buf.getvalue()


def trim_silence(raw_audio, threshold_db=-40, pad_frames=3, frame_ms=20):
    """Trim leading/trailing silence using energy-based VAD."""
    if not raw_audio:
        return raw_audio
    frame_size = int(SAMPLE_RATE * frame_ms / 1000) * SAMPLE_WIDTH
    n_frames = len(raw_audio) // frame_size
    if n_frames == 0:
        return raw_audio

    first_speech = -1
    last_speech = -1
    for i in range(n_frames):
        chunk = raw_audio[i * frame_size:(i + 1) * frame_size]
        samples = struct.unpack(f"<{len(chunk) // 2}h", chunk)
        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        db = 20 * math.log10(rms / 32768) if rms > 0 else -100
        if db > threshold_db:
            if first_speech < 0:
                first_speech = i
            last_speech = i

    if first_speech < 0:
        return b""

    start = max(0, first_speech - pad_frames) * frame_size
    end = min(n_frames, last_speech + pad_frames + 1) * frame_size
    return raw_audio[start:end]


async def transcribe_stream(wav_bytes, port, type_delay_ms, language=None):
    """Send audio to whisper server, stream results, and type each chunk immediately."""
    import aiohttp
    url = f"http://127.0.0.1:{port}/transcribe/stream"
    if language:
        url += f"?language={language}"
    typed_any = False
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, data=wav_bytes) as resp:
            if resp.status != 200:
                log.warning("Stream endpoint returned %d, falling back to batch", resp.status)
                return await transcribe_batch(wav_bytes, port)
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if data.get("done"):
                    break
                chunk = data.get("text", "")
                if chunk:
                    type_text(chunk, delay_ms=type_delay_ms)
                    typed_any = True
    return typed_any


async def transcribe_batch(wav_bytes, port, language=None):
    """Send audio to whisper server and return text (non-streaming fallback)."""
    import aiohttp
    url = f"http://127.0.0.1:{port}/transcribe"
    if language:
        url += f"?language={language}"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=wav_bytes) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("text", "").strip()
            else:
                log.error("Server returned %d", resp.status)
                return ""


async def transcribe_chunk(wav_bytes, port, language=None):
    """Send audio chunk to whisper-cpp server for partial transcription."""
    import aiohttp
    url = f"http://127.0.0.1:{port}/transcribe/stream"
    if language:
        url += f"?language={language}"
    timeout = aiohttp.ClientTimeout(total=15)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, data=wav_bytes) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("text", "").strip()
                else:
                    log.error("Stream chunk returned %d", resp.status)
                    return ""
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        log.warning("Transcription request failed: %s", e)
        return ""


def type_text(text, delay_ms=2):
    """Type text into the focused window using wtype (Wayland) or xdotool (X11)."""
    if not text:
        return

    session_type = os.environ.get("XDG_SESSION_TYPE", "")

    if session_type == "wayland":
        try:
            subprocess.run(["ydotool", "type", "-d", str(delay_ms), "--", text], check=True)
            return
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            log.warning("ydotool failed: %s", e)
        try:
            subprocess.run(["wtype", "-d", str(delay_ms), "--", text], check=True)
            return
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            log.warning("wtype failed: %s", e)
        try:
            subprocess.run(["wl-copy", text], check=True)
            log.info("Text copied to clipboard (install ydotool or wtype for direct typing)")
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            log.error("All typing methods failed: %s", e)
    else:
        try:
            subprocess.run(["xdotool", "type", "--delay", str(delay_ms), "--clearmodifiers", "--", text], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            log.error("xdotool failed: %s", e)


def _try_dbus_call(bus_name, object_path, interface, method, args_variant, reply_type):
    """Try a single D-Bus call, returning the unpacked result or None."""
    try:
        from gi.repository import Gio, GLib
        bus = Gio.bus_get_sync(Gio.BusType.SESSION)
        result = bus.call_sync(
            bus_name, object_path, interface, method,
            args_variant, GLib.VariantType(reply_type) if reply_type else None,
            Gio.DBusCallFlags.NONE, 500, None
        )
        return result.unpack()
    except Exception:
        return None


DBUS_BUS_NAMES = ['org.gnome.Shell', 'com.whisper.LanguageBuddy']
DBUS_OBJECT_PATH = '/com/whisper/LanguageBuddy'
DBUS_INTERFACE = 'com.whisper.LanguageBuddy'


def try_dbus_handoff(text):
    """Try to hand off transcription to desktop extension via D-Bus.
    Tries GNOME Shell first, then standalone KDE service."""
    from gi.repository import GLib
    for bus_name in DBUS_BUS_NAMES:
        result = _try_dbus_call(
            bus_name, DBUS_OBJECT_PATH, DBUS_INTERFACE,
            'HandleTranscription',
            GLib.Variant('(s)', (text,)), '(b)')
        if result and result[0]:
            return True
    return False


def notify(summary, body="", timeout_ms=3000):
    """Show a desktop notification."""
    cmd = ["notify-send", "--expire-time", str(timeout_ms), summary]
    if body:
        cmd.append(body)
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass


def send_keys(scancode_str):
    """Send key events via ydotool (Wayland) or xdotool (X11)."""
    session_type = os.environ.get("XDG_SESSION_TYPE", "")
    if session_type == "wayland":
        try:
            subprocess.run(["ydotool", "key"] + scancode_str.split(), check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    else:
        key_map = {
            "29:1": "ctrl+", "29:0": "", "30:1": "a", "30:0": "",
            "44:1": "z", "44:0": "", "21:1": "y", "21:0": "",
            "46:1": "c", "46:0": "", "47:1": "v", "47:0": "",
            "45:1": "x", "45:0": "", "28:1": "Return", "28:0": "",
            "15:1": "Tab", "15:0": "", "14:1": "BackSpace", "14:0": "",
        }
        keys = []
        combo = ""
        for code in scancode_str.split():
            mapped = key_map.get(code, "")
            if mapped.endswith("+"):
                combo += mapped
            elif mapped:
                keys.append(combo + mapped)
                combo = ""
        for key in keys:
            try:
                subprocess.run(["xdotool", "key", key], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass


def execute_voice_command(text):
    """Check if text matches a voice command and execute it. Returns True if handled."""
    normalized = text.strip().lower().rstrip(".!?,;")
    if normalized.endswith(" please"):
        normalized = normalized[:-7]
    for phrase, actions in VOICE_COMMANDS.items():
        if normalized == phrase:
            for action_type, action_val in actions:
                if action_type == "key":
                    send_keys(action_val)
                elif action_type == "type":
                    type_text(action_val, delay_ms=2)
            return True
    return False


class TranscriptionHistory:
    """Rolling log of recent transcriptions in SQLite."""

    def __init__(self, max_items=50):
        self.max_items = max_items
        db_dir = os.path.expanduser("~/.config/whisper-npu")
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = os.path.join(db_dir, "history.db")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS history "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, timestamp REAL NOT NULL)"
            )

    def add(self, text):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO history (text, timestamp) VALUES (?, ?)",
                         (text, _time.time()))
            conn.execute(
                "DELETE FROM history WHERE id NOT IN "
                "(SELECT id FROM history ORDER BY id DESC LIMIT ?)",
                (self.max_items,))

    def recent(self, n=10):
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT text, timestamp FROM history ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()


def get_focused_app():
    """Query desktop extension for the focused application via D-Bus.
    Tries GNOME Shell first, then standalone KDE service."""
    for bus_name in DBUS_BUS_NAMES:
        result = _try_dbus_call(
            bus_name, DBUS_OBJECT_PATH, DBUS_INTERFACE,
            'GetFocusedApp', None, '(ss)')
        if result:
            return result
    return ("", "")


def load_app_contexts():
    """Load per-app context overrides from config file, merged with defaults."""
    contexts = dict(DEFAULT_APP_CONTEXTS)
    config_path = os.path.expanduser("~/.config/whisper-npu/app-contexts.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                contexts.update(json.load(f))
        except Exception:
            pass
    return contexts


def try_dbus_handoff_with_context(text, context_json):
    """Hand off transcription with per-app context to desktop extension."""
    from gi.repository import GLib
    for bus_name in DBUS_BUS_NAMES:
        result = _try_dbus_call(
            bus_name, DBUS_OBJECT_PATH, DBUS_INTERFACE,
            'HandleTranscriptionWithContext',
            GLib.Variant('(ss)', (text, context_json)), '(b)')
        if result and result[0]:
            return True
    return False


def try_dbus_history_picker(history):
    """Ask desktop extension to show history picker overlay."""
    from gi.repository import GLib
    items = history.recent(10)
    if not items:
        return False
    items_json = json.dumps([{"text": t, "ts": ts} for t, ts in items])
    for bus_name in DBUS_BUS_NAMES:
        result = _try_dbus_call(
            bus_name, DBUS_OBJECT_PATH, DBUS_INTERFACE,
            'ShowHistoryPicker',
            GLib.Variant('(s)', (items_json,)), '(b)')
        if result and result[0]:
            return True
    return False


def play_sound(sound_name):
    """Play a freedesktop sound non-blocking."""
    path = f"/usr/share/sounds/freedesktop/stereo/{sound_name}.oga"
    if not os.path.exists(path):
        return
    for player in ("pw-play", "paplay"):
        try:
            subprocess.Popen([player, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except FileNotFoundError:
            continue


async def auto_punctuate(text, port):
    """Post-process text through LLM for punctuation."""
    import aiohttp
    url = f"http://127.0.0.1:{port}/punctuate"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"text": text}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("text", text).strip()
    except Exception as e:
        log.warning("Auto-punctuate failed: %s", e)
    return text


async def translate_text(text, target_language, port):
    """Translate text through LLM."""
    import aiohttp
    url = f"http://127.0.0.1:{port}/translate"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"text": text, "target_language": target_language}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("text", text).strip()
    except Exception as e:
        log.warning("Translation failed: %s", e)
    return text


import re

WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

ABBREVIATIONS = {
    r"\bmister\b": "Mr.", r"\bmissus\b": "Mrs.", r"\bdoctor\b": "Dr.",
    r"\bversus\b": "vs.", r"\bet cetera\b": "etc.", r"\bfor example\b": "e.g.",
    r"\bthat is\b": "i.e.", r"\bmister\b": "Mr.",
}


def format_dictation(text):
    """Apply dictation formatting rules."""
    for pattern, replacement in ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = re.sub(
        r"\b(\w+)\s+at\s+(\w+)\s+dot\s+(\w+)\b",
        lambda m: f"{m.group(1)}@{m.group(2)}.{m.group(3)}",
        text, flags=re.IGNORECASE
    )

    text = re.sub(
        r"\bw w w dot\s+(\w+)\s+dot\s+(\w+)\b",
        lambda m: f"www.{m.group(1)}.{m.group(2)}",
        text, flags=re.IGNORECASE
    )

    for word, num in sorted(WORD_NUMS.items(), key=lambda x: -len(x[0])):
        text = re.sub(rf"\b{word}\b", str(num), text, flags=re.IGNORECASE)

    text = re.sub(r"\b(\d+)\s+hundred\s+(\d+)\b", lambda m: str(int(m.group(1)) * 100 + int(m.group(2))), text)
    text = re.sub(r"\b(\d+)\s+hundred\b", lambda m: str(int(m.group(1)) * 100), text)

    if text:
        text = text[0].upper() + text[1:]

    return text


async def main():
    parser = argparse.ArgumentParser(description="Push-to-talk voice dictation")
    parser.add_argument("--key", default="KEY_RIGHTCTRL", help="Hold key name (default: KEY_RIGHTCTRL)")
    parser.add_argument("--port", type=int, default=5000, help="Whisper server port (default: 5000)")
    parser.add_argument("--type-delay", type=int, default=4, help="ydotool inter-key delay in ms (default: 4)")
    parser.add_argument("--backend", choices=["openvino", "whisper-cpp"], default="openvino",
                        help="Transcription backend (default: openvino)")
    parser.add_argument("--stream-interval", type=float, default=3.0,
                        help="Seconds between streaming transcription updates (default: 3.0)")
    parser.add_argument("--vad-threshold", type=int, default=-40,
                        help="VAD silence threshold in dB (default: -40)")
    parser.add_argument("--no-notify", action="store_true", default=False,
                        help="Disable desktop notifications after transcription")
    parser.add_argument("--no-commands", action="store_true", default=False,
                        help="Disable voice command detection")
    parser.add_argument("--recall-key", default="KEY_PAUSE",
                        help="Key to show transcription history picker (default: KEY_PAUSE)")
    parser.add_argument("--language", default=None,
                        help="Language code for multilingual models (e.g., 'de', 'fr')")
    parser.add_argument("--auto-punctuate", action="store_true", default=False,
                        help="Post-process transcriptions with LLM for punctuation")
    parser.add_argument("--translate-to", default=None,
                        help="Translate transcriptions to target language (e.g., 'French')")
    parser.add_argument("--no-sound", action="store_true", default=False,
                        help="Disable audio feedback sounds")
    parser.add_argument("--no-formatting", action="store_true", default=False,
                        help="Disable dictation formatting (numbers, abbreviations)")
    args = parser.parse_args()

    language = args.language or os.environ.get("WHISPER_LANGUAGE", "") or None
    do_auto_punctuate = args.auto_punctuate or os.environ.get("WHISPER_AUTO_PUNCTUATE", "") == "1"
    translate_to = args.translate_to or os.environ.get("WHISPER_TRANSLATE_TO", "") or None
    do_sound = not (args.no_sound or os.environ.get("WHISPER_NO_SOUND", "") == "1")
    do_formatting = not (args.no_formatting or os.environ.get("WHISPER_NO_FORMATTING", "") == "1")

    try:
        import aiohttp  # noqa: F401
    except ImportError:
        log.error("aiohttp required: pip install aiohttp")
        sys.exit(1)

    import evdev
    from evdev import ecodes

    target_key = getattr(ecodes, args.key, None)
    if target_key is None:
        log.error("Unknown key: %s", args.key)
        sys.exit(1)

    kbd = find_keyboard()
    if kbd is None:
        log.error("No keyboard found in /dev/input/ — run with sudo or add user to 'input' group")
        sys.exit(1)

    backend_port = args.port if args.backend == "openvino" else 5001
    log.info("Listening on %s (key: %s, backend: %s, port: %d)",
             kbd.name, args.key, args.backend, backend_port)
    log.info("Hold %s to record, release to transcribe and type", args.key)

    history = TranscriptionHistory()
    app_contexts = load_app_contexts()

    recall_key = getattr(ecodes, args.recall_key, None)

    recording_proc = None
    streaming_task = None
    typed_char_count = 0
    streamed_text = ""
    key_down_time = 0.0
    mode = None  # "hold", "toggle", or "correction"
    last_wav_bytes = None
    last_typed_text = ""
    last_transcription_end = 0.0
    last_used_dbus = False

    def backspace_n(n):
        """Send n backspace keys."""
        if n <= 0:
            return
        session_type = os.environ.get("XDG_SESSION_TYPE", "")
        if session_type == "wayland":
            try:
                for _ in range(n):
                    subprocess.run(["ydotool", "key", "14:1", "14:0"], check=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass
        else:
            subprocess.run(["xdotool", "key", "--repeat", str(n), "BackSpace"],
                           check=True)

    def backspace_typed():
        nonlocal typed_char_count
        backspace_n(typed_char_count)
        typed_char_count = 0

    async def streaming_loop_live(audio_buf, port, interval, delay_ms):
        """Periodically transcribe and type incrementally (toggle mode — key not held)."""
        nonlocal typed_char_count, streamed_text
        await asyncio.sleep(interval)
        while audio_buf.proc.poll() is None:
            raw_audio = audio_buf.snapshot()
            raw_audio = trim_silence(raw_audio, threshold_db=args.vad_threshold)
            if len(raw_audio) < SAMPLE_RATE * SAMPLE_WIDTH:
                await asyncio.sleep(0.5)
                continue
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(raw_audio)
            wav_bytes = buf.getvalue()
            duration = len(raw_audio) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
            if duration < 0.5:
                await asyncio.sleep(0.5)
                continue
            log.info("Streaming chunk: %.1fs of audio", duration)
            text = await transcribe_chunk(wav_bytes, port, language=language)
            if text and text != streamed_text:
                common = 0
                for a, b in zip(streamed_text, text):
                    if a == b:
                        common += 1
                    else:
                        break
                chars_to_erase = len(streamed_text) - common
                new_suffix = text[common:]
                if chars_to_erase > 0:
                    backspace_n(chars_to_erase)
                    typed_char_count -= chars_to_erase
                if new_suffix:
                    type_text(new_suffix, delay_ms=delay_ms)
                    typed_char_count += len(new_suffix)
                log.info("Live (-%d +%d): %s", chars_to_erase, len(new_suffix), text[:80])
                streamed_text = text
            await asyncio.sleep(interval)

    async def finalize_recording(port, delay_ms):
        """Stop recording, do final transcription, and type result."""
        nonlocal recording_proc, streaming_task, typed_char_count, streamed_text, mode
        nonlocal last_wav_bytes, last_typed_text, last_transcription_end, last_used_dbus

        if streaming_task:
            streaming_task.cancel()
            try:
                await streaming_task
            except asyncio.CancelledError:
                pass
            streaming_task = None

        wav_bytes = stop_recording(recording_proc, vad_threshold=args.vad_threshold)
        recording_proc = None

        if wav_bytes is None:
            log.info("No speech detected (silence only)")
            typed_char_count = 0
            mode = None
            return

        duration = (len(wav_bytes) - 44) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
        if duration < 0.3:
            log.info("Too short (%.1fs), skipping", duration)
            typed_char_count = 0
            mode = None
            return
        log.info("Recorded %.1fs of audio", duration)

        # Toggle mode with whisper-cpp: live diff correction (no D-Bus handoff)
        if args.backend == "whisper-cpp" and mode == "toggle":
            text = await transcribe_chunk(wav_bytes, port, language=language)
            if text:
                common = 0
                for a, b in zip(streamed_text, text):
                    if a == b:
                        common += 1
                    else:
                        break
                chars_to_erase = len(streamed_text) - common
                new_suffix = text[common:]
                if chars_to_erase > 0:
                    backspace_n(chars_to_erase)
                if new_suffix:
                    type_text(new_suffix, delay_ms=delay_ms)
                log.info("Final (-%d +%d): %s", chars_to_erase, len(new_suffix), text[:80])
            typed_char_count = 0
            streamed_text = ""
            mode = None
            return

        # Hold mode: transcribe, try D-Bus handoff, fallback to direct typing
        log.info("Transcribing...")
        if args.backend == "whisper-cpp":
            text = await transcribe_chunk(wav_bytes, port, language=language)
        else:
            text = await transcribe_batch(wav_bytes, port, language=language)

        if not text:
            log.info("No transcription returned")
            mode = None
            return

        text = text.strip()

        if do_sound:
            play_sound("message")

        # Post-processing pipeline
        if do_formatting:
            text = format_dictation(text)
        if do_auto_punctuate:
            text = await auto_punctuate(text, port)
        if translate_to:
            original = text
            text = await translate_text(text, translate_to, port)
            log.info("Translated: %s -> %s", original[:40], text[:40])

        # Save to history
        history.add(text)

        # Check for voice commands
        if not args.no_commands and execute_voice_command(text):
            log.info("Voice command: %s", text[:80])
            if not args.no_notify:
                notify("Voice Command", text[:80])
            mode = None
            return

        # Per-app context detection
        app_id, window_title = get_focused_app()
        context = app_contexts.get(app_id, {})
        used_dbus = False

        if context.get("tones"):
            context_json = json.dumps({"tones": context["tones"], "app": app_id})
            if try_dbus_handoff_with_context(text, context_json):
                log.info("Context-aware handoff (%s): %s", app_id, text[:80])
                used_dbus = True
        if not used_dbus and try_dbus_handoff(text):
            log.info("Handed off to Language Buddy: %s", text[:80])
            used_dbus = True
        if not used_dbus:
            log.info("Typing: %s", text[:80])
            type_text(text, delay_ms=delay_ms)

        if not args.no_notify:
            notify("Whisper", text[:200])
        if do_sound:
            play_sound("complete")

        # Save state for correction mode
        last_wav_bytes = wav_bytes
        last_typed_text = text
        last_transcription_end = asyncio.get_event_loop().time()
        last_used_dbus = used_dbus

        mode = None

    async def correct_last_transcription():
        """Re-transcribe last audio with a larger model and replace typed text."""
        nonlocal last_wav_bytes, last_typed_text, mode
        import aiohttp

        if not last_wav_bytes or not last_typed_text:
            mode = None
            return

        if last_used_dbus:
            log.info("Correction not available after Language Buddy handoff")
            notify("Whisper", "Correction not available after Language Buddy")
            mode = None
            return

        url = f"http://127.0.0.1:{backend_port}/models"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        mode = None
                        return
                    data = await resp.json()
                    models = data.get("models", [])
        except Exception:
            mode = None
            return

        # Find the current model's size rank and pick a larger one
        current_default = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://127.0.0.1:{backend_port}/model/default") as resp:
                    if resp.status == 200:
                        current_default = (await resp.json()).get("model", "")
        except Exception:
            pass

        current_rank = -1
        for i, size in enumerate(MODEL_SIZE_ORDER):
            if current_default and size in current_default.lower():
                current_rank = i
                break

        larger = None
        best_rank = current_rank
        for m in models:
            for i, size in enumerate(MODEL_SIZE_ORDER):
                if size in m.lower() and i > best_rank:
                    larger = m
                    best_rank = i
                    break

        if not larger:
            log.info("No larger model available for correction")
            notify("Whisper", "No larger model available")
            mode = None
            return

        log.info("Correction: re-transcribing with %s", larger)
        url = f"http://127.0.0.1:{backend_port}/transcribe/{larger}"
        if language:
            url += f"?language={language}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=last_wav_bytes) as resp:
                    if resp.status != 200:
                        mode = None
                        return
                    data = await resp.json()
                    new_text = data.get("text", "").strip()
        except Exception:
            mode = None
            return

        if new_text and new_text != last_typed_text:
            backspace_n(len(last_typed_text))
            type_text(new_text, delay_ms=args.type_delay)
            log.info("Corrected: %s -> %s", last_typed_text[:40], new_text[:40])
            if not args.no_notify:
                notify("Correction", f"{new_text[:200]}")
            last_typed_text = new_text
        else:
            log.info("Same result with larger model")
            if not args.no_notify:
                notify("Whisper", "Same result with larger model")

        mode = None

    async for event in kbd.async_read_loop():
        if event.type != ecodes.EV_KEY:
            continue
        key_event = evdev.categorize(event)

        # Recall key — show history picker
        if recall_key and key_event.scancode == recall_key and key_event.keystate == key_event.key_down:
            if not try_dbus_history_picker(history):
                items = history.recent(1)
                if items:
                    try:
                        subprocess.run(["wl-copy", items[0][0]], check=True,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        notify("Whisper", "Last transcription copied to clipboard")
                    except (FileNotFoundError, subprocess.CalledProcessError):
                        pass
            continue

        if key_event.scancode != target_key:
            continue

        if key_event.keystate == key_event.key_down:
            if recording_proc is None:
                # Fresh press — start recording, decide mode later
                key_down_time = asyncio.get_event_loop().time()
                mode = None
                typed_char_count = 0
                streamed_text = ""
                log.info("Recording...")
                if do_sound:
                    play_sound("message-new-instant")
                recording_proc = record_audio()
            elif mode == "toggle":
                # Second tap in toggle mode — stop and finalize
                log.info("Toggle stop")
                await finalize_recording(backend_port, args.type_delay)

        elif key_event.keystate == key_event.key_up:
            if recording_proc is None:
                continue
            hold_duration = asyncio.get_event_loop().time() - key_down_time
            now = asyncio.get_event_loop().time()

            if mode == "toggle":
                # Key up in toggle mode — ignore, we're waiting for second tap
                pass
            elif hold_duration < 0.5 and last_wav_bytes and (now - last_transcription_end) < CORRECTION_WINDOW:
                # Quick tap shortly after transcription — correction mode
                mode = "correction"
                log.info("Correction mode — re-transcribing with larger model")
                if recording_proc:
                    stop_recording(recording_proc, vad_threshold=args.vad_threshold)
                    recording_proc = None
                await correct_last_transcription()
            elif hold_duration < HOLD_THRESHOLD:
                # Quick tap — enter toggle mode with live streaming
                mode = "toggle"
                log.info("Toggle mode — tap again to stop (held %.1fs)", hold_duration)
                if args.backend == "whisper-cpp":
                    streaming_task = asyncio.create_task(
                        streaming_loop_live(recording_proc, backend_port,
                                           args.stream_interval, args.type_delay)
                    )
            else:
                # Long hold — finalize immediately
                mode = "hold"
                log.info("Hold mode — transcribing (held %.1fs)", hold_duration)
                await finalize_recording(backend_port, args.type_delay)


if __name__ == "__main__":
    asyncio.run(main())
