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
import os
import signal
import subprocess
import sys
import threading
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


def stop_recording(audio_buf):
    """Stop recording and return WAV bytes."""
    raw_audio = audio_buf.stop()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_audio)
    return buf.getvalue()


async def transcribe_stream(wav_bytes, port, type_delay_ms):
    """Send audio to whisper server, stream results, and type each chunk immediately."""
    import aiohttp
    url = f"http://127.0.0.1:{port}/transcribe/stream"
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


async def transcribe_batch(wav_bytes, port):
    """Send audio to whisper server and return text (non-streaming fallback)."""
    import aiohttp
    url = f"http://127.0.0.1:{port}/transcribe"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=wav_bytes) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("text", "").strip()
            else:
                log.error("Server returned %d", resp.status)
                return ""


async def transcribe_chunk(wav_bytes, port):
    """Send audio chunk to whisper-cpp server for partial transcription."""
    import aiohttp
    url = f"http://127.0.0.1:{port}/transcribe/stream"
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


def try_dbus_handoff(text):
    """Try to hand off transcription to GNOME extension via D-Bus.
    Uses native GLib D-Bus with 500ms timeout — returns in <10ms when
    the extension isn't installed, avoiding subprocess overhead."""
    try:
        from gi.repository import Gio, GLib
        bus = Gio.bus_get_sync(Gio.BusType.SESSION)
        result = bus.call_sync(
            'org.gnome.Shell',
            '/com/whisper/LanguageBuddy',
            'com.whisper.LanguageBuddy',
            'HandleTranscription',
            GLib.Variant('(s)', (text,)),
            GLib.VariantType('(b)'),
            Gio.DBusCallFlags.NONE,
            500,
            None
        )
        return result.unpack()[0]
    except Exception:
        return False


async def main():
    parser = argparse.ArgumentParser(description="Push-to-talk voice dictation")
    parser.add_argument("--key", default="KEY_RIGHTCTRL", help="Hold key name (default: KEY_RIGHTCTRL)")
    parser.add_argument("--port", type=int, default=5000, help="Whisper server port (default: 5000)")
    parser.add_argument("--type-delay", type=int, default=4, help="ydotool inter-key delay in ms (default: 4)")
    parser.add_argument("--backend", choices=["openvino", "whisper-cpp"], default="openvino",
                        help="Transcription backend (default: openvino)")
    parser.add_argument("--stream-interval", type=float, default=3.0,
                        help="Seconds between streaming transcription updates (default: 3.0)")
    args = parser.parse_args()

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

    recording_proc = None
    streaming_task = None
    typed_char_count = 0
    streamed_text = ""
    key_down_time = 0.0
    mode = None  # "hold" or "toggle" — decided after HOLD_THRESHOLD
    HOLD_THRESHOLD = 1.5

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
        """Periodically transcribe and type incrementally (toggle mode — key not held).
        Only appends new text. If Whisper corrects earlier words, backspace just the
        changed suffix and retype from the divergence point."""
        nonlocal typed_char_count, streamed_text
        await asyncio.sleep(interval)
        while audio_buf.proc.poll() is None:
            raw_audio = audio_buf.snapshot()
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
            text = await transcribe_chunk(wav_bytes, port)
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
        """Stop recording, do final transcription, and type result.
        Tries D-Bus handoff to GNOME extension first; falls back to direct typing."""
        nonlocal recording_proc, streaming_task, typed_char_count, streamed_text, mode

        if streaming_task:
            streaming_task.cancel()
            try:
                await streaming_task
            except asyncio.CancelledError:
                pass
            streaming_task = None

        wav_bytes = stop_recording(recording_proc)
        recording_proc = None
        duration = (len(wav_bytes) - 44) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
        if duration < 0.3:
            log.info("Too short (%.1fs), skipping", duration)
            typed_char_count = 0
            mode = None
            return
        log.info("Recorded %.1fs of audio", duration)

        # Toggle mode with whisper-cpp: live diff correction (no D-Bus handoff)
        if args.backend == "whisper-cpp" and mode == "toggle":
            text = await transcribe_chunk(wav_bytes, port)
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
            text = await transcribe_chunk(wav_bytes, port)
        else:
            text = await transcribe_batch(wav_bytes, port)

        if not text:
            log.info("No transcription returned")
            mode = None
            return

        text = text.strip()

        if try_dbus_handoff(text):
            log.info("Handed off to Language Buddy: %s", text[:80])
        else:
            log.info("Typing: %s", text[:80])
            type_text(text, delay_ms=delay_ms)

        mode = None

    async for event in kbd.async_read_loop():
        if event.type != ecodes.EV_KEY:
            continue
        key_event = evdev.categorize(event)
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
                recording_proc = record_audio()
            elif mode == "toggle":
                # Second tap in toggle mode — stop and finalize
                log.info("Toggle stop")
                await finalize_recording(backend_port, args.type_delay)

        elif key_event.keystate == key_event.key_up:
            if recording_proc is None:
                continue
            hold_duration = asyncio.get_event_loop().time() - key_down_time

            if mode == "toggle":
                # Key up in toggle mode — ignore, we're waiting for second tap
                pass
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
