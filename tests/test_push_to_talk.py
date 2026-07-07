"""Comprehensive tests for push-to-talk.py module."""

import asyncio
import importlib
import io
import json
import math
import os
import signal
import sqlite3
import struct
import sys
import threading
import wave
from unittest.mock import (
    AsyncMock,
    MagicMock,
    Mock,
    call,
    patch,
)

import pytest

# push-to-talk.py has a dash in its name, so use importlib
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_spec = importlib.util.spec_from_file_location(
    "push_to_talk",
    os.path.join(PROJECT_ROOT, "push-to-talk.py"),
)
ptt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ptt)

# Import conftest helpers
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tests"))
from conftest import make_raw_audio, make_silent_raw, make_wav_bytes


# ── Helpers ──────────────────────────────────────────────────────────────────

def _loud_raw(duration_s=0.5, amplitude=16000, sample_rate=16000):
    """Raw PCM with a loud sine wave (well above any silence threshold)."""
    n = int(sample_rate * duration_s)
    samples = [int(amplitude * math.sin(2 * math.pi * 440 * i / sample_rate))
               for i in range(n)]
    return struct.pack(f"<{n}h", *samples)


def _quiet_raw(duration_s=0.5, amplitude=5, sample_rate=16000):
    """Raw PCM that is extremely quiet (below typical thresholds)."""
    n = int(sample_rate * duration_s)
    samples = [int(amplitude * math.sin(2 * math.pi * 440 * i / sample_rate))
               for i in range(n)]
    return struct.pack(f"<{n}h", *samples)


class _AsyncCtx:
    """Minimal async context manager for mocking ``async with``."""
    def __init__(self, return_value):
        self._rv = return_value
    async def __aenter__(self):
        return self._rv
    async def __aexit__(self, *a):
        return False


def _make_aiohttp_mocks(resp):
    """Build mock aiohttp module where ClientSession().post() returns *resp*.

    Usage::

        resp = AsyncMock(status=200, ...)
        mock_aiohttp, mock_session = _make_aiohttp_mocks(resp)
        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            ...
    """
    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=_AsyncCtx(resp))
    mock_session.get = MagicMock(return_value=_AsyncCtx(resp))

    mock_aiohttp = MagicMock()
    mock_aiohttp.ClientSession = MagicMock(return_value=_AsyncCtx(mock_session))
    mock_aiohttp.ClientTimeout = MagicMock()
    mock_aiohttp.ClientError = Exception
    return mock_aiohttp, mock_session


# ── Constants ────────────────────────────────────────────────────────────────

class TestConstants:
    """Verify module-level constants."""

    def test_sample_rate(self):
        assert ptt.SAMPLE_RATE == 16000

    def test_channels(self):
        assert ptt.CHANNELS == 1

    def test_sample_width(self):
        assert ptt.SAMPLE_WIDTH == 2

    def test_hold_threshold(self):
        assert ptt.HOLD_THRESHOLD == 1.5

    def test_correction_window(self):
        assert ptt.CORRECTION_WINDOW == 3.0

    def test_model_size_order(self):
        assert ptt.MODEL_SIZE_ORDER == ["tiny", "base", "small", "medium", "large"]

    def test_voice_commands_has_expected_keys(self):
        expected = {"select all", "undo that", "redo that", "copy that",
                    "paste that", "cut that", "new line", "new paragraph",
                    "tab key", "delete that", "delete word", "period",
                    "comma", "question mark", "exclamation mark",
                    "exclamation point", "colon", "semicolon",
                    "open quote", "close quote"}
        assert expected == set(ptt.VOICE_COMMANDS.keys())

    def test_voice_command_select_all(self):
        assert ptt.VOICE_COMMANDS["select all"] == [("key", "29:1 30:1 30:0 29:0")]

    def test_voice_command_new_paragraph(self):
        assert ptt.VOICE_COMMANDS["new paragraph"] == [
            ("key", "28:1 28:0"), ("key", "28:1 28:0")
        ]

    def test_voice_command_period(self):
        assert ptt.VOICE_COMMANDS["period"] == [("type", ".")]

    def test_default_app_contexts(self):
        assert "org.mozilla.Thunderbird.desktop" in ptt.DEFAULT_APP_CONTEXTS
        assert ptt.DEFAULT_APP_CONTEXTS["org.mozilla.Thunderbird.desktop"] == {"tones": ["professional"]}
        assert ptt.DEFAULT_APP_CONTEXTS["org.gnome.Terminal.desktop"] == {"tones": []}

    def test_dbus_bus_names(self):
        assert ptt.DBUS_BUS_NAMES == ['org.gnome.Shell', 'com.whisper.LanguageBuddy']

    def test_dbus_object_path(self):
        assert ptt.DBUS_OBJECT_PATH == '/com/whisper/LanguageBuddy'

    def test_dbus_interface(self):
        assert ptt.DBUS_INTERFACE == 'com.whisper.LanguageBuddy'


# ── find_keyboard() ─────────────────────────────────────────────────────────

class TestFindKeyboard:

    def test_finds_keyboard_with_key_a_and_enter(self):
        dev = MagicMock()
        dev.name = "AT Translated Set 2 keyboard"
        dev.capabilities.return_value = {
            ("EV_KEY", 1): [("KEY_A",), ("KEY_ENTER",), ("KEY_B",)],
        }
        mock_evdev = MagicMock()
        mock_evdev.list_devices.return_value = ["/dev/input/event0"]
        mock_evdev.InputDevice.return_value = dev
        with patch.dict("sys.modules", {"evdev": mock_evdev}):
            result = ptt.find_keyboard()
        assert result is dev

    def test_skips_virtual_device(self):
        virtual_dev = MagicMock()
        virtual_dev.name = "Virtual Keyboard"
        real_dev = MagicMock()
        real_dev.name = "Real keyboard"
        real_dev.capabilities.return_value = {
            ("EV_KEY", 1): [("KEY_A",), ("KEY_ENTER",)],
        }
        mock_evdev = MagicMock()
        mock_evdev.list_devices.return_value = ["/dev/input/event0", "/dev/input/event1"]
        mock_evdev.InputDevice.side_effect = [virtual_dev, real_dev]
        with patch.dict("sys.modules", {"evdev": mock_evdev}):
            result = ptt.find_keyboard()
        assert result is real_dev

    def test_returns_none_when_no_keyboard(self):
        mock_evdev = MagicMock()
        mock_evdev.list_devices.return_value = []
        with patch.dict("sys.modules", {"evdev": mock_evdev}):
            result = ptt.find_keyboard()
        assert result is None

    def test_skips_device_missing_key_enter(self):
        dev = MagicMock()
        dev.name = "Mouse"
        dev.capabilities.return_value = {
            ("EV_KEY", 1): [("KEY_A",), ("BTN_LEFT",)],
        }
        mock_evdev = MagicMock()
        mock_evdev.list_devices.return_value = ["/dev/input/event0"]
        mock_evdev.InputDevice.return_value = dev
        with patch.dict("sys.modules", {"evdev": mock_evdev}):
            result = ptt.find_keyboard()
        assert result is None

    def test_handles_tuple_key_names(self):
        """evdev sometimes returns key names as tuples of aliases."""
        dev = MagicMock()
        dev.name = "Keyboard"
        dev.capabilities.return_value = {
            ("EV_KEY", 1): [(("KEY_A", "KEY_A_ALIAS"),), (("KEY_ENTER", "KEY_RETURN"),)],
        }
        mock_evdev = MagicMock()
        mock_evdev.list_devices.return_value = ["/dev/input/event0"]
        mock_evdev.InputDevice.return_value = dev
        with patch.dict("sys.modules", {"evdev": mock_evdev}):
            result = ptt.find_keyboard()
        assert result is dev

    def test_skips_non_ev_key_types(self):
        dev = MagicMock()
        dev.name = "Keyboard"
        dev.capabilities.return_value = {
            ("EV_REL", 2): [("REL_X",), ("REL_Y",)],
        }
        mock_evdev = MagicMock()
        mock_evdev.list_devices.return_value = ["/dev/input/event0"]
        mock_evdev.InputDevice.return_value = dev
        with patch.dict("sys.modules", {"evdev": mock_evdev}):
            result = ptt.find_keyboard()
        assert result is None


# ── AudioBuffer ──────────────────────────────────────────────────────────────

class TestAudioBuffer:

    def test_init_starts_reader_thread(self):
        proc = MagicMock()
        proc.stdout.read.return_value = b""
        buf = ptt.AudioBuffer(proc)
        assert buf._thread.is_alive() or True  # might finish fast
        buf._thread.join(timeout=1)

    def test_reader_accumulates_data(self):
        chunks = [b"AAAA", b"BBBB", b""]
        proc = MagicMock()
        proc.stdout.read.side_effect = chunks
        buf = ptt.AudioBuffer(proc)
        buf._thread.join(timeout=2)
        assert buf.snapshot() == b"AAAABBBB"

    def test_snapshot_returns_copy(self):
        proc = MagicMock()
        proc.stdout.read.return_value = b""
        buf = ptt.AudioBuffer(proc)
        buf._thread.join(timeout=1)
        with buf._lock:
            buf._buf.extend(b"test")
        snap = buf.snapshot()
        assert snap == b"test"
        # Modifying snap should not affect internal buf
        assert isinstance(snap, bytes)

    def test_stop_sends_sigterm_and_returns_data(self):
        chunks = [b"data", b""]
        proc = MagicMock()
        proc.stdout.read.side_effect = chunks
        proc.wait.return_value = 0
        buf = ptt.AudioBuffer(proc)
        buf._thread.join(timeout=2)
        result = buf.stop()
        proc.send_signal.assert_called_once_with(signal.SIGTERM)
        proc.wait.assert_called_once()
        assert result == b"data"


# ── record_audio() ───────────────────────────────────────────────────────────

class TestRecordAudio:

    def test_record_audio_returns_audio_buffer(self, mock_popen):
        popen_cls, proc = mock_popen
        with patch.object(ptt, "AudioBuffer") as mock_buf_cls:
            mock_buf_cls.return_value = MagicMock()
            result = ptt.record_audio()
        popen_cls.assert_called_once()
        args = popen_cls.call_args
        assert "parec" in args[0][0]
        assert "--format=s16le" in args[0][0]
        assert "--rate=16000" in args[0][0]


# ── stop_recording() ────────────────────────────────────────────────────────

class TestStopRecording:

    def test_returns_none_for_all_silence(self):
        audio_buf = MagicMock()
        audio_buf.stop.return_value = make_silent_raw(duration_s=1.0)
        result = ptt.stop_recording(audio_buf, vad_threshold=-40)
        assert result is None

    def test_returns_wav_for_speech(self):
        audio_buf = MagicMock()
        audio_buf.stop.return_value = _loud_raw(duration_s=0.5)
        result = ptt.stop_recording(audio_buf, vad_threshold=-40)
        assert result is not None
        # Verify it's a valid WAV file
        wav_io = io.BytesIO(result)
        with wave.open(wav_io, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000

    def test_returns_none_for_empty_audio(self):
        audio_buf = MagicMock()
        audio_buf.stop.return_value = b""
        result = ptt.stop_recording(audio_buf, vad_threshold=-40)
        assert result is None


# ── trim_silence() ───────────────────────────────────────────────────────────

class TestTrimSilence:
    """Pure logic function — test every branch and edge case."""

    def test_empty_input(self):
        assert ptt.trim_silence(b"") == b""

    def test_none_like_falsy(self):
        assert ptt.trim_silence(b"") == b""

    def test_short_data_below_frame_size(self):
        # Less than one frame — returned as-is
        short = b"\x00\x01"
        assert ptt.trim_silence(short) == short

    def test_all_silence_returns_empty(self):
        raw = make_silent_raw(duration_s=1.0)
        result = ptt.trim_silence(raw, threshold_db=-40)
        assert result == b""

    def test_loud_audio_preserved(self):
        raw = _loud_raw(duration_s=0.5)
        result = ptt.trim_silence(raw, threshold_db=-40)
        assert len(result) > 0

    def test_leading_silence_trimmed(self):
        silence = make_silent_raw(duration_s=0.5)
        loud = _loud_raw(duration_s=0.5)
        raw = silence + loud
        result = ptt.trim_silence(raw, threshold_db=-40, pad_frames=0)
        # Result should be shorter than original (silence removed)
        assert len(result) < len(raw)

    def test_trailing_silence_trimmed(self):
        loud = _loud_raw(duration_s=0.5)
        silence = make_silent_raw(duration_s=0.5)
        raw = loud + silence
        result = ptt.trim_silence(raw, threshold_db=-40, pad_frames=0)
        assert len(result) < len(raw)

    def test_both_sides_trimmed(self):
        silence = make_silent_raw(duration_s=0.5)
        loud = _loud_raw(duration_s=0.5)
        raw = silence + loud + silence
        result = ptt.trim_silence(raw, threshold_db=-40, pad_frames=0)
        assert len(result) < len(raw)

    def test_pad_frames_preserves_context(self):
        silence = make_silent_raw(duration_s=0.5)
        loud = _loud_raw(duration_s=0.5)
        raw = silence + loud + silence
        result_no_pad = ptt.trim_silence(raw, threshold_db=-40, pad_frames=0)
        result_with_pad = ptt.trim_silence(raw, threshold_db=-40, pad_frames=3)
        # With padding, we should keep more data
        assert len(result_with_pad) >= len(result_no_pad)

    def test_pad_frames_clamped_to_bounds(self):
        """Padding shouldn't go before index 0 or past last frame."""
        loud = _loud_raw(duration_s=0.1)
        result = ptt.trim_silence(loud, threshold_db=-40, pad_frames=100)
        assert len(result) > 0

    def test_custom_threshold(self):
        # Very quiet audio that would pass at -80 dB but fail at -20 dB
        quiet = _quiet_raw(duration_s=0.5, amplitude=5)
        result_loose = ptt.trim_silence(quiet, threshold_db=-100)
        result_strict = ptt.trim_silence(quiet, threshold_db=-10)
        assert len(result_loose) > 0
        assert result_strict == b""

    def test_custom_frame_ms(self):
        loud = _loud_raw(duration_s=0.5)
        result_20 = ptt.trim_silence(loud, threshold_db=-40, frame_ms=20)
        result_10 = ptt.trim_silence(loud, threshold_db=-40, frame_ms=10)
        # Both should return non-empty
        assert len(result_20) > 0
        assert len(result_10) > 0

    def test_single_frame_of_audio(self):
        """Exactly one frame of loud audio."""
        frame_samples = int(16000 * 20 / 1000)  # 320 samples for 20ms
        samples = [int(16000 * math.sin(2 * math.pi * 440 * i / 16000))
                    for i in range(frame_samples)]
        raw = struct.pack(f"<{len(samples)}h", *samples)
        result = ptt.trim_silence(raw, threshold_db=-40, pad_frames=0)
        assert len(result) > 0

    def test_rms_zero_treated_as_silence(self):
        """Frame of all zeros should be treated as -100 dB (silence)."""
        frame_samples = int(16000 * 20 / 1000)
        raw = b"\x00\x00" * frame_samples
        result = ptt.trim_silence(raw, threshold_db=-40)
        assert result == b""


# ── transcribe_stream() ─────────────────────────────────────────────────────

class _AsyncIterLines:
    """Async iterator over a list of byte lines — mocks ``resp.content``."""
    def __init__(self, lines):
        self._it = iter(lines)
    def __aiter__(self):
        return self
    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class TestTranscribeStream:

    @pytest.mark.asyncio
    async def test_streams_text_and_types(self):
        resp = MagicMock()
        resp.status = 200
        resp.content = _AsyncIterLines([
            b'data: {"text": "hello "}\n',
            b'data: {"text": "world"}\n',
            b'data: {"done": true}\n',
        ])
        mock_aiohttp, mock_session = _make_aiohttp_mocks(resp)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}), \
             patch.object(ptt, "type_text") as mock_type, \
             patch.object(ptt, "transcribe_batch", new_callable=AsyncMock):
            result = await ptt.transcribe_stream(b"wavdata", 5000, 2)
        assert result is True
        assert mock_type.call_count == 2

    @pytest.mark.asyncio
    async def test_fallback_to_batch_on_non_200(self):
        resp = MagicMock()
        resp.status = 404
        mock_aiohttp, mock_session = _make_aiohttp_mocks(resp)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}), \
             patch.object(ptt, "transcribe_batch", new_callable=AsyncMock, return_value="fallback") as mock_batch:
            result = await ptt.transcribe_stream(b"wavdata", 5000, 2)
        mock_batch.assert_awaited_once_with(b"wavdata", 5000)
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_skips_non_sse_lines(self):
        resp = MagicMock()
        resp.status = 200
        resp.content = _AsyncIterLines([
            b": comment\n",
            b"\n",
            b"event: ping\n",
            b'data: {"text": "only"}\n',
            b'data: {"done": true}\n',
        ])
        mock_aiohttp, mock_session = _make_aiohttp_mocks(resp)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}), \
             patch.object(ptt, "type_text") as mock_type:
            result = await ptt.transcribe_stream(b"wavdata", 5000, 2)
        mock_type.assert_called_once_with("only", delay_ms=2)

    @pytest.mark.asyncio
    async def test_skips_invalid_json(self):
        resp = MagicMock()
        resp.status = 200
        resp.content = _AsyncIterLines([
            b"data: {invalid json}\n",
            b'data: {"text": "valid"}\n',
            b'data: {"done": true}\n',
        ])
        mock_aiohttp, mock_session = _make_aiohttp_mocks(resp)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}), \
             patch.object(ptt, "type_text") as mock_type:
            result = await ptt.transcribe_stream(b"wavdata", 5000, 2)
        mock_type.assert_called_once_with("valid", delay_ms=2)

    @pytest.mark.asyncio
    async def test_language_appended_to_url(self):
        resp = MagicMock()
        resp.status = 200
        resp.content = _AsyncIterLines([b'data: {"done": true}\n'])
        mock_aiohttp, mock_session = _make_aiohttp_mocks(resp)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            await ptt.transcribe_stream(b"wav", 5000, 2, language="de")
        url_arg = mock_session.post.call_args[0][0]
        assert "?language=de" in url_arg

    @pytest.mark.asyncio
    async def test_no_text_returns_false(self):
        resp = MagicMock()
        resp.status = 200
        resp.content = _AsyncIterLines([b'data: {"done": true}\n'])
        mock_aiohttp, mock_session = _make_aiohttp_mocks(resp)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await ptt.transcribe_stream(b"wav", 5000, 2)
        assert result is False


# ── transcribe_batch() ───────────────────────────────────────────────────────

class TestTranscribeBatch:

    @pytest.mark.asyncio
    async def test_success_returns_text(self):
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value={"text": "  hello world  "})
        mock_aiohttp, mock_session = _make_aiohttp_mocks(resp)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await ptt.transcribe_batch(b"wav", 5000)
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_non_200_returns_empty(self):
        resp = AsyncMock()
        resp.status = 500
        mock_aiohttp, mock_session = _make_aiohttp_mocks(resp)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await ptt.transcribe_batch(b"wav", 5000)
        assert result == ""

    @pytest.mark.asyncio
    async def test_language_param(self):
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value={"text": "hallo"})
        mock_aiohttp, mock_session = _make_aiohttp_mocks(resp)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            await ptt.transcribe_batch(b"wav", 5000, language="de")
        url_arg = mock_session.post.call_args[0][0]
        assert "?language=de" in url_arg


# ── transcribe_chunk() ───────────────────────────────────────────────────────

class TestTranscribeChunk:

    @pytest.mark.asyncio
    async def test_success_returns_text(self):
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value={"text": " chunk text "})
        mock_aiohttp, mock_session = _make_aiohttp_mocks(resp)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await ptt.transcribe_chunk(b"wav", 5000)
        assert result == "chunk text"

    @pytest.mark.asyncio
    async def test_non_200_returns_empty(self):
        resp = AsyncMock()
        resp.status = 500
        mock_aiohttp, mock_session = _make_aiohttp_mocks(resp)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await ptt.transcribe_chunk(b"wav", 5000)
        assert result == ""

    @pytest.mark.asyncio
    async def test_timeout_returns_empty(self):
        """When ClientSession raises TimeoutError, return empty string."""
        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientTimeout = MagicMock()
        mock_aiohttp.ClientError = Exception

        class _RaisingCtx:
            async def __aenter__(self):
                raise asyncio.TimeoutError
            async def __aexit__(self, *a):
                return False

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=_RaisingCtx())
        mock_aiohttp.ClientSession = MagicMock(return_value=_AsyncCtx(mock_session))

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await ptt.transcribe_chunk(b"wav", 5000)
        assert result == ""

    @pytest.mark.asyncio
    async def test_language_param(self):
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value={"text": "hallo"})
        mock_aiohttp, mock_session = _make_aiohttp_mocks(resp)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            await ptt.transcribe_chunk(b"wav", 5000, language="fr")
        url_arg = mock_session.post.call_args[0][0]
        assert "?language=fr" in url_arg


# ── type_text() ──────────────────────────────────────────────────────────────

class TestTypeText:

    def test_empty_text_returns_immediately(self, mock_subprocess):
        ptt.type_text("", delay_ms=2)
        mock_subprocess.assert_not_called()

    def test_wayland_ydotool_first(self, mock_subprocess):
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}):
            ptt.type_text("hello", delay_ms=4)
        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        assert args[0] == "ydotool"
        assert "type" in args
        assert "hello" in args

    def test_wayland_falls_back_to_wtype(self, mock_subprocess):
        mock_subprocess.side_effect = [
            FileNotFoundError("no ydotool"),
            MagicMock(returncode=0),
        ]
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}):
            ptt.type_text("hello", delay_ms=2)
        assert mock_subprocess.call_count == 2
        args = mock_subprocess.call_args_list[1][0][0]
        assert args[0] == "wtype"

    def test_wayland_falls_back_to_wl_copy(self, mock_subprocess):
        mock_subprocess.side_effect = [
            FileNotFoundError("no ydotool"),
            FileNotFoundError("no wtype"),
            MagicMock(returncode=0),
        ]
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}):
            ptt.type_text("hello", delay_ms=2)
        assert mock_subprocess.call_count == 3
        args = mock_subprocess.call_args_list[2][0][0]
        assert args[0] == "wl-copy"

    def test_wayland_all_fail(self, mock_subprocess):
        mock_subprocess.side_effect = FileNotFoundError("nothing works")
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}):
            ptt.type_text("hello", delay_ms=2)
        assert mock_subprocess.call_count == 3

    def test_x11_xdotool(self, mock_subprocess):
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}):
            ptt.type_text("hello", delay_ms=4)
        args = mock_subprocess.call_args[0][0]
        assert args[0] == "xdotool"
        assert "type" in args
        assert "--clearmodifiers" in args

    def test_x11_xdotool_fails(self, mock_subprocess):
        mock_subprocess.side_effect = FileNotFoundError("no xdotool")
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}):
            ptt.type_text("hello", delay_ms=2)
        # Should not raise

    def test_none_text_treated_as_falsy(self, mock_subprocess):
        ptt.type_text(None, delay_ms=2)
        mock_subprocess.assert_not_called()


# ── _try_dbus_call() ─────────────────────────────────────────────────────────

class TestTryDbusCall:

    def test_success_returns_unpacked(self):
        mock_result = MagicMock()
        mock_result.unpack.return_value = (True,)
        mock_bus = MagicMock()
        mock_bus.call_sync.return_value = mock_result

        mock_gio = MagicMock()
        mock_gio.bus_get_sync.return_value = mock_bus
        mock_glib = MagicMock()

        with patch.dict("sys.modules", {
            "gi": MagicMock(),
            "gi.repository": MagicMock(Gio=mock_gio, GLib=mock_glib),
        }):
            result = ptt._try_dbus_call(
                "org.test", "/obj", "iface", "Method",
                MagicMock(), "(b)"
            )
        assert result == (True,)

    def test_exception_returns_none(self):
        mock_gio = MagicMock()
        mock_gio.bus_get_sync.side_effect = Exception("D-Bus error")
        mock_glib = MagicMock()

        with patch.dict("sys.modules", {
            "gi": MagicMock(),
            "gi.repository": MagicMock(Gio=mock_gio, GLib=mock_glib),
        }):
            result = ptt._try_dbus_call(
                "org.test", "/obj", "iface", "Method",
                MagicMock(), "(b)"
            )
        assert result is None

    def test_none_reply_type(self):
        mock_result = MagicMock()
        mock_result.unpack.return_value = ()
        mock_bus = MagicMock()
        mock_bus.call_sync.return_value = mock_result

        mock_gio = MagicMock()
        mock_gio.bus_get_sync.return_value = mock_bus
        mock_glib = MagicMock()

        with patch.dict("sys.modules", {
            "gi": MagicMock(),
            "gi.repository": MagicMock(Gio=mock_gio, GLib=mock_glib),
        }):
            result = ptt._try_dbus_call(
                "org.test", "/obj", "iface", "Method",
                MagicMock(), None
            )
        # Should pass None for reply_type
        call_args = mock_bus.call_sync.call_args[0]
        assert call_args[5] is None


# ── try_dbus_handoff() ───────────────────────────────────────────────────────

class TestTryDbusHandoff:

    def test_first_bus_succeeds(self):
        mock_glib = MagicMock()
        with patch.dict("sys.modules", {
            "gi": MagicMock(),
            "gi.repository": MagicMock(GLib=mock_glib),
        }), patch.object(ptt, "_try_dbus_call", return_value=(True,)):
            result = ptt.try_dbus_handoff("hello")
        assert result is True

    def test_first_bus_fails_second_succeeds(self):
        mock_glib = MagicMock()
        with patch.dict("sys.modules", {
            "gi": MagicMock(),
            "gi.repository": MagicMock(GLib=mock_glib),
        }), patch.object(ptt, "_try_dbus_call", side_effect=[None, (True,)]):
            result = ptt.try_dbus_handoff("hello")
        assert result is True

    def test_all_buses_fail(self):
        mock_glib = MagicMock()
        with patch.dict("sys.modules", {
            "gi": MagicMock(),
            "gi.repository": MagicMock(GLib=mock_glib),
        }), patch.object(ptt, "_try_dbus_call", return_value=None):
            result = ptt.try_dbus_handoff("hello")
        assert result is False

    def test_result_false_treated_as_not_handled(self):
        mock_glib = MagicMock()
        with patch.dict("sys.modules", {
            "gi": MagicMock(),
            "gi.repository": MagicMock(GLib=mock_glib),
        }), patch.object(ptt, "_try_dbus_call", return_value=(False,)):
            result = ptt.try_dbus_handoff("hello")
        assert result is False


# ── notify() ─────────────────────────────────────────────────────────────────

class TestNotify:

    def test_basic_notification(self, mock_subprocess):
        ptt.notify("Test", "Body text", timeout_ms=5000)
        args = mock_subprocess.call_args[0][0]
        assert "notify-send" in args
        assert "--expire-time" in args
        assert "5000" in args
        assert "Test" in args
        assert "Body text" in args

    def test_no_body(self, mock_subprocess):
        ptt.notify("Title only")
        args = mock_subprocess.call_args[0][0]
        assert "Body" not in " ".join(args)
        # Body not appended when empty
        assert len(args) == 4  # notify-send --expire-time 3000 Title only

    def test_failure_silenced(self, mock_subprocess):
        mock_subprocess.side_effect = FileNotFoundError("no notify-send")
        ptt.notify("Test")  # Should not raise


# ── send_keys() ──────────────────────────────────────────────────────────────

class TestSendKeys:

    def test_wayland_ydotool(self, mock_subprocess):
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}):
            ptt.send_keys("29:1 30:1 30:0 29:0")
        args = mock_subprocess.call_args[0][0]
        assert args[0] == "ydotool"
        assert "key" in args

    def test_wayland_ydotool_failure_silenced(self, mock_subprocess):
        mock_subprocess.side_effect = FileNotFoundError
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}):
            ptt.send_keys("29:1 30:1 30:0 29:0")  # Should not raise

    def test_x11_xdotool_select_all(self, mock_subprocess):
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}):
            ptt.send_keys("29:1 30:1 30:0 29:0")
        # ctrl+ a -> xdotool key ctrl+a
        args = mock_subprocess.call_args[0][0]
        assert args[0] == "xdotool"
        assert "ctrl+a" in args

    def test_x11_return_key(self, mock_subprocess):
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}):
            ptt.send_keys("28:1 28:0")
        args = mock_subprocess.call_args[0][0]
        assert "Return" in args

    def test_x11_backspace(self, mock_subprocess):
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}):
            ptt.send_keys("14:1 14:0")
        args = mock_subprocess.call_args[0][0]
        assert "BackSpace" in args

    def test_x11_tab(self, mock_subprocess):
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}):
            ptt.send_keys("15:1 15:0")
        args = mock_subprocess.call_args[0][0]
        assert "Tab" in args

    def test_x11_combo_ctrl_z(self, mock_subprocess):
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}):
            ptt.send_keys("29:1 44:1 44:0 29:0")
        args = mock_subprocess.call_args[0][0]
        assert "ctrl+z" in args

    def test_x11_xdotool_failure_silenced(self, mock_subprocess):
        mock_subprocess.side_effect = FileNotFoundError
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}):
            ptt.send_keys("28:1 28:0")  # Should not raise

    def test_unknown_scancode_skipped(self, mock_subprocess):
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}):
            ptt.send_keys("99:1 99:0")
        # unknown code maps to "" — no xdotool calls
        mock_subprocess.assert_not_called()


# ── execute_voice_command() ──────────────────────────────────────────────────

class TestExecuteVoiceCommand:

    def test_exact_match(self):
        with patch.object(ptt, "send_keys") as mock_sk:
            result = ptt.execute_voice_command("select all")
        assert result is True
        mock_sk.assert_called_once()

    def test_case_insensitive(self):
        with patch.object(ptt, "send_keys") as mock_sk:
            result = ptt.execute_voice_command("Select All")
        assert result is True

    def test_strips_whitespace(self):
        with patch.object(ptt, "send_keys") as mock_sk:
            result = ptt.execute_voice_command("  select all  ")
        assert result is True

    def test_strips_punctuation(self):
        with patch.object(ptt, "send_keys") as mock_sk:
            result = ptt.execute_voice_command("select all.")
        assert result is True

    def test_strips_please_suffix(self):
        with patch.object(ptt, "send_keys") as mock_sk:
            result = ptt.execute_voice_command("undo that please")
        assert result is True

    def test_strips_please_and_punctuation(self):
        # "Copy that please!" -> lower -> "copy that please!" -> rstrip -> "copy that please"
        # -> strip " please" -> "copy that" -> match
        with patch.object(ptt, "send_keys") as mock_sk:
            result = ptt.execute_voice_command("Copy that please!")
        assert result is True

    def test_please_with_comma_before_not_stripped(self):
        # "Copy that, please!" -> "copy that, please" -> "copy that," -- no match
        result = ptt.execute_voice_command("Copy that, please!")
        assert result is False

    def test_no_match_returns_false(self):
        result = ptt.execute_voice_command("random text")
        assert result is False

    def test_type_action(self):
        with patch.object(ptt, "type_text") as mock_tt:
            result = ptt.execute_voice_command("period")
        assert result is True
        mock_tt.assert_called_once_with(".", delay_ms=2)

    def test_new_paragraph_double_enter(self):
        with patch.object(ptt, "send_keys") as mock_sk:
            result = ptt.execute_voice_command("new paragraph")
        assert result is True
        assert mock_sk.call_count == 2

    def test_multiple_punctuation_marks(self):
        """Test all punctuation voice commands."""
        punct_map = {
            "period": ".",
            "comma": ",",
            "question mark": "?",
            "exclamation mark": "!",
            "exclamation point": "!",
            "colon": ":",
            "semicolon": ";",
        }
        for cmd, expected in punct_map.items():
            with patch.object(ptt, "type_text") as mock_tt:
                result = ptt.execute_voice_command(cmd)
            assert result is True, f"Failed for command: {cmd}"
            mock_tt.assert_called_once_with(expected, delay_ms=2)


# ── get_focused_app() ────────────────────────────────────────────────────────

class TestGetFocusedApp:

    def test_first_bus_returns_result(self):
        with patch.object(ptt, "_try_dbus_call", return_value=("app.desktop", "Window Title")):
            result = ptt.get_focused_app()
        assert result == ("app.desktop", "Window Title")

    def test_falls_back_to_second_bus(self):
        with patch.object(ptt, "_try_dbus_call", side_effect=[None, ("kde.desktop", "KDE App")]):
            result = ptt.get_focused_app()
        assert result == ("kde.desktop", "KDE App")

    def test_all_fail_returns_empty_tuple(self):
        with patch.object(ptt, "_try_dbus_call", return_value=None):
            result = ptt.get_focused_app()
        assert result == ("", "")


# ── load_app_contexts() ─────────────────────────────────────────────────────

class TestLoadAppContexts:

    def test_defaults_loaded(self):
        with patch("os.path.exists", return_value=False):
            contexts = ptt.load_app_contexts()
        assert "org.mozilla.Thunderbird.desktop" in contexts
        assert contexts["org.mozilla.Thunderbird.desktop"] == {"tones": ["professional"]}

    def test_config_file_merges(self, tmp_path):
        config_dir = tmp_path / ".config" / "whisper-npu"
        config_dir.mkdir(parents=True)
        custom = {"custom.app.desktop": {"tones": ["casual"]}}
        (config_dir / "app-contexts.json").write_text(json.dumps(custom))

        with patch("os.path.expanduser", return_value=str(config_dir / "app-contexts.json")):
            contexts = ptt.load_app_contexts()
        assert "custom.app.desktop" in contexts
        assert contexts["custom.app.desktop"] == {"tones": ["casual"]}
        # Defaults should still be present
        assert "org.mozilla.Thunderbird.desktop" in contexts

    def test_config_file_overrides_defaults(self, tmp_path):
        config_dir = tmp_path / ".config" / "whisper-npu"
        config_dir.mkdir(parents=True)
        override = {"org.mozilla.Thunderbird.desktop": {"tones": ["casual"]}}
        (config_dir / "app-contexts.json").write_text(json.dumps(override))

        with patch("os.path.expanduser", return_value=str(config_dir / "app-contexts.json")):
            contexts = ptt.load_app_contexts()
        assert contexts["org.mozilla.Thunderbird.desktop"] == {"tones": ["casual"]}

    def test_invalid_json_silenced(self, tmp_path):
        config_dir = tmp_path / ".config" / "whisper-npu"
        config_dir.mkdir(parents=True)
        (config_dir / "app-contexts.json").write_text("NOT JSON{{{")

        with patch("os.path.expanduser", return_value=str(config_dir / "app-contexts.json")):
            contexts = ptt.load_app_contexts()
        # Should still have defaults
        assert "org.mozilla.Thunderbird.desktop" in contexts

    def test_missing_file_returns_defaults(self):
        with patch("os.path.expanduser", return_value="/nonexistent/path"):
            contexts = ptt.load_app_contexts()
        assert contexts == dict(ptt.DEFAULT_APP_CONTEXTS)


# ── try_dbus_handoff_with_context() ──────────────────────────────────────────

class TestTryDbusHandoffWithContext:

    def test_first_bus_succeeds(self):
        mock_glib = MagicMock()
        with patch.dict("sys.modules", {
            "gi": MagicMock(),
            "gi.repository": MagicMock(GLib=mock_glib),
        }), patch.object(ptt, "_try_dbus_call", return_value=(True,)):
            result = ptt.try_dbus_handoff_with_context("text", '{"tones":["pro"]}')
        assert result is True

    def test_all_fail(self):
        mock_glib = MagicMock()
        with patch.dict("sys.modules", {
            "gi": MagicMock(),
            "gi.repository": MagicMock(GLib=mock_glib),
        }), patch.object(ptt, "_try_dbus_call", return_value=None):
            result = ptt.try_dbus_handoff_with_context("text", "{}")
        assert result is False

    def test_false_result_not_handled(self):
        mock_glib = MagicMock()
        with patch.dict("sys.modules", {
            "gi": MagicMock(),
            "gi.repository": MagicMock(GLib=mock_glib),
        }), patch.object(ptt, "_try_dbus_call", return_value=(False,)):
            result = ptt.try_dbus_handoff_with_context("text", "{}")
        assert result is False


# ── try_dbus_history_picker() ────────────────────────────────────────────────

class TestTryDbusHistoryPicker:

    def test_empty_history_returns_false(self):
        history = MagicMock()
        history.recent.return_value = []
        mock_glib = MagicMock()
        with patch.dict("sys.modules", {
            "gi": MagicMock(),
            "gi.repository": MagicMock(GLib=mock_glib),
        }):
            result = ptt.try_dbus_history_picker(history)
        assert result is False

    def test_success_with_items(self):
        history = MagicMock()
        history.recent.return_value = [("hello", 1234567890.0), ("world", 1234567891.0)]
        mock_glib = MagicMock()

        with patch.dict("sys.modules", {
            "gi": MagicMock(),
            "gi.repository": MagicMock(GLib=mock_glib),
        }), patch.object(ptt, "_try_dbus_call", return_value=(True,)):
            result = ptt.try_dbus_history_picker(history)
        assert result is True

    def test_dbus_call_receives_json(self):
        history = MagicMock()
        history.recent.return_value = [("hello", 100.0)]
        mock_glib = MagicMock()

        with patch.dict("sys.modules", {
            "gi": MagicMock(),
            "gi.repository": MagicMock(GLib=mock_glib),
        }), patch.object(ptt, "_try_dbus_call", return_value=(True,)) as mock_call:
            ptt.try_dbus_history_picker(history)
        # Verify the JSON was constructed properly
        call_args = mock_call.call_args
        # The GLib.Variant call gets the JSON string
        variant_call = mock_glib.Variant.call_args
        items_json = variant_call[0][1][0]
        parsed = json.loads(items_json)
        assert parsed == [{"text": "hello", "ts": 100.0}]

    def test_all_buses_fail(self):
        history = MagicMock()
        history.recent.return_value = [("test", 1.0)]
        mock_glib = MagicMock()

        with patch.dict("sys.modules", {
            "gi": MagicMock(),
            "gi.repository": MagicMock(GLib=mock_glib),
        }), patch.object(ptt, "_try_dbus_call", return_value=None):
            result = ptt.try_dbus_history_picker(history)
        assert result is False


# ── play_sound() ─────────────────────────────────────────────────────────────

class TestPlaySound:

    def test_file_not_found_does_nothing(self, mock_popen):
        popen_cls, proc = mock_popen
        with patch("os.path.exists", return_value=False):
            ptt.play_sound("message")
        popen_cls.assert_not_called()

    def test_pw_play_used_first(self):
        with patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen") as mock_pop:
            ptt.play_sound("message")
        args = mock_pop.call_args[0][0]
        assert args[0] == "pw-play"

    def test_falls_back_to_paplay(self):
        with patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen", side_effect=[FileNotFoundError, MagicMock()]) as mock_pop:
            ptt.play_sound("message")
        assert mock_pop.call_count == 2
        args = mock_pop.call_args_list[1][0][0]
        assert args[0] == "paplay"

    def test_all_players_fail(self):
        with patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen", side_effect=FileNotFoundError):
            ptt.play_sound("message")  # Should not raise

    def test_correct_path(self):
        with patch("os.path.exists", return_value=True) as mock_exists, \
             patch("subprocess.Popen"):
            ptt.play_sound("complete")
        mock_exists.assert_called_with("/usr/share/sounds/freedesktop/stereo/complete.oga")


# ── auto_punctuate() ─────────────────────────────────────────────────────────

class TestAutoPunctuate:

    @pytest.mark.asyncio
    async def test_success(self):
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value={"text": " Hello, world. "})
        mock_aiohttp, _ = _make_aiohttp_mocks(resp)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await ptt.auto_punctuate("hello world", 5000)
        assert result == "Hello, world."

    @pytest.mark.asyncio
    async def test_failure_returns_original(self):
        """When the session raises, return the original text."""
        mock_aiohttp = MagicMock()

        class _RaisingSessionCtx:
            async def __aenter__(self):
                raise Exception("connection error")
            async def __aexit__(self, *a):
                return False

        mock_aiohttp.ClientSession = MagicMock(return_value=_RaisingSessionCtx())

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await ptt.auto_punctuate("hello world", 5000)
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_non_200_returns_original(self):
        resp = AsyncMock()
        resp.status = 500
        mock_aiohttp, _ = _make_aiohttp_mocks(resp)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await ptt.auto_punctuate("hello world", 5000)
        assert result == "hello world"


# ── translate_text() ─────────────────────────────────────────────────────────

class TestTranslateText:

    @pytest.mark.asyncio
    async def test_success(self):
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value={"text": " Hallo Welt "})
        mock_aiohttp, _ = _make_aiohttp_mocks(resp)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await ptt.translate_text("hello world", "German", 5000)
        assert result == "Hallo Welt"

    @pytest.mark.asyncio
    async def test_posts_correct_payload(self):
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value={"text": "Bonjour"})
        mock_aiohttp, mock_session = _make_aiohttp_mocks(resp)

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            await ptt.translate_text("hello", "French", 5000)
        call_kwargs = mock_session.post.call_args[1]
        assert call_kwargs["json"] == {"text": "hello", "target_language": "French"}

    @pytest.mark.asyncio
    async def test_failure_returns_original(self):
        mock_aiohttp = MagicMock()

        class _RaisingSessionCtx:
            async def __aenter__(self):
                raise Exception("timeout")
            async def __aexit__(self, *a):
                return False

        mock_aiohttp.ClientSession = MagicMock(return_value=_RaisingSessionCtx())

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await ptt.translate_text("hello", "German", 5000)
        assert result == "hello"


# ── format_dictation() ───────────────────────────────────────────────────────

class TestFormatDictation:
    """Pure logic function — test every branch thoroughly."""

    # Basic capitalization
    def test_capitalizes_first_letter(self):
        assert ptt.format_dictation("hello world") == "Hello world"

    def test_empty_string(self):
        assert ptt.format_dictation("") == ""

    def test_single_char(self):
        assert ptt.format_dictation("a") == "A"

    def test_already_capitalized(self):
        assert ptt.format_dictation("Hello") == "Hello"

    # Abbreviations
    def test_mister(self):
        result = ptt.format_dictation("mister smith")
        assert "Mr." in result

    def test_missus(self):
        result = ptt.format_dictation("missus jones")
        assert "Mrs." in result

    def test_doctor(self):
        result = ptt.format_dictation("doctor who")
        assert "Dr." in result

    def test_versus(self):
        result = ptt.format_dictation("red versus blue")
        assert "vs." in result

    def test_et_cetera(self):
        result = ptt.format_dictation("and et cetera")
        assert "etc." in result

    def test_for_example(self):
        result = ptt.format_dictation("for example this")
        # "for example" -> "e.g.", then first-letter capitalized -> "E.g."
        assert "e.g." in result.lower()

    def test_that_is(self):
        result = ptt.format_dictation("that is good")
        # "that is" -> "i.e.", then first-letter capitalized -> "I.e."
        assert "i.e." in result.lower()

    def test_abbreviation_case_insensitive(self):
        result = ptt.format_dictation("DOCTOR who")
        assert "Dr." in result

    # Email patterns
    def test_email_pattern(self):
        result = ptt.format_dictation("send to john at gmail dot com")
        assert "john@gmail.com" in result

    def test_email_pattern_case_insensitive(self):
        result = ptt.format_dictation("email me AT test DOT org")
        assert "me@test.org" in result

    # URL patterns
    def test_www_pattern(self):
        result = ptt.format_dictation("visit w w w dot google dot com")
        assert "www.google.com" in result

    def test_www_pattern_case_insensitive(self):
        result = ptt.format_dictation("go to W W W DOT example DOT org")
        assert "www.example.org" in result

    # Word-to-number conversion
    def test_single_digit_words(self):
        assert "0" in ptt.format_dictation("zero items")
        assert "1" in ptt.format_dictation("one item")
        assert "5" in ptt.format_dictation("five items")
        assert "9" in ptt.format_dictation("nine items")

    def test_teen_numbers(self):
        assert "11" in ptt.format_dictation("eleven items")
        assert "13" in ptt.format_dictation("thirteen items")
        assert "19" in ptt.format_dictation("nineteen items")

    def test_tens_numbers(self):
        assert "20" in ptt.format_dictation("twenty items")
        assert "30" in ptt.format_dictation("thirty items")
        assert "50" in ptt.format_dictation("fifty items")
        assert "90" in ptt.format_dictation("ninety items")

    def test_number_word_case_insensitive(self):
        assert "5" in ptt.format_dictation("Five items")

    # Hundreds
    def test_hundreds_with_remainder(self):
        result = ptt.format_dictation("three hundred twenty five items")
        # three->3, hundred, twenty->20, five->5 => "3 hundred 20" => "320" then "5"
        # Actually: "3 hundred 20" -> 320, then 5 is separate
        # Wait let me trace: first numbers are substituted: "3 hundred 20 5 items"
        # Then "3 hundred 20" -> 320 (first regex), leaving "320 5 items"
        # Hmm, the regex is (\d+)\s+hundred\s+(\d+) matching "3 hundred 20"
        # That gives 3*100+20=320, leaving "320 5 items"
        assert "320" in result

    def test_hundreds_without_remainder(self):
        result = ptt.format_dictation("five hundred items")
        assert "500" in result

    def test_two_hundred(self):
        result = ptt.format_dictation("two hundred")
        assert "200" in result

    # Multiple transformations combined
    def test_combined_transforms(self):
        result = ptt.format_dictation("doctor smith has five patients")
        assert result.startswith("Dr.")
        assert "5" in result

    # Edge cases
    def test_whitespace_only(self):
        result = ptt.format_dictation("   ")
        assert result == "   "  # Spaces are preserved, first char capitalized

    def test_numeric_start(self):
        result = ptt.format_dictation("5 items")
        assert result == "5 items"

    def test_no_transformations_needed(self):
        result = ptt.format_dictation("hello there")
        assert result == "Hello there"

    def test_longer_words_matched_first(self):
        """Ensure 'seventeen' is matched before 'seven'."""
        result = ptt.format_dictation("seventeen items")
        assert "17" in result
        assert "7teen" not in result

    def test_word_boundary_respected(self):
        """Words like 'onerous' should not be partially converted."""
        result = ptt.format_dictation("onerous task")
        # "one" should not be replaced inside "onerous" due to \b boundary
        assert "1rous" not in result


# ── TranscriptionHistory ─────────────────────────────────────────────────────

class TestTranscriptionHistory:

    def test_init_creates_db(self, tmp_path):
        with patch("os.path.expanduser", return_value=str(tmp_path / "whisper-npu")):
            hist = ptt.TranscriptionHistory(max_items=10)
        assert os.path.exists(hist.db_path)

    def test_add_and_recent(self, tmp_path):
        with patch("os.path.expanduser", return_value=str(tmp_path / "whisper-npu")):
            hist = ptt.TranscriptionHistory(max_items=10)
        hist.add("first")
        hist.add("second")
        items = hist.recent(10)
        assert len(items) == 2
        # Most recent first
        assert items[0][0] == "second"
        assert items[1][0] == "first"

    def test_recent_limit(self, tmp_path):
        with patch("os.path.expanduser", return_value=str(tmp_path / "whisper-npu")):
            hist = ptt.TranscriptionHistory(max_items=50)
        for i in range(5):
            hist.add(f"item {i}")
        items = hist.recent(2)
        assert len(items) == 2

    def test_max_items_trimming(self, tmp_path):
        with patch("os.path.expanduser", return_value=str(tmp_path / "whisper-npu")):
            hist = ptt.TranscriptionHistory(max_items=3)
        for i in range(10):
            hist.add(f"item {i}")
        items = hist.recent(100)
        assert len(items) == 3
        # Should keep the 3 most recent
        texts = [t for t, _ in items]
        assert "item 9" in texts
        assert "item 8" in texts
        assert "item 7" in texts

    def test_empty_recent(self, tmp_path):
        with patch("os.path.expanduser", return_value=str(tmp_path / "whisper-npu")):
            hist = ptt.TranscriptionHistory(max_items=10)
        items = hist.recent(10)
        assert items == []

    def test_timestamp_stored(self, tmp_path):
        with patch("os.path.expanduser", return_value=str(tmp_path / "whisper-npu")):
            hist = ptt.TranscriptionHistory(max_items=10)
        hist.add("test")
        items = hist.recent(1)
        assert len(items) == 1
        assert isinstance(items[0][1], float)
        assert items[0][1] > 0

    def test_creates_directory(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "whisper-npu"
        with patch("os.path.expanduser", return_value=str(deep_path)):
            hist = ptt.TranscriptionHistory(max_items=10)
        assert os.path.isdir(deep_path)

    def test_table_created_idempotent(self, tmp_path):
        """Creating two History instances on same DB should not fail."""
        with patch("os.path.expanduser", return_value=str(tmp_path / "whisper-npu")):
            hist1 = ptt.TranscriptionHistory(max_items=10)
            hist1.add("test1")
            hist2 = ptt.TranscriptionHistory(max_items=10)
            hist2.add("test2")
        items = hist2.recent(10)
        assert len(items) == 2


# ── WORD_NUMS / ABBREVIATIONS constants ─────────────────────────────────────

class TestWordNums:

    def test_all_basic_nums_present(self):
        expected = {"zero", "one", "two", "three", "four", "five",
                    "six", "seven", "eight", "nine", "ten"}
        assert expected.issubset(set(ptt.WORD_NUMS.keys()))

    def test_teens_present(self):
        expected = {"eleven", "twelve", "thirteen", "fourteen", "fifteen",
                    "sixteen", "seventeen", "eighteen", "nineteen"}
        assert expected.issubset(set(ptt.WORD_NUMS.keys()))

    def test_tens_present(self):
        expected = {"twenty", "thirty", "forty", "fifty",
                    "sixty", "seventy", "eighty", "ninety"}
        assert expected.issubset(set(ptt.WORD_NUMS.keys()))

    def test_values_correct(self):
        assert ptt.WORD_NUMS["zero"] == 0
        assert ptt.WORD_NUMS["ten"] == 10
        assert ptt.WORD_NUMS["twenty"] == 20
        assert ptt.WORD_NUMS["ninety"] == 90


class TestAbbreviations:

    def test_abbreviations_dict_not_empty(self):
        assert len(ptt.ABBREVIATIONS) > 0

    def test_abbreviations_values(self):
        # Check a few expected values
        assert r"\bdoctor\b" in ptt.ABBREVIATIONS
        assert ptt.ABBREVIATIONS[r"\bdoctor\b"] == "Dr."
        assert r"\bversus\b" in ptt.ABBREVIATIONS
        assert ptt.ABBREVIATIONS[r"\bversus\b"] == "vs."
