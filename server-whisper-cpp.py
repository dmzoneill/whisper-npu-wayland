"""
Whisper.cpp streaming server using libwhisper.so via ctypes.

Loads the model once at startup with OpenVINO NPU encoder acceleration.
Exposes /transcribe (batch) and /transcribe/stream (chunked) endpoints.

Usage:
    python3 server-whisper-cpp.py [--port 5001] [--model ~/.cache/whisper/ggml-base.bin] [--device NPU]
"""

import argparse
import ctypes
import io
import logging
import os
import threading as _threading
import time

import librosa
from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ctypes bindings for libwhisper.so
# ---------------------------------------------------------------------------

import ctypes.util as _ctutil
_whisper_lib_path = (
    _ctutil.find_library("whisper") or
    next((p for p in ["/usr/local/lib64/libwhisper.so", "/usr/local/lib/libwhisper.so"] if os.path.exists(p)), None)
)
if _whisper_lib_path is None:
    raise OSError("libwhisper.so not found — run: make install-whisper-cpp")
_lib = ctypes.CDLL(_whisper_lib_path)

# Probe version — whisper_version() added in v1.5.0; older builds lack it.
try:
    _lib.whisper_version.restype = ctypes.c_char_p
    _lib.whisper_version.argtypes = []
    _WHISPER_VERSION = _lib.whisper_version().decode()
    logger.info("whisper.cpp version: %s", _WHISPER_VERSION)
except AttributeError:
    _WHISPER_VERSION = "unknown"
    logger.warning("whisper_version() not exported — version unknown; struct layout unverifiable")

WHISPER_SAMPLING_GREEDY = 0
WHISPER_SAMPLING_BEAM_SEARCH = 1


class WhisperAhead(ctypes.Structure):
    _fields_ = [("n_text_layer", ctypes.c_int), ("n_head", ctypes.c_int)]


class WhisperAheads(ctypes.Structure):
    _fields_ = [
        ("n_heads", ctypes.c_size_t),
        ("heads", ctypes.POINTER(WhisperAhead)),
    ]


class WhisperContextParams(ctypes.Structure):
    _fields_ = [
        ("use_gpu", ctypes.c_bool),
        ("flash_attn", ctypes.c_bool),
        ("gpu_device", ctypes.c_int),
        ("dtw_token_timestamps", ctypes.c_bool),
        ("dtw_aheads_preset", ctypes.c_int),
        ("dtw_n_top", ctypes.c_int),
        ("dtw_aheads", WhisperAheads),
        ("dtw_mem_size", ctypes.c_size_t),
    ]



class WhisperFullParams(ctypes.Structure):
    _fields_ = [
        ("strategy", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("n_max_text_ctx", ctypes.c_int),
        ("offset_ms", ctypes.c_int),
        ("duration_ms", ctypes.c_int),
        ("translate", ctypes.c_bool),
        ("no_context", ctypes.c_bool),
        ("no_timestamps", ctypes.c_bool),
        ("single_segment", ctypes.c_bool),
        ("print_special", ctypes.c_bool),
        ("print_progress", ctypes.c_bool),
        ("print_realtime", ctypes.c_bool),
        ("print_timestamps", ctypes.c_bool),
        ("token_timestamps", ctypes.c_bool),
        ("thold_pt", ctypes.c_float),
        ("thold_ptsum", ctypes.c_float),
        ("max_len", ctypes.c_int),
        ("split_on_word", ctypes.c_bool),
        ("max_tokens", ctypes.c_int),
        ("debug_mode", ctypes.c_bool),
        ("audio_ctx", ctypes.c_int),
        ("tdrz_enable", ctypes.c_bool),
        ("suppress_regex", ctypes.c_char_p),
        ("initial_prompt", ctypes.c_char_p),
        ("prompt_tokens", ctypes.c_void_p),
        ("prompt_n_tokens", ctypes.c_int),
        ("language", ctypes.c_char_p),
        ("detect_language", ctypes.c_bool),
        ("suppress_blank", ctypes.c_bool),
        ("suppress_nst", ctypes.c_bool),
        ("temperature", ctypes.c_float),
        ("max_initial_ts", ctypes.c_float),
        ("length_penalty", ctypes.c_float),
        ("temperature_inc", ctypes.c_float),
        ("entropy_thold", ctypes.c_float),
        ("logprob_thold", ctypes.c_float),
        ("no_speech_thold", ctypes.c_float),
        ("greedy_best_of", ctypes.c_int),
        ("beam_size", ctypes.c_int),
        ("beam_patience", ctypes.c_float),
        ("new_segment_callback", ctypes.c_void_p),
        ("new_segment_callback_user_data", ctypes.c_void_p),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("encoder_begin_callback", ctypes.c_void_p),
        ("encoder_begin_callback_user_data", ctypes.c_void_p),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_user_data", ctypes.c_void_p),
        ("logits_filter_callback", ctypes.c_void_p),
        ("logits_filter_callback_user_data", ctypes.c_void_p),
        ("grammar_rules", ctypes.c_void_p),
        ("n_grammar_rules", ctypes.c_size_t),
        ("i_start_rule", ctypes.c_size_t),
        ("grammar_penalty", ctypes.c_float),
    ]


# Function signatures
_lib.whisper_context_default_params.restype = WhisperContextParams
_lib.whisper_context_default_params.argtypes = []

_lib.whisper_init_from_file_with_params.restype = ctypes.c_void_p
_lib.whisper_init_from_file_with_params.argtypes = [ctypes.c_char_p, WhisperContextParams]

_lib.whisper_ctx_init_openvino_encoder.restype = ctypes.c_int
_lib.whisper_ctx_init_openvino_encoder.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p
]

_lib.whisper_full_default_params.restype = WhisperFullParams
_lib.whisper_full_default_params.argtypes = [ctypes.c_int]

_lib.whisper_full.restype = ctypes.c_int
_lib.whisper_full.argtypes = [
    ctypes.c_void_p, WhisperFullParams,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
]

_lib.whisper_full_n_segments.restype = ctypes.c_int
_lib.whisper_full_n_segments.argtypes = [ctypes.c_void_p]

_lib.whisper_full_get_segment_text.restype = ctypes.c_char_p
_lib.whisper_full_get_segment_text.argtypes = [ctypes.c_void_p, ctypes.c_int]

_lib.whisper_free.restype = None
_lib.whisper_free.argtypes = [ctypes.c_void_p]


def _verify_struct_layout():
    """Check WhisperFullParams layout matches the loaded libwhisper.so.

    whisper_full_default_params() fills known constants. If our ctypes struct
    misaligns any field before no_speech_thold, the value we read back will be
    garbage. Fatal error on mismatch — silent misalignment is worse than a crash.
    """
    d = _lib.whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
    checks = [
        ("no_speech_thold", d.no_speech_thold, 0.6, 0.05),
        ("entropy_thold",   d.entropy_thold,   2.4, 0.1),
        ("temperature",     d.temperature,     0.0, 0.01),
    ]
    bad = [(name, got, exp) for name, got, exp, tol in checks if abs(got - exp) > tol]
    if bad:
        for name, got, exp in bad:
            logger.error(
                "WhisperFullParams struct mismatch: %s expected ~%.3f got %.3f",
                name, exp, got,
            )
        raise RuntimeError(
            "WhisperFullParams layout does not match libwhisper.so "
            f"(version={_WHISPER_VERSION}). "
            "Update the ctypes struct in server-whisper-cpp.py to match your whisper.cpp build."
        )
    logger.info("WhisperFullParams struct layout verified (version=%s)", _WHISPER_VERSION)


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

class WhisperCppModel:
    def __init__(self, model_path, device="NPU", n_threads=None, language="en", no_speech_thold=0.6):
        self.ctx = None
        self.model_path = model_path
        self.device = device
        self.n_threads = n_threads or max(1, (os.cpu_count() or 4) // 2)
        self.language = language.encode()
        self.no_speech_thold = no_speech_thold
        self._load()

    def _load(self):
        logger.info("Loading model: %s", self.model_path)
        t0 = time.time()

        cparams = _lib.whisper_context_default_params()
        self.ctx = _lib.whisper_init_from_file_with_params(
            self.model_path.encode(), cparams
        )
        if not self.ctx:
            raise RuntimeError(f"Failed to load model: {self.model_path}")

        ov_model = self.model_path.replace(".bin", "-encoder-openvino.xml")
        ov_cache = self.model_path.replace(".bin", "-encoder-openvino-cache")
        if os.path.exists(ov_model):
            logger.info("Initializing OpenVINO encoder on %s", self.device)
            ret = _lib.whisper_ctx_init_openvino_encoder(
                self.ctx,
                ov_model.encode(),
                self.device.encode(),
                ov_cache.encode(),
            )
            if ret != 0:
                logger.warning("OpenVINO encoder init failed (code %d), using CPU fallback", ret)
        else:
            logger.info("No OpenVINO encoder model found, using CPU")

        logger.info("Model loaded in %.1fs", time.time() - t0)

        silence = (ctypes.c_float * 16000)()  # 1s of zeros — warmup
        self.transcribe(silence, no_speech_thold=1.0)
        logger.info("Warmup complete")

        t = _threading.Thread(target=self._keepalive, daemon=True)
        t.start()

    def _keepalive(self, interval=60):
        silence = (ctypes.c_float * 16000)()
        while True:
            time.sleep(interval)
            with _inference_lock:
                self.transcribe(silence, no_speech_thold=1.0)

    def transcribe(self, audio_f32, no_speech_thold=None):
        params = _lib.whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        params.print_realtime = False
        params.print_progress = False
        params.print_timestamps = False
        params.print_special = False
        params.no_timestamps = True
        params.single_segment = False
        params.n_threads = self.n_threads
        params.language = self.language
        params.no_speech_thold = self.no_speech_thold if no_speech_thold is None else no_speech_thold

        arr = (ctypes.c_float * len(audio_f32))(*audio_f32)
        t0 = time.time()
        ret = _lib.whisper_full(self.ctx, params, arr, len(audio_f32))
        elapsed = time.time() - t0

        if ret != 0:
            logger.error("whisper_full failed with code %d", ret)
            return ""

        n_segments = _lib.whisper_full_n_segments(self.ctx)
        segments = []
        for i in range(n_segments):
            text = _lib.whisper_full_get_segment_text(self.ctx, i)
            if text:
                segments.append(text.decode("utf-8"))

        full_text = "".join(segments).strip()
        duration = len(audio_f32) / 16000
        logger.info(
            "Transcribed %.1fs audio in %.2fs (%.1fx realtime)",
            duration, elapsed, duration / elapsed if elapsed > 0 else 0,
        )
        return full_text

    def __del__(self):
        if self.ctx:
            _lib.whisper_free(self.ctx)


_verify_struct_layout()

model = None
_inference_lock = _threading.Lock()

MAX_AUDIO_SECONDS = 30


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_data = request.get_data()
    if not audio_data:
        return jsonify({"error": "No audio data"}), 400
    try:
        audio_f32, _ = librosa.load(io.BytesIO(audio_data), sr=16000)
        with _inference_lock:
            text = model.transcribe(audio_f32)
        return jsonify({"text": text})
    except Exception as e:
        logger.error("Error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/transcribe/stream", methods=["POST"])
def transcribe_stream():
    audio_data = request.get_data()
    if not audio_data:
        return jsonify({"error": "No audio data"}), 400
    try:
        audio_f32, _ = librosa.load(io.BytesIO(audio_data), sr=16000)
        max_samples = MAX_AUDIO_SECONDS * 16000
        if len(audio_f32) > max_samples:
            audio_f32 = audio_f32[-max_samples:]
        with _inference_lock:
            text = model.transcribe(audio_f32)
        return jsonify({"text": text, "is_partial": True})
    except Exception as e:
        logger.error("Error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "backend": "whisper.cpp", "model": model.model_path})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper.cpp streaming server")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--model", default=os.path.expanduser("~/.cache/whisper/ggml-base.bin"))
    parser.add_argument("--device", default="NPU", help="OpenVINO device (NPU, CPU, GPU)")
    parser.add_argument("--threads", type=int, default=None, help="Inference threads (default: cpu_count/2)")
    parser.add_argument("--language", default="en", help="Whisper language code (default: en)")
    parser.add_argument(
        "--no-speech-thold", type=float, default=0.6, dest="no_speech_thold",
        help="no_speech probability threshold 0.0-1.0 (default: 0.6; set 1.0 to disable for PTT)",
    )
    args = parser.parse_args()

    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        "/usr/local/lib/openvino:/usr/local/lib:/usr/local/lib64",
    )

    model = WhisperCppModel(
        args.model,
        device=args.device,
        n_threads=args.threads,
        language=args.language,
        no_speech_thold=args.no_speech_thold,
    )
    app.run(host="0.0.0.0", port=args.port, threaded=True)
