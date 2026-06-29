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
import time

import librosa
from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ctypes bindings for libwhisper.so
# ---------------------------------------------------------------------------

_lib = ctypes.CDLL("/usr/local/lib/libwhisper.so")

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


class WhisperVadParams(ctypes.Structure):
    _fields_ = [
        ("threshold", ctypes.c_float),
        ("min_speech_duration_ms", ctypes.c_int),
        ("min_silence_duration_ms", ctypes.c_int),
        ("max_speech_duration_s", ctypes.c_float),
        ("speech_pad_ms", ctypes.c_int),
        ("samples_overlap", ctypes.c_float),
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
        ("carry_initial_prompt", ctypes.c_bool),
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
        ("vad", ctypes.c_bool),
        ("vad_model_path", ctypes.c_char_p),
        ("vad_params", WhisperVadParams),
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


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

class WhisperCppModel:
    def __init__(self, model_path, device="NPU"):
        self.ctx = None
        self.model_path = model_path
        self.device = device
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

    def transcribe(self, audio_f32):
        params = _lib.whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        params.print_realtime = False
        params.print_progress = False
        params.print_timestamps = False
        params.print_special = False
        params.no_timestamps = True
        params.single_segment = False
        params.n_threads = 4
        params.language = b"en"

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


model = None
import threading as _threading
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
    parser.add_argument(
        "--model",
        default=os.path.expanduser("~/.cache/whisper/ggml-base.bin"),
    )
    parser.add_argument("--device", default="NPU", help="OpenVINO device (NPU, CPU, GPU)")
    args = parser.parse_args()

    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        "/usr/local/lib/openvino:/usr/local/lib:/usr/local/lib64",
    )

    model = WhisperCppModel(args.model, device=args.device)
    app.run(host="0.0.0.0", port=args.port, threaded=True)
