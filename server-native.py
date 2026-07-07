import librosa
import openvino_genai
from flask import Flask, request, jsonify, Response
import io
import json
import os
import logging
import queue
import time
import threading

SERVER_START_TIME = time.time()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = os.environ.get("WHISPER_DEVICE", "NPU")
MODEL_NAME = os.environ.get("WHISPER_MODEL", "whisper-small.en-fp16-ov")
LLM_DEVICE = os.environ.get("WHISPER_LLM_DEVICE", DEVICE)
LLM_MODEL = os.environ.get("WHISPER_LLM_MODEL", "")

DEFAULT_TONES = {
    "diplomatic": "Rewrite the following text to sound warmer, more considerate, and diplomatically phrased. Preserve the original meaning exactly. Return only the rewritten text, nothing else.",
    "professional": "Rewrite the following text in a formal, professional business tone. Preserve the original meaning exactly. Return only the rewritten text, nothing else.",
    "summarize": "Summarize the following text in one concise sentence. Return only the summary, nothing else.",
}

class MetricsCollector:
    def __init__(self):
        self.transcription_count = 0
        self.transcription_errors = 0
        self.total_latency = 0.0
        self.total_audio_duration = 0.0
        self.model_load_times = {}
        self.last_request = {}
        self._lock = threading.Lock()

    def record_transcription(self, latency, audio_duration, perf_metrics=None):
        with self._lock:
            self.transcription_count += 1
            self.total_latency += latency
            self.total_audio_duration += audio_duration
            if perf_metrics:
                self.last_request = perf_metrics

    def record_error(self):
        with self._lock:
            self.transcription_errors += 1

    def record_model_load(self, model_name, load_time):
        with self._lock:
            self.model_load_times[model_name] = round(load_time, 2)

    def snapshot(self):
        with self._lock:
            total = self.transcription_count + self.transcription_errors
            data = {
                "transcription_count": self.transcription_count,
                "transcription_errors": self.transcription_errors,
                "average_latency_seconds": round(self.total_latency / self.transcription_count, 3) if self.transcription_count else 0.0,
                "error_rate": round(self.transcription_errors / total, 4) if total else 0.0,
                "total_audio_seconds": round(self.total_audio_duration, 1),
                "model_load_times": dict(self.model_load_times),
                "uptime_seconds": round(time.time() - SERVER_START_TIME, 1),
            }
            if self.last_request:
                data["last_request"] = dict(self.last_request)
            return data


metrics = MetricsCollector()


class ModelManager:
    def __init__(self):
        self.models_dir = os.path.expanduser("~/.whisper/models")
        self.pipelines = {}
        self.default_model = MODEL_NAME

    def load_model(self, model_name):
        if model_name not in self.pipelines:
            model_path = os.path.join(self.models_dir, model_name)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
            logger.info(f"Loading model: {model_name} on {DEVICE}")
            t0 = time.time()
            self.pipelines[model_name] = openvino_genai.WhisperPipeline(str(model_path), device=DEVICE)
            elapsed = time.time() - t0
            logger.info(f"Model loaded in {elapsed:.1f}s")
            metrics.record_model_load(model_name, elapsed)
        return self.pipelines[model_name]

    def list_models(self):
        return [d for d in os.listdir(self.models_dir)
                if os.path.isdir(os.path.join(self.models_dir, d)) and not d.startswith('.')]


class LLMManager:
    def __init__(self):
        self.models_dir = os.path.expanduser("~/.whisper/llm-models")
        self.pipeline = None
        self.current_model = LLM_MODEL
        if not self.current_model:
            models = self.list_models()
            if models:
                self.current_model = models[0]
                logger.info(f"Auto-selected LLM: {self.current_model}")

    def load_model(self, model_name=None):
        model_name = model_name or self.current_model
        if not model_name:
            raise ValueError("No LLM model configured")
        model_path = os.path.join(self.models_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LLM model {model_name} not found at {model_path}")
        if self.current_model != model_name or self.pipeline is None:
            logger.info(f"Loading LLM: {model_name} on {LLM_DEVICE}")
            t0 = time.time()
            self.pipeline = openvino_genai.LLMPipeline(str(model_path), device=LLM_DEVICE)
            self.current_model = model_name
            logger.info(f"LLM loaded in {time.time()-t0:.1f}s")
        return self.pipeline

    def list_models(self):
        if not os.path.isdir(self.models_dir):
            return []
        return [d for d in os.listdir(self.models_dir)
                if os.path.isdir(os.path.join(self.models_dir, d)) and not d.startswith('.')]

    def rewrite(self, text, tone_name, tone_prompt):
        pipeline = self.load_model()
        prompt = f"{tone_prompt}\n\n{text}"
        t0 = time.time()
        result = pipeline.generate(prompt, max_new_tokens=512, temperature=0.3)
        elapsed = time.time() - t0
        logger.info(f"Rewrote ({tone_name}) in {elapsed:.2f}s")
        return str(result).strip()

    def punctuate(self, text):
        pipeline = self.load_model()
        prompt = f"Add proper punctuation and capitalization to this text. Return only the corrected text, nothing else.\n\n{text}"
        t0 = time.time()
        result = pipeline.generate(prompt, max_new_tokens=256, temperature=0.1)
        elapsed = time.time() - t0
        logger.info(f"Punctuated in {elapsed:.2f}s")
        return str(result).strip()

    def translate(self, text, target_language):
        pipeline = self.load_model()
        prompt = f"Translate the following text to {target_language}. Return only the translation, nothing else.\n\n{text}"
        t0 = time.time()
        result = pipeline.generate(prompt, max_new_tokens=512, temperature=0.3)
        elapsed = time.time() - t0
        logger.info(f"Translated to {target_language} in {elapsed:.2f}s")
        return str(result).strip()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": model_manager.default_model})

@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({"models": model_manager.list_models()})

@app.route("/model/default", methods=["GET"])
def get_default_model():
    return jsonify({"model": model_manager.default_model})

@app.route("/model/default", methods=["PUT"])
def set_default_model():
    data = request.get_json()
    if not data or "model" not in data:
        return jsonify({"error": "model name required"}), 400
    model_name = data["model"]
    available = model_manager.list_models()
    if model_name not in available:
        return jsonify({"error": f"model {model_name} not found"}), 404
    model_manager.load_model(model_name)
    model_manager.default_model = model_name
    logger.info(f"Default model changed to: {model_name}")
    return jsonify({"model": model_name})

@app.route("/transcribe/<model_name>", methods=["POST"])
def transcribe_with_model(model_name):
    try:
        pipeline = model_manager.load_model(model_name)
        audio_data = request.get_data()
        if not audio_data:
            return jsonify({"error": "No audio data"}), 400
        en_raw_speech, _ = librosa.load(io.BytesIO(audio_data), sr=16000)
        t0 = time.time()
        gen_kwargs = {}
        language = request.args.get("language")
        if language:
            gen_kwargs["language"] = f"<|{language}|>"
        result = pipeline.generate(en_raw_speech, **gen_kwargs)
        elapsed = time.time() - t0
        duration = len(en_raw_speech) / 16000
        logger.info(f"Transcribed {duration:.1f}s audio in {elapsed:.2f}s ({duration/elapsed:.1f}x realtime)")
        perf = _extract_perf_metrics(result)
        metrics.record_transcription(elapsed, duration, perf)
        return jsonify({"text": str(result)})
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        metrics.record_error()
        return jsonify({"error": str(e)}), 500

@app.route("/transcribe", methods=["POST"])
def transcribe():
    return transcribe_with_model(model_manager.default_model)


@app.route("/transcribe/stream", methods=["POST"])
def transcribe_stream():
    return transcribe_stream_with_model(model_manager.default_model)


@app.route("/transcribe/stream/<model_name>", methods=["POST"])
def transcribe_stream_with_model(model_name=None):
    model_name = model_name or model_manager.default_model
    try:
        pipeline = model_manager.load_model(model_name)
        audio_data = request.get_data()
        if not audio_data:
            return jsonify({"error": "No audio data"}), 400
        en_raw_speech, _ = librosa.load(io.BytesIO(audio_data), sr=16000)
        duration = len(en_raw_speech) / 16000
        if duration > 30:
            return jsonify({"error": "Streaming requires audio < 30 seconds"}), 400

        q = queue.Queue()
        language = request.args.get("language")

        def streamer_callback(text_chunk):
            q.put(text_chunk)
            return 0

        def generate():
            t0 = time.time()
            def run_inference():
                try:
                    gen_kwargs = {"streamer": streamer_callback, "return_timestamps": False}
                    if language:
                        gen_kwargs["language"] = f"<|{language}|>"
                    pipeline.generate(en_raw_speech, **gen_kwargs)
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                finally:
                    q.put(None)

            thread = threading.Thread(target=run_inference, daemon=True)
            thread.start()

            full_text = []
            while True:
                chunk = q.get()
                if chunk is None:
                    break
                full_text.append(chunk)
                yield f"data: {json.dumps({'text': chunk})}\n\n"

            elapsed = time.time() - t0
            logger.info(f"Streamed {duration:.1f}s audio in {elapsed:.2f}s ({duration/elapsed:.1f}x realtime)")
            metrics.record_transcription(elapsed, duration)
            yield f"data: {json.dumps({'done': True, 'full_text': ''.join(full_text)})}\n\n"

        return Response(generate(), mimetype="text/event-stream")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        metrics.record_error()
        return jsonify({"error": str(e)}), 500

@app.route("/rewrite", methods=["POST"])
def rewrite():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "text field required"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "empty text"}), 400

    tones = data.get("tones", list(DEFAULT_TONES.keys()))
    custom_tones = data.get("custom_tones", {})

    tone_prompts = {}
    for tone in tones:
        if tone in custom_tones:
            tone_prompts[tone] = custom_tones[tone]
        elif tone in DEFAULT_TONES:
            tone_prompts[tone] = DEFAULT_TONES[tone]

    variants = [{"tone": "original", "text": text}]

    for tone_name, tone_prompt in tone_prompts.items():
        try:
            rewritten = llm_manager.rewrite(text, tone_name, tone_prompt)
            variants.append({"tone": tone_name, "text": rewritten})
        except Exception as e:
            logger.error(f"Rewrite ({tone_name}) failed: {e}")
            variants.append({"tone": tone_name, "text": text, "error": str(e)})

    return jsonify({"variants": variants})


@app.route("/llm/models", methods=["GET"])
def list_llm_models():
    return jsonify({"models": llm_manager.list_models(), "current": llm_manager.current_model})


@app.route("/llm/model", methods=["PUT"])
def set_llm_model():
    data = request.get_json()
    if not data or "model" not in data:
        return jsonify({"error": "model name required"}), 400
    model_name = data["model"]
    available = llm_manager.list_models()
    if model_name not in available:
        return jsonify({"error": f"LLM model {model_name} not found"}), 404
    try:
        llm_manager.load_model(model_name)
        return jsonify({"model": model_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/llm/tones", methods=["GET"])
def list_tones():
    return jsonify({"tones": list(DEFAULT_TONES.keys())})


@app.route("/metrics", methods=["GET"])
def get_metrics():
    return jsonify(metrics.snapshot())


def _extract_perf_metrics(result):
    try:
        pm = result.perf_metrics
        return {
            "features_extraction_ms": round(pm.get_features_extraction_duration().mean, 2),
            "inference_ms": round(pm.get_inference_duration().mean, 2),
            "generate_ms": round(pm.get_generate_duration().mean, 2),
            "detokenization_ms": round(pm.get_detokenization_duration().mean, 2),
            "throughput_tokens_per_sec": round(pm.get_throughput().mean, 2),
            "num_generated_tokens": pm.get_num_generated_tokens(),
        }
    except Exception:
        return {}


@app.route("/punctuate", methods=["POST"])
def punctuate():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "text field required"}), 400
    text = data["text"].strip()
    if not text:
        return jsonify({"error": "empty text"}), 400
    try:
        result = llm_manager.punctuate(text)
        return jsonify({"text": result})
    except Exception as e:
        logger.error(f"Punctuate failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "text field required"}), 400
    text = data["text"].strip()
    target_language = data.get("target_language", "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400
    if not target_language:
        return jsonify({"error": "target_language required"}), 400
    try:
        result = llm_manager.translate(text, target_language)
        return jsonify({"text": result, "source": text, "target_language": target_language})
    except Exception as e:
        logger.error(f"Translate failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/transcribe/timestamps/<model_name>", methods=["POST"])
def transcribe_timestamps_with_model(model_name):
    try:
        pipeline = model_manager.load_model(model_name)
        audio_data = request.get_data()
        if not audio_data:
            return jsonify({"error": "No audio data"}), 400
        en_raw_speech, _ = librosa.load(io.BytesIO(audio_data), sr=16000)
        t0 = time.time()
        gen_kwargs = {"return_timestamps": True}
        language = request.args.get("language")
        if language:
            gen_kwargs["language"] = f"<|{language}|>"
        result = pipeline.generate(en_raw_speech, **gen_kwargs)
        elapsed = time.time() - t0
        duration = len(en_raw_speech) / 16000
        chunks = []
        for chunk in result.chunks:
            chunks.append({"text": chunk.text, "start": round(chunk.start_ts, 3), "end": round(chunk.end_ts, 3)})
        logger.info(f"Timestamped {duration:.1f}s audio in {elapsed:.2f}s ({len(chunks)} chunks)")
        metrics.record_transcription(elapsed, duration, _extract_perf_metrics(result))
        return jsonify({"text": str(result), "chunks": chunks, "duration": round(duration, 2), "latency": round(elapsed, 3)})
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        metrics.record_error()
        return jsonify({"error": str(e)}), 500


@app.route("/transcribe/timestamps", methods=["POST"])
def transcribe_timestamps():
    return transcribe_timestamps_with_model(model_manager.default_model)


@app.route("/history/export", methods=["GET"])
def export_history():
    import sqlite3
    import datetime
    fmt = request.args.get("format", "json")
    limit = request.args.get("limit", "50", type=int)
    db_path = os.path.expanduser("~/.config/whisper-npu/history.db")
    if not os.path.exists(db_path):
        return jsonify({"error": "No history database found"}), 404
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT text, timestamp FROM history ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    rows.reverse()
    if fmt == "json":
        items = [{"text": t, "timestamp": ts, "datetime": datetime.datetime.fromtimestamp(ts).isoformat()} for t, ts in rows]
        return jsonify({"items": items, "count": len(items)})
    elif fmt == "markdown":
        lines = ["# Transcription History\n"]
        for text, ts in rows:
            dt = datetime.datetime.fromtimestamp(ts)
            lines.append(f"## {dt.strftime('%H:%M — %Y-%m-%d')}\n\n{text}\n")
        return Response("\n".join(lines), mimetype="text/markdown")
    elif fmt == "srt":
        lines = []
        for i, (text, ts) in enumerate(rows, 1):
            dt = datetime.datetime.fromtimestamp(ts)
            start = dt.strftime("%H:%M:%S,000")
            end_dt = dt + datetime.timedelta(seconds=max(3, len(text) * 0.05))
            end = end_dt.strftime("%H:%M:%S,000")
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")
        return Response("\n".join(lines), mimetype="text/plain")
    else:
        return jsonify({"error": f"Unknown format: {fmt}. Use json, markdown, or srt"}), 400


model_manager = ModelManager()
model_manager.load_model(model_manager.default_model)

llm_manager = LLMManager()
if LLM_MODEL:
    try:
        llm_manager.load_model()
    except Exception as e:
        logger.warning(f"LLM model not loaded: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
