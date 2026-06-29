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

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = os.environ.get("WHISPER_DEVICE", "NPU")
MODEL_NAME = os.environ.get("WHISPER_MODEL", "whisper-small.en-fp16-ov")

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
            logger.info(f"Model loaded in {time.time()-t0:.1f}s")
        return self.pipelines[model_name]

    def list_models(self):
        return [d for d in os.listdir(self.models_dir)
                if os.path.isdir(os.path.join(self.models_dir, d)) and not d.startswith('.')]

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": model_manager.default_model})

@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({"models": model_manager.list_models()})

@app.route("/transcribe/<model_name>", methods=["POST"])
def transcribe_with_model(model_name):
    try:
        pipeline = model_manager.load_model(model_name)
        audio_data = request.get_data()
        if not audio_data:
            return jsonify({"error": "No audio data"}), 400
        en_raw_speech, _ = librosa.load(io.BytesIO(audio_data), sr=16000)
        t0 = time.time()
        result = pipeline.generate(en_raw_speech)
        elapsed = time.time() - t0
        duration = len(en_raw_speech) / 16000
        logger.info(f"Transcribed {duration:.1f}s audio in {elapsed:.2f}s ({duration/elapsed:.1f}x realtime)")
        return jsonify({"text": str(result)})
    except Exception as e:
        logger.error(f"Error: {str(e)}")
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

        def streamer_callback(text_chunk):
            q.put(text_chunk)
            return 0

        def generate():
            t0 = time.time()
            def run_inference():
                try:
                    pipeline.generate(
                        en_raw_speech,
                        streamer=streamer_callback,
                        return_timestamps=False,
                    )
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
            yield f"data: {json.dumps({'done': True, 'full_text': ''.join(full_text)})}\n\n"

        return Response(generate(), mimetype="text/event-stream")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

model_manager = ModelManager()
model_manager.load_model(model_manager.default_model)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
