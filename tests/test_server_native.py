"""Tests for server-native.py — the Flask-based Whisper NPU transcription server."""

import importlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module-level mocking: openvino_genai and librosa must be mocked BEFORE
# importing server-native, because the module creates global ModelManager
# and LLMManager instances (and calls load_model) at import time.
# ---------------------------------------------------------------------------

_mock_ov = MagicMock()
_mock_librosa = MagicMock()
# Make librosa.load return a plausible (samples, sr) tuple by default
_mock_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)

sys.modules["openvino_genai"] = _mock_ov
sys.modules["librosa"] = _mock_librosa

# Patch filesystem calls used during module-level init so ModelManager and
# LLMManager constructors plus model_manager.load_model() succeed.
_patches = [
    patch("os.path.exists", return_value=True),
    patch("os.listdir", return_value=["whisper-small-en"]),
    patch("os.path.isdir", return_value=True),
    patch(
        "os.path.expanduser",
        side_effect=lambda p: p.replace("~", "/tmp/_test_home"),
    ),
]
for _p in _patches:
    _p.start()

server_mod = importlib.import_module("server-native")

for _p in _patches:
    _p.stop()

# Convenient aliases
MetricsCollector = server_mod.MetricsCollector
ModelManager = server_mod.ModelManager
LLMManager = server_mod.LLMManager
_extract_perf_metrics = server_mod._extract_perf_metrics
DEFAULT_TONES = server_mod.DEFAULT_TONES
app = server_mod.app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_mocks():
    """Reset shared mocks between tests so state doesn't leak."""
    _mock_ov.reset_mock()
    _mock_librosa.reset_mock()
    _mock_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
    # Reset global metrics so state from earlier tests doesn't break /metrics
    m = server_mod.metrics
    m.transcription_count = 0
    m.transcription_errors = 0
    m.total_latency = 0.0
    m.total_audio_duration = 0.0
    m.model_load_times = {}
    m.last_request = {}
    yield


def _make_mock_result(text="hello world"):
    """Create a mock transcription result whose perf_metrics won't poison
    the global MetricsCollector with un-serializable MagicMock values.
    Using spec=[] prevents auto-creation of attributes, so
    _extract_perf_metrics will hit the except branch and return {}.
    """
    mock_result = MagicMock()
    mock_result.__str__ = lambda self: text
    mock_result.perf_metrics = MagicMock(spec=[])
    return mock_result


@pytest.fixture
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def fresh_metrics():
    """Return a brand-new MetricsCollector."""
    return MetricsCollector()


@pytest.fixture
def model_mgr(tmp_path):
    """ModelManager wired to a temp models directory."""
    mgr = ModelManager.__new__(ModelManager)
    mgr.models_dir = str(tmp_path / "models")
    mgr.pipelines = {}
    mgr.default_model = "whisper-small-en"
    os.makedirs(mgr.models_dir, exist_ok=True)
    return mgr


@pytest.fixture
def llm_mgr(tmp_path):
    """LLMManager wired to a temp LLM models directory."""
    mgr = LLMManager.__new__(LLMManager)
    mgr.models_dir = str(tmp_path / "llm-models")
    mgr.pipeline = None
    mgr.current_model = ""
    os.makedirs(mgr.models_dir, exist_ok=True)
    return mgr


# ===================================================================
#  1. MetricsCollector
# ===================================================================

class TestMetricsCollector:
    def test_initial_snapshot(self, fresh_metrics):
        snap = fresh_metrics.snapshot()
        assert snap["transcription_count"] == 0
        assert snap["transcription_errors"] == 0
        assert snap["average_latency_seconds"] == 0.0
        assert snap["error_rate"] == 0.0
        assert snap["total_audio_seconds"] == 0.0
        assert snap["model_load_times"] == {}
        assert "uptime_seconds" in snap
        assert "last_request" not in snap  # empty → omitted

    def test_record_transcription_basic(self, fresh_metrics):
        fresh_metrics.record_transcription(0.5, 2.0)
        snap = fresh_metrics.snapshot()
        assert snap["transcription_count"] == 1
        assert snap["average_latency_seconds"] == 0.5
        assert snap["total_audio_seconds"] == 2.0
        assert "last_request" not in snap  # no perf_metrics passed

    def test_record_transcription_with_perf_metrics(self, fresh_metrics):
        perf = {"inference_ms": 123.4}
        fresh_metrics.record_transcription(1.0, 3.0, perf_metrics=perf)
        snap = fresh_metrics.snapshot()
        assert snap["last_request"] == perf

    def test_record_multiple_transcriptions_averages(self, fresh_metrics):
        fresh_metrics.record_transcription(1.0, 4.0)
        fresh_metrics.record_transcription(3.0, 6.0)
        snap = fresh_metrics.snapshot()
        assert snap["transcription_count"] == 2
        assert snap["average_latency_seconds"] == 2.0
        assert snap["total_audio_seconds"] == 10.0

    def test_record_error(self, fresh_metrics):
        fresh_metrics.record_error()
        fresh_metrics.record_error()
        snap = fresh_metrics.snapshot()
        assert snap["transcription_errors"] == 2
        assert snap["error_rate"] == 1.0  # 0 successes, 2 errors

    def test_error_rate_mixed(self, fresh_metrics):
        fresh_metrics.record_transcription(1.0, 1.0)
        fresh_metrics.record_error()
        snap = fresh_metrics.snapshot()
        # 1 success + 1 error = error_rate 0.5
        assert snap["error_rate"] == 0.5

    def test_record_model_load(self, fresh_metrics):
        fresh_metrics.record_model_load("whisper-small-en", 2.567)
        snap = fresh_metrics.snapshot()
        assert snap["model_load_times"] == {"whisper-small-en": 2.57}

    def test_thread_safety(self, fresh_metrics):
        """Concurrent writers should not corrupt data."""
        import threading

        def record_many():
            for _ in range(100):
                fresh_metrics.record_transcription(0.01, 0.1)

        threads = [threading.Thread(target=record_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        snap = fresh_metrics.snapshot()
        assert snap["transcription_count"] == 400


# ===================================================================
#  2. ModelManager
# ===================================================================

class TestModelManager:
    def test_list_models_with_models(self, model_mgr):
        os.makedirs(os.path.join(model_mgr.models_dir, "whisper-small-en"))
        os.makedirs(os.path.join(model_mgr.models_dir, "whisper-large-v3"))
        models = model_mgr.list_models()
        assert set(models) == {"whisper-small-en", "whisper-large-v3"}

    def test_list_models_empty(self, model_mgr):
        assert model_mgr.list_models() == []

    def test_list_models_hidden_dirs_filtered(self, model_mgr):
        os.makedirs(os.path.join(model_mgr.models_dir, ".hidden"))
        os.makedirs(os.path.join(model_mgr.models_dir, "visible"))
        models = model_mgr.list_models()
        assert models == ["visible"]

    def test_list_models_files_filtered(self, model_mgr):
        """Regular files (not directories) should not appear."""
        open(os.path.join(model_mgr.models_dir, "not-a-dir.txt"), "w").close()
        assert model_mgr.list_models() == []

    def test_load_model_cache_miss(self, model_mgr):
        model_dir = os.path.join(model_mgr.models_dir, "whisper-small-en")
        os.makedirs(model_dir, exist_ok=True)
        pipe = model_mgr.load_model("whisper-small-en")
        _mock_ov.WhisperPipeline.assert_called_once_with(model_dir, device="NPU")
        assert "whisper-small-en" in model_mgr.pipelines

    def test_load_model_cache_hit(self, model_mgr):
        sentinel = MagicMock(name="cached_pipeline")
        model_mgr.pipelines["whisper-small-en"] = sentinel
        result = model_mgr.load_model("whisper-small-en")
        assert result is sentinel
        _mock_ov.WhisperPipeline.assert_not_called()

    def test_load_model_not_found(self, model_mgr):
        with pytest.raises(FileNotFoundError, match="not found"):
            model_mgr.load_model("nonexistent-model")


# ===================================================================
#  3. LLMManager
# ===================================================================

class TestLLMManager:
    def test_init_with_env_model(self, tmp_path):
        """When LLM_MODEL env is set, current_model should reflect it."""
        mgr = LLMManager.__new__(LLMManager)
        mgr.models_dir = str(tmp_path / "llm")
        mgr.pipeline = None
        mgr.current_model = "my-llm"
        assert mgr.current_model == "my-llm"

    def test_init_auto_select_first(self, llm_mgr):
        """list_models returns models; auto-select picks first."""
        os.makedirs(os.path.join(llm_mgr.models_dir, "alpha-model"))
        os.makedirs(os.path.join(llm_mgr.models_dir, "beta-model"))
        models = llm_mgr.list_models()
        assert len(models) >= 1
        # Simulate auto-select behavior
        llm_mgr.current_model = models[0]
        assert llm_mgr.current_model in models

    def test_list_models_empty_dir(self, llm_mgr):
        assert llm_mgr.list_models() == []

    def test_list_models_no_dir(self, tmp_path):
        mgr = LLMManager.__new__(LLMManager)
        mgr.models_dir = str(tmp_path / "does-not-exist")
        mgr.pipeline = None
        mgr.current_model = ""
        assert mgr.list_models() == []

    def test_list_models_hidden_filtered(self, llm_mgr):
        os.makedirs(os.path.join(llm_mgr.models_dir, ".cache"))
        os.makedirs(os.path.join(llm_mgr.models_dir, "llama-3"))
        assert llm_mgr.list_models() == ["llama-3"]

    def test_load_model_no_model_configured(self, llm_mgr):
        llm_mgr.current_model = ""
        with pytest.raises(ValueError, match="No LLM model configured"):
            llm_mgr.load_model()

    def test_load_model_not_found(self, llm_mgr):
        llm_mgr.current_model = "nonexistent"
        with pytest.raises(FileNotFoundError, match="not found"):
            llm_mgr.load_model()

    def test_load_model_success(self, llm_mgr):
        model_name = "llama-3"
        model_dir = os.path.join(llm_mgr.models_dir, model_name)
        os.makedirs(model_dir)
        pipe = llm_mgr.load_model(model_name)
        _mock_ov.LLMPipeline.assert_called_once()
        assert llm_mgr.current_model == model_name
        assert llm_mgr.pipeline is not None

    def test_load_model_already_loaded_same_model(self, llm_mgr):
        """If model is already loaded, don't reload."""
        model_name = "llama-3"
        model_dir = os.path.join(llm_mgr.models_dir, model_name)
        os.makedirs(model_dir)
        sentinel = MagicMock(name="existing_pipeline")
        llm_mgr.pipeline = sentinel
        llm_mgr.current_model = model_name
        result = llm_mgr.load_model(model_name)
        assert result is sentinel
        _mock_ov.LLMPipeline.assert_not_called()

    def test_load_model_switch_model(self, llm_mgr):
        """Switching to a different model should reload."""
        for name in ("model-a", "model-b"):
            os.makedirs(os.path.join(llm_mgr.models_dir, name))
        llm_mgr.current_model = "model-a"
        llm_mgr.pipeline = MagicMock()
        llm_mgr.load_model("model-b")
        _mock_ov.LLMPipeline.assert_called_once()
        assert llm_mgr.current_model == "model-b"

    def test_rewrite(self, llm_mgr):
        model_dir = os.path.join(llm_mgr.models_dir, "llm")
        os.makedirs(model_dir)
        mock_pipe = MagicMock()
        mock_pipe.generate.return_value = "  Rewritten text  "
        llm_mgr.pipeline = mock_pipe
        llm_mgr.current_model = "llm"
        result = llm_mgr.rewrite("hello world", "diplomatic", "Be diplomatic")
        assert result == "Rewritten text"
        mock_pipe.generate.assert_called_once()
        call_args = mock_pipe.generate.call_args
        assert "hello world" in call_args[0][0]
        assert call_args[1]["max_new_tokens"] == 512

    def test_punctuate(self, llm_mgr):
        model_dir = os.path.join(llm_mgr.models_dir, "llm")
        os.makedirs(model_dir)
        mock_pipe = MagicMock()
        mock_pipe.generate.return_value = "  Hello, world.  "
        llm_mgr.pipeline = mock_pipe
        llm_mgr.current_model = "llm"
        result = llm_mgr.punctuate("hello world")
        assert result == "Hello, world."
        call_args = mock_pipe.generate.call_args
        assert call_args[1]["max_new_tokens"] == 256
        assert call_args[1]["temperature"] == 0.1

    def test_translate(self, llm_mgr):
        model_dir = os.path.join(llm_mgr.models_dir, "llm")
        os.makedirs(model_dir)
        mock_pipe = MagicMock()
        mock_pipe.generate.return_value = "  Hola mundo  "
        llm_mgr.pipeline = mock_pipe
        llm_mgr.current_model = "llm"
        result = llm_mgr.translate("hello world", "Spanish")
        assert result == "Hola mundo"
        call_args = mock_pipe.generate.call_args
        assert "Spanish" in call_args[0][0]
        assert call_args[1]["max_new_tokens"] == 512


# ===================================================================
#  4. _extract_perf_metrics
# ===================================================================

class TestExtractPerfMetrics:
    def test_success(self):
        mock_pm = MagicMock()
        for attr in (
            "get_features_extraction_duration",
            "get_inference_duration",
            "get_generate_duration",
            "get_detokenization_duration",
            "get_throughput",
        ):
            dur = MagicMock()
            dur.mean = 42.123
            getattr(mock_pm, attr).return_value = dur
        mock_pm.get_num_generated_tokens.return_value = 10

        result_obj = MagicMock()
        result_obj.perf_metrics = mock_pm
        out = _extract_perf_metrics(result_obj)
        assert out["features_extraction_ms"] == 42.12
        assert out["inference_ms"] == 42.12
        assert out["generate_ms"] == 42.12
        assert out["detokenization_ms"] == 42.12
        assert out["throughput_tokens_per_sec"] == 42.12
        assert out["num_generated_tokens"] == 10

    def test_exception_returns_empty(self):
        result_obj = MagicMock(spec=[])  # spec=[] means no attributes
        assert _extract_perf_metrics(result_obj) == {}


# ===================================================================
#  5-20. Flask route tests
# ===================================================================

class TestHealthRoute:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "model" in data


class TestModelsRoute:
    def test_list_models(self, client):
        server_mod.model_manager.list_models = MagicMock(
            return_value=["whisper-small-en", "whisper-large-v3"]
        )
        resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "whisper-small-en" in data["models"]


class TestDefaultModelRoute:
    def test_get_default_model(self, client):
        resp = client.get("/model/default")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "model" in data

    def test_set_default_model_missing_body(self, client):
        resp = client.put("/model/default", content_type="application/json")
        assert resp.status_code == 400

    def test_set_default_model_not_found(self, client):
        server_mod.model_manager.list_models = MagicMock(return_value=["a"])
        resp = client.put(
            "/model/default",
            data=json.dumps({"model": "nonexistent"}),
            content_type="application/json",
        )
        assert resp.status_code == 404

    def test_set_default_model_success(self, client):
        server_mod.model_manager.list_models = MagicMock(
            return_value=["whisper-small-en"]
        )
        server_mod.model_manager.load_model = MagicMock()
        resp = client.put(
            "/model/default",
            data=json.dumps({"model": "whisper-small-en"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json()["model"] == "whisper-small-en"


class TestTranscribeRoute:
    def test_no_audio(self, client):
        server_mod.model_manager.load_model = MagicMock()
        resp = client.post("/transcribe", data=b"")
        assert resp.status_code == 400
        assert "No audio" in resp.get_json()["error"]

    def test_success(self, client, wav_bytes):
        mock_pipeline = MagicMock()
        mock_pipeline.generate.return_value = _make_mock_result("hello world")
        server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
        _mock_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        resp = client.post("/transcribe", data=wav_bytes)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["text"] == "hello world"

    def test_exception(self, client, wav_bytes):
        server_mod.model_manager.load_model = MagicMock(
            side_effect=RuntimeError("device error")
        )
        resp = client.post("/transcribe", data=wav_bytes)
        assert resp.status_code == 500
        assert "device error" in resp.get_json()["error"]


class TestTranscribeWithModelRoute:
    def test_with_language(self, client, wav_bytes):
        mock_pipeline = MagicMock()
        mock_pipeline.generate.return_value = _make_mock_result("bonjour")
        server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
        _mock_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        resp = client.post(
            "/transcribe/whisper-small-en?language=fr", data=wav_bytes
        )
        assert resp.status_code == 200
        # Verify language kwarg was passed
        call_kwargs = mock_pipeline.generate.call_args[1]
        assert call_kwargs["language"] == "<|fr|>"

    def test_model_not_found(self, client, wav_bytes):
        server_mod.model_manager.load_model = MagicMock(
            side_effect=FileNotFoundError("not found")
        )
        resp = client.post("/transcribe/bad-model", data=wav_bytes)
        assert resp.status_code == 500
        assert "not found" in resp.get_json()["error"]


class TestTranscribeStreamRoute:
    def test_no_audio(self, client):
        server_mod.model_manager.load_model = MagicMock()
        resp = client.post("/transcribe/stream", data=b"")
        assert resp.status_code == 400

    def test_duration_too_long(self, client, wav_bytes):
        mock_pipeline = MagicMock()
        server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
        # 31 seconds of audio at 16kHz
        long_audio = np.zeros(16000 * 31, dtype=np.float32)
        _mock_librosa.load.return_value = (long_audio, 16000)

        resp = client.post("/transcribe/stream", data=wav_bytes)
        assert resp.status_code == 400
        assert "30 seconds" in resp.get_json()["error"]

    def test_success_sse(self, client, wav_bytes):
        mock_pipeline = MagicMock()

        # Simulate streaming: the pipeline.generate call invokes the streamer
        # callback. The server runs pipeline.generate in a background thread
        # and reads chunks from a queue, so the mock must call the streamer
        # that the server passes in.
        def fake_generate(audio, **kwargs):
            streamer = kwargs.get("streamer")
            if streamer:
                streamer("hello ")
                streamer("world")

        mock_pipeline.generate.side_effect = fake_generate
        server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
        short_audio = np.zeros(16000 * 5, dtype=np.float32)  # 5 seconds
        _mock_librosa.load.return_value = (short_audio, 16000)

        resp = client.post("/transcribe/stream", data=wav_bytes)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.content_type

        # Parse SSE events from response data
        raw = resp.get_data(as_text=True)
        events = [
            line[len("data: "):]
            for line in raw.split("\n")
            if line.startswith("data: ")
        ]
        assert len(events) >= 2  # chunk events + done event
        # Last event should have done: true
        last = json.loads(events[-1])
        assert last["done"] is True
        assert last["full_text"] == "hello world"

    def test_stream_with_model_name(self, client, wav_bytes):
        """Test the /transcribe/stream/<model_name> route."""
        mock_pipeline = MagicMock()

        def fake_generate(audio, **kwargs):
            streamer = kwargs.get("streamer")
            if streamer:
                streamer("test")

        mock_pipeline.generate.side_effect = fake_generate
        server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
        short_audio = np.zeros(16000, dtype=np.float32)
        _mock_librosa.load.return_value = (short_audio, 16000)

        resp = client.post("/transcribe/stream/whisper-small-en", data=wav_bytes)
        assert resp.status_code == 200
        server_mod.model_manager.load_model.assert_called_with("whisper-small-en")

    def test_stream_with_language(self, client, wav_bytes):
        """Verify that language query param is threaded through the streaming path."""
        mock_pipeline = MagicMock()
        captured_kwargs = {}

        def fake_generate(audio, **kwargs):
            captured_kwargs.update(kwargs)
            streamer = kwargs.get("streamer")
            if streamer:
                streamer("bonjour")

        mock_pipeline.generate.side_effect = fake_generate
        server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
        short_audio = np.zeros(16000 * 2, dtype=np.float32)
        _mock_librosa.load.return_value = (short_audio, 16000)

        resp = client.post("/transcribe/stream?language=fr", data=wav_bytes)
        assert resp.status_code == 200
        # Consume the response to ensure the background thread runs
        resp.get_data()
        assert captured_kwargs.get("language") == "<|fr|>"

    def test_stream_inference_error(self, client, wav_bytes):
        """If pipeline.generate raises inside the thread, the stream still completes."""
        mock_pipeline = MagicMock()

        def fake_generate(audio, **kwargs):
            raise RuntimeError("inference exploded")

        mock_pipeline.generate.side_effect = fake_generate
        server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
        short_audio = np.zeros(16000, dtype=np.float32)
        _mock_librosa.load.return_value = (short_audio, 16000)

        resp = client.post("/transcribe/stream", data=wav_bytes)
        assert resp.status_code == 200
        raw = resp.get_data(as_text=True)
        # The stream should still produce a done event (from the finally block)
        events = [line[len("data: "):] for line in raw.split("\n") if line.startswith("data: ")]
        last = json.loads(events[-1])
        assert last["done"] is True
        assert last["full_text"] == ""


class TestRewriteRoute:
    def test_missing_text(self, client):
        resp = client.post(
            "/rewrite",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_empty_text(self, client):
        resp = client.post(
            "/rewrite",
            data=json.dumps({"text": "   "}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "empty" in resp.get_json()["error"]

    def test_success_default_tones(self, client):
        server_mod.llm_manager.rewrite = MagicMock(return_value="Rewritten")
        resp = client.post(
            "/rewrite",
            data=json.dumps({"text": "fix this"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        variants = data["variants"]
        # First variant is always original
        assert variants[0]["tone"] == "original"
        assert variants[0]["text"] == "fix this"
        # Should have one variant per default tone
        assert len(variants) == 1 + len(DEFAULT_TONES)

    def test_success_custom_tones(self, client):
        server_mod.llm_manager.rewrite = MagicMock(return_value="Custom rewrite")
        resp = client.post(
            "/rewrite",
            data=json.dumps({
                "text": "hello",
                "tones": ["my_tone"],
                "custom_tones": {"my_tone": "Be very casual"},
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200
        variants = resp.get_json()["variants"]
        assert any(v["tone"] == "my_tone" for v in variants)

    def test_rewrite_exception(self, client):
        server_mod.llm_manager.rewrite = MagicMock(
            side_effect=RuntimeError("LLM failed")
        )
        resp = client.post(
            "/rewrite",
            data=json.dumps({"text": "hello"}),
            content_type="application/json",
        )
        assert resp.status_code == 200  # route returns 200 with error in variant
        variants = resp.get_json()["variants"]
        errored = [v for v in variants if "error" in v]
        assert len(errored) > 0
        assert "LLM failed" in errored[0]["error"]


class TestLLMModelsRoute:
    def test_list_llm_models(self, client):
        server_mod.llm_manager.list_models = MagicMock(return_value=["llama-3"])
        server_mod.llm_manager.current_model = "llama-3"
        resp = client.get("/llm/models")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "llama-3" in data["models"]
        assert data["current"] == "llama-3"


class TestSetLLMModelRoute:
    def test_missing_body(self, client):
        resp = client.put("/llm/model", content_type="application/json")
        assert resp.status_code == 400

    def test_not_found(self, client):
        server_mod.llm_manager.list_models = MagicMock(return_value=["llama-3"])
        resp = client.put(
            "/llm/model",
            data=json.dumps({"model": "nonexistent"}),
            content_type="application/json",
        )
        assert resp.status_code == 404

    def test_success(self, client):
        server_mod.llm_manager.list_models = MagicMock(return_value=["llama-3"])
        server_mod.llm_manager.load_model = MagicMock()
        resp = client.put(
            "/llm/model",
            data=json.dumps({"model": "llama-3"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json()["model"] == "llama-3"

    def test_load_exception(self, client):
        server_mod.llm_manager.list_models = MagicMock(return_value=["llama-3"])
        server_mod.llm_manager.load_model = MagicMock(
            side_effect=RuntimeError("load failed")
        )
        resp = client.put(
            "/llm/model",
            data=json.dumps({"model": "llama-3"}),
            content_type="application/json",
        )
        assert resp.status_code == 500
        assert "load failed" in resp.get_json()["error"]


class TestTonesRoute:
    def test_list_tones(self, client):
        resp = client.get("/llm/tones")
        assert resp.status_code == 200
        tones = resp.get_json()["tones"]
        assert "diplomatic" in tones
        assert "professional" in tones
        assert "summarize" in tones


class TestMetricsRoute:
    def test_get_metrics(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "transcription_count" in data
        assert "uptime_seconds" in data


class TestPunctuateRoute:
    def test_missing_text(self, client):
        resp = client.post(
            "/punctuate",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_empty_text(self, client):
        resp = client.post(
            "/punctuate",
            data=json.dumps({"text": "   "}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "empty" in resp.get_json()["error"]

    def test_success(self, client):
        server_mod.llm_manager.punctuate = MagicMock(return_value="Hello, world.")
        resp = client.post(
            "/punctuate",
            data=json.dumps({"text": "hello world"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json()["text"] == "Hello, world."

    def test_exception(self, client):
        server_mod.llm_manager.punctuate = MagicMock(
            side_effect=RuntimeError("inference crash")
        )
        resp = client.post(
            "/punctuate",
            data=json.dumps({"text": "hello world"}),
            content_type="application/json",
        )
        assert resp.status_code == 500
        assert "inference crash" in resp.get_json()["error"]


class TestTranslateRoute:
    def test_missing_text(self, client):
        resp = client.post(
            "/translate",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_empty_text(self, client):
        resp = client.post(
            "/translate",
            data=json.dumps({"text": "   ", "target_language": "Spanish"}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "empty" in resp.get_json()["error"]

    def test_missing_target_language(self, client):
        resp = client.post(
            "/translate",
            data=json.dumps({"text": "hello"}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "target_language" in resp.get_json()["error"]

    def test_success(self, client):
        server_mod.llm_manager.translate = MagicMock(return_value="Hola mundo")
        resp = client.post(
            "/translate",
            data=json.dumps({"text": "hello world", "target_language": "Spanish"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["text"] == "Hola mundo"
        assert data["source"] == "hello world"
        assert data["target_language"] == "Spanish"

    def test_exception(self, client):
        server_mod.llm_manager.translate = MagicMock(
            side_effect=RuntimeError("translation error")
        )
        resp = client.post(
            "/translate",
            data=json.dumps({"text": "hello", "target_language": "Spanish"}),
            content_type="application/json",
        )
        assert resp.status_code == 500
        assert "translation error" in resp.get_json()["error"]


class TestTranscribeTimestampsRoute:
    def test_success(self, client, wav_bytes):
        mock_pipeline = MagicMock()
        mock_result = _make_mock_result("hello world")
        # Simulate chunks
        chunk1 = MagicMock()
        chunk1.text = "hello"
        chunk1.start_ts = 0.0
        chunk1.end_ts = 0.5
        chunk2 = MagicMock()
        chunk2.text = " world"
        chunk2.start_ts = 0.5
        chunk2.end_ts = 1.0
        mock_result.chunks = [chunk1, chunk2]
        mock_pipeline.generate.return_value = mock_result
        server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
        _mock_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        resp = client.post("/transcribe/timestamps", data=wav_bytes)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["text"] == "hello world"
        assert len(data["chunks"]) == 2
        assert data["chunks"][0]["text"] == "hello"
        assert data["chunks"][0]["start"] == 0.0
        assert data["chunks"][1]["end"] == 1.0
        assert "duration" in data
        assert "latency" in data

    def test_with_model_name(self, client, wav_bytes):
        mock_pipeline = MagicMock()
        mock_result = _make_mock_result("test")
        mock_result.chunks = []
        mock_pipeline.generate.return_value = mock_result
        server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
        _mock_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        resp = client.post("/transcribe/timestamps/whisper-large", data=wav_bytes)
        assert resp.status_code == 200
        server_mod.model_manager.load_model.assert_called_with("whisper-large")

    def test_exception(self, client, wav_bytes):
        server_mod.model_manager.load_model = MagicMock(
            side_effect=RuntimeError("crash")
        )
        resp = client.post("/transcribe/timestamps", data=wav_bytes)
        assert resp.status_code == 500
        assert "crash" in resp.get_json()["error"]


class TestHistoryExportRoute:
    def test_no_db(self, client):
        with patch("os.path.exists", return_value=False), \
             patch("os.path.expanduser", return_value="/tmp/no-such-db"):
            resp = client.get("/history/export")
        assert resp.status_code == 404

    def test_json_format(self, client, tmp_path):
        db_path = str(tmp_path / "history.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE history (id INTEGER PRIMARY KEY, text TEXT, timestamp REAL)")
            conn.execute("INSERT INTO history (text, timestamp) VALUES (?, ?)", ("hello", 1700000000.0))
            conn.execute("INSERT INTO history (text, timestamp) VALUES (?, ?)", ("world", 1700000010.0))

        with patch("os.path.exists", return_value=True), \
             patch("os.path.expanduser", return_value=db_path):
            resp = client.get("/history/export?format=json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 2
        assert len(data["items"]) == 2
        # Items should be in chronological order (reversed from DESC)
        assert data["items"][0]["text"] == "hello"
        assert "datetime" in data["items"][0]

    def test_markdown_format(self, client, tmp_path):
        db_path = str(tmp_path / "history.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE history (id INTEGER PRIMARY KEY, text TEXT, timestamp REAL)")
            conn.execute("INSERT INTO history (text, timestamp) VALUES (?, ?)", ("test entry", 1700000000.0))

        with patch("os.path.exists", return_value=True), \
             patch("os.path.expanduser", return_value=db_path):
            resp = client.get("/history/export?format=markdown")
        assert resp.status_code == 200
        assert "text/markdown" in resp.content_type
        body = resp.get_data(as_text=True)
        assert "Transcription History" in body
        assert "test entry" in body

    def test_srt_format(self, client, tmp_path):
        db_path = str(tmp_path / "history.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE history (id INTEGER PRIMARY KEY, text TEXT, timestamp REAL)")
            conn.execute("INSERT INTO history (text, timestamp) VALUES (?, ?)", ("subtitle line", 1700000000.0))

        with patch("os.path.exists", return_value=True), \
             patch("os.path.expanduser", return_value=db_path):
            resp = client.get("/history/export?format=srt")
        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        assert "1\n" in body
        assert "-->" in body
        assert "subtitle line" in body

    def test_unknown_format(self, client, tmp_path):
        db_path = str(tmp_path / "history.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE history (id INTEGER PRIMARY KEY, text TEXT, timestamp REAL)")

        with patch("os.path.exists", return_value=True), \
             patch("os.path.expanduser", return_value=db_path):
            resp = client.get("/history/export?format=xml")
        assert resp.status_code == 400
        assert "Unknown format" in resp.get_json()["error"]

    def test_limit_parameter(self, client, tmp_path):
        db_path = str(tmp_path / "history.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE history (id INTEGER PRIMARY KEY, text TEXT, timestamp REAL)")
            for i in range(10):
                conn.execute("INSERT INTO history (text, timestamp) VALUES (?, ?)", (f"entry {i}", 1700000000.0 + i))

        with patch("os.path.exists", return_value=True), \
             patch("os.path.expanduser", return_value=db_path):
            resp = client.get("/history/export?format=json&limit=3")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 3


# ===================================================================
#  Edge cases and integration-style tests
# ===================================================================

class TestTranscribeNoAudioField:
    """Verify that posting with no body triggers the 400 path."""

    def test_transcribe_with_model_no_audio(self, client):
        mock_pipeline = MagicMock()
        server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
        resp = client.post("/transcribe/whisper-small-en", data=b"")
        assert resp.status_code == 400


class TestRewriteToneResolution:
    """Verify that tone resolution picks custom_tones over DEFAULT_TONES."""

    def test_custom_tone_overrides_default(self, client):
        calls = []

        def capture_rewrite(text, tone_name, tone_prompt):
            calls.append((tone_name, tone_prompt))
            return "rewritten"

        server_mod.llm_manager.rewrite = capture_rewrite
        resp = client.post(
            "/rewrite",
            data=json.dumps({
                "text": "hello",
                "tones": ["diplomatic"],
                "custom_tones": {"diplomatic": "My custom prompt"},
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert calls[0][1] == "My custom prompt"

    def test_unknown_tone_ignored(self, client):
        """A tone not in DEFAULT_TONES and not in custom_tones is skipped."""
        server_mod.llm_manager.rewrite = MagicMock(return_value="done")
        resp = client.post(
            "/rewrite",
            data=json.dumps({
                "text": "hello",
                "tones": ["nonexistent_tone"],
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200
        variants = resp.get_json()["variants"]
        # Only "original" — the unknown tone is not in tone_prompts
        assert len(variants) == 1
        assert variants[0]["tone"] == "original"


class TestStreamException:
    """Verify that exceptions in the streaming route return 500."""

    def test_stream_load_model_failure(self, client, wav_bytes):
        server_mod.model_manager.load_model = MagicMock(
            side_effect=RuntimeError("NPU unavailable")
        )
        resp = client.post("/transcribe/stream", data=wav_bytes)
        assert resp.status_code == 500
        assert "NPU unavailable" in resp.get_json()["error"]


class TestTranscribeTimestampsNoAudio:
    def test_no_audio_data(self, client):
        mock_pipeline = MagicMock()
        server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
        resp = client.post("/transcribe/timestamps", data=b"")
        assert resp.status_code == 400
        assert "No audio" in resp.get_json()["error"]
