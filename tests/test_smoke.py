"""Smoke tests — lightweight assembly checks that verify components load and
respond without requiring real models, hardware, or external services."""

import importlib
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# server-native.py smoke tests
# ---------------------------------------------------------------------------

class TestServerSmoke:
    """Verify the Flask app assembles and its routes respond."""

    @pytest.fixture(autouse=True)
    def _setup_server(self):
        mock_ov = MagicMock()
        mock_librosa = MagicMock()

        saved = {}
        for mod in ("openvino_genai", "librosa"):
            saved[mod] = sys.modules.get(mod)
        sys.modules["openvino_genai"] = mock_ov
        sys.modules["librosa"] = mock_librosa

        with patch("os.path.exists", return_value=True), \
             patch("os.listdir", return_value=["whisper-small-en"]), \
             patch("os.path.isdir", return_value=True), \
             patch("os.path.expanduser", side_effect=lambda p: p.replace("~", "/tmp/smoke")):
            self.mod = importlib.import_module("server-native")
            importlib.reload(self.mod)

        self.client = self.mod.app.test_client()
        yield

        for mod, orig in saved.items():
            if orig is None:
                sys.modules.pop(mod, None)
            else:
                sys.modules[mod] = orig

    def test_app_is_flask(self):
        from flask import Flask
        assert isinstance(self.mod.app, Flask)

    def test_health_returns_200(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "ok"
        assert "model" in data

    def test_models_returns_200(self):
        with patch("os.listdir", return_value=["whisper-small-en"]), \
             patch("os.path.isdir", return_value=True):
            resp = self.client.get("/models")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "models" in data

    def test_llm_tones_returns_200(self):
        resp = self.client.get("/llm/tones")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "tones" in data
        assert isinstance(data["tones"], list)

    def test_llm_models_returns_200(self):
        resp = self.client.get("/llm/models")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "models" in data
        assert "current" in data

    def test_metrics_returns_200(self):
        resp = self.client.get("/metrics")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "transcription_count" in data
        assert "uptime_seconds" in data

    def test_model_default_get(self):
        resp = self.client.get("/model/default")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "model" in data

    def test_all_expected_routes_registered(self):
        rules = [r.rule for r in self.mod.app.url_map.iter_rules()]
        expected = [
            "/health", "/models", "/model/default",
            "/transcribe", "/transcribe/<model_name>",
            "/transcribe/stream", "/transcribe/stream/<model_name>",
            "/rewrite", "/llm/models", "/llm/model", "/llm/tones",
            "/metrics", "/punctuate", "/translate",
            "/transcribe/timestamps", "/transcribe/timestamps/<model_name>",
            "/history/export",
        ]
        for route in expected:
            assert route in rules, f"Route {route} not registered"

    def test_transcribe_rejects_empty_body(self):
        resp = self.client.post("/transcribe", data=b"")
        assert resp.status_code == 400

    def test_rewrite_rejects_missing_text(self):
        resp = self.client.post("/rewrite",
                                data=json.dumps({}),
                                content_type="application/json")
        assert resp.status_code == 400

    def test_punctuate_rejects_empty_text(self):
        resp = self.client.post("/punctuate",
                                data=json.dumps({"text": "  "}),
                                content_type="application/json")
        assert resp.status_code == 400

    def test_translate_requires_target_language(self):
        resp = self.client.post("/translate",
                                data=json.dumps({"text": "hello"}),
                                content_type="application/json")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# push-to-talk.py smoke tests
# ---------------------------------------------------------------------------

class TestPushToTalkSmoke:
    """Verify the push-to-talk module imports and key components work."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        spec = importlib.util.spec_from_file_location(
            "push_to_talk",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "push-to-talk.py"),
            submodule_search_locations=[]
        )
        self.mod = importlib.util.module_from_spec(spec)
        sys.modules["push_to_talk"] = self.mod

        mock_evdev = MagicMock()
        mock_evdev.list_devices.return_value = []
        with patch.dict("sys.modules", {"evdev": mock_evdev}):
            spec.loader.exec_module(self.mod)

        yield
        sys.modules.pop("push_to_talk", None)

    def test_module_imports(self):
        assert hasattr(self.mod, "trim_silence")
        assert hasattr(self.mod, "format_dictation")
        assert hasattr(self.mod, "type_text")

    def test_constants_are_sane(self):
        assert self.mod.SAMPLE_RATE == 16000
        assert self.mod.CHANNELS == 1
        assert self.mod.SAMPLE_WIDTH == 2
        assert len(self.mod.VOICE_COMMANDS) > 10
        assert len(self.mod.DBUS_BUS_NAMES) == 2

    def test_dbus_bus_names(self):
        assert "org.gnome.Shell" in self.mod.DBUS_BUS_NAMES
        assert "com.whisper.LanguageBuddy" in self.mod.DBUS_BUS_NAMES

    def test_trim_silence_empty(self):
        assert self.mod.trim_silence(b"") == b""

    def test_format_dictation_passthrough(self):
        assert self.mod.format_dictation("Hello world") == "Hello world"

    def test_format_dictation_capitalizes(self):
        assert self.mod.format_dictation("hello")[0] == "H"

    def test_type_text_empty_is_noop(self):
        with patch("subprocess.run") as m:
            self.mod.type_text("")
            m.assert_not_called()

    def test_try_dbus_handoff_no_bus(self):
        with patch.object(self.mod, "_try_dbus_call", return_value=None):
            assert self.mod.try_dbus_handoff("test") is False

    def test_notify_graceful_failure(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            self.mod.notify("test")

    def test_play_sound_missing_file(self):
        with patch("os.path.exists", return_value=False):
            self.mod.play_sound("nonexistent")

    def test_load_app_contexts_returns_defaults(self):
        with patch("os.path.exists", return_value=False):
            ctx = self.mod.load_app_contexts()
            assert isinstance(ctx, dict)
            assert len(ctx) > 0

    def test_voice_commands_have_actions(self):
        for cmd, actions in self.mod.VOICE_COMMANDS.items():
            assert len(actions) > 0, f"Command '{cmd}' has no actions"
            for action_type, _ in actions:
                assert action_type in ("key", "type"), \
                    f"Command '{cmd}' has invalid action type '{action_type}'"


# ---------------------------------------------------------------------------
# KDE service smoke tests
# ---------------------------------------------------------------------------

class TestKdeServiceSmoke:
    """Verify the KDE D-Bus service module loads and key functions work."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        for mod_name in ["dbus", "dbus.service", "dbus.mainloop", "dbus.mainloop.glib"]:
            if mod_name not in sys.modules:
                sys.modules[mod_name] = MagicMock()

        mock_gi = MagicMock()
        for mod_name in ["gi", "gi.repository"]:
            if mod_name not in sys.modules:
                sys.modules[mod_name] = mock_gi

        spec = importlib.util.spec_from_file_location(
            "whisper_npu_kde_service",
            os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "kde-plasmoid", "whisper-npu-kde-service.py"),
            submodule_search_locations=[]
        )
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        yield

    def test_module_imports(self):
        assert hasattr(self.mod, "load_config")
        assert hasattr(self.mod, "server_url")
        assert hasattr(self.mod, "http_post")
        assert hasattr(self.mod, "type_text")
        assert hasattr(self.mod, "backspace_n")

    def test_load_config_missing_file(self):
        result = self.mod.load_config()
        assert isinstance(result, dict)

    def test_server_url_default(self):
        url = self.mod.server_url("/health", {"server-host": "localhost", "server-port": 8080})
        assert url == "http://localhost:8080/health"

    def test_server_url_default_config(self):
        url = self.mod.server_url("/test")
        assert url.startswith("http://")
        assert "/test" in url

    def test_default_tones_constant(self):
        assert self.mod.DEFAULT_TONES == ["diplomatic", "professional"]

    def test_config_path_exists(self):
        assert isinstance(self.mod.CONFIG_PATH, str)
        assert "whisper-npu" in self.mod.CONFIG_PATH

    def test_type_text_calls_ydotool(self):
        with patch("subprocess.run") as m:
            self.mod.type_text("hello")
            m.assert_called_once()
            args = m.call_args[0][0]
            assert args[0] == "ydotool"

    def test_backspace_n_zero(self):
        with patch("subprocess.run") as m:
            self.mod.backspace_n(0)
            m.assert_not_called()

    def test_http_post_handles_error(self):
        result = self.mod.http_post("http://127.0.0.1:1/bad", {"x": 1})
        assert result is None
