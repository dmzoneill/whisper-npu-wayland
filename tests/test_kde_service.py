"""Tests for the KDE D-Bus overlay service (whisper-npu-kde-service.py).

Mocks dbus before importing the module so tests work without a session bus.
Uses real PyQt6 for widget tests (QApplication created once per session).
"""

import importlib
import importlib.util
import json
import os
import sys
import time
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock dbus before importing the service module
# ---------------------------------------------------------------------------

# Save originals if present, then replace with mocks
_real_dbus = sys.modules.pop("dbus", None)
_real_dbus_service = sys.modules.pop("dbus.service", None)
_real_dbus_mainloop = sys.modules.pop("dbus.mainloop", None)
_real_dbus_mainloop_glib = sys.modules.pop("dbus.mainloop.glib", None)

mock_dbus = MagicMock()
# dbus.service.Object must be a real class so LanguageBuddyService can inherit
mock_dbus_service_Object = type("Object", (), {
    "__init__": lambda self, *a, **kw: None,
})
mock_dbus.service = MagicMock()
mock_dbus.service.Object = mock_dbus_service_Object
mock_dbus.service.method = lambda *a, **kw: lambda f: f
mock_dbus.service.BusName = MagicMock
mock_dbus.SessionBus = MagicMock

sys.modules["dbus"] = mock_dbus
sys.modules["dbus.service"] = mock_dbus.service
sys.modules["dbus.mainloop"] = MagicMock()
sys.modules["dbus.mainloop.glib"] = MagicMock()

# Mock GLib (gi.repository.GLib) to avoid needing gobject-introspection in tests
_real_gi = sys.modules.pop("gi", None)
_real_gi_repo = sys.modules.pop("gi.repository", None)
mock_gi = MagicMock()
sys.modules["gi"] = mock_gi
sys.modules["gi.repository"] = mock_gi.repository

# ---------------------------------------------------------------------------
# Import the module under test (filename has hyphens, use importlib)
# ---------------------------------------------------------------------------

_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "kde-plasmoid", "whisper-npu-kde-service.py"
)

spec = importlib.util.spec_from_file_location("whisper_npu_kde_service", _MODULE_PATH)
kde_svc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kde_svc)

# ---------------------------------------------------------------------------
# QApplication fixture (session-scoped, created once)
# ---------------------------------------------------------------------------

from PyQt6.QtWidgets import QApplication  # noqa: E402

@pytest.fixture(scope="session")
def qapp():
    """Create a single QApplication for the entire test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# =========================================================================
# 1. load_config
# =========================================================================

class TestLoadConfig:

    def test_valid_json(self, tmp_path):
        cfg = {"server-host": "10.0.0.1", "server-port": 9999}
        cfg_file = tmp_path / "settings.json"
        cfg_file.write_text(json.dumps(cfg))
        with patch.object(kde_svc, "CONFIG_PATH", str(cfg_file)):
            result = kde_svc.load_config()
        assert result == cfg

    def test_file_missing(self, tmp_path):
        with patch.object(kde_svc, "CONFIG_PATH", str(tmp_path / "nope.json")):
            result = kde_svc.load_config()
        assert result == {}

    def test_invalid_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json!!!")
        with patch.object(kde_svc, "CONFIG_PATH", str(bad)):
            result = kde_svc.load_config()
        assert result == {}

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.json"
        empty.write_text("")
        with patch.object(kde_svc, "CONFIG_PATH", str(empty)):
            result = kde_svc.load_config()
        assert result == {}

    def test_valid_nested_config(self, tmp_path):
        cfg = {
            "server-host": "192.168.1.100",
            "server-port": 8080,
            "language-buddy-enabled": True,
            "language-buddy-bypass": True,
            "language-buddy-timeout": 60,
        }
        cfg_file = tmp_path / "settings.json"
        cfg_file.write_text(json.dumps(cfg))
        with patch.object(kde_svc, "CONFIG_PATH", str(cfg_file)):
            result = kde_svc.load_config()
        assert result["language-buddy-enabled"] is True
        assert result["language-buddy-timeout"] == 60


# =========================================================================
# 2. server_url
# =========================================================================

class TestServerUrl:

    def test_default_config(self):
        url = kde_svc.server_url("/rewrite", config={})
        assert url == "http://127.0.0.1:5000/rewrite"

    def test_custom_config(self):
        cfg = {"server-host": "10.0.0.5", "server-port": 8888}
        url = kde_svc.server_url("/api/v1", config=cfg)
        assert url == "http://10.0.0.5:8888/api/v1"

    def test_none_config_calls_load_config(self):
        with patch.object(kde_svc, "load_config", return_value={"server-host": "host1", "server-port": 1234}):
            url = kde_svc.server_url("/test")
        assert url == "http://host1:1234/test"

    def test_empty_path(self):
        url = kde_svc.server_url("", config={})
        assert url == "http://127.0.0.1:5000"

    def test_path_with_query_string(self):
        url = kde_svc.server_url("/search?q=hello", config={"server-host": "localhost", "server-port": 3000})
        assert url == "http://localhost:3000/search?q=hello"

    def test_partial_config_uses_defaults(self):
        # Only host specified, port should default
        url = kde_svc.server_url("/x", config={"server-host": "myhost"})
        assert url == "http://myhost:5000/x"

        # Only port specified, host should default
        url = kde_svc.server_url("/x", config={"server-port": 7777})
        assert url == "http://127.0.0.1:7777/x"


# =========================================================================
# 3. http_post
# =========================================================================

class TestHttpPost:

    def test_success(self):
        response_data = {"result": "ok"}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as m:
            result = kde_svc.http_post("http://localhost:5000/test", {"key": "val"})
        assert result == response_data
        m.assert_called_once()

    def test_http_error(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
                "http://x", 500, "Server Error", {}, None)):
            result = kde_svc.http_post("http://x", {})
        assert result is None

    def test_timeout(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timed out")):
            result = kde_svc.http_post("http://x", {})
        assert result is None

    def test_json_decode_error(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = kde_svc.http_post("http://x", {})
        assert result is None

    def test_connection_refused(self):
        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("refused")):
            result = kde_svc.http_post("http://x", {})
        assert result is None


# =========================================================================
# 4. type_text
# =========================================================================

class TestTypeText:

    def test_success(self):
        with patch.object(kde_svc, "subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0)
            mock_sub.DEVNULL = -1
            kde_svc.type_text("hello world", delay_ms=4)
        mock_sub.run.assert_called_once_with(
            ["ydotool", "type", "-d", "4", "--", "hello world"],
            check=True, stdout=-1, stderr=-1
        )

    def test_custom_delay(self):
        with patch.object(kde_svc, "subprocess") as mock_sub:
            mock_sub.DEVNULL = -1
            kde_svc.type_text("test", delay_ms=10)
        args = mock_sub.run.call_args[0][0]
        assert args[3] == "10"

    def test_ydotool_missing(self):
        with patch.object(kde_svc, "subprocess") as mock_sub:
            mock_sub.run.side_effect = FileNotFoundError("ydotool not found")
            mock_sub.DEVNULL = -1
            # Should not raise, just log
            kde_svc.type_text("hello")

    def test_subprocess_error(self):
        import subprocess as real_subprocess
        with patch.object(kde_svc, "subprocess") as mock_sub:
            mock_sub.run.side_effect = real_subprocess.CalledProcessError(1, "ydotool")
            mock_sub.DEVNULL = -1
            # Should not raise
            kde_svc.type_text("test")


# =========================================================================
# 5. backspace_n
# =========================================================================

class TestBackspaceN:

    def test_zero(self):
        with patch.object(kde_svc, "subprocess") as mock_sub:
            mock_sub.DEVNULL = -1
            kde_svc.backspace_n(0)
        mock_sub.run.assert_not_called()

    def test_three(self):
        with patch.object(kde_svc, "subprocess") as mock_sub:
            mock_sub.DEVNULL = -1
            kde_svc.backspace_n(3)
        assert mock_sub.run.call_count == 3
        for c in mock_sub.run.call_args_list:
            assert c[0][0] == ["ydotool", "key", "14:1", "14:0"]

    def test_subprocess_failure_continues(self):
        with patch.object(kde_svc, "subprocess") as mock_sub:
            mock_sub.DEVNULL = -1
            mock_sub.run.side_effect = OSError("fail")
            # Should not raise, failures are silently caught
            kde_svc.backspace_n(3)
        assert mock_sub.run.call_count == 3

    def test_one(self):
        with patch.object(kde_svc, "subprocess") as mock_sub:
            mock_sub.DEVNULL = -1
            kde_svc.backspace_n(1)
        assert mock_sub.run.call_count == 1


# =========================================================================
# 6. OverlayCard
# =========================================================================

class TestOverlayCard:

    def test_init_ready(self, qapp):
        card = kde_svc.OverlayCard("diplomatic", "Hello world", ready=True)
        assert card.tone == "diplomatic"
        assert card.card_text == "Hello world"
        assert card.ready is True
        assert card.text_label.text() == "Hello world"
        assert card.tone_label.text() == "Diplomatic"

    def test_init_not_ready(self, qapp):
        card = kde_svc.OverlayCard("casual", "Processing...", ready=False)
        assert card.ready is False
        assert card.tone_label.text() == "Casual"

    def test_update_text(self, qapp):
        card = kde_svc.OverlayCard("formal", "Loading...", ready=False)
        assert card.ready is False
        card.update_text("This is the rewritten text.")
        assert card.ready is True
        assert card.card_text == "This is the rewritten text."
        assert card.text_label.text() == "This is the rewritten text."

    def test_update_style_called_on_init(self, qapp):
        card = kde_svc.OverlayCard("test", "text", ready=False)
        style = card.styleSheet()
        assert "QPushButton" in style
        assert "border-radius" in style

    def test_tone_capitalization(self, qapp):
        card = kde_svc.OverlayCard("summarize", "text")
        assert card.tone_label.text() == "Summarize"

    def test_enter_leave_events(self, qapp):
        """enterEvent / leaveEvent just call super, ensure no crash."""
        card = kde_svc.OverlayCard("test", "text", ready=True)
        # Qt6 expects QEnterEvent|None and QEvent|None — pass None
        card.enterEvent(None)
        card.leaveEvent(None)


# =========================================================================
# 7. LanguageBuddyOverlay
# =========================================================================

class TestLanguageBuddyOverlay:

    def test_show_overlay_creates_cards(self, qapp):
        overlay = kde_svc.LanguageBuddyOverlay()
        tones = ["diplomatic", "professional"]
        overlay.show_overlay(tones, "Hello world", timeout_sec=30)

        assert "original" in overlay._cards
        assert "diplomatic" in overlay._cards
        assert "professional" in overlay._cards
        assert len(overlay._cards) == 3  # original + 2 tones

        # Original card should be ready
        assert overlay._cards["original"].ready is True
        assert overlay._cards["original"].card_text == "Hello world"

        # Tone cards should not be ready
        assert overlay._cards["diplomatic"].ready is False
        assert overlay._cards["professional"].ready is False

        overlay.dismiss()

    def test_show_overlay_with_timeout(self, qapp):
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["casual"], "Test", timeout_sec=10)
        assert overlay._timer.isActive()
        overlay.dismiss()

    def test_show_overlay_zero_timeout(self, qapp):
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["casual"], "Test", timeout_sec=0)
        assert not overlay._timer.isActive()
        overlay.dismiss()

    def test_update_card_existing(self, qapp):
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["diplomatic"], "Original text")
        overlay.update_card("diplomatic", "Rewritten text")

        card = overlay._cards["diplomatic"]
        assert card.ready is True
        assert card.card_text == "Rewritten text"
        overlay.dismiss()

    def test_update_card_missing_tone(self, qapp):
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["diplomatic"], "text")
        # Should not crash
        overlay.update_card("nonexistent", "text")
        overlay.dismiss()

    def test_on_card_click_ready(self, qapp):
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["diplomatic"], "original")
        overlay.update_card("diplomatic", "rewritten")

        selected = []
        overlay.card_selected.connect(lambda t: selected.append(t))
        overlay._on_card_click("diplomatic")

        assert len(selected) == 1
        assert selected[0] == "rewritten"

    def test_on_card_click_not_ready(self, qapp):
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["diplomatic"], "original")
        # Card is still "Processing...", not ready

        selected = []
        overlay.card_selected.connect(lambda t: selected.append(t))
        overlay._on_card_click("diplomatic")

        assert len(selected) == 0
        overlay.dismiss()

    def test_on_card_click_missing_tone(self, qapp):
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["diplomatic"], "original")

        selected = []
        overlay.card_selected.connect(lambda t: selected.append(t))
        overlay._on_card_click("nonexistent")

        assert len(selected) == 0
        overlay.dismiss()

    def test_on_select_emits_signal(self, qapp):
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["diplomatic"], "original")

        selected = []
        overlay.card_selected.connect(lambda t: selected.append(t))
        overlay._on_select("chosen text")

        assert selected == ["chosen text"]

    def test_dismiss(self, qapp):
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["diplomatic"], "text", timeout_sec=30)
        assert overlay._timer.isActive()

        overlay.dismiss()
        assert not overlay._timer.isActive()
        assert overlay.isHidden()
        assert len(overlay._cards) == 0

    def test_clear_removes_cards(self, qapp):
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["a", "b"], "text")
        assert len(overlay._cards) == 3  # original + a + b
        overlay._clear()
        assert len(overlay._cards) == 0

    def test_position_with_screen(self, qapp):
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["test"], "text")
        # Just checking it doesn't crash with a real screen
        overlay._position()
        overlay.dismiss()

    def test_position_without_screen(self, qapp):
        overlay = kde_svc.LanguageBuddyOverlay()
        with patch.object(kde_svc.QGuiApplication, "primaryScreen", return_value=None):
            overlay._position()  # Should return early, no crash

    def test_show_overlay_replaces_previous(self, qapp):
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["tone1"], "first")
        assert "tone1" in overlay._cards

        overlay.show_overlay(["tone2"], "second")
        assert "tone2" in overlay._cards
        assert "tone1" not in overlay._cards
        overlay.dismiss()


# =========================================================================
# 8. HistoryOverlay
# =========================================================================

class TestHistoryOverlay:

    def test_show_history(self, qapp):
        overlay = kde_svc.HistoryOverlay()
        items = [
            {"text": "Hello world", "ts": time.time() - 30},
            {"text": "Second item", "ts": time.time() - 3600},
        ]
        overlay.show_history(items, timeout_sec=10)
        assert overlay._timer.isActive()
        overlay.dismiss()

    def test_show_history_long_text_truncated(self, qapp):
        overlay = kde_svc.HistoryOverlay()
        long_text = "x" * 200
        items = [{"text": long_text, "ts": time.time()}]
        overlay.show_history(items)
        # The display text in the card should be truncated, but the full_text
        # bound to the click handler should still be the full text
        overlay.dismiss()

    def test_show_history_zero_timeout(self, qapp):
        overlay = kde_svc.HistoryOverlay()
        items = [{"text": "test", "ts": time.time()}]
        overlay.show_history(items, timeout_sec=0)
        assert not overlay._timer.isActive()
        overlay.dismiss()

    def test_dismiss(self, qapp):
        overlay = kde_svc.HistoryOverlay()
        items = [{"text": "test", "ts": time.time()}]
        overlay.show_history(items, timeout_sec=10)
        overlay.dismiss()
        assert not overlay._timer.isActive()
        assert overlay.isHidden()

    def test_on_select_types_text(self, qapp):
        overlay = kde_svc.HistoryOverlay()
        with patch.object(kde_svc, "type_text") as mock_type:
            overlay._on_select("typed text")
        mock_type.assert_called_once_with("typed text")

    def test_position_with_screen(self, qapp):
        overlay = kde_svc.HistoryOverlay()
        overlay._position()  # Should not crash

    def test_position_without_screen(self, qapp):
        overlay = kde_svc.HistoryOverlay()
        with patch.object(kde_svc.QGuiApplication, "primaryScreen", return_value=None):
            overlay._position()  # Should return early

    def test_clear(self, qapp):
        overlay = kde_svc.HistoryOverlay()
        items = [{"text": "a", "ts": time.time()}, {"text": "b", "ts": time.time()}]
        overlay.show_history(items)
        overlay._clear()
        assert not overlay._timer.isActive()
        overlay.dismiss()

    def test_show_replaces_previous(self, qapp):
        overlay = kde_svc.HistoryOverlay()
        overlay.show_history([{"text": "first", "ts": time.time()}])
        overlay.show_history([{"text": "second", "ts": time.time()}])
        overlay.dismiss()


# =========================================================================
# 8a. _time_ago — thorough pure logic tests
# =========================================================================

class TestTimeAgo:

    @pytest.fixture
    def overlay(self, qapp):
        return kde_svc.HistoryOverlay()

    def test_just_now_0_seconds(self, overlay):
        assert overlay._time_ago(time.time()) == "just now"

    def test_just_now_30_seconds(self, overlay):
        assert overlay._time_ago(time.time() - 30) == "just now"

    def test_just_now_59_seconds(self, overlay):
        assert overlay._time_ago(time.time() - 59) == "just now"

    def test_1_minute(self, overlay):
        assert overlay._time_ago(time.time() - 60) == "1m ago"

    def test_2_minutes(self, overlay):
        assert overlay._time_ago(time.time() - 120) == "2m ago"

    def test_30_minutes(self, overlay):
        assert overlay._time_ago(time.time() - 1800) == "30m ago"

    def test_59_minutes(self, overlay):
        assert overlay._time_ago(time.time() - 3599) == "59m ago"

    def test_1_hour(self, overlay):
        assert overlay._time_ago(time.time() - 3600) == "1h ago"

    def test_2_hours(self, overlay):
        assert overlay._time_ago(time.time() - 7200) == "2h ago"

    def test_23_hours(self, overlay):
        assert overlay._time_ago(time.time() - 82800) == "23h ago"

    def test_1_day(self, overlay):
        assert overlay._time_ago(time.time() - 86400) == "1d ago"

    def test_7_days(self, overlay):
        assert overlay._time_ago(time.time() - 604800) == "7d ago"

    def test_30_days(self, overlay):
        assert overlay._time_ago(time.time() - 2592000) == "30d ago"

    def test_future_timestamp(self, overlay):
        # Future timestamp -> negative diff -> falls through to "Xd ago"
        # Actually: diff < 0 means diff < 60 is True, so "just now"
        result = overlay._time_ago(time.time() + 3600)
        assert result == "just now"

    def test_boundary_60_seconds(self, overlay):
        # Exactly 60 seconds -> diff >= 60 -> minutes
        result = overlay._time_ago(time.time() - 60)
        assert result == "1m ago"

    def test_boundary_3600_seconds(self, overlay):
        # Exactly 3600 -> diff >= 3600 -> hours
        result = overlay._time_ago(time.time() - 3600)
        assert result == "1h ago"

    def test_boundary_86400_seconds(self, overlay):
        # Exactly 86400 -> diff >= 86400 -> days
        result = overlay._time_ago(time.time() - 86400)
        assert result == "1d ago"

    def test_90_seconds_is_1m(self, overlay):
        # 90 // 60 = 1
        assert overlay._time_ago(time.time() - 90) == "1m ago"

    def test_5400_seconds_is_90m_which_is_1h(self, overlay):
        # 5400 seconds = 90 minutes = 1.5 hours, but 5400 >= 3600 so hours branch
        # 5400 // 3600 = 1
        assert overlay._time_ago(time.time() - 5400) == "1h ago"

    def test_very_old_timestamp(self, overlay):
        # 365 days
        assert overlay._time_ago(time.time() - 365 * 86400) == "365d ago"

    def test_zero_timestamp(self, overlay):
        # ts=0 means diff ~ current time (very large)
        result = overlay._time_ago(0)
        assert result.endswith("d ago")


# =========================================================================
# 9. RewriteWorker
# =========================================================================

class TestRewriteWorker:

    def test_successful_rewrite(self, qapp):
        config = {"server-host": "localhost", "server-port": 5000}
        worker = kde_svc.RewriteWorker("hello", "diplomatic", config)

        results = []
        worker.finished.connect(lambda tone, text: results.append((tone, text)))

        api_response = {
            "variants": [
                {"tone": "diplomatic", "text": "Greetings"}
            ]
        }

        with patch.object(kde_svc, "http_post", return_value=api_response):
            worker.run()

        assert results == [("diplomatic", "Greetings")]

    def test_failed_http(self, qapp):
        config = {"server-host": "localhost", "server-port": 5000}
        worker = kde_svc.RewriteWorker("hello", "diplomatic", config)

        results = []
        worker.finished.connect(lambda tone, text: results.append((tone, text)))

        with patch.object(kde_svc, "http_post", return_value=None):
            worker.run()

        # Falls back to original text
        assert results == [("diplomatic", "hello")]

    def test_missing_variants_key(self, qapp):
        config = {}
        worker = kde_svc.RewriteWorker("hello", "diplomatic", config)

        results = []
        worker.finished.connect(lambda tone, text: results.append((tone, text)))

        with patch.object(kde_svc, "http_post", return_value={"status": "ok"}):
            worker.run()

        assert results == [("diplomatic", "hello")]

    def test_wrong_tone_in_variants(self, qapp):
        config = {}
        worker = kde_svc.RewriteWorker("hello", "diplomatic", config)

        results = []
        worker.finished.connect(lambda tone, text: results.append((tone, text)))

        api_response = {
            "variants": [
                {"tone": "casual", "text": "Hey there"}
            ]
        }

        with patch.object(kde_svc, "http_post", return_value=api_response):
            worker.run()

        # Tone doesn't match, falls back
        assert results == [("diplomatic", "hello")]

    def test_variant_with_error_field(self, qapp):
        config = {}
        worker = kde_svc.RewriteWorker("hello", "diplomatic", config)

        results = []
        worker.finished.connect(lambda tone, text: results.append((tone, text)))

        api_response = {
            "variants": [
                {"tone": "diplomatic", "text": "Greetings", "error": "LLM timeout"}
            ]
        }

        with patch.object(kde_svc, "http_post", return_value=api_response):
            worker.run()

        # Has error field, so skipped -> falls back
        assert results == [("diplomatic", "hello")]

    def test_server_url_called_correctly(self, qapp):
        config = {"server-host": "myhost", "server-port": 9999}
        worker = kde_svc.RewriteWorker("test", "formal", config)

        results = []
        worker.finished.connect(lambda tone, text: results.append((tone, text)))

        with patch.object(kde_svc, "http_post", return_value=None) as mock_post:
            worker.run()

        mock_post.assert_called_once_with(
            "http://myhost:9999/rewrite",
            {"text": "test", "tones": ["formal"]}
        )

    def test_multiple_variants_picks_matching(self, qapp):
        config = {}
        worker = kde_svc.RewriteWorker("hello", "professional", config)

        results = []
        worker.finished.connect(lambda tone, text: results.append((tone, text)))

        api_response = {
            "variants": [
                {"tone": "casual", "text": "Hey"},
                {"tone": "professional", "text": "Good day"},
                {"tone": "diplomatic", "text": "Greetings"},
            ]
        }

        with patch.object(kde_svc, "http_post", return_value=api_response):
            worker.run()

        assert results == [("professional", "Good day")]


# =========================================================================
# 10. LanguageBuddyService
# =========================================================================

class TestLanguageBuddyService:

    @pytest.fixture
    def service(self, qapp):
        overlay = kde_svc.LanguageBuddyOverlay()
        history_overlay = kde_svc.HistoryOverlay()
        bus_name = MagicMock()
        svc = kde_svc.LanguageBuddyService(bus_name, overlay, history_overlay)
        # Connect a dummy slot so that _process()'s disconnect() call won't fail
        overlay.card_selected.connect(lambda t: None)
        return svc

    # --- HandleTranscription ---

    def test_handle_transcription_enabled(self, service):
        config = {"language-buddy-enabled": True, "language-buddy-bypass": False, "language-buddy-timeout": 30}
        with patch.object(kde_svc, "load_config", return_value=config), \
             patch.object(service, "_process") as mock_proc:
            result = service.HandleTranscription("hello world")
        assert result is True
        mock_proc.assert_called_once_with("hello world", config)

    def test_handle_transcription_disabled(self, service):
        config = {"language-buddy-enabled": False}
        with patch.object(kde_svc, "load_config", return_value=config), \
             patch.object(service, "_process") as mock_proc:
            result = service.HandleTranscription("hello world")
        assert result is False
        mock_proc.assert_not_called()

    def test_handle_transcription_missing_key(self, service):
        # No language-buddy-enabled key -> defaults to False
        with patch.object(kde_svc, "load_config", return_value={}), \
             patch.object(service, "_process") as mock_proc:
            result = service.HandleTranscription("hello")
        assert result is False
        mock_proc.assert_not_called()

    # --- HandleTranscriptionWithContext ---

    def test_handle_with_context_enabled_valid(self, service):
        config = {"language-buddy-enabled": True, "language-buddy-bypass": False, "language-buddy-timeout": 30}
        context = json.dumps({"tones": ["casual", "formal"]})

        with patch.object(kde_svc, "load_config", return_value=config), \
             patch.object(service, "_process") as mock_proc:
            result = service.HandleTranscriptionWithContext("hello", context)
        assert result is True
        mock_proc.assert_called_once_with("hello", config, ["casual", "formal"])

    def test_handle_with_context_invalid_json_fallback(self, service):
        config = {"language-buddy-enabled": True, "language-buddy-bypass": False, "language-buddy-timeout": 30}

        with patch.object(kde_svc, "load_config", return_value=config), \
             patch.object(service, "_process") as mock_proc:
            result = service.HandleTranscriptionWithContext("hello", "not json!!!")
        assert result is True
        mock_proc.assert_called_once_with("hello", config, kde_svc.DEFAULT_TONES)

    def test_handle_with_context_disabled(self, service):
        config = {"language-buddy-enabled": False}

        with patch.object(kde_svc, "load_config", return_value=config), \
             patch.object(service, "_process") as mock_proc:
            result = service.HandleTranscriptionWithContext("hello", "{}")
        assert result is False
        mock_proc.assert_not_called()

    def test_handle_with_context_empty_tones_uses_default(self, service):
        config = {"language-buddy-enabled": True, "language-buddy-bypass": False, "language-buddy-timeout": 30}
        context = json.dumps({"tones": []})

        with patch.object(kde_svc, "load_config", return_value=config), \
             patch.object(service, "_process") as mock_proc:
            result = service.HandleTranscriptionWithContext("hello", context)
        assert result is True
        # Empty tones list is falsy, so DEFAULT_TONES used
        mock_proc.assert_called_once_with("hello", config, kde_svc.DEFAULT_TONES)

    def test_handle_with_context_none_tones_uses_default(self, service):
        config = {"language-buddy-enabled": True, "language-buddy-bypass": False, "language-buddy-timeout": 30}
        context = json.dumps({"tones": None})

        with patch.object(kde_svc, "load_config", return_value=config), \
             patch.object(service, "_process") as mock_proc:
            result = service.HandleTranscriptionWithContext("hello", context)
        assert result is True
        mock_proc.assert_called_once_with("hello", config, kde_svc.DEFAULT_TONES)

    # --- GetFocusedApp ---

    def test_get_focused_app_kwin_success(self, service):
        mock_bus = MagicMock()
        mock_result = {"resourceClass": "firefox", "caption": "Mozilla Firefox"}
        mock_obj = MagicMock()
        mock_obj.Get.return_value = mock_result
        mock_bus.get_object.return_value = mock_obj

        with patch.object(kde_svc.dbus, "SessionBus", return_value=mock_bus), \
             patch.object(kde_svc.dbus, "Interface", return_value=MagicMock()):
            result = service.GetFocusedApp()
        assert result == ("firefox", "Mozilla Firefox")

    def test_get_focused_app_kwin_fail_xdotool_success(self, service):
        mock_bus = MagicMock()
        mock_bus.get_object.side_effect = Exception("KWin not available")

        with patch.object(kde_svc.dbus, "SessionBus", return_value=mock_bus), \
             patch.object(kde_svc, "subprocess") as mock_sub:
            mock_sub.check_output.side_effect = [
                b"konsole\n",  # class name
                b"Terminal\n",  # window name
            ]
            mock_sub.DEVNULL = -1
            result = service.GetFocusedApp()

        assert result == ("konsole", "Terminal")

    def test_get_focused_app_all_fail(self, service):
        mock_bus = MagicMock()
        mock_bus.get_object.side_effect = Exception("fail")

        with patch.object(kde_svc.dbus, "SessionBus", return_value=mock_bus), \
             patch.object(kde_svc, "subprocess") as mock_sub:
            mock_sub.check_output.side_effect = Exception("xdotool fail")
            mock_sub.DEVNULL = -1
            result = service.GetFocusedApp()

        assert result == ("", "")

    # --- ShowHistoryPicker ---

    def test_show_history_picker_valid(self, service):
        items = [{"text": "hello", "ts": time.time()}]
        with patch.object(service._history_overlay, "show_history") as mock_show:
            result = service.ShowHistoryPicker(json.dumps(items))
        assert result is True
        mock_show.assert_called_once()

    def test_show_history_picker_empty_items(self, service):
        result = service.ShowHistoryPicker("[]")
        assert result is False

    def test_show_history_picker_invalid_json(self, service):
        result = service.ShowHistoryPicker("not json")
        assert result is False

    def test_show_history_picker_null(self, service):
        result = service.ShowHistoryPicker("null")
        assert result is False

    # --- _process ---

    def _run_process(self, service, text, config, tones=None):
        """Helper to run _process with threading mocked out.

        Patches RewriteWorker so moveToThread is never called on a real
        QObject with a MagicMock QThread.
        """
        mock_worker = MagicMock()
        with patch.object(kde_svc, "RewriteWorker", return_value=mock_worker), \
             patch.object(kde_svc, "QThread") as mock_thread_cls:
            mock_thread_cls.return_value = MagicMock()
            if tones is not None:
                service._process(text, config, tones)
            else:
                service._process(text, config)

    def test_process_bypass_mode(self, service, qapp):
        config = {"language-buddy-bypass": True, "language-buddy-timeout": 30}
        tones = ["diplomatic"]

        with patch.object(kde_svc, "type_text") as mock_type:
            self._run_process(service, "hello", config, tones)

        # In bypass mode, type_text is called with original text
        mock_type.assert_called_once_with("hello")

    def test_process_non_bypass(self, service, qapp):
        config = {"language-buddy-bypass": False, "language-buddy-timeout": 30}
        tones = ["diplomatic"]

        with patch.object(kde_svc, "type_text") as mock_type:
            self._run_process(service, "hello", config, tones)

        # Non-bypass: type_text should NOT be called immediately
        mock_type.assert_not_called()

    def test_process_text_over_50_adds_summarize(self, service, qapp):
        config = {"language-buddy-bypass": False, "language-buddy-timeout": 30}
        long_text = "x" * 51  # > 50 chars

        show_args = {}

        def capture_show(tones, text, timeout_sec=30):
            show_args["tones"] = list(tones)
            show_args["text"] = text
            show_args["timeout"] = timeout_sec

        with patch.object(service._overlay, "show_overlay", side_effect=capture_show):
            self._run_process(service, long_text, config, tones=None)

        assert "summarize" in show_args["tones"]
        assert "diplomatic" in show_args["tones"]
        assert "professional" in show_args["tones"]

    def test_process_text_under_50_no_summarize(self, service, qapp):
        config = {"language-buddy-bypass": False, "language-buddy-timeout": 30}
        short_text = "x" * 50  # exactly 50, not > 50

        show_args = {}

        def capture_show(tones, text, timeout_sec=30):
            show_args["tones"] = list(tones)

        with patch.object(service._overlay, "show_overlay", side_effect=capture_show):
            self._run_process(service, short_text, config, tones=None)

        assert "summarize" not in show_args["tones"]

    def test_process_custom_tones(self, service, qapp):
        config = {"language-buddy-bypass": False, "language-buddy-timeout": 15}
        custom_tones = ["casual", "formal", "pirate"]

        show_args = {}

        def capture_show(tones, text, timeout_sec=30):
            show_args["tones"] = list(tones)
            show_args["timeout"] = timeout_sec

        with patch.object(service._overlay, "show_overlay", side_effect=capture_show):
            self._run_process(service, "hello", config, tones=custom_tones)

        assert show_args["tones"] == ["casual", "formal", "pirate"]
        assert show_args["timeout"] == 15

    def test_process_bypass_on_select_different_text(self, service, qapp):
        """In bypass mode, selecting different text should backspace and retype."""
        config = {"language-buddy-bypass": True, "language-buddy-timeout": 30}

        with patch.object(kde_svc, "type_text"):
            self._run_process(service, "hello", config, tones=["diplomatic"])

        # Now simulate card_selected signal
        with patch.object(kde_svc, "type_text") as mock_type2, \
             patch.object(kde_svc, "backspace_n") as mock_bs2:
            service._overlay.card_selected.emit("different text")
        mock_bs2.assert_called_once_with(5)  # len("hello") = 5
        mock_type2.assert_called_once_with("different text")

    def test_process_bypass_on_select_same_text(self, service, qapp):
        """In bypass mode, selecting the same text should not backspace."""
        config = {"language-buddy-bypass": True, "language-buddy-timeout": 30}

        with patch.object(kde_svc, "type_text"):
            self._run_process(service, "hello", config, tones=["diplomatic"])

        with patch.object(kde_svc, "type_text") as mock_type2, \
             patch.object(kde_svc, "backspace_n") as mock_bs2:
            service._overlay.card_selected.emit("hello")  # same as original
        mock_bs2.assert_not_called()
        mock_type2.assert_not_called()

    def test_process_non_bypass_on_select(self, service, qapp):
        """In non-bypass mode, selecting text should type it."""
        config = {"language-buddy-bypass": False, "language-buddy-timeout": 30}

        self._run_process(service, "hello", config, tones=["diplomatic"])

        with patch.object(kde_svc, "type_text") as mock_type:
            service._overlay.card_selected.emit("selected text")
        mock_type.assert_called_once_with("selected text")


# =========================================================================
# Edge cases and integration-style tests
# =========================================================================

class TestModuleConstants:

    def test_default_tones(self):
        assert kde_svc.DEFAULT_TONES == ["diplomatic", "professional"]

    def test_config_path_is_string(self):
        assert isinstance(kde_svc.CONFIG_PATH, str)
        assert "settings.json" in kde_svc.CONFIG_PATH


class TestServerUrlIntegration:

    def test_load_config_and_server_url_together(self, tmp_path):
        cfg = {"server-host": "10.0.0.1", "server-port": 8080}
        cfg_file = tmp_path / "settings.json"
        cfg_file.write_text(json.dumps(cfg))

        with patch.object(kde_svc, "CONFIG_PATH", str(cfg_file)):
            url = kde_svc.server_url("/rewrite")
        assert url == "http://10.0.0.1:8080/rewrite"
