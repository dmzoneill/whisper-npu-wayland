#!/usr/bin/env python3
"""D-Bus service + Qt6 overlay for Whisper NPU on KDE Plasma.

Registers com.whisper.LanguageBuddy on the session bus with the same
interface used by the GNOME extension, so push-to-talk.py works
unmodified on KDE.

Requires: PyQt6, dbus-python
"""

import json
import logging
import os
import subprocess
import sys
import time
import urllib.request

import dbus
import dbus.mainloop.glib
import dbus.service
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QGuiApplication, QFont, QColor
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QGraphicsDropShadowEffect
)

log = logging.getLogger("whisper-npu-kde")
logging.basicConfig(level=logging.DEBUG, format="[whisper-npu-kde] %(message)s")

CONFIG_PATH = os.path.expanduser("~/.config/whisper-npu/settings.json")
DEFAULT_TONES = ["diplomatic", "professional"]


def load_config():
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def server_url(path, config=None):
    if config is None:
        config = load_config()
    host = config.get("server-host", "127.0.0.1")
    port = config.get("server-port", 5000)
    return f"http://{host}:{port}{path}"


def http_post(url, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data,
                                headers={"Content-Type": "application/json"},
                                method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        log.error("HTTP POST %s failed: %s", url, e)
        return None


def type_text(text, delay_ms=4):
    try:
        subprocess.run(["ydotool", "type", "-d", str(delay_ms), "--", text],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log.error("ydotool type failed: %s", e)


def backspace_n(n):
    for _ in range(n):
        try:
            subprocess.run(["ydotool", "key", "14:1", "14:0"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Overlay window — matches GNOME extension's floating card style
# ---------------------------------------------------------------------------

class OverlayCard(QPushButton):
    """A single card in the overlay."""

    def __init__(self, tone, text, ready=False, parent=None):
        super().__init__(parent)
        self.tone = tone
        self.card_text = text
        self.ready = ready

        self.setCursor(Qt.CursorShape.PointingHandCursor if ready
                       else Qt.CursorShape.ArrowCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.setFlat(True)
        self._update_style()
        self._build_layout(tone, text)

    def _build_layout(self, tone, text):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        self.tone_label = QLabel(tone[0].upper() + tone[1:])
        self.tone_label.setFont(QFont("Sans", 9, QFont.Weight.Bold))
        self.tone_label.setStyleSheet("color: rgba(120, 174, 237, 0.9);")
        layout.addWidget(self.tone_label)

        self.text_label = QLabel(text)
        self.text_label.setWordWrap(True)
        self.text_label.setFont(QFont("Sans", 11))
        self.text_label.setStyleSheet("color: rgba(255, 255, 255, 0.9);")
        layout.addWidget(self.text_label)

    def update_text(self, text):
        self.card_text = text
        self.text_label.setText(text)
        self.ready = True
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_style()

    def _update_style(self):
        self.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 15);
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 20);
                text-align: left;
                padding: 0;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 30);
                border-color: rgba(120, 174, 237, 128);
            }
        """)

    def enterEvent(self, event):
        super().enterEvent(event)

    def leaveEvent(self, event):
        super().leaveEvent(event)


class LanguageBuddyOverlay(QWidget):
    """Floating overlay at bottom-right showing rewrite variants."""

    card_selected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self._cards = {}
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.dismiss)

        self._container = QWidget(self)
        self._container.setStyleSheet("""
            QWidget {
                background-color: rgba(30, 30, 30, 242);
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 25);
            }
        """)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 128))
        self._container.setGraphicsEffect(shadow)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.addWidget(self._container)

        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(16, 16, 16, 16)
        self._layout.setSpacing(8)

    def show_overlay(self, tones, original_text, timeout_sec=30):
        self._clear()

        header_layout = QHBoxLayout()
        header = QLabel("Language Buddy")
        header.setFont(QFont("Sans", 10, QFont.Weight.Bold))
        header.setStyleSheet(
            "color: rgba(255, 255, 255, 153); text-transform: uppercase; letter-spacing: 1px;")
        header_layout.addWidget(header, stretch=1)

        dismiss_btn = QPushButton("✕")
        dismiss_btn.setFixedSize(28, 28)
        dismiss_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border-radius: 14px;
                color: rgba(255, 255, 255, 128);
                font-size: 14px;
                border: none;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 38);
                color: rgba(255, 255, 255, 230);
            }
        """)
        dismiss_btn.clicked.connect(self.dismiss)
        header_layout.addWidget(dismiss_btn)
        self._layout.addLayout(header_layout)

        orig_card = OverlayCard("original", original_text, ready=True)
        orig_card.clicked.connect(lambda: self._on_select(original_text))
        self._layout.addWidget(orig_card)
        self._cards["original"] = orig_card

        for tone in tones:
            card = OverlayCard(tone, "Processing...", ready=False)
            card.clicked.connect(lambda checked=False, t=tone: self._on_card_click(t))
            self._layout.addWidget(card)
            self._cards[tone] = card

        self.setMinimumWidth(380)
        self.setMaximumWidth(480)
        self.adjustSize()
        self._position()
        self.show()

        if timeout_sec > 0:
            self._timer.start(timeout_sec * 1000)

    def update_card(self, tone, text):
        card = self._cards.get(tone)
        if card:
            card.update_text(text)
            self.adjustSize()
            self._position()

    def _on_card_click(self, tone):
        card = self._cards.get(tone)
        if card and card.ready:
            self._on_select(card.card_text)

    def _on_select(self, text):
        self.card_selected.emit(text)
        self.dismiss()

    def _position(self):
        screen = QGuiApplication.primaryScreen()
        if not screen:
            return
        geom = screen.availableGeometry()
        padding = 100
        x = geom.x() + geom.width() - self.width() - padding
        y = geom.y() + geom.height() - self.height() - padding
        self.move(x, y)

    def _clear(self):
        self._timer.stop()
        self._cards.clear()
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()

    def dismiss(self):
        self._timer.stop()
        self.hide()
        self._clear()


class HistoryOverlay(QWidget):
    """Floating overlay for transcription history recall."""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.dismiss)

        self._container = QWidget(self)
        self._container.setStyleSheet("""
            QWidget {
                background-color: rgba(30, 30, 30, 242);
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 25);
            }
        """)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 128))
        self._container.setGraphicsEffect(shadow)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.addWidget(self._container)

        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(16, 16, 16, 16)
        self._layout.setSpacing(8)

    def show_history(self, items, timeout_sec=10):
        self._clear()

        header_layout = QHBoxLayout()
        header = QLabel("Transcription History")
        header.setFont(QFont("Sans", 10, QFont.Weight.Bold))
        header.setStyleSheet(
            "color: rgba(255, 255, 255, 153); text-transform: uppercase; letter-spacing: 1px;")
        header_layout.addWidget(header, stretch=1)

        dismiss_btn = QPushButton("✕")
        dismiss_btn.setFixedSize(28, 28)
        dismiss_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent; border-radius: 14px;
                color: rgba(255, 255, 255, 128); font-size: 14px; border: none;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 38);
                color: rgba(255, 255, 255, 230);
            }
        """)
        dismiss_btn.clicked.connect(self.dismiss)
        header_layout.addWidget(dismiss_btn)
        self._layout.addLayout(header_layout)

        for item in items:
            text = item.get("text", "")
            ts = item.get("ts", 0)
            ago = self._time_ago(ts)
            display = text[:120] + "..." if len(text) > 120 else text

            card = OverlayCard(ago, display, ready=True)
            full_text = text
            card.clicked.connect(lambda checked=False, t=full_text: self._on_select(t))
            self._layout.addWidget(card)

        self.setMinimumWidth(380)
        self.setMaximumWidth(480)
        self.adjustSize()
        self._position()
        self.show()

        if timeout_sec > 0:
            self._timer.start(timeout_sec * 1000)

    def _time_ago(self, ts):
        diff = time.time() - ts
        if diff < 60:
            return "just now"
        if diff < 3600:
            return f"{int(diff // 60)}m ago"
        if diff < 86400:
            return f"{int(diff // 3600)}h ago"
        return f"{int(diff // 86400)}d ago"

    def _on_select(self, text):
        self.dismiss()
        type_text(text)

    def _position(self):
        screen = QGuiApplication.primaryScreen()
        if not screen:
            return
        geom = screen.availableGeometry()
        padding = 100
        x = geom.x() + geom.width() - self.width() - padding
        y = geom.y() + geom.height() - self.height() - padding
        self.move(x, y)

    def _clear(self):
        self._timer.stop()
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()

    def dismiss(self):
        self._timer.stop()
        self.hide()
        self._clear()


# ---------------------------------------------------------------------------
# Rewrite worker — runs HTTP calls off the main thread
# ---------------------------------------------------------------------------

class RewriteWorker(QObject):
    finished = pyqtSignal(str, str)  # tone, text

    def __init__(self, text, tone, config):
        super().__init__()
        self.text = text
        self.tone = tone
        self.config = config

    def run(self):
        url = server_url("/rewrite", self.config)
        result = http_post(url, {"text": self.text, "tones": [self.tone]})
        if result and "variants" in result:
            for v in result["variants"]:
                if v.get("tone") == self.tone and not v.get("error"):
                    self.finished.emit(self.tone, v["text"])
                    return
        self.finished.emit(self.tone, self.text)


# ---------------------------------------------------------------------------
# D-Bus service
# ---------------------------------------------------------------------------

class LanguageBuddyService(dbus.service.Object):

    def __init__(self, bus_name, overlay, history_overlay):
        super().__init__(bus_name, "/com/whisper/LanguageBuddy")
        self._overlay = overlay
        self._history_overlay = history_overlay
        self._threads = []

    @dbus.service.method("com.whisper.LanguageBuddy",
                         in_signature="s", out_signature="b")
    def HandleTranscription(self, text):
        config = load_config()
        if not config.get("language-buddy-enabled", False):
            return False
        log.info("HandleTranscription: %s", text[:80])
        self._process(str(text), config)
        return True

    @dbus.service.method("com.whisper.LanguageBuddy",
                         in_signature="ss", out_signature="b")
    def HandleTranscriptionWithContext(self, text, context_json):
        config = load_config()
        if not config.get("language-buddy-enabled", False):
            return False
        log.info("HandleTranscriptionWithContext: %s", text[:80])
        try:
            context = json.loads(str(context_json))
            tones = context.get("tones", []) or DEFAULT_TONES
        except (json.JSONDecodeError, TypeError):
            tones = DEFAULT_TONES
        self._process(str(text), config, tones)
        return True

    @dbus.service.method("com.whisper.LanguageBuddy",
                         in_signature="", out_signature="ss")
    def GetFocusedApp(self):
        try:
            bus = dbus.SessionBus()
            kwin = bus.get_object("org.kde.KWin", "/KWin")
            iface = dbus.Interface(kwin, "org.kde.KWin")
            iface.loadScript("/dev/null", "whisper-npu-query")
        except Exception:
            pass

        try:
            bus = dbus.SessionBus()
            obj = bus.get_object("org.kde.KWin", "/Scripting")
            result = obj.Get("org.kde.KWin.Scripting", "activeClient",
                             dbus_interface="org.freedesktop.DBus.Properties")
            return (str(result.get("resourceClass", "")),
                    str(result.get("caption", "")))
        except Exception:
            pass

        try:
            output = subprocess.check_output(
                ["xdotool", "getactivewindow", "getwindowclassname"],
                timeout=2, stderr=subprocess.DEVNULL).decode().strip()
            title = subprocess.check_output(
                ["xdotool", "getactivewindow", "getwindowname"],
                timeout=2, stderr=subprocess.DEVNULL).decode().strip()
            return (output, title)
        except Exception:
            return ("", "")

    @dbus.service.method("com.whisper.LanguageBuddy",
                         in_signature="s", out_signature="b")
    def ShowHistoryPicker(self, items_json):
        try:
            items = json.loads(str(items_json))
            if not items:
                return False
            self._history_overlay.show_history(items, timeout_sec=10)
            return True
        except Exception as e:
            log.error("ShowHistoryPicker: %s", e)
            return False

    def _process(self, text, config, tones=None):
        bypass = config.get("language-buddy-bypass", False)
        timeout_sec = config.get("language-buddy-timeout", 30)

        if bypass:
            type_text(text)

        if tones is None:
            tones = list(DEFAULT_TONES)
            if len(text) > 50:
                tones.append("summarize")

        original_text = text

        if bypass:
            def on_select(selected):
                if selected != original_text:
                    backspace_n(len(original_text))
                    type_text(selected)
        else:
            def on_select(selected):
                type_text(selected)

        self._overlay.card_selected.disconnect()
        self._overlay.card_selected.connect(on_select)
        self._overlay.show_overlay(tones, text, timeout_sec)

        for tone in tones:
            worker = RewriteWorker(text, tone, config)
            thread = QThread()
            worker.moveToThread(thread)
            worker.finished.connect(self._overlay.update_card)
            worker.finished.connect(thread.quit)
            thread.started.connect(worker.run)
            thread.finished.connect(lambda t=thread: self._threads.remove(t))
            self._threads.append(thread)
            thread.start()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    app = QApplication(sys.argv)
    app.setApplicationName("whisper-npu-kde")
    app.setQuitOnLastWindowClosed(False)

    overlay = LanguageBuddyOverlay()
    history_overlay = HistoryOverlay()

    bus = dbus.SessionBus()
    bus_name = dbus.service.BusName("com.whisper.LanguageBuddy", bus)
    service = LanguageBuddyService(bus_name, overlay, history_overlay)  # noqa: F841

    log.info("D-Bus service registered: com.whisper.LanguageBuddy")
    log.info("Listening for transcription handoffs...")

    # Integrate GLib main loop with Qt
    from gi.repository import GLib
    glib_loop = GLib.MainLoop()

    def glib_tick():
        ctx = glib_loop.get_context()
        while ctx.pending():
            ctx.iteration(False)

    timer = QTimer()
    timer.timeout.connect(glib_tick)
    timer.start(50)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
