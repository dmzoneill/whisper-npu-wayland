.PHONY: install uninstall install-service uninstall-service enable-service disable-service \
       dev logs logs-service preview preview-viewer preview-dbus preview-stop deps clean

SHELL := /bin/bash
CWD := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PLUGIN_ID = com.whisper.npu
INSTALL_DIR = ~/.local/share/plasma/plasmoids/$(PLUGIN_ID)
SERVICE_DIR = ~/.config/systemd/user
CONFIG_DIR = ~/.config/whisper-npu

# Fedora package names — adjust for other distros
KDE_RUNTIME_DEPS = plasma-workspace python3-dbus python3-pyqt6 dbus-tools
KDE_PREVIEW_DEPS = plasma-sdk
KDE_ALL_DEPS = $(KDE_RUNTIME_DEPS) $(KDE_PREVIEW_DEPS)

install:
	mkdir -p $(dir $(INSTALL_DIR))
	ln -snf $(CWD) $(INSTALL_DIR)
	mkdir -p $(CONFIG_DIR)
	@echo "Plasmoid installed. Add 'Whisper NPU' widget to your panel or system tray."
	@echo ""
	@echo "To install the D-Bus service (required for Language Buddy): make install-service"

uninstall:
	rm -f $(INSTALL_DIR)
	@echo "Plasmoid uninstalled."

install-service:
	mkdir -p $(SERVICE_DIR)
	cp $(CWD)/whisper-npu-kde.service $(SERVICE_DIR)/
	systemctl --user daemon-reload
	systemctl --user enable whisper-npu-kde.service
	systemctl --user start whisper-npu-kde.service
	@echo "D-Bus service installed and started."

uninstall-service: disable-service
	rm -f $(SERVICE_DIR)/whisper-npu-kde.service
	systemctl --user daemon-reload
	@echo "D-Bus service uninstalled."

enable-service:
	systemctl --user enable whisper-npu-kde.service
	systemctl --user start whisper-npu-kde.service

disable-service:
	-systemctl --user stop whisper-npu-kde.service
	-systemctl --user disable whisper-npu-kde.service

dev: install install-service
	@echo ""
	@echo "========================================"
	@echo "Whisper NPU KDE plasmoid installed!"
	@echo "========================================"
	@echo ""
	@echo "1. Right-click panel > Add Widgets > search 'Whisper NPU'"
	@echo "2. Or add to system tray via System Tray Settings"
	@echo ""
	@echo "Watch logs:   make logs"
	@echo "Service logs: make logs-service"

# ---------- Dependencies ------------------------------------------

deps:
	@echo "Installing KDE plasmoid dependencies..."
	sudo dnf install -y $(KDE_ALL_DEPS)
	@echo "Dependencies installed."

# ---------- Preview / testing (no Plasma session required) --------

preview: deps install
	@echo "Starting plasmoid preview with plasmawindowed..."
	@echo "Close the window or press Ctrl+C to stop."
	@echo ""
	plasmawindowed $(PLUGIN_ID) 2>&1

preview-viewer: deps install
	@echo "Starting plasmoid preview with plasmoidviewer..."
	@echo "Use the toolbar to switch form factor (panel / desktop / sidebar)."
	@echo "Close the window or press Ctrl+C to stop."
	@echo ""
	plasmoidviewer -a $(PLUGIN_ID) 2>&1

preview-dbus: deps
	@echo "Starting D-Bus overlay service in foreground..."
	@echo "Test with:"
	@echo '  dbus-send --session --dest=com.whisper.LanguageBuddy --print-reply \'
	@echo '    /com/whisper/LanguageBuddy com.whisper.LanguageBuddy.HandleTranscription \'
	@echo '    string:"Hello this is a test transcription"'
	@echo ""
	python3 $(CWD)/whisper-npu-kde-service.py

preview-stop:
	-pkill -f "plasmawindowed $(PLUGIN_ID)" 2>/dev/null
	-pkill -f "plasmoidviewer.*$(PLUGIN_ID)" 2>/dev/null
	-pkill -f "whisper-npu-kde-service" 2>/dev/null
	@echo "Preview processes stopped."

# ------------------------------------------------------------------

logs:
	journalctl --user -f -u push-to-talk.service -u whisper-server.service

logs-service:
	journalctl --user -f -u whisper-npu-kde.service

clean:
	@echo "Nothing to clean."
