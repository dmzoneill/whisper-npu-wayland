.PHONY: install uninstall dev reload enable disable logs clean

SHELL := /bin/bash
CWD := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
UUID = whisper-npu@dmz.oneill
INSTALL_DIR = ~/.local/share/gnome-shell/extensions/$(UUID)
SCHEMA_DIR = ~/.local/share/glib-2.0/schemas

install:
	mkdir -p $(SCHEMA_DIR)
	glib-compile-schemas schemas/
	cp schemas/*.gschema.xml $(SCHEMA_DIR)/
	glib-compile-schemas $(SCHEMA_DIR)/
	ln -snf $(CWD) $(INSTALL_DIR)
	@echo "Extension installed. Enable with: make enable"
	@echo "On Wayland, log out and back in to load the extension."

uninstall: disable
	rm -f $(INSTALL_DIR)
	rm -f $(SCHEMA_DIR)/org.gnome.shell.extensions.whisper-npu.gschema.xml
	glib-compile-schemas $(SCHEMA_DIR)/
	@echo "Extension uninstalled."

enable:
	gnome-extensions enable $(UUID)

disable:
	-gnome-extensions disable $(UUID)

reload: disable enable
	@echo "Extension reloaded (JS/CSS changes may require logout/login on Wayland)"

dev: install enable
	@echo ""
	@echo "========================================"
	@echo "Whisper NPU extension installed and enabled!"
	@echo "========================================"
	@echo ""
	@echo "Watch logs: make logs"
	@echo ""
	@echo "Note: JS/CSS changes require logout/login on Wayland."

logs:
	journalctl -f -o cat /usr/bin/gnome-shell

clean:
	rm -f schemas/gschemas.compiled
	glib-compile-schemas schemas/
