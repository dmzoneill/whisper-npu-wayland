#!/bin/bash
set -euo pipefail

UUID="whisper-npu@dmz.oneill"
EXT_DIR="/root/.local/share/gnome-shell/extensions/$UUID"
GLIB_SCHEMA_DIR="/root/.local/share/glib-2.0/schemas"

# Compile extension schemas
if [ -d "$EXT_DIR/schemas" ]; then
    glib-compile-schemas "$EXT_DIR/schemas" 2>/dev/null || true
    cp "$EXT_DIR/schemas"/*.gschema.xml "$GLIB_SCHEMA_DIR"/ 2>/dev/null || true
    glib-compile-schemas "$GLIB_SCHEMA_DIR" 2>/dev/null || true
fi

# Enable extension
gsettings set org.gnome.shell enabled-extensions "[\"$UUID\"]"

# Run GNOME Shell as a Wayland compositor in devkit mode.
# --devkit creates a GTK window on WAYLAND_DISPLAY (the host desktop) that mirrors the
# virtual monitor output — no --virtual-monitor flag, which was causing the panel to render
# to a second monitor invisible in the devkit window.
exec gnome-shell --wayland --no-x11 --devkit "$@"
