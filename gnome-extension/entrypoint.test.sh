#!/bin/bash
set -euo pipefail

UUID="whisper-npu@dmz.oneill"
EXT_DIR="/root/.local/share/gnome-shell/extensions/$UUID"
GLIB_SCHEMA_DIR="/root/.local/share/glib-2.0/schemas"

# Compile extension schemas into the user schema dir so gsettings finds them
if [ -d "$EXT_DIR/schemas" ]; then
    glib-compile-schemas "$EXT_DIR/schemas" 2>/dev/null || true
    cp "$EXT_DIR/schemas"/*.gschema.xml "$GLIB_SCHEMA_DIR"/ 2>/dev/null || true
    glib-compile-schemas "$GLIB_SCHEMA_DIR" 2>/dev/null || true
fi

# Enable extension before starting the shell
gsettings set org.gnome.shell enabled-extensions "[\"$UUID\"]"

exec gnome-shell "$@"
