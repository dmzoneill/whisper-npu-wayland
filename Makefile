SHELL := /bin/bash

PROJECT_DIR  := $(CURDIR)
SYSTEMD_DIR  := $(HOME)/.config/systemd/user
PYTHON       := /usr/bin/python3

WHISPER_DEVICE     ?= auto
WHISPER_MODEL      ?= whisper-small.en-fp16-ov
WHISPER_CT2_MODEL  ?= small.en
WHISPER_CPP_PORT   ?= 5001
WHISPER_CPP_DEVICE ?= NPU

SYSTEM_PKGS := ydotool pipewire-pulseaudio pipewire-utils pulseaudio-utils \
               wtype wl-clipboard xdotool \
               git git-lfs curl \
               cmake gcc-c++ \
               pciutils libnotify

WHISPER_CPP_VERSION ?= v1.7.4
WHISPER_CPP_SRC     := $(PROJECT_DIR)/.whisper-cpp

SERVICE_FILES := $(SYSTEMD_DIR)/whisper-server.service \
                 $(SYSTEMD_DIR)/whisper-cpp-server.service \
                 $(SYSTEMD_DIR)/push-to-talk.service

WHISPER_MODELS_DIR := $(HOME)/.whisper/models
HF_ORG := OpenVINO

.PHONY: help install install-python install-system install-whisper-cpp \
        install-services install-permissions install-models enable start \
        stop restart status logs logs-server logs-cpp logs-ptt \
        test uninstall clean \
        extension extension-install extension-uninstall extension-enable \
        extension-disable extension-reload extension-dev extension-logs

.DEFAULT_GOAL := help

# ----------------------------------------------------------------------------
# Help
# ----------------------------------------------------------------------------

help: ## Show available targets
	@echo "whisper-npu-server"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables (override with make VAR=value):"
	@echo "  WHISPER_DEVICE       Device: auto|NPU|GPU|CUDA|CPU [$(WHISPER_DEVICE)]"
	@echo "  WHISPER_MODEL        OpenVINO model name         [$(WHISPER_MODEL)]"
	@echo "  WHISPER_CT2_MODEL    faster-whisper model name   [$(WHISPER_CT2_MODEL)]"
	@echo "  WHISPER_CPP_DEVICE   Device for whisper.cpp      [$(WHISPER_CPP_DEVICE)]"
	@echo "  WHISPER_CPP_PORT     Port for whisper.cpp        [$(WHISPER_CPP_PORT)]"

# ----------------------------------------------------------------------------
# Full install
# ----------------------------------------------------------------------------

install: install-python install-system install-whisper-cpp install-permissions install-models install-services enable start ## Install everything

# ----------------------------------------------------------------------------
# Python dependencies
# ----------------------------------------------------------------------------

install-python: ## Install Python packages (auto-detects NPU/GPU/CUDA hardware)
	@echo "=== Hardware Detection ==="; \
	HAS_INTEL=0; HAS_NPU=0; HAS_ARC=0; HAS_CUDA=0; \
	if grep -qi "intel" /proc/cpuinfo 2>/dev/null || lspci 2>/dev/null | grep -qi "intel"; then \
		HAS_INTEL=1; \
	fi; \
	if ls /dev/accel* 2>/dev/null | grep -q . || lspci 2>/dev/null | grep -qi "VPU\|NPU\|8087:"; then \
		HAS_NPU=1; echo "  [x] Intel NPU detected"; \
	fi; \
	if lspci 2>/dev/null | grep -qi "intel.*DG\|intel.*arc\|intel.*battlemage\|intel.*alchemist"; then \
		HAS_ARC=1; echo "  [x] Intel ARC GPU detected"; \
	fi; \
	if [ $$HAS_INTEL -eq 1 ] && [ $$HAS_NPU -eq 0 ] && [ $$HAS_ARC -eq 0 ]; then \
		echo "  [x] Intel iGPU/CPU detected"; \
	fi; \
	if nvidia-smi -L 2>/dev/null | grep -q "GPU"; then \
		HAS_CUDA=1; \
		CUDA_MAJOR=$$(nvidia-smi 2>/dev/null | grep -oP 'CUDA( UMD)? Version: \K\d+' | head -1); \
		echo "  [x] NVIDIA CUDA GPU detected (CUDA $${CUDA_MAJOR:-unknown})"; \
		if [ -n "$$CUDA_MAJOR" ] && [ "$$CUDA_MAJOR" -lt 12 ]; then \
			echo "  [!] Warning: CUDA $${CUDA_MAJOR} detected — faster-whisper requires CUDA 12+."; \
			echo "      Upgrade driver or install manually: pip install 'faster-whisper<1.0' 'ctranslate2<4'"; \
			HAS_CUDA=0; \
		fi; \
	fi; \
	if [ $$HAS_INTEL -eq 0 ] && [ $$HAS_CUDA -eq 0 ]; then \
		echo "  [ ] No accelerator found — installing OpenVINO for CPU fallback"; \
		HAS_INTEL=1; \
	fi; \
	echo ""; \
	echo "=== Installing Base Packages ==="; \
	$(PYTHON) -m pip install --user -r $(PROJECT_DIR)/requirements-base.txt; \
	if [ $$HAS_INTEL -eq 1 ] || [ $$HAS_NPU -eq 1 ] || [ $$HAS_ARC -eq 1 ]; then \
		echo ""; \
		echo "=== Installing OpenVINO (NPU / Intel GPU / ARC / CPU) ==="; \
		$(PYTHON) -m pip install --user -r $(PROJECT_DIR)/requirements-openvino.txt; \
	fi; \
	if [ $$HAS_CUDA -eq 1 ]; then \
		echo ""; \
		echo "=== Installing faster-whisper (CUDA) ==="; \
		$(PYTHON) -m pip install --user -r $(PROJECT_DIR)/requirements-cuda.txt; \
	fi; \
	echo ""; \
	echo "Done. Set WHISPER_DEVICE=auto (default) to let the server pick the best device."

# ----------------------------------------------------------------------------
# System packages
# ----------------------------------------------------------------------------

install-system: ## Install system packages via dnf (requires sudo)
	sudo dnf install -y $(SYSTEM_PKGS)
	sudo mkdir -p /etc/systemd/system/ydotool.service.d
	@printf '%s\n' \
		'[Service]' \
		'RestartSec=3' \
		"ExecStartPost=/bin/bash -c 'sleep 0.5 && for s in /tmp/.ydotool_socket /run/user/*/.ydotool_socket; do [ -S \"$$s\" ] && chmod 666 \"$$s\"; done'" \
		| sudo tee /etc/systemd/system/ydotool.service.d/socket-permissions.conf > /dev/null
	sudo systemctl daemon-reload
	sudo systemctl enable ydotool.service
	sudo systemctl restart ydotool.service
	@sleep 1
	@YDOTOOL_SOCKET=$$(ls /tmp/.ydotool_socket /run/user/*/.ydotool_socket 2>/dev/null | head -1); \
	if [ -n "$$YDOTOOL_SOCKET" ]; then \
		YDOTOOL_SOCKET=$$YDOTOOL_SOCKET ydotool type -- "" 2>/dev/null \
			&& echo "  [x] ydotool OK ($$YDOTOOL_SOCKET)" \
			|| echo "  [!] ydotool socket found at $$YDOTOOL_SOCKET but command failed — check permissions"; \
	else \
		echo "  [!] ydotool socket not found — daemon may not have started yet"; \
	fi

# ----------------------------------------------------------------------------
# whisper.cpp (libwhisper.so)
# ----------------------------------------------------------------------------

install-whisper-cpp: ## Build and install libwhisper.so from source
	@if [ -f /usr/local/lib/libwhisper.so ]; then \
		echo "libwhisper.so already installed"; \
	else \
		echo "Building whisper.cpp $(WHISPER_CPP_VERSION)..."; \
		git clone --depth 1 --branch $(WHISPER_CPP_VERSION) https://github.com/ggerganov/whisper.cpp.git $(WHISPER_CPP_SRC) && \
		cmake -B $(WHISPER_CPP_SRC)/build -S $(WHISPER_CPP_SRC) \
			-DCMAKE_BUILD_TYPE=Release \
			-DWHISPER_OPENVINO=ON \
			-DOpenVINO_DIR=/usr/local/lib64/python3.14/site-packages/openvino/cmake && \
		cmake --build $(WHISPER_CPP_SRC)/build --config Release -j$$(nproc) && \
		sudo cmake --install $(WHISPER_CPP_SRC)/build && \
		sudo ldconfig && \
		rm -rf $(WHISPER_CPP_SRC) && \
		echo "libwhisper.so installed"; \
	fi

# ----------------------------------------------------------------------------
# User permissions
# ----------------------------------------------------------------------------

install-permissions: ## Add user to input group for evdev access
	@if id -nG "$(USER)" | grep -qw input; then \
		echo "$(USER) already in input group"; \
	else \
		sudo usermod -aG input "$(USER)"; \
		echo "Added $(USER) to input group — log out and back in to apply"; \
	fi

# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

install-models: ## Download default OpenVINO model if not already present
	@mkdir -p $(WHISPER_MODELS_DIR)
	@if [ -d "$(WHISPER_MODELS_DIR)/$(WHISPER_MODEL)" ]; then \
		echo "Model $(WHISPER_MODEL) already present"; \
	else \
		echo "Downloading $(WHISPER_MODEL) from $(HF_ORG)..."; \
		GIT_LFS_SKIP_SMUDGE=1 git clone "https://huggingface.co/$(HF_ORG)/$(WHISPER_MODEL)" "$(WHISPER_MODELS_DIR)/$(WHISPER_MODEL)" && \
		cd "$(WHISPER_MODELS_DIR)/$(WHISPER_MODEL)" && git lfs pull; \
	fi

# ----------------------------------------------------------------------------
# Systemd services
# ----------------------------------------------------------------------------

install-services: ## Install systemd user service files (always regenerates)
	@rm -f $(SERVICE_FILES)
	@$(MAKE) $(SERVICE_FILES)
	systemctl --user daemon-reload

$(SYSTEMD_DIR)/whisper-server.service:
	@mkdir -p $(SYSTEMD_DIR)
	@printf '%s\n' \
		'[Unit]' \
		'Description=Whisper Speech-to-Text Server' \
		'After=basic.target' \
		'' \
		'[Service]' \
		'Type=simple' \
		'WorkingDirectory=$(PROJECT_DIR)' \
		'ExecStart=$(PYTHON) $(PROJECT_DIR)/server-native.py' \
		'Environment=WHISPER_DEVICE=$(WHISPER_DEVICE)' \
		'Environment=WHISPER_MODEL=$(WHISPER_MODEL)' \
		'Environment=WHISPER_CT2_MODEL=$(WHISPER_CT2_MODEL)' \
		'Restart=on-failure' \
		'RestartSec=5' \
		'' \
		'[Install]' \
		'WantedBy=default.target' > $@

$(SYSTEMD_DIR)/whisper-cpp-server.service:
	@mkdir -p $(SYSTEMD_DIR)
	@printf '%s\n' \
		'[Unit]' \
		'Description=Whisper.cpp Speech-to-Text Server (NPU)' \
		'After=basic.target' \
		'' \
		'[Service]' \
		'Type=simple' \
		'WorkingDirectory=$(PROJECT_DIR)' \
		'Environment=LD_LIBRARY_PATH=/usr/local/lib/openvino:/usr/local/lib:/usr/local/lib64' \
		'ExecStart=$(PYTHON) $(PROJECT_DIR)/server-whisper-cpp.py --port $(WHISPER_CPP_PORT) --device $(WHISPER_CPP_DEVICE)' \
		'Restart=on-failure' \
		'RestartSec=5' \
		'' \
		'[Install]' \
		'WantedBy=default.target' > $@

$(SYSTEMD_DIR)/push-to-talk.service:
	@mkdir -p $(SYSTEMD_DIR)
	@printf '%s\n' \
		'[Unit]' \
		'Description=Push-to-Talk Voice Dictation' \
		'After=whisper-server.service' \
		'Wants=whisper-server.service' \
		'' \
		'[Service]' \
		'Type=simple' \
		'Environment=XDG_SESSION_TYPE=wayland' \
		'Environment=YDOTOOL_SOCKET=/tmp/.ydotool_socket' \
		'ExecStartPre=/bin/bash -c '"'"'i=0; while [ $$i -lt 60 ]; do curl -sf http://127.0.0.1:5000/health >/dev/null 2>&1 && exit 0; sleep 1; i=$$((i+1)); done; echo whisper-server not ready after 60s; exit 1'"'"'' \
		'ExecStart=$(PYTHON) $(PROJECT_DIR)/push-to-talk.py --key KEY_RIGHTCTRL --backend openvino' \
		'Restart=on-failure' \
		'RestartSec=3' \
		'' \
		'[Install]' \
		'WantedBy=default.target' > $@

# ----------------------------------------------------------------------------
# Service management
# ----------------------------------------------------------------------------

enable: ## Enable services (whisper-server, whisper-cpp-server, push-to-talk)
	systemctl --user enable whisper-server.service
	systemctl --user enable whisper-cpp-server.service
	systemctl --user enable push-to-talk.service

start: ## Start services
	systemctl --user start whisper-server.service
	systemctl --user start whisper-cpp-server.service
	@echo "Waiting for whisper-server to load model..."
	@for i in $$(seq 1 60); do curl -sf http://127.0.0.1:5000/health >/dev/null 2>&1 && break; sleep 1; done
	systemctl --user start push-to-talk.service

stop: ## Stop all services
	-systemctl --user stop push-to-talk.service
	-systemctl --user stop whisper-server.service
	-systemctl --user stop whisper-cpp-server.service

restart: stop start ## Restart all services

status: ## Show service status
	@echo "=== whisper-server (server-native.py :5000) ==="
	@systemctl --user status whisper-server.service --no-pager 2>/dev/null || echo "  not installed"
	@echo ""
	@echo "=== whisper-cpp-server (server-whisper-cpp.py :$(WHISPER_CPP_PORT)) ==="
	@systemctl --user status whisper-cpp-server.service --no-pager 2>/dev/null || echo "  not installed"
	@echo ""
	@echo "=== push-to-talk ==="
	@systemctl --user status push-to-talk.service --no-pager 2>/dev/null || echo "  not installed"
	@echo ""
	@echo "=== ydotoold ==="
	@systemctl status ydotool.service --no-pager 2>/dev/null || echo "  not installed"
	@ls -la /tmp/.ydotool_socket 2>/dev/null || echo "  socket not found at /tmp/.ydotool_socket"

# ----------------------------------------------------------------------------
# Logs
# ----------------------------------------------------------------------------

logs: logs-server logs-cpp logs-ptt ## Show logs for all services

logs-server: ## Show whisper-server logs
	journalctl --user -u whisper-server.service --no-pager -n 30

logs-cpp: ## Show whisper-cpp-server logs
	journalctl --user -u whisper-cpp-server.service --no-pager -n 30

logs-ptt: ## Show push-to-talk logs
	journalctl --user -u push-to-talk.service --no-pager -n 30

# ----------------------------------------------------------------------------
# Health check
# ----------------------------------------------------------------------------

test: ## Health check against running servers
	@echo "--- whisper-server (:5000) ---"
	@curl -sf http://127.0.0.1:5000/models 2>/dev/null | $(PYTHON) -m json.tool || echo "  not reachable"
	@echo ""
	@echo "--- whisper-cpp-server (:$(WHISPER_CPP_PORT)) ---"
	@curl -sf http://127.0.0.1:$(WHISPER_CPP_PORT)/health 2>/dev/null | $(PYTHON) -m json.tool || echo "  not reachable"

# ----------------------------------------------------------------------------
# Uninstall
# ----------------------------------------------------------------------------

uninstall: stop ## Stop and remove all services
	-systemctl --user disable whisper-server.service 2>/dev/null
	-systemctl --user disable whisper-cpp-server.service 2>/dev/null
	-systemctl --user disable push-to-talk.service 2>/dev/null
	rm -f $(SERVICE_FILES)
	systemctl --user daemon-reload

# ----------------------------------------------------------------------------
# Cleanup
# ----------------------------------------------------------------------------

clean: ## Remove downloaded models (destructive, prompts for confirmation)
	@echo "This will delete all models in ~/.whisper/models and ~/.cache/whisper."
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf $(HOME)/.whisper/models
	rm -rf $(HOME)/.cache/whisper

# ----------------------------------------------------------------------------
# GNOME Extension
# ----------------------------------------------------------------------------

EXTENSION_DIR     := $(PROJECT_DIR)/gnome-extension
EXTENSION_UUID    := whisper-npu@dmz.oneill
EXTENSION_INSTALL := $(HOME)/.local/share/gnome-shell/extensions/$(EXTENSION_UUID)
SCHEMA_DIR        := $(HOME)/.local/share/glib-2.0/schemas

extension-install: ## Install GNOME extension (symlink + schemas)
	mkdir -p $(SCHEMA_DIR)
	glib-compile-schemas $(EXTENSION_DIR)/schemas/
	cp $(EXTENSION_DIR)/schemas/*.gschema.xml $(SCHEMA_DIR)/
	glib-compile-schemas $(SCHEMA_DIR)/
	ln -snf $(EXTENSION_DIR) $(EXTENSION_INSTALL)
	@echo "Extension installed. Enable with: make extension-enable"
	@echo "On Wayland, log out and back in to load the extension."

extension-uninstall: extension-disable ## Uninstall GNOME extension
	rm -f $(EXTENSION_INSTALL)
	rm -f $(SCHEMA_DIR)/org.gnome.shell.extensions.whisper-npu.gschema.xml
	glib-compile-schemas $(SCHEMA_DIR)/
	@echo "Extension uninstalled."

extension-enable: ## Enable GNOME extension
	gnome-extensions enable $(EXTENSION_UUID)

extension-disable: ## Disable GNOME extension
	-gnome-extensions disable $(EXTENSION_UUID)

extension-reload: extension-disable extension-enable ## Reload GNOME extension
	@echo "Extension reloaded (JS/CSS changes may require logout/login on Wayland)"

extension-dev: extension-install extension-enable ## Install and enable GNOME extension for development
	@echo ""
	@echo "========================================"
	@echo "Whisper NPU extension installed and enabled!"
	@echo "========================================"
	@echo ""
	@echo "Watch logs: make extension-logs"
	@echo ""
	@echo "Note: JS/CSS changes require logout/login on Wayland."

extension-logs: ## Show GNOME Shell logs (for extension debugging)
	journalctl -f -o cat /usr/bin/gnome-shell
