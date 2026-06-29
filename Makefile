SHELL := /bin/bash

PROJECT_DIR  := $(CURDIR)
SYSTEMD_DIR  := $(HOME)/.config/systemd/user
PYTHON       := /usr/bin/python3

WHISPER_DEVICE     ?= NPU
WHISPER_MODEL      ?= whisper-small.en-fp16-ov
WHISPER_CPP_PORT   ?= 5001
WHISPER_CPP_DEVICE ?= NPU

SYSTEM_PKGS := ydotool pipewire-pulseaudio wtype wl-clipboard xdotool

SERVICE_FILES := $(SYSTEMD_DIR)/whisper-server.service \
                 $(SYSTEMD_DIR)/whisper-legacy.service \
                 $(SYSTEMD_DIR)/whisper-cpp-server.service \
                 $(SYSTEMD_DIR)/push-to-talk.service

WHISPER_MODELS_DIR := $(HOME)/.whisper/models
HF_ORG := mecattaf

.PHONY: help install install-python install-system install-services \
        install-permissions install-models enable start stop restart status \
        logs logs-server logs-cpp logs-ptt \
        test uninstall clean

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
	@echo "  WHISPER_DEVICE       Device for server-native    [$(WHISPER_DEVICE)]"
	@echo "  WHISPER_MODEL        Model for server-native     [$(WHISPER_MODEL)]"
	@echo "  WHISPER_CPP_DEVICE   Device for whisper.cpp      [$(WHISPER_CPP_DEVICE)]"
	@echo "  WHISPER_CPP_PORT     Port for whisper.cpp        [$(WHISPER_CPP_PORT)]"
	@echo ""
	@echo "whisper-server and whisper-legacy both bind port 5000; only run one."

# ----------------------------------------------------------------------------
# Full install
# ----------------------------------------------------------------------------

install: install-python install-system install-permissions install-models install-services enable ## Install everything

# ----------------------------------------------------------------------------
# Python dependencies
# ----------------------------------------------------------------------------

install-python: ## Install Python packages
	$(PYTHON) -m pip install --user -r requirements.txt
	$(PYTHON) -m pip install --user aiohttp evdev

# ----------------------------------------------------------------------------
# System packages
# ----------------------------------------------------------------------------

install-system: ## Install system packages via dnf (requires sudo)
	sudo dnf install -y $(SYSTEM_PKGS)

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

install-services: $(SERVICE_FILES) ## Install systemd user service files
	systemctl --user daemon-reload

$(SYSTEMD_DIR)/whisper-server.service:
	@mkdir -p $(SYSTEMD_DIR)
	@printf '%s\n' \
		'[Unit]' \
		'Description=Whisper Speech-to-Text Server (OpenVINO GenAI)' \
		'After=basic.target' \
		'Conflicts=whisper-legacy.service' \
		'' \
		'[Service]' \
		'Type=simple' \
		'WorkingDirectory=$(PROJECT_DIR)' \
		'ExecStart=$(PYTHON) $(PROJECT_DIR)/server-native.py' \
		'Environment=WHISPER_DEVICE=$(WHISPER_DEVICE)' \
		'Environment=WHISPER_MODEL=$(WHISPER_MODEL)' \
		'Restart=on-failure' \
		'RestartSec=5' \
		'' \
		'[Install]' \
		'WantedBy=default.target' > $@

$(SYSTEMD_DIR)/whisper-legacy.service:
	@mkdir -p $(SYSTEMD_DIR)
	@printf '%s\n' \
		'[Unit]' \
		'Description=Whisper Speech-to-Text Server (Legacy)' \
		'After=basic.target' \
		'Conflicts=whisper-server.service' \
		'' \
		'[Service]' \
		'Type=simple' \
		'WorkingDirectory=$(PROJECT_DIR)' \
		'ExecStart=$(PYTHON) $(PROJECT_DIR)/server.py' \
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
		'After=whisper-cpp-server.service' \
		'Wants=whisper-cpp-server.service' \
		'' \
		'[Service]' \
		'Type=simple' \
		'Environment=XDG_SESSION_TYPE=wayland' \
		'ExecStart=$(PYTHON) $(PROJECT_DIR)/push-to-talk.py --key KEY_RIGHTCTRL --backend whisper-cpp' \
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
	systemctl --user start push-to-talk.service

stop: ## Stop all services
	-systemctl --user stop push-to-talk.service
	-systemctl --user stop whisper-server.service
	-systemctl --user stop whisper-legacy.service
	-systemctl --user stop whisper-cpp-server.service

restart: stop start ## Restart all services

status: ## Show service status
	@echo "=== whisper-server (server-native.py :5000) ==="
	@systemctl --user status whisper-server.service --no-pager 2>/dev/null || echo "  not installed"
	@echo ""
	@echo "=== whisper-legacy (server.py :5000) ==="
	@systemctl --user status whisper-legacy.service --no-pager 2>/dev/null || echo "  not installed"
	@echo ""
	@echo "=== whisper-cpp-server (server-whisper-cpp.py :$(WHISPER_CPP_PORT)) ==="
	@systemctl --user status whisper-cpp-server.service --no-pager 2>/dev/null || echo "  not installed"
	@echo ""
	@echo "=== push-to-talk ==="
	@systemctl --user status push-to-talk.service --no-pager 2>/dev/null || echo "  not installed"
	@echo ""
	@echo "=== ydotoold ==="
	@pgrep -a ydotoold || echo "  not running"

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
	-systemctl --user disable whisper-legacy.service 2>/dev/null
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
