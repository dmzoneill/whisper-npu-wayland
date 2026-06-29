# Whisper NPU Server

Local speech-to-text transcription server that runs entirely on Intel NPU hardware via OpenVINO. Includes a push-to-talk voice dictation client for Wayland desktops.

## Quick Start

```bash
git clone git@github.com:dmzoneill/whisper-npu-wayland.git
cd whisper-npu-wayland
make install        # installs deps, system packages, default model, services, permissions
make start          # starts whisper-server, whisper-cpp-server, push-to-talk
make status         # check service status
make test           # hit health/model endpoints
```

That's it. `make install` downloads the default model (`whisper-small.en-fp16-ov`) automatically. See [Models](#models) below for installing additional models.

## Usage Examples

### Transcribe an audio file

```bash
curl -X POST http://127.0.0.1:5000/transcribe --data-binary @recording.wav
# {"text": "This is the transcribed text."}
```

### Stream transcription (real-time token-by-token via SSE)

```bash
curl -N -X POST http://127.0.0.1:5000/transcribe/stream --data-binary @recording.wav
# data: {"text": "This is "}
# data: {"text": "the transcribed "}
# data: {"text": "text."}
# data: {"done": true, "full_text": "This is the transcribed text."}
```

### Use a specific model

```bash
curl http://127.0.0.1:5000/models
# {"models": ["whisper-small.en-fp16-ov", "whisper-base.en"]}

curl -X POST http://127.0.0.1:5000/transcribe/whisper-base.en --data-binary @recording.wav
```

### Voice dictation (push-to-talk)

The push-to-talk service starts automatically. By default it uses Right Ctrl:

- **Hold** Right Ctrl for >1.5s — records while held, transcribes on release, types into the focused window
- **Tap** Right Ctrl quickly — starts recording with live streaming, tap again to stop

Check its logs with `make logs-ptt`.

## Architecture

```mermaid
graph TB
    subgraph Client Layer
        PTT[push-to-talk.py<br/>Voice Dictation Client]
        CURL[curl / HTTP Client]
    end

    subgraph Server Layer
        SN[server-native.py<br/>OpenVINO GenAI · NPU<br/>Port 5000]
        SC[server-whisper-cpp.py<br/>whisper.cpp · NPU encoder + CPU decoder<br/>Port 5001]
    end

    subgraph Inference Layer
        OV[OpenVINO GenAI<br/>WhisperPipeline<br/>Full NPU]
        WC[libwhisper.so<br/>NPU encoder · CPU decoder]
    end

    subgraph Hardware
        NPU[Intel NPU<br/>/dev/accel/accel0]
    end

    subgraph Models
        OVM[(~/.whisper/models/<br/>OpenVINO Format)]
        GGML[(~/.cache/whisper/<br/>GGML Format)]
    end

    PTT -->|HTTP POST| SN
    CURL -->|HTTP POST| SN
    CURL -->|HTTP POST| SC

    SN --> OV
    SC --> WC

    OV --> NPU
    WC --> NPU

    OV --> OVM
    WC --> GGML

    style SN fill:#2d6a4f,color:#fff
    style SC fill:#40916c,color:#fff
    style PTT fill:#1d3557,color:#fff
    style NPU fill:#e76f51,color:#fff
```

The primary server (`server-native.py`) runs the full Whisper pipeline on the NPU. The secondary server (`server-whisper-cpp.py`) runs the encoder on NPU but the decoder on CPU. Push-to-talk defaults to the primary NPU server.

## Data Flow

### Batch Transcription

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Flask Server
    participant L as librosa
    participant P as WhisperPipeline
    participant N as Intel NPU

    C->>S: POST /transcribe (WAV audio)
    S->>L: Load audio bytes
    L-->>S: float32 array @ 16kHz
    S->>P: pipeline.generate(audio)
    P->>N: Full inference on NPU
    N-->>P: Token sequence
    P-->>S: Transcribed text
    S-->>C: {"text": "..."}
```

### Streaming Transcription

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Flask Server
    participant Q as Queue
    participant T as Inference Thread
    participant N as Intel NPU

    C->>S: POST /transcribe/stream (WAV audio)
    S->>T: Start inference thread
    T->>N: pipeline.generate(audio, streamer=callback)

    loop Token chunks
        N-->>T: Text chunk
        T->>Q: Push chunk
        Q-->>S: Read chunk
        S-->>C: SSE: data: {"text": "chunk"}
    end

    T->>Q: Push None (done)
    S-->>C: SSE: data: {"done": true, "full_text": "..."}
```

### Push-to-Talk Voice Dictation

```mermaid
sequenceDiagram
    participant K as Keyboard (evdev)
    participant PTT as push-to-talk.py
    participant R as parec (PipeWire)
    participant S as Whisper Server :5000
    participant Y as ydotool / wtype

    Note over K,Y: Hold Mode (key held > 1.5s)
    K->>PTT: Key down
    PTT->>R: Start recording
    K->>PTT: Key up
    PTT->>R: Stop recording
    R-->>PTT: WAV audio
    PTT->>S: POST /transcribe
    S-->>PTT: {"text": "..."}
    PTT->>Y: Type text into focused window

    Note over K,Y: Toggle Mode (quick tap < 1.5s)
    K->>PTT: Tap (key down + up)
    PTT->>R: Start recording

    loop Every 3s
        PTT->>S: POST /transcribe/stream (audio so far)
        S-->>PTT: {"text": "partial..."}
        PTT->>Y: Diff and type new text
    end

    K->>PTT: Tap again
    PTT->>R: Stop recording
    PTT->>S: POST /transcribe/stream (final audio)
    S-->>PTT: {"text": "final result"}
    PTT->>Y: Backspace diff + type correction
```

## Servers

### server-native.py (Primary — Full NPU)

OpenVINO GenAI server running the complete Whisper pipeline on the Intel NPU. Supports batch and real-time SSE streaming transcription.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/models` | GET | List available models |
| `/transcribe` | POST | Batch transcription (default model) |
| `/transcribe/<model>` | POST | Batch transcription (named model) |
| `/transcribe/stream` | POST | SSE streaming transcription (default model) |
| `/transcribe/stream/<model>` | POST | SSE streaming transcription (named model) |

Environment variables:
- `WHISPER_DEVICE` — inference device: `NPU`, `CPU`, `GPU` (default: `NPU`)
- `WHISPER_MODEL` — default model name (default: `whisper-small.en-fp16-ov`)

Models are loaded from `~/.whisper/models/` in OpenVINO format.

### server-whisper-cpp.py (Secondary — NPU encoder, CPU decoder)

Whisper.cpp server using ctypes bindings to `libwhisper.so`. The encoder runs on NPU via OpenVINO acceleration, but the decoder runs on CPU.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/transcribe` | POST | Batch transcription |
| `/transcribe/stream` | POST | Transcribe last 30s of audio |
| `/health` | GET | Health check |

```
python3 server-whisper-cpp.py --port 5001 --model ~/.cache/whisper/ggml-base.bin --device NPU
```

Models are GGML format stored in `~/.cache/whisper/`.

## Push-to-Talk Client

Desktop voice dictation client for GNOME/Wayland. Listens for a hotkey via evdev, records audio through PipeWire, sends it to the whisper server on port 5000, and types the result into the focused window.

```
python3 push-to-talk.py --key KEY_RIGHTCTRL --backend openvino --stream-interval 3.0
```

**Two modes of operation:**

- **Hold mode** — hold the key for >1.5 seconds. Audio is recorded while held, transcribed on release, and typed as a single block.
- **Toggle mode** — quick tap (<1.5s). Recording starts immediately with live incremental transcription every few seconds. Tap again to finalize. Uses a diff algorithm to backspace and retype only the changed suffix when the model corrects earlier words.

**Typing backends** (tried in order on Wayland): `ydotool` → `wtype` → `wl-copy` (clipboard fallback). On X11: `xdotool`.

## Service Architecture

```mermaid
graph LR
    subgraph "systemd --user"
        WS[whisper-server.service<br/>server-native.py :5000]
        WC[whisper-cpp-server.service<br/>server-whisper-cpp.py :5001]
        PT[push-to-talk.service]
    end

    PT -->|Wants=| WS

    style WS fill:#2d6a4f,color:#fff
    style WC fill:#40916c,color:#fff
    style PT fill:#1d3557,color:#fff
```

`push-to-talk` depends on `whisper-server` (the full NPU server) by default.

## Installation

### Prerequisites

- Linux (Fedora) with Intel Core Ultra processor (NPU)
- Device access: `/dev/accel/accel0` (NPU)
- Python 3

### What `make install` Does

1. **Python packages** — installs from `requirements.txt` (openvino, flask, librosa, aiohttp, evdev)
2. **System packages** — `ydotool`, `pipewire-pulseaudio`, `wtype`, `wl-clipboard`, `xdotool`
3. **Permissions** — adds your user to the `input` group (for evdev keyboard access)
4. **Default model** — downloads `whisper-small.en-fp16-ov` from HuggingFace if not already present
5. **Services** — generates and installs three systemd user service files
6. **Enable** — enables services to start on login

### Makefile Targets

| Target | Description |
|--------|-------------|
| `make install` | Full install: deps + model + services + permissions |
| `make install-models` | Download the default OpenVINO model |
| `make start` | Start all services |
| `make stop` | Stop all services |
| `make restart` | Restart all services |
| `make status` | Show service and ydotoold status |
| `make logs` | Show recent logs for all services |
| `make logs-server` | Show whisper-server logs |
| `make logs-cpp` | Show whisper-cpp-server logs |
| `make logs-ptt` | Show push-to-talk logs |
| `make test` | Health checks against running servers |
| `make uninstall` | Stop, disable, and remove services |
| `make clean` | Remove downloaded models (prompts first) |

### Configuration

Override defaults at install time:

```bash
make install WHISPER_DEVICE=CPU WHISPER_MODEL=whisper-base.en
```

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_DEVICE` | `NPU` | OpenVINO device for server-native |
| `WHISPER_MODEL` | `whisper-small.en-fp16-ov` | Default model for server-native |
| `WHISPER_CPP_DEVICE` | `NPU` | OpenVINO encoder device for whisper.cpp |
| `WHISPER_CPP_PORT` | `5001` | Port for whisper.cpp server |

## Models

Stored in `~/.whisper/models/` in OpenVINO format. The default model is downloaded automatically by `make install`.

To install additional models:

```bash
cd ~/.whisper/models
for model in whisper-small.en-fp16-ov whisper-base.en whisper-tiny.en; do
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/OpenVINO/$model
    cd $model && git lfs pull && cd ..
done
```

Available models: `whisper-tiny`, `whisper-base`, `whisper-small`, `whisper-medium`, `whisper-large-v3` (and `.en` variants).

## API Reference

### Batch Transcription

```bash
curl -X POST http://127.0.0.1:5000/transcribe \
    --data-binary @audio.wav
```

```json
{"text": "The transcribed text appears here."}
```

### Streaming Transcription (SSE)

```bash
curl -N -X POST http://127.0.0.1:5000/transcribe/stream \
    --data-binary @audio.wav
```

```
data: {"text": "The "}
data: {"text": "transcribed "}
data: {"text": "text "}
data: {"done": true, "full_text": "The transcribed text appears here."}
```

### List Models

```bash
curl http://127.0.0.1:5000/models
```

```json
{"models": ["whisper-small.en-fp16-ov", "whisper-base.en"]}
```

## Dependencies

### Python

| Package | Version | Used By |
|---------|---------|---------|
| openvino | >= 2025.4.0 | server-native.py |
| openvino-genai | >= 2025.4.0 | server-native.py |
| openvino-tokenizers | >= 2025.4.0 | server-native.py |
| librosa | >= 0.10.2 | server-native.py, server-whisper-cpp.py |
| flask | >= 3.1.0 | server-native.py, server-whisper-cpp.py |
| aiohttp | >= 3.9.0 | push-to-talk.py |
| evdev | >= 1.9.0 | push-to-talk.py |

### System

| Package | Purpose |
|---------|---------|
| `ydotool` | Type text into focused window (Wayland) |
| `wtype` | Fallback Wayland text input |
| `wl-clipboard` | Clipboard fallback (`wl-copy`) |
| `xdotool` | X11 text input fallback |
| `pipewire-pulseaudio` | Audio recording via `parec` |

### Native Libraries

| Library | Used By |
|---------|---------|
| `libwhisper.so` | server-whisper-cpp.py (ctypes) |
| OpenVINO runtime | server-native.py |

## Hardware

Tested on Lenovo ThinkPad P1 Gen 7 with Intel Core Ultra 7 165H (Meteor Lake NPU).

Required device files:
- `/dev/accel/accel0` — Intel NPU
