# GPU Session CLI

A command-line tool to provision 48GB+ VRAM GPUs on RunPod Secure Cloud and launch SGLang-powered OpenAI-compatible inference servers using HuggingFace models (FP8 checkpoints for optimal efficiency).

## Installation

```bash
pip install gpu-session
uv tool install gpu-session
git clone https://github.com/your-username/gpu-session
cd gpu-session
pip install -e .
```

## Setup

1. Get a RunPod API key from https://www.runpod.io/console/api
2. Set the environment variable:
```bash
export RUNPOD_API_KEY='your-api-key'
```

## Usage

### Start a session
```bash
gpu-session start \
  --model "Qwen/Qwen3.6-27B-FP8" \
  --api-key "my-secret-key" \
  --context-length 262144 \
  --max-running-requests 2 \
  --idle-timeout 30
```

### Stop a session
```bash
gpu-session stop
```

### Check status
```bash
gpu-session status
```

### List available GPUs
```bash
gpu-session list-gpus
```

### Dry run
```bash
gpu-session start --model "Qwen/Qwen3.6-27B-FP8" --api-key "my-secret-key" --dry-run
```

## Configuration
Optional config file at `~/.gpu-session/config.toml`:
```toml
[defaults]
context_length = 262144
idle_timeout = 0
preferred_gpus = ["NVIDIA RTX PRO 6000 Blackwell"]

[runpod]
cloud = "SECURE"
```

## Environment Variables (passed to pod)

| Variable | Required | Description |
|---|---|---|
| `MODEL_REPO` | Yes | HuggingFace repo ID (e.g. `Qwen/Qwen3.6-27B-FP8`) |
| `API_KEY` | Yes | Server API key |
| `CTX_LEN` | No | Context length (default: 262144) |
| `MAX_RUNNING_REQUESTS` | No | Concurrent requests (default: 2) |
| `PORT` | No | Server port (default: 8080) |
| `HF_TOKEN` | No | HuggingFace token for gated models |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` | No | Allow extended context (default: 1) |

## Target Hardware

- **GPU**: NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM)
- **Model**: Qwen3.6-27B FP8 (28B params at ~1 byte/param)
- **Memory**: ~60 GB total for 2 concurrent 262K-context sessions
- **Architecture**: Hybrid Gated DeltaNet — 48 GDN layers + 16 attention layers

## Building the Docker Image
```bash
cd docker
docker build -t ghcr.io/your-username/gpu-session-runtime:latest .
docker push ghcr.io/your-username/gpu-session-runtime:latest
```
Update `DEFAULT_DOCKER_IMAGE` in `gpu_session/cli.py` to point to your image.

## License
MIT
