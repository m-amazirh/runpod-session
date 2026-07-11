# GPU Session Docker Image (SGLang)

Docker image for serving Qwen3.6-27B-FP8 via SGLang on RunPod (RTX PRO 6000 Blackwell, 96 GB).

## Base Image

`lmsysorg/sglang:latest` — CUDA 13, FlashInfer, PyTorch, SGLang pre-installed.

## Build

```bash
cd docker
docker build -t ghcr.io/your-username/gpu-session-runtime:latest .
docker push ghcr.io/your-username/gpu-session-runtime:latest
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_REPO` | *(required)* | HF repo ID, e.g. `Qwen/Qwen3.6-27B-FP8` |
| `API_KEY` | *(required)* | Server API key |
| `CTX_LEN` | `262144` | Maximum context length per request |
| `MAX_RUNNING_REQUESTS` | `2` | Concurrent conversation limit |
| `PORT` | `8080` | Server port |
| `MEM_FRACTION` | `0.90` | GPU memory fraction for KV cache |
| `CHUNKED_PREFILL_SIZE` | `8192` | Tokens per prefill chunk |
| `ENABLE_MTP` | `true` | Multi-Token Prediction (native in Qwen3.6-27B) |
| `ENABLE_METRICS` | `true` | Prometheus metrics |
| `ENABLE_TORCH_COMPILE` | `false` | torch.compile optimization |
| `HF_TOKEN` | *(optional)* | HuggingFace token for gated models |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` | `1` | Allow context-length override |

## Architecture Notes

- Qwen3.6-27B has a hybrid Gated DeltaNet architecture (48 GDN layers + 16 attention layers)
- The FP8 checkpoint (`Qwen/Qwen3.6-27B-FP8`) reduces weight memory to ~28 GB
- KV cache only spans 16 attention layers; two 262K-context sessions fit in ~60 GB total
- MTP speculative decoding is built into the checkpoint (no separate draft model)
- Mamba scheduler uses `extra_buffer` strategy with `page-size 64` (FLA kernel backend)

## Health Check

- `GET /health` → `{"status": "OK"}`
- `GET /health_generate` → thorough check (generates a token)

## API

OpenAI-compatible at `http://<host>:<port>/v1`
