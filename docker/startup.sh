#!/bin/bash
set -euo pipefail

# ------------------------------------------------------------------------------
# GPU Session Startup (SGLang)
# Serves Qwen3.6-27B via OpenAI-compatible API using SGLang
# ------------------------------------------------------------------------------

echo "=== GPU Session Startup (SGLang) ==="

# --- Required environment variables -------------------------------------------
: "${MODEL_REPO:?ERROR: MODEL_REPO is required (e.g. Qwen/Qwen3.6-27B-FP8)}"
: "${API_KEY:?ERROR: API_KEY is required}"

# --- Configurable defaults ----------------------------------------------------
CTX_LEN="${CTX_LEN:-262144}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"
PORT="${PORT:-8080}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
ENABLE_MTP="${ENABLE_MTP:-true}"
ENABLE_METRICS="${ENABLE_METRICS:-true}"
ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-false}"
HF_TOKEN="${HF_TOKEN:-}"
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}"

# --- Startup diagnostics (never print API key) --------------------------------
echo "Model:                ${MODEL_REPO}"
echo "Context Length:       ${CTX_LEN}"
echo "Max Running Requests: ${MAX_RUNNING_REQUESTS}"
echo "Port:                 ${PORT}"
echo "Memory Fraction:      ${MEM_FRACTION}"
echo "Chunked Prefill:      ${CHUNKED_PREFILL_SIZE}"
echo "MTP (NEXTN):          ${ENABLE_MTP}"
echo "Torch Compile:        ${ENABLE_TORCH_COMPILE}"
echo "API Key:              [set]"
echo "HF Token:             [$(if [ -n "$HF_TOKEN" ]; then echo 'set'; else echo 'not set'; fi)]"

# --- HuggingFace authentication (reuses existing HF_TOKEN variable) -----------
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
fi

# --- Build argument array -----------------------------------------------------
ARGS=(
    --model-path "${MODEL_REPO}"
    --host 0.0.0.0
    --port "${PORT}"
    --api-key "${API_KEY}"
    --context-length "${CTX_LEN}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --mamba-scheduler-strategy extra_buffer
    --page-size 64
    --cuda-graph-bs 1 2
    --mem-fraction-static "${MEM_FRACTION}"
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
    --served-model-name "${MODEL_REPO}"
    --trust-remote-code
)

# --- MTP speculative decoding (Qwen3.6-27B has native MTP heads) --------------
if [ "${ENABLE_MTP}" = "true" ]; then
    ARGS+=(
        --speculative-algorithm NEXTN
        --speculative-num-steps 3
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens 4
    )
fi

# --- Metrics ------------------------------------------------------------------
if [ "${ENABLE_METRICS}" = "true" ]; then
    ARGS+=(--enable-metrics)
fi

# --- Torch compile (optional, may improve throughput on Blackwell) ------------
if [ "${ENABLE_TORCH_COMPILE}" = "true" ]; then
    ARGS+=(--enable-torch-compile --torch-compile-max-bs "${MAX_RUNNING_REQUESTS}")
fi

# --- Launch -------------------------------------------------------------------
echo ""
echo "Launching SGLang server..."
exec python3 -m sglang.launch_server "${ARGS[@]}"
