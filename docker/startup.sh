#!/bin/bash
set -e

echo "=== GPU Session Startup ==="
echo "Model: ${MODEL_REPO}"
echo "Filename: ${MODEL_FILENAME}"
echo "Engine: ${ENGINE}"
echo "Context Length: ${CTX_LEN}"
echo "API Key: [set]"

# Download model from HuggingFace using direct URL
echo "Downloading model..."
MODEL_URL="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILENAME}"

if [ -n "$HF_TOKEN" ]; then
    echo "Using HF_TOKEN for authentication..."
    curl -L -o "/models/${MODEL_FILENAME}" "${MODEL_URL}" \
        -H "Authorization: Bearer ${HF_TOKEN}"
else
    curl -L -o "/models/${MODEL_FILENAME}" "${MODEL_URL}"
fi

if [ ! -f "/models/${MODEL_FILENAME}" ]; then
    echo "ERROR: Model file not found after download"
    exit 1
fi

echo "Model downloaded: /models/${MODEL_FILENAME}"

# Start inference server based on engine
case "${ENGINE}" in
    "llama-cpp")
        echo "Starting llama.cpp server..."
        exec /app/llama-server \
            -m "/models/${MODEL_FILENAME}" \
            -c "${CTX_LEN}" \
            -ngl 99 \
            --host 0.0.0.0 \
            --port 8080 \
            --api-key "${API_KEY}" \
            --ctx-size "${CTX_LEN}"
        ;;
    "vllm")
        echo "Starting vLLM server..."
        # Install vLLM if not present
        pip3 install vllm --quiet
        exec vllm serve "/models/${MODEL_FILENAME}" \
            --host 0.0.0.0 \
            --port 8080 \
            --api-key "${API_KEY}" \
            --max-model-len "${CTX_LEN}"
        ;;
    *)
        echo "ERROR: Unknown engine: ${ENGINE}"
        exit 1
        ;;
esac
