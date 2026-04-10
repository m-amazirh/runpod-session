FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir huggingface-hub

# Download latest llama.cpp server with CUDA support
RUN curl -L https://github.com/ggerganov/llama.cpp/releases/download/b8748/llama-b8748-bin-ubuntu-x64.tar.gz | \
    tar xz -C /usr/local/bin --strip-components=1 llama-b8748/llama-server

# Create models directory
RUN mkdir -p /models

# Copy startup script
COPY startup.sh /startup.sh
RUN chmod +x /startup.sh

# Expose port
EXPOSE 8080

# Set entrypoint
ENTRYPOINT ["/startup.sh"]
