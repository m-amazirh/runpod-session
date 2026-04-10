FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir huggingface-hub

# Build llama.cpp with CUDA support
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp && \
    cd /opt/llama.cpp && \
    mkdir build && cd build && \
    cmake .. -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON && \
    make -j$(nproc) llama-server && \
    cp llama-server /usr/local/bin/ && \
    cp libggml*.so libllama*.so /usr/local/lib/ && \
    ldconfig

# Create models directory
RUN mkdir -p /models

# Copy startup script
COPY startup.sh /startup.sh
RUN chmod +x /startup.sh

# Expose port
EXPOSE 8080

# Set entrypoint
ENTRYPOINT ["/startup.sh"]
