FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

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
    ninja-build \
    ccache \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /ccache && \
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp && \
    cd /opt/llama.cpp && \
    mkdir build && cd build && \
    cmake .. -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,/usr/local/cuda/lib64" \
      -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,/usr/local/cuda/lib64" \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache && \
    ninja llama-server && \
    cp bin/llama-server /usr/local/bin/ && \
    cp bin/libggml*.so bin/libllama*.so /usr/local/lib/ && \
    echo "/usr/local/lib" >> /etc/ld.so.conf.d/llama-cpp.conf && \
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
