FROM ghcr.io/ggml-org/llama.cpp:server-cuda

# Create models directory
RUN mkdir -p /models

# Copy startup script
COPY startup.sh /startup.sh
RUN chmod +x /startup.sh

# Expose port
EXPOSE 8080

# Set entrypoint
ENTRYPOINT ["/startup.sh"]
