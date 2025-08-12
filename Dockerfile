FROM debian:bookworm-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH="/home/benchmarkuser/.local/bin:${PATH}" \
    WORKSPACE_DIR="/workspace"

# Install base utilities & dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    git \
    curl \
    wget \
    vim \
    nano \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    unzip \
    zip \
    tar \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for safety
RUN useradd -m -s /bin/bash benchmarkuser

# Create workspace directory and set permissions
RUN mkdir -p ${WORKSPACE_DIR} && chown -R benchmarkuser:benchmarkuser ${WORKSPACE_DIR}

# Switch to non-root user
USER benchmarkuser

# Set working directory
WORKDIR ${WORKSPACE_DIR}

# Default command
CMD ["/bin/bash"]