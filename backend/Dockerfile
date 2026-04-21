FROM python:3.11-slim AS builder

ENV PATH="/opt/venv/bin:$PATH"
RUN python -m venv /opt/venv

# Install build dependencies for liblc3
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    meson \
    ninja-build \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Build liblc3 and create wheel
WORKDIR /tmp
RUN git clone https://github.com/google/liblc3.git && \
    cd liblc3 && \
    meson setup build && \
    cd build && \
    meson install && \
    ldconfig && \
    # Stage shared libs to a fixed path (meson install path varies by arch)
    mkdir -p /opt/liblc3 && \
    cp /usr/local/lib/*/liblc3.so* /opt/liblc3/ 2>/dev/null || \
    cp /usr/local/lib/liblc3.so* /opt/liblc3/ && \
    cd /tmp/liblc3 && \
    python3 -m pip wheel --no-cache-dir --wheel-dir /tmp/wheels .

# Install Python requirements
WORKDIR /opt/venv
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

FROM python:3.11-slim

WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH"
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Trust the pendant-stack local CA for DNS-redirected hosts (e.g. api.deepgram.com).
# The CA file is mounted by docker-compose at /usr/local/share/ca-certificates/pendant-ca.crt.
# Runtime update-ca-certificates (via entrypoint) picks it up because the mount isn't
# available at image-build time.
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN apt-get update && apt-get -y install ffmpeg curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy compiled liblc3 library from builder staging dir
COPY --from=builder /opt/liblc3/ /usr/local/lib/
COPY --from=builder /tmp/wheels /tmp/wheels

# Install liblc3 Python package and set library path
RUN ldconfig && \
    pip install --no-cache-dir /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

COPY --from=builder /opt/venv /opt/venv
COPY backend/ .

COPY backend/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--loop", "uvloop"]
