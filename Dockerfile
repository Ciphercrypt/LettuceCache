# ── Stage 1: Build ─────────────────────────────────────────────────────────
FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libhiredis-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libfaiss-dev \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY CMakeLists.txt .
COPY src/ src/

RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
    && cmake --build build --parallel "$(nproc)"

# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libhiredis0.14 \
    libcurl4 \
    libssl3 \
    libfaiss1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/build/lettucecache ./lettucecache

RUN mkdir -p /data

EXPOSE 8080

ENTRYPOINT ["/app/lettucecache"]
