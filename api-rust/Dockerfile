# delivery-eta Rust 서비스. 빌드 컨텍스트는 저장소 루트다 (uv.lock, artifacts/ 를 읽는다).
#   docker build -f api-rust/Dockerfile -t eta-rust .
#   docker run --rm -p 8082:8080 eta-rust
# 먼저 artifacts/model_sklearn.onnx 가 있어야 한다 (uv run --group train python -m training.export_onnx).

ARG RUST_IMAGE=rust:1.100-bookworm

# ── 빌드: ort 는 load-dynamic 이라 ONNX Runtime 없이 빌드된다 (실행 시 ORT_LIB 를 dlopen) ──
FROM ${RUST_IMAGE} AS build
WORKDIR /src
COPY api-rust/Cargo.toml api-rust/Cargo.lock ./
# 의존성만 먼저 빌드해 캐시한다 (소스만 바뀌면 이 층을 다시 쓰지 않는다)
RUN mkdir src && echo 'fn main() {}' > src/main.rs \
 && cargo build --release --locked \
 && rm -rf src
COPY api-rust/src ./src
RUN touch src/main.rs \
 && cargo test --release --locked \
 && cargo build --release --locked \
 && cp target/release/eta-rust /eta-rust

# ── ONNX Runtime: Python 쪽과 같은 버전을 uv.lock 에서 읽어 받는다 ──
FROM ${RUST_IMAGE} AS ort
ARG TARGETARCH
COPY uv.lock /tmp/uv.lock
RUN set -eu; \
    version=$(awk -F'"' '/^name = /{ hit = ($2 == "onnxruntime") } hit && /^version = /{ print $2; exit }' /tmp/uv.lock); \
    [ -n "$version" ] || { echo "uv.lock 에서 onnxruntime 버전을 찾지 못했다" >&2; exit 1; }; \
    case "${TARGETARCH:-}" in \
      amd64) arch=x64 ;; \
      arm64) arch=aarch64 ;; \
      *) echo "지원하지 않는 TARGETARCH: '${TARGETARCH:-}' (amd64, arm64 만 지원)" >&2; exit 1 ;; \
    esac; \
    name="onnxruntime-linux-${arch}-${version}"; \
    echo "ONNX Runtime ${version} (${arch})"; \
    curl -fsSL --retry 3 -o /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${version}/${name}.tgz"; \
    tar -xzf /tmp/ort.tgz -C /tmp; \
    mkdir -p /opt/onnxruntime; \
    cp -a "/tmp/${name}/lib" /opt/onnxruntime/lib; \
    test -e /opt/onnxruntime/lib/libonnxruntime.so

# ── 실행 ──
FROM debian:bookworm-slim
COPY --from=ort /opt/onnxruntime/lib /opt/onnxruntime/lib
COPY --from=build /eta-rust /usr/local/bin/eta-rust
COPY artifacts/model_sklearn.onnx /app/model_sklearn.onnx
ENV ORT_LIB=/opt/onnxruntime/lib/libonnxruntime.so \
    MODEL_PATH=/app/model_sklearn.onnx \
    PORT=8080
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/eta-rust"]
