# delivery-eta Go 서비스. 빌드 컨텍스트는 저장소 루트다 (uv.lock, artifacts/ 를 읽는다).
#   docker build -f api-go/Dockerfile -t eta-go .
#   docker run --rm -p 8081:8080 eta-go
# 먼저 artifacts/model_sklearn.onnx 가 있어야 한다 (uv run --group train python -m training.export_onnx).

ARG GO_IMAGE=golang:1.25-bookworm

# ── 빌드: onnxruntime_go 는 cgo 를 쓴다. ONNX Runtime 은 실행 시 dlopen 하므로 빌드에는 필요 없다 ──
FROM ${GO_IMAGE} AS build
WORKDIR /src
COPY api-go/go.mod api-go/go.sum ./
RUN go mod download
COPY api-go/*.go ./
RUN CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o /out/eta-go .

# ── ONNX Runtime: Python 쪽과 같은 버전을 uv.lock 에서 읽어 받는다 ──
FROM ${GO_IMAGE} AS ort
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
COPY --from=build /out/eta-go /usr/local/bin/eta-go
COPY artifacts/model_sklearn.onnx /app/model_sklearn.onnx
ENV ORT_LIB=/opt/onnxruntime/lib/libonnxruntime.so \
    MODEL_PATH=/app/model_sklearn.onnx \
    PORT=8080
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/eta-go"]
