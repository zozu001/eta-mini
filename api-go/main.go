// delivery-eta 의 Go 구현. api/app.py 와 같은 계약(POST /eta, GET /health)을 ONNX 모델로 서빙한다.
//
// 환경 변수:
//
//	PORT       듣는 포트 (기본 8080)
//	MODEL_PATH ONNX 모델 경로 (기본 artifacts/model_sklearn.onnx)
//	ORT_LIB    ONNX Runtime 공유 라이브러리 경로 (기본 /opt/onnxruntime/lib/libonnxruntime.so)
package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

// /health, /eta 응답의 model 값. Python 의 onnx 구현체와 같은 모델 파일이라 같은 이름을 쓴다.
const modelName = "onnx"

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func main() {
	p, err := newONNXPredictor(
		getenv("ORT_LIB", "/opt/onnxruntime/lib/libonnxruntime.so"),
		getenv("MODEL_PATH", "artifacts/model_sklearn.onnx"),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer p.Close()
	// 모델 입출력 이름·모양이 계약과 다르면 요청을 받기 전에 실패한다.
	if _, err := p.Predict(3, 2); err != nil {
		log.Fatalf("모델 시험 실행 실패: %v", err)
	}

	srv := &http.Server{
		Addr:              ":" + getenv("PORT", "8080"),
		Handler:           NewHandler(p, modelName),
		ReadHeaderTimeout: 5 * time.Second,
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()

	log.Printf("listening on %s (model=%s)", srv.Addr, modelName)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}
