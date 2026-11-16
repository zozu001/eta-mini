//! delivery-eta 의 Rust 구현. api/app.py 와 같은 계약(POST /eta, GET /health)을 ONNX 모델로 서빙한다.
//!
//! 환경 변수:
//!   PORT       듣는 포트 (기본 8080)
//!   MODEL_PATH ONNX 모델 경로 (기본 artifacts/model_sklearn.onnx)
//!   ORT_LIB    ONNX Runtime 공유 라이브러리 경로 (기본 /opt/onnxruntime/lib/libonnxruntime.so)
//!              ort 의 load-dynamic 기능으로 실행 시 이 파일을 연다. ort 가 런타임을 내려받지 않는다.

mod eta;
mod predictor;

use std::path::PathBuf;
use std::process::ExitCode;

use eta::Predictor;
use predictor::OnnxPredictor;

/// /health, /eta 응답의 model 값. Python 의 onnx 구현체와 같은 모델 파일이라 같은 이름을 쓴다.
const MODEL_NAME: &str = "onnx";

fn getenv(key: &str, default: &str) -> String {
    std::env::var(key).ok().filter(|v| !v.is_empty()).unwrap_or_else(|| default.to_string())
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), String> {
    let ort_lib = PathBuf::from(getenv("ORT_LIB", "/opt/onnxruntime/lib/libonnxruntime.so"));
    let model_path = PathBuf::from(getenv("MODEL_PATH", "artifacts/model_sklearn.onnx"));
    let port = getenv("PORT", "8080");

    // 환경은 프로세스에 하나이며 세션보다 오래 살아야 한다.
    let env = ort::init_from(&ort_lib)
        .map_err(|e| format!("ONNX Runtime 불러오기 실패 ({}): {e}", ort_lib.display()))?
        .with_name("eta-rust")
        .build()
        .map_err(|e| format!("ONNX Runtime 초기화 실패: {e}"))?;

    let p = OnnxPredictor::new(&env, &model_path)?;
    // 모델 입출력 이름·모양이 계약과 다르면 요청을 받기 전에 실패한다.
    p.predict(3.0, 2.0).map_err(|e| format!("모델 시험 실행 실패: {e}"))?;

    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| format!("{addr} 에서 듣기 실패: {e}"))?;
    println!("listening on {addr} (model={MODEL_NAME})");
    axum::serve(listener, eta::router(Box::new(p), MODEL_NAME))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| e.to_string())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
}
