//! training/export_onnx.py 가 만든 모델을 ONNX Runtime(ort 크레이트)으로 실행한다.
//! 계약: 입력 "X" [1, 2] float32 (거리 km, 대기 주문 수) → 출력 "variable" [1, 1] float32 (분).
//! 정규화는 모델 안에 있으므로 원래 단위 값을 그대로 넣는다.

use std::path::Path;
use std::sync::Mutex;

use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;

use crate::eta::Predictor;

pub struct OnnxPredictor {
    // Session::run 은 &mut self 를 받는다. 모델이 작아 실행이 짧으므로 Mutex 하나로 충분하다.
    session: Mutex<Session>,
}

impl OnnxPredictor {
    pub fn new(env: &Environment, model_path: &Path) -> Result<Self, String> {
        let session = Session::builder(env)
            .and_then(|mut b| b.commit_from_file(model_path))
            .map_err(|e| format!("모델 로드 실패 ({}): {e}", model_path.display()))?;
        Ok(Self { session: Mutex::new(session) })
    }
}

impl Predictor for OnnxPredictor {
    fn predict(&self, distance_km: f32, pending_orders: f32) -> Result<f32, String> {
        let input = Tensor::from_array(([1usize, 2], vec![distance_km, pending_orders])).map_err(|e| e.to_string())?;
        let mut session = self.session.lock().map_err(|_| "세션 잠금 실패 (poisoned)".to_string())?;
        let outputs = session.run(ort::inputs!["X" => input]).map_err(|e| e.to_string())?;
        let (shape, data) = outputs["variable"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
        match data {
            [minutes] => Ok(*minutes),
            _ => Err(format!("출력 모양이 [1, 1] 이 아니다: {shape:?}")),
        }
    }
}
