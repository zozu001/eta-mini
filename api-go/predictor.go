package main

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// onnxPredictor 는 training/export_onnx.py 가 만든 모델을 실행한다.
// 계약: 입력 "X" [1, 2] float32 (거리 km, 대기 주문 수) → 출력 "variable" [1, 1] float32 (분).
// 정규화는 모델 안에 있으므로 원래 단위 값을 그대로 넣는다.
type onnxPredictor struct {
	session *ort.DynamicAdvancedSession
}

// newONNXPredictor 는 ONNX Runtime 공유 라이브러리를 불러오고 세션을 만든다.
// DynamicAdvancedSession 은 호출마다 텐서를 받으므로 여러 요청에서 동시에 Run 해도 안전하다.
func newONNXPredictor(libPath, modelPath string) (*onnxPredictor, error) {
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("ONNX Runtime 초기화 실패 (%s): %w", libPath, err)
	}
	s, err := ort.NewDynamicAdvancedSession(modelPath, []string{"X"}, []string{"variable"}, nil)
	if err != nil {
		return nil, fmt.Errorf("모델 로드 실패 (%s): %w", modelPath, err)
	}
	return &onnxPredictor{session: s}, nil
}

func (p *onnxPredictor) Predict(distanceKm, pendingOrders float32) (float32, error) {
	in, err := ort.NewTensor(ort.NewShape(1, 2), []float32{distanceKm, pendingOrders})
	if err != nil {
		return 0, err
	}
	defer in.Destroy()
	out, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1))
	if err != nil {
		return 0, err
	}
	defer out.Destroy()
	if err := p.session.Run([]ort.Value{in}, []ort.Value{out}); err != nil {
		return 0, err
	}
	return out.GetData()[0], nil
}

func (p *onnxPredictor) Close() {
	p.session.Destroy()
	ort.DestroyEnvironment()
}
