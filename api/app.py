import html
import importlib
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

MODEL_IMPL = os.getenv("MODEL_IMPL", "rule")
estimate_delivery_minutes = importlib.import_module(f"api.predictors.{MODEL_IMPL}").estimate_delivery_minutes

class EtaRequest(BaseModel):
    """배달 예상 시간 요청"""
    # 모델이 더 많은 정보를 원하면 여기에 필드를 추가
    distance_km: float = Field(ge=0, description="가게에서 배달지까지 거리(km)")
    pending_orders: int = Field(ge=0, description="지금 밀려 있는 주문 수")

class EtaResponse(BaseModel):
    """배달 예상 시간 응답"""
    minutes: float
    model: str  # 어떤 구현체가 낸 예측인지 (장애 추적용)

APP_VERSION = "v0.2"

app = FastAPI(title="delivery-eta", version=APP_VERSION)

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """API 소개 페이지를 HTML로 돌려준다."""
    model = html.escape(MODEL_IMPL)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>delivery-eta API</title>
</head>
<body style="font-family: system-ui, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; line-height: 1.6;">
  <h1>🛵 delivery-eta API</h1>
  <p>거리와 밀린 주문 수로 배달 예상 시간(분)을 계산하는 API입니다.</p>
  <p>버전: <b>{APP_VERSION}</b> · 모델: <b>{model}</b></p>
  <h2>엔드포인트</h2>
  <ul>
    <li><code>GET /health</code> — 서비스 상태와 현재 모델</li>
    <li><code>POST /eta</code> — 배달 예상 시간 계산<br>
      요청 예: <code>{{"distance_km": 2, "pending_orders": 3}}</code><br>
      응답 예: <code>{{"minutes": 22.0, "model": "{model}"}}</code></li>
    <li><a href="/docs">/docs</a> — Swagger 문서에서 직접 호출해 보기</li>
  </ul>
</body>
</html>"""

@app.get("/health")
def health() -> dict[str, str]:
    """서비스 상태와 현재 구현체 이름을 돌려준다."""
    return {"status": "ok", "model": MODEL_IMPL}

@app.post("/eta", response_model=EtaResponse)
def eta(req: EtaRequest) -> EtaResponse:
    """배달 예상 시간을 계산"""
    minutes = estimate_delivery_minutes(req.distance_km, req.pending_orders)
    return EtaResponse(minutes=minutes, model=MODEL_IMPL)

