import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.predictors.rule import estimate_delivery_minutes

client = TestClient(app)


# ---------- /health ----------

def test_health_returns_ok_and_model_name():
    """/health 는 200 과 {"status": "ok", "model": "rule"} 을 돌려준다."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "model": "rule"}


# ---------- /eta 정상 입력 ----------

def test_eta_response_shape():
    """/eta 응답은 minutes(숫자)와 model(문자열) 두 필드만 가진다."""
    res = client.post("/eta", json={"distance_km": 2, "pending_orders": 3})
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"minutes", "model"}
    assert isinstance(body["minutes"], (int, float))
    assert body["model"] == "rule"


@pytest.mark.parametrize(
    ("distance_km", "pending_orders", "expected"),
    [
        (2, 3, 22.0),        # 10 + 3*2 + 2*3
        (1.5, 0, 14.5),      # 소수 거리
        (0, 5, 20.0),        # 거리 0
        (4, 0, 22.0),        # 대기 주문 0
    ],
)
def test_eta_known_values(distance_km, pending_orders, expected):
    """알려진 입력에 대해 기준선 공식(10 + 3*km + 2*주문)의 값을 돌려준다."""
    res = client.post(
        "/eta", json={"distance_km": distance_km, "pending_orders": pending_orders}
    )
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(expected)


# ---------- /eta 경계값 ----------

def test_eta_zero_inputs_returns_base_time():
    """거리 0, 대기 주문 0 이면 기본 10분을 돌려준다."""
    res = client.post("/eta", json={"distance_km": 0, "pending_orders": 0})
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(10.0)


def test_eta_large_inputs():
    """큰 값도 422 없이 공식대로 계산한다."""
    res = client.post("/eta", json={"distance_km": 1000, "pending_orders": 10000})
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(10 + 3 * 1000 + 2 * 10000)


# ---------- /eta 잘못된 입력 ----------

@pytest.mark.parametrize(
    "payload",
    [
        {"distance_km": -1, "pending_orders": 0},          # 음수 거리
        {"distance_km": -0.01, "pending_orders": 0},       # 0 바로 아래
        {"distance_km": 1, "pending_orders": -1},          # 음수 주문 수
        {"distance_km": "abc", "pending_orders": 0},       # 숫자가 아닌 거리
        {"distance_km": 1, "pending_orders": "many"},      # 숫자가 아닌 주문 수
        {"distance_km": 1, "pending_orders": 1.5},         # 정수가 아닌 주문 수
        {"distance_km": None, "pending_orders": 0},        # null
        {"pending_orders": 0},                             # 거리 누락
        {"distance_km": 1},                                # 주문 수 누락
        {},                                                # 빈 본문
    ],
)
def test_eta_invalid_input_returns_422(payload):
    """음수·숫자 아님·누락 등 잘못된 입력은 422 를 돌려준다."""
    res = client.post("/eta", json=payload)
    assert res.status_code == 422


def test_eta_get_method_not_allowed():
    """/eta 는 POST 만 받으며 GET 은 405 를 돌려준다."""
    res = client.get("/eta")
    assert res.status_code == 405


# ---------- rule 구현체 직접 ----------

@pytest.mark.parametrize(
    ("distance_km", "pending_orders", "expected"),
    [(0, 0, 10.0), (2, 3, 22.0), (10, 10, 60.0)],
)
def test_rule_estimate_known_values(distance_km, pending_orders, expected):
    """rule 구현체는 10 + 3*km + 2*주문 을 float 로 돌려준다."""
    result = estimate_delivery_minutes(distance_km, pending_orders)
    assert isinstance(result, float)
    assert result == pytest.approx(expected)
