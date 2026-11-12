import math

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


# ---------- /eta 엣지 케이스: 알려진 결함 ----------

# 서버 예외를 500 응답으로 받기 위한 별도 클라이언트
lenient_client = TestClient(app, raise_server_exceptions=False)


def _post_raw(body: bytes):
    """JSON 표준 밖 토큰(NaN, Infinity)을 그대로 보내기 위한 헬퍼."""
    return lenient_client.post(
        "/eta", content=body, headers={"content-type": "application/json"}
    )


@pytest.mark.xfail(strict=True, reason="버그: NaN 입력 시 422 응답 직렬화 중 500 발생")
def test_eta_nan_distance_returns_422():
    """distance_km 가 NaN 이면 500 이 아니라 422 를 돌려준다."""
    res = _post_raw(b'{"distance_km":NaN,"pending_orders":0}')
    assert res.status_code == 422


@pytest.mark.xfail(strict=True, reason="버그: inf/오버플로 시 minutes 가 null 로 응답됨")
@pytest.mark.parametrize(
    "body",
    [
        b'{"distance_km":Infinity,"pending_orders":0}',   # 무한대
        b'{"distance_km":1e400,"pending_orders":0}',      # 파싱 시 inf
        b'{"distance_km":1e308,"pending_orders":0}',      # 계산 중 오버플로
    ],
)
def test_eta_non_finite_never_returns_null_minutes(body):
    """무한대·오버플로 입력은 422 이거나, 200 이면 minutes 가 유한한 숫자다."""
    res = _post_raw(body)
    if res.status_code == 200:
        minutes = res.json()["minutes"]
        assert isinstance(minutes, (int, float)) and math.isfinite(minutes)
    else:
        assert res.status_code == 422


# ---------- /eta 엣지 케이스: 현재 정책(관대한 변환)을 계약으로 고정 ----------

@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"distance_km": "2", "pending_orders": "3"}, 22.0),     # 숫자 문자열
        ({"distance_km": 2, "pending_orders": 3.0}, 22.0),       # 정수형 float
        ({"distance_km": True, "pending_orders": False}, 13.0),  # bool → 1/0
        ({"distance_km": 1, "pending_orders": 1, "foo": 1}, 15.0),  # 모르는 필드 무시
    ],
)
def test_eta_lenient_coercion(payload, expected):
    """숫자 문자열·정수형 float·bool 은 변환되고, 모르는 필드는 무시된다."""
    res = client.post("/eta", json=payload)
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(expected)


# ---------- /eta 엣지 케이스: 본문 형식 오류 ----------

@pytest.mark.parametrize(
    ("content", "content_type"),
    [
        (b"{bad", "application/json"),                                   # 깨진 JSON
        (b"", "application/json"),                                       # 빈 본문
        (b"[1, 2]", "application/json"),                                 # 배열 본문
        (b'{"distance_km":1,"pending_orders":1}', "text/plain"),         # 잘못된 content-type
    ],
)
def test_eta_malformed_body_returns_422(content, content_type):
    """JSON 이 아니거나 객체가 아닌 본문은 422 를 돌려준다."""
    res = client.post("/eta", content=content, headers={"content-type": content_type})
    assert res.status_code == 422


def test_eta_pending_orders_null_returns_422():
    """pending_orders 가 null 이면 422 를 돌려준다."""
    res = client.post("/eta", json={"distance_km": 1, "pending_orders": None})
    assert res.status_code == 422


@pytest.mark.parametrize(
    ("payload", "bad_field"),
    [
        ({"distance_km": -1, "pending_orders": 0}, "distance_km"),
        ({"distance_km": 1, "pending_orders": "x"}, "pending_orders"),
    ],
)
def test_eta_422_detail_points_to_bad_field(payload, bad_field):
    """422 응답의 detail[].loc 는 문제가 된 필드를 가리킨다."""
    res = client.post("/eta", json=payload)
    assert res.status_code == 422
    locs = [tuple(e["loc"]) for e in res.json()["detail"]]
    assert locs == [("body", bad_field)]


# ---------- /eta 엣지 케이스: 경계값 ----------

@pytest.mark.parametrize(
    ("distance_km", "pending_orders", "expected"),
    [
        (-0.0, 0, 10.0),          # 음의 0 은 0 으로 취급
        (0.001, 0, 10.003),       # 아주 작은 양수
        (0, 10**30, 2e30),        # 아주 큰 정수
    ],
)
def test_eta_boundary_values(distance_km, pending_orders, expected):
    """-0.0, 아주 작은 양수, 아주 큰 정수도 200 과 공식값을 돌려준다."""
    res = client.post(
        "/eta", json={"distance_km": distance_km, "pending_orders": pending_orders}
    )
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(expected)


def test_eta_monotonic_in_inputs():
    """거리나 대기 주문이 늘면 예상 시간은 줄어들지 않는다."""
    def minutes(km, orders):
        return client.post(
            "/eta", json={"distance_km": km, "pending_orders": orders}
        ).json()["minutes"]

    base = minutes(2, 3)
    assert minutes(3, 3) >= base
    assert minutes(2, 4) >= base


# ---------- 라우팅 ----------

def test_health_post_method_not_allowed():
    """/health 는 GET 만 받으며 POST 는 405 를 돌려준다."""
    assert client.post("/health").status_code == 405


def test_unknown_path_returns_404():
    """정의되지 않은 경로는 404 를 돌려준다."""
    assert client.get("/nope").status_code == 404
