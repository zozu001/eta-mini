"""Python API 와 다른 언어 서비스(Go, Rust)의 예측이 같은지 비교한다.
CI 에서 실패(exit 1)하면 이후 단계(배포)가 진행되지 않는다.

기준: Python API (PY_URL, 기본 http://127.0.0.1:8080). 학습한 원본 모델(MODEL_IMPL=sklearn)로 띄운다.
비교 대상: TARGETS (기본 go=http://127.0.0.1:8081,rust=http://127.0.0.1:8082)
판정: 대상별 최대 오차가 TOLERANCE 분 미만이면 PASS.

서버들이 떠 있어야 하므로 pytest 가 아니라 스크립트로 실행한다. 표준 라이브러리만 쓴다.
    python -m training.check_parity
"""

import json
import os
import random
import sys
import urllib.error
import urllib.request

TOLERANCE = 1e-4  # 분
TIMEOUT = 5       # 초

PY_URL = os.getenv("PY_URL", "http://127.0.0.1:8080").rstrip("/")
TARGETS = dict(
    item.split("=", 1)
    for item in os.getenv("TARGETS", "go=http://127.0.0.1:8081,rust=http://127.0.0.1:8082").split(",")
    if item.strip()
)


def make_inputs() -> list[tuple[float, int]]:
    """고정 입력 3건 + seed 0 무작위 20건 (거리 0.5~8km, 주문 0~10건)."""
    rng = random.Random(0)
    fixed = [(3.0, 2), (0.0, 0), (8.0, 10)]
    rand = [(round(rng.uniform(0.5, 8.0), 2), rng.randint(0, 10)) for _ in range(20)]
    return fixed + rand


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
        return json.load(resp)


def predict(base: str, distance_km: float, pending_orders: int) -> float:
    body = json.dumps({"distance_km": distance_km, "pending_orders": pending_orders}).encode()
    req = urllib.request.Request(f"{base}/eta", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return float(json.load(resp)["minutes"])


def main() -> int:
    inputs = make_inputs()

    # 기준 서비스가 원본 모델로 떠 있는지 먼저 확인한다 (다른 구현체와 비교하면 의미가 없다)
    try:
        model = get_json(f"{PY_URL}/health").get("model")
        if model != "sklearn":
            print(f"FAIL [python] 기준 서비스의 모델이 sklearn 이 아니다 (model={model!r}, {PY_URL})")
            return 1
        expected = [predict(PY_URL, d, o) for d, o in inputs]
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        print(f"FAIL [python] 기준 서비스 호출 실패 ({PY_URL}): {e}")
        return 1

    ok = True
    for name, base in TARGETS.items():
        base = base.rstrip("/")
        try:
            got = [predict(base, d, o) for d, o in inputs]
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            print(f"FAIL [{name}] 호출 실패 ({base}): {e}")
            ok = False
            continue

        diffs = [abs(g - w) for g, w in zip(got, expected)]
        worst = max(range(len(diffs)), key=diffs.__getitem__)
        passed = diffs[worst] < TOLERANCE
        ok &= passed
        d, o = inputs[worst]
        print(f"{'PASS' if passed else 'FAIL'} [{name}] 최대 오차 {diffs[worst]:.2e}분 "
              f"(< {TOLERANCE:g}) · {len(inputs)}건 · 최악 입력 ({d}, {o}): "
              f"python {expected[worst]:.6f} vs {name} {got[worst]:.6f}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
