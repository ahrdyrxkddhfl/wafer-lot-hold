"""HTTP API 테스트: 오류 → 상태 코드 대응, 판정 배치 엔드포인트, Lot 생성 재요청. 테스트 DB를 쓴다."""
import logging
from collections.abc import Callable, Iterator

import psycopg
import pytest
from fastapi.testclient import TestClient

from mes import service
from mes.api import app, get_conn

MODEL = "c" * 64


@pytest.fixture
def client(connect_test: Callable[[], psycopg.Connection]) -> Iterator[TestClient]:
    """API가 개발용 DB 대신 테스트 DB를 쓰게 한 클라이언트. 서버 예외도 500 응답으로 받는다."""
    def test_conn() -> Iterator[psycopg.Connection]:
        with connect_test() as c:
            yield c

    app.dependency_overrides[get_conn] = test_conn
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


def lot_body(lot_id: str, n: int = 2) -> dict:
    return {"lot_id": lot_id, "wafers": [{"wafer_id": f"{lot_id}_W{i:02d}", "wafer_index": i} for i in range(1, n + 1)]}


def batch(lot_id: str, label_by_wafer: dict[str, str], labels: list[str], inspect_round: int = 1) -> dict:
    return {"work_order_id": f"{lot_id}-INSPECT", "inspect_round": inspect_round, "model_sha256": MODEL,
            "results": [{"wafer_id": w, "pred_label": lab,
                         "probabilities": {c: (1.0 if c == lab else 0.0) for c in labels}}
                        for w, lab in label_by_wafer.items()]}


def to_inspect(client: TestClient, lot_id: str, steps: list[dict]) -> None:
    """Lot을 만들고 검사 공정 앞까지 처리한 뒤 검사 공정에 투입한다."""
    assert client.post("/lots", json=lot_body(lot_id)).status_code == 200
    for step in steps:
        wo = f"{lot_id}-{step['step_code']}"
        r = client.post("/work-orders", json={"work_order_id": wo, "lot_id": lot_id, "step_code": step["step_code"],
                                              "equipment_id": step["equipment"][0]["equipment_id"]})
        assert r.status_code == 200, r.json()
        if step["step_code"] == service.INSPECT_STEP:
            return
        assert client.post(f"/work-orders/{wo}/complete").status_code == 200


def test_get_equipment_returns_route_and_inspect_step(client, mes_cfg):
    body = client.get("/equipment").json()
    assert body["inspect_step"] == service.INSPECT_STEP
    assert [s["step_code"] for s in body["steps"]] == mes_cfg["route"]
    assert [e["equipment_id"] for e in body["steps"][0]["equipment"]] == mes_cfg["equipment"][mes_cfg["route"][0]]


def test_create_lot_again_returns_existing_and_different_wafers_conflict(client):
    first = client.post("/lots", json=lot_body("LOT_A"))
    again = client.post("/lots", json=lot_body("LOT_A"))
    other = client.post("/lots", json=lot_body("LOT_A", n=3))
    assert (first.status_code, first.json()["already_processed"]) == (200, False)
    assert (again.status_code, again.json()["already_processed"]) == (200, True)
    assert (other.status_code, other.json()["rule"]) == (409, "LOT_EXISTS")


@pytest.mark.parametrize("body", [
    {"action": "RELEASE", "decided_by": "engineer", "reason": "   "},
    {"action": "RELEASE", "decided_by": "", "reason": "사유"},
    {"action": "IGNORE", "decided_by": "engineer", "reason": "사유"},
])
def test_invalid_disposition_is_rejected_at_entrance_with_422(client, body):
    assert client.post("/holds/1/disposition", json=body).status_code == 422


def test_rule_violation_maps_to_409_and_not_found_to_404(client, mes_cfg):
    route, eqs = mes_cfg["route"], mes_cfg["equipment"]
    client.post("/lots", json=lot_body("LOT_A"))
    skip = client.post("/work-orders", json={"work_order_id": "WO-1", "lot_id": "LOT_A", "step_code": route[1],
                                             "equipment_id": eqs[route[1]][0]})
    missing = client.post("/work-orders/NO-SUCH/complete")
    assert (skip.status_code, skip.json()["rule"]) == (409, "I1")
    assert (missing.status_code, missing.json()["rule"]) == (404, "NOT_FOUND")


def test_inspection_batch_flow_over_http(client, mes_cfg):
    steps = client.get("/equipment").json()["steps"]
    labels = mes_cfg["labels"]
    to_inspect(client, "LOT_A", steps)
    defect = batch("LOT_A", {"LOT_A_W01": "Center", "LOT_A_W02": "none"}, labels)

    first = client.post("/lots/LOT_A/inspection-results", json=defect).json()
    again = client.post("/lots/LOT_A/inspection-results", json=defect).json()
    assert (first["inserted"], first["duplicates"], first["hold"]["created"]) == (2, 0, True)
    assert (again["inserted"], again["duplicates"], again["hold"]) == (0, 2, {"hold_id": first["hold"]["hold_id"],
                                                                               "created": False})

    changed = batch("LOT_A", {"LOT_A_W01": "none"}, labels)
    r = client.post("/lots/LOT_A/inspection-results", json=changed)
    assert (r.status_code, r.json()["rule"]) == (409, "I8")

    inspect_idx = [s["step_code"] for s in steps].index(service.INSPECT_STEP)
    nxt = steps[inspect_idx + 1]
    assert client.post(f"/work-orders/LOT_A-{service.INSPECT_STEP}/complete").status_code == 200
    ship = {"work_order_id": "LOT_A-NEXT", "lot_id": "LOT_A", "step_code": nxt["step_code"],
            "equipment_id": nxt["equipment"][0]["equipment_id"]}
    blocked = client.post("/work-orders", json=ship)
    assert (blocked.status_code, blocked.json()["rule"]) == (409, "I4")

    released = client.post(f"/holds/{first['hold']['hold_id']}/disposition",
                           json={"action": "RELEASE", "decided_by": "engineer", "reason": "판정 확인 결과 정상"})
    assert released.status_code == 200
    assert client.post("/work-orders", json=ship).status_code == 200


def test_bad_inspection_batch_format_is_422(client, mes_cfg):
    body = batch("LOT_A", {"LOT_A_W01": "none"}, mes_cfg["labels"])
    body["model_sha256"] = "not-a-hash"
    assert client.post("/lots/LOT_A/inspection-results", json=body).status_code == 422


def test_db_constraint_error_maps_to_409(client, monkeypatch):
    def raise_unique(*args, **kwargs):
        raise psycopg.errors.UniqueViolation("duplicate key value violates unique constraint")

    monkeypatch.setattr(service, "track_out", raise_unique)
    r = client.post("/work-orders/WO-1/complete")
    assert (r.status_code, r.json()["rule"]) == (409, "DB_CONSTRAINT")


def test_unexpected_error_is_500_and_logged(client, monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise RuntimeError("예상 못 한 오류")

    monkeypatch.setattr(service, "track_out", boom)
    with caplog.at_level(logging.ERROR, logger="mes.api"):
        r = client.post("/work-orders/WO-1/complete")
    assert (r.status_code, r.json()["rule"]) == (500, "INTERNAL")
    assert any(rec.exc_info and "예상 못 한 오류" in str(rec.exc_info[1]) for rec in caplog.records)
