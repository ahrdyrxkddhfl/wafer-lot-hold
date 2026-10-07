"""MES HTTP API(FastAPI).

검사 장비는 DB에 직접 쓰지 않고 이 API로만 기록한다. 장비는 공정 규칙을 모르므로, 모든 기록이 MES 규칙(mes/service.py)을
지나가는 입구를 하나로 두기 위해서다.

엔드포인트는 일반(동기) 함수다. DB 드라이버를 동기로 쓰고, 처리 속도보다 기록 정합성이 우선이기 때문이다.
FastAPI는 동기 엔드포인트를 스레드 풀에서 돌리므로 요청이 동시에 처리될 수 있고, 그 정합성은 서비스 계층의 행 잠금이 맡는다.

실행: .venv/bin/uvicorn mes.api:app
"""
import logging
from collections.abc import Iterator
from dataclasses import asdict
from typing import Annotated, Literal

import psycopg
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StringConstraints

from mes import service
from mes.db import connect, load_config
from mes.errors import RuleViolation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CFG = load_config()
LABELS: list[str] = CFG["labels"]
RULE = service.StopRule.from_config(CFG["stop_rule"])
MAX_RESULTS: int = CFG["api"]["max_results_per_request"]

# 규칙 코드 → HTTP 상태 코드. 여기 없는 코드(I1~I13, HOLD_CLOSED 등 상태 규칙)는 409 Conflict.
STATUS_BY_RULE = {"NOT_FOUND": 404, "INVALID": 422}

# 앞뒤 공백을 지운 뒤 비어 있으면 입구에서 422로 거절한다(I5의 입력 검사도 여기서 먼저 걸린다).
NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def get_conn() -> Iterator[psycopg.Connection]:
    """요청마다 DB 연결을 열고 응답 후 닫는다(연결 풀 라이브러리는 쓰지 않는다)."""
    with connect(CFG, CFG["db"]["dbname"]) as conn:
        yield conn


Conn = Annotated[psycopg.Connection, Depends(get_conn)]
app = FastAPI(title="wafer-lot-hold MES")


# ── 요청 형식 ────────────────────────────────────────────────

class WaferIn(BaseModel):
    wafer_id: NonBlank
    wafer_index: int = Field(gt=0)


class LotIn(BaseModel):
    lot_id: NonBlank
    wafers: list[WaferIn] = Field(min_length=1)


class WorkOrderIn(BaseModel):
    work_order_id: NonBlank
    lot_id: NonBlank
    step_code: NonBlank
    equipment_id: NonBlank


class EquipmentStatusIn(BaseModel):
    status: Literal["AVAILABLE", "DOWN", "MAINTENANCE"]
    changed_by: NonBlank
    reason: NonBlank


class DispositionIn(BaseModel):
    action: Literal["RELEASE", "RETEST", "SCRAP"]
    decided_by: NonBlank
    reason: NonBlank


class ResultItem(BaseModel):
    wafer_id: NonBlank
    pred_label: NonBlank
    probabilities: dict[str, float]


class InspectionBatchIn(BaseModel):
    work_order_id: NonBlank
    inspect_round: int = Field(ge=1)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    results: list[ResultItem] = Field(min_length=1, max_length=MAX_RESULTS)


# ── 오류 → HTTP 상태 코드 ────────────────────────────────────

@app.exception_handler(RuleViolation)
def handle_rule_violation(request: Request, exc: RuleViolation) -> JSONResponse:
    """규칙 위반: 없는 대상 404, 잘못된 입력 422, 상태 규칙 위반 409."""
    return JSONResponse(status_code=STATUS_BY_RULE.get(exc.rule, 409),
                        content={"rule": exc.rule, "detail": str(exc)})


@app.exception_handler(psycopg.errors.IntegrityError)
def handle_integrity_error(request: Request, exc: psycopg.errors.IntegrityError) -> JSONResponse:
    """경쟁 상황에서 서비스 검사를 지나 DB 제약에 걸린 경우(예: 같은 작업 지시 ID가 서로 다른 Lot으로 동시에 옴)."""
    logger.warning("DB 제약 위반 %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=409, content={"rule": "DB_CONSTRAINT", "detail": str(exc).splitlines()[0]})


@app.exception_handler(Exception)
def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    """예상 못 한 오류는 스택까지 로그로 남기고 500으로 돌려준다(삼키지 않는다)."""
    # 처리기는 except 블록 밖에서 불리므로 logger.exception으로는 스택이 남지 않는다. 예외를 직접 넘긴다.
    logger.error("처리하지 못한 오류 %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"rule": "INTERNAL", "detail": "서버 오류(로그 확인)"})


# ── 엔드포인트 ───────────────────────────────────────────────

@app.get("/equipment")
def get_equipment(conn: Conn) -> dict:
    """공정 순서, 공정별 설비·상태, 검사 공정 이름."""
    return service.get_equipment_layout(conn)


@app.post("/lots")
def post_lot(body: LotIn, conn: Conn) -> dict:
    """Lot 생성. 같은 구성으로 다시 오면 기존 결과, 다른 구성이면 409."""
    return asdict(service.create_lot(conn, body.lot_id, [(w.wafer_id, w.wafer_index) for w in body.wafers]))


@app.post("/work-orders")
def post_work_order(body: WorkOrderIn, conn: Conn) -> dict:
    """작업 지시에 따른 투입(track_in)."""
    return asdict(service.track_in(conn, body.work_order_id, body.lot_id, body.step_code, body.equipment_id))


@app.post("/work-orders/{work_order_id}/complete")
def complete_work_order(work_order_id: str, conn: Conn) -> dict:
    """작업 지시 완료(track_out)."""
    return asdict(service.track_out(conn, work_order_id))


@app.post("/equipment/{equipment_id}/status")
def post_equipment_status(equipment_id: str, body: EquipmentStatusIn, conn: Conn) -> dict:
    """설비 상태 변경. 고장(DOWN)이면 처리 중인 Lot에 Hold가 열린다."""
    hold = service.set_equipment_status(conn, equipment_id, body.status, body.changed_by, body.reason)
    return {"equipment_id": equipment_id, "status": body.status, "hold": None if hold is None else asdict(hold)}


@app.post("/holds/{hold_id}/disposition")
def post_disposition(hold_id: int, body: DispositionIn, conn: Conn) -> dict:
    """Hold 처분(해제·재검사·폐기). 결정자와 사유가 필요하다."""
    return asdict(service.dispose_hold(conn, hold_id, body.action, body.decided_by, body.reason))


@app.post("/lots/{lot_id}/inspection-results")
def post_inspection_results(lot_id: str, body: InspectionBatchIn, conn: Conn) -> dict:
    """Lot 하나의 한 차수 검사 판정 배치. 하나라도 잘못되면 전체 거절, 같은 내용 재전송은 중복으로 센다."""
    results = [service.ResultIn(r.wafer_id, r.pred_label, r.probabilities) for r in body.results]
    return asdict(service.receive_inspection_results(conn, lot_id, body.work_order_id, body.inspect_round,
                                                     body.model_sha256, results, LABELS, RULE))
