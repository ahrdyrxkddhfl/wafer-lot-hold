"""MES 핵심 동작: Lot 생성, 공정 투입·완료, 설비 상태 변경, Hold 열기·처분.

모든 함수는 autocommit 연결을 받아 안에서 트랜잭션 하나를 연다.
Lot을 바꾸는 동작은 먼저 lot 행을 `SELECT ... FOR UPDATE`로 잠근다. 같은 Lot에 대한 요청이
동시에 와도 한 번에 하나씩 검사·기록되게 하기 위해서다. 설비도 필요하면 Lot 다음에 잠근다
(잠그는 순서를 항상 Lot → 설비로 맞춰 교착을 피한다).
"""
import logging
from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row

from mes.errors import RuleViolation

logger = logging.getLogger(__name__)

EQUIPMENT_STATUSES = ("AVAILABLE", "DOWN", "MAINTENANCE")
DISPOSITION_ACTIONS = ("RELEASE", "RETEST", "SCRAP")
MANUAL_RULE = "MANUAL"


@dataclass(frozen=True)
class WorkOrderResult:
    """투입·완료 결과. 같은 요청이 다시 오면 already_processed=True로 같은 내용을 돌려준다."""
    work_order_id: str
    lot_id: str
    step_code: str
    equipment_id: str
    status: str
    already_processed: bool


@dataclass(frozen=True)
class HoldResult:
    """Hold 열기 결과. 이미 열린 Hold가 있으면 created=False로 그 Hold를 돌려준다."""
    hold_id: int
    created: bool


def _require_text(value: str | None, field: str, rule: str) -> str:
    """빈 문자열·공백·None을 거절한다."""
    if value is None or not value.strip():
        raise RuleViolation(rule, f"{field}가 비어 있음")
    return value


def _lock_lot(cur: psycopg.Cursor, lot_id: str) -> dict:
    """Lot 행을 잠그고 읽는다. 같은 Lot을 바꾸려는 다른 트랜잭션은 커밋될 때까지 여기서 기다린다."""
    cur.execute("SELECT lot_id, status, current_step_code, inspect_round FROM lot "
                "WHERE lot_id = %s FOR UPDATE", (lot_id,))
    lot = cur.fetchone()
    if lot is None:
        raise RuleViolation("NOT_FOUND", f"Lot 없음: {lot_id}")
    return lot


def _lock_equipment(cur: psycopg.Cursor, equipment_id: str) -> dict:
    """설비 행을 잠그고 읽는다.

    FOR SHARE가 아니라 FOR UPDATE를 쓴다. FOR SHARE는 여러 트랜잭션이 함께 잡을 수 있어서
    두 Lot이 같은 설비를 동시에 "비어 있음"으로 읽을 수 있기 때문이다(I11).
    """
    cur.execute("SELECT equipment_id, step_code, status FROM equipment "
                "WHERE equipment_id = %s FOR UPDATE", (equipment_id,))
    eq = cur.fetchone()
    if eq is None:
        raise RuleViolation("NOT_FOUND", f"설비 없음: {equipment_id}")
    return eq


def _step_seq(cur: psycopg.Cursor, step_code: str) -> int:
    """공정의 순번을 읽는다."""
    cur.execute("SELECT seq FROM route_step WHERE step_code = %s", (step_code,))
    row = cur.fetchone()
    if row is None:
        raise RuleViolation("NOT_FOUND", f"공정 없음: {step_code}")
    return row["seq"]


def _open_hold_id(cur: psycopg.Cursor, lot_id: str) -> int | None:
    """Lot의 열린 Hold ID. 없으면 None."""
    cur.execute("SELECT hold_id FROM hold WHERE lot_id = %s AND closed_at IS NULL", (lot_id,))
    row = cur.fetchone()
    return None if row is None else row["hold_id"]


def _add_history(cur: psycopg.Cursor, lot_id: str, event: str, step_code: str | None = None,
                 equipment_id: str | None = None, work_order_id: str | None = None,
                 hold_id: int | None = None) -> None:
    """Lot 이력을 한 줄 추가한다."""
    cur.execute("INSERT INTO lot_history (lot_id, event, step_code, equipment_id, work_order_id, hold_id) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (lot_id, event, step_code, equipment_id, work_order_id, hold_id))


def create_lot(conn: psycopg.Connection, lot_id: str, wafers: list[tuple[str, int]]) -> None:
    """Lot과 웨이퍼를 만들고 첫 공정 대기 상태로 둔다.

    Args:
        conn: autocommit 연결.
        lot_id: Lot ID.
        wafers: (웨이퍼 ID, 웨이퍼 번호) 목록.
    """
    with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT step_code FROM route_step ORDER BY seq LIMIT 1")
        first = cur.fetchone()
        if first is None:
            raise RuleViolation("NOT_FOUND", "공정 순서가 비어 있음")
        cur.execute("INSERT INTO lot (lot_id, status, current_step_code) VALUES (%s, 'WAITING', %s)",
                    (lot_id, first["step_code"]))
        cur.executemany("INSERT INTO wafer (wafer_id, lot_id, wafer_index) VALUES (%s, %s, %s)",
                        [(wafer_id, lot_id, idx) for wafer_id, idx in wafers])
        _add_history(cur, lot_id, "CREATED", step_code=first["step_code"])
    logger.info("Lot 생성 %s (웨이퍼 %d장)", lot_id, len(wafers))


def set_equipment_status(conn: psycopg.Connection, equipment_id: str, status: str,
                         changed_by: str, reason: str) -> None:
    """설비 상태를 바꾸고 이력을 남긴다.

    Args:
        conn: autocommit 연결.
        equipment_id: 설비 ID.
        status: AVAILABLE, DOWN, MAINTENANCE 중 하나.
        changed_by: 바꾼 사람.
        reason: 사유.
    """
    if status not in EQUIPMENT_STATUSES:
        raise RuleViolation("INVALID", f"알 수 없는 설비 상태: {status}")
    _require_text(changed_by, "changed_by", "INVALID")
    _require_text(reason, "reason", "INVALID")
    with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        eq = _lock_equipment(cur, equipment_id)
        if eq["status"] == status:
            raise RuleViolation("INVALID", f"{equipment_id}는 이미 {status}")
        cur.execute("UPDATE equipment SET status = %s, updated_at = now() WHERE equipment_id = %s",
                    (status, equipment_id))
        cur.execute("INSERT INTO equipment_status_history "
                    "(equipment_id, from_status, to_status, changed_by, reason) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (equipment_id, eq["status"], status, changed_by, reason))
    logger.info("설비 %s: %s -> %s", equipment_id, eq["status"], status)


def track_in(conn: psycopg.Connection, work_order_id: str, lot_id: str, step_code: str,
             equipment_id: str) -> WorkOrderResult:
    """작업 지시에 따라 Lot을 설비에 투입한다.

    Args:
        conn: autocommit 연결.
        work_order_id: 지시하는 쪽이 정한 작업 지시 ID.
        lot_id: Lot ID.
        step_code: 처리할 공정.
        equipment_id: 투입할 설비.

    Returns:
        작업 지시 결과. 같은 ID·같은 내용의 요청이 이미 처리됐으면 already_processed=True.

    Raises:
        RuleViolation: I1~I4, I6, I7, I10, I11 규칙에 어긋날 때.
    """
    _require_text(work_order_id, "work_order_id", "INVALID")
    with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        lot = _lock_lot(cur, lot_id)

        cur.execute("SELECT * FROM work_order WHERE work_order_id = %s", (work_order_id,))
        existing = cur.fetchone()
        if existing is not None:
            if (existing["lot_id"], existing["step_code"], existing["equipment_id"]) != \
                    (lot_id, step_code, equipment_id):
                raise RuleViolation("I3", f"작업 지시 {work_order_id}는 다른 내용으로 이미 있음")
            return WorkOrderResult(work_order_id, lot_id, step_code, equipment_id,
                                   existing["status"], already_processed=True)

        if lot["status"] == "SCRAPPED":
            raise RuleViolation("I10", f"폐기된 Lot: {lot_id}")
        if lot["status"] == "FINISHED":
            raise RuleViolation("I2", f"모든 공정을 마친 Lot: {lot_id}")
        if _open_hold_id(cur, lot_id) is not None:
            raise RuleViolation("I4", f"Hold 중인 Lot: {lot_id}")
        if lot["status"] == "IN_PROCESS":
            raise RuleViolation("I7", f"이미 처리 중인 Lot: {lot_id}")

        req_seq = _step_seq(cur, step_code)
        cur.execute("SELECT 1 FROM work_order WHERE lot_id = %s AND step_code = %s",
                    (lot_id, step_code))
        if cur.fetchone() is not None or req_seq < _step_seq(cur, lot["current_step_code"]):
            raise RuleViolation("I2", f"{lot_id}는 {step_code}를 이미 처리함")
        if step_code != lot["current_step_code"]:
            raise RuleViolation("I1", f"{lot_id}의 다음 공정은 {lot['current_step_code']} (요청 {step_code})")

        eq = _lock_equipment(cur, equipment_id)
        if eq["step_code"] != step_code:
            raise RuleViolation("EQUIPMENT_STEP", f"{equipment_id}는 {eq['step_code']} 설비")
        if eq["status"] != "AVAILABLE":
            raise RuleViolation("I6", f"{equipment_id} 상태 {eq['status']}")
        cur.execute("SELECT lot_id FROM work_order WHERE equipment_id = %s AND status = 'STARTED'",
                    (equipment_id,))
        busy = cur.fetchone()
        if busy is not None:
            raise RuleViolation("I11", f"{equipment_id}는 {busy['lot_id']} 처리 중")

        cur.execute("INSERT INTO work_order (work_order_id, lot_id, step_code, equipment_id, status) "
                    "VALUES (%s, %s, %s, %s, 'STARTED')",
                    (work_order_id, lot_id, step_code, equipment_id))
        cur.execute("UPDATE lot SET status = 'IN_PROCESS' WHERE lot_id = %s", (lot_id,))
        _add_history(cur, lot_id, "TRACK_IN", step_code, equipment_id, work_order_id)
    logger.info("투입 %s: %s %s @ %s", work_order_id, lot_id, step_code, equipment_id)
    return WorkOrderResult(work_order_id, lot_id, step_code, equipment_id, "STARTED",
                           already_processed=False)


def track_out(conn: psycopg.Connection, work_order_id: str) -> WorkOrderResult:
    """작업 지시를 완료하고 Lot을 다음 공정 대기(마지막이면 FINISHED)로 넘긴다.

    Args:
        conn: autocommit 연결.
        work_order_id: 완료할 작업 지시 ID.

    Returns:
        작업 지시 결과. 이미 완료된 지시면 already_processed=True.

    Raises:
        RuleViolation: I4, I10 규칙에 어긋나거나 작업 지시가 없을 때.
    """
    with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT lot_id FROM work_order WHERE work_order_id = %s", (work_order_id,))
        row = cur.fetchone()
        if row is None:
            raise RuleViolation("NOT_FOUND", f"작업 지시 없음: {work_order_id}")
        lot = _lock_lot(cur, row["lot_id"])
        # 잠금을 기다리는 동안 바뀌었을 수 있으므로 잠근 뒤에 다시 읽는다.
        cur.execute("SELECT * FROM work_order WHERE work_order_id = %s", (work_order_id,))
        wo = cur.fetchone()
        result = WorkOrderResult(work_order_id, wo["lot_id"], wo["step_code"], wo["equipment_id"],
                                 "COMPLETED", already_processed=True)
        if wo["status"] == "COMPLETED":
            return result
        if lot["status"] == "SCRAPPED":
            raise RuleViolation("I10", f"폐기된 Lot: {lot['lot_id']}")
        if _open_hold_id(cur, lot["lot_id"]) is not None:
            raise RuleViolation("I4", f"Hold 중인 Lot: {lot['lot_id']}")

        cur.execute("UPDATE work_order SET status = 'COMPLETED', ended_at = now() "
                    "WHERE work_order_id = %s", (work_order_id,))
        cur.execute("SELECT step_code FROM route_step WHERE seq > %s ORDER BY seq LIMIT 1",
                    (_step_seq(cur, wo["step_code"]),))
        nxt = cur.fetchone()
        if nxt is None:
            cur.execute("UPDATE lot SET status = 'FINISHED', current_step_code = NULL "
                        "WHERE lot_id = %s", (lot["lot_id"],))
        else:
            cur.execute("UPDATE lot SET status = 'WAITING', current_step_code = %s WHERE lot_id = %s",
                        (nxt["step_code"], lot["lot_id"]))
        _add_history(cur, lot["lot_id"], "TRACK_OUT", wo["step_code"], wo["equipment_id"], work_order_id)
    logger.info("완료 %s: %s %s", work_order_id, wo["lot_id"], wo["step_code"])
    return WorkOrderResult(work_order_id, wo["lot_id"], wo["step_code"], wo["equipment_id"],
                           "COMPLETED", already_processed=False)


def open_hold(conn: psycopg.Connection, lot_id: str, rule_name: str, *,
              trigger_result_id: int | None = None, opened_by: str | None = None) -> HoldResult:
    """Lot에 Hold를 연다. 이미 열린 Hold가 있으면 새로 만들지 않고 그 Hold를 돌려준다(I9).

    규칙이 연 Hold는 trigger_result_id(그 판정 결과), 사람이 연 Hold는 opened_by와
    rule_name='MANUAL' 중 정확히 한쪽만 준다.

    Args:
        conn: autocommit 연결(바깥 트랜잭션 안에서 부르면 savepoint가 된다).
        lot_id: Lot ID.
        rule_name: Hold를 연 규칙 이름.
        trigger_result_id: Hold를 열게 한 inspection_result ID.
        opened_by: Hold를 연 사람.

    Returns:
        Hold ID와 새로 만들었는지 여부.

    Raises:
        RuleViolation: I10에 어긋나거나 입력이 잘못됐을 때.
    """
    _require_text(rule_name, "rule_name", "INVALID")
    if (trigger_result_id is None) == (opened_by is None):
        raise RuleViolation("INVALID", "trigger_result_id와 opened_by 중 정확히 하나만 줘야 함")
    if opened_by is not None:
        _require_text(opened_by, "opened_by", "INVALID")
    if (opened_by is not None) != (rule_name == MANUAL_RULE):
        raise RuleViolation("INVALID", f"사람이 연 Hold만 rule_name이 {MANUAL_RULE}")

    with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        lot = _lock_lot(cur, lot_id)
        if lot["status"] == "SCRAPPED":
            raise RuleViolation("I10", f"폐기된 Lot: {lot_id}")
        if lot["status"] == "FINISHED":
            raise RuleViolation("INVALID", f"모든 공정을 마친 Lot: {lot_id}")
        existing = _open_hold_id(cur, lot_id)
        if existing is not None:
            return HoldResult(existing, created=False)
        cur.execute("INSERT INTO hold (lot_id, rule_name, trigger_result_id, opened_by, inspect_round) "
                    "VALUES (%s, %s, %s, %s, %s) RETURNING hold_id",
                    (lot_id, rule_name, trigger_result_id, opened_by, lot["inspect_round"]))
        hold_id = cur.fetchone()["hold_id"]
        _add_history(cur, lot_id, "HOLD", step_code=lot["current_step_code"], hold_id=hold_id)
    logger.info("Hold %d 열림: %s (%s)", hold_id, lot_id, rule_name)
    return HoldResult(hold_id, created=True)


def dispose_hold(conn: psycopg.Connection, hold_id: int, action: str, decided_by: str,
                 reason: str) -> int:
    """열린 Hold를 처분(해제·재검사·폐기)하고 닫는다.

    RELEASE는 Hold만 닫는다. RETEST는 lot.inspect_round를 1 올린다(작업 지시는 새로 만들지 않음).
    SCRAP은 Lot을 SCRAPPED(끝 상태)로 두고 처리 중인 작업 지시를 ABORTED로 닫는다.

    Args:
        conn: autocommit 연결.
        hold_id: 처분할 Hold.
        action: RELEASE, RETEST, SCRAP 중 하나.
        decided_by: 결정한 사람.
        reason: 사유.

    Returns:
        처분 ID.

    Raises:
        RuleViolation: 결정자·사유가 비었거나(I5) Hold가 없거나 이미 닫혔을 때.
    """
    _require_text(decided_by, "decided_by", "I5")
    _require_text(reason, "reason", "I5")
    if action not in DISPOSITION_ACTIONS:
        raise RuleViolation("INVALID", f"알 수 없는 처분: {action}")

    with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT lot_id FROM hold WHERE hold_id = %s", (hold_id,))
        row = cur.fetchone()
        if row is None:
            raise RuleViolation("NOT_FOUND", f"Hold 없음: {hold_id}")
        lot = _lock_lot(cur, row["lot_id"])
        cur.execute("SELECT closed_at FROM hold WHERE hold_id = %s", (hold_id,))
        if cur.fetchone()["closed_at"] is not None:
            raise RuleViolation("INVALID", f"이미 처분된 Hold: {hold_id}")

        cur.execute("INSERT INTO hold_disposition (hold_id, action, decided_by, reason) "
                    "VALUES (%s, %s, %s, %s) RETURNING disposition_id",
                    (hold_id, action, decided_by, reason))
        disposition_id = cur.fetchone()["disposition_id"]
        cur.execute("UPDATE hold SET closed_at = now() WHERE hold_id = %s", (hold_id,))
        if action == "RETEST":
            cur.execute("UPDATE lot SET inspect_round = inspect_round + 1 WHERE lot_id = %s",
                        (lot["lot_id"],))
        elif action == "SCRAP":
            cur.execute("UPDATE work_order SET status = 'ABORTED', ended_at = now() "
                        "WHERE lot_id = %s AND status = 'STARTED'", (lot["lot_id"],))
            cur.execute("UPDATE lot SET status = 'SCRAPPED' WHERE lot_id = %s", (lot["lot_id"],))
        _add_history(cur, lot["lot_id"], action, step_code=lot["current_step_code"], hold_id=hold_id)
    logger.info("Hold %d 처분 %s by %s", hold_id, action, decided_by)
    return disposition_id
