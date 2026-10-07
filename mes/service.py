"""MES 핵심 동작: Lot 생성, 공정 투입·완료, 설비 상태 변경, Hold 열기·처분.

모든 함수는 autocommit 연결을 받아 안에서 트랜잭션 하나를 연다.
Lot을 바꾸는 동작은 먼저 lot 행을 잠근다. 같은 Lot에 대한 요청이 동시에 와도 한 번에 하나씩
검사·기록되게 하기 위해서다. 설비도 필요하면 Lot 다음에 잠근다(설비 DOWN 처리만 예외, set_equipment_status 참고).

행 잠금 규칙: 이 프로젝트는 기본키를 바꾸지 않으므로 행 잠금은 모두 `FOR NO KEY UPDATE`를 쓴다.
`FOR UPDATE`는 외래키 확인이 참조 행에 거는 `FOR KEY SHARE`와도 충돌해, 그 행을 참조하는 행을 넣는
다른 트랜잭션을 코드에 보이지 않게 기다리게 만든다(실제로 설비 DOWN과 track_out이 교착했다).
`FOR NO KEY UPDATE`끼리는 충돌하므로 같은 Lot·같은 설비를 동시에 바꾸지 못하는 직렬화는 그대로다.
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
EQUIPMENT_DOWN_RULE = "EQUIPMENT_DOWN"
# 검사 판정을 받는 공정. config/mes.yaml의 route에 반드시 있어야 한다(seed_master_data가 확인).
INSPECT_STEP = "INSPECT"


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
class DispositionResult:
    """처분 결과. 같은 내용의 처분 요청이 다시 오면 already_processed=True로 기존 처분을 돌려준다."""
    disposition_id: int
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
                "WHERE lot_id = %s FOR NO KEY UPDATE", (lot_id,))
    lot = cur.fetchone()
    if lot is None:
        raise RuleViolation("NOT_FOUND", f"Lot 없음: {lot_id}")
    return lot


def _lock_equipment(cur: psycopg.Cursor, equipment_id: str) -> dict:
    """설비 행을 잠그고 읽는다.

    FOR SHARE는 여러 트랜잭션이 함께 잡을 수 있어서 두 Lot이 같은 설비를 동시에 "비어 있음"으로
    읽을 수 있다(I11). FOR NO KEY UPDATE는 서로 충돌하므로 한 번에 하나만 잡는다(모듈 설명의 행 잠금 규칙).
    """
    cur.execute("SELECT equipment_id, step_code, status FROM equipment "
                "WHERE equipment_id = %s FOR NO KEY UPDATE", (equipment_id,))
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


def _open_hold_id(cur: psycopg.Cursor, lot_id: str, rule_name: str | None = None) -> int | None:
    """Lot의 열린 Hold ID. rule_name을 주면 그 종류만 본다. 없으면 None(여럿이면 가장 먼저 열린 것)."""
    if rule_name is None:
        cur.execute("SELECT hold_id FROM hold WHERE lot_id = %s AND closed_at IS NULL ORDER BY hold_id LIMIT 1",
                    (lot_id,))
    else:
        cur.execute("SELECT hold_id FROM hold WHERE lot_id = %s AND rule_name = %s AND closed_at IS NULL",
                    (lot_id, rule_name))
    row = cur.fetchone()
    return None if row is None else row["hold_id"]


def _next_step(cur: psycopg.Cursor, step_code: str) -> str | None:
    """공정 순서에서 다음 공정. 마지막 공정이면 None."""
    cur.execute("SELECT step_code FROM route_step WHERE seq > %s ORDER BY seq LIMIT 1",
                (_step_seq(cur, step_code),))
    row = cur.fetchone()
    return None if row is None else row["step_code"]


def _in_inspect_window(cur: psycopg.Cursor, lot: dict) -> bool:
    """Lot이 "INSPECT를 시작한 뒤부터 다음 공정에 투입하기 전까지"에 있는지.

    이 기간에만 판정을 받고 재검사(RETEST)를 허용한다. Hold는 완료를 막지 않으므로(I4는 투입만 막음)
    INSPECT 완료 뒤에도 다음 공정에 들어가기 전이면 이 기간이다.
    """
    if lot["status"] in ("SCRAPPED", "FINISHED"):
        return False
    cur.execute("SELECT 1 FROM work_order WHERE lot_id = %s AND step_code = %s "
                "AND status IN ('STARTED', 'COMPLETED')", (lot["lot_id"], INSPECT_STEP))
    if cur.fetchone() is None:
        return False
    if lot["current_step_code"] == INSPECT_STEP:
        return True
    return lot["status"] == "WAITING" and lot["current_step_code"] == _next_step(cur, INSPECT_STEP)


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


def _started_lot_on(cur: psycopg.Cursor, equipment_id: str) -> str | None:
    """설비에서 처리 중(STARTED)인 Lot ID. 없으면 None."""
    cur.execute("SELECT lot_id FROM work_order WHERE equipment_id = %s AND status = 'STARTED'",
                (equipment_id,))
    row = cur.fetchone()
    return None if row is None else row["lot_id"]


def set_equipment_status(conn: psycopg.Connection, equipment_id: str, status: str,
                         changed_by: str, reason: str) -> HoldResult | None:
    """설비 상태를 바꾸고 이력을 남긴다.

    MAINTENANCE(계획 정비)는 처리 중인 Lot이 있으면 거절한다(I12).
    DOWN(고장)은 받고, 그 설비에서 처리 중인 Lot에 같은 트랜잭션으로 EQUIPMENT_DOWN Hold를 연다.

    잠금 순서는 다른 함수와 반대로 설비 → Lot이다. 그래도 교착이 없는 이유: Lot을 잡은 채 설비를 기다리는
    함수는 track_in뿐인데, track_in은 처리 중인 Lot(I7)과 다른 공정 설비(EQUIPMENT_STEP)를 설비 잠금 전에
    거절하므로, 이 설비에서 처리 중인 Lot을 잡고 이 설비를 기다리는 트랜잭션은 생기지 않는다.

    Args:
        conn: autocommit 연결.
        equipment_id: 설비 ID.
        status: AVAILABLE, DOWN, MAINTENANCE 중 하나.
        changed_by: 바꾼 사람.
        reason: 사유.

    Returns:
        DOWN으로 EQUIPMENT_DOWN Hold를 열었거나 같은 종류가 이미 열려 있으면 그 결과, 처리 중인 Lot이 없으면 None.

    Raises:
        RuleViolation: I12에 어긋나거나 입력이 잘못됐을 때.
    """
    if status not in EQUIPMENT_STATUSES:
        raise RuleViolation("INVALID", f"알 수 없는 설비 상태: {status}")
    _require_text(changed_by, "changed_by", "INVALID")
    _require_text(reason, "reason", "INVALID")
    hold = None
    with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        eq = _lock_equipment(cur, equipment_id)
        if eq["status"] == status:
            raise RuleViolation("INVALID", f"{equipment_id}는 이미 {status}")
        busy_lot = _started_lot_on(cur, equipment_id)
        if status == "MAINTENANCE" and busy_lot is not None:
            raise RuleViolation("I12", f"{equipment_id}는 {busy_lot} 처리 중이라 정비로 바꿀 수 없음")

        cur.execute("UPDATE equipment SET status = %s, updated_at = now() WHERE equipment_id = %s",
                    (status, equipment_id))
        cur.execute("INSERT INTO equipment_status_history "
                    "(equipment_id, from_status, to_status, changed_by, reason) "
                    "VALUES (%s, %s, %s, %s, %s) RETURNING history_id",
                    (equipment_id, eq["status"], status, changed_by, reason))
        history_id = cur.fetchone()["history_id"]

        if status == "DOWN" and busy_lot is not None:
            _lock_lot(cur, busy_lot)
            # 잠금을 기다리는 동안 그 Lot이 완료됐을 수 있으므로 잠근 뒤 다시 확인한다.
            if _started_lot_on(cur, equipment_id) == busy_lot:
                hold = open_hold(conn, busy_lot, EQUIPMENT_DOWN_RULE,
                                 trigger_equipment_history_id=history_id)
    logger.info("설비 %s: %s -> %s", equipment_id, eq["status"], status)
    return hold


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
        open_hold_id = _open_hold_id(cur, lot_id)
        if open_hold_id is not None:
            raise RuleViolation("I4", f"Hold 중인 Lot: {lot_id} (열린 Hold {open_hold_id} 등)")
        if lot["status"] == "IN_PROCESS":
            raise RuleViolation("I7", f"이미 처리 중인 Lot: {lot_id}")

        req_seq = _step_seq(cur, step_code)
        cur.execute("SELECT 1 FROM work_order WHERE lot_id = %s AND step_code = %s",
                    (lot_id, step_code))
        if cur.fetchone() is not None or req_seq < _step_seq(cur, lot["current_step_code"]):
            raise RuleViolation("I2", f"{lot_id}는 {step_code}를 이미 처리함")
        if step_code != lot["current_step_code"]:
            raise RuleViolation("I1", f"{lot_id}의 다음 공정은 {lot['current_step_code']} (요청 {step_code})")

        # 교착 방지 전제: Lot을 잡은 채 설비 잠금을 기다리는 것은 "이 공정의 설비에 처음 투입하는" 요청뿐이어야 한다.
        # 그래야 설비 DOWN 처리(설비 → 그 설비에서 처리 중인 Lot 순서로 잠금)와 서로 기다리지 않는다.
        #  - 처리 중 확인(I7)은 위에서 설비 잠금 전에 끝냈다.
        #  - 공정-설비 일치는 바뀌지 않는 기준정보라 잠금 없이 읽어 여기서 먼저 확인한다.
        cur.execute("SELECT step_code FROM equipment WHERE equipment_id = %s", (equipment_id,))
        eq_step = cur.fetchone()
        if eq_step is None:
            raise RuleViolation("NOT_FOUND", f"설비 없음: {equipment_id}")
        if eq_step["step_code"] != step_code:
            raise RuleViolation("EQUIPMENT_STEP", f"{equipment_id}는 {eq_step['step_code']} 설비")

        eq = _lock_equipment(cur, equipment_id)
        if eq["status"] != "AVAILABLE":
            raise RuleViolation("I6", f"{equipment_id} 상태 {eq['status']}")
        busy_lot = _started_lot_on(cur, equipment_id)
        if busy_lot is not None:
            raise RuleViolation("I11", f"{equipment_id}는 {busy_lot} 처리 중")

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

    열린 Hold가 있어도 완료는 막지 않는다(I4는 다음 공정 투입만 막는다). Hold된 Lot이 설비를 계속
    차지하면, 사람이 처분할 때까지 그 설비에 다른 Lot을 넣을 수 없기 때문이다.

    Raises:
        RuleViolation: I10 규칙에 어긋나거나 작업 지시가 없을 때.
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

        cur.execute("UPDATE work_order SET status = 'COMPLETED', ended_at = now() "
                    "WHERE work_order_id = %s", (work_order_id,))
        nxt = _next_step(cur, wo["step_code"])
        if nxt is None:
            cur.execute("UPDATE lot SET status = 'FINISHED', current_step_code = NULL "
                        "WHERE lot_id = %s", (lot["lot_id"],))
        else:
            cur.execute("UPDATE lot SET status = 'WAITING', current_step_code = %s WHERE lot_id = %s",
                        (nxt, lot["lot_id"]))
        _add_history(cur, lot["lot_id"], "TRACK_OUT", wo["step_code"], wo["equipment_id"], work_order_id)
    logger.info("완료 %s: %s %s", work_order_id, wo["lot_id"], wo["step_code"])
    return WorkOrderResult(work_order_id, wo["lot_id"], wo["step_code"], wo["equipment_id"],
                           "COMPLETED", already_processed=False)


def open_hold(conn: psycopg.Connection, lot_id: str, rule_name: str, *,
              trigger_result_id: int | None = None, trigger_equipment_history_id: int | None = None,
              opened_by: str | None = None) -> HoldResult:
    """Lot에 Hold를 연다. 같은 종류(rule_name)의 열린 Hold가 있으면 새로 만들지 않고 그 Hold를 돌려준다(I9).

    종류가 다른 Hold는 함께 열린다. 그래야 설비 고장 Hold 중에 들어온 불량 판정이 묻히지 않는다.

    무엇이 열었는지를 정확히 하나만 준다.
        판정 규칙: trigger_result_id(그 판정 결과)
        설비 고장: trigger_equipment_history_id(그 상태 변경 이력), rule_name='EQUIPMENT_DOWN'
        사람: opened_by, rule_name='MANUAL'

    Args:
        conn: autocommit 연결(바깥 트랜잭션 안에서 부르면 savepoint가 된다).
        lot_id: Lot ID.
        rule_name: Hold를 연 규칙 이름.
        trigger_result_id: Hold를 열게 한 inspection_result ID.
        trigger_equipment_history_id: Hold를 열게 한 equipment_status_history ID.
        opened_by: Hold를 연 사람.

    Returns:
        Hold ID와 새로 만들었는지 여부.

    Raises:
        RuleViolation: I10에 어긋나거나 입력이 잘못됐을 때.
    """
    _require_text(rule_name, "rule_name", "INVALID")
    sources = [trigger_result_id, trigger_equipment_history_id, opened_by]
    if sum(s is not None for s in sources) != 1:
        raise RuleViolation("INVALID", "trigger_result_id, trigger_equipment_history_id, opened_by 중 "
                                       "정확히 하나만 줘야 함")
    if opened_by is not None:
        _require_text(opened_by, "opened_by", "INVALID")
    if (opened_by is not None) != (rule_name == MANUAL_RULE):
        raise RuleViolation("INVALID", f"사람이 연 Hold만 rule_name이 {MANUAL_RULE}")
    if (trigger_equipment_history_id is not None) != (rule_name == EQUIPMENT_DOWN_RULE):
        raise RuleViolation("INVALID", f"설비 고장으로 연 Hold만 rule_name이 {EQUIPMENT_DOWN_RULE}")

    with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        lot = _lock_lot(cur, lot_id)
        if lot["status"] == "SCRAPPED":
            raise RuleViolation("I10", f"폐기된 Lot: {lot_id}")
        if lot["status"] == "FINISHED":
            raise RuleViolation("INVALID", f"모든 공정을 마친 Lot: {lot_id}")
        existing = _open_hold_id(cur, lot_id, rule_name)
        if existing is not None:
            return HoldResult(existing, created=False)
        cur.execute("INSERT INTO hold (lot_id, rule_name, trigger_result_id, trigger_equipment_history_id, "
                    "opened_by, inspect_round) VALUES (%s, %s, %s, %s, %s, %s) RETURNING hold_id",
                    (lot_id, rule_name, trigger_result_id, trigger_equipment_history_id, opened_by,
                     lot["inspect_round"]))
        hold_id = cur.fetchone()["hold_id"]
        _add_history(cur, lot_id, "HOLD", step_code=lot["current_step_code"], hold_id=hold_id)
    logger.info("Hold %d 열림: %s (%s)", hold_id, lot_id, rule_name)
    return HoldResult(hold_id, created=True)


def dispose_hold(conn: psycopg.Connection, hold_id: int, action: str, decided_by: str,
                 reason: str) -> DispositionResult:
    """열린 Hold를 처분(해제·재검사·폐기)하고 닫는다.

    RELEASE는 이 Hold만 닫는다. 다른 종류의 열린 Hold가 남아 있으면 Lot은 계속 투입이 막힌다(I4).
    RETEST는 이 Hold를 닫고 lot.inspect_round를 1 올린다(작업 지시는 새로 만들지 않음).
    "INSPECT를 시작한 뒤부터 다음 공정 투입 전까지"에만 허용한다.
    SCRAP은 Lot을 SCRAPPED(끝 상태)로 두고, 처리 중인 작업 지시를 ABORTED로, 그 Lot의 다른 열린 Hold도
    같은 결정자·사유의 SCRAP 처분으로 함께 닫는다(폐기된 Lot에 열린 Hold가 남지 않게).

    Args:
        conn: autocommit 연결.
        hold_id: 처분할 Hold.
        action: RELEASE, RETEST, SCRAP 중 하나.
        decided_by: 결정한 사람.
        reason: 사유.

    Returns:
        처분 결과. 같은 내용으로 이미 처분된 Hold면 already_processed=True로 기존 처분을 돌려준다.

    Raises:
        RuleViolation: 결정자·사유가 비었거나(I5), Hold가 없거나, 다른 내용으로 이미 처분됐거나(HOLD_CLOSED),
            검사 기간 밖의 Lot을 재검사하려 할 때(RETEST_NOT_AT_INSPECT).
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
        cur.execute("SELECT disposition_id, action, decided_by, reason FROM hold_disposition "
                    "WHERE hold_id = %s", (hold_id,))
        done = cur.fetchone()
        if done is not None:
            # 작업 지시(I3)와 같은 방식: 같은 내용의 재요청은 기존 결과, 다른 내용이면 거절.
            if (done["action"], done["decided_by"], done["reason"]) != (action, decided_by, reason):
                raise RuleViolation("HOLD_CLOSED", f"Hold {hold_id}는 이미 {done['action']}로 처분됨")
            return DispositionResult(done["disposition_id"], already_processed=True)
        # 재검사는 같은 웨이퍼를 다시 판정받는 것이므로, 판정을 받는 기간(INSPECT 시작 ~ 다음 공정 투입 전)이
        # 아니면 차수를 올리지 않는다. 다른 공정에서 차수가 오르면 나중에 받을 첫 판정(1차)이 차수 불일치로 거절된다.
        if action == "RETEST" and not _in_inspect_window(cur, lot):
            raise RuleViolation("RETEST_NOT_AT_INSPECT",
                                f"{lot['lot_id']}는 {lot['current_step_code']} {lot['status']}")

        to_close = [hold_id]
        if action == "SCRAP":
            cur.execute("SELECT hold_id FROM hold WHERE lot_id = %s AND closed_at IS NULL AND hold_id <> %s "
                        "ORDER BY hold_id", (lot["lot_id"], hold_id))
            to_close += [r["hold_id"] for r in cur.fetchall()]
        for h in to_close:
            cur.execute("INSERT INTO hold_disposition (hold_id, action, decided_by, reason) "
                        "VALUES (%s, %s, %s, %s) RETURNING disposition_id",
                        (h, action, decided_by, reason))
            if h == hold_id:
                disposition_id = cur.fetchone()["disposition_id"]
            cur.execute("UPDATE hold SET closed_at = now() WHERE hold_id = %s", (h,))
        if action == "RETEST":
            cur.execute("UPDATE lot SET inspect_round = inspect_round + 1 WHERE lot_id = %s",
                        (lot["lot_id"],))
        elif action == "SCRAP":
            cur.execute("UPDATE work_order SET status = 'ABORTED', ended_at = now() "
                        "WHERE lot_id = %s AND status = 'STARTED'", (lot["lot_id"],))
            cur.execute("UPDATE lot SET status = 'SCRAPPED' WHERE lot_id = %s", (lot["lot_id"],))
        _add_history(cur, lot["lot_id"], action, step_code=lot["current_step_code"], hold_id=hold_id)
    logger.info("Hold %d 처분 %s by %s", hold_id, action, decided_by)
    return DispositionResult(disposition_id, already_processed=False)
