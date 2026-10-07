"""막아야 할 상태(I1~I13), 판정 수신·정지 규칙, 정상 흐름 테스트. 실제 PostgreSQL 테스트 DB에 대해 돈다."""
import dataclasses
import threading
import time
from collections.abc import Callable

import psycopg
import pytest

from mes import service
from mes.errors import RuleViolation
from mes.service import (ResultIn, StopRule, create_lot, dispose_hold, open_hold, receive_inspection_results,
                         set_equipment_status, track_in, track_out)

# 잠금이 없을 때 두 요청이 모두 검사를 통과하도록, 행을 읽은 직후 기다리는 시간.
# 잠금이 있으면 두 번째 요청은 이 시간 동안 잠금에서 기다린다. 너무 짧으면 경쟁이 재현되지 않을 수 있다.
RACE_WINDOW_SECONDS = 0.3
THREAD_TIMEOUT_SECONDS = 10
# 잠금을 기다리면 안 되는 요청에 거는 대기 한도. 넘으면 LockNotAvailable로 실패해 "기다렸다"는 것이 드러난다.
LOCK_TIMEOUT_MS = 500
# 동시성 테스트에서 순서를 강제할 때 나중에 출발할 쪽의 출발 지연. RACE_WINDOW_SECONDS보다 작아야
# 먼저 출발한 쪽이 잠금을 쥐고 기다리는 동안 나중 쪽이 도착해 실제로 잠금을 기다리게 된다.
START_DELAY_SECONDS = 0.1
MODEL_A = "a" * 64  # 테스트용 모델 해시
MODEL_B = "b" * 64


@pytest.fixture
def route(mes_cfg: dict) -> list[str]:
    return mes_cfg["route"]


@pytest.fixture
def eqs(mes_cfg: dict) -> dict[str, list[str]]:
    return mes_cfg["equipment"]


@pytest.fixture
def labels(mes_cfg: dict) -> list[str]:
    return mes_cfg["labels"]


@pytest.fixture
def rule(mes_cfg: dict) -> StopRule:
    return StopRule.from_config(mes_cfg["stop_rule"])


def make_lot(conn: psycopg.Connection, lot_id: str, n_wafers: int = 2) -> None:
    create_lot(conn, lot_id, [(f"{lot_id}_W{i:02d}", i) for i in range(1, n_wafers + 1)])


def judged(wafer_id: str, label: str, labels: list[str]) -> ResultIn:
    """판정 유형의 확률이 1.0인 판정."""
    return ResultIn(wafer_id, label, {c: (1.0 if c == label else 0.0) for c in labels})


def send(conn: psycopg.Connection, lot_id: str, label_by_wafer: dict[str, str], labels: list[str],
         rule: StopRule, inspect_round: int = 1, model: str = MODEL_A, wo_id: str | None = None):
    """Lot의 INSPECT 작업 지시(기본 ID {lot_id}-INSPECT)로 판정 배치를 보낸다."""
    return receive_inspection_results(
        conn, lot_id, wo_id or f"{lot_id}-{service.INSPECT_STEP}", inspect_round, model,
        [judged(w, lab, labels) for w, lab in label_by_wafer.items()], labels, rule)


def count(conn: psycopg.Connection, query: str, params: tuple = ()) -> int:
    return conn.execute(query, params).fetchone()[0]


def lot_row(conn: psycopg.Connection, lot_id: str) -> tuple:
    return conn.execute("SELECT status, current_step_code, inspect_round FROM lot WHERE lot_id = %s",
                        (lot_id,)).fetchone()


def manual_hold(conn: psycopg.Connection, lot_id: str) -> int:
    return open_hold(conn, lot_id, "MANUAL", opened_by="tester").hold_id


def slow_after(original: Callable) -> Callable:
    """행을 잠그고 읽은 뒤 RACE_WINDOW_SECONDS만큼 기다리게 감싼다(경쟁 구간 넓히기)."""
    def wrapper(cur, key):
        row = original(cur, key)
        time.sleep(RACE_WINDOW_SECONDS)
        return row
    return wrapper


def run_at_same_time(connect_test: Callable[[], psycopg.Connection],
                     calls: list[Callable[[psycopg.Connection], object]],
                     start_delays: list[float] | None = None) -> list[tuple[str, object]]:
    """각 호출을 스레드·연결 하나씩에서 배리어로 동시에 출발시키고 (결과 종류, 값) 목록을 돌려준다.

    start_delays를 주면 배리어 뒤에 그만큼 늦게 출발해, 어느 쪽이 먼저 잠금을 잡을지 정할 수 있다.
    """
    barrier = threading.Barrier(len(calls))
    outcomes: list[tuple[str, object]] = [("not_run", None)] * len(calls)
    delays = start_delays or [0.0] * len(calls)

    def worker(i: int, call: Callable[[psycopg.Connection], object]) -> None:
        with connect_test() as c:
            barrier.wait()
            time.sleep(delays[i])
            try:
                outcomes[i] = ("ok", call(c))
            except Exception as e:  # 결과로 기록해 테스트에서 종류를 확인한다
                outcomes[i] = ("error", e)

    threads = [threading.Thread(target=worker, args=(i, call)) for i, call in enumerate(calls)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(THREAD_TIMEOUT_SECONDS)
        assert not t.is_alive(), "스레드가 시간 안에 끝나지 않음(교착 의심)"
    return outcomes


# ── 정상 흐름 ──────────────────────────────────────────────

def test_lot_goes_through_all_steps_to_finished(conn, route, eqs, labels, rule):
    make_lot(conn, "LOT_A")
    for i, step in enumerate(route):
        track_in(conn, f"WO-{i}", "LOT_A", step, eqs[step][0])
        if step == service.INSPECT_STEP:  # 다음 공정에 들어가려면 모든 웨이퍼의 판정이 있어야 한다(I13)
            send(conn, "LOT_A", {"LOT_A_W01": "none", "LOT_A_W02": "none"}, labels, rule, wo_id=f"WO-{i}")
        track_out(conn, f"WO-{i}")

    assert lot_row(conn, "LOT_A")[:2] == ("FINISHED", None)
    events = [r[0] for r in conn.execute(
        "SELECT event FROM lot_history WHERE lot_id = 'LOT_A' ORDER BY event_id").fetchall()]
    assert events == ["CREATED"] + ["TRACK_IN", "TRACK_OUT"] * len(route)
    assert count(conn, "SELECT count(*) FROM work_order WHERE status = 'COMPLETED'") == len(route)


# ── I1 공정 순서 건너뛰기 ─────────────────────────────────────

def test_i1_cannot_skip_step(conn, route, eqs):
    make_lot(conn, "LOT_A")
    with pytest.raises(RuleViolation) as e:
        track_in(conn, "WO-1", "LOT_A", route[1], eqs[route[1]][0])
    assert e.value.rule == "I1"
    assert count(conn, "SELECT count(*) FROM work_order") == 0


# ── I2 끝난 공정 다시 처리 ────────────────────────────────────

def test_i2_cannot_reprocess_finished_step(conn, route, eqs):
    make_lot(conn, "LOT_A")
    track_in(conn, "WO-1", "LOT_A", route[0], eqs[route[0]][0])
    track_out(conn, "WO-1")

    with pytest.raises(RuleViolation) as e:
        track_in(conn, "WO-2", "LOT_A", route[0], eqs[route[0]][1])
    assert e.value.rule == "I2"


def test_i2_db_rejects_second_work_order_for_same_lot_and_step(conn, route, eqs):
    make_lot(conn, "LOT_A")
    track_in(conn, "WO-1", "LOT_A", route[0], eqs[route[0]][0])
    track_out(conn, "WO-1")
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute("INSERT INTO work_order (work_order_id, lot_id, step_code, equipment_id, status, ended_at) "
                     "VALUES ('WO-X', 'LOT_A', %s, %s, 'COMPLETED', now())", (route[0], eqs[route[0]][1]))


# ── I3 같은 작업 지시 두 번 처리 ──────────────────────────────

def test_i3_same_track_in_twice_returns_same_result(conn, route, eqs):
    make_lot(conn, "LOT_A")
    first = track_in(conn, "WO-1", "LOT_A", route[0], eqs[route[0]][0])
    second = track_in(conn, "WO-1", "LOT_A", route[0], eqs[route[0]][0])

    assert not first.already_processed and second.already_processed
    assert (first.work_order_id, first.lot_id, first.step_code, first.equipment_id) == \
           (second.work_order_id, second.lot_id, second.step_code, second.equipment_id)
    assert count(conn, "SELECT count(*) FROM work_order") == 1
    assert count(conn, "SELECT count(*) FROM lot_history WHERE event = 'TRACK_IN'") == 1


def test_i3_same_id_with_different_content_is_rejected(conn, route, eqs):
    make_lot(conn, "LOT_A")
    make_lot(conn, "LOT_B")
    track_in(conn, "WO-1", "LOT_A", route[0], eqs[route[0]][0])
    with pytest.raises(RuleViolation) as e:
        track_in(conn, "WO-1", "LOT_B", route[0], eqs[route[0]][1])
    assert e.value.rule == "I3"


def test_i3_same_track_out_twice_returns_same_result(conn, route, eqs):
    make_lot(conn, "LOT_A")
    track_in(conn, "WO-1", "LOT_A", route[0], eqs[route[0]][0])
    first = track_out(conn, "WO-1")
    second = track_out(conn, "WO-1")

    assert not first.already_processed and second.already_processed
    assert lot_row(conn, "LOT_A")[:2] == ("WAITING", route[1])
    assert count(conn, "SELECT count(*) FROM lot_history WHERE event = 'TRACK_OUT'") == 1


# ── I4 Hold된 Lot 진행 ───────────────────────────────────────

def test_i4_hold_blocks_track_in_but_not_track_out(conn, route, eqs):
    make_lot(conn, "LOT_A")
    make_lot(conn, "LOT_B")
    track_in(conn, "WO-A1", "LOT_A", route[0], eqs[route[0]][0])
    hold_a = manual_hold(conn, "LOT_A")
    manual_hold(conn, "LOT_B")

    # 완료(설비에서 내리기)는 막지 않는다. 다음 공정 투입만 막는다.
    assert not track_out(conn, "WO-A1").already_processed
    with pytest.raises(RuleViolation) as e_next:
        track_in(conn, "WO-A2", "LOT_A", route[1], eqs[route[1]][0])
    with pytest.raises(RuleViolation) as e_in:
        track_in(conn, "WO-B1", "LOT_B", route[0], eqs[route[0]][1])
    assert e_next.value.rule == "I4" and e_in.value.rule == "I4"

    dispose_hold(conn, hold_a, "RELEASE", "engineer", "판정 확인 결과 문제 없음")
    assert not track_in(conn, "WO-A2", "LOT_A", route[1], eqs[route[1]][0]).already_processed


def test_i4_release_of_one_hold_keeps_lot_blocked_by_another(conn, route, eqs):
    eq1 = eqs[route[0]][0]
    make_lot(conn, "LOT_A")
    track_in(conn, "WO-A1", "LOT_A", route[0], eq1)
    manual_id = manual_hold(conn, "LOT_A")
    set_equipment_status(conn, eq1, "DOWN", "engineer", "고장")
    track_out(conn, "WO-A1")

    dispose_hold(conn, manual_id, "RELEASE", "engineer", "확인 완료")
    with pytest.raises(RuleViolation) as e:
        track_in(conn, "WO-A2", "LOT_A", route[1], eqs[route[1]][0])
    assert e.value.rule == "I4"  # EQUIPMENT_DOWN Hold가 아직 열려 있다


# ── I5 사유·결정자 없는 처분 ──────────────────────────────────

@pytest.mark.parametrize("decided_by,reason", [
    ("", "사유"), ("   ", "사유"), (None, "사유"),
    ("engineer", ""), ("engineer", "   "), ("engineer", None),
])
def test_i5_disposition_needs_decider_and_reason(conn, decided_by, reason):
    make_lot(conn, "LOT_A")
    hold_id = manual_hold(conn, "LOT_A")
    with pytest.raises(RuleViolation) as e:
        dispose_hold(conn, hold_id, "RELEASE", decided_by, reason)
    assert e.value.rule == "I5"
    assert count(conn, "SELECT count(*) FROM hold_disposition") == 0
    assert count(conn, "SELECT count(*) FROM hold WHERE closed_at IS NULL") == 1


def test_i5_db_rejects_blank_or_null_reason(conn):
    make_lot(conn, "LOT_A")
    hold_id = manual_hold(conn, "LOT_A")
    insert = ("INSERT INTO hold_disposition (hold_id, action, decided_by, reason) "
              "VALUES (%s, 'RELEASE', 'engineer', %s)")
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(insert, (hold_id, "  "))
    with pytest.raises(psycopg.errors.NotNullViolation):
        conn.execute(insert, (hold_id, None))


# ── 처분 재요청 ─────────────────────────────────────────────

def test_same_disposition_twice_returns_existing(conn):
    make_lot(conn, "LOT_A")
    hold_id = manual_hold(conn, "LOT_A")
    first = dispose_hold(conn, hold_id, "RELEASE", "engineer", "확인 완료")
    second = dispose_hold(conn, hold_id, "RELEASE", "engineer", "확인 완료")

    assert not first.already_processed and second.already_processed
    assert first.disposition_id == second.disposition_id
    assert count(conn, "SELECT count(*) FROM hold_disposition") == 1
    assert count(conn, "SELECT count(*) FROM lot_history WHERE event = 'RELEASE'") == 1


@pytest.mark.parametrize("action,decided_by,reason", [
    ("SCRAP", "engineer", "확인 완료"),
    ("RELEASE", "other", "확인 완료"),
    ("RELEASE", "engineer", "다른 사유"),
])
def test_different_disposition_on_closed_hold_is_rejected(conn, action, decided_by, reason):
    make_lot(conn, "LOT_A")
    hold_id = manual_hold(conn, "LOT_A")
    dispose_hold(conn, hold_id, "RELEASE", "engineer", "확인 완료")

    with pytest.raises(RuleViolation) as e:
        dispose_hold(conn, hold_id, action, decided_by, reason)
    assert e.value.rule == "HOLD_CLOSED"
    assert lot_row(conn, "LOT_A")[0] == "WAITING"
    assert count(conn, "SELECT count(*) FROM hold_disposition") == 1


# ── I6 정지·정비 중인 설비에 투입 ─────────────────────────────

@pytest.mark.parametrize("status", ["DOWN", "MAINTENANCE"])
def test_i6_cannot_track_in_to_unavailable_equipment(conn, route, eqs, status):
    eq1, eq2 = eqs[route[0]][:2]
    make_lot(conn, "LOT_A")
    set_equipment_status(conn, eq1, status, "engineer", "점검")

    with pytest.raises(RuleViolation) as e:
        track_in(conn, "WO-1", "LOT_A", route[0], eq1)
    assert e.value.rule == "I6"
    assert not track_in(conn, "WO-2", "LOT_A", route[0], eq2).already_processed
    assert conn.execute("SELECT from_status, to_status, changed_by FROM equipment_status_history "
                        "WHERE equipment_id = %s", (eq1,)).fetchall() == [("AVAILABLE", status, "engineer")]


def test_equipment_of_other_step_is_rejected(conn, route, eqs):
    make_lot(conn, "LOT_A")
    with pytest.raises(RuleViolation) as e:
        track_in(conn, "WO-1", "LOT_A", route[0], eqs[route[1]][0])
    assert e.value.rule == "EQUIPMENT_STEP"


def test_equipment_step_is_checked_without_waiting_for_equipment_lock(conn, connect_test, route, eqs):
    """다른 공정 설비로의 투입은 그 설비의 잠금을 기다리지 않고 바로 거절돼야 한다.

    기다리게 되면 "Lot을 잡은 채 설비를 기다리는" 경로가 생겨 설비 DOWN 처리(설비 → Lot)와 교착할 수 있다.
    """
    other_eq = eqs[route[1]][0]
    make_lot(conn, "LOT_A")  # route[0] 대기
    conn.execute("SELECT set_config('lock_timeout', %s, false)", (f"{LOCK_TIMEOUT_MS}ms",))
    with connect_test() as holder, holder.transaction():
        holder.execute("SELECT 1 FROM equipment WHERE equipment_id = %s FOR NO KEY UPDATE", (other_eq,))
        with pytest.raises(RuleViolation) as e:
            track_in(conn, "WO-1", "LOT_A", route[0], other_eq)
    assert e.value.rule == "EQUIPMENT_STEP"


# ── I7 두 설비가 같은 Lot을 동시에 처리 ────────────────────────

def test_i7_concurrent_track_in_of_same_lot_only_one_succeeds(conn, connect_test, monkeypatch, route, eqs):
    make_lot(conn, "LOT_A")
    eq1, eq2 = eqs[route[0]][:2]
    monkeypatch.setattr(service, "_lock_lot", slow_after(service._lock_lot))

    outcomes = run_at_same_time(connect_test, [
        lambda c: track_in(c, "WO-1", "LOT_A", route[0], eq1),
        lambda c: track_in(c, "WO-2", "LOT_A", route[0], eq2),
    ])

    kinds = sorted(k for k, _ in outcomes)
    errors = [v for k, v in outcomes if k == "error"]
    assert kinds == ["error", "ok"], outcomes
    assert isinstance(errors[0], RuleViolation) and errors[0].rule == "I7", errors[0]
    assert count(conn, "SELECT count(*) FROM work_order WHERE lot_id = 'LOT_A' AND status = 'STARTED'") == 1
    assert count(conn, "SELECT count(*) FROM lot_history WHERE event = 'TRACK_IN'") == 1


# ── I9 한 Lot에 열린 Hold 두 개 ──────────────────────────────

def test_i9_second_open_hold_of_same_type_returns_existing(conn):
    make_lot(conn, "LOT_A")
    first = open_hold(conn, "LOT_A", "MANUAL", opened_by="tester")
    second = open_hold(conn, "LOT_A", "MANUAL", opened_by="tester")

    assert first.created and not second.created and first.hold_id == second.hold_id
    assert count(conn, "SELECT count(*) FROM hold WHERE closed_at IS NULL") == 1

    dispose_hold(conn, first.hold_id, "RELEASE", "engineer", "확인 완료")
    third = open_hold(conn, "LOT_A", "MANUAL", opened_by="tester")
    assert third.created and third.hold_id != first.hold_id


def test_i9_holds_of_different_types_open_together(conn, route, eqs):
    eq1 = eqs[route[0]][0]
    make_lot(conn, "LOT_A")
    track_in(conn, "WO-A", "LOT_A", route[0], eq1)
    manual_hold(conn, "LOT_A")
    down = set_equipment_status(conn, eq1, "DOWN", "engineer", "고장")

    assert down is not None and down.created
    assert sorted(r[0] for r in conn.execute(
        "SELECT rule_name FROM hold WHERE closed_at IS NULL").fetchall()) == ["EQUIPMENT_DOWN", "MANUAL"]


def test_i9_db_rejects_second_open_hold_of_same_type(conn):
    make_lot(conn, "LOT_A")
    manual_hold(conn, "LOT_A")
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute("INSERT INTO hold (lot_id, rule_name, opened_by, inspect_round) "
                     "VALUES ('LOT_A', 'MANUAL', 'tester', 1)")


# ── Hold를 연 주체는 정확히 하나 ──────────────────────────────

@pytest.mark.parametrize("rule_name,kwargs", [
    ("MANUAL", {}),                                          # 둘 다 없음
    ("MANUAL", {"trigger_result_id": 1, "opened_by": "x"}),  # 둘 다 있음
    ("DEFECT_RULE", {"opened_by": "tester"}),                # 사람이 열었는데 MANUAL이 아님
    ("MANUAL", {"opened_by": "   "}),                        # 공백 이름
    ("MANUAL", {"trigger_equipment_history_id": 1, "opened_by": "x"}),  # 둘 다 있음
    ("DEFECT_RULE", {"trigger_equipment_history_id": 1}),    # 설비 고장인데 EQUIPMENT_DOWN이 아님
    ("EQUIPMENT_DOWN", {"trigger_result_id": 1}),            # EQUIPMENT_DOWN인데 설비 이력이 없음
])
def test_hold_needs_exactly_one_opener(conn, rule_name, kwargs):
    make_lot(conn, "LOT_A")
    with pytest.raises(RuleViolation) as e:
        open_hold(conn, "LOT_A", rule_name, **kwargs)
    assert e.value.rule == "INVALID"
    assert count(conn, "SELECT count(*) FROM hold") == 0


def test_hold_db_rejects_missing_opener(conn):
    make_lot(conn, "LOT_A")
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute("INSERT INTO hold (lot_id, rule_name, inspect_round) VALUES ('LOT_A', 'MANUAL', 1)")


def test_hold_db_rejects_equipment_down_rule_without_equipment_history(conn):
    make_lot(conn, "LOT_A")
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute("INSERT INTO hold (lot_id, rule_name, opened_by, inspect_round) "
                     "VALUES ('LOT_A', 'EQUIPMENT_DOWN', 'tester', 1)")


# ── I10 폐기된 Lot 진행 ──────────────────────────────────────

def test_i10_scrapped_lot_is_terminal(conn, route, eqs):
    make_lot(conn, "LOT_A")
    track_in(conn, "WO-1", "LOT_A", route[0], eqs[route[0]][0])
    dispose_hold(conn, manual_hold(conn, "LOT_A"), "SCRAP", "engineer", "불량 확정")

    assert lot_row(conn, "LOT_A")[0] == "SCRAPPED"
    assert count(conn, "SELECT count(*) FROM work_order WHERE status = 'ABORTED'") == 1
    for action in (lambda: track_out(conn, "WO-1"),
                   lambda: track_in(conn, "WO-2", "LOT_A", route[0], eqs[route[0]][1]),
                   lambda: open_hold(conn, "LOT_A", "MANUAL", opened_by="tester")):
        with pytest.raises(RuleViolation) as e:
            action()
        assert e.value.rule == "I10"
    assert lot_row(conn, "LOT_A")[0] == "SCRAPPED"


# ── I11 한 설비에 두 Lot 동시 처리 ─────────────────────────────

def test_i11_busy_equipment_rejects_another_lot(conn, route, eqs):
    eq1, eq2 = eqs[route[0]][:2]
    make_lot(conn, "LOT_A")
    make_lot(conn, "LOT_B")
    track_in(conn, "WO-A", "LOT_A", route[0], eq1)

    with pytest.raises(RuleViolation) as e:
        track_in(conn, "WO-B", "LOT_B", route[0], eq1)
    assert e.value.rule == "I11"
    assert not track_in(conn, "WO-B2", "LOT_B", route[0], eq2).already_processed


def test_i11_concurrent_track_in_to_same_equipment_only_one_succeeds(conn, connect_test, monkeypatch,
                                                                     route, eqs):
    eq1 = eqs[route[0]][0]
    make_lot(conn, "LOT_A")
    make_lot(conn, "LOT_B")
    monkeypatch.setattr(service, "_lock_equipment", slow_after(service._lock_equipment))

    outcomes = run_at_same_time(connect_test, [
        lambda c: track_in(c, "WO-A", "LOT_A", route[0], eq1),
        lambda c: track_in(c, "WO-B", "LOT_B", route[0], eq1),
    ])

    kinds = sorted(k for k, _ in outcomes)
    errors = [v for k, v in outcomes if k == "error"]
    assert kinds == ["error", "ok"], outcomes
    assert isinstance(errors[0], RuleViolation) and errors[0].rule == "I11", errors[0]
    assert count(conn, "SELECT count(*) FROM work_order WHERE equipment_id = %s AND status = 'STARTED'",
                 (eq1,)) == 1


# ── I12 처리 중인 설비를 계획 정비로 / 설비 고장 시 자동 Hold ────────────

def test_i12_maintenance_rejected_while_equipment_processing(conn, route, eqs):
    eq1, eq2 = eqs[route[0]][:2]
    make_lot(conn, "LOT_A")
    track_in(conn, "WO-A", "LOT_A", route[0], eq1)

    with pytest.raises(RuleViolation) as e:
        set_equipment_status(conn, eq1, "MAINTENANCE", "engineer", "정기 점검")
    assert e.value.rule == "I12"
    assert count(conn, "SELECT count(*) FROM equipment WHERE equipment_id = %s AND status = 'AVAILABLE'",
                 (eq1,)) == 1
    assert count(conn, "SELECT count(*) FROM equipment_status_history") == 0
    assert set_equipment_status(conn, eq2, "MAINTENANCE", "engineer", "정기 점검") is None


def test_down_opens_equipment_down_hold_on_processing_lot(conn, route, eqs):
    eq1 = eqs[route[0]][0]
    make_lot(conn, "LOT_A")
    track_in(conn, "WO-A", "LOT_A", route[0], eq1)

    result = set_equipment_status(conn, eq1, "DOWN", "engineer", "고장")
    assert result is not None and result.created
    history_id = conn.execute("SELECT history_id FROM equipment_status_history WHERE equipment_id = %s",
                              (eq1,)).fetchone()[0]
    assert conn.execute("SELECT lot_id, rule_name, trigger_equipment_history_id, closed_at FROM hold "
                        "WHERE hold_id = %s", (result.hold_id,)).fetchone() == \
        ("LOT_A", "EQUIPMENT_DOWN", history_id, None)
    assert count(conn, "SELECT count(*) FROM lot_history WHERE event = 'HOLD'") == 1
    assert not track_out(conn, "WO-A").already_processed  # 고장 난 설비에서 내리는 것은 막지 않는다
    with pytest.raises(RuleViolation) as e:
        track_in(conn, "WO-A2", "LOT_A", route[1], eqs[route[1]][0])
    assert e.value.rule == "I4"


def test_down_on_idle_equipment_opens_no_hold(conn, route, eqs):
    assert set_equipment_status(conn, eqs[route[0]][0], "DOWN", "engineer", "고장") is None
    assert count(conn, "SELECT count(*) FROM hold") == 0


def test_down_opens_its_own_hold_even_if_manual_hold_is_open(conn, route, eqs):
    eq1 = eqs[route[0]][0]
    make_lot(conn, "LOT_A")
    track_in(conn, "WO-A", "LOT_A", route[0], eq1)
    manual_id = manual_hold(conn, "LOT_A")

    result = set_equipment_status(conn, eq1, "DOWN", "engineer", "고장")
    assert result is not None and result.created and result.hold_id != manual_id
    assert count(conn, "SELECT count(*) FROM hold WHERE closed_at IS NULL") == 2


def no_deadlock(outcomes: list[tuple[str, object]]) -> None:
    for kind, value in outcomes:
        assert not isinstance(value, psycopg.errors.DeadlockDetected), outcomes


@pytest.mark.parametrize("first", ["track_in", "down"])
def test_down_and_track_in_to_same_equipment_in_both_orders(conn, connect_test, monkeypatch, route, eqs, first):
    """먼저 출발한 쪽이 설비를 잡고 RACE_WINDOW 동안 쥐고 있는 사이에 나중 쪽이 도착해 기다린다."""
    eq1 = eqs[route[0]][0]
    make_lot(conn, "LOT_B")
    monkeypatch.setattr(service, "_lock_equipment", slow_after(service._lock_equipment))

    outcomes = run_at_same_time(connect_test, [
        lambda c: track_in(c, "WO-B", "LOT_B", route[0], eq1),
        lambda c: set_equipment_status(c, eq1, "DOWN", "engineer", "고장"),
    ], start_delays=[0.0, START_DELAY_SECONDS] if first == "track_in" else [START_DELAY_SECONDS, 0.0])

    no_deadlock(outcomes)
    (in_kind, in_value), (down_kind, _) = outcomes
    assert down_kind == "ok", outcomes
    open_down_holds = count(conn, "SELECT count(*) FROM hold WHERE lot_id = 'LOT_B' "
                                  "AND rule_name = 'EQUIPMENT_DOWN' AND closed_at IS NULL")
    if first == "track_in":  # 투입이 먼저: 고장 처리가 그 Lot에 Hold를 연다
        assert in_kind == "ok" and open_down_holds == 1, outcomes
    else:                    # 고장이 먼저: 투입이 I6으로 거절된다
        assert isinstance(in_value, RuleViolation) and in_value.rule == "I6", outcomes
        assert open_down_holds == 0 and count(conn, "SELECT count(*) FROM work_order") == 0


@pytest.mark.parametrize("first", ["track_out", "down"])
def test_down_and_track_out_on_same_equipment_in_both_orders(conn, connect_test, monkeypatch, route, eqs, first):
    """고장 처리는 설비 → Lot, 완료는 Lot만 잠근다. 어느 쪽이 Lot을 먼저 잡아도 교착 없이 끝나야 한다."""
    eq1 = eqs[route[0]][0]
    make_lot(conn, "LOT_A")
    track_in(conn, "WO-A", "LOT_A", route[0], eq1)
    monkeypatch.setattr(service, "_lock_lot", slow_after(service._lock_lot))

    outcomes = run_at_same_time(connect_test, [
        lambda c: track_out(c, "WO-A"),
        lambda c: set_equipment_status(c, eq1, "DOWN", "engineer", "고장"),
    ], start_delays=[0.0, START_DELAY_SECONDS] if first == "track_out" else [START_DELAY_SECONDS, 0.0])

    no_deadlock(outcomes)
    (out_kind, _), (down_kind, _) = outcomes
    assert out_kind == "ok" and down_kind == "ok", outcomes  # 완료는 Hold와 상관없이 성공한다
    assert lot_row(conn, "LOT_A")[:2] == ("WAITING", route[1])
    open_holds = count(conn, "SELECT count(*) FROM hold WHERE lot_id = 'LOT_A' AND closed_at IS NULL")
    # 완료가 먼저면 고장 처리는 Lot을 잠근 뒤 다시 확인해 Hold를 열지 않고, 고장이 먼저면 Hold가 열린다.
    assert open_holds == (0 if first == "track_out" else 1)


# ── 처분 RETEST ─────────────────────────────────────────────

def track_in_to_inspect(conn: psycopg.Connection, lot_id: str, route: list[str],
                        eqs: dict[str, list[str]]) -> str:
    """검사 공정 앞까지 처리하고 검사 공정에 투입한다. 검사 공정 작업 지시 ID를 돌려준다."""
    for step in route[:route.index(service.INSPECT_STEP)]:
        track_in(conn, f"{lot_id}-{step}", lot_id, step, eqs[step][0])
        track_out(conn, f"{lot_id}-{step}")
    wo_id = f"{lot_id}-{service.INSPECT_STEP}"
    track_in(conn, wo_id, lot_id, service.INSPECT_STEP, eqs[service.INSPECT_STEP][0])
    return wo_id


def test_retest_increments_round_and_keeps_lot_in_process(conn, route, eqs):
    make_lot(conn, "LOT_A")
    wo_id = track_in_to_inspect(conn, "LOT_A", route, eqs)
    dispose_hold(conn, manual_hold(conn, "LOT_A"), "RETEST", "engineer", "재검사 요청")

    assert lot_row(conn, "LOT_A") == ("IN_PROCESS", service.INSPECT_STEP, 2)
    assert count(conn, "SELECT count(*) FROM hold WHERE closed_at IS NULL") == 0
    assert not track_out(conn, wo_id).already_processed


def test_retest_only_allowed_at_inspect(conn):
    service.create_lot(conn, "LOT_R", [("LOT_R_W01", 1)])
    h = service.open_hold(conn, "LOT_R", "MANUAL", opened_by="engineer")
    with pytest.raises(RuleViolation) as e:
        service.dispose_hold(conn, h.hold_id, "RETEST", "engineer", "재검사 요청")
    assert e.value.rule == "RETEST_NOT_AT_INSPECT"
    row = conn.execute("SELECT current_step_code, inspect_round FROM lot WHERE lot_id='LOT_R'").fetchone()
    assert row == ("STEP_1", 1)
    assert count(conn, "SELECT count(*) FROM hold WHERE closed_at IS NULL") == 1
    assert count(conn, "SELECT count(*) FROM hold_disposition") == 0


def test_retest_allowed_after_inspect_completed_before_next_step(conn, route, eqs):
    make_lot(conn, "LOT_A")
    wo_id = track_in_to_inspect(conn, "LOT_A", route, eqs)
    track_out(conn, wo_id)
    dispose_hold(conn, manual_hold(conn, "LOT_A"), "RETEST", "engineer", "재검사 요청")
    assert lot_row(conn, "LOT_A")[2] == 2


def test_scrap_closes_all_open_holds_of_lot(conn, route, eqs):
    eq1 = eqs[route[0]][0]
    make_lot(conn, "LOT_A")
    track_in(conn, "WO-A", "LOT_A", route[0], eq1)
    manual_id = manual_hold(conn, "LOT_A")
    set_equipment_status(conn, eq1, "DOWN", "engineer", "고장")

    dispose_hold(conn, manual_id, "SCRAP", "engineer", "불량 확정")
    assert count(conn, "SELECT count(*) FROM hold WHERE closed_at IS NULL") == 0
    assert conn.execute("SELECT action, decided_by, reason FROM hold_disposition ORDER BY disposition_id"
                        ).fetchall() == [("SCRAP", "engineer", "불량 확정")] * 2


# ── 2단계: 판정 수신과 정지 규칙 ─────────────────────────────────

def lot_at_inspect(conn: psycopg.Connection, lot_id: str, route: list[str], eqs: dict[str, list[str]],
                   n_wafers: int = 2) -> str:
    make_lot(conn, lot_id, n_wafers)
    return track_in_to_inspect(conn, lot_id, route, eqs)


def wafers(lot_id: str, label: str, n: int = 2) -> dict[str, str]:
    return {f"{lot_id}_W{i:02d}": label for i in range(1, n + 1)}


def result_id(conn: psycopg.Connection, wafer_id: str, model: str = MODEL_A, inspect_round: int = 1) -> int:
    return conn.execute("SELECT result_id FROM inspection_result WHERE wafer_id = %s AND model_sha256 = %s "
                        "AND inspect_round = %s", (wafer_id, model, inspect_round)).fetchone()[0]


def test_results_without_defect_open_no_hold(conn, route, eqs, labels, rule):
    lot_at_inspect(conn, "LOT_A", route, eqs)
    r = send(conn, "LOT_A", wafers("LOT_A", "none"), labels, rule)
    assert (r.received, r.inserted, r.duplicates, r.hold) == (2, 2, 0, None)


def test_defect_result_opens_rule_hold_with_trigger_and_params(conn, route, eqs, labels, rule):
    lot_at_inspect(conn, "LOT_A", route, eqs)
    r = send(conn, "LOT_A", {"LOT_A_W01": "none", "LOT_A_W02": "Center"}, labels, rule)

    assert r.hold is not None and r.hold.created
    assert conn.execute("SELECT rule_name, trigger_result_id, rule_params, inspect_round FROM hold "
                        "WHERE hold_id = %s", (r.hold.hold_id,)).fetchone() == \
        (rule.name, result_id(conn, "LOT_A_W02"), rule.params(), 1)
    assert count(conn, "SELECT count(*) FROM lot_history WHERE event = 'HOLD'") == 1


def test_i8_same_batch_resent_counts_duplicates_and_keeps_hold(conn, route, eqs, labels, rule):
    lot_at_inspect(conn, "LOT_A", route, eqs)
    first = send(conn, "LOT_A", {"LOT_A_W01": "Center", "LOT_A_W02": "none"}, labels, rule)
    second = send(conn, "LOT_A", {"LOT_A_W01": "Center", "LOT_A_W02": "none"}, labels, rule)

    assert (second.inserted, second.duplicates) == (0, 2)
    assert second.hold == service.HoldResult(first.hold.hold_id, created=False)
    assert count(conn, "SELECT count(*) FROM inspection_result") == 2
    assert count(conn, "SELECT count(*) FROM hold") == 1


def test_i8_results_of_different_model_hash_are_recorded_separately(conn, route, eqs, labels, rule):
    lot_at_inspect(conn, "LOT_A", route, eqs)
    send(conn, "LOT_A", wafers("LOT_A", "none"), labels, rule, model=MODEL_A)
    r = send(conn, "LOT_A", wafers("LOT_A", "none"), labels, rule, model=MODEL_B)
    assert r.inserted == 2
    assert conn.execute("SELECT model_sha256, count(*) FROM inspection_result GROUP BY 1 ORDER BY 1"
                        ).fetchall() == [(MODEL_A, 2), (MODEL_B, 2)]


def test_i8_same_key_with_different_content_rejects_whole_batch(conn, route, eqs, labels, rule):
    lot_at_inspect(conn, "LOT_A", route, eqs, n_wafers=3)
    send(conn, "LOT_A", {"LOT_A_W01": "none"}, labels, rule)
    with pytest.raises(RuleViolation) as e:
        send(conn, "LOT_A", {"LOT_A_W01": "Center", "LOT_A_W02": "none"}, labels, rule)
    assert e.value.rule == "I8"
    assert conn.execute("SELECT wafer_id, pred_label FROM inspection_result").fetchall() == [("LOT_A_W01", "none")]
    assert count(conn, "SELECT count(*) FROM hold") == 0


def test_round_mismatch_is_rejected(conn, route, eqs, labels, rule):
    lot_at_inspect(conn, "LOT_A", route, eqs)
    with pytest.raises(RuleViolation) as e:
        send(conn, "LOT_A", wafers("LOT_A", "none"), labels, rule, inspect_round=2)
    assert e.value.rule == "ROUND_MISMATCH"


def test_results_outside_inspect_window_are_rejected(conn, route, eqs, labels, rule):
    wo_id = lot_at_inspect(conn, "LOT_A", route, eqs)
    with pytest.raises(RuleViolation) as e_wo:  # INSPECT가 아닌 작업 지시로 보냄
        send(conn, "LOT_A", wafers("LOT_A", "none"), labels, rule, wo_id=f"LOT_A-{route[0]}")
    send(conn, "LOT_A", wafers("LOT_A", "none"), labels, rule)
    track_out(conn, wo_id)
    nxt = route[route.index(service.INSPECT_STEP) + 1]
    track_in(conn, "WO-NEXT", "LOT_A", nxt, eqs[nxt][0])
    with pytest.raises(RuleViolation) as e_late:  # 다음 공정에 들어간 뒤
        send(conn, "LOT_A", wafers("LOT_A", "none"), labels, rule, model=MODEL_B)
    assert e_wo.value.rule == "RESULT_NOT_ACCEPTED" and e_late.value.rule == "RESULT_NOT_ACCEPTED"


def test_batch_with_one_bad_wafer_is_rejected_entirely(conn, route, eqs, labels, rule):
    lot_at_inspect(conn, "LOT_A", route, eqs)
    make_lot(conn, "LOT_B")
    with pytest.raises(RuleViolation) as e_other:  # 다른 Lot의 웨이퍼가 섞임
        send(conn, "LOT_A", {"LOT_A_W01": "Center", "LOT_B_W01": "none"}, labels, rule)
    with pytest.raises(RuleViolation) as e_label:  # 약속에 없는 판정 유형
        receive_inspection_results(conn, "LOT_A", f"LOT_A-{service.INSPECT_STEP}", 1, MODEL_A,
                                   [ResultIn("LOT_A_W01", "Unknown", {c: 0.0 for c in labels})], labels, rule)
    assert e_other.value.rule == "NOT_FOUND" and e_label.value.rule == "INVALID"
    assert count(conn, "SELECT count(*) FROM inspection_result") == 0
    assert count(conn, "SELECT count(*) FROM hold") == 0


def test_min_count_counts_across_batches(conn, route, eqs, labels, rule):
    rule2 = dataclasses.replace(rule, min_count=2)
    lot_at_inspect(conn, "LOT_A", route, eqs)
    assert send(conn, "LOT_A", {"LOT_A_W01": "Center"}, labels, rule2).hold is None
    r = send(conn, "LOT_A", {"LOT_A_W02": "Scratch"}, labels, rule2)
    assert r.hold is not None and r.hold.created
    assert conn.execute("SELECT trigger_result_id, rule_params ->> 'min_count' FROM hold").fetchone() == \
        (result_id(conn, "LOT_A_W02"), "2")


def test_min_count_counts_distinct_wafers_not_results(conn, route, eqs, labels, rule):
    rule2 = dataclasses.replace(rule, min_count=2)
    lot_at_inspect(conn, "LOT_A", route, eqs)
    assert send(conn, "LOT_A", {"LOT_A_W01": "Center"}, labels, rule2, model=MODEL_A).hold is None
    assert send(conn, "LOT_A", {"LOT_A_W01": "Center"}, labels, rule2, model=MODEL_B).hold is None
    r = send(conn, "LOT_A", {"LOT_A_W02": "Center"}, labels, rule2, model=MODEL_A)
    assert r.hold is not None and r.hold.created
    assert conn.execute("SELECT trigger_result_id FROM hold").fetchone()[0] == result_id(conn, "LOT_A_W02")


def test_rule_hold_is_not_reopened_in_same_round_after_release(conn, route, eqs, labels, rule):
    lot_at_inspect(conn, "LOT_A", route, eqs)
    first = send(conn, "LOT_A", {"LOT_A_W01": "Center", "LOT_A_W02": "none"}, labels, rule)
    dispose_hold(conn, first.hold.hold_id, "RELEASE", "engineer", "엔지니어 확인 결과 정상")

    again = send(conn, "LOT_A", {"LOT_A_W01": "Center", "LOT_A_W02": "none"}, labels, rule)
    other_model = send(conn, "LOT_A", {"LOT_A_W01": "Center"}, labels, rule, model=MODEL_B)
    assert again.hold == other_model.hold == service.HoldResult(first.hold.hold_id, created=False)
    assert count(conn, "SELECT count(*) FROM hold") == 1
    assert count(conn, "SELECT count(*) FROM hold WHERE closed_at IS NULL") == 0


def test_retest_opens_new_rule_hold_in_new_round(conn, route, eqs, labels, rule):
    lot_at_inspect(conn, "LOT_A", route, eqs)
    first = send(conn, "LOT_A", wafers("LOT_A", "Center"), labels, rule)
    dispose_hold(conn, first.hold.hold_id, "RETEST", "engineer", "재검사 요청")

    r = send(conn, "LOT_A", wafers("LOT_A", "Center"), labels, rule, inspect_round=2)
    assert r.inserted == 2 and r.hold is not None and r.hold.created
    assert conn.execute("SELECT inspect_round FROM hold WHERE hold_id = %s", (r.hold.hold_id,)).fetchone()[0] == 2


def test_result_after_inspect_completed_opens_hold_and_blocks_next_step(conn, route, eqs, labels, rule):
    wo_id = lot_at_inspect(conn, "LOT_A", route, eqs)
    track_out(conn, wo_id)
    r = send(conn, "LOT_A", {"LOT_A_W01": "Center", "LOT_A_W02": "none"}, labels, rule)
    assert r.hold is not None and r.hold.created
    nxt = route[route.index(service.INSPECT_STEP) + 1]
    with pytest.raises(RuleViolation) as e:
        track_in(conn, "WO-NEXT", "LOT_A", nxt, eqs[nxt][0])
    assert e.value.rule == "I4"


def test_defect_result_during_equipment_down_hold_is_not_hidden(conn, route, eqs, labels, rule):
    inspect_eq = eqs[service.INSPECT_STEP][0]
    nxt = route[route.index(service.INSPECT_STEP) + 1]
    wo_id = lot_at_inspect(conn, "LOT_A", route, eqs)
    down = set_equipment_status(conn, inspect_eq, "DOWN", "engineer", "검사 중 고장")
    r = send(conn, "LOT_A", {"LOT_A_W01": "Center", "LOT_A_W02": "none"}, labels, rule)
    assert r.hold is not None and r.hold.created and r.hold.hold_id != down.hold_id

    dispose_hold(conn, down.hold_id, "RELEASE", "engineer", "설비 복구")
    track_out(conn, wo_id)
    with pytest.raises(RuleViolation) as e:  # 규칙 Hold가 아직 열려 있다
        track_in(conn, "WO-NEXT", "LOT_A", nxt, eqs[nxt][0])
    assert e.value.rule == "I4"

    dispose_hold(conn, r.hold.hold_id, "RELEASE", "engineer", "판정 확인 결과 정상")
    assert not track_in(conn, "WO-NEXT", "LOT_A", nxt, eqs[nxt][0]).already_processed


# ── I13 검사 결과 없이 검사 공정 통과 ───────────────────────────

def next_after_inspect(conn: psycopg.Connection, wo_id: str, route: list[str], eqs: dict[str, list[str]]):
    track_out(conn, wo_id)
    nxt = route[route.index(service.INSPECT_STEP) + 1]
    return track_in(conn, "WO-NEXT", "LOT_A", nxt, eqs[nxt][0])


@pytest.mark.parametrize("label_by_wafer", [{}, {"LOT_A_W01": "none"}], ids=["no_results", "some_wafers"])
def test_i13_next_step_needs_results_for_all_wafers(conn, route, eqs, labels, rule, label_by_wafer):
    wo_id = lot_at_inspect(conn, "LOT_A", route, eqs)
    if label_by_wafer:
        send(conn, "LOT_A", label_by_wafer, labels, rule)
    with pytest.raises(RuleViolation) as e:
        next_after_inspect(conn, wo_id, route, eqs)
    assert e.value.rule == "I13"


def test_i13_all_wafers_judged_without_defect_pass(conn, route, eqs, labels, rule):
    wo_id = lot_at_inspect(conn, "LOT_A", route, eqs)
    send(conn, "LOT_A", wafers("LOT_A", "none"), labels, rule)
    assert not next_after_inspect(conn, wo_id, route, eqs).already_processed


def test_i13_after_retest_needs_results_of_new_round(conn, route, eqs, labels, rule):
    wo_id = lot_at_inspect(conn, "LOT_A", route, eqs)
    send(conn, "LOT_A", wafers("LOT_A", "none"), labels, rule)
    dispose_hold(conn, manual_hold(conn, "LOT_A"), "RETEST", "engineer", "재검사 요청")
    with pytest.raises(RuleViolation) as e:
        next_after_inspect(conn, wo_id, route, eqs)
    assert e.value.rule == "I13"


# ── 규칙 Hold의 DB 제약 ─────────────────────────────────────

def test_hold_db_rejects_rule_hold_without_params(conn, route, eqs, labels, rule):
    lot_at_inspect(conn, "LOT_A", route, eqs)
    send(conn, "LOT_A", wafers("LOT_A", "none"), labels, rule)
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute("INSERT INTO hold (lot_id, rule_name, trigger_result_id, inspect_round) "
                     "VALUES ('LOT_A', 'DEFECT_TYPE_IN_LOT', %s, 1)", (result_id(conn, "LOT_A_W01"),))


def test_hold_db_rejects_second_rule_hold_in_same_round(conn, route, eqs, labels, rule):
    lot_at_inspect(conn, "LOT_A", route, eqs)
    first = send(conn, "LOT_A", wafers("LOT_A", "Center"), labels, rule)
    dispose_hold(conn, first.hold.hold_id, "RELEASE", "engineer", "확인 완료")
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute("INSERT INTO hold (lot_id, rule_name, trigger_result_id, rule_params, inspect_round) "
                     "VALUES ('LOT_A', %s, %s, '{}', 1)", (rule.name, result_id(conn, "LOT_A_W02")))


# ── 판정 수신 동시성 ────────────────────────────────────────

def test_concurrent_batches_for_same_lot_open_exactly_one_hold(conn, connect_test, monkeypatch,
                                                               route, eqs, labels, rule):
    """min_count=2, 두 배치에 정지 대상이 1장씩. Lot 잠금이 없으면 두 쪽 모두 1장만 세어 Hold를 놓친다."""
    rule2 = dataclasses.replace(rule, min_count=2)
    lot_at_inspect(conn, "LOT_A", route, eqs)
    monkeypatch.setattr(service, "_lock_lot", slow_after(service._lock_lot))

    outcomes = run_at_same_time(connect_test, [
        lambda c: send(c, "LOT_A", {"LOT_A_W01": "Center"}, labels, rule2),
        lambda c: send(c, "LOT_A", {"LOT_A_W02": "Center"}, labels, rule2),
    ])

    no_deadlock(outcomes)
    assert [k for k, _ in outcomes] == ["ok", "ok"], outcomes
    assert count(conn, "SELECT count(*) FROM hold WHERE trigger_result_id IS NOT NULL") == 1


@pytest.mark.parametrize("first", ["results", "down"])
def test_results_and_inspect_equipment_down_in_both_orders(conn, connect_test, monkeypatch,
                                                           route, eqs, labels, rule, first):
    inspect_eq = eqs[service.INSPECT_STEP][0]
    lot_at_inspect(conn, "LOT_A", route, eqs)
    monkeypatch.setattr(service, "_lock_lot", slow_after(service._lock_lot))

    outcomes = run_at_same_time(connect_test, [
        lambda c: send(c, "LOT_A", {"LOT_A_W01": "Center", "LOT_A_W02": "none"}, labels, rule),
        lambda c: set_equipment_status(c, inspect_eq, "DOWN", "engineer", "검사 중 고장"),
    ], start_delays=[0.0, START_DELAY_SECONDS] if first == "results" else [START_DELAY_SECONDS, 0.0])

    no_deadlock(outcomes)
    assert [k for k, _ in outcomes] == ["ok", "ok"], outcomes
    # 순서와 상관없이 규칙 Hold와 고장 Hold가 각각 열린다(종류별 하나).
    assert sorted(r[0] for r in conn.execute("SELECT rule_name FROM hold WHERE closed_at IS NULL").fetchall()) \
        == sorted([rule.name, "EQUIPMENT_DOWN"])
