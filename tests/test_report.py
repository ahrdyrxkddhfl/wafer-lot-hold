"""생산 현황 보고(MIS) 테스트: 사람이 입력한 문자열 이스케이프, 집계 숫자."""
from mes.report import build_report
from mes.service import create_lot, dispose_hold, open_hold


def test_report_escapes_user_input_and_counts_dispositions(conn):
    create_lot(conn, "LOT_A", [("LOT_A_W01", 1)])
    create_lot(conn, "LOT_B", [("LOT_B_W01", 1)])
    hold = open_hold(conn, "LOT_A", "MANUAL", opened_by="tester")
    open_hold(conn, "LOT_B", "MANUAL", opened_by="tester")
    dispose_hold(conn, hold.hold_id, "RELEASE", "<b>operator</b>", "<script>alert(1)</script> 확인 완료")

    doc = build_report(conn)

    assert "<script>alert(1)</script>" not in doc and "<b>operator</b>" not in doc
    assert "&lt;script&gt;alert(1)&lt;/script&gt; 확인 완료" in doc
    # Lot 상태: WAITING 2개 중 열린 Hold 있음 1(LOT_B), 없음 1(LOT_A)
    assert '<tr><td>WAITING</td><td class="num">1</td><td class="num">1</td><td class="num">2</td></tr>' in doc
    # 처분 결과: RELEASE 직접 결정 1, 딸려서 닫힘 0
    assert '<tr><td>RELEASE</td><td class="num">1</td><td class="num">0</td><td class="num">1</td></tr>' in doc
