"""생산 현황 보고(MIS): DB를 SQL로 집계해 HTML 한 장을 만든다.

MES DB만 읽는다(정답 라벨 파일은 읽지 않는다). 새 라이브러리 없이 표준 라이브러리 html.escape로 모든 값을 이스케이프한다.
사유처럼 사람이 입력한 문자열이 HTML로 해석되지 않게 하기 위해서다.

    .venv/bin/python -m mes.report
"""
import html
import logging
import sys
from pathlib import Path

import psycopg

from mes.db import PROJECT_ROOT, connect, load_config

logger = logging.getLogger("report")

# (제목, 설명, SQL). 모든 SQL은 값 바인딩이 필요 없는 집계 조회다.
SECTIONS: list[tuple[str, str, str]] = [
    ("Lot 상태", "열린 Hold가 하나라도 있는 Lot은 다음 공정 투입이 막혀 있다(I4).",
     """SELECT l.status AS "상태",
               count(*) FILTER (WHERE EXISTS (SELECT 1 FROM hold h WHERE h.lot_id = l.lot_id AND h.closed_at IS NULL))
                   AS "열린 Hold 있음",
               count(*) FILTER (WHERE NOT EXISTS (SELECT 1 FROM hold h WHERE h.lot_id = l.lot_id AND h.closed_at IS NULL))
                   AS "열린 Hold 없음",
               count(*) AS "합계"
        FROM lot l GROUP BY l.status ORDER BY l.status"""),
    ("정지 사유(규칙)별 Hold", "규칙이 연 Hold는 그때의 기준값(rule_params)을 함께 남긴다.",
     """SELECT rule_name AS "규칙", inspect_round AS "검사 차수", count(*) AS "Hold",
               count(*) FILTER (WHERE closed_at IS NULL) AS "열림", count(*) FILTER (WHERE closed_at IS NOT NULL) AS "닫힘"
        FROM hold GROUP BY rule_name, inspect_round ORDER BY rule_name, inspect_round"""),
    ("판정 유형 분포", "검사 장비가 보낸 AI 판정. 같은 웨이퍼라도 차수·모델이 다르면 따로 센다.",
     """SELECT pred_label AS "판정 유형",
               count(*) FILTER (WHERE inspect_round = 1) AS "1차", count(*) FILTER (WHERE inspect_round > 1) AS "2차 이상",
               count(*) AS "합계"
        FROM inspection_result GROUP BY pred_label ORDER BY count(*) DESC, pred_label"""),
    ("설비별 처리량", "완료한 작업 지시 수. 폐기로 중단된 작업 지시(ABORTED)는 따로 센다.",
     """SELECT e.step_code AS "공정", e.equipment_id AS "설비", e.status AS "현재 상태",
               count(w.work_order_id) FILTER (WHERE w.status = 'COMPLETED') AS "완료",
               count(w.work_order_id) FILTER (WHERE w.status = 'STARTED') AS "처리 중",
               count(w.work_order_id) FILTER (WHERE w.status = 'ABORTED') AS "중단"
        FROM equipment e JOIN route_step r USING (step_code) LEFT JOIN work_order w USING (equipment_id)
        GROUP BY e.step_code, r.seq, e.equipment_id, e.status ORDER BY r.seq, e.equipment_id"""),
    ("처분 결과", "사람이 직접 결정한 처분과, 폐기·재검사에 딸려서 닫힌 처분을 구분한다.",
     """SELECT action AS "처분",
               count(*) FILTER (WHERE cascaded_from_disposition_id IS NULL) AS "직접 결정",
               count(*) FILTER (WHERE cascaded_from_disposition_id IS NOT NULL) AS "딸려서 닫힘",
               count(*) AS "합계"
        FROM hold_disposition GROUP BY action ORDER BY action"""),
    ("처분 기록", "결정자와 사유는 사람이 입력한 그대로다.",
     """SELECT d.disposition_id AS "처분 ID", h.lot_id AS "Lot", h.hold_id AS "Hold", h.rule_name AS "규칙",
               h.inspect_round AS "차수", d.action AS "처분", d.decided_by AS "결정자",
               coalesce(d.cascaded_from_disposition_id::text, '') AS "딸려서 닫힘(원래 처분)", d.reason AS "사유",
               to_char(d.decided_at AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS') || ' UTC' AS "결정 시각"
        FROM hold_disposition d JOIN hold h USING (hold_id) ORDER BY d.disposition_id"""),
    ("설비 상태 변경 이력", "",
     """SELECT equipment_id AS "설비", from_status AS "이전", to_status AS "이후", changed_by AS "변경자",
               reason AS "사유", to_char(changed_at AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS') || ' UTC' AS "시각"
        FROM equipment_status_history ORDER BY history_id"""),
]

STYLE = """body{font-family:sans-serif;margin:24px;color:#222}h1{font-size:20px}h2{font-size:16px;margin-top:28px}
table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #ccc;padding:4px 8px;text-align:left;vertical-align:top}
th{background:#f3f3f3}td.num{text-align:right}p.note{color:#555;font-size:13px;margin:4px 0}"""


def html_table(columns: list[str], rows: list[tuple]) -> str:
    """조회 결과를 HTML 표로. 모든 값을 이스케이프하고, 숫자는 오른쪽 정렬한다."""
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body = []
    for row in rows:
        cells = []
        for v in row:
            cls = ' class="num"' if isinstance(v, int) else ""
            cells.append(f"<td{cls}>{html.escape('' if v is None else str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    if not rows:
        body.append(f'<tr><td colspan="{len(columns)}">(없음)</td></tr>')
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_report(conn: psycopg.Connection) -> str:
    """DB를 집계해 HTML 문서 문자열을 만든다.

    Args:
        conn: MES DB 연결.

    Returns:
        HTML 문서.
    """
    generated = conn.execute("SELECT to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')").fetchone()[0]
    parts = ['<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>생산 현황 보고</title>',
             f"<style>{STYLE}</style></head><body>",
             "<h1>생산 현황 보고 (MIS)</h1>",
             f'<p class="note">MES DB 집계, 생성 {html.escape(generated)} UTC. 공정·설비는 합성값이다. '
             "AI 판정은 Lot을 멈추게만 하고, 처분은 사람이 사유를 남겨 결정한다.</p>"]
    for title, note, sql in SECTIONS:
        with conn.cursor() as cur:
            cur.execute(sql)
            columns = [d.name for d in cur.description]
            rows = cur.fetchall()
        parts.append(f"<h2>{html.escape(title)}</h2>")
        if note:
            parts.append(f'<p class="note">{html.escape(note)}</p>')
        parts.append(html_table(columns, rows))
    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> int:
    """개발용 DB로 보고서를 만들어 config의 report.output에 저장한다."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    out = PROJECT_ROOT / cfg["report"]["output"]
    with connect(cfg, cfg["db"]["dbname"]) as conn:
        doc = build_report(conn)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(doc, encoding="utf-8")
    logger.info("저장 %s (%d bytes)", out.relative_to(PROJECT_ROOT), out.stat().st_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
