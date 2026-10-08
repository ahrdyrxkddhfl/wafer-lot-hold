"""AI 판정 평가(3단계). MES가 실제로 멈춘 Lot을 정답 라벨과 비교한다. 현장에는 정답이 없으므로 평가용이다.

MIS(mes/report.py)와 분리된 스크립트다. 정답 파일은 이 스크립트만 읽고 MES 코드는 읽지 않는다.
기준은 DB에 기록된 첫 차수(inspect_round=1)의 판정 규칙 Hold다. 그 뒤의 처분·재검사 시연과 상관없이 결과가 같다.
정답 정의는 4단계 보완과 같다: 라벨 있는 웨이퍼에 같은 규칙(정지 대상 유형, 최소 장수)을 적용하고,
라벨 없는 웨이퍼는 알 수 없으므로 빼며, 라벨 있는 웨이퍼가 0장인 Lot은 괜히 멈춤·놓침 계산에서 빼고 따로 센다.

이 저장소의 .venv(psycopg 포함, pandas 없음)로 실행한다.
    .venv/bin/python -m analysis.evaluate_holds
"""
import csv
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from mes.db import PROJECT_ROOT, connect, load_config

logger = logging.getLogger("evaluate_holds")

EQUIPMENT_CONFIG = PROJECT_ROOT / "config" / "equipment.yaml"
OUTPUT = PROJECT_ROOT / "evaluation" / "hold_evaluation_test.csv"
FIRST_ROUND = 1
LABELED = "labeled"


@dataclass(frozen=True)
class LotFacts:
    """Lot 하나의 평가 재료."""
    stopped: bool          # 첫 차수 판정 규칙 Hold가 열렸는가
    n_labeled: int         # 라벨 있는 웨이퍼 수
    n_true_hit: int        # 라벨 있는 웨이퍼 중 정답이 정지 대상 유형인 수
    n_labeled_trig: int    # 라벨 있는 웨이퍼 중 AI 판정이 정지 대상(확률 하한 이상)인 수


def count_outcomes(facts: dict[str, LotFacts], min_count: int) -> dict[str, int]:
    """Lot 단위로 정지·괜히 멈춤·놓침을 센다.

    Args:
        facts: Lot ID → 평가 재료.
        min_count: Lot당 최소 장수(정답 정의에도 같은 값을 쓴다).

    Returns:
        지표 이름 → 값.
    """
    known = {k: f for k, f in facts.items() if f.n_labeled > 0}
    should = {k for k, f in known.items() if f.n_true_hit >= min_count}
    false_stop = [k for k, f in known.items() if f.stopped and k not in should]
    return {
        "lots": len(facts),
        "stopped_lots": sum(f.stopped for f in facts.values()),
        "unknown_truth_lots": len(facts) - len(known),
        "known_truth_lots": len(known),
        "should_stop_lots": len(should),
        "should_not_stop_lots": len(known) - len(should),
        "false_stop_lots": len(false_stop),
        # 괜히 멈춘 Lot 중 라벨 있는 웨이퍼의 판정만으로는 멈추지 않았을 Lot(정지 근거가 라벨 없는 웨이퍼 판정)
        "false_stop_by_unlabeled": sum(facts[k].n_labeled_trig < min_count for k in false_stop),
        "missed_lots": sum(1 for k in should if not known[k].stopped),
    }


def rule_hits_from_file(path: Path, defect_types: set[str], min_prob: float) -> dict[str, set[str]]:
    """판정 파일에 같은 규칙을 다시 적용: Lot → 정지 대상 판정을 받은 웨이퍼 집합(교차 확인용)."""
    hits: dict[str, set[str]] = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row["pred_label"]
            if label in defect_types and float(row["prob_" + label]) >= min_prob:
                hits[row["lot_name"]].add(row["wafer_id"])
    return hits


def main() -> int:
    """평가 표를 출력하고 CSV로 저장한다."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    with open(EQUIPMENT_CONFIG, encoding="utf-8") as f:
        out_cfg = yaml.safe_load(f)["output"]

    with connect(cfg, cfg["db"]["dbname"]) as conn:
        lot_of = dict(conn.execute("SELECT wafer_id, lot_id FROM wafer").fetchall())
        stopped_rows = conn.execute("SELECT lot_id, rule_params FROM hold WHERE trigger_result_id IS NOT NULL "
                                    "AND inspect_round = %s", (FIRST_ROUND,)).fetchall()
        first_preds = conn.execute("SELECT wafer_id, pred_label, (probabilities ->> pred_label)::float8 "
                                   "FROM inspection_result WHERE inspect_round = %s", (FIRST_ROUND,)).fetchall()

    # 기준값은 Hold에 남은 스냅샷에서 읽는다(config를 나중에 바꿔도 그때의 기준으로 평가).
    params = {(tuple(p["defect_types"]), p["min_prob"], p["min_count"]) for _, p in stopped_rows}
    if len(params) != 1:
        raise SystemExit(f"첫 차수 규칙 Hold의 기준값이 하나가 아님: {params}")
    types, min_prob, min_count = next(iter(params))
    defect_types = set(types)
    stopped = {lot for lot, _ in stopped_rows}

    labels = {}
    with open(PROJECT_ROOT / out_cfg["lot_labels"], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["wafer_id"]] = (row["true_label"], row["label_source"] == LABELED)
    if set(labels) != set(lot_of):
        raise SystemExit("정답 파일의 웨이퍼와 DB의 웨이퍼가 다름")
    trig = {w for w, label, p in first_preds if label in defect_types and p >= min_prob}

    n_labeled, n_true, n_trig = defaultdict(int), defaultdict(int), defaultdict(int)
    for wafer, lot in lot_of.items():
        true_label, labeled = labels[wafer]
        if labeled:
            n_labeled[lot] += 1
            n_true[lot] += true_label in defect_types
            n_trig[lot] += wafer in trig
    lots = set(lot_of.values())
    facts = {lot: LotFacts(lot in stopped, n_labeled[lot], n_true[lot], n_trig[lot]) for lot in lots}
    result = count_outcomes(facts, min_count)

    file_hits = rule_hits_from_file(PROJECT_ROOT / out_cfg["lot_predictions"], defect_types, min_prob)
    file_stopped = {lot for lot, wafers in file_hits.items() if len(wafers) >= min_count}
    agree = sum((lot in stopped) == (lot in file_stopped) for lot in lots)

    logger.info("평가용: 현장에는 정답이 없으므로 이 결과는 평가를 위한 것이다(MES는 정답을 읽지 않는다).")
    logger.info("기준: DB의 첫 차수 판정 규칙 Hold. 규칙 기준값(Hold 스냅샷) 유형 %d종, 확률 하한 %s, 최소 %d장",
                len(defect_types), min_prob, min_count)
    logger.info("주의: 이 규칙의 시험용 Lot 정지율(82.6%)은 4단계 종료 보고에서 이미 계산해 본 상태다. "
                "기준값은 검증용 Lot으로 골랐지만 이 규칙의 시험용 결과는 미리 봤다.")
    logger.info("교차 확인: 판정 파일에 같은 규칙을 다시 적용한 결과와 Lot 단위 일치 %d / %d (불일치 %d)",
                agree, len(lots), len(lots) - agree)
    for key, value in result.items():
        logger.info("  %-24s %d", key, value)
    logger.info("정지율 %.1f%%, 정답을 아는 Lot 중 멈추지 않아야 할 Lot %d개 중 괜히 멈춤 %d개, "
                "멈춰야 할 Lot %d개 중 놓침 %d개", 100 * result["stopped_lots"] / result["lots"],
                result["should_not_stop_lots"], result["false_stop_lots"], result["should_stop_lots"],
                result["missed_lots"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(result.items())
        w.writerow(["file_rule_lot_agreement", agree])
    logger.info("저장 %s", OUTPUT.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
