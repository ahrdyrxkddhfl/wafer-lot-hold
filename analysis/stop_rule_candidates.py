"""정지 규칙 후보를 검증용 Lot에서 비교한다(4단계 보완).

기준값을 시험용 Lot으로 고르면 시험 결과에 맞춰 고른 셈이 되므로 검증용 Lot 파일만 읽는다.
정답 파일을 읽는 평가용 스크립트이며, MES 코드(mes/)와는 분리돼 있다.

웨이퍼 저장소의 .venv(pandas 포함)로 이 저장소 루트에서 실행한다.
    ../SKALA_CNN-Optimization/.venv/bin/python -m analysis.stop_rule_candidates

정의
    정지: 정지 대상 유형으로 판정되고 그 판정 확률이 하한 이상인 웨이퍼가 Lot당 최소 장수 이상.
    멈춰야 할 Lot(정답): 라벨 있는 웨이퍼만으로 같은 규칙(유형, 최소 장수)을 적용한 결과. 라벨 없는 웨이퍼는 알 수 없으므로 뺀다.
    정답을 알 수 없는 Lot(라벨 있는 웨이퍼가 0장)은 괜히 멈춤·놓침 계산에서 뺀다.
"""
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger("stop_rule_candidates")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EQUIPMENT_CONFIG = PROJECT_ROOT / "config" / "equipment.yaml"
CANDIDATE_CONFIG = PROJECT_ROOT / "config" / "stop_rule_candidates.yaml"
NONE_LABEL = "none"
LABELED = "labeled"
OUTPUT = PROJECT_ROOT / "evaluation" / "stop_rule_candidates_valid.csv"


def load_yaml(path: Path) -> dict:
    """YAML 파일을 읽는다."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_valid(equipment_cfg: dict) -> pd.DataFrame:
    """검증용 Lot의 판정과 정답을 웨이퍼 ID로 붙인다.

    Returns:
        lot_name, pred_label, pred_prob(판정한 유형의 확률), true_label, labeled 열.
    """
    out = equipment_cfg["output"]
    pred = pd.read_csv(PROJECT_ROOT / out["lot_valid_predictions"])
    labels = pd.read_csv(PROJECT_ROOT / out["lot_valid_labels"])
    d = pred.merge(labels, on="wafer_id", validate="one_to_one")
    prob_cols = {c[len("prob_"):]: c for c in pred.columns if c.startswith("prob_")}
    d["pred_prob"] = [row[prob_cols[label]] for label, row in zip(d["pred_label"], d.to_dict("records"))]
    d["labeled"] = d["label_source"] == LABELED
    return d[["lot_name", "pred_label", "pred_prob", "true_label", "labeled"]]


def defect_types(d: pd.DataFrame) -> list[str]:
    """none을 뺀 불량 유형 목록(판정 파일의 확률 열 순서와 무관하게 이름순)."""
    types = sorted((set(d["pred_label"]) | set(d["true_label"])) - {NONE_LABEL})
    assert NONE_LABEL not in types
    return types


def false_alarm_order(d: pd.DataFrame, types: list[str]) -> pd.Series:
    """라벨 있는 none 웨이퍼가 불량으로 잘못 판정된 횟수를 유형별로, 많은 순서로."""
    fa = d[d["labeled"] & (d["true_label"] == NONE_LABEL) & (d["pred_label"] != NONE_LABEL)]
    return fa["pred_label"].value_counts().reindex(types).fillna(0).astype(int).sort_values(ascending=False)


def evaluate_rule(d: pd.DataFrame, types: list[str], min_prob: float, min_count: int,
                  reference_should_stop: pd.Series) -> dict:
    """규칙 하나를 Lot 단위로 적용해 정지·괜히 멈춤·놓침을 센다.

    Args:
        d: load_valid 결과.
        types: 정지 대상 유형.
        min_prob: 판정 확률 하한.
        min_count: Lot당 최소 장수.
        reference_should_stop: 기준 규칙(전체 유형, 1장)으로 정의한 멈춰야 할 Lot. 유형·장수를 바꾸면
            "멈춰야 할 Lot"의 정의도 바뀌므로, 놓침을 같은 잣대로 비교하기 위해 함께 센다.

    Returns:
        표 한 줄.
    """
    trig = d["pred_label"].isin(types) & (d["pred_prob"] >= min_prob)
    true_hit = d["labeled"] & d["true_label"].isin(types)
    lot = pd.DataFrame({
        "n_trig": trig.groupby(d["lot_name"]).sum(),
        "n_trig_labeled": (trig & d["labeled"]).groupby(d["lot_name"]).sum(),
        "n_true": true_hit.groupby(d["lot_name"]).sum(),
        "n_labeled": d["labeled"].groupby(d["lot_name"]).sum(),
    })
    stopped = lot["n_trig"] >= min_count
    known = lot["n_labeled"] > 0
    should = lot["n_true"] >= min_count
    false_stop = stopped & known & ~should
    miss = ~stopped & known & should
    return {
        "stopped_lot_pct": round(100 * stopped.mean(), 1),
        "should_stop_lots": int((known & should).sum()),
        # 괜히 멈출 수 있는 Lot의 수. 이 데이터는 불량 Lot 위주라 작으므로 괜히 멈춘 수와 함께 봐야 한다.
        "should_not_stop_lots": int((known & ~should).sum()),
        "false_stop_lots": int(false_stop.sum()),
        # 괜히 멈춘 Lot 중, 라벨 있는 웨이퍼의 판정만으로는 멈추지 않았을 Lot(정지 근거가 라벨 없는 웨이퍼 판정)
        "false_stop_by_unlabeled": int((false_stop & (lot["n_trig_labeled"] < min_count)).sum()),
        "missed_lots": int(miss.sum()),
        "missed_vs_reference": int((~stopped & known & reference_should_stop.reindex(lot.index)).sum()),
    }


def main() -> None:
    """후보별 표를 출력하고 CSV로 저장한다."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cand = load_yaml(CANDIDATE_CONFIG)
    d = load_valid(load_yaml(EQUIPMENT_CONFIG))
    types = defect_types(d)

    per_lot = d.groupby("lot_name")["labeled"].agg(["sum", "size"])
    logger.info("검증용 Lot %d개, 웨이퍼 %d장. 라벨 있는 웨이퍼가 0장인 Lot(정답 알 수 없음) %d개, "
                "일부만 라벨 있는 Lot %d개", len(per_lot), len(d), int((per_lot["sum"] == 0).sum()),
                int(((per_lot["sum"] > 0) & (per_lot["sum"] < per_lot["size"])).sum()))
    order = false_alarm_order(d, types)
    logger.info("라벨 있는 none 웨이퍼의 유형별 오판정 수(많은 순)\n%s", order.to_string())

    lot_names = d["lot_name"].unique()
    ref_hit = (d["labeled"] & d["true_label"].isin(types)).groupby(d["lot_name"]).sum()
    reference_should_stop = (ref_hit >= 1).reindex(lot_names)

    rows = []
    for k in cand["exclude_top_false_alarm_types"]:
        excluded = list(order.index[:k])
        rows.append({"variable": "(a) 유형", "candidate": f"전체-{k}" + (f" ({', '.join(excluded)} 제외)" if k else ""),
                     **evaluate_rule(d, [t for t in types if t not in excluded], 0.0, 1, reference_should_stop)})
    for p in cand["min_probs"]:
        rows.append({"variable": "(b) 확률 하한", "candidate": f"{p:.1f}",
                     **evaluate_rule(d, types, p, 1, reference_should_stop)})
    for c in cand["min_counts"]:
        rows.append({"variable": "(c) 최소 장수", "candidate": f"{c}장",
                     **evaluate_rule(d, types, 0.0, c, reference_should_stop)})

    table = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        logger.info("정지 규칙 후보(검증용 Lot, 기준: 전체 유형·하한 0.0·1장)\n%s", table.to_string(index=False))
    logger.info("저장 %s (%d bytes)", OUTPUT.relative_to(PROJECT_ROOT), OUTPUT.stat().st_size)


if __name__ == "__main__":
    main()
