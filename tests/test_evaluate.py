"""AI 판정 평가 집계 테스트(정답 정의: 라벨 있는 웨이퍼만, 라벨 0장 Lot은 따로)."""
from analysis.evaluate_holds import LotFacts, count_outcomes


def test_count_outcomes_separates_unknown_lots_and_counts_false_stops_and_misses():
    facts = {
        "OK_STOP": LotFacts(stopped=True, n_labeled=5, n_true_hit=2, n_labeled_trig=2),    # 맞게 멈춤
        "FALSE": LotFacts(stopped=True, n_labeled=5, n_true_hit=0, n_labeled_trig=1),      # 괜히 멈춤
        "FALSE_UNL": LotFacts(stopped=True, n_labeled=3, n_true_hit=0, n_labeled_trig=0),  # 라벨 없는 웨이퍼 판정 때문
        "MISS": LotFacts(stopped=False, n_labeled=4, n_true_hit=1, n_labeled_trig=0),      # 놓침
        "OK_PASS": LotFacts(stopped=False, n_labeled=4, n_true_hit=0, n_labeled_trig=0),   # 맞게 통과
        "UNKNOWN": LotFacts(stopped=True, n_labeled=0, n_true_hit=0, n_labeled_trig=0),    # 정답 알 수 없음
    }
    r = count_outcomes(facts, min_count=1)
    assert (r["lots"], r["stopped_lots"], r["unknown_truth_lots"], r["known_truth_lots"]) == (6, 4, 1, 5)
    assert (r["should_stop_lots"], r["should_not_stop_lots"]) == (2, 3)
    assert (r["false_stop_lots"], r["false_stop_by_unlabeled"], r["missed_lots"]) == (2, 1, 1)


def test_count_outcomes_uses_min_count_for_truth():
    facts = {"ONE_DEFECT": LotFacts(stopped=False, n_labeled=5, n_true_hit=1, n_labeled_trig=1)}
    assert count_outcomes(facts, min_count=2)["missed_lots"] == 0  # 정답도 2장 이상이어야 멈춰야 할 Lot
