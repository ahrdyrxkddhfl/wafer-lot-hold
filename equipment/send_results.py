"""2단계: 검사 장비 역할 스크립트. 시험용 Lot의 판정 파일을 MES API로 보낸다.

이 스크립트는 작업 지시를 내리는 역할(Lot 생성, 공정 투입·완료)도 함께 맡는다. Lot마다
공정 순서대로 투입·완료하고, 검사 공정에 투입한 뒤 그 Lot의 판정을 배치 하나로 보낸다.
판정으로 Hold가 열린 Lot은 검사 다음 공정 투입이 I4(409)로 막혀 거기서 멈춘다.

DB에 직접 쓰지 않고, MES 설정 파일도 읽지 않는다. 공정 순서와 설비는 GET /equipment로 받는다.
정답 파일은 읽지 않는다. 작업 지시 ID를 {Lot}-{공정}으로 정해, 같은 파일을 다시 보내도 같은 요청이 된다.

웨이퍼 저장소의 .venv(requests, pandas 포함)로 이 저장소 루트에서 실행한다. MES API가 떠 있어야 한다.
    ../SKALA_CNN-Optimization/.venv/bin/python -m equipment.send_results
    ../SKALA_CNN-Optimization/.venv/bin/python -m equipment.send_results --lot lot10084 --round 2
뒤의 형태는 재검사(RETEST)로 차수가 오른 Lot 하나의 판정만 지정한 차수로 다시 보낸다(Lot 생성·투입은 하지 않음).
"""
import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
import yaml

logger = logging.getLogger("send_results")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "equipment.yaml"
PROB_PREFIX = "prob_"
HTTP_OK = 200
HTTP_CONFLICT = 409


class MesClient:
    """MES API 호출. 예상하지 못한 응답은 어디서 왜 실패했는지 남기고 멈춘다(삼키지 않는다)."""

    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()  # 연결을 재사용해 요청마다 TCP 연결을 새로 맺지 않는다

    def call(self, method: str, path: str, body: dict | None = None,
             allowed: tuple[int, ...] = (HTTP_OK,)) -> requests.Response:
        """요청을 보내고, allowed에 없는 상태 코드면 RuntimeError로 멈춘다."""
        r = self.session.request(method, self.base_url + path, json=body, timeout=self.timeout)
        if r.status_code not in allowed:
            raise RuntimeError(f"{method} {path} → {r.status_code} {r.text[:300]}")
        return r


def load_lots(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """판정 파일을 읽어 Lot·웨이퍼 순서로 정렬한다.

    Returns:
        판정 DataFrame과 판정 유형 목록(prob_ 열 순서).
    """
    df = pd.read_csv(path).sort_values(["lot_name", "wafer_index"]).reset_index(drop=True)
    labels = [c[len(PROB_PREFIX):] for c in df.columns if c.startswith(PROB_PREFIX)]
    return df, labels


def batch_body(lot: pd.DataFrame, labels: list[str], work_order_id: str, inspect_round: int) -> dict:
    """Lot 하나의 판정 배치 요청."""
    hashes = lot["model_sha256"].unique()
    if len(hashes) != 1:
        raise ValueError(f"{lot['lot_name'].iat[0]}: 모델 해시가 하나가 아님 {hashes}")
    return {"work_order_id": work_order_id, "inspect_round": inspect_round, "model_sha256": hashes[0],
            "results": [{"wafer_id": row["wafer_id"], "pred_label": row["pred_label"],
                         "probabilities": {c: float(row[PROB_PREFIX + c]) for c in labels}}
                        for _, row in lot.iterrows()]}


def send_lot(mes: MesClient, lot_id: str, lot: pd.DataFrame, labels: list[str], steps: list[dict],
             inspect_step: str, lot_index: int, counts: Counter) -> None:
    """Lot 하나를 만들고 공정 순서대로 진행하며, 검사 공정에서 판정을 보낸다."""
    wafers = [{"wafer_id": w, "wafer_index": int(i)} for w, i in zip(lot["wafer_id"], lot["wafer_index"])]
    mes.call("POST", "/lots", {"lot_id": lot_id, "wafers": wafers})
    for step in steps:
        available = [e["equipment_id"] for e in step["equipment"] if e["status"] == "AVAILABLE"]
        if not available:
            raise RuntimeError(f"{step['step_code']}에 쓸 수 있는 설비가 없음")
        wo = f"{lot_id}-{step['step_code']}"
        r = mes.call("POST", "/work-orders", {"work_order_id": wo, "lot_id": lot_id, "step_code": step["step_code"],
                                              "equipment_id": available[lot_index % len(available)]},
                     allowed=(HTTP_OK, HTTP_CONFLICT))
        if r.status_code == HTTP_CONFLICT:
            if r.json().get("rule") != "I4":
                raise RuntimeError(f"{wo} 투입 거절: {r.text[:300]}")
            counts["held_lots"] += 1  # 열린 Hold 때문에 다음 공정 투입이 막혔다
            return
        if step["step_code"] == inspect_step:
            result = mes.call("POST", f"/lots/{lot_id}/inspection-results",
                              batch_body(lot, labels, wo, inspect_round=1)).json()
            counts["results_sent"] += result["received"]
            counts["inserted"] += result["inserted"]
            counts["duplicates"] += result["duplicates"]
            if result["hold"] is not None:
                counts["new_holds" if result["hold"]["created"] else "existing_holds"] += 1
        mes.call("POST", f"/work-orders/{wo}/complete")
    counts["finished_lots"] += 1


def send_round(mes: MesClient, df: pd.DataFrame, labels: list[str], inspect_step: str, lot_id: str,
               inspect_round: int) -> None:
    """Lot 하나의 판정을 지정한 차수로 다시 보낸다(재검사 뒤 새 차수 판정)."""
    lot = df[df["lot_name"] == lot_id]
    if lot.empty:
        raise ValueError(f"판정 파일에 없는 Lot: {lot_id}")
    result = mes.call("POST", f"/lots/{lot_id}/inspection-results",
                      batch_body(lot, labels, f"{lot_id}-{inspect_step}", inspect_round)).json()
    logger.info("%s %d차 판정 전송: %s", lot_id, inspect_round, json.dumps(result, ensure_ascii=False))


def main() -> int:
    """판정 파일 전체(또는 --lot으로 지정한 Lot 하나의 한 차수)를 보내고 건수를 출력한다."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="검사 판정을 MES API로 보낸다")
    parser.add_argument("--lot", help="이 Lot 하나의 판정만 보낸다(--round와 함께)")
    parser.add_argument("--round", type=int, help="--lot의 판정을 보낼 검사 차수(재검사 뒤 2 이상)")
    args = parser.parse_args()
    if (args.lot is None) != (args.round is None):
        parser.error("--lot과 --round는 함께 줘야 한다")

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    api = cfg["mes_api"]
    mes = MesClient(api["base_url"], api["timeout_seconds"])

    layout = mes.call("GET", "/equipment").json()
    steps = sorted(layout["steps"], key=lambda s: s["seq"])
    df, labels = load_lots(PROJECT_ROOT / cfg["output"]["lot_predictions"])
    if args.lot is not None:
        send_round(mes, df, labels, layout["inspect_step"], args.lot, args.round)
        return 0
    lots = list(df.groupby("lot_name", sort=True))
    logger.info("공정 %s, 검사 공정 %s, Lot %d개, 판정 %d건", [s["step_code"] for s in steps],
                layout["inspect_step"], len(lots), len(df))

    counts: Counter = Counter()
    start = time.perf_counter()
    for i, (lot_id, lot) in enumerate(lots):
        send_lot(mes, lot_id, lot, labels, steps, layout["inspect_step"], i, counts)
        if (i + 1) % api["progress_every_lots"] == 0:
            logger.info("%d / %d Lot", i + 1, len(lots))
    logger.info("Lot %d개 처리, %.1f초", len(lots), time.perf_counter() - start)
    logger.info("보낸 판정 %d, 새 기록 %d, 중복 %d, 새 Hold %d, 기존 Hold %d, FINISHED Lot %d, Hold로 멈춘 Lot %d",
                counts["results_sent"], counts["inserted"], counts["duplicates"], counts["new_holds"],
                counts["existing_holds"], counts["finished_lots"], counts["held_lots"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
