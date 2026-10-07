"""4단계: 검사 모델 Lot 단위 재평가.

웨이퍼 저장소의 .venv로 이 저장소 루트에서 실행한다.
    ../SKALA_CNN-Optimization/.venv/bin/python -m equipment.stage4 {plan|train|evaluate|predict}

plan:     Lot 이름 재사용 확인, 분할 결과, 두 Test의 겹침, 시험용 Lot 전체 미리보기, 1에폭 시간(저장 안 함)
train:    Lot 단위 분할로 12번과 같은 설정으로 학습(MPS), 체크포인트는 artifacts/, 에폭별 기록은 CSV
evaluate: 12번 모델과 새 모델의 Test 성능 비교(CPU)
predict:  시험용 Lot의 원본 전체 웨이퍼를 새 모델로 판정(CPU), 판정 파일과 정답 파일을 따로 저장
검증이 실패하면 이유를 남기고 exit 1로 멈춘다.
"""
import argparse
import logging
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from equipment.inspector import load_model, predict, sha256_of
from equipment.prepare_stage0 import (CONFIG_PATH, Stage0Failure, compare_split, load_config,
                                      macro_f1, make_wafer_ids)
from equipment.wafer_repo import (PROJECT_ROOT, UNLABELED, build_reduced, load_lswmd, make_xy,
                                  preprocess_wafer_map, resolve_wafer_repo, split_by_lot,
                                  split_like_notebook)

logger = logging.getLogger("stage4")

SPLITS = ["train", "valid", "test"]
HISTORY_KEYS = ["tr_loss", "tr_acc", "va_loss", "va_acc", "va_f1", "va_minor", "lr"]


@dataclass
class Prepared:
    """4단계 하위 명령이 함께 쓰는 데이터."""
    df: pd.DataFrame                 # 원본 전체
    df_reduced: pd.DataFrame         # 축소 데이터 30,519장
    X: np.ndarray
    y: np.ndarray
    class_names: list[str]
    wafer_rows: dict[str, np.ndarray]  # 12번의 웨이퍼 단위 분할(축소 데이터 행 번호)
    lot_rows: dict[str, np.ndarray]    # Lot 단위 분할(축소 데이터 행 번호)

    def lot_data(self) -> dict[str, np.ndarray]:
        """Lot 단위 분할을 make_loaders가 받는 형태로."""
        return {f"{k}_{s}": (self.X if k == "X" else self.y)[self.lot_rows[f"row_{s}"]]
                for k in ("X", "y") for s in SPLITS}


def check_lot_name_reuse(df: pd.DataFrame) -> None:
    """원본 전체에서 (lotName, waferIndex)가 겹치는 행이 있으면 멈춘다.

    시험용 Lot의 웨이퍼를 원본 전체에서 가져오므로, 겹치면 서로 다른 Lot이 한 Lot으로 섞여 들어온다.

    Args:
        df: load_lswmd 결과(원본 전체).

    Raises:
        Stage0Failure: 겹치는 행이 있을 때.
    """
    n_dup = int(df.duplicated(["lotName", "waferIndex"], keep=False).sum())
    logger.info("원본 %d행 중 (lotName, waferIndex)가 겹치는 행 %d", len(df), n_dup)
    if n_dup:
        raise Stage0Failure(f"(lotName, waferIndex)가 겹치는 행 {n_dup}개. Lot 이름이 재사용됐을 수 있음")


def log_split_report(df_reduced: pd.DataFrame, rows: dict[str, np.ndarray], class_names: list[str],
                     min_class_warn: int) -> None:
    """분할별 Lot 수·웨이퍼 수·클래스 분포와 분할 사이 Lot 교집합을 출력한다.

    Args:
        df_reduced: 축소 데이터.
        rows: split_by_lot 결과.
        class_names: 클래스 이름 순서.
        min_class_warn: 이보다 적은 클래스를 경고한다.

    Raises:
        Stage0Failure: 두 분할에 같은 Lot이 있을 때.
    """
    lots = {s: set(df_reduced["lotName"].iloc[rows[f"row_{s}"]]) for s in SPLITS}
    logger.info("축소 데이터 %d장, Lot %d개", len(df_reduced), df_reduced["lotName"].nunique())
    for s in SPLITS:
        n = len(rows[f"row_{s}"])
        logger.info("%-5s 웨이퍼 %5d장 (%.1f%%), Lot %4d개", s, n, 100 * n / len(df_reduced), len(lots[s]))

    dist = pd.DataFrame({s: df_reduced["failureType"].iloc[rows[f"row_{s}"]].value_counts()
                         for s in SPLITS}).reindex(class_names).fillna(0).astype(int)
    dist["total"] = dist.sum(axis=1)
    logger.info("분할별 클래스 분포\n%s", dist.to_string())
    for s in SPLITS:
        small = dist.index[dist[s] < min_class_warn].tolist()
        if small:
            logger.warning("%s에 %d장 미만인 클래스 %s: 그 클래스 F1은 흔들린다",
                           s, min_class_warn, {c: int(dist.at[c, s]) for c in small})

    overlaps = {f"{a}&{b}": len(lots[a] & lots[b])
                for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]}
    logger.info("분할 사이 Lot 교집합 %s", overlaps)
    if any(overlaps.values()):
        raise Stage0Failure("두 분할에 같은 Lot이 있음")


def prepare(cfg: dict) -> Prepared:
    """원본을 읽고 두 분할(웨이퍼 단위·Lot 단위)을 만들어 확인한다.

    Args:
        cfg: 장비 설정.

    Returns:
        Prepared.

    Raises:
        Stage0Failure: Lot 이름 재사용, 기존 split.npz와 불일치, Lot 교집합, 두 Test가 완전히 같을 때.
    """
    repo = resolve_wafer_repo()
    from src.utils import CLASS_NAMES

    d, ls = cfg["data"], cfg["lot_split"]
    df = load_lswmd(repo / cfg["wafer_repo"]["lswmd"])
    check_lot_name_reuse(df)
    df_reduced = build_reduced(df, d["none_sample_n"], d["seed"])
    X, y = make_xy(df_reduced, d["image_size"], CLASS_NAMES)

    wafer_split = split_like_notebook(X, y, d["test_size"], d["valid_size"], d["seed"])
    compare_split(wafer_split, repo / cfg["wafer_repo"]["split_npz"])  # (가)가 0단계와 같은 Test임을 보장
    wafer_rows = {k: v for k, v in wafer_split.items() if k.startswith("row_")}

    lot_rows = split_by_lot(y, df_reduced["lotName"].to_numpy(), ls["test_n_splits"], ls["valid_n_splits"],
                            ls["fold_index"], d["seed"])
    log_split_report(df_reduced, lot_rows, CLASS_NAMES, ls["min_class_warn"])

    n_overlap = len(np.intersect1d(wafer_rows["row_test"], lot_rows["row_test"]))
    logger.info("12번 Test %d장과 Lot 단위 Test %d장이 겹치는 웨이퍼 %d장 (%.1f%%)",
                len(wafer_rows["row_test"]), len(lot_rows["row_test"]), n_overlap,
                100 * n_overlap / len(lot_rows["row_test"]))
    if n_overlap == len(lot_rows["row_test"]):
        raise Stage0Failure("두 Test가 완전히 같음. Lot 단위 분할이 적용되지 않았을 수 있음")
    return Prepared(df, df_reduced, X, y, CLASS_NAMES, wafer_rows, lot_rows)


def build_training(cfg: dict, data: dict[str, np.ndarray], device: torch.device) -> tuple:
    """12번(02-exp.ipynb 셀 16)과 같은 순서·설정으로 로더·모델·옵티마이저·스케줄러를 만든다.

    Args:
        cfg: 장비 설정.
        data: X_train, y_train, X_valid, y_valid, X_test, y_test.
        device: 학습 장치.

    Returns:
        (model, train_loader, valid_loader, criterion, optimizer, scheduler).
    """
    from src.models import ImprovedCNN
    from src.utils import make_loaders, set_seed

    seed, m, t = cfg["data"]["seed"], cfg["model"], cfg["train"]
    set_seed(seed)
    train_loader, valid_loader, _ = make_loaders(data, batch_size=t["batch_size"], seed=seed,
                                                 augment=t["augment"])
    model = ImprovedCNN(dropout=m["dropout"], use_bn=m["use_bn"], activation=m["activation"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=t["scheduler_factor"], patience=t["scheduler_patience"])
    return model, train_loader, valid_loader, torch.nn.CrossEntropyLoss(), optimizer, scheduler


def log_full_test_lots(df: pd.DataFrame, test_lots: set[str], bytes_per_row: float) -> None:
    """시험용 Lot에 속한 원본 전체 웨이퍼 수와 판정 파일 크기 예상치를 출력한다(판정 전 미리보기)."""
    full = df[df["lotName"].isin(test_lots)]
    n_unlabeled = int((full["label_source"] == UNLABELED).sum())
    logger.info("시험용 Lot %d개의 원본 전체 웨이퍼 %d장 (라벨 있음 %d, 없음 %d), waferIndex %g~%g, "
                "판정 파일 예상 %.1f MB", len(test_lots), len(full), len(full) - n_unlabeled, n_unlabeled,
                full["waferIndex"].min(), full["waferIndex"].max(), len(full) * bytes_per_row / 1e6)


def cmd_plan(cfg: dict) -> None:
    """분할 결과, 두 Test의 겹침, 시험용 Lot 미리보기, 1에폭 시간을 출력한다."""
    p = prepare(cfg)  # 웨이퍼 저장소를 import 경로에 넣는다
    from src.utils import train_model

    stage0 = PROJECT_ROOT / cfg["output"]["predictions"]
    bytes_per_row = stage0.stat().st_size / len(pd.read_csv(stage0, usecols=["wafer_id"]))
    log_full_test_lots(p.df, set(p.df_reduced["lotName"].iloc[p.lot_rows["row_test"]]), bytes_per_row)

    device = torch.device(cfg["train"]["device"])
    model, tl, vl, crit, opt, sched = build_training(cfg, p.lot_data(), device)
    start = time.perf_counter()
    train_model(model, tl, vl, crit, opt, device, epochs=1, patience=None, ckpt_path=None,
                verbose=True, scheduler=sched)
    seconds = time.perf_counter() - start
    logger.info("장치 %s, 1에폭 %.1f초. 최대 %d에폭이면 약 %.0f분", device, seconds, cfg["train"]["epochs"],
                seconds * cfg["train"]["epochs"] / 60)


def cmd_train(cfg: dict) -> None:
    """Lot 단위 분할로 학습하고 체크포인트와 에폭별 기록을 저장한다."""
    p = prepare(cfg)  # 웨이퍼 저장소를 import 경로에 넣는다
    from src.utils import train_model

    t = cfg["train"]
    device = torch.device(t["device"])
    ckpt = PROJECT_ROOT / t["checkpoint"]
    model, tl, vl, crit, opt, sched = build_training(cfg, p.lot_data(), device)
    start = time.perf_counter()
    hist = train_model(model, tl, vl, crit, opt, device, epochs=t["epochs"], patience=t["early_stop_patience"],
                       ckpt_path=str(ckpt), verbose=True, scheduler=sched)
    seconds = time.perf_counter() - start

    n_epochs = len(hist["va_f1"])
    best = hist["best_epoch"]
    history = pd.DataFrame({k: hist[k] for k in HISTORY_KEYS})
    history.insert(0, "epoch", range(1, n_epochs + 1))
    out = PROJECT_ROOT / cfg["output"]["stage4_train_history"]
    out.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(out, index=False, float_format="%.6f")
    logger.info("장치 %s, 전체 %.1f초(%.1f분), 에폭당 평균 %.1f초", device, seconds, seconds / 60,
                seconds / n_epochs)
    logger.info("최고 Valid Macro-F1 %.4f @ 에폭 %d, 멈춘 에폭 %d (최대 %d)", hist["va_f1"][best - 1], best,
                n_epochs, t["epochs"])
    logger.info("체크포인트 %s SHA-256 %s", t["checkpoint"], sha256_of(ckpt))
    logger.info("저장 %s", cfg["output"]["stage4_train_history"])


def f1_row(case: str, model_name: str, y_true: np.ndarray, y_pred: np.ndarray,
           class_names: list[str]) -> dict:
    """비교 표 한 줄. Macro-F1은 원본 평가와 같은 방식, 클래스별 F1은 정답이 없는 클래스를 빈칸으로 둔다."""
    per_class = f1_score(y_true, y_pred, labels=list(range(len(class_names))), average=None,
                         zero_division=np.nan)
    support = np.bincount(y_true, minlength=len(class_names))
    row = {"case": case, "model": model_name, "n_wafers": len(y_true), "macro_f1": macro_f1(y_true, y_pred)}
    for i, c in enumerate(class_names):
        row[f"f1_{c}"] = per_class[i] if support[i] else np.nan
    for i, c in enumerate(class_names):
        row[f"n_{c}"] = int(support[i])  # 장수가 작은 클래스의 F1은 한두 장에 크게 흔들린다
    return row


def cmd_evaluate(cfg: dict) -> None:
    """(가) 12번/자기 Test, (나) 새 모델/Lot 단위 Test, (다) 두 Test가 겹치는 웨이퍼를 비교한다."""
    p = prepare(cfg)
    repo = resolve_wafer_repo()
    m, inf = cfg["model"], cfg["inference"]
    cpu = torch.device(inf["device"])
    ckpt12, ckpt_new = repo / cfg["wafer_repo"]["checkpoint"], PROJECT_ROOT / cfg["train"]["checkpoint"]
    models = {"exp12": load_model(ckpt12, m["dropout"], m["use_bn"], m["activation"], cpu),
              "lotsplit": load_model(ckpt_new, m["dropout"], m["use_bn"], m["activation"], cpu)}
    logger.info("모델 exp12 %s / lotsplit %s", sha256_of(ckpt12), sha256_of(ckpt_new))

    def judge(name: str, rows: np.ndarray) -> np.ndarray:
        return predict(models[name], p.X[rows], inf["batch_size"], cpu)[0]

    names = p.class_names
    w_test, l_test = p.wafer_rows["row_test"], p.lot_rows["row_test"]
    table = [f1_row("(가) 웨이퍼 단위 Test", "exp12", p.y[w_test], judge("exp12", w_test), names),
             f1_row("(나) Lot 단위 Test", "lotsplit", p.y[l_test], judge("lotsplit", l_test), names)]

    overlap = np.intersect1d(w_test, l_test)
    lot_of = p.df_reduced["lotName"].to_numpy()
    seen_train = np.isin(lot_of[overlap], lot_of[p.wafer_rows["row_train"]])
    seen_train_valid = np.isin(lot_of[overlap], lot_of[np.concatenate([p.wafer_rows["row_train"],
                                                                       p.wafer_rows["row_valid"]])])
    logger.info("겹치는 웨이퍼 %d장 중 12번 train에 같은 Lot 웨이퍼가 있었던 것 %d장, train+valid 기준 %d장",
                len(overlap), int(seen_train.sum()), int(seen_train_valid.sum()))
    for case, rows in [("(다) 겹치는 웨이퍼 전체", overlap),
                       ("(다-1) 12번 train에 같은 Lot 있음", overlap[seen_train]),
                       ("(다-2) 12번 train에 같은 Lot 없음", overlap[~seen_train])]:
        for name in models:
            table.append(f1_row(case, name, p.y[rows], judge(name, rows), names))

    result = pd.DataFrame(table)
    out = PROJECT_ROOT / cfg["output"]["stage4_comparison"]
    result.to_csv(out, index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        logger.info("비교 표\n%s", result.round(4).to_string(index=False))
    logger.info("저장 %s", cfg["output"]["stage4_comparison"])


def cmd_predict(cfg: dict) -> None:
    """시험용 Lot에 속한 원본 전체 웨이퍼를 새 모델로 판정해 판정 파일과 정답 파일을 저장한다."""
    p = prepare(cfg)
    m, inf, out = cfg["model"], cfg["inference"], cfg["output"]
    cpu = torch.device(inf["device"])
    ckpt = PROJECT_ROOT / cfg["train"]["checkpoint"]
    model_sha = sha256_of(ckpt)
    model = load_model(ckpt, m["dropout"], m["use_bn"], m["activation"], cpu)

    test_lots = set(p.df_reduced["lotName"].iloc[p.lot_rows["row_test"]])
    full = p.df[p.df["lotName"].isin(test_lots)].sort_values(["lotName", "waferIndex"]).reset_index(drop=True)
    full["wafer_id"] = make_wafer_ids(full["lotName"], full["waferIndex"])
    X_full = np.array([preprocess_wafer_map(w, cfg["data"]["image_size"]) for w in full["waferMap"]],
                      dtype=np.uint8)[:, np.newaxis, :, :]
    pred, prob = predict(model, X_full, inf["batch_size"], cpu)

    predictions = pd.DataFrame({
        "wafer_id": full["wafer_id"],
        "lot_name": full["lotName"],
        "wafer_index": full["waferIndex"].astype(int),
        "src_row": full["src_row"],
        "pred_label": [p.class_names[i] for i in pred],
    })
    for i, c in enumerate(p.class_names):
        predictions[f"prob_{c}"] = prob[:, i]
    predictions["model_sha256"] = model_sha
    labels = pd.DataFrame({"wafer_id": full["wafer_id"], "true_label": full["failureType"],
                           "label_source": full["label_source"]})
    for frame, key in [(predictions, "lot_predictions"), (labels, "lot_labels")]:
        path = PROJECT_ROOT / out[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, float_format="%.6f")
        logger.info("저장 %s (%d행, %.2f MB)", out[key], len(frame), path.stat().st_size / 1e6)

    sizes = full.groupby("lotName").size()
    logger.info("시험용 Lot %d개, 웨이퍼 %d장, Lot당 최소 %d / 중앙값 %.1f / 평균 %.2f / 최대 %d",
                len(sizes), len(full), sizes.min(), sizes.median(), sizes.mean(), sizes.max())
    logger.info("Lot당 웨이퍼 수별 Lot 수: %s",
                ", ".join(f"{k}장:{v}" for k, v in sizes.value_counts().sort_index().items()))
    labeled = full[full["label_source"] != UNLABELED]
    logger.info("라벨 있음 %d (불량 %d, none %d), 라벨 없음 %d", len(labeled),
                int((labeled["failureType"] != "none").sum()), int((labeled["failureType"] == "none").sum()),
                len(full) - len(labeled))
    in_reduced = full["src_row"].isin(p.df_reduced["src_row"].iloc[p.lot_rows["row_test"]])
    logger.info("이 중 축소 데이터(Lot 단위 Test)에 있던 웨이퍼 %d, 없던 웨이퍼 %d", int(in_reduced.sum()),
                int((~in_reduced).sum()))
    logger.info("waferIndex 최소 %d, 최대 %d", full["waferIndex"].min(), full["waferIndex"].max())
    logger.info("판정 유형 분포\n%s", predictions["pred_label"].value_counts().reindex(p.class_names)
                .fillna(0).astype(int).to_string())


def main() -> int:
    """하위 명령을 실행한다."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["plan", "train", "evaluate", "predict"])
    args = parser.parse_args()
    cfg = load_config(CONFIG_PATH)
    {"plan": cmd_plan, "train": cmd_train, "evaluate": cmd_evaluate, "predict": cmd_predict}[args.command](cfg)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Stage0Failure as e:
        logger.error("4단계 중단: %s", e)
        sys.exit(1)
