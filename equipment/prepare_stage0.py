"""0단계: 데이터 준비와 12번 모델 재현 확인.

웨이퍼 저장소의 .venv로 이 저장소 루트에서 실행한다.
    ../SKALA_CNN-Optimization/.venv/bin/python -m equipment.prepare_stage0

검증 단계 하나라도 실패하면 이유를 남기고 exit 1로 멈추며 산출물을 쓰지 않는다.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import skimage
import sklearn
import torch
import yaml
from sklearn.metrics import f1_score

from equipment.inspector import load_model, predict, sha256_of
from equipment.wafer_repo import (PROJECT_ROOT, UNLABELED, build_reduced, load_lswmd,
                                  make_xy, resolve_wafer_repo, split_like_notebook)

logger = logging.getLogger("stage0")

CONFIG_PATH = PROJECT_ROOT / "config" / "equipment.yaml"
SPLIT_KEYS = ["X_train", "y_train", "X_valid", "y_valid", "X_test", "y_test"]


class Stage0Failure(Exception):
    """0단계 검증이 실패해 더 진행하면 안 되는 상황."""


def load_config(path: Path) -> dict:
    """YAML 설정을 읽는다.

    Args:
        path: 설정 파일 경로.

    Returns:
        설정 dict.
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compare_split(new: dict[str, np.ndarray], saved_path: Path) -> None:
    """새로 만든 분할이 기존 split.npz와 한 원소도 다르지 않은지 확인한다.

    Args:
        new: split_like_notebook 결과.
        saved_path: 웨이퍼 저장소 split.npz.

    Raises:
        Stage0Failure: 배열 하나라도 shape·dtype·값이 다를 때.
    """
    saved = np.load(saved_path)
    all_same = True
    for k in SPLIT_KEYS:
        a, b = new[k], saved[k]
        same = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
        logger.info("분할 비교 %-8s new %s %s / saved %s %s -> %s",
                    k, a.shape, a.dtype, b.shape, b.dtype, "같음" if same else "다름")
        all_same &= same
    if not all_same:
        raise Stage0Failure("기존 split.npz와 다른 배열이 있음")


def make_wafer_ids(lot_names: pd.Series, wafer_indexes: pd.Series) -> pd.Series:
    """웨이퍼 ID를 f"{lotName}_W{int(waferIndex):02d}" 형식으로 만든다.

    Args:
        lot_names: lotName 열.
        wafer_indexes: waferIndex 열(원본은 float).

    Returns:
        웨이퍼 ID 열.

    Raises:
        Stage0Failure: waferIndex에 결측이나 정수가 아닌 값이 있을 때.
    """
    wi = wafer_indexes.astype(float)
    bad = wi.isna() | (wi != np.floor(wi))
    if bad.any():
        raise Stage0Failure(
            f"정수가 아닌 waferIndex {int(bad.sum())}개: {wafer_indexes[bad].head().tolist()}")
    return pd.Series([f"{lot}_W{int(w):02d}" for lot, w in zip(lot_names, wi)],
                     index=lot_names.index)


def log_label_source(df_reduced: pd.DataFrame, test_meta: pd.DataFrame) -> None:
    """none 클래스 안에서 실제 라벨 none과 라벨 없음 웨이퍼 수를 출력한다.

    Args:
        df_reduced: 축소 데이터 전체.
        test_meta: Test 웨이퍼 메타데이터.
    """
    for name, frame in [("축소 데이터 none", df_reduced), ("Test none", test_meta)]:
        none = frame[frame["failureType"] == "none"]
        n_unlabeled = int((none["label_source"] == UNLABELED).sum())
        logger.info("%s %d장: 실제 라벨 none %d, 라벨 없음 %d",
                    name, len(none), len(none) - n_unlabeled, n_unlabeled)


def log_lot_sizes(test_meta: pd.DataFrame) -> None:
    """Test 안에서 Lot당 웨이퍼 수 분포를 출력한다.

    Args:
        test_meta: Test 웨이퍼 메타데이터.
    """
    sizes = test_meta.groupby("lotName").size()
    logger.info("Test Lot 수 %d, Lot당 웨이퍼 수 최소 %d / 중앙값 %.1f / 평균 %.2f / 최대 %d",
                len(sizes), sizes.min(), sizes.median(), sizes.mean(), sizes.max())
    dist = sizes.value_counts().sort_index()
    logger.info("Lot당 웨이퍼 수별 Lot 수: %s",
                ", ".join(f"{k}장:{v}" for k, v in dist.items()))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """원본 평가와 같은 방식의 Macro-F1."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def main() -> int:
    """0단계 전체를 실행한다.

    Returns:
        종료 코드. 성공 0, 검증 실패 1.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(CONFIG_PATH)
    repo = resolve_wafer_repo()
    from src.utils import CLASS_NAMES

    logger.info("웨이퍼 저장소 %s", repo)
    logger.info("버전 numpy %s, pandas %s, scikit-image %s, scikit-learn %s, torch %s",
                np.__version__, pd.__version__, skimage.__version__,
                sklearn.__version__, torch.__version__)

    # 1. 체크포인트 해시
    ckpt = repo / cfg["wafer_repo"]["checkpoint"]
    model_sha = sha256_of(ckpt)
    logger.info("체크포인트 SHA-256 %s", model_sha)
    if model_sha != cfg["model"]["checkpoint_sha256"]:
        raise Stage0Failure(f"체크포인트 해시가 config와 다름: {model_sha}")

    # 2. 원본 로드·축소·분할
    d = cfg["data"]
    df = load_lswmd(repo / cfg["wafer_repo"]["lswmd"])
    df_reduced = build_reduced(df, d["none_sample_n"], d["seed"])
    del df
    X, y = make_xy(df_reduced, d["image_size"], CLASS_NAMES)
    split = split_like_notebook(X, y, d["test_size"], d["valid_size"], d["seed"])

    # 3. 기존 분할과 비교
    compare_split(split, repo / cfg["wafer_repo"]["split_npz"])

    test_meta = df_reduced.iloc[split["row_test"]][
        ["lotName", "waferIndex", "src_row", "label_source", "failureType"]
    ].reset_index(drop=True)
    y_test = split["y_test"]
    if not np.array_equal(test_meta["failureType"].map(CLASS_NAMES.index).values, y_test):
        raise Stage0Failure("Test 메타데이터와 y_test 순서가 맞지 않음")
    log_label_source(df_reduced, test_meta)

    # 4. 웨이퍼 ID와 중복
    test_meta["wafer_id"] = make_wafer_ids(test_meta["lotName"], test_meta["waferIndex"])
    n_dup = int(test_meta.duplicated(["lotName", "waferIndex"], keep=False).sum())
    logger.info("Test %d장 중 (lotName, waferIndex)가 겹치는 웨이퍼 %d장", len(test_meta), n_dup)

    # 5. 점수 재현: CPU(산출물) + 비교 장치
    m, inf, v = cfg["model"], cfg["inference"], cfg["verify"]
    expected, decimals = v["expected_test_macro_f1"], v["f1_decimals"]

    def matches(f1: float) -> bool:
        return round(f1, decimals) == round(expected, decimals)

    cpu = torch.device(inf["device"])
    model = load_model(ckpt, m["dropout"], m["use_bn"], m["activation"], cpu)
    pred, prob = predict(model, split["X_test"], inf["batch_size"], cpu)
    f1_cpu = macro_f1(y_test, pred)
    logger.info("%s Test Macro-F1 %.6f (기준 %.4f, %s)",
                inf["device"], f1_cpu, expected, "같음" if matches(f1_cpu) else "다름")

    cmp_name = inf["compare_device"]
    f1_cmp = None
    if cmp_name == "mps" and not torch.backends.mps.is_available():
        logger.warning("비교 장치 %s를 쓸 수 없어 건너뜀", cmp_name)
    else:
        cmp_dev = torch.device(cmp_name)
        model_cmp = load_model(ckpt, m["dropout"], m["use_bn"], m["activation"], cmp_dev)
        pred_cmp, _ = predict(model_cmp, split["X_test"], inf["batch_size"], cmp_dev)
        f1_cmp = macro_f1(y_test, pred_cmp)
        logger.info("%s Test Macro-F1 %.6f (기준 %.4f, %s)",
                    cmp_name, f1_cmp, expected, "같음" if matches(f1_cmp) else "다름")
        logger.info("%s와 %s의 예측이 다른 웨이퍼 %d장",
                    inf["device"], cmp_name, int((pred != pred_cmp).sum()))

    if not matches(f1_cpu) and not (f1_cmp is not None and matches(f1_cmp)):
        raise Stage0Failure("어느 장치의 Macro-F1도 기준값과 같지 않음")
    if not matches(f1_cpu):
        logger.warning("CPU 점수만 기준과 다름. README에 적을 것")

    # 6. 활성함수를 잘못 넣은 경우
    wrong = load_model(ckpt, m["dropout"], m["use_bn"], m["wrong_activation"], cpu)
    pred_wrong, _ = predict(wrong, split["X_test"], inf["batch_size"], cpu)
    logger.info("activation=%s로도 load_state_dict(strict=True) 성공. Test Macro-F1 %.6f, "
                "%s 대비 예측이 바뀐 웨이퍼 %d장",
                m["wrong_activation"], macro_f1(y_test, pred_wrong), m["activation"],
                int((pred != pred_wrong).sum()))

    # 7. Lot 크기 분포
    log_lot_sizes(test_meta)

    if n_dup:
        raise Stage0Failure(f"웨이퍼 ID 중복 {n_dup}장. ID 규칙을 정한 뒤 다시 실행")

    # 8. 산출물
    out = cfg["output"]
    predictions = pd.DataFrame({
        "wafer_id": test_meta["wafer_id"],
        "lot_name": test_meta["lotName"],
        "wafer_index": test_meta["waferIndex"].astype(int),
        "src_row": test_meta["src_row"],
        "pred_label": [CLASS_NAMES[i] for i in pred],
    })
    for i, c in enumerate(CLASS_NAMES):
        predictions[f"prob_{c}"] = prob[:, i]
    predictions["model_sha256"] = model_sha
    labels = pd.DataFrame({
        "wafer_id": test_meta["wafer_id"],
        "true_label": test_meta["failureType"],
        "label_source": test_meta["label_source"],
    })
    for frame, rel in [(predictions, out["predictions"]), (labels, out["labels"])]:
        path = PROJECT_ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, float_format="%.6f")
        logger.info("저장 %s (%d행)", rel, len(frame))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Stage0Failure as e:
        logger.error("0단계 중단: %s", e)
        sys.exit(1)
