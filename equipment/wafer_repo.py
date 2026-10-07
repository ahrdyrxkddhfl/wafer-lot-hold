"""웨이퍼 저장소(SKALA_CNN-Optimization) 연결과 데이터 준비.

웨이퍼 저장소의 파일은 읽기만 한다. 모델·클래스 이름은 그 저장소에서 import하고,
노트북에만 있는 코드(pickle 호환 처리, 축소, 분할)는 출처를 적어 옮긴다.
"""
import logging
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.transform import resize
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WAFER_REPO = PROJECT_ROOT.parent / "SKALA_CNN-Optimization"

LABELED = "labeled"
UNLABELED = "unlabeled"


def resolve_wafer_repo() -> Path:
    """웨이퍼 저장소 경로를 정하고 그 저장소의 `src` 패키지를 import할 수 있게 한다.

    Returns:
        웨이퍼 저장소 경로. 환경변수 WAFER_REPO가 있으면 그 값, 없으면 이 저장소 옆의 기본 위치.

    Raises:
        FileNotFoundError: 경로에 `src/models.py`가 없을 때.
    """
    repo = Path(os.environ.get("WAFER_REPO", DEFAULT_WAFER_REPO)).resolve()
    if not (repo / "src" / "models.py").exists():
        raise FileNotFoundError(f"웨이퍼 저장소를 찾을 수 없음: {repo} (WAFER_REPO 확인)")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def install_pickle_shim() -> None:
    """구버전 pandas로 저장된 LSWMD.pkl을 열기 위한 모듈 별칭을 등록한다.

    출처: 웨이퍼 저장소 notebooks/01-prep.ipynb 셀 0 (그대로 옮김).
    """
    from pandas.core.indexes.base import Index, _new_Index

    def _shim(name: str, **attrs: object) -> None:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m

    _shim("pandas.indexes")
    _shim("pandas.indexes.base", _new_Index=_new_Index, Index=Index)
    _shim("pandas.indexes.numeric", Int64Index=Index, Float64Index=Index)
    _shim("pandas.indexes.range", RangeIndex=pd.RangeIndex)
    _shim("pandas.indexes.multi", MultiIndex=pd.MultiIndex)


def load_lswmd(pkl_path: Path) -> pd.DataFrame:
    """원본 LSWMD.pkl을 읽어 라벨을 문자열로 바꾼다.

    출처: 01-prep.ipynb 셀 1. 원본은 lotName·waferIndex를 지우지만 여기서는 남긴다.
    원본은 라벨 없는 웨이퍼(failureType 길이 0)도 'none'으로 넣는다. 그 동작은 그대로 두고,
    구분을 잃지 않도록 label_source 열을 더한다.

    Args:
        pkl_path: LSWMD.pkl 경로.

    Returns:
        waferMap, lotName, waferIndex, failureType, label_source, src_row 열을 가진 DataFrame.
    """
    install_pickle_shim()
    with open(pkl_path, "rb") as f:
        df = pickle.load(f, encoding="latin1")
    logger.info("원본: %d행, 열 %s", len(df), df.columns.tolist())

    # src_row: LSWMD.pkl을 pickle.load한 직후의 행 위치(0부터). 불량/정상 축소와 reset_index 전 번호다.
    df["src_row"] = np.arange(len(df), dtype=np.int64)
    df = df.drop(["dieSize"], axis=1)
    df["label_source"] = df["failureType"].apply(lambda x: LABELED if len(x) > 0 else UNLABELED)
    df["failureType"] = df["failureType"].apply(lambda x: x[0][0] if len(x) > 0 else "none")
    return df


def build_reduced(df: pd.DataFrame, none_sample_n: int, seed: int) -> pd.DataFrame:
    """불량은 전부, none(라벨 없음 포함)은 일부만 뽑아 축소 데이터를 만든다.

    출처: 01-prep.ipynb 셀 2. sample은 행 위치만 뽑으므로 열을 더해도 뽑히는 행은 같다.

    Args:
        df: load_lswmd 결과.
        none_sample_n: none 또는 라벨 없음 웨이퍼에서 뽑을 장수.
        seed: sample의 random_state.

    Returns:
        불량 다음 none 순서로 이어 붙이고 인덱스를 0부터 다시 매긴 DataFrame.
    """
    df_failure = df[df["failureType"] != "none"].copy()
    df_none = df[df["failureType"] == "none"].sample(n=none_sample_n, random_state=seed).copy()
    df_reduced = pd.concat([df_failure, df_none]).reset_index(drop=True)
    logger.info("축소: %d장 (불량 %d, none 또는 라벨 없음 %d)",
                len(df_reduced), len(df_failure), none_sample_n)
    return df_reduced


def preprocess_wafer_map(wm: np.ndarray, image_size: int) -> np.ndarray:
    """웨이퍼맵을 최근접 보간으로 축소한다.

    출처: 01-prep.ipynb 셀 2. order=0이라 픽셀값 0/1/2가 중간값으로 섞이지 않는다.

    Args:
        wm: 원본 웨이퍼맵(2차원, 값 0/1/2).
        image_size: 축소 후 가로·세로 크기.

    Returns:
        (image_size, image_size) uint8 배열.
    """
    return resize(wm, (image_size, image_size), order=0,
                  preserve_range=True, anti_aliasing=False).astype(np.uint8)


def make_xy(df_reduced: pd.DataFrame, image_size: int,
            class_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """축소 데이터를 모델 입력 배열과 정답 번호로 바꾼다.

    출처: 01-prep.ipynb 셀 2.

    Args:
        df_reduced: build_reduced 결과.
        image_size: 축소 크기.
        class_names: 웨이퍼 저장소 src.utils.CLASS_NAMES.

    Returns:
        X (N, 1, H, W) uint8, y (N,) int64.
    """
    resized = [preprocess_wafer_map(wm, image_size) for wm in df_reduced["waferMap"]]
    X = np.array(resized, dtype=np.uint8)[:, np.newaxis, :, :]
    mapping = {c: i for i, c in enumerate(class_names)}
    y = df_reduced["failureType"].map(mapping).values.astype(np.int64)
    return X, y


def split_like_notebook(X: np.ndarray, y: np.ndarray, test_size: float,
                        valid_size: float, seed: int) -> dict[str, np.ndarray]:
    """Test를 먼저 떼고 남은 것에서 Valid를 뗀다.

    출처: 01-prep.ipynb 셀 3. 원본과 달리 축소 데이터의 행 번호(row)를 같이 넘겨,
    각 웨이퍼가 어느 행이었는지 되찾을 수 있게 한다. 배열을 더 넘겨도 섞는 순서는 같다.

    Args:
        X: 입력 배열.
        y: 정답 번호.
        test_size: 전체 중 Test 비율.
        valid_size: Test를 뺀 나머지 중 Valid 비율.
        seed: random_state.

    Returns:
        X_train, y_train, X_valid, y_valid, X_test, y_test와
        각 분할의 축소 데이터 행 번호 row_train, row_valid, row_test.
    """
    row = np.arange(len(y))
    X_temp, X_test, y_temp, y_test, row_temp, row_test = train_test_split(
        X, y, row, test_size=test_size, random_state=seed, stratify=y)
    X_train, X_valid, y_train, y_valid, row_train, row_valid = train_test_split(
        X_temp, y_temp, row_temp, test_size=valid_size, random_state=seed, stratify=y_temp)
    return {"X_train": X_train, "y_train": y_train,
            "X_valid": X_valid, "y_valid": y_valid,
            "X_test": X_test, "y_test": y_test,
            "row_train": row_train, "row_valid": row_valid, "row_test": row_test}


def split_by_lot(y: np.ndarray, groups: np.ndarray, test_n_splits: int, valid_n_splits: int,
                 fold_index: int, seed: int) -> dict[str, np.ndarray]:
    """같은 Lot이 두 분할에 걸치지 않게 나눈다(4단계).

    StratifiedGroupKFold는 그룹(lotName)을 쪼개지 않으면서 각 묶음의 클래스 비율을 전체와 비슷하게 맞춘다.
    Lot마다 웨이퍼 수와 클래스 구성이 달라 비율이 정확히 맞지는 않는다.
    전체를 test_n_splits 묶음으로 나눠 fold_index번째를 Test로, 나머지를 valid_n_splits 묶음으로 나눠
    fold_index번째를 Valid로 쓴다.

    Args:
        y: 정답 번호.
        groups: 각 웨이퍼의 lotName.
        test_n_splits: Test를 고를 때 나눌 묶음 수.
        valid_n_splits: Valid를 고를 때 나눌 묶음 수.
        fold_index: 몇 번째 묶음을 쓸지.
        seed: random_state.

    Returns:
        축소 데이터 행 번호 row_train, row_valid, row_test.
    """
    rows = np.arange(len(y))
    outer = StratifiedGroupKFold(n_splits=test_n_splits, shuffle=True, random_state=seed)
    temp, test = list(outer.split(rows, y, groups))[fold_index]
    inner = StratifiedGroupKFold(n_splits=valid_n_splits, shuffle=True, random_state=seed)
    train_rel, valid_rel = list(inner.split(temp, y[temp], groups[temp]))[fold_index]
    return {"row_train": temp[train_rel], "row_valid": temp[valid_rel], "row_test": test}
