"""처분 시연용 Lot 웨이퍼맵 그림. 사람이 이 그림을 보고 Hold 처분(해제·재검사·폐기)을 정한다.

Lot의 웨이퍼 전부를 격자로, 축소 전 원본 웨이퍼맵을 그리고, 칸마다 웨이퍼 번호·AI 판정 유형·그 확률을 적는다.
정답 라벨은 넣지 않는다(현장 작업자는 정답을 모른다). 판정 파일과 원본 LSWMD.pkl만 읽는다.
그림은 원본 데이터에서 만든 것이라 저장소에 올리지 않는다(.gitignore).

웨이퍼 저장소의 .venv(matplotlib 포함)로 이 저장소 루트에서 실행한다.
    ../SKALA_CNN-Optimization/.venv/bin/python -m equipment.draw_wafer_maps lot1 lot10496
"""
import argparse
import logging
import math
import sys

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from equipment.prepare_stage0 import CONFIG_PATH, load_config  # noqa: E402
from equipment.wafer_repo import PROJECT_ROOT, load_lswmd, resolve_wafer_repo  # noqa: E402

logger = logging.getLogger("draw_wafer_maps")

# 웨이퍼맵 픽셀값(WM-811K 약속): 0 웨이퍼 밖, 1 정상 칩, 2 불량 칩
PIXEL_LEGEND = [(0, "#eeeeee", "0: outside wafer"), (1, "#9ecae1", "1: normal die"), (2, "#d62728", "2: defect die")]
NONE_LABEL = "none"


def draw_lot(lot_id: str, lot: pd.DataFrame, wafer_maps: pd.Series, cfg: dict) -> str:
    """Lot 하나의 그림을 저장하고 경로를 돌려준다.

    Args:
        lot_id: Lot ID.
        lot: 그 Lot의 판정 행(wafer_index 순).
        wafer_maps: src_row → 원본 웨이퍼맵.
        cfg: demo_wafers 설정.

    Returns:
        저장한 파일 경로(저장소 기준).
    """
    cols = cfg["grid_columns"]
    rows = math.ceil(len(lot) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cfg["cell_inches"], rows * cfg["cell_inches"] + 0.9),
                             squeeze=False)
    cmap = ListedColormap([color for _, color, _ in PIXEL_LEGEND])
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, row) in zip(axes.flat, lot.iterrows()):
        prob = row["prob_" + row["pred_label"]]
        ax.imshow(wafer_maps[row["src_row"]], cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(f"W{row['wafer_index']:02d}  {row['pred_label']}  {prob:.2f}", fontsize=9,
                     color="black" if row["pred_label"] == NONE_LABEL else "#b00000")
    n_defect = int((lot["pred_label"] != NONE_LABEL).sum())
    fig.suptitle(f"{lot_id}: AI judgment per wafer (label  probability), {n_defect}/{len(lot)} non-none. "
                 f"Ground truth not shown.", fontsize=11, y=0.995)
    fig.legend(handles=[Patch(facecolor=color, edgecolor="gray", label=label) for _, color, label in PIXEL_LEGEND],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.975), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = PROJECT_ROOT / cfg["output_dir"] / f"{lot_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=cfg["dpi"])
    plt.close(fig)
    return str(out.relative_to(PROJECT_ROOT))


def main() -> int:
    """지정한 Lot들의 그림을 만든다."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="처분 시연용 Lot 웨이퍼맵 그림")
    parser.add_argument("lots", nargs="+", help="그릴 Lot ID")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    pred = pd.read_csv(PROJECT_ROOT / cfg["output"]["lot_predictions"])
    missing = sorted(set(args.lots) - set(pred["lot_name"]))
    if missing:
        raise SystemExit(f"판정 파일에 없는 Lot: {missing}")
    pred = pred[pred["lot_name"].isin(args.lots)].sort_values(["lot_name", "wafer_index"])

    repo = resolve_wafer_repo()
    df = load_lswmd(repo / cfg["wafer_repo"]["lswmd"])
    # src_row는 LSWMD.pkl을 읽은 직후의 행 위치다. 정답 열은 쓰지 않고 웨이퍼맵만 가져온다.
    wafer_maps = df["waferMap"].iloc[pred["src_row"].to_numpy()]
    wafer_maps.index = pred["src_row"].to_numpy()

    for lot_id in args.lots:
        path = draw_lot(lot_id, pred[pred["lot_name"] == lot_id], wafer_maps, cfg["demo_wafers"])
        logger.info("저장 %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
