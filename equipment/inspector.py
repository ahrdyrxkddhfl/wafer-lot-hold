"""검사 모델(웨이퍼 저장소 12번 모델) 불러오기와 판정."""
import hashlib
from pathlib import Path

import numpy as np
import torch

HASH_CHUNK_BYTES = 1024 * 1024  # 해시 계산 시 한 번에 읽는 크기(1 MiB)


def sha256_of(path: Path) -> str:
    """파일의 SHA-256을 구한다. 어떤 모델이 판정했는지 기록·확인하는 데 쓴다.

    Args:
        path: 파일 경로.

    Returns:
        16진수 문자열.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(ckpt_path: Path, dropout: float, use_bn: bool, activation: str,
               device: torch.device) -> torch.nn.Module:
    """웨이퍼 저장소의 ImprovedCNN을 만들고 체크포인트를 불러와 판정 모드로 둔다.

    resolve_wafer_repo()를 먼저 불러 웨이퍼 저장소가 import 경로에 있어야 한다.
    activation이 학습 때와 달라도 가중치 모양은 같아서 strict=True로도 오류 없이 불러와진다.

    Args:
        ckpt_path: state_dict를 저장한 체크포인트.
        dropout: ImprovedCNN dropout.
        use_bn: ImprovedCNN use_bn.
        activation: ImprovedCNN activation.
        device: 모델을 올릴 장치.

    Returns:
        eval() 상태의 모델.
    """
    from src.models import ImprovedCNN

    model = ImprovedCNN(dropout=dropout, use_bn=use_bn, activation=activation)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def predict(model: torch.nn.Module, X: np.ndarray, batch_size: int,
            device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """웨이퍼를 판정해 예측 클래스 번호와 클래스별 확률을 돌려준다.

    12번 모델은 정규화 없이 학습했으므로(make_loaders 기본값 normalize=False)
    픽셀값 0/1/2를 float로만 바꿔 넣는다. eval()과 no_grad()가 없으면 BatchNorm·Dropout이
    학습 모드로 돌아 오류 없이 점수만 틀어지므로 여기서 다시 한 번 eval()을 건다.

    Args:
        model: load_model 결과.
        X: (N, 1, H, W) uint8 입력.
        batch_size: 한 번에 판정할 장수.
        device: 판정 장치.

    Returns:
        pred (N,) int64: 로짓 최댓값의 클래스 번호(원본 evaluate의 torch.max와 같은 기준).
        prob (N, C) float32: softmax 확률.
    """
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size].astype(np.float32)).to(device)
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds).astype(np.int64), np.concatenate(probs)
