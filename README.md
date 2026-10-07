# wafer-lot-hold

검사 장비 역할의 스크립트가 웨이퍼 불량 판정 결과를 보내면, Lot의 공정 진행을 관리하다가 지정한 불량이 나온
Lot을 자동으로 멈추고(Hold) 사람이 사유와 함께 처분(해제·재검사·폐기)하게 하는 시스템을 만드는 개인 프로젝트다.
MES 전체가 아니라 MES의 일부 기능(Lot 이력, 공정 순서, 정지·처분)을 다룬다.

## 현재 상태

| 단계 | 내용 | 상태 |
|---|---|---|
| 0 | 데이터 준비와 검사 모델 재현 확인 | 완료 |
| 1 | Lot·공정 순서·설비·이력 DB와 상태 규칙 | 예정 |
| 2 | 판정 결과 수신 API와 자동 정지 | 예정 |
| 3 | 생산 현황 보고(HTML), AI 판정 평가 | 예정 |
| 4 | 검사 모델 Lot 단위 재평가 | 예정 |

## 0단계: 데이터 준비와 모델 재현 확인

검사 모델과 데이터는 [SKALA_CNN-Optimization](https://github.com/ahrdyrxkddhfl/SKALA_CNN-Optimization)
(이하 웨이퍼 저장소)의 12번 실험 결과를 쓴다. 모델 코드(`ImprovedCNN`)와 클래스 이름은 그 저장소에서
import하고, 노트북에만 있는 전처리·분할 코드는 출처를 주석으로 적어 옮겼다.

### 실행

웨이퍼 저장소를 이 저장소 옆에 두고, 그 저장소의 가상환경(torch 포함)으로 실행한다.
위치가 다르면 환경변수 `WAFER_REPO`로 경로를 준다.

```bash
../SKALA_CNN-Optimization/.venv/bin/python -m equipment.prepare_stage0
```

검증이 하나라도 실패하면 이유를 남기고 exit 1로 멈추며 산출물을 쓰지 않는다.
설정값은 [config/equipment.yaml](config/equipment.yaml)에 있다.

### 확인한 것

| 항목 | 결과 |
|---|---|
| 체크포인트 SHA-256 | `438a0473…3704faf` 일치 |
| 원본에서 다시 만든 분할 vs 웨이퍼 저장소 `split.npz` | X·y의 train/valid/test 6개 배열 모두 완전히 같음 |
| Test 안의 (lotName, waferIndex) 중복 | 0장 |
| Test Macro-F1 (CPU) | 0.880501 (기준 0.8805) |
| Test Macro-F1 (MPS) | 0.880501, CPU와 예측이 다른 웨이퍼 0장 |
| 활성함수를 `relu`로 잘못 넣은 경우 | `load_state_dict(strict=True)`가 오류 없이 성공, Macro-F1 0.882389, 예측이 바뀐 웨이퍼 34장 |
| Test Lot | 3,875개, Lot당 웨이퍼 1~13장(중앙값 1, 평균 1.58) |

활성함수 확인은 "설정을 잘못 넣어도 오류가 나지 않고 판정만 바뀐다"는 것을 보이기 위한 것이다. 이번에는
LeakyReLU(기울기 0.01)와 ReLU의 차이가 작아 점수가 오히려 조금 올랐지만, 판정 34건이 소리 없이 바뀌었다.
그래서 활성함수는 config에 적고 점수 재현으로 확인한다.

### 산출물 (커밋됨)

원본 데이터와 체크포인트 없이도 이후 단계를 재현할 수 있게 Test 6,104장의 판정 결과와 정답을 파일로 남긴다.

- [data/test_predictions_exp12.csv](data/test_predictions_exp12.csv): 판정 결과.
  `wafer_id, lot_name, wafer_index, src_row, pred_label, prob_<클래스 9개>, model_sha256`
- [evaluation/test_labels.csv](evaluation/test_labels.csv): 정답 라벨. `wafer_id, true_label, label_source`.
  평가 전용이며 MES 코드는 이 파일을 읽지 않는다.

`wafer_id`는 `{lotName}_W{waferIndex:02d}` 형식이다(예: `lot46082_W17`).
`src_row`는 `LSWMD.pkl`을 읽은 직후, 축소·재번호 전의 원본 행 위치(0부터)다.
판정은 CPU로 했다(다른 컴퓨터에서도 같은 결과가 나오게).

## 한계 (현재까지)

- **none 클래스에 라벨 없는 웨이퍼가 섞여 학습됐다.** 원본 전처리가 라벨 없는 웨이퍼도 `none`으로 넣었기 때문이다.
  학습용 축소 데이터의 none 또는 라벨 없음 5,000장 중 실제 라벨 none은 974장, 라벨 없음은 4,026장이다.
  Test의 none 1,000장 중에서는 206장과 794장이다. 정답 파일의 `label_source`로 구분하고, AI 판정 평가에서는
  라벨 없음을 정상으로 치지 않고 따로 센다.
- 학습용 축소 데이터는 불량 25,519장 전부와 none 또는 라벨 없음 5,000장으로 만들어 불량이 대부분이다.
- 검사 모델(12번)은 웨이퍼 단위 무작위 분할로 학습돼 같은 Lot의 웨이퍼가 학습과 시험에 섞였을 수 있다(4단계에서 측정 예정).
- MES 전체가 아니라 일부 기능이다.

## 데이터와 모델 출처

- 데이터: WM-811K (`LSWMD.pkl`). 원본 데이터와 체크포인트는 이 저장소에 없고, 판정 결과 파일과 모델 해시만 있다.
- 검사 모델: SKALA 2인 팀 과제 저장소 [SKALA_CNN-Optimization](https://github.com/ahrdyrxkddhfl/SKALA_CNN-Optimization)의
  12번 실험 모델. 전처리·모델 설계·실험은 본인이 맡았다.
