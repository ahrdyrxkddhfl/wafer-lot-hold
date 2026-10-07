# wafer-lot-hold

검사 장비 역할의 스크립트가 웨이퍼 불량 판정 결과를 보내면, Lot의 공정 진행을 관리하다가 지정한 불량이 나온
Lot을 자동으로 멈추고(Hold) 사람이 사유와 함께 처분(해제·재검사·폐기)하게 하는 시스템을 만드는 개인 프로젝트다.
MES 전체가 아니라 MES의 일부 기능(Lot 이력, 공정 순서, 정지·처분)을 다룬다.

## 현재 상태

진행 순서는 0 → 1 → 4 → 2 → 3이다. MES에 올릴 판정 데이터를 4단계(Lot 단위 분할 모델)에서 만들기 때문이다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 0 | 데이터 준비와 검사 모델 재현 확인 | 완료 |
| 1 | Lot·공정 순서·설비·이력 DB와 상태 규칙 | 완료 |
| 4 | 검사 모델 Lot 단위 재평가, MES에 올릴 판정 데이터 생성 | 예정 |
| 2 | 판정 결과 수신 API와 자동 정지 | 예정 |
| 3 | 생산 현황 보고(HTML), AI 판정 평가 | 예정 |

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

원본 데이터와 체크포인트 없이도 0단계 결과를 확인할 수 있게 Test 6,104장의 판정 결과와 정답을 파일로 남긴다.
이 판정은 웨이퍼 단위 분할 모델(12번)의 결과이며, MES 시연에는 4단계의 Lot 단위 분할 모델 판정을 쓴다.

- [data/test_predictions_exp12.csv](data/test_predictions_exp12.csv): 판정 결과.
  `wafer_id, lot_name, wafer_index, src_row, pred_label, prob_<클래스 9개>, model_sha256`
- [evaluation/test_labels.csv](evaluation/test_labels.csv): 정답 라벨. `wafer_id, true_label, label_source`.
  평가 전용이며 MES 코드는 이 파일을 읽지 않는다.

`wafer_id`는 `{lotName}_W{waferIndex:02d}` 형식이다(예: `lot46082_W17`).
`src_row`는 `LSWMD.pkl`을 읽은 직후, 축소·재번호 전의 원본 행 위치(0부터)다.
판정은 CPU로 했다(다른 컴퓨터에서도 같은 결과가 나오게).

## 1단계: MES 핵심 (Lot·공정·설비·Hold)

PostgreSQL에 Lot, 웨이퍼, 공정 순서, 설비와 설비 상태 이력, 작업 지시, Lot 이력, 검사 판정, Hold·처분을 둔다.
테이블 정의는 [schema.sql](schema.sql) 하나이고, 동작은 [mes/service.py](mes/service.py)에 있다.
ORM 없이 SQL을 직접 써서 잠금과 트랜잭션 범위가 코드에 그대로 보이게 했다.

- **공정 순서와 설비는 합성값이다**([config/mes.yaml](config/mes.yaml)). 공정은 `STEP_1 → STEP_2 → INSPECT → SHIP`,
  앞 세 공정에 설비 2대씩, SHIP에 출하 스테이션 1대를 둔다.
- **작업 지시**는 "Lot 하나를 정해진 공정·설비에서 처리하라는 지시 한 건"이다. 투입(track_in)으로 시작하고
  완료(track_out)로 끝나며, ID는 지시하는 쪽이 정한다.
- **Lot 상태**는 `WAITING`(현재 공정 대기) → `IN_PROCESS` → 다음 공정 `WAITING` … → `FINISHED`. `SCRAPPED`는 끝 상태다.
  Hold 여부는 Lot 상태에 넣지 않고 열린 Hold 행으로 판단한다.
- **Hold**는 무엇이 열었는지 남긴다. 규칙이 열었으면 그 판정 결과(`trigger_result_id`), 사람이 열었으면 연 사람(`opened_by`,
  규칙 이름 `MANUAL`) 중 정확히 하나가 있다.
- **처분**은 해제(RELEASE), 재검사(RETEST: 검사 차수 `inspect_round` +1), 폐기(SCRAP)이며 결정자와 사유가 필요하다.
  AI 판정(`inspection_result`)과 사람의 결정(`hold_disposition`)은 다른 테이블에 기록한다.
- 시각은 모두 DB가 찍는다(`timestamptz DEFAULT now()`).

### 막는 상태

| # | 상태 | 서비스 검사 | DB 제약 |
|---|---|---|---|
| I1 | 공정 순서 건너뛰기 | 요청 공정이 Lot의 현재 공정보다 뒤면 거절 | 공정 순서는 `route_step.seq` 기준 |
| I2 | 끝난 공정 다시 처리 | 그 공정의 작업 지시 이력이 있으면 거절 | `UNIQUE (lot_id, step_code)` |
| I3 | 같은 작업 지시 두 번 처리 | 같은 ID·같은 내용이면 "이미 처리됨"으로 같은 결과 반환, 다른 내용이면 거절 | `work_order_id` 기본키 |
| I4 | Hold된 Lot 진행 | 열린 Hold가 있으면 투입·완료 거절 | — |
| I5 | 사유·결정자 없는 처분 | 빈 값·공백 거절 | `NOT NULL` + `CHECK (btrim(...) <> '')` |
| I6 | 정지·정비 중인 설비에 투입 | 설비가 `AVAILABLE`이 아니면 거절 | 상태 값 `CHECK` |
| I7 | 두 설비가 같은 Lot을 동시에 처리 | Lot 행 `SELECT ... FOR UPDATE` 후 처리 중이면 거절 | 부분 유일 인덱스 `(lot_id) WHERE status='STARTED'` |
| I8 | 같은 판정 결과를 다시 보냄 | (2단계) | `UNIQUE (wafer_id, model_sha256, inspect_round)` |
| I9 | 한 Lot에 열린 Hold가 두 개 | 열린 Hold가 있으면 새로 만들지 않고 그 Hold 반환 | 부분 유일 인덱스 `(lot_id) WHERE closed_at IS NULL` |
| I10 | 폐기된 Lot 진행 | `SCRAPPED`면 투입·완료·Hold 거절 | — |
| I11 | 한 설비가 두 Lot을 동시에 처리 | 설비 행 `SELECT ... FOR UPDATE` 후 처리 중인 Lot이 있으면 거절 | 부분 유일 인덱스 `(equipment_id) WHERE status='STARTED'` |

잠금이 정상 경로이고 제약은 마지막 방어선이다. 잠금이 제대로 걸리면 제약 위반은 일어날 수 없으므로,
제약 위반은 규칙 오류로 바꾸지 않고 그대로 실패시킨다. 규칙마다 테스트가 [tests/test_rules.py](tests/test_rules.py)에 있고,
I7·I11은 연결 두 개를 배리어로 같은 순간에 출발시켜 하나만 성공하는지 확인한다.

### 테스트 실행

```bash
cp .env.example .env              # POSTGRES_PASSWORD 채우기
docker compose up -d --wait       # PostgreSQL 17.11, 호스트 포트 5434
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest -v               # 테스트 전용 DB(wafer_mes_test)를 만들어 쓴다
```

테스트는 개발용 DB(`wafer_mes`)가 아니라 테스트 전용 DB에서 돌고, 매 테스트 전에 모든 테이블을 비운다.
GitHub Actions도 같은 PostgreSQL 버전의 서비스 컨테이너로 같은 테스트를 돌린다.

## 한계 (현재까지)

- 공정 순서, 설비 목록·상태는 합성값이다.

- **none 클래스에 라벨 없는 웨이퍼가 섞여 학습됐다.** 원본 전처리가 라벨 없는 웨이퍼도 `none`으로 넣었기 때문이다.
  학습용 축소 데이터의 none 또는 라벨 없음 5,000장 중 실제 라벨 none은 974장, 라벨 없음은 4,026장이다.
  Test의 none 1,000장 중에서는 206장과 794장이다. 정답 파일의 `label_source`로 구분하고, AI 판정 평가에서는
  라벨 없음을 정상으로 치지 않고 따로 센다.
- 학습용 축소 데이터는 불량 25,519장 전부와 none 또는 라벨 없음 5,000장으로 만들어 불량이 대부분이다.
- 0단계 판정 파일(12번 모델)은 웨이퍼 단위 분할 모델의 결과이며, MES 시연에는 4단계의 Lot 단위 분할 모델을 쓴다.
- MES 전체가 아니라 일부 기능이다.

## 데이터와 모델 출처

- 데이터: WM-811K (`LSWMD.pkl`). 원본 데이터와 체크포인트는 이 저장소에 없고, 판정 결과 파일과 모델 해시만 있다.
- 검사 모델: 2인 팀 과제 저장소 [SKALA_CNN-Optimization](https://github.com/ahrdyrxkddhfl/SKALA_CNN-Optimization)의
  12번 실험 모델. 전처리·모델 설계·실험은 본인이 맡았다.
