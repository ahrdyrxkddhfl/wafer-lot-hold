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
| 4 | 검사 모델 Lot 단위 재평가, MES에 올릴 판정 데이터 생성 | 완료 |
| 2 | 판정 결과 수신 API와 자동 정지 | 완료 |
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
- **Hold**는 무엇이 열었는지 남긴다. 판정 규칙이면 그 판정 결과(`trigger_result_id`)와 그때의 기준값(`rule_params`),
  설비 고장이면 그 상태 변경 이력(`trigger_equipment_history_id`, 규칙 이름 `EQUIPMENT_DOWN`), 사람이면 연 사람
  (`opened_by`, 규칙 이름 `MANUAL`) 중 정확히 하나가 있다(`CHECK num_nonnulls(...) = 1`).
  열린 Hold는 **종류(규칙 이름)별로** Lot당 하나다. 종류가 다르면 함께 열린다.
- **Hold는 다음 공정 투입만 막는다(I4).** 완료(설비에서 내리기)는 막지 않는다. Hold된 Lot이 설비를 계속 차지하면
  사람이 처분할 때까지 그 설비에 다른 Lot을 넣을 수 없기 때문이다.
- **설비 상태 변경**: 계획 정비(MAINTENANCE)는 처리 중인 Lot이 있으면 거절한다(I12). 고장(DOWN)은 받고, 그 설비에서
  처리 중인 Lot에 같은 트랜잭션으로 `EQUIPMENT_DOWN` Hold를 연다.
- **처분**은 해제(RELEASE), 재검사(RETEST: 검사 차수 `inspect_round` +1), 폐기(SCRAP)이며 결정자와 사유가 필요하다.
  처분은 Hold 하나만 닫으므로, 다른 종류의 Hold가 열려 있으면 Lot은 계속 멈춰 있다. 예외로 폐기는 그 Lot의 열린 Hold를
  모두, 재검사는 열린 판정 규칙 Hold를 함께 닫는다. 이때 **직접 결정한 처분과 딸려서 닫힌 처분을 구분한다**
  (딸려서 닫힌 처분은 `cascaded_from_disposition_id`로 원래 처분을 가리킨다).
  재검사는 INSPECT를 시작한 뒤부터 다음 공정 투입 전까지만 받는다. 이미 처분된 Hold에 같은 내용의 처분이 다시 오면
  기존 처분을 돌려주고, 다른 내용이면 거절한다. AI 판정(`inspection_result`)과 사람의 결정(`hold_disposition`)은 다른 테이블에 기록한다.
- 시각은 모두 DB가 찍는다(`timestamptz DEFAULT now()`).

### 막는 상태

| # | 상태 | 서비스 검사 | DB 제약 |
|---|---|---|---|
| I1 | 공정 순서 건너뛰기 | 요청 공정이 Lot의 현재 공정보다 뒤면 거절 | 공정 순서는 `route_step.seq` 기준 |
| I2 | 끝난 공정 다시 처리 | 그 공정의 작업 지시 이력이 있으면 거절 | `UNIQUE (lot_id, step_code)` |
| I3 | 같은 작업 지시 두 번 처리 | 같은 ID·같은 내용이면 "이미 처리됨"으로 같은 결과 반환, 다른 내용이면 거절 | `work_order_id` 기본키 |
| I4 | Hold된 Lot 진행 | 열린 Hold가 하나라도 있으면 투입 거절(완료는 허용) | — |
| I5 | 사유·결정자 없는 처분 | 빈 값·공백 거절 | `NOT NULL` + `CHECK (btrim(...) <> '')` |
| I6 | 정지·정비 중인 설비에 투입 | 설비가 `AVAILABLE`이 아니면 거절 | 상태 값 `CHECK` |
| I7 | 두 설비가 같은 Lot을 동시에 처리 | Lot 행 `SELECT ... FOR NO KEY UPDATE` 후 처리 중이면 거절 | 부분 유일 인덱스 `(lot_id) WHERE status='STARTED'` |
| I8 | 같은 판정 결과를 다시 보냄 | 같은 (웨이퍼, 모델 해시, 차수)에 같은 내용이면 중복으로 세고, 다른 내용이면 배치 전체 거절 | `UNIQUE (wafer_id, model_sha256, inspect_round)` |
| I9 | 한 Lot에 같은 종류의 열린 Hold가 두 개 | 같은 종류의 열린 Hold가 있으면 새로 만들지 않고 그 Hold 반환 | 부분 유일 인덱스 `(lot_id, rule_name) WHERE closed_at IS NULL` |
| I10 | 폐기된 Lot 진행 | `SCRAPPED`면 투입·완료·Hold 거절 | — |
| I11 | 한 설비가 두 Lot을 동시에 처리 | 설비 행 `SELECT ... FOR NO KEY UPDATE` 후 처리 중인 Lot이 있으면 거절 | 부분 유일 인덱스 `(equipment_id) WHERE status='STARTED'` |
| I12 | 처리 중인 설비를 계획 정비로 변경 | 그 설비에 처리 중인 Lot이 있으면 거절 (고장은 받고 그 Lot에 Hold) | — |
| I13 | 검사 결과 없이 검사 공정 통과 | 검사 다음 공정에 투입하려면 모든 웨이퍼에 현재 차수 판정이 있어야 함 | — |

잠금이 정상 경로이고 제약은 마지막 방어선이다. 잠금이 제대로 걸리면 제약 위반은 일어날 수 없으므로,
제약 위반은 규칙 오류로 바꾸지 않고 그대로 실패시킨다. 규칙마다 테스트가 [tests/test_rules.py](tests/test_rules.py)에 있고,
I7·I11은 연결 두 개를 배리어로 같은 순간에 출발시켜 하나만 성공하는지 확인한다. 설비 고장과 투입·완료·판정 수신이
겹치는 경우는 출발 지연으로 두 순서를 각각 강제해 결과와 교착 여부를 확인한다.

- I7 잠금을 뺀 실험에서는 같은 공정 중복을 막는 I2 제약(`UNIQUE (lot_id, step_code)`)이 먼저 막았다.
- 외래키 확인이 참조 행에 거는 잠금 때문에 생긴 교착을 동시성 테스트로 찾아 행 잠금 방식을 바꿨다
  (`FOR UPDATE` → `FOR NO KEY UPDATE`, 경위는 [docs/decisions.md](docs/decisions.md)).

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

## 2단계: 장비 연동과 자동 정지

검사 장비는 DB에 직접 쓰지 않고 MES API([mes/api.py](mes/api.py))로만 기록한다. **장비는 공정 규칙을 모르므로,
모든 기록이 MES 규칙을 지나가는 입구를 하나로 두기 위해서다.** 실제 장비 통신 규격(SECS/GEM) 대신 HTTP를 썼다.

```
검사 장비 역할 스크립트 ──HTTP──▶ MES API (FastAPI, 동기) ──▶ mes/service.py (규칙·잠금) ──▶ PostgreSQL
(equipment/send_results.py)        입력 형식 검사(pydantic)        한 요청 = 한 트랜잭션
```

| 메서드 | 경로 | 하는 일 |
|---|---|---|
| GET | `/equipment` | 공정 순서, 공정별 설비·상태, 검사 공정 이름 |
| POST | `/lots` | Lot·웨이퍼 생성 (같은 구성 재요청은 기존 결과, 다른 구성은 409) |
| POST | `/work-orders` | 작업 지시에 따른 투입 |
| POST | `/work-orders/{id}/complete` | 완료 |
| POST | `/lots/{lot_id}/inspection-results` | 검사 판정 배치 수신 → 정지 규칙 → Hold (한 트랜잭션) |
| POST | `/equipment/{id}/status` | 설비 상태 변경 (고장이면 처리 중인 Lot에 Hold) |
| POST | `/holds/{id}/disposition` | 처분(해제·재검사·폐기) + 결정자 + 사유 |

- **판정 배치 하나 = Lot 하나의 한 검사 차수**(최대 25장). 하나라도 잘못되면(다른 Lot의 웨이퍼, 약속에 없는 유형 등)
  배치 전체를 거절한다. 같은 내용의 재전송은 실패가 아니라 중복으로 센다(I8).
- **판정을 받는 기간**은 INSPECT를 시작한 뒤부터 다음 공정 투입 전까지이고, 차수는 Lot의 현재 차수와 같아야 한다.
- **정지 규칙**([config/mes.yaml](config/mes.yaml) `stop_rule`): 불량 8종 중 하나로 판정된 웨이퍼가 같은 Lot·같은 차수에
  1장 이상이면 Hold. 장수는 판정 건수가 아니라 서로 다른 웨이퍼 수로 센다. 같은 Lot에 판정이 동시에 들어와도
  Lot 행을 잠근 뒤 세므로 한 번에 하나씩 센다. 같은 Lot·같은 차수에서 규칙 Hold는 한 번만 연다.
- **오류 → HTTP**: 입력 형식 오류 422, 없는 대상 404, 상태 규칙 위반 409, 경쟁 상황에서 DB 제약에 걸린 오류 409,
  예상 못 한 오류는 스택을 로그로 남기고 500.

### API 실행과 판정 전송

```bash
docker compose up -d --wait
.venv/bin/python -m mes.init_db --reset          # 개발용 DB(wafer_mes)에 스키마·기준정보
.venv/bin/uvicorn mes.api:app --port 8000        # 다른 터미널에서
../SKALA_CNN-Optimization/.venv/bin/python -m equipment.send_results   # 시험용 Lot 판정 전송
```

장비 역할 스크립트는 작업 지시를 내리는 역할(Lot 생성, 공정 투입·완료)도 함께 맡는다. Lot마다 공정 순서대로 진행하고,
검사 공정에서 판정을 보낸 뒤, Hold가 열린 Lot은 다음 공정(SHIP) 투입이 막혀 멈춘다. 작업 지시 ID를 `{Lot}-{공정}`으로
정해 같은 파일을 다시 보내도 같은 요청이 된다.

| 실행 | 보낸 판정 | 새 기록 | 중복 | 새 Hold | 기존 Hold | FINISHED Lot | Hold로 멈춘 Lot | 시간 |
|---|---|---|---|---|---|---|---|---|
| 첫 실행 | 51,087 | 51,087 | 0 | 1,952 | 0 | 410 | 1,952 | 164초 |
| 같은 파일 재실행 | 51,087 | 0 | 51,087 | 0 | 1,952 | 410 | 1,952 | 146초 |

멈춘 Lot 1,952개(82.6%)는 4단계에서 같은 규칙을 판정 파일에 적용해 계산한 수와 같다.

## 4단계: 검사 모델 Lot 단위 재평가

12번 모델은 웨이퍼 단위 무작위 분할로 학습해서, 같은 Lot의 다른 웨이퍼를 학습한 채 시험을 봤을 수 있다.
같은 30,519장을 Lot 단위로 나눠(같은 Lot이 학습·검증·시험에 걸치지 않게) 12번과 같은 설정으로 다시 학습해 비교했다.

```bash
../SKALA_CNN-Optimization/.venv/bin/python -m equipment.stage4 plan      # 분할 확인, 1에폭 시간
../SKALA_CNN-Optimization/.venv/bin/python -m equipment.stage4 train     # MPS 학습 → artifacts/
../SKALA_CNN-Optimization/.venv/bin/python -m equipment.stage4 evaluate  # 비교 표
../SKALA_CNN-Optimization/.venv/bin/python -m equipment.stage4 predict   # 시험용 Lot 전체 판정
```

- **분할**: `StratifiedGroupKFold`(groups=lotName, 클래스 층화, seed 42). 축소 데이터 Lot 11,823개를
  train 7,093 / valid 2,368 / test 2,362개(웨이퍼 18,311 / 6,104 / 6,104장)로 나눴고 분할 사이 Lot 교집합은 0이다.
  가장 적은 클래스는 Test의 Near-full 30장이다. 12번 Test와 겹치는 웨이퍼는 1,224장(20.1%)이다.
- **학습**: 12번과 같은 설정([config/equipment.yaml](config/equipment.yaml) `train`), MPS, 6.2분(에폭당 6.0초).
  최고 Valid Macro-F1 0.9022(52에폭), 62에폭에서 조기 종료. 에폭별 기록은
  [evaluation/stage4_train_history.csv](evaluation/stage4_train_history.csv). 체크포인트 SHA-256 `1e02579e…32a7bb`(저장소에 없음).

### 비교 (Test Macro-F1, CPU 판정)

| | 데이터 | 장수 | 12번 모델 | Lot 단위 모델 |
|---|---|---|---|---|
| (가) | 12번의 웨이퍼 단위 Test | 6,104 | **0.8805** | — |
| (나) | Lot 단위 Test | 6,104 | — | **0.8887** |
| (다) | 두 Test가 겹치는 웨이퍼 | 1,224 | 0.8821 | 0.8758 |
| (다-1) | 그중 12번 train에 같은 Lot 웨이퍼가 있었던 것 | 857 | 0.8606 | 0.8529 |
| (다-2) | 그중 없었던 것 | 367 | 0.8535 | 0.8465 |

클래스별 F1과 장수는 [evaluation/stage4_comparison.csv](evaluation/stage4_comparison.csv)에 있다.
"12번 train"은 train만 센 것이다(train+valid로 세면 1,224장 중 891장). valid는 가중치 학습에 쓰이지 않고
멈출 에폭 선택에만 쓰였기 때문이다.

- 같은 Lot을 미리 본 효과가 있다면 12번 모델은 (다-1)에서만 새 모델보다 높아야 한다. 결과는 (다-1)에서 +0.0077,
  (다-2)에서 +0.0070으로 두 쪽 차이가 비슷해, **이 데이터에서는 같은 Lot을 미리 본 효과가 보이지 않았다.**
  Lot 단위 모델도 자기 Test에서 0.8887로 12번(0.8805)보다 낮지 않았다.
- 한계: 시드 하나로 한 비교라 결론이 아니라 방향이다. (다-2)는 367장으로 작고 클래스 구성이 (다-1)과 크게 다르다
  (Edge-Ring 8장 대 402장, none 160장 대 40장). 그래서 (다-1)과 (다-2)의 점수끼리는 비교하지 않고, 같은 칸에서 두 모델만 비교했다.
  (다)의 Near-full은 4장, Donut은 14장이라 이 두 클래스의 F1은 한두 장에 크게 흔들린다.
  MPS 학습은 같은 seed로 다시 돌려도 결과가 조금 다를 수 있다.

### MES에 올릴 판정 데이터

Lot 단위 Test의 Lot은 어떤 웨이퍼도 학습에 쓰이지 않았으므로, 원본에서 그 Lot에 속한 웨이퍼 **전부**를 새 모델로 판정했다
(축소 데이터에 없던 라벨 없음·none 웨이퍼 포함, CPU 판정).

- [data/lot_test_predictions_exp12_lotsplit.csv](data/lot_test_predictions_exp12_lotsplit.csv): Lot 2,362개, 웨이퍼 51,087장, 9.34 MB.
  Lot당 웨이퍼 1~25장(중앙값 25, 25장인 Lot 1,615개), waferIndex 1~25. 열은 0단계 판정 파일과 같다.
- [evaluation/lot_test_labels.csv](evaluation/lot_test_labels.csv): 정답(평가 전용, MES 코드는 읽지 않음).
  라벨 있음 23,192장(불량 5,104, none 18,088), 라벨 없음 27,895장.
- 판정 유형 분포: none 39,122 / Edge-Loc 3,701 / Edge-Ring 2,465 / Center 2,391 / Loc 2,196 / Random 474 /
  Scratch 457 / Donut 202 / Near-full 79.

### 정지 규칙 후보 (검증용 Lot)

정지 규칙의 기준값(정지 대상 유형, 판정 확률 하한, Lot당 최소 장수)은 시험용 Lot이 아니라 **검증용 Lot**으로 고른다.
시험용 Lot으로 고르면 시험 결과에 맞춰 고른 셈이 되기 때문이다. 검증용 Lot 2,368개의 원본 웨이퍼 51,747장을 같은 모델로 판정했다
([data/lot_valid_predictions_exp12_lotsplit.csv](data/lot_valid_predictions_exp12_lotsplit.csv), 정답은
[evaluation/lot_valid_labels.csv](evaluation/lot_valid_labels.csv)).

```bash
../SKALA_CNN-Optimization/.venv/bin/python -m equipment.stage4 predict --split valid
../SKALA_CNN-Optimization/.venv/bin/python -m analysis.stop_rule_candidates
```

후보별 정지율·괜히 멈춘 Lot·놓친 Lot은 [evaluation/stop_rule_candidates_valid.csv](evaluation/stop_rule_candidates_valid.csv)에 있다
(후보 값은 [config/stop_rule_candidates.yaml](config/stop_rule_candidates.yaml)). 라벨 있는 웨이퍼가 없는 Lot 660개는 정답을 알 수 없어
괜히 멈춤·놓침 계산에서 뺐다.

**결정한 정지 규칙: 불량 8종 전부, 확률 하한 없음, Lot당 1장 이상이면 정지.** 표를 보기 전에 정한 기준
"놓친 Lot 최소 → 그다음 괜히 멈춘 Lot 최소"를 그대로 적용한 결과다(놓침 14, 괜히 멈춤 77). 이유와 버린 후보는
[docs/decisions.md](docs/decisions.md)에 있다.

**이 표가 보여주는 것: 어떤 규칙 값을 써도 깨끗한 Lot을 많이 멈춘다.** 정답 기준으로 멈추지 않아야 할 검증용 Lot 96개 중
결정한 규칙은 77개를 멈추고, 확률 하한을 0.9까지 올려도 16개를 멈추면서 놓침이 336개로 늘어난다. 원인은 규칙이 아니라 모델이다.
정상 웨이퍼가 적고(5,000장) 그나마 대부분 라벨 없는 웨이퍼인 데이터로 학습해서, 실제 Lot에 많은 라벨 none 웨이퍼를
Edge-Loc·Loc 등으로 판정한다(검증용 Lot의 라벨 none 웨이퍼 중 Edge-Loc 1,049장, Loc 725장).
그래서 이 시스템에서 **AI 판정은 Lot을 "멈추게"만 하고, 해제·재검사·폐기는 사람이 사유를 남겨 결정한다.**

## 한계 (현재까지)

- 공정 순서, 설비 목록·상태는 합성값이다.
- 실제 장비 통신 규격(SECS/GEM) 대신 HTTP를 썼다. 장비 역할 스크립트가 작업 지시를 내리는 역할도 함께 맡는다.
- 재검사(RETEST)가 검사 설비를 다시 쓰는 과정은 표현하지 않는다(Lot이 설비에 올라가지 않은 채 새 차수 판정을 받는다).
- 판정 재전송의 내용 비교(I8)는 확률 값을 그대로 비교하므로, 같은 값이라도 소수점 표현이 다르면 다른 내용으로 본다.

- **none 클래스에 라벨 없는 웨이퍼가 섞여 학습됐다.** 원본 전처리가 라벨 없는 웨이퍼도 `none`으로 넣었기 때문이다.
  학습용 축소 데이터의 none 또는 라벨 없음 5,000장 중 실제 라벨 none은 974장, 라벨 없음은 4,026장이다.
  Test의 none 1,000장 중에서는 206장과 794장이다. 정답 파일의 `label_source`로 구분하고, AI 판정 평가에서는
  라벨 없음을 정상으로 치지 않고 따로 센다.
- 학습용 축소 데이터는 불량 25,519장 전부와 none 또는 라벨 없음 5,000장으로 만들어 불량이 대부분이다.
- 0단계 판정 파일(12번 모델)은 웨이퍼 단위 분할 모델의 결과이며, MES 시연에는 4단계의 Lot 단위 분할 모델을 쓴다.
- 4단계 비교(웨이퍼 단위 vs Lot 단위)는 시드 하나로 한 것이다.
- 정지 규칙을 고른 검증용 Lot 중 정답을 아는 Lot에서 "멈추지 않아야 할 Lot"은 96개뿐이라, 괜히 멈춘 Lot 비교는 흔들린다.
  일부만 라벨이 있는 Lot은 라벨 있는 웨이퍼로만 정답을 계산해, 놓침은 적게·괜히 멈춤은 많게 잡혔을 수 있다.
- 시험용 Lot은 축소 데이터에 들어간 Lot에서 뽑았고, 축소 데이터는 불량 웨이퍼를 전부 넣었으므로 불량 Lot 비율이 높다
  (라벨 있는 불량 웨이퍼가 1장 이상인 Lot이 2,362개 중 1,607개, 68.0%).
- MES 전체가 아니라 일부 기능이다.

## 다음에 할 수 있는 것

- 정상 웨이퍼(특히 라벨 있는 none)를 충분히 넣어 검사 모델을 다시 학습해 정상 오판정을 줄이기(이번 범위 밖).

## 데이터와 모델 출처

- 데이터: WM-811K (`LSWMD.pkl`). 원본 데이터와 체크포인트는 이 저장소에 없고, 판정 결과 파일과 모델 해시만 있다.
- 검사 모델: 2인 팀 과제 저장소 [SKALA_CNN-Optimization](https://github.com/ahrdyrxkddhfl/SKALA_CNN-Optimization)의
  12번 실험 모델. 전처리·모델 설계·실험은 본인이 맡았다.
