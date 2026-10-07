-- wafer-lot-hold MES 테이블 정의. 변경 이력 도구 없이 이 파일 하나로 관리한다.
-- 시각은 모두 DB가 찍는다(timestamptz DEFAULT now(), now()는 트랜잭션 시작 시각).
-- 각 제약 옆의 I번호는 README "막는 상태" 표의 번호다.

-- 공정 순서. seq 순서대로만 진행한다(I1).
CREATE TABLE route_step (
    step_code text PRIMARY KEY,
    seq       int  NOT NULL UNIQUE CHECK (seq > 0)
);

-- 설비와 현재 상태.
CREATE TABLE equipment (
    equipment_id text PRIMARY KEY,
    step_code    text NOT NULL REFERENCES route_step (step_code),
    status       text NOT NULL CHECK (status IN ('AVAILABLE', 'DOWN', 'MAINTENANCE')),  -- I6
    updated_at   timestamptz NOT NULL DEFAULT now()
);

-- 설비 상태 변경 이력. 추가만 한다.
CREATE TABLE equipment_status_history (
    history_id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    equipment_id text NOT NULL REFERENCES equipment (equipment_id),
    from_status  text NOT NULL CHECK (from_status IN ('AVAILABLE', 'DOWN', 'MAINTENANCE')),
    to_status    text NOT NULL CHECK (to_status IN ('AVAILABLE', 'DOWN', 'MAINTENANCE')),
    changed_by   text NOT NULL CHECK (btrim(changed_by) <> ''),
    reason       text NOT NULL CHECK (btrim(reason) <> ''),
    changed_at   timestamptz NOT NULL DEFAULT now(),
    CHECK (from_status <> to_status)
);

-- Lot. Hold 여부는 상태에 넣지 않고 열린 hold 행으로 판단한다(같은 정보를 두 곳에 두지 않기 위해).
CREATE TABLE lot (
    lot_id            text PRIMARY KEY,
    status            text NOT NULL CHECK (status IN ('WAITING', 'IN_PROCESS', 'FINISHED', 'SCRAPPED')),
    current_step_code text REFERENCES route_step (step_code),  -- 대기 중이거나 처리 중인 공정
    inspect_round     int  NOT NULL DEFAULT 1 CHECK (inspect_round >= 1),  -- RETEST마다 1 증가
    created_at        timestamptz NOT NULL DEFAULT now(),
    CHECK ((status = 'FINISHED') = (current_step_code IS NULL))
);

CREATE TABLE wafer (
    wafer_id    text PRIMARY KEY,
    lot_id      text NOT NULL REFERENCES lot (lot_id),
    wafer_index int  NOT NULL CHECK (wafer_index > 0),
    UNIQUE (lot_id, wafer_index)
);

-- 작업 지시: Lot 하나를 정해진 공정·설비에서 처리하라는 지시 한 건. ID는 지시하는 쪽이 정한다.
CREATE TABLE work_order (
    work_order_id text PRIMARY KEY,  -- I3: 같은 지시는 한 번만 기록
    lot_id        text NOT NULL REFERENCES lot (lot_id),
    step_code     text NOT NULL REFERENCES route_step (step_code),
    equipment_id  text NOT NULL REFERENCES equipment (equipment_id),
    status        text NOT NULL CHECK (status IN ('STARTED', 'COMPLETED', 'ABORTED')),
    started_at    timestamptz NOT NULL DEFAULT now(),
    ended_at      timestamptz,
    UNIQUE (lot_id, step_code),       -- I2: 한 Lot은 한 공정을 한 번만 처리
    CHECK ((status = 'STARTED') = (ended_at IS NULL))
);
-- I7: 한 Lot은 동시에 한 설비에서만 처리 중일 수 있다(잠금이 정상 경로, 이 인덱스는 마지막 방어선).
CREATE UNIQUE INDEX work_order_one_started_per_lot ON work_order (lot_id) WHERE status = 'STARTED';
-- I11: 한 설비는 동시에 한 Lot만 처리한다.
CREATE UNIQUE INDEX work_order_one_started_per_equipment ON work_order (equipment_id) WHERE status = 'STARTED';

-- 검사 판정(AI). 기록 후 수정하지 않는다. 정답 라벨은 넣지 않는다.
CREATE TABLE inspection_result (
    result_id     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    wafer_id      text NOT NULL REFERENCES wafer (wafer_id),
    work_order_id text NOT NULL REFERENCES work_order (work_order_id),  -- 판정을 낸 INSPECT 작업 지시
    inspect_round int  NOT NULL CHECK (inspect_round >= 1),
    model_sha256  char(64) NOT NULL CHECK (model_sha256 ~ '^[0-9a-f]{64}$'),
    pred_label    text NOT NULL CHECK (btrim(pred_label) <> ''),
    probabilities jsonb NOT NULL,  -- {"클래스 이름": 확률, ...}
    received_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (wafer_id, model_sha256, inspect_round)  -- I8
);

-- Hold(정지). 규칙이 열었으면 trigger_result_id, 사람이 열었으면 opened_by 중 정확히 하나가 있다.
CREATE TABLE hold (
    hold_id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    lot_id            text NOT NULL REFERENCES lot (lot_id),
    rule_name         text NOT NULL CHECK (btrim(rule_name) <> ''),  -- 사람이 연 Hold는 'MANUAL'
    trigger_result_id bigint REFERENCES inspection_result (result_id),
    opened_by         text CHECK (btrim(opened_by) <> ''),
    inspect_round     int  NOT NULL CHECK (inspect_round >= 1),
    opened_at         timestamptz NOT NULL DEFAULT now(),
    closed_at         timestamptz,  -- 처분과 같은 트랜잭션에서 채운다
    CHECK (num_nonnulls(trigger_result_id, opened_by) = 1),
    CHECK ((opened_by IS NOT NULL) = (rule_name = 'MANUAL'))
);
-- I9: Lot당 열린 Hold는 하나.
CREATE UNIQUE INDEX hold_one_open_per_lot ON hold (lot_id) WHERE closed_at IS NULL;

-- 처분(사람의 결정). Hold 하나에 처분 하나.
CREATE TABLE hold_disposition (
    disposition_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    hold_id        bigint NOT NULL UNIQUE REFERENCES hold (hold_id),
    action         text NOT NULL CHECK (action IN ('RELEASE', 'RETEST', 'SCRAP')),
    decided_by     text NOT NULL CHECK (btrim(decided_by) <> ''),  -- I5
    reason         text NOT NULL CHECK (btrim(reason) <> ''),      -- I5
    decided_at     timestamptz NOT NULL DEFAULT now()
);

-- Lot 이력. 추가만 한다.
CREATE TABLE lot_history (
    event_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    lot_id        text NOT NULL REFERENCES lot (lot_id),
    event         text NOT NULL CHECK (event IN
                      ('CREATED', 'TRACK_IN', 'TRACK_OUT', 'HOLD', 'RELEASE', 'RETEST', 'SCRAP')),
    step_code     text REFERENCES route_step (step_code),
    equipment_id  text REFERENCES equipment (equipment_id),
    work_order_id text REFERENCES work_order (work_order_id),
    hold_id       bigint REFERENCES hold (hold_id),
    event_at      timestamptz NOT NULL DEFAULT now()
);
