"""MES 설정 읽기, DB 연결, 스키마 적용, 기준정보(공정·설비) 적재."""
import logging
import os
from pathlib import Path

import psycopg
import yaml
from dotenv import load_dotenv

from mes.service import INSPECT_STEP

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "mes.yaml"
SCHEMA_PATH = PROJECT_ROOT / "schema.sql"
PORT_ENV = "POSTGRES_PORT"  # 있으면 config의 db.port 대신 쓴다


def load_config(path: Path = CONFIG_PATH) -> dict:
    """MES 설정(YAML)을 읽는다.

    Args:
        path: 설정 파일 경로.

    Returns:
        설정 dict.
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_password() -> str:
    """DB 비밀번호를 환경변수에서 읽는다. 저장소 루트의 .env가 있으면 먼저 불러온다.

    Returns:
        POSTGRES_PASSWORD 값.

    Raises:
        RuntimeError: 값이 없을 때.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError("POSTGRES_PASSWORD가 없음 (.env.example을 .env로 복사해 채울 것)")
    return password


def connect(cfg: dict, dbname: str) -> psycopg.Connection:
    """DB에 연결한다.

    autocommit으로 열고, 트랜잭션은 호출하는 쪽에서 `with conn.transaction():`으로 직접 연다.
    그래야 어디서 어디까지가 한 트랜잭션인지 코드에 그대로 보인다.

    Args:
        cfg: load_config 결과.
        dbname: 접속할 DB 이름.

    Returns:
        autocommit 연결.
    """
    db = cfg["db"]
    password = get_password()  # .env를 먼저 불러와야 POSTGRES_PORT도 .env에서 읽힌다
    return psycopg.connect(host=db["host"], port=db_port(cfg), user=db["user"],
                           password=password, dbname=dbname, autocommit=True)


def db_port(cfg: dict) -> int:
    """DB 포트. 환경변수 POSTGRES_PORT가 있으면 그 값, 없으면 config의 db.port.

    같은 컴퓨터에서 이 저장소를 하나 더 띄울 때 docker compose와 같은 값을 쓰기 위해서다.
    """
    return int(os.environ.get(PORT_ENV) or cfg["db"]["port"])


def apply_schema(conn: psycopg.Connection) -> None:
    """schema.sql을 실행해 테이블을 만든다. 이미 있으면 오류로 멈춘다.

    Args:
        conn: 대상 DB 연결.
    """
    with conn.transaction():
        conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    logger.info("schema.sql 적용")


def seed_master_data(conn: psycopg.Connection, cfg: dict) -> None:
    """config의 공정 순서와 설비 목록을 넣는다. 설비는 모두 AVAILABLE로 시작한다.

    Args:
        conn: 대상 DB 연결.
        cfg: load_config 결과.
    """
    if INSPECT_STEP not in cfg["route"]:
        raise ValueError(f"config route에 검사 공정 {INSPECT_STEP}가 없음: {cfg['route']}")
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany("INSERT INTO route_step (step_code, seq) VALUES (%s, %s)",
                        [(step, seq) for seq, step in enumerate(cfg["route"], start=1)])
        cur.executemany(
            "INSERT INTO equipment (equipment_id, step_code, status) VALUES (%s, %s, 'AVAILABLE')",
            [(eq, step) for step, eqs in cfg["equipment"].items() for eq in eqs])
    logger.info("기준정보 적재: 공정 %d개, 설비 %d대",
                len(cfg["route"]), sum(len(v) for v in cfg["equipment"].values()))
