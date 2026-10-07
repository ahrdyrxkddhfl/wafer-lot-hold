"""테스트 전용 DB 준비와 테스트마다 정리.

동시성 테스트는 연결 두 개가 각자 커밋하므로 "테스트 끝에 롤백"으로는 격리되지 않는다.
그래서 모든 테스트가 같은 방식으로, 시작 전에 모든 테이블을 TRUNCATE하고 기준정보를 다시 넣는다.
시작 전에 비우므로 실패한 테스트가 남긴 데이터는 다음 테스트 전까지 DB에서 들여다볼 수 있다.
"""
from collections.abc import Callable, Iterator

import psycopg
import pytest
from psycopg import sql

from mes.db import apply_schema, connect, load_config, seed_master_data

MAINTENANCE_DB = "postgres"  # PostgreSQL이 항상 만들어 두는 관리용 DB


@pytest.fixture(scope="session")
def mes_cfg() -> dict:
    """MES 설정."""
    return load_config()


@pytest.fixture(scope="session")
def test_dbname(mes_cfg: dict) -> str:
    """테스트 DB를 (없으면 만들고) 스키마를 새로 깐 뒤 이름을 돌려준다."""
    name = mes_cfg["db"]["test_dbname"]
    if name == mes_cfg["db"]["dbname"]:
        pytest.exit("test_dbname이 개발용 dbname과 같아 테스트를 멈춤 (개발 데이터 보호)")
    with connect(mes_cfg, MAINTENANCE_DB) as admin:
        exists = admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if exists is None:
            # DB 이름은 값 바인딩을 쓸 수 없어 sql.Identifier로 따옴표 처리해 넣는다.
            admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    with connect(mes_cfg, name) as conn:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
        apply_schema(conn)
    return name


@pytest.fixture
def connect_test(mes_cfg: dict, test_dbname: str) -> Callable[[], psycopg.Connection]:
    """테스트 DB에 새 연결을 여는 함수(동시성 테스트에서 스레드마다 하나씩 쓴다)."""
    return lambda: connect(mes_cfg, test_dbname)


@pytest.fixture(autouse=True)
def clean_db(mes_cfg: dict, connect_test: Callable[[], psycopg.Connection]) -> None:
    """테스트 시작 전에 모든 테이블을 비우고 공정·설비 기준정보를 다시 넣는다."""
    with connect_test() as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'").fetchall()]
        conn.execute(sql.SQL("TRUNCATE {} RESTART IDENTITY CASCADE").format(
            sql.SQL(", ").join(sql.Identifier(t) for t in tables)))
        seed_master_data(conn, mes_cfg)


@pytest.fixture
def conn(connect_test: Callable[[], psycopg.Connection]) -> Iterator[psycopg.Connection]:
    """테스트 하나가 쓰는 연결."""
    with connect_test() as c:
        yield c
