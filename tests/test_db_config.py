"""DB 접속 설정 테스트: 환경변수 POSTGRES_PORT가 config의 포트를 덮어쓴다."""
from mes.db import db_port


def test_db_port_uses_env_when_set(monkeypatch):
    cfg = {"db": {"port": 5434}}
    monkeypatch.setenv("POSTGRES_PORT", "5435")
    assert db_port(cfg) == 5435


def test_db_port_falls_back_to_config(monkeypatch):
    monkeypatch.delenv("POSTGRES_PORT", raising=False)
    assert db_port({"db": {"port": 5434}}) == 5434
