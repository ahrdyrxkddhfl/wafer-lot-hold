"""개발용 DB(config db.dbname)에 스키마와 기준정보(공정·설비)를 넣는다.

    .venv/bin/python -m mes.init_db            # 비어 있는 DB에만
    .venv/bin/python -m mes.init_db --reset    # 스키마를 지우고 처음부터(기존 데이터 삭제)
"""
import argparse
import logging
import sys

from mes.db import apply_schema, connect, load_config, seed_master_data

logger = logging.getLogger("init_db")


def main() -> int:
    """스키마와 기준정보를 넣는다.

    Returns:
        종료 코드. 성공 0, 테이블이 이미 있는데 --reset이 없으면 1.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="개발용 DB 초기화")
    parser.add_argument("--reset", action="store_true", help="public 스키마를 지우고 다시 만든다(데이터 삭제)")
    args = parser.parse_args()

    cfg = load_config()
    dbname = cfg["db"]["dbname"]
    with connect(cfg, dbname) as conn:
        n_tables = conn.execute("SELECT count(*) FROM pg_tables WHERE schemaname = 'public'").fetchone()[0]
        if n_tables and not args.reset:
            logger.error("%s에 테이블 %d개가 이미 있음. 지우고 다시 만들려면 --reset", dbname, n_tables)
            return 1
        if args.reset:
            conn.execute("DROP SCHEMA public CASCADE")
            conn.execute("CREATE SCHEMA public")
            logger.info("%s public 스키마를 지움", dbname)
        apply_schema(conn)
        seed_master_data(conn, cfg)
    logger.info("%s 초기화 완료", dbname)
    return 0


if __name__ == "__main__":
    sys.exit(main())
