"""
PostgreSQL 및 pgvector 확장 연결 테스트 스크립트
"""
import sys
import psycopg2
from src.config import PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD

def test_connection():
    print("=" * 60)
    print("PostgreSQL 및 pgvector 연결 테스트")
    print("=" * 60)
    print(f"Host: {PG_HOST}")
    print(f"Port: {PG_PORT}")
    print(f"Database: {PG_DATABASE}")
    print(f"User: {PG_USER}")
    print(f"Password: {'*' * len(PG_PASSWORD) if PG_PASSWORD else '(비어있음)'}")
    print("-" * 60)

    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            pg_version = cur.fetchone()[0]
            print(f"[OK] PostgreSQL 연결 성공!\n     버전: {pg_version}")

            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            ext = cur.fetchone()
            if ext:
                print(f"[OK] pgvector 확장 활성화 확인 (version: {ext[0]})")
            else:
                print("[WARNING] 'vector' 확장이 설치되어 있지 않습니다.")
                print("           DB에 접속하여 'CREATE EXTENSION IF NOT EXISTS vector;' 를 실행하세요.")
        conn.close()
        print("=" * 60)
        print("모든 연결 테스트가 성공적으로 통과되었습니다.")
        return 0
    except Exception as e:
        print(f"[ERROR] 연결 실패: {e}")
        print("\n[트러블슈팅 안내]")
        print("1. PostgreSQL 서비스가 실행 중인지 확인하세요.")
        print("2. 프로젝트 루트의 .env 파일에 올바른 PG_PASSWORD 가 설정되어 있는지 확인하세요.")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(test_connection())
