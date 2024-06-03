import psycopg2
import contextlib
import os
from dotenv import load_dotenv
from pathlib import Path

@contextlib.contextmanager
def create_database_connection(**kwds):
    load_dotenv(dotenv_path=Path('./embedding_store/.env'))
    load_dotenv()
    db_conn = psycopg2.connect(
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        host="localhost",
        port=5432,  # The port you exposed in docker-compose.yml
        database=os.getenv('POSTGRES_DB')
    )
    try:
        yield db_conn
    finally:
        db_conn.close()