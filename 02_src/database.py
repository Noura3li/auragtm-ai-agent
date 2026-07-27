import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:AuraGTM_123@localhost:5432/auragtm_db"
)

SQL_ECHO = os.getenv("SQL_ECHO", "false").lower() == "true"

engine = create_engine(
    DATABASE_URL,
    echo=SQL_ECHO
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)