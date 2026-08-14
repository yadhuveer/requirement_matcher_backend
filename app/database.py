import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL,pool_pre_ping=True,pool_recycle=300)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        # A long-running request can leave this session's connection idle long
        # enough that Neon drops it; closing a dead connection then raises during
        # teardown. Swallow it so a finished request isn't turned into a 500.
        try:
            db.close()
        except Exception:
            pass

    