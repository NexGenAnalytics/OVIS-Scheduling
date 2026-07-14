import os
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DEFAULT = "sqlite:///data/output/db.sqlite3"

class Base(DeclarativeBase):
  pass

def get_database_url(database_url: Optional[str] = None) -> str:
  return database_url or DEFAULT

def _prepare_database_url(database_url: str) -> str:
  if not database_url.startswith("sqlite:///"):
    return database_url

  sqlite_path = database_url.removeprefix("sqlite:///")
  if sqlite_path == ":memory:":
    return database_url

  path = Path(sqlite_path)
  if not path.is_absolute():
    path = Path.cwd() / path

  path.parent.mkdir(parents=True, exist_ok=True)
  return f"sqlite:///{path}"

def get_engine(database_url: Optional[str] = None) -> Engine:
  return create_engine(_prepare_database_url(get_database_url(database_url)))

def get_session_factory(engine: Engine) -> sessionmaker:
  return sessionmaker(bind=engine, expire_on_commit=False)

def init_db(database_url: Optional[str] = None) -> Engine:
  import database.models

  engine = get_engine(database_url=database_url)
  Base.metadata.create_all(engine)
  return engine

def open_session(database_url: Optional[str] = None) -> Session:
  engine = init_db(database_url=database_url)
  return get_session_factory(engine)()
