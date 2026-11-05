from sqlalchemy import BigInteger, Integer, PickleType, String
from sqlalchemy.ext.mutable import MutableSet
from sqlalchemy.orm import Mapped, mapped_column

from database.setup import Base

class Job(Base):
  __tablename__ = "jobs"

  id: Mapped[int] = mapped_column(Integer, primary_key=True)
  name: Mapped[str] = mapped_column(String(255), nullable=False)
  max_cpu: Mapped[int] = mapped_column(Integer, nullable=False)
  max_memory: Mapped[int] = mapped_column(Integer, nullable=False)

  def __repr__(self) -> str:
    return (
      f"Job("
      f"id={self.id!r}, "
      f"name={self.name!r}, "
      f"max_cpu={self.max_cpu!r}, "
      f"max_memory={self.max_memory!r}"
      f")"
    )
