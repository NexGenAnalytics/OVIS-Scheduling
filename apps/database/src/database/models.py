from sqlalchemy import BigInteger, Integer, PickleType, String
from sqlalchemy.ext.mutable import MutableSet
from sqlalchemy.orm import Mapped, mapped_column

from database.setup import Base

class Job(Base):
  __tablename__ = "jobs"

  id: Mapped[int] = mapped_column(Integer, primary_key=True)
  filename: Mapped[str] = mapped_column(String(255), nullable=False)
  method: Mapped[str] = mapped_column(String(255), nullable=False)

  normalization: Mapped[set[str]] = mapped_column(
    MutableSet.as_mutable(PickleType),
    nullable=False,
    default=set,
  )

  simhash32: Mapped[int] = mapped_column(
    BigInteger,
    nullable=False
  )

  def __repr__(self) -> str:
    return (
      f"Job(id={self.id!r}, filename={self.filename!r}, method={self.method!r},\n normalization={self.normalization!r}, simhash32={self.simhash32!r})"
    )
