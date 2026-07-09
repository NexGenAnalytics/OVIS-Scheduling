from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from database.setup import Base

class Job(Base):
  __tablename__ = "jobs"

  id: Mapped[int] = mapped_column(Integer, primary_key=True)
  name: Mapped[str] = mapped_column(String(255), nullable=False)

  def __repr__(self) -> str:
    return (
      f"Job(id={self.id!r}, name={self.name!r})"
    )
