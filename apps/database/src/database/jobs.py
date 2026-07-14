from typing import List

from database.models import Job
from database.setup import open_session
from sqlalchemy import select

def create_job(name: str) -> Job:
  with open_session() as session:
    job = Job(name=name)
    session.add(job)
    session.commit()
    session.refresh(job)
    return job

def list_jobs() -> List[Job]:
  with open_session() as session:
    statement = (
      select(Job)
      .order_by(Job.id)
    )
    return list(session.scalars(statement))
