from typing import List

from database.models import Job
from database.setup import open_session
from sqlalchemy import select

def create_or_edit_job(
  filename: str,
  normalization: set[str],
  simhash32: int,
) -> Job:
  with open_session() as session:
    statement = select(Job).where(Job.filename == filename)
    job = session.scalar(statement)

    if job is None:
      job = Job(
        filename=filename,
        normalization=normalization,
        simhash32=simhash32,
      )
      session.add(job)
      print("D, create job")
    else:
      job.normalization = normalization
      job.simhash32 = simhash32
      print("D, update job")

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
