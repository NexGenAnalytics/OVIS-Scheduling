from typing import List
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from database.models import Job
from database.setup import open_session
from database.utils import hamming_distance

def insert_jobs(session_factory: sessionmaker, *jobs: Job) -> None:
  with session_factory() as session:
    session.add_all(jobs)
    session.commit()

def create_or_edit_job(
  filename: str,
  method: str,
  normalization: set[str],
  simhash32: int,
) -> Job:
  with open_session() as session:
    statement = select(Job).where(
      Job.filename == filename,
      Job.method == method,
    )
    job = session.scalar(statement)

    if job is None:
      job = Job(
        filename=filename,
        method=method,
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

def get_by_method(method: str) -> List[Job]:
  with open_session() as session:
    statement = (
      select(Job)
      .where(Job.method == method)
      .order_by(Job.id)
    )
    return list(session.scalars(statement))

def get_by_id(job_id: int) -> Job | None:
  with open_session() as session:
    statement = (
      select(Job)
      .where(Job.id == job_id)
    )
    return session.scalar(statement)

def get_by_distance(simhash32: int, method: str) -> Job | None:
  jobs = get_by_method(method)

  if not jobs:
    return None

  closest_job = min(
    jobs,
    key=lambda job: hamming_distance(simhash32, job.simhash32)
  )

  return closest_job
