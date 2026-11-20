from typing import List
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from database.models import Job
from database.setup import open_session

def insert_jobs(session_factory: sessionmaker, *jobs: Job) -> None:
  with session_factory() as session:
    session.add_all(jobs)
    session.commit()

def create_or_edit_job(
  name: str,
  cpu_value: int,
  memory_value: int,
) -> Job:
  with open_session() as session:
    statement = select(Job).where(
      Job.name == name,
    )
    job = session.scalar(statement)

    if job is None:
      job = Job(
        name=name, max_cpu=cpu_value, max_memory=memory_value,
      )
      session.add(job)
      print("D, create job")
    else:
      job.max_cpu = cpu_value
      job.max_memory = memory_value
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

def get_by_id(job_id: int) -> Job | None:
  with open_session() as session:
    statement = (
      select(Job)
      .where(Job.id == job_id)
    )
    return session.scalar(statement)
