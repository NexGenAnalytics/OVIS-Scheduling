import argparse
import sys

from database.utils import print_jobs
from database.jobs import list_jobs, get_by_id

def main() -> None:
  print("D, start")

  parser = argparse.ArgumentParser(description="Inspect the jobs database.")

  commands = parser.add_mutually_exclusive_group(required=True)

  commands.add_argument(
    "--list",
    dest="list_jobs",
    action="store_true",
    help="List every job ordered by ID.",
  )

  commands.add_argument(
    "--getid",
    dest="get_id",
    type=int,
    metavar="ID",
    help="Get a job by its ID.",
  )

  args = parser.parse_args()

  match vars(args):
    case {"list_jobs": True}:
      print_jobs(list_jobs())

    case {"get_id": int(job_id)}:
      job = get_by_id(job_id)
      print(job if job is not None else "Job not found.")

  print("D, end")
