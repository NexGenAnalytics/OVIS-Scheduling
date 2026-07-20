import argparse

from database.jobs import list_jobs

def main() -> None:
  print("D, start")

  parser = argparse.ArgumentParser(description="Inspect the jobs database.")

  commands = parser.add_mutually_exclusive_group(required=True)

  commands.add_argument(
    "--list-jobs",
    action="store_true",
    help="List every job ordered by ID.",
  )

  commands.add_argument(
    "--list-by-method",
    metavar="METHOD",
    help="List jobs using the specified method.",
  )

  commands.add_argument(
    "--get-id",
    type=int,
    metavar="ID",
    help="Get a job by its ID.",
  )

  args = parser.parse_args()

  match vars(args):
    case {"list_jobs": True}:
      print_jobs(list_jobs())

    case {"list_by_method": str(method)}:
      print_jobs(get_by_method(method))

    case {"get_id": int(job_id)}:
      job = get_by_id(job_id)
      print(job if job is not None else "Job not found.")

  print("D, end")
