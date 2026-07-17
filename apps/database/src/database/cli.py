import argparse

from database.jobs import list_jobs

def main() -> None:
  print("D, start")

  parser = argparse.ArgumentParser(description="Inspect the jobs database.")
  parser.add_argument(
    "--list-jobs",
    action="store_true",
    help="list every job ordered by ID",
  )
  args = parser.parse_args()

  if args.list_jobs:
    jobs = list_jobs()
    print(*jobs, sep="\n")

  print("D, end")

