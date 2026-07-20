from database.models import Job

def print_jobs(jobs: list[Job]) -> None:
  if not jobs:
    print("No jobs found.")
    return

  print(*jobs, sep="\n")

def hamming_distance(x: set, y: set):
  return (x ^ y).bit_count()
