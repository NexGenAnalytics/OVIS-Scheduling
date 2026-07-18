import argparse
import sys

from database.utils import print_jobs, hamming_distance
from database.jobs import list_jobs, get_by_method, get_by_id

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

  commands.add_argument(
    "--get-hash-distance-between",
    nargs=2,
    type=int,
    metavar=("ID_A", "ID_B"),
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

    case {"get_hash_distance_between": [int(id_a), int(id_b)]}:
      job_a = get_by_id(id_a)
      job_b = get_by_id(id_b)
      if job_a is None or job_b is None:
        print("Job(s) not found.")
      else:
        distance = hamming_distance(job_a.simhash32, job_b.simhash32)
        print(f"Hamming distance is {distance}.")
        print("Note: Hamming distance gives results between 0 (same) and 16 (very different).")

  print("D, end")
