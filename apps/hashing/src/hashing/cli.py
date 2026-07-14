from hashing.utils import jaccard_distance
# from database.jobs import create_job, list_jobs

def main() -> None:
  print("hello")

  a = { 1, 2, 3 }
  b = { 1, 2, 4 }

  distance = jaccard_distance(a, b)

  print(f"distance is {distance}")

  # usage example
  # job = create_job("job test #1")
  # jobs = list_jobs()
