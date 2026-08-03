import argparse

from hashing.utils import create_hash
from database.jobs import create_or_edit_job
from hashing.normalizers import NORMALIZERS

def main() -> None:
  print("#, start")
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True)
  parser.add_argument("--normalizer", required=True, choices=NORMALIZERS)
  parser.add_argument("--save", action="store_true")
  parser.add_argument("--weights")
  args = parser.parse_args()

  filename = args.input
  method = args.normalizer
  weights = args.weights

  normalization, simhash32 = create_hash(filename, method, weights)
  print(f"#, hash: {simhash32}")

  if args.save:
    job = create_or_edit_job(filename, method, normalization, simhash32)

  print("#, end")
