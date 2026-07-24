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
  args = parser.parse_args()

  filename = args.input
  method = args.normalizer

  normalizer = get_normalizer(method)
  normalization, simhash32 = create_hash(filename, normalizer)

  if args.save:
    job = create_or_edit_job(filename, method, normalization, simhash32)

  print("#, end")
