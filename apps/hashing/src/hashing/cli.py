import argparse

from hashing.utils import read_file, simhash
from database.jobs import create_or_edit_job
from hashing.normalizers import NORMALIZERS

def main() -> None:
  print("#, start")
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True)
  parser.add_argument("--normalizer", required=True, choices=NORMALIZERS)
  args = parser.parse_args()

  filename = args.input
  method = args.normalizer

  lines = read_file(filename)
  normalizer = NORMALIZERS[method]

  normalization = normalizer(lines)
  simhash32 = simhash(normalization)

  job = create_or_edit_job(filename, method, normalization, simhash32)
  print("#, end")
