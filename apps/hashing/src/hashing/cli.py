import argparse

from hashing.utils import normalize_deck, read_file, simhash
from database.jobs import create_or_edit_job

def main() -> None:
  print("#, start")
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True)
  args = parser.parse_args()

  filename = args.input
  lines = read_file(filename)
  normalization = normalize_deck(lines)
  simhash32 = simhash(normalization)

  job = create_or_edit_job(filename, normalization, simhash32)
  print("#, end")
