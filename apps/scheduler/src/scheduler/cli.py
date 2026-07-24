import argparse

from database.jobs import get_by_distance
from hashing.utils import create_hash
from hashing.normalizers import get_normalizer

def main() -> None:
  print("S, start")

  parser = argparse.ArgumentParser()
  commands = parser.add_mutually_exclusive_group(required=True)

  commands.add_argument(
    "--find-nearest-to",
    nargs=2,
    type=str,
    metavar=("FILENAME", "METHOD")
  )

  args = parser.parse_args()

  match vars(args):
    case {"find_nearest_to": [str(filename), str(method)]}:
      normalizer = get_normalizer(method)
      _, simhash32 = create_hash(filename, normalizer)
      job = get_by_distance(simhash32)
      print(f"Nearest: {job}")

  print("S, end")
