import argparse

from database.jobs import get_by_distance
from hashing.utils import create_hash

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
      _, simhash32 = create_hash(filename, method)
      job = get_by_distance(simhash32)
      print(f"Nearest: {job}")

  print("S, end")
