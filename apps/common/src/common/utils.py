import sys
from typing import Callable, TypeVar

T = TypeVar("T")

def read_file(path: str, convert: Callable[[str], T]) -> list[T]:
  """
  Read text file, line by line.
  """
  try:
    with open(path, "r") as file:
      return [
        convert(line.strip())
        for line in file
        if line.strip()
      ]
  except FileNotFoundError:
    print(f"File not found: {path}")
    sys.exit(1) # failure
  except ValueError as error:
    print(f"Invalid value in {path}: {error}")
    sys.exit(1)
