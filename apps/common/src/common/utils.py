import sys

def read_file(path: str) -> list[str]:
  """
  Read text file, line by line.
  """
  try:
    with open(path, "r") as file:
      lines = file.readlines()
      return lines
  except FileNotFoundError:
    print("File not found.")
    sys.exit(1) # failure
