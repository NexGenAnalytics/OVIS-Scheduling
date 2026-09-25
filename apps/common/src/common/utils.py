import sys

def read_file(path: str, converter=str) -> list:
  """
  Read and convert the non-empty lines of a text file.
  """
  try:
    with open(path, "r", encoding="utf-8") as file:
      lines = (line.strip() for line in file)
      return [converter(line) for line in lines if line]
  except FileNotFoundError:
    print("File not found.")
    sys.exit(1) # failure
