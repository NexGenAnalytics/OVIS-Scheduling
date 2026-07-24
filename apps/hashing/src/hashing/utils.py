import hashlib
import sys

def _read_file(path: str) -> list[str]:
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

def _simhash(tokens, bits=32) -> int:
  """
  Create hash from a set[str].
  """
  vector = [0] * bits

  for token in tokens:
    h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
    for i in range(bits):
      bit = (h >> i) & 1
      vector[i] += 1 if bit else -1

  fingerprint = 0
  for i, value in enumerate(vector):
    if value > 0:
      fingerprint |= 1 << i

  return fingerprint

def create_hash(path: str, normalizer) -> tuple[set[str], int]:
  lines = _read_file(path)
  normalization = normalizer(lines)
  simhash32 = _simhash(normalization)
  return normalization, simhash32
