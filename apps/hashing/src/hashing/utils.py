import hashlib
import sys

def normalize_deck(lines: list[str]) -> set:
  """
  Method:
  - From a list of strings, extract "-D", clean, strip and stack them.

  Example 1:
  - Input: ["cmake" "-S", ".", "-B", "build-cuda", "-DCMAKE_BUILD_TYPE=Release"]
  - Output: { "CMAKE_BUILD_TYPE=Release" }

  Example 2:
  - Input: ["cmake", "-DCMAKE_BUILD_TYPE=Debug", "-DTPL_ENABLE_MPI=ON"]
  - Output: { "CMAKE_BUILD_TYPE=Debug", "TPL_ENABLE_MPI=ON" }
  """
  tokens = set()

  for line in lines:
    # clean
    line = line.strip().rstrip("\\").strip()

    # get persistent variable
    if line.startswith("-D"):
      keyval = line[2:]
      key, value = keyval.split("=", 1)
      tokens.add(f"{key.strip()}={value.strip()}")

  return tokens

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

def simhash(tokens, bits=32) -> int:
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
