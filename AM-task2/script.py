import hashlib
import sys

def readFile(path: str):
  try:
    with open(path, "r") as file:
      lines = file.readlines()
      return lines
  except FileNotFoundError:
    print("File not found.")
    return []

def normalize_deck(lines: list[str]):
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

def jaccard_distance(a, b):
  if not a and not b:
    return 0.0

  return 1 - len(a & b) / len(a | b)

def simhash(tokens, bits=64):
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

def hamming_distance(x, y):
  return (x ^ y).bit_count()

if __name__ == "__main__":
  path1 = sys.argv[1]
  path2 = sys.argv[2]

  configContent1 = readFile(path1)
  configContent2 = readFile(path2)

  a = normalize_deck(configContent1)
  b = normalize_deck(configContent2)

  print(f"Between *{path1.split('/')[-1]}* and *{path2.split('/')[-1]}*:")
  print(f"Jaccard distance (0 same - 1 different): {jaccard_distance(a, b)}")

  A = simhash(a)
  B = simhash(b)

  print(f"Hamming distance (0 same - >16 different): {hamming_distance(A, B)}")
