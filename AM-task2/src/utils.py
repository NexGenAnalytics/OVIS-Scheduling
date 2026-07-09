import hashlib
import json

def jaccard_distance(a: set, b: set):
  if not a and not b:
    return 0.0

  return 1 - len(a & b) / len(a | b)

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

def read_file(path: str):
  try:
    with open(path, "r") as file:
      lines = file.readlines()
      return lines
  except FileNotFoundError:
    print("File not found.")
    return []

def read_json(path: str):
  try:
    with open(path, "r", encoding="utf-8") as file:
      db = json.load(file)
      return db
  except FileNotFoundError:
    print("File not found.")
    return []

def save_db(path: str, data):
  try:
    with open(path, "w") as file:
      json.dump(data, file, indent=4, default=list)
  except FileNotFoundError:
    print("File not found.")

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
