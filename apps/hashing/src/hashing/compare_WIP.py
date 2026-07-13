import sys

from utils import jaccard_distance, normalize_deck, read_file, simhash

def hamming_distance(x: set, y: set):
  return (x ^ y).bit_count()

"""
if __name__ == "__main__":
  path1 = sys.argv[1]
  path2 = sys.argv[2]

  print(f"Compare *{path1.split('/')[-1]}* and *{path2.split('/')[-1]}*:\n")

  configContent1 = read_file(path1)
  configContent2 = read_file(path2)

  a = normalize_deck(configContent1)
  b = normalize_deck(configContent2)

  print(f"Normalization a: {a}")
  print(f"Normalization b: {b}")
  print(f"Jaccard distance (0 same - 1 different): {jaccard_distance(a, b)}\n")

  A = simhash(a)
  B = simhash(b)

  print(f"Hash A: {A}")
  print(f"Hash B: {B}")

  print(f"Hamming distance (0 same - >16 different): {hamming_distance(A, B)}")
"""
