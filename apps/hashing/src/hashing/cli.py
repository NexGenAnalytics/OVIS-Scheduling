from hashing.utils import jaccard_distance

def main() -> None:
  print("hello")

  a = { 1, 2, 3 }
  b = { 1, 2, 4 }

  distance = jaccard_distance(a, b)

  print(f"distance is {distance}")

