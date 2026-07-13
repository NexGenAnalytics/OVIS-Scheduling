import sys

from hashing.utils import jaccard_distance, normalize_deck, read_json, read_file

def find_nearest(db, file):
  best = None

  for entry in db:
    entry_tokens = set(entry["normalization"])
    distance = jaccard_distance(file, entry_tokens)

    result = {
      "distance": distance,
      "entry": entry
    }

    if (best is None or distance < best["distance"]):
      best = result

  return best

"""
if __name__ == "__main__":
  db_path = sys.argv[1]
  file_path = sys.argv[2]

  db = read_json(db_path)
  file_content = read_file(file_path)
  file = normalize_deck(file_content)

  nearest = find_nearest(db, file)
  print(nearest)
"""
