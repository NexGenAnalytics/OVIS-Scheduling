import pathlib2
import sys

from utils import normalize_deck, read_file, save_db, simhash

"""
if __name__ == "__main__":
  path = sys.argv[1]
  print(f"Save files from folder *{path}*:")

  directory = pathlib2.Path(path)
  files = [f for f in directory.iterdir() if f.is_file()]

  objects = []
  for file in files:
    configContent = read_file(file)
    normal = normalize_deck(configContent)
    myhash = simhash(normal)
    obj = {
      "path": str(file),
      "normalization": normal,
      "hash": myhash,
      "metadata": {
        "cpu": {
          "utilization": "",
        },
        "memory": {
          "usage": "",
          "occupancy": ""
        },
        "network": "",
      }
    }
    objects.append(obj)
    print(f"- {file} ready")

  save_db("./db.json", objects)
  print("Files saved!")
"""
