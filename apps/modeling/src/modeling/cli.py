import argparse
import joblib
import pandas as pd

from pathlib import Path

from common.utils import read_file

def filter_data(foldername: str, verbose = True):
  """
  Given a path to a folder, return each first level folders where
  'required_files' are in.
  """
  parent = Path(foldername)
  required_files = ["cpu.txt", "deck.txt", "memory.txt"]
  folders = []

  for child in parent.iterdir():
    if not child.is_dir():
      if verbose: print(f"{child.name}: is a file")
      continue

    missing_files = [
      filename
      for filename in required_files
      if not (child / filename).is_file()
    ]

    if missing_files:
      if verbose: print(f"{child.name}: missing {' and '.join(missing_files)}")
      continue

    if verbose: print(f"{child.name} ✅")
    folders.append(child)

  return folders

def create_features(folders: list, verbose = True):
  """
  Given correct folders, extract data and return a features list
  """
  features = []

  for folder in folders:
    feature = {}

    feature["NAME"] = str(folder)

    cpu_file = folder / "cpu.txt"
    cpu_content = read_file(cpu_file, int)
    feature["CPU"] = cpu_content

    memory_file = folder / "memory.txt"
    memory_content = read_file(memory_file, int)
    feature["MEMORY"] = memory_content

    deck_file = folder / "deck.txt"
    deck_content = read_file(deck_file, str)
    feature["DECK"] = deck_content

    features.append(feature)

  return features

def train_model(data):
  print("train TODO TODO TODO TODO TODO TODO TODO TODO TODO")

  # TODO

  return []

def main() -> None:
  print("M, start")

  parser = argparse.ArgumentParser()
  commands = parser.add_mutually_exclusive_group(required=True)
  commands.add_argument("--train", metavar="PATH")
  commands.add_argument("--test", metavar="PATH")
  args = parser.parse_args()

  MODEL_PATH = Path("data/output/model.joblib")

  match vars(args):

    case {"train": str(path)}:
      folders = filter_data(path, verbose = False)
      print("M, folders filtered")

      features = create_features(folders, verbose = True)
      data = pd.DataFrame(features)
      print("M, data extracted")

      model = train_model(data)
      print("M, model trained")

      MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
      joblib.dump(model, MODEL_PATH)
      print("M, model saved")

    case {"test": str(path)}:
      features = [] # TODO TODO TODO TODO TODO TODO TODO TODO
      data = pd.DataFrame(features)
      print("M, data loaded")

      model = joblib.load(MODEL_PATH)
      print("M, model loaded")

      prediction = model.predict(data)
      print(f"M, prediction => {prediction}")

  print("M, end")
