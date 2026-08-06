import argparse
import joblib
import pandas as pd

from pathlib import Path

from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

def create_features(folders: list):
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
  """
  Train a small MLP to predict the CPU and memory time series.
  """
  if data.empty:
    raise ValueError("No complete training profiles were found")

  target_lengths = {
    (len(row.CPU), len(row.MEMORY))
    for row in data.itertuples()
  }
  if len(target_lengths) != 1:
    raise ValueError("All CPU and memory series must have the same length")

  cpu_length, memory_length = target_lengths.pop()
  if cpu_length == 0 or cpu_length != memory_length:
    raise ValueError("CPU and memory series must be non-empty and equally sized")

  decks = data["DECK"].map("\n".join)
  targets = [
    cpu + memory
    for cpu, memory in zip(data["CPU"], data["MEMORY"])
  ]

  model = TransformedTargetRegressor(
    regressor=make_pipeline(
      TfidfVectorizer(),
      MLPRegressor(
        hidden_layer_sizes=(8,),
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
      ),
    ),
    transformer=StandardScaler(),
  )
  model.fit(decks, targets)
  return model

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

      features = create_features(folders)
      data = pd.DataFrame(features)
      print("M, data extracted")

      model = train_model(data)
      print("M, model trained")

      MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
      joblib.dump(model, MODEL_PATH)
      print("M, model saved")

    case {"test": str(path)}:
      model = joblib.load(MODEL_PATH)
      print("M, model loaded")

      deck = "\n".join(read_file(path, str))
      print("M, deck loaded")

      prediction = model.predict([deck])[0]
      print("M, prediction made")

      midpoint = len(prediction) // 2
      cpu = [max(0.0, round(float(value), 2)) for value in prediction[:midpoint]]
      memory = [max(0.0, round(float(value), 2)) for value in prediction[midpoint:]]
      print(f"M, prediction => cpu: {cpu} / memory: {memory}")

  print("M, end")
