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

OUTPUT_FOLDER = Path("data/output/")
MODEL_PATH = OUTPUT_FOLDER / "model.joblib"

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

def create_targets(data):
  """
  Create padded targets: duration, CPU values, then memory values.
  """
  if data.empty:
    raise ValueError("No complete training profiles were found")

  for row in data.itertuples():
    if not row.CPU or len(row.CPU) != len(row.MEMORY):
      raise ValueError(
        "CPU and memory series in each profile must be non-empty and equally sized"
      )

  maximum_length = max(len(cpu) for cpu in data["CPU"])
  targets = []

  for cpu, memory in zip(data["CPU"], data["MEMORY"]):
    duration = len(cpu)
    cpu_padding = [0] * (maximum_length - duration)
    memory_padding = [0] * (maximum_length - duration)
    targets.append(
      [duration] + cpu + cpu_padding + memory + memory_padding
    )

  return targets

def train_model(data):
  """
  Train one small MLP to predict duration, CPU, and memory.
  """
  targets = create_targets(data)

  decks = data["DECK"].map("\n".join)

  model = TransformedTargetRegressor(
    regressor=make_pipeline(
      TfidfVectorizer(token_pattern=r"(?u)\b\w+\b"),
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

def predict_model(model, deck):
  """
  Predict and remove the padded values.
  """
  prediction = model.predict([deck])[0]
  maximum_length = (len(prediction) - 1) // 2
  duration = max(1, min(maximum_length, round(float(prediction[0]))))

  cpu_start = 1
  memory_start = cpu_start + maximum_length
  cpu = prediction[cpu_start:memory_start][:duration]
  memory = prediction[memory_start:][:duration]

  return duration, cpu, memory

def display_prediction(duration, cpu, memory):
  """
  Display one CPU/memory prediction per minute.
  """
  print(f"M, predicted duration => {duration} minutes")

  for time, (cpu_value, memory_value) in enumerate(zip(cpu, memory)):
    cpu_value = max(0.0, round(float(cpu_value), 2))
    memory_value = max(0.0, round(float(memory_value), 2))
    print(f"time: {time}, cpu: {cpu_value}%, memory: {memory_value}%")

def main() -> None:
  print("M, start")

  parser = argparse.ArgumentParser()
  commands = parser.add_mutually_exclusive_group(required=True)
  commands.add_argument("--train", metavar="PATH")
  commands.add_argument("--test", metavar="PATH")
  args = parser.parse_args()

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

      duration, cpu, memory = predict_model(model, deck)
      print("M, prediction made")

      display_prediction(duration, cpu, memory)

  print("M, end")
