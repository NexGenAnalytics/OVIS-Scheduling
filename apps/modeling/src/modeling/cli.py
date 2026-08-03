import argparse
import joblib
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common.utils import read_file

def load_training(folders):
  rows = []

  for run_folder in folders.iterdir():
    if run_folder.is_dir():
      lines = read_file(run_folder / "deck.txt")
      features = parse_lammps_deck(lines)
      features["NAME"] = run_folder.name

      features["cpu"] = float(read_file(run_folder / "cpu.txt")[0])
      features["memory"] = float(read_file(run_folder / "memory.txt")[0])
      rows.append(features)

  return rows

def parse_lammps_deck(lines: list[str]):
  features = {}

  for line in lines:
    # remove inline comments, then clean whitespace
    line = line.split("#", maxsplit=1)[0].strip()

    # avoid empty
    if not line:
      continue

    # split line
    tokens = line.split()
    command = tokens[0]
    args = tokens[1:]

    # feature definitions
    if command == "dimension":
      features["dimension"] = int(args[0])
    elif command == "atom_style":
      features["atom_style"] = args[0]

  return features

def train_model(data):

  feature_columns = ["dimension", "atom_style"]

  X = data[feature_columns]
  y = data[["cpu", "memory"]].astype(float)

  atom_style_columns = ["atom_style"]

  preprocessing = ColumnTransformer(
    transformers=[
      ("numeric", StandardScaler(), ["dimension"]),
      (
        "atom_style",
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        atom_style_columns,
      ),
    ]
  )

  neural_network = TransformedTargetRegressor(
    regressor=MLPRegressor(
      hidden_layer_sizes=(32, 16),
      max_iter=1000,
      early_stopping=False,
      random_state=42,
    ),
    transformer=StandardScaler(),
  )

  model = make_pipeline(preprocessing, neural_network)

  model.fit(X, y)
  return model

def main() -> None:
  print("M, start")

  parser = argparse.ArgumentParser()
  commands = parser.add_mutually_exclusive_group(required=True)

  commands.add_argument(
    "--train",
    action="store_true",
    help="Train or re-train, the neural network.",
  )

  commands.add_argument(
    "--test",
    metavar="PATH",
  )

  args = parser.parse_args()

  MODEL_PATH = Path("data/output/model.joblib")

  match vars(args):
    case {"train": True}:
      # load data
      trainingfolder = Path("apps/modeling/training_single")
      rows = load_training(trainingfolder)
      data = pd.DataFrame(rows)
      # print(data)

      # train
      model = train_model(data)
      print("M, model trained")

      # save
      MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
      joblib.dump(model, MODEL_PATH)
      print("M, model saved")

    case {"test": str(path)}:
      # input
      lines = read_file(path)
      features = parse_lammps_deck(lines)
      data = pd.DataFrame([features])

      # load and usage
      model = joblib.load(MODEL_PATH)
      prediction = model.predict(data)[0]

      cpu, memory = prediction
      print(f"M, prediction => cpu: {cpu} / memory: {memory}")

  print("M, end")
