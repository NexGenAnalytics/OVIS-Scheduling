import argparse
from common import define_deck_vector, Feature, parse_inputdeck
import joblib
import numpy as np
from pathlib import Path

def init_parser() -> argparse.Namespace:
  parser = argparse.ArgumentParser(prog="modeling-prediction")
  parser.add_argument("--model", type=Path, required=True)
  parser.add_argument("--input", type=Path, required=True)
  return parser.parse_args()

def main() -> None:
  args: argparse.Namespace = init_parser()

  model = joblib.load(args.model)
  print(f"Model: {args.model}")

  content = Path(args.input).read_text(encoding="utf-8")
  inputdeck = parse_inputdeck(content)
  deck_vector = define_deck_vector(inputdeck)

  X_new = np.asarray([deck_vector], dtype=float)

  predicted_values = model.predict(X_new)[0]

  TARGET_NAMES = ("max", "mean", "min")
  cpu_prediction = dict(zip(TARGET_NAMES, predicted_values))
  print(cpu_prediction)
