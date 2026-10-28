import argparse
from common import DECK_FEATURE_NAMES, define_deck, parse_inputdeck
import joblib
import numpy as np
from pathlib import Path

def init_parser() -> argparse.Namespace:
  parser = argparse.ArgumentParser(prog="modeling-prediction")
  parser.add_argument("--model", type=Path, required=True)
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument(
    "--particle-count", type=int, required=True,
    help="Initial number of simulated particles (training's particle_count).",
  )
  args = parser.parse_args()
  if args.particle_count <= 0:
    parser.error("--particle-count must be a positive integer")
  return args

def main() -> None:
  args: argparse.Namespace = init_parser()

  model = joblib.load(args.model)
  print(f"Model: {args.model}")

  content = Path(args.input).read_text(encoding="utf-8")
  inputdeck = parse_inputdeck(content)

  deck_vector = define_deck(inputdeck)
  deck_vector.append(float(args.particle_count))

  X_new = np.asarray([deck_vector], dtype=float)

  expected_schema = (*DECK_FEATURE_NAMES, "particle_count")
  saved_schema = getattr(model, "deck_feature_names_", None)
  if ((saved_schema is not None and tuple(saved_schema) != expected_schema)
      or (saved_schema is None and getattr(model, "n_features_in_", None) != X_new.shape[1])):
    raise ValueError(
      "Model uses an incompatible deck feature schema. Retrain it with "
      "modeling-training before predicting with the expanded DeckConcepts."
    )

  predicted_values = model.predict(X_new)[0]

  TARGET_NAMES = ("max", "mean", "min")
  prediction = dict(zip(TARGET_NAMES, predicted_values))
  print(prediction)
