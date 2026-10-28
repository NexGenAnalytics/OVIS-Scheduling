import argparse
from .benchmark import (display_benchmark, run_benchmark, select_best_model)
from common import (DECK_FEATURE_NAMES, Simulation, load_simulations)
from .dataset import prepare_dataset
from .evaluation import TARGET_NAMES, evaluate_predictions
import joblib
from .models import RANDOM_SEED
import numpy as np
from pathlib import Path
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split

RESOURCE_METRICS = {
  "cpu": "cpu_cores_used",
  "memory": "mem_active_kb",
}

def init_parser() -> argparse.Namespace:
  parser = argparse.ArgumentParser(prog="modeling-training")
  parser.add_argument("--simu", type=Path, required=True)
  # parser.add_argument("--feat", nargs="+", type=str, required=False)
  return parser.parse_args()

def split_set(simulations: list[Simulation]) -> tuple[
  list[Simulation],
  list[Simulation],
  list[Simulation]
]:
  """
  Split in:
  - train: 70%
  - validation: 20%
  - test: 10%
  """
  training_set, remaining_set = train_test_split(
    simulations, train_size=0.7,
    random_state=RANDOM_SEED, shuffle=True,
  )
  validation_set, testing_set = train_test_split(
    remaining_set, train_size=2/3,
    random_state=RANDOM_SEED, shuffle=True,
  )
  return training_set, validation_set, testing_set

def save_model(model: any, name: str) -> Path:
  directory = Path("output/models")
  directory.mkdir(parents=True, exist_ok=True)

  model_path = directory / f"{name}_model.joblib"
  model.deck_feature_names_ = (*DECK_FEATURE_NAMES, "particle_count")
  joblib.dump(model, model_path)

  return model_path

def display_peak_memory_check(
  expected: np.ndarray,
  predicted: np.ndarray,
) -> None:
  peak_index = TARGET_NAMES.index("max")
  actual_peaks = expected[:, peak_index]
  predicted_peaks = predicted[:, peak_index]

  print("\nPeak-memory accuracy (held-out test):")
  if actual_peaks.size == 0 or np.any(actual_peaks <= 0):
    print("Cannot assess relative accuracy: actual peaks must be positive.")
    return

  relative_errors = np.abs(predicted_peaks - actual_peaks) / actual_peaks
  within_tolerance = int(np.count_nonzero(relative_errors <= 0.20))
  total = len(actual_peaks)
  coverage = within_tolerance / total
  status = "PASS" if coverage >= 0.90 else "FAIL"

  underestimated = int(np.count_nonzero(predicted_peaks < actual_peaks))
  large_underestimates = int(np.count_nonzero(
    (predicted_peaks < actual_peaks) & (relative_errors > 0.20)
  ))

  print(f"Within ±20%: {within_tolerance}/{total} ({coverage:.1%})")
  print(f"Target ≥90% within ±20%: {status}")
  print(f"Underestimated: {underestimated}/{total} ({underestimated / total:.1%})")
  print(
    f"Underestimated by >20%: {large_underestimates}/{total} "
    f"({large_underestimates / total:.1%})"
  )

def main() -> None:
  args: argparse.Namespace = init_parser()
  simulations: list[Simulation] = load_simulations(args.simu)

  training_set, validation_set, testing_set = split_set(simulations)

  for resource, metric in RESOURCE_METRICS.items():
    print(f"\n---------------- RESOURCE: {resource} ----------------")

    X_train, y_train = prepare_dataset(training_set, metric)
    X_validation, y_validation = prepare_dataset(
      validation_set,
      metric,
    )

    benchmark_results = run_benchmark(
      X_train,
      y_train,
      X_validation,
      y_validation,
    )

    display_benchmark(benchmark_results)

    best_result = select_best_model(benchmark_results)
    print(f"Selected model: {best_result.name}")

    # Test
    X_test, y_test = prepare_dataset(testing_set, metric)
    test_predictions = best_result.model.predict(X_test)
    test_scores = evaluate_predictions(y_test, test_predictions)

    print(f"Held-out test evaluation: {best_result.name} ({len(y_test)} simulations)")
    print(f"{'TARGET':<10}{'MAE':>14}{'RMSE':>14}")
    for score in test_scores:
      print(f"{score.target:<10}{score.mae:>14.4f}{score.rmse:>14.4f}")

    display_peak_memory_check(y_test, test_predictions)

    # Save model
    model_path = save_model(best_result.model, resource)
    print(f"Saved {best_result.name} to {model_path}")

