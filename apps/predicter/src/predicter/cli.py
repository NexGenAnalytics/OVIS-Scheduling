import argparse
from .benchmark import display_benchmark, ModelBenchmark, run_benchmark
from common.loading import load_simulations
from common.metrics import RESOURCE_METRICS
from common.Simulation import Simulation
from .decker import numeric_features
from .evaluation import evaluate_predictions
import matplotlib.pyplot as plt
from .models import RANDOM_SEED
from pathlib import Path
from .prediction import define_predict
from .prepare import prepare_dataset
from .save import save_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def init_parser() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="TODO")

  commands = parser.add_mutually_exclusive_group(required=True)
  commands.add_argument("--train", metavar="Path")
  commands.add_argument("--predict", metavar="Path")

  args = parser.parse_args()
  return args

def split_set(
  simulations: list[Simulation]
) -> tuple[list[Simulation], list[Simulation], list[Simulation]]:
  """
  Train / Validation / Test => 70% / 20% / 10%
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

def define_train(path: Path):
  sims: list[Simulation] = load_simulations(path, True) # debug is True

  print("Split into:")
  training_set, validation_set, testing_set = split_set(sims)
  print(f"> {len(training_set)} file(s) for training")
  print(f"> {len(validation_set)} file(s) for validation")
  print(f"> {len(testing_set)} file(s) for testing\n")

  for resource, metric in RESOURCE_METRICS.items():
    print(f"-> RESOURCE: {resource}")

    # discover the columns automatically
    columns = sorted({
      name for simulation in training_set
      for name in numeric_features(simulation.inputdeck)
    })
    print(f"> {len(columns)} column(s) discovered: {columns[:3]} ...")

    X_train, y_train = prepare_dataset(training_set, metric, columns)
    print(f"> {len(X_train[0])} column(s) total (numeric + present)")
    X_validation, y_validation = prepare_dataset(
      validation_set, metric, columns)

    results: list[ModelBenchmark] = run_benchmark(
      X_train, y_train, X_validation, y_validation)
    display_benchmark(results)

    # Select the best model using validation results.
    best_result = min(results, key=lambda result: result.scores[0].rmse)
    print(f"> Selected model (using best validation RMSE): {best_result.name}")

    # Save best model.
    model_path = save_model(best_result, columns, resource)
    print(f"> Saved {best_result.name} to {model_path}")

    # Predict on the test set.
    X_test, y_test = prepare_dataset(testing_set, metric, columns)
    test_predictions = best_result.model.predict(X_test)

    # Memory: kB -> MB. CPU stays in cores.
    scale = 1000 if resource == "memory" else 1
    units = "MB" if resource == "memory" else "cores"

    real = y_test / scale
    predicted = test_predictions / scale

    score = evaluate_predictions(real, predicted)[0]
    r2 = r2_score(real, predicted)

    print(f"Test MAE: {score.mae:.2f} {units}")
    print(f"Test RMSE: {score.rmse:.2f} {units}")
    print(f"Test R²: {r2:.3f}")

    # Use identical limits for both axes.
    lower = min(real.min(), predicted.min())
    upper = max(real.max(), predicted.max())
    padding = max(upper - lower, 1) * 0.05
    limits = [lower - padding, upper + padding]

    # Figure and plot
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(real, predicted, alpha=0.7, label="Test simulations")
    ax.plot(
      limits, limits,
      "--", color="orange", label="Perfect prediction: x = y"
    )

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(f"Real maximum {resource} ({units})")
    ax.set_ylabel(f"Predicted maximum {resource} ({units})")
    ax.set_title(
      f"{best_result.name} — {resource}\n"
      f"MAE: {score.mae:.2f} {units} | "
      f"RMSE: {score.rmse:.2f} {units} | R²: {r2:.3f}"
    )

    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()

    fig.tight_layout()
    plt.show()


def main() -> None:
  print("P, start")

  args: argparse.Namespace = init_parser()

  match vars(args):
    case {"train": str(path_train)}:
      define_train(Path(path_train))
    case {"predict": str(path_predict)}:
      define_predict(path_predict)

  print("P, end")
