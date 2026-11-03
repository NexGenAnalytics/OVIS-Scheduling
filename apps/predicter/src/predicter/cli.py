import argparse
from .benchmark import display_benchmark, ModelBenchmark, run_benchmark
from common.loading import load_simulations
from common.metrics import RESOURCE_METRICS
from common.Simulation import Simulation
from .decker import numeric_features
from .models import RANDOM_SEED
from pathlib import Path
from .prepare import prepare_dataset
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

    # TODO: best_result = select_best_model(results)
    # TODO: print(f"Selected model: {best_result.name}")

    # Test
    X_test, y_test = prepare_dataset(testing_set, metric, columns)
    # TODO: test_predictions = best_result.model.predict(X_test)
    # TODO: test_scores = evaluate_predictions(y_test, test_predictions)

    # Save model
    # TODO: model_path = save_model(best_result.model, resource)
    # TODO: print(f"Saved {best_result.name} to {model_path}")

    # TODO: Draw graph x = y


def define_predict(path: str):
  # TODO
  print("TODO, incoming")


def main() -> None:
  print("P, start")

  args: argparse.Namespace = init_parser()

  match vars(args):
    case {"train": str(path_train)}:
      define_train(Path(path_train))
    case {"predict": str(path_predict)}:
      define_predict(path_predict)

  print("P, end")
