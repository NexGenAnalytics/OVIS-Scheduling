import argparse
from .benchmark import display_benchmark, ModelBenchmark, run_benchmark
from common.loading import load_simulations
from common.metrics import RESOURCE_METRICS
from common.Simulation import Simulation
from .decker import numeric_features
from .models import RANDOM_SEED
import numpy as np
from pathlib import Path
from .prepare import prepare_dataset
from sklearn.model_selection import train_test_split

def init_parser() -> argparse.Namespace:
  parser = argparse.ArgumentParser(prog="TODO")
  parser.add_argument("--train", type=Path, required=True)
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

def main() -> None:
  print("P, start")

  args: argparse.Namespace = init_parser()
  sims: list[Simulation] = load_simulations(args.train, True) # debug is True

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
    X_validation, y_validation = prepare_dataset(validation_set, metric, columns)

    run_benchmark(X_train, y_train, X_validation, y_validation)


  # move here code from https://github.com/NexGenAnalytics/OVIS-Scheduling/pull/44/changes
  # store (id job, max cpu, max memory, input decks features)
  # train NN
  # predict (max cpu, max memory)

  print("P, end")
