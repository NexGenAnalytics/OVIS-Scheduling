import argparse
from common import (Feature, Metric, Simulation)
from common import (parse_inputdeck, parse_runprofile)
from common import find_variable
import csv
import joblib
import numpy as np
from pathlib import Path
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (
  mean_absolute_error, r2_score, root_mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42

def init_parser() -> argparse.Namespace:
  parser = argparse.ArgumentParser(prog="modeling-training")
  parser.add_argument("--simu", type=Path, required=True)
  # parser.add_argument("--feat", nargs="+", type=str, required=False)
  return parser.parse_args()

def load_simulations(path: Path) -> list[Simulation]:
  simulations: list[Simulation] = []

  path_manifest = path / "ldms" / "ldms_manifest_20260921.csv"
  path_inputdecks = path / "input-decks" / "lammps"
  path_profiles = path / "ldms" / "cpu_mpi_omp"

  with path_manifest.open(newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)

    for row in reader:
      status = row["status"]

      if row["status"] != "ok": continue

      job_id = row["job_id"]
      problem = row["problem"]
      filename = f"in.{problem}"

      path_inputdeck = path_inputdecks / filename
      path_profile = path_profiles / problem / job_id / "ldms_metrics.csv"

      inputdeck_str: str = path_inputdeck.read_text(encoding="utf-8")
      runprofile_str: str = path_profile.read_text(encoding="utf-8")

      inputdeck_content: list[Feature] = parse_inputdeck(inputdeck_str)
      runprofile_content, total_time = parse_runprofile(runprofile_str)

      simulation = Simulation(
        id = int(job_id),
        inputdeck = inputdeck_content,
        runprofile = runprofile_content,
        totaltime = int(total_time),
      )
      simulations.append(simulation)

      totaltimeminutes = int(simulation.totaltime / 60)
      print(f"Load {simulation.id}, exec. time of {totaltimeminutes} minutes")

  return simulations

def create_model() -> TransformedTargetRegressor:
  mlp_regressor = MLPRegressor(
    hidden_layer_sizes=(64, 32), solver="lbfgs", max_iter=2000,
    random_state=RANDOM_SEED
  )
  model = TransformedTargetRegressor(
    regressor=make_pipeline(StandardScaler(), mlp_regressor),
    transformer=StandardScaler(),
  )
  return model

def prepare_dataset(
  simulations: list[Simulation],
  metric: str
) -> (np.ndarray, np.ndarray):
  # X is the numeric description of each LAMMPS input deck.
  # X.shape == (number_of_simulations, number_of_input_features)
  X: list[list[float]] = []

  # y is the observed [column] (example: cpu_cores_used) profile corresponding
  # to that deck.
  # y.shape == (number_of_simulations, 3)
  y: list[list[float]] = []

  for simulation in simulations:
    # TODO: below, use features??
    deck_vector = [
      find_variable(simulation.inputdeck, "nx"),
      #find_variable(inputdeck, "rho"),
      #find_variable(inputdeck, "temp"),
      #find_variable(inputdeck, "rc"),
      find_variable(simulation.inputdeck, "nsteps"),
    ]

    # Only the requested metric
    values = np.asarray(
      [
        point.value
        for point in simulation.runprofile
        if point.metric == metric
      ],
      dtype=float
    )

    # Checks
    if values.size == 0:
      raise ValueError(
        f"Simulation {simulation.id} has no values for metric {metric!r}"
      )

    if not np.all(np.isfinite(values)):
      raise ValueError(
        f"Simulation {simulation.id} contains invalid values for {metric!r}"
      )

    # Targets in y
    target = [
      float(np.max(values)),
      float(np.mean(values)),
      float(np.min(values)),
    ]

    X.append(deck_vector)
    y.append(target)

  return (np.asarray(X, dtype=float), np.asarray(y, dtype=float))

def train_model(
  simulations: list[Simulation],
  metric: str
) -> TransformedTargetRegressor:
  if not simulations:
    raise ValueError("No simulations available for training")

  X, y = prepare_dataset(simulations, metric)

  model = create_model()
  model.fit(X, y)

  return model

def save_model(model: TransformedTargetRegressor, name: str) -> bool:
  directory = Path("output/models")
  directory.mkdir(parents=True, exist_ok=True)

  filename = f"{name}_model.joblib"
  modelpath = directory / filename
  joblib.dump(model, modelpath)

  return True

def split_set(
  simulations: list[Simulation]
) -> (list[Simulation], list[Simulation], list[Simulation]):
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

def evaluate_model(
  model: TransformedTargetRegressor,
  simulations: list[Simulation],
  metric: str
) -> (str, float):
  X, expected = prepare_dataset(simulations, metric)
  predicted = model.predict(X)

  # Mean Absolute Error
  mae = mean_absolute_error(expected, predicted)

  # Root Mean Squared Error
  rmse = root_mean_squared_error(expected, predicted)

  # Coefficient of determination
  # Closer to 1 is better; 0 means approx. no better than predicting the mean.
  #r2 = r2_score(expected, predicted, multioutput="variance_weighted")

  return {"mae": float(mae), "rmse": float(rmse)} #, "r2": float(r2)}

def main() -> None:
  args: argparse.Namespace = init_parser()

  print(f"-------------------- LOAD SIMULATIONS --------------------")
  simulations: list[Simulation] = load_simulations(args.simu)

  # split
  training_set, validation_set, testing_set = split_set(simulations)

  METRICS = { "cpu": "cpu_cores_used", "memory": "mem_active_kb" }
  for resource, metric in METRICS.items():
    print(f"-------------------- RESOURCE: {resource} --------------------")
    print("Model development:")

    cpu_model: TransformedTargetRegressor = train_model(training_set, metric)
    print("- train OK")

    valid_scores = evaluate_model(cpu_model, validation_set, metric)
    print("- validation:", valid_scores)

    # Adjust architecture/hyperparameters using validation results only.
    # Once all model choices are final, evaluate the test set exactly once.

    print("Model evaluation:")
    test_scores = evaluate_model(cpu_model, testing_set, metric)
    print("- test:", test_scores)

    print("Save model:")
    saved = save_model(cpu_model, resource)
    print("- OK")
