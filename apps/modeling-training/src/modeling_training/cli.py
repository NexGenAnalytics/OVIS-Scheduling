import argparse
from common import (Feature, Metric, Simulation)
from common import (parse_inputdeck, parse_runprofile)
from common import find_variable
import csv
import joblib
import numpy as np
from pathlib import Path
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def init_parser() -> argparse.Namespace:
  parser = argparse.ArgumentParser(prog="modeling-training")
  parser.add_argument("--simu", type=Path, required=True)
  parser.add_argument("--deta", type=int, required=True)
  parser.add_argument("--feat", nargs="+", type=str, required=False)
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
    hidden_layer_sizes=(64, 32), solver="lbfgs", max_iter=2000, random_state=42
  )
  model = TransformedTargetRegressor(
    regressor=make_pipeline(StandardScaler(), mlp_regressor),
    transformer=StandardScaler(),
  )
  return model

def train_model(
  simulations: list[Simulation],
  details: int,
  features: list[str] | None,
  metric: str
) -> TransformedTargetRegressor:
  # Pre-checks
  if not simulations:
    raise ValueError("No simulations available for training")

  if details < 2:
    raise ValueError("details must be at least 2")

  # X is the numeric description of each LAMMPS input deck.
  # X.shape == (number_of_simulations, number_of_input_features)
  X: list[list[float]] = []

  # y is the observed [column] (example: cpu_cores_used) profile corresponding
  # to that deck.
  # y.shape == (number_of_simulations, details)
  y: list[np.ndarray] = []

  for simulation in simulations:
    # Fixed-length description extracted from the input deck
    inputdeck = simulation.inputdeck
    deck_vector = [
      find_variable(inputdeck, "nx"),
      #find_variable(inputdeck, "rho"),
      #find_variable(inputdeck, "temp"),
      #find_variable(inputdeck, "rc"),
      find_variable(inputdeck, "nsteps"),
    ]

    # Only the requested metric
    metrics = [
      point
      for point in simulation.runprofile
      if point.metric == metric
    ]

    # Ensure chronological order
    metrics.sort(key=lambda point: point.time_rel_s)

    times = np.asarray([point.time_rel_s for point in metrics], dtype=float)
    values = np.asarray([point.value for point in metrics], dtype=float)

    # Exactly `details` positions for every simulation
    requested_times = np.linspace(times[0], times[-1], details)

    resampled_values = np.interp(requested_times, times, values)

    X.append(deck_vector)
    y.append(resampled_values)

  X = np.asarray(X, dtype=float)
  y = np.asarray(y, dtype=float)

  print("X shape:", X.shape)
  print("y shape:", y.shape)
  print("First X row:", X[0])
  print("First y row:", y[0])

  model = create_model()
  model.fit(X, y)

  return model

def save_model(model: TransformedTargetRegressor, name: str) -> bool:
  directory = Path("output/models")
  directory.mkdir(parents=True, exist_ok=True)

  filename = f"{name}_model.joblib"
  modelpath = directory / filename
  joblib.dump(model, modelpath)
  print(f"Save {name} ok")

  return True

def main() -> None:
  args: argparse.Namespace = init_parser()

  simulations: list[Simulation] = load_simulations(args.simu)

  cpu_model: TransformedTargetRegressor = train_model(
    simulations, args.deta, args.feat, "cpu_cores_used"
  )
  saved = save_model(cpu_model, "cpu")

  mem_model: TransformedTargetRegressor = train_model(
    simulations, args.deta, args.feat, "mem_active_kb"
  )
  saved = save_model(mem_model, "mem")
