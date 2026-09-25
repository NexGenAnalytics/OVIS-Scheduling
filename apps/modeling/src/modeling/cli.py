import argparse
import csv
import joblib
import numpy as np
import pandas as pd

from dataclasses import dataclass
from hashing.utils import create_hash
from pathlib import Path
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from typing import List

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Every CPU and memory profile is interpolated to this number of points.
# This is necessary because a machine-learning model needs fixed-size targets.
PROFILE_POINTS = 128

MODEL_DIRECTORY = Path("data/output/models")

CPU_MODEL_PATH = MODEL_DIRECTORY / "cpu_profile_model.joblib"
MEMORY_MODEL_PATH = MODEL_DIRECTORY / "memory_profile_model.joblib"

# ------------------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class JobSummary:
  id: int
  build: str
  problem: str
  rows: int
  status: str

@dataclass(frozen=True)
class Simulation:
  id: int
  pathinputdeck: str
  pathrunprofile: str

@dataclass(frozen=True)
class SimulationContent:
  id: int
  inputdeckhash: int
  cpuprofile: pd.DataFrame
  memoryprofile: pd.DataFrame

# ------------------------------------------------------------------------------
# Manifest loading
# ------------------------------------------------------------------------------

def read_job_summaries(path: Path) -> List[JobSummary]:
  with path.open(newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)

    return [
      JobSummary(
        id=int(row["job_id"]),
        build=row["build"], problem=row["problem"],
        rows=int(row["rows"]), status=row["status"],
      )
      for row in reader
    ]

# ------------------------------------------------------------------------------
# Job validation
# ------------------------------------------------------------------------------

def get_valid_jobs(jobs: List[JobSummary]) -> List[Simulation]:
  # Job is "valid" if files exists
  valid_jobs = []

  for job in jobs:
    problem = job.problem

    path_profiles = (
      Path("data/ldms") / job.build / job.problem / str(job.id)
      / "ldms_metrics.csv"
    )
    path_input_deck = (Path("data/input-decks/lammps") / f"in.{job.problem}")

    if path_profiles.is_file() and path_input_deck.is_file():
      valid_jobs.append(
        Simulation(
          id=job.id,
          pathinputdeck=path_input_deck,
          pathrunprofile=path_profiles,
        )
      )
    else:
      print(f"Job {job.id}: required files are missing")

  return valid_jobs

# ------------------------------------------------------------------------------
# Profile loading
# ------------------------------------------------------------------------------

def load_jobs(jobs: List[Simulation]) -> List[SimulationContent]:
  loaded_jobs = []

  for job in jobs:
    path_input_deck = Path(job.pathinputdeck)
    path_run_profile = Path(job.pathrunprofile)

    _, input_deck_hash = create_hash(
      str(path_input_deck),
      "lammps_in_files",
    )

    metrics = pd.read_csv(
      path_run_profile,
      dtype={
        "job_id": "int64",
        "component_id": "int64",
        "metric": "string",
        "unit": "string",
      },
    )

    required_columns = {
      "timestamp",
      "time_rel_s",
      "job_id",
      "component_id",
      "metric",
      "unit",
      "value",
    }
    missing_columns = required_columns.difference(metrics.columns)

    if missing_columns:
      raise ValueError(
        f"{path_run_profile}: missing columns "
        f"{sorted(missing_columns)}"
      )

    cpu_profile = (
      metrics.loc[metrics["metric"] == "cpu_cores_used"]
      .sort_values(["component_id", "time_rel_s"])
      .reset_index(drop=True)
    )

    memory_profile = (
      metrics.loc[metrics["metric"] == "mem_active_kb"]
      .sort_values(["component_id", "time_rel_s"])
      .reset_index(drop=True)
    )

    if cpu_profile.empty:
      raise ValueError(f"{path_run_profile}: no CPU profile was found")

    if memory_profile.empty:
      raise ValueError(f"{path_run_profile}: no memory profile was found")

    loaded_jobs.append(
      SimulationContent(
        id=job.id,
        inputdeckhash=input_deck_hash,
        cpuprofile=cpu_profile,
        memoryprofile=memory_profile,
      )
    )

  return loaded_jobs

# ------------------------------------------------------------------------------
# Model creation
# ------------------------------------------------------------------------------

def create_model() -> TransformedTargetRegressor:
  regressor = make_pipeline(
    StandardScaler(),
    MLPRegressor(
      hidden_layer_sizes=(64, 32),
      solver="lbfgs",
      max_iter=2000,
      random_state=42,
    ),
  )

  return TransformedTargetRegressor(
    regressor=regressor,
    transformer=StandardScaler(),
  )

# ------------------------------------------------------------------------------
# SimHash preparation
# ------------------------------------------------------------------------------

def hash_to_features(simhash: int, number_of_bits: int = 64) -> np.ndarray:
  bit_mask = (1 << number_of_bits) - 1
  simhash = int(simhash) & bit_mask

  return np.array(
    [(simhash >> bit) & 1 for bit in range(number_of_bits)],
    dtype=np.float64,
  )

# ------------------------------------------------------------------------------
# Profile preparation
# ------------------------------------------------------------------------------

def profile_to_target(
  profile: pd.DataFrame,
  number_of_points: int = PROFILE_POINTS,
) -> np.ndarray:
  """
  Convert a variable-length resource profile into a fixed-size target.

  First, values from all components at the same timestamp are summed.
  The combined profile is then interpolated over normalized execution
  time, from 0.0 to 1.0.

  The returned array always contains `number_of_points` values.
  """

  prepared_profile = profile.loc[:, ["time_rel_s", "value"]].copy()

  prepared_profile["time_rel_s"] = pd.to_numeric(
    prepared_profile["time_rel_s"],
    errors="coerce",
  )

  prepared_profile["value"] = pd.to_numeric(
    prepared_profile["value"],
    errors="coerce",
  )

  # Remove measurements that cannot be used for interpolation.
  prepared_profile = prepared_profile.replace(
    [np.inf, -np.inf],
    np.nan,
  ).dropna()

  if prepared_profile.empty:
    raise ValueError("Cannot create a target from an empty profile")

  # Obtain one total resource value for each timestamp.
  prepared_profile = (
    prepared_profile
    .groupby("time_rel_s", as_index=False)["value"]
    .sum()
    .sort_values("time_rel_s")
    .reset_index(drop=True)
  )

  measured_times = prepared_profile["time_rel_s"].to_numpy(dtype=np.float64)

  measured_values = prepared_profile["value"].to_numpy(dtype=np.float64)

  # A profile with only one timestamp cannot be interpolated.
  # In that case, use the same value for the entire profile.
  if (
    len(measured_values) == 1
    or measured_times[-1] == measured_times[0]
  ):
    return np.full(
      number_of_points,
      measured_values[0],
      dtype=np.float64,
    )

  # Convert the real execution time to a range between 0.0 and 1.0.
  normalized_measured_times = (
    measured_times - measured_times[0]
  ) / (
    measured_times[-1] - measured_times[0]
  )

  normalized_target_times = np.linspace(0.0, 1.0, number_of_points)

  return np.interp(
    normalized_target_times,
    normalized_measured_times,
    measured_values,
  )

# ------------------------------------------------------------------------------
# Model training
# ------------------------------------------------------------------------------

def train_models(
  simulations: List[SimulationContent]
) -> tuple[TransformedTargetRegressor, TransformedTargetRegressor]:
  """
  Train one model for CPU profiles and one for memory profiles.
  """

  x = len(simulations)
  if x < 2:
    raise ValueError("At least two simulations are required for training")

  # Each SimHash becomes HASH_BITS binary input features.
  inputs = np.vstack([
    hash_to_features(simulation.inputdeckhash)
    for simulation in simulations
  ])

  # Each CPU profile becomes PROFILE_POINTS output values.
  cpu_targets = np.vstack([
    profile_to_target(simulation.cpuprofile)
    for simulation in simulations
  ])

  # Each memory profile becomes PROFILE_POINTS output values.
  memory_targets = np.vstack([
    profile_to_target(simulation.memoryprofile)
    for simulation in simulations
  ])

  cpu_model = create_model()
  memory_model = create_model()

  print(f"Training CPU model with {x} simulations...")
  cpu_model.fit(inputs, cpu_targets)

  print(f"Training memory model with {x} simulations...")
  memory_model.fit(inputs, memory_targets)

  return cpu_model, memory_model

# ------------------------------------------------------------------------------
# Model prediction
# ------------------------------------------------------------------------------

def predict_profiles(
  input_deck_hash: int,
  cpu_model: TransformedTargetRegressor,
  memory_model: TransformedTargetRegressor,
) -> pd.DataFrame:
  """
  Predict CPU and memory profiles from an input-deck SimHash.

  The returned time axis is normalized:
    0.0 means the beginning of the simulation.
    1.0 means the end of the simulation.
  """

  # Convert the 32-bit SimHash to the same binary feature representation
  # that was used during training.
  model_input = hash_to_features(input_deck_hash).reshape(1, -1)

  predicted_cpu = cpu_model.predict(model_input)[0]
  predicted_memory = memory_model.predict(model_input)[0]

  # Regression models can produce small negative values even though CPU
  # and memory usage cannot physically be negative.
  predicted_cpu = np.maximum(predicted_cpu, 0.0)
  predicted_memory = np.maximum(predicted_memory, 0.0)

  normalized_time = np.linspace(0.0, 1.0, PROFILE_POINTS)

  return pd.DataFrame({
    "normalized_time": normalized_time,
    "cpu_cores_used": predicted_cpu,
    "mem_active_kb": predicted_memory,
  })

# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_models(
  cpu_model: TransformedTargetRegressor,
  memory_model: TransformedTargetRegressor,
) -> None:
  """Save both trained models to disk."""

  MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

  joblib.dump(cpu_model, CPU_MODEL_PATH)
  joblib.dump(memory_model, MEMORY_MODEL_PATH)

  print(f"CPU model saved to {CPU_MODEL_PATH}")
  print(f"Memory model saved to {MEMORY_MODEL_PATH}")

def load_models() -> tuple[
  TransformedTargetRegressor,
  TransformedTargetRegressor
]:
  """Load previously trained CPU and memory models."""

  cpu_model = joblib.load(CPU_MODEL_PATH)
  memory_model = joblib.load(MEMORY_MODEL_PATH)

  return cpu_model, memory_model

# ------------------------------------------------------------------------------
# CLI operations
# ------------------------------------------------------------------------------

def run_training(path_manifest: Path) -> None:
  """
  Load simulations from a manifest, train both models, and save them.
  """

  # LOADING MANIFEST
  raw_jobs: List[JobSummary] = read_job_summaries(Path(path_manifest))
  print(f"Manifest jobs: {len(raw_jobs)}")
  ok_jobs: List[JobSummary] = [job for job in raw_jobs if job.status == "ok"]
  print(f"Successful jobs: {len(ok_jobs)}")

  # VALIDATING MANIFEST
  valid_jobs: List[Simulation] = get_valid_jobs(ok_jobs)
  print(f"Valid jobs: {len(valid_jobs)}")

  # LOADING DATA
  simulations: List[SimulationContent] = load_jobs(valid_jobs)
  print(f"Loaded simulations: {len(simulations)}")

  # TRAINING MODELS
  cpu_model, memory_model = train_models(simulations)
  print("Training completed")

  # SAVE MODELS
  save_models(cpu_model, memory_model)
  print("Saving completed")

def run_prediction(input_deck_path: Path) -> None:
  """
  Load the trained models and predict the resource profile of one input deck.
  """

  if not input_deck_path.is_file():
    raise FileNotFoundError(
      f"Input deck does not exist: {input_deck_path}"
    )

  if not CPU_MODEL_PATH.is_file():
    raise FileNotFoundError(
      f"CPU model does not exist: {CPU_MODEL_PATH}. "
      "Run modeling --train first."
    )

  if not MEMORY_MODEL_PATH.is_file():
    raise FileNotFoundError(
      f"Memory model does not exist: {MEMORY_MODEL_PATH}. "
      "Run modeling --train first."
    )

  print(f"Loading input deck: {input_deck_path}")

  # Use the same LAMMPS normalizer used during model training.
  _, input_deck_hash = create_hash(
    str(input_deck_path),
    "lammps_in_files",
  )

  print(f"Input-deck SimHash: {input_deck_hash}")

  cpu_model, memory_model = load_models()

  prediction = predict_profiles(
    input_deck_hash=input_deck_hash,
    cpu_model=cpu_model,
    memory_model=memory_model,
  )

  # Save the prediction next to the trained-model output directory.
  prediction_directory = Path("data/output/predictions")
  prediction_directory.mkdir(parents=True, exist_ok=True)

  prediction_path = (
    prediction_directory
    / f"{input_deck_path.stem}_predicted_profile.csv"
  )

  prediction.to_csv(prediction_path, index=False)

  print(prediction.to_string(index=False))
  print(f"Prediction saved to {prediction_path}")

# ------------------------------------------------------------------------------
# Main program
# ------------------------------------------------------------------------------

def main() -> None:
  print("M, start")

  parser = argparse.ArgumentParser(prog="modeling")
  commands = parser.add_mutually_exclusive_group(required=True)

  commands.add_argument(
    "--train", type=Path, metavar="MANIFEST",
    help="Train the models using an manifest CSV file",
  )

  commands.add_argument(
    "--predict", type=Path, metavar="INPUT_DECK",
    help="Predict CPU and memory profiles for an input deck",
  )

  args = parser.parse_args()

  if args.train is not None:
    run_training(args.train)
  else:
    run_prediction(args.predict)

  print("M, end")
