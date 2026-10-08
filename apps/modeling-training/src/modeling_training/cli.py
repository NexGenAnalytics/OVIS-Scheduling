import argparse
from .benchmark import (display_benchmark, run_benchmark, select_best_model)
from common import (Feature, Metric, Simulation)
from common import (parse_inputdeck, parse_runprofile)
from .dataset import prepare_dataset
import csv
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

def load_simulations(path: Path) -> list[Simulation]:
  simulations: list[Simulation] = []

  path_manifest = path / "ldms" / "ldms_manifest_20261006-084316.csv"
  path_inputdecks = path / "input-decks" / "lammps"
  path_profiles = path / "ldms" / "cpu_mpi_omp"

  with path_manifest.open(newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    rows = list(reader)
    print(f"- Total {len(rows)} files")

    rc_nodes, rc_ntasks, rc_ntasks_per_node, rc_omp_num_threads = 1, 48, 48, 1
    print(f"- Reference configuration: \
      nodes={rc_nodes}, \
      ntasks={rc_ntasks}, \
      ntasks_per_node={rc_ntasks_per_node} \
      and omp_num_threads={rc_omp_num_threads}")

    for row in rows:
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

      if not (
        int(row["nodes"]) == rc_nodes
        and int(row["ntasks"]) == rc_ntasks
        and int(row["ntasks_per_node"]) == rc_ntasks_per_node
        and int(row["omp_num_threads"]) == rc_omp_num_threads
      ):
        continue

      simulation = Simulation(
        id = int(job_id),
        inputdeck = inputdeck_content,
        runprofile = runprofile_content,
        totaltime = int(total_time),
      )
      simulations.append(simulation)

      totaltimeminutes = int(simulation.totaltime / 60)
      # print(f"Load {simulation.id}, exec. time of {totaltimeminutes} minutes")

  print(f"- Load {len(simulations)} simulations")
  return simulations

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
  joblib.dump(model, model_path)

  return model_path

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

    best_result = select_best_model(benchmark_results, target="mean")
    print(f"Selected model: {best_result.name}")

    model_path = save_model(best_result.model, resource)
    print(f"Saved {best_result.name} to {model_path}")

