import argparse
import csv
import numpy as np

from pathlib import Path

from .benchmark import run_benchmark

from .Feature import Feature
from .models import RANDOM_SEED
from .parsing import parse_inputdeck
from .prepare import prepare_dataset
from .Simulation import Simulation

from visualizer.metrics import max_value, filter_metric

from sklearn.model_selection import train_test_split

RESOURCE_METRICS = {
  "cpu": "cpu_cores_used",
  "memory": "mem_active_kb",
}

def init_parser() -> argparse.Namespace:
  parser = argparse.ArgumentParser(prog="TODO")
  parser.add_argument("--train", type=Path, required=True)
  args = parser.parse_args()
  return args

def load_datas(path: Path) -> list[Simulation]:
  simulations: list[Simulation] = []

  path_manifest = path / "lammps_results" / "summary.csv"
  path_inputdecks = path / "lammps_input_decks"
  path_profiles = path / "lammps_results"

  with path_manifest.open(newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    rows = list(reader)
    print(f"> {len(rows)} file(s)")

    count_ok = sum(row["status"] == "ok" for row in rows)
    print(f"> {count_ok} file(s) OK")

    for row in rows:
      status = row["status"]

      if row["status"] != "ok": continue

      job_id = row["job_id"]
      problem = row["problem"]
      filename = f"in.{problem}"

      path_inputdeck = path_inputdecks / filename
      path_profile = path_profiles / problem / job_id / "ldms_metrics.csv"

      inputdeck_str: str = path_inputdeck.read_text(encoding="utf-8")
      inputdeckcontent: list[Feature] = parse_inputdeck(inputdeck_str)

      with open(path_profile, newline="") as file:
        rows = list(csv.DictReader(file))

      cpu = filter_metric(rows, RESOURCE_METRICS["cpu"])
      memory = filter_metric(rows, RESOURCE_METRICS["memory"])

      cpu_values = [float(row["value"]) for row in cpu]
      memory_values = [float(row["value"]) for row in memory]
      time_values = [float(row["time_rel_s"]) for row in cpu]

      max_cpu = max_value(cpu_values)
      max_memory = max_value(memory_values)
      total_time = max_value(time_values)

      simulation = Simulation(
        id = int(job_id),
        name = problem,
        maxcpu = int(max_cpu), # cores
        maxmemory = int(max_memory), # kB
        totaltime = int(total_time), # seconds
        inputdeck = inputdeckcontent,
      )
      # print(simulation)
      simulations.append(simulation)

  print(f"> {len(simulations)} file(s) loaded")
  return simulations

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
  sims = load_datas(args.train)
  training_set, validation_set, testing_set = split_set(sims)

  for resource, metric in RESOURCE_METRICS.items():
    print(f"\n---------------- RESOURCE: {resource} ----------------")

    X_train, y_train = prepare_dataset(training_set, metric)
    X_validation, y_validation = prepare_dataset(validation_set, metric)

    run_benchmark(X_train, y_train, X_validation, y_validation)


  # move here code from https://github.com/NexGenAnalytics/OVIS-Scheduling/pull/44/changes
  # store (id job, max cpu, max memory, input decks features)
  # train NN
  # predict (max cpu, max memory)

  print("P, end")
