import argparse
import csv

from pathlib import Path

from visualizer.metrics import max_value, filter_metric

def load_datas(path: Path):
  simulations: list() = []

  path_manifest = path / "lammps_results" / "summary.csv"
  path_inputdecks = path / "lammps_input_decks"
  path_profiles = path / "lammps_results"

  with path_manifest.open(newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    rows = list(reader)
    print(f"> {len(rows)} files")

    count_ok = sum(row["status"] == "ok" for row in rows)
    print(f"> {count_ok} files OK")

    for row in rows:
      status = row["status"]

      if row["status"] != "ok": continue

      job_id = row["job_id"]
      problem = row["problem"]
      filename = f"in.{problem}"

      path_inputdeck = path_inputdecks / filename
      path_profile = path_profiles / problem / job_id / "ldms_metrics.csv"

      with open(path_profile, newline="") as file:
        rows = list(csv.DictReader(file))

      cpu = filter_metric(rows, "cpu_cores_used")
      memory = filter_metric(rows, "mem_active_kb")

      cpu_values = [float(row["value"]) for row in cpu]
      memory_values = [float(row["value"]) for row in memory]

      max_cpu = max_value(cpu_values)
      max_memory = max_value(memory_values)

      # print(f"- name: {problem}, max_CPU: {max_cpu} and max_M: {max_memory}")

      simulations.append((problem, max_cpu, max_memory))

  print(f"> {len(simulations)} loaded")
  return simulations


def main() -> None:
  print("P, start")

  parser = argparse.ArgumentParser(prog="TODO")
  parser.add_argument("--train", type=Path, required=True)
  args = parser.parse_args()

  sims = load_datas(args.train)

  print("TODO, train then infer neural network")

  # move here code from https://github.com/NexGenAnalytics/OVIS-Scheduling/pull/44/changes
  # store (id job, max cpu, max memory, input decks features)
  # train NN
  # predict (max cpu, max memory)

  print("P, end")
