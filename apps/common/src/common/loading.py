from common.Simulation import Simulation
import csv
from .metrics import filter_metric, max_value, RESOURCE_METRICS
from .parsing import parse_inputdeck

def load_one_simulation(path: str):
  with open(path, newline="") as file:
    rows_file = list(csv.DictReader(file))

  cpu = filter_metric(rows_file, RESOURCE_METRICS["cpu"])
  memory = filter_metric(rows_file, RESOURCE_METRICS["memory"])

  cpu_values = [float(row["value"]) for row in cpu]
  memory_values = [float(row["value"]) for row in memory]

  cpu_times = [float(row["time_rel_s"]) for row in cpu]
  memory_times = [float(row["time_rel_s"]) for row in memory]

  return cpu_values, memory_values, cpu_times, memory_times

def load_simulations(path, debug = False):
  simulations: list[Simulation] = []

  path_manifest = path / "lammps_results" / "summary.csv"
  path_inputdecks = path / "lammps_input_decks"
  path_profiles = path / "lammps_results"

  with path_manifest.open(newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    rows = list(reader)

  if debug:
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

    cpu_values, memory_values, cpu_times, memory_times = (
      load_one_simulation(path_profile)
    )

    max_cpu = max_value(cpu_values)
    max_memory = max_value(memory_values) # kB
    total_time = max_value(cpu_times) # seconds

    simulation = Simulation(
      id = int(job_id),
      name = problem,
      maxcpu = int(max_cpu), # cores
      maxmemory = int(max_memory), # kB
      totaltime = int(total_time), # seconds
      inputdeck = inputdeckcontent,
    )
    simulations.append(simulation)

  if debug:
    print(f"> {len(simulations)} file(s) loaded")
  return simulations
