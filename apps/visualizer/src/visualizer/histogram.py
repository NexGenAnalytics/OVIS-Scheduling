from common.metrics import RESOURCE_METRICS
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from visualizer.metrics import max_value, filter_metric

def display_histogram(metric: str):

  path: Path = Path("data")

  # TODO, its like load data in predicter, maybe create a common
  # load data
  simulations = []
  path_manifest = path / "lammps_results" / "summary.csv"
  path_inputdecks = path / "lammps_input_decks"
  path_profiles = path / "lammps_results"

  with path_manifest.open(newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    rows = list(reader)

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

      cpu = filter_metric(rows, RESOURCE_METRICS["cpu"])
      memory = filter_metric(rows, RESOURCE_METRICS["memory"])

      cpu_values = [float(row["value"]) for row in cpu]
      memory_values = [float(row["value"]) for row in memory]

      max_cpu = max_value(cpu_values)
      max_memory = max_value(memory_values) # kB
      max_memory_MB = max_memory / 1000 # MB

      simulation = {
        "id": int(job_id),
        "name": problem,
        "maxcpu": int(max_cpu), # cores
        "maxmemory": int(max_memory_MB) # MB
      }
      simulations.append(simulation)

  if metric == "memory":
    values = np.array([sim["maxmemory"] for sim in simulations])
    units = "MB"
  else:
    values = np.array([sim["maxcpu"] for sim in simulations])
    units = "cores"

  val_min = values.min()
  val_max = values.max()
  print(f"Min memory: {val_min:.2f} {units}")
  print(f"Max memory: {val_max:.2f} {units}")

  bins = np.arange(
    math.floor(val_min / 10) * 10,
    math.ceil(val_max / 10) * 10 + 1,
    10
  )
  counts, edges = np.histogram(values, bins=bins)

  median = float(np.median(values))
  p95 = float(np.percentile(values, 95))

  # ===========> drawing
  fig, axes = plt.subplots(figsize=(10, 6))

  axes.hist(
    values, bins=bins,
    edgecolor="black", alpha=0.7, label="Simulation runs"
  )

  axes.axvline(
    median, color="orange",
    linestyle="--", linewidth=2,
    label=f"Median: {median:.1f} {units}"
  )

  axes.axvline(
    p95, color="red",
    linestyle="--", linewidth=2,
    label=f"P95: {p95:.1f} {units}"
  )

  axes.set_xlim(edges[0], edges[-1])
  axes.set_ylim(0, max(counts.max() * 1.12, 1))

  axes.set_ylabel("Number of simulation runs", labelpad=10)
  axes.set_xlabel(f"Maximum recorded active {metric} ({units})", labelpad=10)

  axes.legend()
  axes.grid(axis="y", alpha=0.3)

  plt.tight_layout()
  plt.show()
