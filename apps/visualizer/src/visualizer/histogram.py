from common.loading import load_simulations
from common.metrics import RESOURCE_METRICS
from common.Simulation import Simulation
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from visualizer.metrics import max_value, filter_metric

def display_histogram(metric: str):
  path: Path = Path("data")
  simulations: list[Simulation] = load_simulations(path)

  # Compute data
  if metric == "memory":
    values = np.array([
      sim.maxmemory / 1000 # from kB to MB
      for sim in simulations])
    units = "MB"
  else:
    values = np.array([sim.maxcpu for sim in simulations])
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

  # Figure and plot
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
