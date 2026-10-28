import argparse
import csv

import matplotlib.pyplot as plt

from visualizer.metrics import max_value, filter_metric

def main() -> None:
  print("V, start")

  parser = argparse.ArgumentParser(description="Plot CPU and memory usage.")
  parser.add_argument("--file", required=True, metavar="PATH")
  args = parser.parse_args()

  with open(args.file, newline="") as file:
    rows = list(csv.DictReader(file))

  cpu = filter_metric(rows, "cpu_cores_used")
  memory = filter_metric(rows, "mem_active_kb")

  cpu_values = [float(row["value"]) for row in cpu]
  memory_values = [float(row["value"]) for row in memory]

  max_cpu = max_value(cpu_values)
  max_memory = max_value(memory_values)

  fig, axes = plt.subplots(2, 1, sharex=True)

  axes[0].plot(
    [float(row["time_rel_s"]) for row in cpu],
    cpu_values,
  )
  axes[0].axhline(
    max_cpu, color="red", linestyle="--",
    label=f"Max: {max_cpu:.2f} cores",
  )
  axes[0].set_ylabel("CPU (cores)")
  axes[0].legend()

  axes[1].plot(
    [float(row["time_rel_s"]) for row in memory],
    memory_values,
  )
  axes[1].axhline(
    max_memory, color="red", linestyle="--",
    label=f"Max: {max_memory:.0f} kB",
  )
  axes[1].set_ylabel("Memory (kB)")
  axes[1].set_xlabel("Time (seconds)")
  axes[1].legend()

  plt.tight_layout()
  plt.show()

  print("V, end")
