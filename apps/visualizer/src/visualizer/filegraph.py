from common.loading import load_one_simulation
from common.metrics import filter_metric, max_value
import csv
import matplotlib.pyplot as plt
from pathlib import Path

def display_graph(path_file: str):

  path: Path = Path("data")
  path_file = path / "lammps_results" / path_file

  # Loading
  cpu_values, memory_values, cpu_times, memory_times = (
    load_one_simulation(path_file)
  )

  # Computation
  max_cpu = max_value(cpu_values)
  max_memory = max_value(memory_values) # kB

  # Figure and plot
  fig, axes = plt.subplots(2, 1, sharex=True)

  axes[0].plot(cpu_times, cpu_values)
  axes[0].axhline(
    max_cpu, color="red", linestyle="--",
    label=f"Max: {max_cpu:.2f} cores",
  )
  axes[0].set_ylabel("CPU (cores)")
  axes[0].legend()

  axes[1].plot(memory_times, memory_values)
  axes[1].axhline(
    max_memory, color="red", linestyle="--",
    label=f"Max: {max_memory:.0f} kB",
  )
  axes[1].set_ylabel("Memory (kB)")
  axes[1].set_xlabel("Time (seconds)")
  axes[1].legend()

  plt.tight_layout()
  plt.show()
