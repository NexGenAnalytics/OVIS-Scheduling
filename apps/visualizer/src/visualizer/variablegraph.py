from common.loading import load_simulations
from common.Simulation import Simulation
import matplotlib.pyplot as plt
from pathlib import Path

def display_vs_graph(variable_name: str):
  path: Path = Path("data")
  simulations: list[Simulation] = load_simulations(path)

  x_values = []
  memory_values = []

  for simulation in simulations:
    """
    inputdeck example => [
      Feature(command='nx', arguments='24'),
      Feature(command='temp', arguments='1000.0'),
      Feature(command='nsteps', arguments='300000')
    ]
    """
    variables = {
      feature.command: feature.arguments
      for feature in simulation.inputdeck
    }
    # variable example => {'nx': '24', 'temp': '1000.0', 'nsteps': '300000'}

    value = variables.get(variable_name)

    if isinstance(value, list):
      value = value[0]

    try:
      value = float(value)
    except (TypeError, ValueError):
      continue  # Variable is missing or not numeric.

    x_values.append(value)
    memory_values.append(simulation.maxmemory / 1000)  # kB -> MB

  if not x_values:
    print(f"No numeric values found for '{variable_name}'")
    return

  # Figure and plot
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.scatter(x_values, memory_values, alpha=0.6)

  ax.set_xlabel(variable_name)
  ax.set_ylabel("Maximum memory (MB)")
  ax.set_title(
    f"{variable_name} vs max memory — {len(x_values)} runs"
  )

  ax.grid(alpha=0.3)
  ax.set_axisbelow(True)

  plt.tight_layout()
  plt.show()
