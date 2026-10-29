import numpy as np

from .decker import define_deck
from .Simulation import Simulation

def prepare_dataset(
  simulations: list[Simulation], metric: str
) -> tuple[np.ndarray, np.ndarray]:
  X: list[list[float]] = [] # (number_of_simulations, number_of_input_features)
  y: list[list[float]] = [] # (number_of_simulations, number_of_target)

  for simulation in simulations:
    deckvector = define_deck(simulation.inputdeck)
    print(f"dv: {deckvector}")

    if metric == "cpu_cores_used":
      target = float(simulation.maxcpu)
    else:
      target = float(simulation.maxmemory)

    X.append(deckvector)
    y.append(target)

  print(X, y)
  return np.asarray(X, dtype=float), np.asarray(y, dtype=float)
