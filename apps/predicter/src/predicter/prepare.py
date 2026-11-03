from common.Simulation import Simulation
from .decker import define_deck
import numpy as np

def prepare_dataset(
  simulations: list[Simulation], metric: str, columns: list[str]
) -> tuple[np.ndarray, np.ndarray]:
  X: list[list[float]] = [] # (number_of_simulations, number_of_input_features)
  y: list[list[float]] = [] # (number_of_simulations, number_of_target)

  for simulation in simulations:
    deckvector = define_deck(simulation.inputdeck, columns)

    if metric == "cpu_cores_used":
      target = float(simulation.maxcpu)
    else:
      target = float(simulation.maxmemory)

    X.append(deckvector)
    y.append(target)

  return np.asarray(X, dtype=float), np.asarray(y, dtype=float)
