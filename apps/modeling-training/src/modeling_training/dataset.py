import numpy as np
from common import Simulation, define_deck

def prepare_dataset(
  simulations: list[Simulation],
  metric: str,
) -> tuple[np.ndarray, np.ndarray]:
  X: list[list[float]] = [] # (number_of_simulations, number_of_input_features)
  y: list[list[float]] = [] # (number_of_simulations, number_of_target)

  for simulation in simulations:
    deck_vector = define_deck(simulation.inputdeck)
    # print(deck_vector)

    values = np.asarray(
      [
        point.value
        for point in simulation.runprofile
        if point.metric == metric
      ],
      dtype=float,
    )

    if values.size == 0:
      raise ValueError(
        f"Simulation {simulation.id} has no values for {metric!r}"
      )

    if not np.all(np.isfinite(values)):
      raise ValueError(
        f"Simulation {simulation.id} has invalid values for {metric!r}"
      )

    target = [
      float(np.max(values)),
      float(np.mean(values)),
      float(np.min(values)),
    ]

    X.append(deck_vector)
    y.append(target)

  return np.asarray(X, dtype=float), np.asarray(y, dtype=float)
