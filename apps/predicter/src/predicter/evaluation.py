from dataclasses import dataclass

import numpy as np

from sklearn.metrics import (mean_absolute_error, root_mean_squared_error)

@dataclass(frozen=True)
class TargetScore:
  target: str
  mae: float
  rmse: float

def evaluate_predictions(
  expected: np.ndarray,
  predicted: np.ndarray
) -> list[TargetScore]:
  scores: list[TargetScore] = []

  # Mean Absolute Error
  mae = mean_absolute_error(expected, predicted)

  # Root Mean Squared Error
  rmse = root_mean_squared_error(expected, predicted)

  score = TargetScore(
    target="find max",
    mae=float(mae),
    rmse=float(rmse),
  )
  scores.append(score)

  return scores
