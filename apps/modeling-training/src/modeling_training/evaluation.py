from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
  mean_absolute_error,
  root_mean_squared_error,
)

TARGET_NAMES = ("max", "mean", "min")

@dataclass(frozen=True)
class TargetScore:
  target: str
  mae: float
  rmse: float

def evaluate_predictions(
  expected: np.ndarray,
  predicted: np.ndarray,
) -> list[TargetScore]:
  scores: list[TargetScore] = []

  for index, target_name in enumerate(TARGET_NAMES):
    target_expected = expected[:, index]
    target_predicted = predicted[:, index]

    # Mean Absolute Error
    mae = mean_absolute_error(target_expected, target_predicted)

    # Root Mean Squared Error
    rmse = root_mean_squared_error(target_expected, target_predicted)

    scores.append(TargetScore(
      target=target_name,
      mae=float(mae),
      rmse=float(rmse),
    ))

  return scores