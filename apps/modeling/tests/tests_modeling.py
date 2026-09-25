import unittest

import numpy as np
import pandas as pd

from modeling.cli import (
  HASH_BITS,
  SimulationContent,
  get_simulation_duration,
  hash_to_features,
  predict_profiles,
)


class ConstantProfileModel:
  def __init__(self, value: float) -> None:
    self.value = value

  def predict(self, inputs: np.ndarray) -> np.ndarray:
    return np.full((len(inputs), 128), self.value)


class ConstantDurationModel:
  def __init__(self, duration_seconds: float) -> None:
    self.duration_seconds = duration_seconds

  def predict(self, inputs: np.ndarray) -> np.ndarray:
    return np.array([self.duration_seconds])


class ModelingTests(unittest.TestCase):

  def test_hash_to_features_uses_32_simhash_bits(self) -> None:
    features = hash_to_features(0b1011)

    self.assertEqual(HASH_BITS, len(features))
    np.testing.assert_array_equal(features[:4], [1.0, 1.0, 0.0, 1.0])

  def test_get_simulation_duration_uses_last_measurement(self) -> None:
    profile = pd.DataFrame({
      "time_rel_s": [1.0, 5.0, 12.5],
      "value": [1.0, 2.0, 3.0],
    })
    simulation = SimulationContent(1, 7, profile, profile)

    self.assertEqual(12.5, get_simulation_duration(simulation))

  def test_prediction_uses_seconds_for_time_axis(self) -> None:
    prediction, duration = predict_profiles(
      input_deck_hash=7,
      cpu_model=ConstantProfileModel(2.0),
      memory_model=ConstantProfileModel(3.0),
      duration_model=ConstantDurationModel(12.5),
    )

    self.assertEqual(12.5, duration)
    self.assertEqual(
      ["time_s", "cpu_cores_used", "mem_active_kb"],
      list(prediction.columns),
    )
    self.assertEqual(0.0, prediction["time_s"].iloc[0])
    self.assertEqual(12.5, prediction["time_s"].iloc[-1])
