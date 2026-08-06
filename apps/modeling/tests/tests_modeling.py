import unittest

import pandas as pd

from modeling.cli import create_targets, display_prediction, predict_model, train_model

class TestModeling(unittest.TestCase):

  def create_data(self) -> pd.DataFrame:
    return pd.DataFrame([
      {
        "DECK": ["dimension 3", "atom_style atomic"],
        "CPU": [2, 5],
        "MEMORY": [100, 110],
      },
      {
        "DECK": ["dimension 1", "atom_style full"],
        "CPU": [3, 7, 8, 4, 5],
        "MEMORY": [90, 95, 102, 123, 89],
      },
    ])

  def test_accepts_profiles_with_different_durations(self) -> None:
    targets = create_targets(self.create_data())
    self.assertEqual(len(targets[0]), len(targets[1]))
    self.assertEqual(targets[0][0], 2)
    self.assertEqual(targets[1][0], 5)

  def test_train_model(self) -> None:
    model = train_model(self.create_data())
    duration, cpu, memory = predict_model(
      model,
      "dimension 3\natom_style atomic",
    )

    self.assertEqual(duration, 2)
    self.assertEqual(len(cpu), duration)
    self.assertEqual(len(memory), duration)

  def test_rejects_inconsistent_series(self) -> None:
    data = pd.DataFrame([
      {
        "DECK": ["dimension 3"],
        "CPU": [2, 5],
        "MEMORY": [100],
      },
    ])

    with self.assertRaisesRegex(ValueError, "equally sized"):
      create_targets(data)
