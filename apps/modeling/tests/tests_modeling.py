import unittest

import pandas as pd

from modeling.cli import train_model

class TestModeling(unittest.TestCase):

  def test_train_model(self) -> None:
    data = pd.DataFrame([
      {
        "DECK": ["dimension 3", "atom_style atomic"],
        "CPU": [2, 5],
        "MEMORY": [100, 110],
      },
      {
        "DECK": ["dimension 1", "atom_style full"],
        "CPU": [3, 7],
        "MEMORY": [90, 95],
      },
    ])

    model = train_model(data)
    prediction = model.predict(["dimension 2\natom_style atomic"])

    self.assertEqual(prediction.shape, (1, 4))

  def test_rejects_inconsistent_series(self) -> None:
    data = pd.DataFrame([
      {
        "DECK": ["dimension 3"],
        "CPU": [2, 5],
        "MEMORY": [100],
      },
    ])

    with self.assertRaisesRegex(ValueError, "equally sized"):
      train_model(data)
