from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor

from common import DECK_FEATURE_NAMES, Metric, Simulation, define_deck, parse_inputdeck
from modeling_training.cli import save_model
from modeling_training.dataset import prepare_dataset


class TestTrainingDeckSchema(unittest.TestCase):
  def test_dataset_uses_shared_vector_and_keeps_particle_count_last(self):
    deck = parse_inputdeck("units lj\natom_style atomic\nneighbor 1 nsq\nrun 100")
    profile = [Metric(0, 0, "cpu", 2), Metric(1, 1, "cpu", 6)]
    simulation = Simulation(1, deck, profile, 1, 256000)
    X, y = prepare_dataset([simulation], "cpu")
    np.testing.assert_array_equal(X[0], define_deck(deck) + [256000.0])
    np.testing.assert_array_equal(y, [[6, 4, 2]])
    self.assertEqual(X.shape, (1, len(DECK_FEATURE_NAMES) + 1))

  def test_saved_model_carries_schema_and_predicts_after_loading(self):
    model = DummyRegressor().fit(np.zeros((2, len(DECK_FEATURE_NAMES) + 1)), [[3, 2, 1], [5, 3, 2]])
    with tempfile.TemporaryDirectory() as directory:
      with patch("modeling_training.cli.Path", return_value=Path(directory)):
        path = save_model(model, "test")
      loaded = joblib.load(path)
      self.assertEqual(loaded.deck_feature_names_, (*DECK_FEATURE_NAMES, "particle_count"))
      np.testing.assert_array_equal(loaded.predict(np.zeros((1, len(DECK_FEATURE_NAMES) + 1))), [[4, 2.5, 1.5]])
