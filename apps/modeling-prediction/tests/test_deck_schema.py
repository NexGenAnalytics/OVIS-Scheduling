from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from common import DECK_FEATURE_NAMES, define_deck, parse_inputdeck
from modeling_prediction.cli import main


class TestPredictionDeckSchema(unittest.TestCase):
  def invoke(self, model, content="units lj\natom_style atomic\nneighbor 1 nsq\nrun 100"):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "in.test"
      path.write_text(content)
      args = SimpleNamespace(model=Path(directory) / "model.joblib", input=path, particle_count=256000)
      with patch("modeling_prediction.cli.init_parser", return_value=args), patch("modeling_prediction.cli.joblib.load", return_value=model), redirect_stdout(StringIO()):
        main()
    return content

  def test_prediction_uses_shared_vector_and_initial_particle_count(self):
    model = SimpleNamespace(deck_feature_names_=(*DECK_FEATURE_NAMES, "particle_count"), predict=Mock(return_value=np.array([[3, 2, 1]])))
    content = self.invoke(model)
    model.predict.assert_called_once()
    np.testing.assert_array_equal(model.predict.call_args.args[0], [define_deck(parse_inputdeck(content)) + [256000.0]])

  def test_old_models_get_retraining_message_before_predict(self):
    model = SimpleNamespace(n_features_in_=83, predict=Mock())
    with self.assertRaisesRegex(ValueError, "Retrain"):
      self.invoke(model)
    model.predict.assert_not_called()

  def test_same_width_but_different_column_order_is_rejected(self):
    model = SimpleNamespace(deck_feature_names_=(*reversed(DECK_FEATURE_NAMES), "particle_count"), predict=Mock())
    with self.assertRaisesRegex(ValueError, "incompatible"):
      self.invoke(model)
    model.predict.assert_not_called()
