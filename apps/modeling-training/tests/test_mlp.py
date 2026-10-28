from io import BytesIO
import unittest

import joblib
import numpy as np

from modeling_training.models import create_mlp


class TestMLPConstantFeatures(unittest.TestCase):
  def test_training_constant_columns_cannot_change_predictions(self):
    # Include both zero and nonzero training constants, plus varying inputs.
    X = np.array([[0, value, 7] for value in range(6)], dtype=float)
    y = np.array([[value + 3, value + 2, value + 1] for value in range(6)])
    model = create_mlp().fit(X, y)
    normal = np.array([[0, 2.5, 7], [0, 4.5, 7]])
    unseen = np.array([[50000, 2.5, -10000], [-50000, 4.5, 10000]])

    # Validation/prediction values must not refit which features are retained.
    np.testing.assert_allclose(model.predict(unseen), model.predict(normal))
    np.testing.assert_allclose(
      model.regressor_.named_steps["standardscaler"].mean_, [2.5],
    )
    self.assertEqual(model.n_features_in_, 3)

    # The fitted column mask must survive saving and loading the full model.
    artifact = BytesIO()
    joblib.dump(model, artifact)
    artifact.seek(0)
    loaded = joblib.load(artifact)
    np.testing.assert_allclose(loaded.predict(unseen), model.predict(normal))
