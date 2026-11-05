from collections.abc import Callable
from typing import Any

from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
  ExtraTreesRegressor,
  HistGradientBoostingRegressor,
  RandomForestRegressor,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42

ModelFactory = Callable[[], Any]

def create_ridge() -> TransformedTargetRegressor:
  return TransformedTargetRegressor(
    regressor=make_pipeline(
      StandardScaler(),
      Ridge(alpha=1.0),
    ),
    transformer=StandardScaler(),
  )

def create_mlp() -> TransformedTargetRegressor:
  mlp_regressor = MLPRegressor(
    # Means one hidden layer containing 4 neurons.
    # Architecture: One layer, 4 neurons.
    # Effect: Simpler, may miss complex relationships.
    hidden_layer_sizes=(4,),
    # Controls how strongly training penalizes large weights.
    # Regularization: Strong.
    alpha=0.1,
    solver="lbfgs",
    max_iter=3000,
    random_state=RANDOM_SEED
  )
  model = TransformedTargetRegressor(
    regressor=make_pipeline(
      # Fit on training data and reuse the same columns during prediction.
      VarianceThreshold(),
      StandardScaler(),
      mlp_regressor,
    ),
    transformer=StandardScaler(),
  )
  return model

def create_random_forest() -> RandomForestRegressor:
  return RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=2,
    random_state=RANDOM_SEED,
  )

def create_extra_trees() -> ExtraTreesRegressor:
  return ExtraTreesRegressor(
    n_estimators=200,
    min_samples_leaf=2,
    random_state=RANDOM_SEED,
  )

def create_hist_gradient_boosting() -> MultiOutputRegressor:
  # HistGradientBoostingRegressor needs a separate estimator for each target.
  return MultiOutputRegressor(HistGradientBoostingRegressor(
    # Keep trees small for the limited number of training simulations.
    max_leaf_nodes=7,
    min_samples_leaf=5,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=RANDOM_SEED,
  ))

MODEL_FACTORIES: dict[str, ModelFactory] = {
  "dummy": lambda: DummyRegressor(strategy="mean"),
  "ridge": create_ridge,
  "mlp": create_mlp,
  "random_forest": create_random_forest,
  "extra_trees": create_extra_trees,
  "hist_gb": create_hist_gradient_boosting,
}
