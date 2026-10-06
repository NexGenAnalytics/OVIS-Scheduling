from collections.abc import Callable
from typing import Any

from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
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
    hidden_layer_sizes=(16,),
    solver="lbfgs",
    max_iter=2000,
    random_state=RANDOM_SEED
  )
  model = TransformedTargetRegressor(
    regressor=make_pipeline(StandardScaler(), mlp_regressor),
    transformer=StandardScaler(),
  )
  return model

MODEL_FACTORIES: dict[str, ModelFactory] = {
  "dummy": lambda: DummyRegressor(strategy="mean"),
  "ridge": create_ridge,
  "mlp": create_mlp,
}
