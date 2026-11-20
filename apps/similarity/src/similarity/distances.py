import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modeling_training.models import create_extra_trees

def _standardizer():
  return make_pipeline(VarianceThreshold(), StandardScaler())

class ChangedParams:
  """Number of differing features, the closest analogue of the old simhash distance."""

  def fit(self, X: np.ndarray, y: np.ndarray) -> "ChangedParams":
    return self

  def pairwise(self, Xq: np.ndarray, Xref: np.ndarray) -> np.ndarray:
    return (Xq[:, None, :] != Xref[None, :, :]).sum(axis=2).astype(float)

class StandardizedEuclidean:
  """Unweighted distance over standardized features."""

  def fit(self, X: np.ndarray, y: np.ndarray) -> "StandardizedEuclidean":
    self.scaler_ = _standardizer().fit(X)
    return self

  def pairwise(self, Xq: np.ndarray, Xref: np.ndarray) -> np.ndarray:
    return pairwise_distances(self.scaler_.transform(Xq), self.scaler_.transform(Xref))

class LearnedWeights:
  """Weighted L1 distance whose non-negative feature weights predict |Δtarget| over training pairs."""

  def __init__(self, alpha: float = 1.0):
    self.alpha = alpha

  def fit(self, X: np.ndarray, y: np.ndarray) -> "LearnedWeights":
    self.scaler_ = _standardizer().fit(X)
    Z = self.scaler_.transform(X)
    i, j = np.triu_indices(len(Z), k=1)
    pair_model = Ridge(alpha=self.alpha, positive=True).fit(np.abs(Z[i] - Z[j]), np.abs(y[i] - y[j]))
    self.weights_ = pair_model.coef_
    return self

  def pairwise(self, Xq: np.ndarray, Xref: np.ndarray) -> np.ndarray:
    q = self.scaler_.transform(Xq) * self.weights_
    r = self.scaler_.transform(Xref) * self.weights_
    return pairwise_distances(q, r, metric="manhattan")

  def feature_weights(self, feature_names: tuple[str, ...]) -> list[tuple[str, float]]:
    """Weights by feature name, largest first."""
    kept = np.asarray(feature_names)[self.scaler_[0].get_support()]
    return sorted(zip(kept, self.weights_), key=lambda pair: -pair[1])

class ForestProximity:
  """1 - share of trees in which two decks land in the same leaf of a target-trained forest."""

  def fit(self, X: np.ndarray, y: np.ndarray) -> "ForestProximity":
    self.forest_ = create_extra_trees().fit(X, y)
    return self

  def pairwise(self, Xq: np.ndarray, Xref: np.ndarray) -> np.ndarray:
    leaves_q, leaves_ref = self.forest_.apply(Xq), self.forest_.apply(Xref)
    return 1.0 - (leaves_q[:, None, :] == leaves_ref[None, :, :]).mean(axis=2)

DISTANCES = {
  "changed_params": ChangedParams,
  "euclidean": StandardizedEuclidean,
  "learned_weights": LearnedWeights,
  "forest": ForestProximity,
}

def nearest(distances: np.ndarray, targets: np.ndarray, k: int) -> tuple[float, np.ndarray]:
  """Inverse-distance-weighted target of the k nearest references, and their indices."""
  indices = np.argsort(distances, kind="stable")[:k]
  weights = 1.0 / (distances[indices] + 1e-9)
  return float(np.average(targets[indices], weights=weights)), indices
