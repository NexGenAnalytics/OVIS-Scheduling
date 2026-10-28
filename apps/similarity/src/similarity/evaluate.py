from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, KFold
from sklearn.multioutput import MultiOutputRegressor

from modeling_training.models import MODEL_FACTORIES, RANDOM_SEED
from .distances import DISTANCES, LearnedWeights, nearest
from .profile import RESOURCES, Dataset, from_targets

N_FOLDS = 10
NEIGHBOR_COUNTS = (1, 3)
TOLERANCE = 0.20

@dataclass(frozen=True)
class Score:
  method: str
  median_error: float
  within_tolerance: float
  underestimated: float
  # Spearman correlation between nearest-neighbor distance and error; None for regressors.
  distance_error_correlation: float | None

def family_of(name: str) -> str:
  """Problem-name prefix before the first size token, e.g. lj_base_256k_160k_seed -> lj_base."""
  prefix = []
  for token in name.split("_"):
    if token[:1].isdigit(): break
    prefix.append(token)
  return "_".join(prefix) or name

def _folds(dataset: Dataset, scenario: str) -> list[tuple[np.ndarray, np.ndarray]]:
  if scenario == "deck":
    return list(KFold(N_FOLDS, shuffle=True, random_state=RANDOM_SEED).split(dataset.X))
  families = [family_of(name) for name in dataset.names]
  return list(GroupKFold(min(N_FOLDS, len(set(families)))).split(dataset.X, groups=families))

def _cross_validate(dataset: Dataset, scenario: str, resource: int):
  """Out-of-fold log-target predictions per method, plus nearest distances for neighbor methods."""
  y = dataset.targets[:, resource]
  predictions: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(y)))
  nearest_distances: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(y)))

  for train, test in _folds(dataset, scenario):
    for name, distance_class in DISTANCES.items():
      matrix = distance_class().fit(dataset.X[train], y[train]).pairwise(dataset.X[test], dataset.X[train])
      for k in NEIGHBOR_COUNTS:
        method = f"{name} k={k}"
        for row, index in zip(matrix, test):
          predictions[method][index], _ = nearest(row, y[train], k)
          nearest_distances[method][index] = row.min()
    for name, factory in MODEL_FACTORIES.items():
      model = factory()
      # MultiOutputRegressor needs a 2-D target; the others expect 1-D.
      model.fit(dataset.X[train], y[train][:, None] if isinstance(model, MultiOutputRegressor) else y[train])
      predictions[f"regressor {name}"][test] = np.asarray(model.predict(dataset.X[test])).reshape(-1)

  return predictions, nearest_distances

def evaluate(dataset: Dataset, scenario: str) -> dict[str, list[Score]]:
  observed = dataset.observed
  scores: dict[str, list[Score]] = {}
  for resource, resource_name in enumerate(RESOURCES):
    predictions, nearest_distances = _cross_validate(dataset, scenario, resource)
    scores[resource_name] = []
    for method, predicted in predictions.items():
      absolute = from_targets(predicted, dataset.scales[:, resource])
      relative = absolute / observed[:, resource] - 1.0
      correlation = None
      if method in nearest_distances:
        correlation = float(spearmanr(nearest_distances[method], np.abs(relative)).statistic)
      scores[resource_name].append(Score(
        method=method,
        median_error=float(np.median(np.abs(relative))),
        within_tolerance=float(np.mean(np.abs(relative) <= TOLERANCE)),
        underestimated=float(np.mean(relative < -TOLERANCE)),
        distance_error_correlation=correlation,
      ))
  return scores

def identical_deck_spread(dataset: Dataset) -> list[tuple[str, str, np.ndarray]]:
  """Relative resource differences between runs with identical feature rows: a measurement-noise proxy."""
  groups: dict[bytes, list[int]] = defaultdict(list)
  for index, row in enumerate(dataset.X):
    groups[row.tobytes()].append(index)
  pairs = []
  for indices in groups.values():
    for a, b in zip(indices, indices[1:]):
      difference = np.abs(dataset.observed[a] - dataset.observed[b]) / dataset.observed[[a, b]].min(axis=0)
      pairs.append((dataset.names[a], dataset.names[b], difference))
  return pairs

def learned_weights(dataset: Dataset, feature_names: tuple[str, ...], top: int = 10) -> dict[str, list[tuple[str, float]]]:
  """Most influential features per resource, fitted on all runs."""
  return {
    name: LearnedWeights().fit(dataset.X, dataset.targets[:, resource]).feature_weights(feature_names)[:top]
    for resource, name in enumerate(RESOURCES)
  }
