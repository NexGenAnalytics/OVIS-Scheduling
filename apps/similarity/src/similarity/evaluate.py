from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor

from modeling_training.models import MODEL_FACTORIES, RANDOM_SEED
from .distances import DISTANCES, LearnedWeights, nearest
from .family import fit_families
from .profile import RESOURCES, Dataset, from_targets

N_FOLDS = 10
NEIGHBOR_COUNTS = (1, 3)
TOLERANCE = 0.20
SCENARIOS = ("deck", "family")
# A deck farther from every known run than this share of held-out known-family decks gets no match.
CUTOFF_PERCENTILE = 90

@dataclass(frozen=True)
class Score:
  method: str
  median_error: float
  within_tolerance: float
  underestimated: float
  # Spearman correlation between nearest-neighbor distance and error; None for regressors.
  distance_error_correlation: float | None
  # Share of runs beyond the distance cutoff, and the median error of the rest; None for regressors.
  flagged: float | None
  matched_median_error: float | None

def _folds(dataset: Dataset, scenario: str) -> list[tuple[np.ndarray, np.ndarray]]:
  # Repeats of a deck share a name, so they are held out together.
  groups = dataset.names if scenario == "deck" else dataset.families
  folds = GroupKFold(min(N_FOLDS, len(set(groups))), shuffle=True, random_state=RANDOM_SEED)
  return list(folds.split(dataset.X, groups=groups))

def out_of_fold_nearest(dataset: Dataset, distance: str, resource: int) -> np.ndarray:
  """Distance from each run to its nearest run in other folds, holding out decks."""
  y = dataset.targets[:, resource]
  nearest_distances = np.zeros(len(y))
  for train, test in _folds(dataset, "deck"):
    matrix = DISTANCES[distance]().fit(dataset.X[train], y[train]).pairwise(dataset.X[test], dataset.X[train])
    nearest_distances[test] = matrix.min(axis=1)
  return nearest_distances

def _cross_validate(dataset: Dataset, scenario: str, resource: int):
  """Out-of-fold log-target predictions per method, plus nearest distances for neighbor methods."""
  y = dataset.targets[:, resource]
  predictions: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(y)))
  nearest_distances: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(y)))

  for train, test in _folds(dataset, scenario):
    families = fit_families([dataset.families[i] for i in train], dataset.log_atoms[train], y[train])
    for name, distance_class in DISTANCES.items():
      matrix = distance_class().fit(dataset.X[train], y[train]).pairwise(dataset.X[test], dataset.X[train])
      for row, index in zip(matrix, test):
        for k in NEIGHBOR_COUNTS:
          predictions[f"{name} k={k}"][index], _ = nearest(row, y[train], k)
        # Route to the nearest run's family and apply that family's model.
        family = dataset.families[train[np.argmin(row)]]
        predictions[f"{name} family"][index] = families[family].predict(dataset.log_atoms[index])
        for method in (*(f"{name} k={k}" for k in NEIGHBOR_COUNTS), f"{name} family"):
          nearest_distances[method][index] = row.min()
    for name, factory in MODEL_FACTORIES.items():
      model = factory()
      # MultiOutputRegressor needs a 2-D target; the others expect 1-D.
      model.fit(dataset.X[train], y[train][:, None] if isinstance(model, MultiOutputRegressor) else y[train])
      predictions[f"regressor {name}"][test] = np.asarray(model.predict(dataset.X[test])).reshape(-1)

  return predictions, nearest_distances

def _score(method: str, relative: np.ndarray, distances: np.ndarray | None, cutoff: float | None) -> Score:
  error = np.abs(relative)
  correlation = flagged = matched_error = None
  if distances is not None:
    correlation = float(spearmanr(distances, error).statistic)
    matched = distances <= cutoff
    flagged = float(np.mean(~matched))
    matched_error = float(np.median(error[matched])) if matched.any() else None
  return Score(
    method=method,
    median_error=float(np.median(error)),
    within_tolerance=float(np.mean(error <= TOLERANCE)),
    underestimated=float(np.mean(relative < -TOLERANCE)),
    distance_error_correlation=correlation,
    flagged=flagged,
    matched_median_error=matched_error,
  )

def evaluate(dataset: Dataset) -> dict[str, dict[str, list[Score]]]:
  """Scores per scenario and resource; distance cutoffs come from the held-out-deck scenario."""
  scores: dict[str, dict[str, list[Score]]] = {scenario: {} for scenario in SCENARIOS}
  for resource, resource_name in enumerate(RESOURCES):
    results = {scenario: _cross_validate(dataset, scenario, resource) for scenario in SCENARIOS}
    cutoffs = {method: np.percentile(d, CUTOFF_PERCENTILE) for method, d in results["deck"][1].items()}
    for scenario, (predictions, nearest_distances) in results.items():
      scores[scenario][resource_name] = []
      for method, predicted in predictions.items():
        absolute = from_targets(predicted, dataset.scales[:, resource])
        relative = absolute / dataset.observed[:, resource] - 1.0
        scores[scenario][resource_name].append(
          _score(method, relative, nearest_distances.get(method), cutoffs.get(method))
        )
  return scores

def repeat_spread(dataset: Dataset) -> list[tuple[str, np.ndarray]]:
  """Largest relative resource difference among repeat runs of each deck: the measurement noise floor."""
  groups: dict[str, list[int]] = defaultdict(list)
  for index, name in enumerate(dataset.names):
    groups[name].append(index)
  spreads = []
  for name, indices in groups.items():
    if len(indices) < 2: continue
    runs = dataset.observed[indices]
    spreads.append((name, runs.max(axis=0) / runs.min(axis=0) - 1.0))
  return spreads

def learned_weights(dataset: Dataset, feature_names: tuple[str, ...], top: int = 10) -> dict[str, list[tuple[str, float]]]:
  """Most influential features per resource, fitted on all runs."""
  return {
    name: LearnedWeights().fit(dataset.X, dataset.targets[:, resource]).feature_weights(feature_names)[:top]
    for resource, name in enumerate(RESOURCES)
  }
