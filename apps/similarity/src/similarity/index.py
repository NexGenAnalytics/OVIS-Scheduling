from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from common import get_converter
from .distances import DISTANCES, nearest
from .evaluate import CUTOFF_PERCENTILE, out_of_fold_nearest
from .profile import RESOURCES, Dataset, from_targets, resource_scales

@dataclass(frozen=True)
class SimilarityIndex:
  converter: str
  feature_names: tuple[str, ...]
  dataset: Dataset
  distance: str
  # Per resource, keyed like RESOURCES: fitted distance and no-match cutoff.
  fitted: dict[str, object]
  cutoffs: dict[str, float]

@dataclass(frozen=True)
class Match:
  neighbors: list[str]
  distance: float
  cutoff: float
  predicted: float

  @property
  def matched(self) -> bool:
    return self.distance <= self.cutoff

def build_index(dataset: Dataset, converter: str, distance: str) -> SimilarityIndex:
  fitted, cutoffs = {}, {}
  for resource, name in enumerate(RESOURCES):
    fitted[name] = DISTANCES[distance]().fit(dataset.X, dataset.targets[:, resource])
    cutoffs[name] = float(np.percentile(out_of_fold_nearest(dataset, distance, resource), CUTOFF_PERCENTILE))
  return SimilarityIndex(converter, get_converter(converter).feature_names(), dataset, distance, fitted, cutoffs)

def save_index(index: SimilarityIndex, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  joblib.dump(index, path)

def load_index(path: Path) -> SimilarityIndex:
  index = joblib.load(path)
  if index.feature_names != get_converter(index.converter).feature_names():
    raise ValueError("Index uses an incompatible deck feature schema. Rebuild it with 'similarity build'.")
  return index

def query(index: SimilarityIndex, content: str, particle_count: float, k: int) -> dict[str, Match]:
  """Per resource: inverse-distance-weighted target of the k nearest runs, rescaled to this deck."""
  converter = get_converter(index.converter)
  deck = converter.parse(content)
  x = np.asarray([converter.featurize(deck, particle_count)], dtype=float)
  scales = resource_scales(converter, deck, particle_count)
  matches = {}
  for resource, name in enumerate(RESOURCES):
    distances = index.fitted[name].pairwise(x, index.dataset.X)[0]
    target, indices = nearest(distances, index.dataset.targets[:, resource], k)
    matches[name] = Match(
      neighbors=[index.dataset.names[i] for i in indices],
      distance=float(distances[indices[0]]),
      cutoff=index.cutoffs[name],
      predicted=float(from_targets(np.asarray(target), scales[resource])),
    )
  return matches
