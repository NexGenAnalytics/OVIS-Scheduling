from collections import defaultdict

import numpy as np

def family_of(name: str) -> str:
  """Problem-name prefix before the first size token, e.g. lj_base_256k_160k_seed -> lj_base."""
  prefix = []
  for token in name.split("_"):
    if token[:1].isdigit(): break
    prefix.append(token)
  return "_".join(prefix) or name

class FamilyModel:
  """Log target as a line in log particle count; the family mean when only one size was run."""

  def fit(self, log_atoms: np.ndarray, y: np.ndarray) -> "FamilyModel":
    if len(np.unique(log_atoms)) >= 2:
      self.slope_, self.intercept_ = np.polyfit(log_atoms, y, deg=1)
    else:
      self.slope_, self.intercept_ = 0.0, float(np.mean(y))
    return self

  def predict(self, log_atoms: float) -> float:
    return float(self.slope_ * log_atoms + self.intercept_)

def fit_families(families: list[str], log_atoms: np.ndarray, y: np.ndarray) -> dict[str, FamilyModel]:
  members: dict[str, list[int]] = defaultdict(list)
  for index, family in enumerate(families):
    members[family].append(index)
  return {family: FamilyModel().fit(log_atoms[indices], y[indices]) for family, indices in members.items()}
