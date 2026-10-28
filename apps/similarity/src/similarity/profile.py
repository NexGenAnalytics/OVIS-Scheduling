from dataclasses import dataclass

import numpy as np

from common import Converter, Simulation

RESOURCES = ("memory", "cpu", "duration")
RESOURCE_METRICS = {"memory": "mem_active_kb", "cpu": "cpu_cores_used"}

@dataclass(frozen=True)
class Dataset:
  names: list[str]
  X: np.ndarray
  # Measured peak memory (kB), peak CPU (cores) and duration (s) per run.
  observed: np.ndarray
  # Divisors that make each resource comparable across run sizes.
  scales: np.ndarray

  @property
  def targets(self) -> np.ndarray:
    return to_targets(self.observed, self.scales)

def measure(simulation: Simulation) -> list[float]:
  peaks = [
    max(point.value for point in simulation.runprofile if point.metric == metric)
    for metric in RESOURCE_METRICS.values()
  ]
  return [*peaks, max(float(simulation.totaltime), 1.0)]

def resource_scales(converter: Converter, deck, particle_count: float) -> list[float]:
  # Only wall time is size-normalized: peak memory is dominated by a fixed node baseline in the current data.
  return [1.0, 1.0, converter.work(deck, particle_count)]

def to_targets(observed: np.ndarray, scales: np.ndarray) -> np.ndarray:
  return np.log(observed / scales)

def from_targets(targets: np.ndarray, scales: np.ndarray) -> np.ndarray:
  return np.exp(targets) * scales

def build_dataset(simulations: list[Simulation], converter: Converter) -> Dataset:
  return Dataset(
    names=[simulation.name for simulation in simulations],
    X=np.asarray([converter.featurize(s.inputdeck, s.particlecount) for s in simulations], dtype=float),
    observed=np.asarray([measure(simulation) for simulation in simulations], dtype=float),
    scales=np.asarray([resource_scales(converter, s.inputdeck, s.particlecount) for s in simulations], dtype=float),
  )
