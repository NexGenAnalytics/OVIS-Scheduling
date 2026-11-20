from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from common import Converter, Simulation
from .family import family_of

RESOURCES = ("memory", "cpu", "duration")
# Job memory is anonymous plus shared pages above the idle node; repeats agree within ~1%.
MEMORY_METRICS = ("mem_anon_kb", "mem_shmem_kb")
CPU_METRIC = "cpu_cores_used"

@dataclass(frozen=True)
class Dataset:
  names: list[str]
  families: list[str]
  X: np.ndarray
  particle_counts: np.ndarray
  # Measured peak per-node job memory (kB), peak CPU (cores) and duration (s) per run.
  observed: np.ndarray
  # Divisors that make each resource comparable across run sizes.
  scales: np.ndarray

  @property
  def targets(self) -> np.ndarray:
    return to_targets(self.observed, self.scales)

  @property
  def log_atoms(self) -> np.ndarray:
    return np.log(self.particle_counts)

def _peak_memory(simulation: Simulation) -> float | None:
  """Largest memory any one node uses above its own idle baseline."""
  totals: dict[tuple[int, float], float] = defaultdict(float)
  for point in simulation.runprofile:
    if point.metric in MEMORY_METRICS: totals[point.component_id, point.timestamp] += point.value
  if not totals: return None
  idle = {}
  for node in {node for node, _ in totals}:
    node_baseline = simulation.baseline.get(node, {})
    if not all(metric in node_baseline for metric in MEMORY_METRICS): return None
    idle[node] = sum(node_baseline[metric] for metric in MEMORY_METRICS)
  return max(max(total - idle[node] for (node, _), total in totals.items()), 1.0)

def measure(simulation: Simulation) -> list[float] | None:
  """Peak per-node memory, peak CPU and duration, or None if the run lacks those metrics or a baseline."""
  memory = _peak_memory(simulation)
  cpu = [point.value for point in simulation.runprofile if point.metric == CPU_METRIC]
  if memory is None or not cpu: return None
  return [memory, max(cpu), max(float(simulation.totaltime), 1.0)]

def resource_scales(converter: Converter, deck, particle_count: float) -> list[float]:
  # Only wall time is size-normalized: job memory carries a fixed per-rank MPI overhead that dwarfs small decks.
  return [1.0, 1.0, converter.work(deck, particle_count)]

def to_targets(observed: np.ndarray, scales: np.ndarray) -> np.ndarray:
  return np.log(observed / scales)

def from_targets(targets: np.ndarray, scales: np.ndarray) -> np.ndarray:
  return np.exp(targets) * scales

def build_dataset(simulations: list[Simulation], converter: Converter) -> Dataset:
  measured = [(simulation, measure(simulation)) for simulation in simulations]
  measured = [(simulation, observed) for simulation, observed in measured if observed is not None]
  print(f"- Measurable {len(measured)} of {len(simulations)} simulations (others lack memory/CPU metrics or baseline)")
  return Dataset(
    names=[s.name for s, _ in measured],
    families=[family_of(s.name) for s, _ in measured],
    X=np.asarray([converter.featurize(s.inputdeck, s.particlecount) for s, _ in measured], dtype=float),
    particle_counts=np.asarray([s.particlecount for s, _ in measured], dtype=float),
    observed=np.asarray([observed for _, observed in measured], dtype=float),
    scales=np.asarray([resource_scales(converter, s.inputdeck, s.particlecount) for s, _ in measured], dtype=float),
  )
