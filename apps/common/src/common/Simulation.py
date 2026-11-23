from dataclasses import dataclass, field

from .Feature import Feature

@dataclass(frozen=True)
class Simulation:
  id: int
  name: str
  maxcpu: int # cores
  maxmemory: int # kB
  totaltime: int # seconds
  particlecount: int # particles
  inputdeck: list[Feature]
  # Idle value per node, then per metric, from ldms_baseline.csv; empty when the run has none.
  baseline: dict[int, dict[str, float]] = field(default_factory=dict)
