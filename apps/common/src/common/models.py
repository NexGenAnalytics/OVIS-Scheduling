from dataclasses import dataclass

@dataclass(frozen=True)
class DeckConcepts:
  total_steps: list[float]
  units: str
  atom_style: str
  pair_styles: tuple[str, ...]
  density: list[float]
  simulation_family: str
  interaction_cutoff: list[float]
  atom_mapping: list[float]
  long_range_settings: list[float]
  execution: list[float]
  neighbor_settings: list[float]
  geometry_settings: list[float]
  interaction_settings: list[float]
  dynamics_settings: list[float]
  output_settings: list[float]

@dataclass(frozen=True)
class Feature:
  """
  Each LAMMPS instruction/feature as (command, arguments).
  """
  command: str
  arguments: list[str]

@dataclass(frozen=True)
class Metric:
  timestamp: float
  time_rel_s: float
  #job_id: int
  #component_id: int
  metric: str
  #unit: str
  value: float

@dataclass(frozen=True)
class Simulation:
  id: int
  inputdeck: list[Feature]
  runprofile: list[Metric]
  # Execution time of the simulation in seconds
  totaltime: int
  # Number simulated particles (atoms, molecules)
  # that LAMMPS must process.
  particlecount: float

