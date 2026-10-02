from dataclasses import dataclass

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
  totaltime: int # in seconds
