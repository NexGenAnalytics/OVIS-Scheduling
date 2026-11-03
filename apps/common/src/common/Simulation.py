from dataclasses import dataclass

from .Feature import Feature

@dataclass(frozen=True)
class Simulation:
  id: int
  name: str
  maxcpu: int # cores
  maxmemory: int # kB
  totaltime: int # seconds
  inputdeck: list[Feature]
  #runprofile: list[Metric]
