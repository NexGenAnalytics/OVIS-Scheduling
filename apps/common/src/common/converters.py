from typing import Protocol

from common.models import Feature
from common.parsing import parse_inputdeck
from common.utils import DECK_FEATURE_NAMES, define_deck, find_total_steps

class Converter(Protocol):
  """Turns one application's input deck into a fixed numeric row."""
  name: str

  def parse(self, content: str) -> list[Feature]: ...

  def featurize(self, deck: list[Feature], particle_count: float) -> list[float]: ...

  def feature_names(self) -> tuple[str, ...]: ...

  def work(self, deck: list[Feature], particle_count: float) -> float:
    """Amount of computation that wall time should scale with."""
    ...

class LammpsConverter:
  name = "lammps"

  def parse(self, content: str) -> list[Feature]:
    return parse_inputdeck(content)

  def featurize(self, deck: list[Feature], particle_count: float) -> list[float]:
    return [*define_deck(deck), float(particle_count)]

  def feature_names(self) -> tuple[str, ...]:
    return (*DECK_FEATURE_NAMES, "particle_count")

  def work(self, deck: list[Feature], particle_count: float) -> float:
    # Atom-steps: LAMMPS cost per step is roughly linear in atom count.
    steps, found = find_total_steps(deck)
    return float(particle_count) * (steps if found and steps > 0 else 1.0)

CONVERTERS: dict[str, Converter] = {"lammps": LammpsConverter()}

def get_converter(name: str) -> Converter:
  if name not in CONVERTERS:
    raise KeyError(f"unknown converter '{name}'; known: {', '.join(sorted(CONVERTERS))}")
  return CONVERTERS[name]
